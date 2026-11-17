"""Create leakage-safe candidate slates from the existing RecBole splits."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Iterable

import numpy as np
import torch

from common.normalization import masked_minmax


def canonical_group(
    positions: torch.Tensor,
    item_ids: torch.Tensor,
    base_scores: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match the current evaluator's last-write duplicate and deterministic tie rules."""

    occurrence: dict[int, int] = {}
    for position in positions.tolist():
        occurrence[int(item_ids[position])] = int(position)
    ordered_items = sorted(occurrence)
    representatives = torch.tensor([occurrence[item] for item in ordered_items], dtype=torch.long)
    items = torch.tensor(ordered_items, dtype=torch.long)
    scores = base_scores[representatives]
    id_order = torch.argsort(items, stable=True)
    score_order = torch.argsort(scores[id_order], descending=True, stable=True)
    order = id_order[score_order]
    return items[order], scores[order]


def build_eval_batches(adapter, loader, split: str) -> tuple[list[dict], list[int]]:
    batches = []
    lengths = []
    for interaction, row_index, positive_u, positive_i in loader:
        base = adapter.score_interaction(interaction)
        users = interaction[adapter.uid_field].detach().cpu()
        items = interaction[adapter.iid_field].detach().cpu()
        row_index = row_index.detach().cpu()
        groups = []
        # Group in one stable sort instead of scanning the complete candidate
        # vector once per row.  This preserves the former sorted-row order and
        # the original position order within each row (needed for last-write
        # duplicate semantics), while avoiding quadratic preparation time.
        row_order = torch.argsort(row_index, stable=True)
        sorted_rows = row_index[row_order]
        _, row_counts = torch.unique_consecutive(sorted_rows, return_counts=True)
        for positions in row_order.split(row_counts.tolist()):
            user_values = torch.unique(users[positions])
            if user_values.numel() != 1:
                raise ValueError("one evaluator row must correspond to one user")
            candidate_items, scores = canonical_group(positions, items, base)
            groups.append({
                "user": int(user_values.item()),
                "items": candidate_items.to(torch.int32),
                "base_score_raw": scores.to(torch.float32),
            })
            lengths.append(len(candidate_items))
        batches.append({
            "split": split,
            "groups": groups,
            "positive_u": positive_u.detach().cpu().to(torch.long),
            "positive_i": positive_i.detach().cpu().to(torch.long),
        })
    return batches, lengths


def _unique_negatives(
    rng: np.random.Generator,
    item_num: int,
    excluded: set[int],
    count: int,
) -> list[int]:
    available = item_num - 1 - len(excluded)
    wanted = min(count, max(0, available))
    result: set[int] = set()
    while len(result) < wanted:
        draw = rng.integers(1, item_num, size=max(256, 2 * (wanted - len(result))))
        result.update(int(item) for item in draw if int(item) not in excluded)
    return sorted(result)[:wanted]


def build_training_samples(
    adapter, *, sample_num: int, seed: int, targets_per_slate: int
) -> tuple[list[dict], dict[int, list[int]], dict]:
    interaction = adapter.train_data._dataset.inter_feat
    users = interaction[adapter.uid_field].detach().cpu().numpy()
    items = interaction[adapter.iid_field].detach().cpu().numpy()
    by_user: dict[int, list[int]] = defaultdict(list)
    for user, item in zip(users.tolist(), items.tolist()):
        if user > 0 and item > 0:
            by_user[int(user)].append(int(item))

    all_histories: dict[int, list[int]] = {}
    samples = []
    separation = (
        "fixed-seed random 80% history / 20% target pool within train; target pool is split into "
        f"slates of at most {targets_per_slate} positives so training covers full evaluation positions; "
        "each slate's targets are excluded from its history; no timestamp loaded"
    )
    for user in sorted(by_user):
        distinct = list(dict.fromkeys(by_user[user]))
        all_histories[user] = distinct
        rng = np.random.default_rng(seed * 1_000_003 + user)
        shuffled = np.asarray(distinct, dtype=np.int64)
        rng.shuffle(shuffled)
        target_count = max(1, int(math.ceil(0.2 * len(shuffled))))
        if len(shuffled) > 1:
            target_count = min(target_count, len(shuffled) - 1)
        target_pool = shuffled[:target_count].tolist()
        excluded = set(distinct)
        for offset in range(0, len(target_pool), targets_per_slate):
            targets = target_pool[offset:offset + targets_per_slate]
            target_set = set(targets)
            history = [item for item in distinct if item not in target_set]
            negatives = _unique_negatives(
                rng, adapter.dataset.item_num, excluded, sample_num * len(targets)
            )
            candidate_items = targets + negatives
            labels = torch.tensor(
                [1.0] * len(targets) + [0.0] * len(negatives), dtype=torch.float32
            )
            candidate_tensor = torch.tensor(candidate_items, dtype=torch.long)
            samples.append({
                "user": user,
                "history": torch.tensor(history, dtype=torch.int32),
                "history_feedback": torch.ones(len(history), dtype=torch.int8),
                "items": candidate_tensor.to(torch.int32),
                "labels": labels,
            })
    # Score many users per backbone call without materializing the entire
    # dataset's repeated user vector at once.
    for first in range(0, len(samples), 512):
        group = samples[first:first + 512]
        flat_users: list[int] = []
        flat_items: list[int] = []
        for sample in group:
            flat_users.extend([sample["user"]] * len(sample["items"]))
            flat_items.extend(sample["items"].tolist())
        all_scores = adapter.score_candidates(flat_users, flat_items)
        cursor = 0
        for sample in group:
            length = len(sample["items"])
            raw_scores = all_scores[cursor:cursor + length]
            cursor += length
            # Sort without labels: item ID ascending first, then stable score descending.
            id_order = torch.argsort(sample["items"], stable=True)
            score_order = torch.argsort(raw_scores[id_order], descending=True, stable=True)
            order = id_order[score_order]
            sample["items"] = sample["items"][order]
            sample["base_score_raw"] = raw_scores[order].to(torch.float32)
            sample["labels"] = sample["labels"][order]
    lengths = [len(sample["items"]) for sample in samples]
    history_lengths = [len(sample["history"]) for sample in samples]
    summary = {
        "construction": separation,
        "users_with_samples": len(samples),
        "distinct_training_users": len(all_histories),
        "targets_per_slate_cap": targets_per_slate,
        "target_positive_count": int(sum(float(s["labels"].sum()) for s in samples)),
        "candidate_length": length_summary(lengths),
        "training_history_length": length_summary(history_lengths),
        "evaluation_history_length": length_summary([len(v) for v in all_histories.values()]),
        "negative_sampling": f"uniform without replacement, up to {sample_num} per target positive",
    }
    return samples, all_histories, summary


def length_summary(values: Iterable[int]) -> dict:
    array = np.asarray(list(values), dtype=np.int64)
    if not len(array):
        return {"min": 0, "median": 0, "max": 0, "mean": 0.0}
    return {
        "min": int(array.min()),
        "median": float(np.median(array)),
        "max": int(array.max()),
        "mean": float(array.mean()),
    }


def truncate_ranked_records(records: list[dict], limit: int) -> list[dict]:
    """Copy base-score-ranked records with only their label-blind Top-M prefix."""

    if limit < 1:
        raise ValueError("candidate limit must be positive")
    output = []
    for record in records:
        copied = dict(record)
        for key in ("items", "base_score_raw", "labels"):
            if key in copied:
                copied[key] = copied[key][:limit].clone()
        output.append(copied)
    return output


def truncate_eval_batches(batches: list[dict], limit: int) -> tuple[list[dict], list[int]]:
    output, lengths = [], []
    for cached in batches:
        groups = truncate_ranked_records(cached["groups"], limit)
        lengths.extend(len(record["items"]) for record in groups)
        output.append({
            **cached,
            "groups": groups,
            "positive_u": cached["positive_u"].clone(),
            "positive_i": cached["positive_i"].clone(),
        })
    return output, lengths


def transform_scores(
    scores: torch.Tensor, mask: torch.Tensor, transform: str
) -> torch.Tensor:
    if transform == "identity_probability":
        return torch.where(mask, scores, torch.zeros_like(scores))
    if transform == "valid_minmax":
        return masked_minmax(scores, mask)
    raise ValueError(f"unknown DNR score transform: {transform}")


def collate_slates(
    records: list[dict],
    histories: dict[int, list[int]] | None,
    *,
    item_num: int,
    score_transform: str,
    include_labels: bool,
    permute: bool = False,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    if not records:
        raise ValueError("cannot collate an empty slate batch")
    candidate_length = max(len(record["items"]) for record in records)
    if histories is None:
        history_values = [record["history"].tolist() for record in records]
        feedback_values = [record["history_feedback"].tolist() for record in records]
    else:
        history_values = [histories.get(int(record["user"]), []) for record in records]
        feedback_values = [[1] * len(value) for value in history_values]
    history_length = max(1, max(map(len, history_values), default=0))
    batch = len(records)
    users = torch.tensor([record["user"] for record in records], dtype=torch.long)
    candidate_items = torch.full((batch, candidate_length), item_num, dtype=torch.long)
    raw_scores = torch.zeros((batch, candidate_length), dtype=torch.float32)
    labels = torch.zeros((batch, candidate_length), dtype=torch.float32)
    candidate_mask = torch.zeros((batch, candidate_length), dtype=torch.bool)
    history_items = torch.full((batch, history_length), item_num, dtype=torch.long)
    history_feedback = torch.zeros((batch, history_length), dtype=torch.long)
    history_mask = torch.zeros((batch, history_length), dtype=torch.bool)
    for index, record in enumerate(records):
        length = len(record["items"])
        candidate_items[index, :length] = record["items"].long()
        raw_scores[index, :length] = record["base_score_raw"].float()
        candidate_mask[index, :length] = True
        if include_labels:
            labels[index, :length] = record["labels"].float()
        history = history_values[index]
        feedback = feedback_values[index]
        if history:
            history_items[index, :len(history)] = torch.as_tensor(history, dtype=torch.long)
            history_feedback[index, :len(history)] = torch.as_tensor(feedback, dtype=torch.long)
            history_mask[index, :len(history)] = True
        if permute and length:
            order = torch.randperm(length, generator=generator)
            candidate_items[index, :length] = candidate_items[index, :length][order]
            raw_scores[index, :length] = raw_scores[index, :length][order]
            labels[index, :length] = labels[index, :length][order]
    scores = transform_scores(raw_scores, candidate_mask, score_transform)
    return {
        "users": users,
        "history_items": history_items,
        "history_feedback": history_feedback,
        "history_mask": history_mask,
        "candidate_items": candidate_items,
        "candidate_mask": candidate_mask,
        "base_score_raw": raw_scores,
        "scores": scores,
        "labels": labels,
    }


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}
