"""Corrected, shared candidate construction. Dependencies: numpy, torch, stdlib.

Outer splits and the binary relevance labels come from the original loader.
The helper never creates outer splits, scores candidates, queries a graph, or
fits a scaler. Item 0 is padding. All real item IDs are in [1, item_num).
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from typing import Mapping

import numpy as np
import torch


SAMPLER_VERSION = "choice-without-replacement-v1"
TARGET_POLICY = "rating-ge4-pair-any-relevant-original-slate-v1"
PROTOCOL_VERSION = "corrected-100-per-relevant-top50-v1"


def protocol_spec() -> dict:
    return {
        "sampler_version": SAMPLER_VERSION,
        "target_policy": TARGET_POLICY,
        "protocol_version": PROTOCOL_VERSION,
        "numpy_version": np.__version__,
        "rng": "NumPy PCG64 Generator, SHA256 group key -> SeedSequence",
        "U": "Generator.choice(eligible, size=min(requested, eligible), replace=False)",
        "W": "Generator.choice without replacement, p proportional to 1+binary all-observed train item degree",
        "equal_weight_W": "same uniform choice algorithm as U",
        "negative_request": "100 times number of unique relevant target items, capped at eligible catalog",
        "train_partition": "original first-seen distinct item order, default_rng(seed*1000003+user), shuffled 20% target pool, unchanged slate boundaries",
        "duplicate_pair_relevance": "any binary label1 on that pair, while observed history and graph retain one binary edge",
        "train_exclusion": "all observed train items, including low-rated targets",
        "valid_exclusion": "all observed train items plus current validation relevant positives",
        "test_exclusion": "all observed train items plus validation relevant positives plus current test relevant positives",
        "evaluation_history": "all observed train items only, separate from negative exclusion",
        "truth_denominator": "all unique relevant targets before Top-50 truncation",
        "top50": "raw score descending, item ID ascending, label blind, no forced positive insertion",
    }


def _numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _integer_vector(value, name: str) -> np.ndarray:
    array = _numpy(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.dtype.kind not in "biu" and (not np.all(np.isfinite(array)) or not np.all(array == np.floor(array))):
        raise ValueError(f"{name} contains non-integral IDs")
    return array.astype(np.int64, copy=False)


def _read_interactions(interactions: Mapping, item_num: int) -> tuple[dict, dict]:
    """Retain first occurrence order, aggregate binary pair relevance by OR."""
    users = _integer_vector(interactions["user_ids"], "user_ids")
    items = _integer_vector(interactions["item_ids"], "item_ids")
    labels = _numpy(interactions["labels"])
    if labels.ndim != 1 or not (len(users) == len(items) == len(labels)):
        raise ValueError("interaction columns must be equal-length vectors")
    if not np.all(np.isfinite(labels)) or not np.all((labels == 0) | (labels == 1)):
        raise ValueError("labels must be original-loader binary relevance, not raw ratings")
    if np.any(users < 0) or np.any(items < 0) or np.any(items >= item_num):
        raise ValueError("negative IDs or out-of-catalog item IDs")
    by_user: dict[int, dict[int, bool]] = defaultdict(dict)
    duplicates = conflicts = padding = 0
    seen_bits: dict[tuple[int, int], int] = {}
    for user, item, label in zip(users.tolist(), items.tolist(), labels.tolist()):
        if user == 0 or item == 0:
            padding += 1
            continue
        relevant = bool(label)
        key = (user, item)
        bit = 2 if relevant else 1
        old_bits = seen_bits.get(key, 0)
        if old_bits:
            duplicates += 1
        if old_bits in (1, 2) and old_bits != bit:
            conflicts += 1
        seen_bits[key] = old_bits | bit
        by_user[user][item] = by_user[user].get(item, False) or relevant
    return dict(by_user), {
        "input_rows": len(users), "padding_rows_ignored": padding,
        "unique_observed_pairs": sum(map(len, by_user.values())),
        "duplicate_pair_rows": duplicates, "conflicting_label_pairs": conflicts,
        "unique_relevant_pairs": sum(sum(items.values()) for items in by_user.values()),
    }


def binary_item_degree(train: Mapping, item_num: int) -> np.ndarray:
    """Number of distinct train users adjacent to each item, including label0."""
    observed, _ = _read_interactions(train, item_num)
    degree = np.zeros(item_num, dtype=np.int64)
    for values in observed.values():
        degree[np.fromiter(values, dtype=np.int64)] += 1
    return degree


def group_key(*, dataset: str, seed: int, phase: str, user: int,
              slate_index: int = 0, protocol: str = "W") -> str:
    protocol = protocol.upper()
    if protocol not in ("U", "W"):
        raise ValueError("protocol must be U or W")
    if int(seed) < 0 or int(user) < 1 or int(slate_index) < 0:
        raise ValueError("seed, user, or slate_index outside supported range")
    payload = {
        "dataset": str(dataset), "seed": int(seed), "phase": str(phase),
        "user": int(user), "slate_index": int(slate_index), "protocol": protocol,
        "sampler_version": SAMPLER_VERSION, "target_policy": TARGET_POLICY,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def group_rng(**kwargs) -> np.random.Generator:
    encoded = group_key(**kwargs).encode("utf-8")
    words = np.frombuffer(hashlib.sha256(encoded).digest(), dtype="<u4").tolist()
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(words)))


def sample_negatives(rng: np.random.Generator, *, item_num: int,
                     excluded, count: int, protocol: str,
                     item_degree=None) -> tuple[np.ndarray, dict]:
    """Choose the subset before sorting. W uses NumPy's weighted choice.

    NumPy's weighted, replace=False choice is sequential weighted sampling
    without replacement. Equal weights dispatch to the exact U implementation.
    No set oversampling, ID-prefix truncation, or low-ID shortage fill occurs.
    """
    if int(item_num) != item_num or item_num < 1:
        raise ValueError("item_num must include a nonnegative padding position")
    if int(count) != count or count < 0:
        raise ValueError("count must be a nonnegative integer")
    protocol = protocol.upper()
    if protocol not in ("U", "W"):
        raise ValueError("protocol must be U or W")
    excluded_ids = _integer_vector(list(excluded), "excluded")
    if np.any(excluded_ids < 0) or np.any(excluded_ids >= item_num):
        raise ValueError("excluded IDs outside catalog")
    eligible_mask = np.ones(item_num, dtype=bool)
    eligible_mask[0] = False
    eligible_mask[excluded_ids] = False
    eligible = np.flatnonzero(eligible_mask)
    wanted = min(int(count), len(eligible))
    meta = {
        "requested": int(count), "eligible_count": len(eligible),
        "sampled": wanted, "capped": int(count) > len(eligible),
        "protocol": protocol, "sampler_version": SAMPLER_VERSION,
    }
    probabilities = None
    if protocol == "W":
        if item_degree is None:
            raise ValueError("W requires binary all-observed train item_degree")
        degree = _numpy(item_degree)
        if degree.shape != (item_num,) or not np.all(np.isfinite(degree)) or np.any(degree < 0):
            raise ValueError("item_degree must be finite, nonnegative, and length item_num")
        weights = 1.0 + degree[eligible].astype(np.float64)
        if len(weights) and not np.all(weights == weights[0]):
            probabilities = weights / weights.sum()
    if wanted == 0:
        return np.empty(0, dtype=np.int64), meta
    sampled = rng.choice(eligible, size=wanted, replace=False, p=probabilities)
    # Canonical presentation only. The random subset has already been selected.
    return np.sort(sampled.astype(np.int64, copy=False)), meta


def _record(user, targets, negatives, history, key, meta, original_targets=None):
    items = torch.tensor(list(targets) + negatives.tolist(), dtype=torch.int32)
    labels = torch.zeros(len(items), dtype=torch.float32)
    labels[:len(targets)] = 1
    record = {
        "user": int(user), "items": items, "labels": labels,
        "history": torch.tensor(history, dtype=torch.int32),
        "history_feedback": torch.ones(len(history), dtype=torch.int8),
        "positive_items": torch.tensor(targets, dtype=torch.int32),
        "positive_count": len(targets), "group_key": key,
        "sampling_meta": meta,
    }
    if original_targets is not None:
        record["original_target_items"] = torch.tensor(original_targets, dtype=torch.int32)
    return record


def _summarize(records, *, phase: str, protocol: str, source_stats: dict) -> dict:
    positive = [int(r["positive_count"]) for r in records]
    requested_groups = sum(r["sampling_meta"]["requested"] > 0 for r in records)
    capped = sum(r["sampling_meta"]["capped"] for r in records)
    result = {
        **protocol_spec(), "phase": phase, "protocol": protocol.upper(),
        "source": source_stats, "records": len(records),
        "distinct_users": len({r["user"] for r in records}),
        "zero_positive_records": sum(n == 0 for n in positive),
        "positive_count": sum(positive),
        "negative_requested": sum(r["sampling_meta"]["requested"] for r in records),
        "negative_sampled": sum(r["sampling_meta"]["sampled"] for r in records),
        "capped_groups": capped, "groups_with_nonzero_request": requested_groups,
        "cap_fraction_all_groups": capped / max(1, len(records)),
        "cap_fraction_requested_groups": capped / max(1, requested_groups),
    }
    if phase == "train":
        original = sum(len(r["original_target_items"]) for r in records)
        result.update(original_target_count=original,
                      low_relevance_targets_removed=original - sum(positive),
                      zero_positive_record_policy="retained empty for accounting, excluded from pairwise fit")
    return result


def build_training_records(train: Mapping, *, item_num: int, dataset: str,
                           seed: int, protocol: str, item_degree,
                           targets_per_slate: int,
                           negatives_per_positive: int = 100):
    """Preserve historical partition/slates, then filter relevance inside each."""
    if targets_per_slate < 1 or int(targets_per_slate) != targets_per_slate:
        raise ValueError("targets_per_slate must be a positive integer")
    if negatives_per_positive < 0 or int(negatives_per_positive) != negatives_per_positive:
        raise ValueError("negatives_per_positive must be a nonnegative integer")
    by_user, stats = _read_interactions(train, item_num)
    histories, records = {}, []
    for user in sorted(by_user):
        distinct = list(by_user[user])
        histories[user] = distinct
        # This RNG and item order are the ORIGINAL partition rule. Negative
        # draws use a separate group RNG and cannot perturb future partitions.
        partition_rng = np.random.default_rng(int(seed) * 1_000_003 + user)
        shuffled = np.asarray(distinct, dtype=np.int64)
        partition_rng.shuffle(shuffled)
        target_count = max(1, int(math.ceil(0.2 * len(shuffled))))
        if len(shuffled) > 1:
            target_count = min(target_count, len(shuffled) - 1)
        pool = shuffled[:target_count].tolist()
        for offset in range(0, len(pool), targets_per_slate):
            original_targets = pool[offset:offset + targets_per_slate]
            targets = [item for item in original_targets if by_user[user][item]]
            target_set = set(original_targets)
            history = [item for item in distinct if item not in target_set]
            key_args = dict(dataset=dataset, seed=seed, phase="train", user=user,
                            slate_index=offset // targets_per_slate, protocol=protocol)
            negatives, meta = sample_negatives(
                group_rng(**key_args), item_num=item_num, excluded=distinct,
                count=negatives_per_positive * len(targets), protocol=protocol,
                item_degree=item_degree)
            meta.update(original_target_count=len(original_targets),
                        relevant_target_count=len(targets), slate_offset=offset)
            records.append(_record(user, targets, negatives, history,
                                   group_key(**key_args), meta, original_targets))
    summary = _summarize(records, phase="train", protocol=protocol, source_stats=stats)
    summary["targets_per_slate_cap"] = int(targets_per_slate)
    summary["negatives_per_positive"] = int(negatives_per_positive)
    return records, histories, summary


def build_evaluation_records(train: Mapping, valid: Mapping, test: Mapping, *,
                             phase: str, item_num: int, dataset: str, seed: int,
                             protocol: str, item_degree,
                             negatives_per_positive: int = 100):
    """Create complete heldout positive pools with phase-correct exclusions.

    The original ranking loader has already filtered validation/test to
    label1. Reject unfiltered input instead of silently changing its semantics.
    Train/history remains all observed, regardless of binary relevance.
    """
    if phase == "validation":
        phase = "valid"
    if phase not in ("valid", "test"):
        raise ValueError("evaluation phase must be valid or test")
    if negatives_per_positive < 0 or int(negatives_per_positive) != negatives_per_positive:
        raise ValueError("negatives_per_positive must be a nonnegative integer")
    observed, train_stats = _read_interactions(train, item_num)
    valid_by_user, valid_stats = _read_interactions(valid, item_num)
    test_by_user, test_stats = _read_interactions(test, item_num)
    for values in (valid_by_user, test_by_user):
        if any(not label for items in values.values() for label in items.values()):
            raise ValueError("heldout input must be original-loader relevant-only rows")
    truth = valid_by_user if phase == "valid" else test_by_user
    histories = {user: list(items) for user, items in observed.items()}
    records = []
    for user in sorted(truth):
        targets = list(truth[user])
        excluded = set(histories.get(user, [])) | set(targets)
        if phase == "test":
            excluded.update(valid_by_user.get(user, {}))
        key_args = dict(dataset=dataset, seed=seed, phase=phase, user=user,
                        slate_index=0, protocol=protocol)
        negatives, meta = sample_negatives(
            group_rng(**key_args), item_num=item_num, excluded=excluded,
            count=negatives_per_positive * len(targets), protocol=protocol,
            item_degree=item_degree)
        records.append(_record(user, targets, negatives, histories.get(user, []),
                               group_key(**key_args), meta))
    stats = {"train": train_stats, "valid": valid_stats, "test": test_stats}
    summary = _summarize(records, phase=phase, protocol=protocol, source_stats=stats)
    summary["negatives_per_positive"] = int(negatives_per_positive)
    return records, histories, summary


def rank_record(record: Mapping, raw_scores, top_m: int = 50) -> dict:
    """Score-desc/ID-asc Top-M. Preserve full pre-truncation truth denominator."""
    if top_m < 1 or int(top_m) != top_m:
        raise ValueError("top_m must be a positive integer")
    items = torch.as_tensor(record["items"]).detach().cpu()
    scores = torch.as_tensor(raw_scores).detach().cpu().reshape(-1)
    labels = torch.as_tensor(record["labels"]).detach().cpu().reshape(-1)
    if items.ndim != 1 or len(scores) != len(items) or len(labels) != len(items):
        raise ValueError("candidate IDs, scores, and labels must align")
    if len(items) != len(torch.unique(items)) or bool((items <= 0).any()):
        raise ValueError("candidates must be unique real item IDs")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("non-finite raw backbone scores")
    if "positive_items" not in record or "positive_count" not in record:
        raise ValueError("full relevant positive set is required")
    positives = torch.as_tensor(record["positive_items"]).detach().cpu().reshape(-1)
    if len(positives) != len(torch.unique(positives)) or int(record["positive_count"]) != len(positives):
        raise ValueError("positive_count must equal full unique relevant truth set")
    if not torch.equal(labels.bool(), torch.isin(items, positives)):
        raise ValueError("candidate labels disagree with full relevant truth set")
    id_order = torch.argsort(items, stable=True)
    order = id_order[torch.argsort(scores[id_order], descending=True, stable=True)][:top_m]
    output = dict(record)
    output.update(items=items[order].clone(), labels=labels[order].clone(),
                  base_score_raw=scores[order].to(torch.float32),
                  positive_items=positives.clone(), positive_count=len(positives))
    retained = int(output["labels"].sum())
    output["ranking_meta"] = {
        "pre_top_m_candidates": len(items), "retained_candidates": len(order),
        "full_positive_count": len(positives), "retained_positives": retained,
        "pairwise_eligible": retained > 0 and retained < len(order),
        "tie_rule": "score_desc_item_id_asc", "top_m": int(top_m),
    }
    return output
