"""One-pass frozen-backbone/minimal-graph feature preparation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from baselines.dnr.backbone_adapter import FrozenBackboneAdapter, file_sha256
from common.io import load_torch
from graph_rerank.graph import GraphScorerConfig, load_or_build_tables, training_graph

from .evaluate import assert_metrics_close, ranking_metrics


FEATURE_NAMES = ("s", "h", "h_minus_s", "log1p_q", "variance", "n_eff")


def cache_signature(applied: dict) -> dict:
    source = Path(applied["dnr_cache"])
    return {
        "version": 1,
        "dataset": applied["dataset"], "backbone": applied["backbone"], "seed": applied["seed"],
        "checkpoint_sha256": file_sha256(Path(applied["checkpoint"])),
        "dnr_cache_size": source.stat().st_size,
        "candidate_top_m": 50, "neighbor_k": applied["neighbor_k"],
        "score_domain": "raw frozen-backbone predict score",
    }


def _records_from_eval(batches: list[dict]):
    records, positives = [], []
    for batch in batches:
        for row, record in enumerate(batch["groups"]):
            records.append(record)
            positives.append(batch["positive_i"][batch["positive_u"] == row].long())
    return records, positives


@torch.inference_mode()
def _compute_records(adapter, user_table, item_table, records, positive_items=None, chunk_groups=128):
    width = 50
    fields = {name: [] for name in (
        "candidate_ids", "valid_mask", "s", "h", "Q", "V", "N_eff", "gamma",
        "has_graph_evidence", "labels", "positive_count", "features",
    )}
    direct_check = None
    for first in range(0, len(records), chunk_groups):
        group = records[first:first + chunk_groups]
        lengths = [len(record["items"]) for record in group]
        flat_items = np.concatenate([record["items"].long().numpy() for record in group])
        flat_users = np.concatenate([
            np.full(length, int(record["user"]), dtype=np.int64)
            for record, length in zip(group, lengths)
        ])
        flat_s = torch.cat([record["base_score_raw"].float() for record in group]).to(torch.float64)

        user_ids = user_table.ids[flat_users]
        user_weights = user_table.weights[flat_users]
        item_ids = item_table.ids[flat_items]
        item_weights = item_table.weights[flat_items]
        pair_users = np.broadcast_to(user_ids[:, :, None], (len(flat_items), user_ids.shape[1], item_ids.shape[1]))
        pair_items = np.broadcast_to(item_ids[:, None, :], pair_users.shape)
        pair_q = user_weights[:, :, None] * item_weights[:, None, :]
        valid = (pair_users >= 0) & (pair_items >= 0) & (pair_q > 0)
        prediction = np.zeros(pair_q.shape, dtype=np.float64)
        active_users = pair_users[valid].astype(np.int64, copy=False)
        active_items = pair_items[valid].astype(np.int64, copy=False)
        if len(active_users):
            prediction[valid] = adapter.score_candidates(active_users, active_items).double().numpy()
        q = np.where(valid, pair_q, 0.0)
        sum_q = q.sum(axis=(1, 2))
        sum_q_score = (q * prediction).sum(axis=(1, 2))
        sum_q_score_squared = (q * prediction * prediction).sum(axis=(1, 2))
        sum_q_squared = (q * q).sum(axis=(1, 2))
        has = sum_q > 0
        h = flat_s.numpy().copy()
        h[has] = sum_q_score[has] / sum_q[has]
        variance = np.zeros_like(sum_q)
        variance[has] = sum_q_score_squared[has] / sum_q[has] - h[has] ** 2
        if np.isnan(variance).any() or (variance < -1e-4).any():
            raise FloatingPointError("invalid graph prediction variance")
        variance = np.maximum(variance, 0.0)
        n_eff = np.zeros_like(sum_q)
        n_eff[has] = sum_q[has] ** 2 / sum_q_squared[has]
        if not all(np.isfinite(value).all() for value in (sum_q, h, variance, n_eff)):
            raise FloatingPointError("non-finite graph sufficient statistic")
        if direct_check is None and len(flat_items):
            # Direct and sufficient-statistics forms must agree before publishing a cache.
            index = int(np.flatnonzero(has)[0]) if has.any() else 0
            selected = valid[index]
            weights = pair_q[index][selected]
            scores = prediction[index][selected]
            if len(weights):
                direct_h = float(np.average(scores, weights=weights))
                direct_v = float(np.average((scores - direct_h) ** 2, weights=weights))
                direct_n = float(weights.sum() ** 2 / np.square(weights).sum())
                if not np.allclose(
                    [h[index], variance[index], n_eff[index]],
                    [direct_h, direct_v, direct_n], atol=1e-4, rtol=0.0,
                ):
                    raise AssertionError("direct graph statistics mismatch sufficient statistics")
            direct_check = True

        cursor = 0
        for local, (record, length) in enumerate(zip(group, lengths)):
            end = cursor + length
            ids = torch.zeros(width, dtype=torch.int32)
            mask = torch.zeros(width, dtype=torch.bool)
            s = torch.zeros(width, dtype=torch.float32)
            hh = torch.zeros(width, dtype=torch.float32)
            qq = torch.zeros(width, dtype=torch.float32)
            vv = torch.zeros(width, dtype=torch.float32)
            nn = torch.zeros(width, dtype=torch.float32)
            evidence = torch.zeros(width, dtype=torch.bool)
            ids[:length] = record["items"].to(torch.int32)
            mask[:length] = True
            s[:length] = flat_s[cursor:end].float()
            hh[:length] = torch.from_numpy(h[cursor:end]).float()
            qq[:length] = torch.from_numpy(sum_q[cursor:end]).float()
            vv[:length] = torch.from_numpy(variance[cursor:end]).float()
            nn[:length] = torch.from_numpy(n_eff[cursor:end]).float()
            evidence[:length] = torch.from_numpy(has[cursor:end])
            gamma = torch.clamp(qq, max=1.0)
            labels = torch.zeros(width, dtype=torch.bool)
            if positive_items is None:
                source_labels = record["labels"].bool()
                labels[:length] = source_labels
                positive_count = int(source_labels.sum())
            else:
                positive = positive_items[first + local]
                labels[:length] = torch.isin(record["items"].long(), positive)
                positive_count = len(positive)
            features = torch.stack((s, hh, hh - s, torch.log1p(qq), vv, nn), dim=-1)
            values = {
                "candidate_ids": ids, "valid_mask": mask, "s": s, "h": hh, "Q": qq,
                "V": vv, "N_eff": nn, "gamma": gamma,
                "has_graph_evidence": evidence, "labels": labels,
                "positive_count": torch.tensor(positive_count, dtype=torch.long),
                "features": features,
            }
            for name, value in values.items():
                fields[name].append(value)
            cursor = end
    return {name: torch.stack(values) for name, values in fields.items()}


def prepare_feature_cache(applied: dict, output_path: Path, device: torch.device) -> dict:
    signature = cache_signature(applied)
    if output_path.is_file():
        cached = load_torch(output_path)
        if cached.get("signature") == signature:
            return cached["summary"]
    started = time.perf_counter()
    source = load_torch(Path(applied["dnr_cache"]))
    adapter = FrozenBackboneAdapter(
        Path(applied["checkpoint"]), device=device, sample_num=100, predict_batch_size=8192,
    )
    initial_state = adapter.state_digest()
    graph = training_graph(adapter.dataset)
    candidates = sorted(Path(applied["neighbor_cache_dir"]).glob("*.npz"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one minimal neighbor cache, found {len(candidates)}")
    scorer_config = GraphScorerConfig.preset("minimal", k=int(applied["neighbor_k"]))
    user_table, item_table = load_or_build_tables(graph, candidates[0], scorer_config=scorer_config)

    train = _compute_records(adapter, user_table, item_table, source["train_samples"])
    valid_records, valid_positive = _records_from_eval(source["valid_batches"])
    test_records, test_positive = _records_from_eval(source["test_batches"])
    valid = _compute_records(adapter, user_table, item_table, valid_records, valid_positive)
    test = _compute_records(adapter, user_table, item_table, test_records, test_positive)
    for split in (train, valid, test):
        split["item_num"] = int(adapter.dataset.item_num)
    adapter.assert_frozen_and_unchanged()
    if adapter.state_digest() != initial_state:
        raise AssertionError("backbone changed during graph feature preparation")

    valid_values = train["features"][train["valid_mask"]]
    mean = valid_values.mean(dim=0)
    std = valid_values.std(dim=0, unbiased=False).clamp_min(1e-6)
    score_scale = max(float(train["s"][train["valid_mask"]].std(unbiased=False)), 1e-6)
    scaler = {"mean": mean, "std": std, "score_scale": score_scale, "feature_names": FEATURE_NAMES}
    base_metrics = ranking_metrics(test["s"], test, device=device)
    current_scores = test["s"] + test["gamma"] * (test["h"] - test["s"])
    current_metrics = ranking_metrics(current_scores, test, device=device)
    print(f"base passthrough check item_num={test.get('item_num')} metrics={base_metrics}", flush=True)
    print(f"current-gamma check metrics={current_metrics}", flush=True)
    # Historic dense torch.topk results can differ at exact ties.  Use the
    # experiment-wide numerical reproduction tolerance requested by the user.
    validation_atol = float(applied.get("numerical_validation_atol", 1e-3))
    assert_metrics_close(base_metrics, applied["expected_base_metrics"], atol=validation_atol)
    assert_metrics_close(current_metrics, applied["expected_current_metrics"], atol=validation_atol)
    summary = {
        "signature": signature, "dataset": applied["dataset"], "backbone": applied["backbone"],
        "seed": applied["seed"], "candidate_support": "label-blind base-score Top-50 only",
        "outside_top50": "excluded from ranking; original positive count retained in metric denominator",
        "score_domain": "raw frozen-backbone predict score; no extra sigmoid/minmax",
        "neighbor_k": applied["neighbor_k"], "backbone_state_before": initial_state,
        "backbone_state_after": adapter.state_digest(), "backbone_frozen": True,
        "groups": {name: len(value["s"]) for name, value in (("train", train), ("valid", valid), ("test", test))},
        "train_pairwise_excluded_no_positive": int(((train["labels"].sum(1) == 0) & train["valid_mask"].any(1)).sum()),
        "train_pairwise_excluded_no_negative": int(((train["labels"].sum(1) == train["valid_mask"].sum(1)) & train["valid_mask"].any(1)).sum()),
        "base_passthrough_test_deterministic": base_metrics,
        "legacy_base_metrics": applied["expected_base_metrics"],
        "base_topk_tie_note": "legacy CUDA torch.topk was unspecified for exact ties; diagnostic uses score-desc/item-ID-asc",
        "current_gamma_reproduction_test": current_metrics,
        "feature_preparation_seconds": time.perf_counter() - started,
        "feature_query_policy": "one graph scoring pass shared by all three learned modules",
        "train_predictions_out_of_fold": False,
    }
    payload = {"signature": signature, "summary": summary, "scaler": scaler, "train": train, "valid": valid, "test": test}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(output_path)
    return summary
