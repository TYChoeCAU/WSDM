"""Metric and prediction helpers for fixed Top-50 feature caches."""

from __future__ import annotations

import math
import time

import torch

from . import METRICS
from .models import fuse_scores


def ranking_metrics(scores: torch.Tensor, split: dict, device=None, chunk_groups: int = 128) -> dict[str, float]:
    """Binary Recall/NDCG with deterministic item-ID tie handling.

    RecBole's dense ``torch.topk`` leaves exact ties unspecified.  The matched
    candidate builder documents item ID ascending as its tie rule, so we make
    that rule effective here with stable sorts.
    """
    mask = split["valid_mask"]
    labels = split["labels"]
    items = split["candidate_ids"]
    positive_count = split["positive_count"].to(torch.float64)
    del device, chunk_groups
    values = scores.detach().cpu().to(torch.float64).masked_fill(~mask, -torch.inf)
    item_order = torch.argsort(items, dim=1, stable=True)
    ordered_values = torch.gather(values, 1, item_order)
    score_order = torch.argsort(ordered_values, dim=1, descending=True, stable=True)
    order = torch.gather(item_order, 1, score_order)
    ranked_labels = torch.gather(labels & mask, 1, order).to(torch.float64)
    result = {}
    for k in (10, 20):
        effective_k = min(k, ranked_labels.shape[1])
        gain = ranked_labels[:, :effective_k]
        hits = gain.sum(dim=1)
        recall = torch.where(positive_count > 0, hits / positive_count, torch.zeros_like(hits))
        discount = 1.0 / torch.log2(torch.arange(2, effective_k + 2, dtype=torch.float64))
        dcg = (gain * discount).sum(dim=1)
        ideal_n = torch.minimum(positive_count.to(torch.long), torch.tensor(effective_k))
        prefix = torch.cat((torch.zeros(1, dtype=torch.float64), torch.cumsum(discount, 0)))
        idcg = prefix[ideal_n]
        ndcg = torch.where(idcg > 0, dcg / idcg, torch.zeros_like(dcg))
        result[f"ndcg@{k}"] = float(ndcg.mean())
        result[f"recall@{k}"] = float(recall.mean())
    return {name: result[name] for name in METRICS}


@torch.inference_mode()
def predict_split(
    model, method: str, split: dict, scaler: dict, device, batch_groups=4096,
    feature_mask=None, fusion_method=None,
):
    output, gates = [], []
    mean = scaler["mean"].to(device)
    std = scaler["std"].to(device)
    for start in range(0, len(split["s"]), batch_groups):
        end = min(start + batch_groups, len(split["s"]))
        raw = split["features"][start:end].to(device)
        standardized = (raw - mean) / std
        if feature_mask is not None:
            mask = torch.as_tensor(feature_mask, dtype=standardized.dtype, device=device)
            standardized = standardized * mask
        scores, gate = fuse_scores(
            fusion_method or method, model, standardized,
            split["s"][start:end].to(device), split["h"][start:end].to(device),
            float(scaler["score_scale"]),
        )
        output.append(scores.cpu())
        if gate is not None:
            gates.append(gate.cpu())
    return torch.cat(output), (torch.cat(gates) if gates else None)


def evaluate_model(model, method, split, scaler, device, feature_mask=None, fusion_method=None):
    started = time.perf_counter()
    scores, gates = predict_split(
        model, method, split, scaler, device,
        feature_mask=feature_mask, fusion_method=fusion_method,
    )
    return ranking_metrics(scores, split, device=device), scores, gates, time.perf_counter() - started


def assert_metrics_close(actual: dict, expected: dict, atol: float = 1e-3) -> None:
    for name in METRICS:
        if not math.isclose(float(actual[name]), float(expected[name]), rel_tol=0.0, abs_tol=atol):
            raise AssertionError(f"metric mismatch {name}: {actual[name]} != {expected[name]}")
