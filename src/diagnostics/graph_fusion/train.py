"""Pairwise training for the three fixed-budget diagnostic modules."""

from __future__ import annotations

import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from common.io import load_torch
from .evaluate import evaluate_model
from .models import FusionMLP, fuse_scores


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pairwise_group_loss(scores, labels, mask):
    positive = labels & mask
    negative = (~labels) & mask
    pair_mask = positive.unsqueeze(2) & negative.unsqueeze(1)
    eligible = pair_mask.flatten(1).any(1)
    if not eligible.any():
        return None, 0
    differences = scores.unsqueeze(2) - scores.unsqueeze(1)
    per_pair = torch.nn.functional.softplus(-differences)
    counts = pair_mask.sum(dim=(1, 2)).clamp_min(1)
    per_group = (per_pair * pair_mask).sum(dim=(1, 2)) / counts
    return per_group[eligible].mean(), int(eligible.sum())


def train_module(
    applied: dict, method: str, cache_path: Path, run_dir: Path, device, smoke=False,
    feature_mask=None, fusion_method=None,
):
    checkpoint_path = run_dir / ("smoke_checkpoint.pt" if smoke else "best_checkpoint.pt")
    summary_path = run_dir / ("smoke_training_summary.json" if smoke else "training_summary.json")
    if summary_path.is_file() and checkpoint_path.is_file() and not smoke:
        return json.loads(summary_path.read_text())
    cache = load_torch(cache_path)
    train = cache["train"]
    valid = cache["valid"]
    scaler = cache["scaler"]
    _seed_everything(int(applied["training_seed"]))
    model = FusionMLP().to(device)
    initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(applied["learning_rate"]),
        weight_decay=float(applied["weight_decay"]),
    )
    epochs = 2 if smoke else int(applied["maximum_epochs"])
    minimum = 1 if smoke else int(applied["minimum_epochs"])
    patience = int(applied["early_stopping_patience"])
    batch_size = int(applied["batch_size"])
    indices = torch.where(
        (train["labels"].sum(1) > 0)
        & (train["labels"].sum(1) < train["valid_mask"].sum(1))
    )[0]
    if smoke:
        indices = indices[: min(len(indices), 512)]
        limit = min(len(valid["s"]), 512)
        valid = {key: (value[:limit] if torch.is_tensor(value) else value) for key, value in valid.items()}
    mean, std = scaler["mean"].to(device), scaler["std"].to(device)
    scale = float(scaler["score_scale"])
    best_metric, best_epoch, stale = -float("inf"), None, 0
    history, gradient_seen = [], False
    started = time.perf_counter()
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(epochs):
        model.train()
        generator = torch.Generator().manual_seed(int(applied["training_seed"]) + epoch)
        order = indices[torch.randperm(len(indices), generator=generator)]
        total_loss = total_groups = 0
        for start in range(0, len(order), batch_size):
            chosen = order[start:start + batch_size]
            raw = train["features"][chosen].to(device)
            standardized = (raw - mean) / std
            if feature_mask is not None:
                mask = torch.as_tensor(feature_mask, dtype=standardized.dtype, device=device)
                standardized = standardized * mask
            scores, _ = fuse_scores(
                fusion_method or method, model, standardized, train["s"][chosen].to(device),
                train["h"][chosen].to(device), scale,
            )
            loss, groups = pairwise_group_loss(
                scores / scale, train["labels"][chosen].to(device), train["valid_mask"][chosen].to(device)
            )
            if loss is None: continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_seen |= any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in model.parameters())
            clip_grad_norm_(model.parameters(), float(applied["gradient_clip_norm"]))
            optimizer.step()
            total_loss += float(loss.detach()) * groups
            total_groups += groups
        model.eval()
        metrics, _, _, seconds = evaluate_model(
            model, method, valid, scaler, device,
            feature_mask=feature_mask, fusion_method=fusion_method,
        )
        value = metrics["ndcg@10"]
        improved = value > best_metric + 1e-4
        if improved:
            best_metric, best_epoch, stale = value, epoch, 0
            run_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "method": method, "epoch": epoch, "state_dict": model.state_dict(),
                "validation_metrics": metrics, "cache_signature": cache["signature"],
                "applied": applied, "fusion_method": fusion_method or method,
                "feature_names": list(scaler["feature_names"]),
                "feature_mask": list(feature_mask) if feature_mask is not None else None,
                "train_scaler": {"mean": scaler["mean"], "std": scaler["std"]},
                "residual_scale_t": scale,
            }, checkpoint_path)
        else:
            stale += 1
        history.append({
            "epoch": epoch, "training_loss": total_loss / max(total_groups, 1),
            "eligible_groups": total_groups, "validation_metrics": metrics,
            "validation_seconds": seconds,
        })
        atomic_json(run_dir / ("smoke_validation_history.json" if smoke else "validation_history.json"), {"epochs": history})
        if epoch + 1 >= minimum and stale >= patience:
            break
    if not gradient_seen or all(torch.equal(initial[name], model.state_dict()[name]) for name in initial):
        raise AssertionError("fusion module did not receive a finite nonzero gradient/update")
    elapsed = time.perf_counter() - started
    summary = {
        "method": method, "selected_epoch": best_epoch,
        "best_validation_ndcg@10": best_metric, "epochs_ran": len(history),
        "training_wall_seconds": elapsed,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "eligible_training_groups": len(indices), "gradient_and_update_verified": True,
    }
    atomic_json(summary_path, summary)
    return summary


def load_selected_model(checkpoint_path: Path, device):
    checkpoint = load_torch(checkpoint_path)
    model = FusionMLP().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint
