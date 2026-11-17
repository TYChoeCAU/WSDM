"""Published shared candidate and residual functions, with legacy orchestration removed."""
from __future__ import annotations
import hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from . import JOINT_FEATURES, METRICS
from .common import *
from diagnostics.graph_fusion.train import pairwise_group_loss

def joint_applied(config: dict, coord, inventory: dict, method: str) -> dict:
    checkpoint = torch.load(full_checkpoint(inventory[coord]), map_location="cpu", weights_only=False)
    applied = dict(checkpoint["applied"])
    applied.update({
        "version": config["version"], "method": method, "feature_order": JOINT_FEATURES,
        "feature_mask": config["joint_methods"][method], "training_seed": coord[2],
        "selection_metric": "validation ndcg@10", "selection_min_delta": 1e-4,
        "source_full_applied": checkpoint["applied"],
    })
    return applied


def joint_run_dir(config: dict, run_id: str, coord, method: str) -> Path:
    return output_root(config, run_id) / "runs" / coord[0] / coord[1] / f"seed_{coord[2]}" / method


def train_joint_one(config: dict, run_id: str, coord, method: str, cache: dict, inventory: dict, device, smoke=False) -> dict:
    run = output_root(config, run_id) / "smoke" / method if smoke else joint_run_dir(config, run_id, coord, method)
    checkpoint_path = run / ("smoke_checkpoint.pt" if smoke else "best_checkpoint.pt")
    summary_path = run / ("smoke_training_summary.json" if smoke else "training_summary.json")
    result_path = run / "result.json"
    applied = joint_applied(config, coord, inventory, method)
    if result_path.is_file() and checkpoint_path.is_file() and not smoke:
        return json.loads(result_path.read_text())
    seed_everything(coord[2])
    model = JointResidualMLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(applied["learning_rate"]), weight_decay=float(applied["weight_decay"]))
    train, valid = cache["train"], cache["valid"]
    indices = torch.where((train["labels"].sum(1) > 0) & (train["labels"].sum(1) < train["valid_mask"].sum(1)))[0]
    epochs = 2 if smoke else int(applied["maximum_epochs"])
    minimum = 1 if smoke else int(applied["minimum_epochs"])
    if smoke:
        indices = indices[:min(512, len(indices))]
        valid = {key: value[:min(512, len(valid["s"]))] if torch.is_tensor(value) else value for key, value in valid.items()}
    mask = torch.as_tensor(config["joint_methods"][method], dtype=torch.float32, device=device)
    mean, std = cache["scaler"]["mean"].to(device), cache["scaler"]["std"].to(device)
    scale = float(cache["scaler"]["score_scale"])
    best, best_epoch, stale = -float("inf"), None, 0
    history, gradient_seen = [], False
    started = time.perf_counter()
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(epochs):
        model.train()
        generator = torch.Generator().manual_seed(coord[2] + epoch)
        order = indices[torch.randperm(len(indices), generator=generator)]
        loss_sum = groups_sum = 0
        for start in range(0, len(order), int(applied["batch_size"])):
            chosen = order[start:start+int(applied["batch_size"])]
            z = ((train["features"][chosen].to(device) - mean) / std) * mask
            scores = train["s"][chosen].to(device) + scale * model(z)
            loss, groups = pairwise_group_loss(scores / scale, train["labels"][chosen].to(device), train["valid_mask"][chosen].to(device))
            if loss is None: continue
            optimizer.zero_grad(set_to_none=True); loss.backward()
            gradient_seen |= any(parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0 for parameter in model.parameters())
            clip_grad_norm_(model.parameters(), float(applied["gradient_clip_norm"])); optimizer.step()
            loss_sum += float(loss.detach()) * groups; groups_sum += groups
        scores = predict_joint(model, valid, cache["scaler"], config["joint_methods"][method], device)
        metrics = ranking_metrics(scores, valid)
        improved = metrics["ndcg@10"] > best + 1e-4
        if improved:
            best, best_epoch, stale = metrics["ndcg@10"], epoch, 0
            run.mkdir(parents=True, exist_ok=True)
            torch.save({
                "method": method, "epoch": epoch, "state_dict": model.state_dict(), "validation_metrics": metrics,
                "cache_signature": cache["signature"], "applied": applied, "feature_names": JOINT_FEATURES,
                "feature_mask": config["joint_methods"][method],
                "train_scaler": {"mean": cache["scaler"]["mean"], "std": cache["scaler"]["std"]},
                "residual_scale_t": scale,
            }, checkpoint_path)
        else:
            stale += 1
        history.append({"epoch": epoch, "training_loss": loss_sum/max(groups_sum, 1), "eligible_groups": groups_sum, "validation_metrics": metrics})
        atomic_json(run / ("smoke_validation_history.json" if smoke else "validation_history.json"), {"epochs": history})
        if epoch + 1 >= minimum and stale >= int(applied["early_stopping_patience"]): break
    if not gradient_seen:
        raise AssertionError("joint module received no finite nonzero gradient")
    elapsed = time.perf_counter() - started
    summary = {"method": method, "selected_epoch": best_epoch, "best_validation_ndcg@10": best, "epochs_ran": len(history), "training_wall_seconds": elapsed, "trainable_parameters": 801, "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0, "gradient_and_update_verified": True}
    atomic_json(summary_path, summary)
    if smoke:
        return summary
    selected, checkpoint = load_joint_checkpoint(checkpoint_path, device)
    test_scores = predict_joint(selected, cache["test"], cache["scaler"], config["joint_methods"][method], device)
    test_metrics = ranking_metrics(test_scores, cache["test"])
    payload = {
        "dataset": coord[0], "backbone": coord[1], "seed": coord[2], "method": method, "status": "EVALUATED",
        "metrics": test_metrics, "validation_metrics": checkpoint["validation_metrics"], "selected_epoch": checkpoint["epoch"],
        "feature_order": JOINT_FEATURES, "feature_mask": config["joint_methods"][method],
        "residual_scale_t": scale, "training_wall_seconds": elapsed, "checkpoint": str(checkpoint_path),
    }
    atomic_json(result_path, payload)
    torch.save({"scores": test_scores, "user_ids": cache["test"]["user_ids"], "candidate_ids": cache["test"]["candidate_ids"], "valid_mask": cache["test"]["valid_mask"], "labels": cache["test"]["labels"], "positive_count": cache["test"]["positive_count"]}, run / "test_predictions.pt")
    return payload


def positive_records(batches: list[dict]):
    records = []
    for batch in batches:
        for row_index, group in enumerate(batch["groups"]):
            records.append((int(group["user"]), group, batch["positive_i"][batch["positive_u"] == row_index].long()))
    return records


def train_degrees(dnr: dict):
    user_num, item_num = int(dnr["model_config"]["user_num"]), int(dnr["model_config"]["item_num"])
    user_degree = np.zeros(user_num, dtype=np.int64); item_degree = np.zeros(item_num, dtype=np.int64)
    histories = {}
    for user, items in dnr["evaluation_histories"].items():
        unique = np.asarray(sorted(set(map(int, items))), dtype=np.int64)
        histories[int(user)] = set(unique.tolist()); user_degree[int(user)] = len(unique); item_degree[unique] += 1
    return histories, user_degree, item_degree


def sample_unique_negatives(rng, item_num: int, count: int, excluded: set[int], weights: np.ndarray | None):
    available = (item_num - 1) - sum(1 <= item < item_num for item in excluded)
    target = min(count, available)
    if target <= 0:
        return np.empty(0, dtype=np.int64), available < count
    chosen = set()
    attempts = 0
    while len(chosen) < target and attempts < 30:
        need = target - len(chosen)
        draw_count = max(need * 3, 128)
        if weights is None:
            draw = rng.integers(1, item_num, size=draw_count)
        else:
            draw = rng.choice(item_num - 1, size=draw_count, replace=True, p=weights) + 1
        for item in draw.tolist():
            if item not in excluded:
                chosen.add(int(item))
                if len(chosen) == target: break
        attempts += 1
    if len(chosen) < target:
        eligible = np.asarray([item for item in range(1, item_num) if item not in excluded and item not in chosen], dtype=np.int64)
        need = target - len(chosen)
        if weights is None:
            extra = rng.choice(eligible, size=need, replace=False)
        else:
            probability = weights[eligible-1]; probability = probability/probability.sum()
            extra = rng.choice(eligible, size=need, replace=False, p=probability)
        chosen.update(map(int, extra.tolist()))
    return np.asarray(sorted(chosen), dtype=np.int64), available < count


def group_rng(dataset: str, seed: int, user: int, protocol: str):
    digest = hashlib.blake2b(f"{dataset}|{seed}|{user}|{protocol}".encode(), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "little"))


def factor_relation(user: int, items: np.ndarray, factors: dict):
    q = factors["user_sum"][user] * factors["item_sum"][items]
    denom = factors["user_sq"][user] * factors["item_sq"][items]
    n_eff = np.zeros_like(q)
    valid = denom > 0
    n_eff[valid] = np.square(q[valid]) / denom[valid]
    return np.log1p(q).astype(np.float32), n_eff.astype(np.float32)


def top50_from_records(adapter, records, dataset, seed, protocol, histories, valid_map, item_degree, factors, batch_groups=256):
    item_num = len(item_degree)
    weights = None
    if protocol == "P_control":
        weights = np.power(item_degree[1:].astype(np.float64) + 1.0, 1.0); weights /= weights.sum()
    output = []
    short = 0
    for first in range(0, len(records), batch_groups):
        batch = records[first:first+batch_groups]
        pools = []
        for user, original, positives in batch:
            positive_set = set(map(int, positives.tolist()))
            original_items = set(map(int, original["items"].tolist()))
            target_negatives = len(original_items - positive_set)
            excluded = set(histories.get(user, set())) | set(valid_map.get(user, set())) | positive_set | {0}
            sampled, limited = sample_unique_negatives(group_rng(dataset, seed, user, protocol), item_num, target_negatives, excluded, weights)
            short += int(limited)
            pool = np.asarray(sorted(positive_set | set(map(int, sampled.tolist()))), dtype=np.int64)
            pools.append(pool)
        flat_users = torch.cat([torch.full((len(pool),), user, dtype=torch.long) for (user,_,_),pool in zip(batch,pools)])
        flat_items = torch.from_numpy(np.concatenate(pools)).long()
        flat_scores = adapter.score_candidates(flat_users, flat_items).cpu()
        cursor = 0
        for (user,_,positives), pool in zip(batch,pools):
            length = len(pool); scores = flat_scores[cursor:cursor+length]; cursor += length
            items = torch.from_numpy(pool).long()
            item_order = torch.argsort(items, stable=True); ordered=scores[item_order]; score_order=torch.argsort(ordered,descending=True,stable=True); order=item_order[score_order[:min(50,length)]]
            selected_items=items[order]; selected_scores=scores[order]
            if length < 50:
                pad=50-length; selected_items=torch.cat((selected_items,torch.zeros(pad,dtype=torch.long))); selected_scores=torch.cat((selected_scores,torch.zeros(pad)))
            valid=torch.arange(50)<min(50,length); labels=torch.isin(selected_items,positives)
            logq,neff=factor_relation(user,selected_items.numpy(),factors)
            output.append({"user":user,"items":selected_items.int(),"s":selected_scores.float(),"valid":valid,"labels":labels & valid,"positive_count":len(positives),"logq":torch.from_numpy(logq),"neff":torch.from_numpy(neff),"target_negatives":target_negatives})
    return output, short


def protocol_split(records, item_log_degree):
    return {
        "user_ids": torch.tensor([row["user"] for row in records],dtype=torch.int32),
        "candidate_ids": torch.stack([row["items"] for row in records]),
        "s": torch.stack([row["s"] for row in records]),
        "valid_mask": torch.stack([row["valid"] for row in records]),
        "labels": torch.stack([row["labels"] for row in records]),
        "positive_count": torch.tensor([row["positive_count"] for row in records]),
        "logq": torch.stack([row["logq"] for row in records]), "neff": torch.stack([row["neff"] for row in records]),
        "item_degree": torch.stack([torch.from_numpy(item_log_degree[row["items"].numpy()]).float() for row in records]),
    }

