from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

from . import JOINT_FEATURES, METRICS

ROOT = Path(__file__).resolve().parents[2]


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = fieldnames or (list(rows[0]) if rows else [])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    config["config_path"] = str(path.resolve())
    for key in ("source_table1", "source_score_degree", "source_table2", "source_degree", "source_audit", "output_base"):
        config[key] = str((ROOT / config[key]).resolve())
    return config


def now_kst() -> str:
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y%m%d_%H%M%S_KST")


def resolve_run_id(config: dict, requested: str | None, create=False) -> str:
    marker = Path(config["output_base"]) / "ACTIVE_RUN_ID"
    run_id = requested or os.environ.get("RUN_ID") or (marker.read_text().strip() if marker.is_file() else None)
    if not run_id and create:
        run_id = now_kst()
    if not run_id:
        raise RuntimeError("No active revision-evidence RUN_ID")
    if create:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(run_id + "\n", encoding="utf-8")
    return run_id


def output_root(config: dict, run_id: str) -> Path:
    return Path(config["output_base"]) / run_id


def coordinates(config: dict):
    return [(d, b, int(s)) for d in config["datasets"] for b in config["backbones"] for s in config["seeds"]]


def joint_coordinates(config: dict):
    return [(d, b, int(s)) for d in config["joint_datasets"] for b in config["joint_backbones"] for s in config["seeds"]]


def coordinate_key(row: dict):
    return row["dataset"], row["backbone"], int(row["seed"])


def source_inventory(config: dict) -> dict[tuple[str, str, int], dict]:
    rows = read_csv(Path(config["source_degree"]) / "run_inventory.csv")
    output = {coordinate_key(row): row for row in rows}
    if len(output) != 54:
        raise AssertionError(f"expected 54 degree coordinates, got {len(output)}")
    return output


def original_cache(row: dict) -> Path:
    return Path(row["source_feature_cache"])


def degree_cache(row: dict) -> Path:
    return Path(row["degree_feature_cache"])


def full_checkpoint(row: dict) -> Path:
    return Path(row["source_full_checkpoint"])


def dnr_cache(row: dict) -> Path:
    return Path(row["source_dnr_cache"])


def full_candidate_cache(row: dict) -> Path:
    dnr = torch.load(dnr_cache(row), map_location="cpu", weights_only=False, mmap=True)
    value = dnr["preparation_summary"].get("reused_full_candidate_cache")
    if not value:
        raise RuntimeError(f"full pre-Top50 candidate cache not recorded: {dnr_cache(row)}")
    return Path(value)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_digest(*values: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for value in values:
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class JointResidualMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(7, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def assert_joint_model() -> None:
    count = sum(parameter.numel() for parameter in JointResidualMLP().parameters())
    if count != 801:
        raise AssertionError(f"7-32-16-1 model has {count} parameters, expected 801")


def stable_order(scores: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
    item_order = torch.argsort(items, dim=1, stable=True)
    ordered = torch.gather(scores, 1, item_order)
    score_order = torch.argsort(ordered, dim=1, descending=True, stable=True)
    return torch.gather(item_order, 1, score_order)


def per_group_metrics(scores: torch.Tensor, split: dict) -> dict[str, torch.Tensor]:
    mask = split["valid_mask"].cpu()
    labels = split["labels"].cpu() & mask
    items = split["candidate_ids"].cpu().long()
    values = scores.detach().cpu().double().masked_fill(~mask, -torch.inf)
    order = stable_order(values, items)
    ranked = torch.gather(labels, 1, order).double()
    positive_count = split["positive_count"].cpu().double()
    result = {}
    for k in (10, 20):
        gain = ranked[:, :min(k, ranked.shape[1])]
        hits = gain.sum(1)
        result[f"recall@{k}"] = torch.where(positive_count > 0, hits / positive_count, torch.zeros_like(hits))
        discount = 1.0 / torch.log2(torch.arange(2, gain.shape[1] + 2, dtype=torch.float64))
        dcg = (gain * discount).sum(1)
        prefix = torch.cat((torch.zeros(1, dtype=torch.float64), torch.cumsum(discount, 0)))
        ideal = torch.minimum(positive_count.long(), torch.tensor(gain.shape[1]))
        idcg = prefix[ideal]
        result[f"ndcg@{k}"] = torch.where(idcg > 0, dcg / idcg, torch.zeros_like(dcg))
    return result


def ranking_metrics(scores: torch.Tensor, split: dict) -> dict[str, float]:
    return {name: float(values.mean()) for name, values in per_group_metrics(scores, split).items() if name in METRICS}


def predict_joint(model, split: dict, scaler: dict, mask, device, batch_groups=4096):
    model.eval()
    mean = scaler["mean"].to(device)
    std = scaler["std"].to(device)
    feature_mask = torch.as_tensor(mask, dtype=torch.float32, device=device)
    scale = float(scaler["score_scale"])
    output = []
    with torch.inference_mode():
        for start in range(0, len(split["s"]), batch_groups):
            end = min(len(split["s"]), start + batch_groups)
            raw = split["features"][start:end].to(device)
            z = ((raw - mean) / std) * feature_mask
            output.append((split["s"][start:end].to(device) + scale * model(z)).cpu())
    return torch.cat(output)


def load_joint_checkpoint(path: Path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = JointResidualMLP().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint

