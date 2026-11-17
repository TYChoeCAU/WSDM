"""Frozen RecBole backbone adapter used by the reproduction pipelines."""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
from typing import Iterable

import numpy as np

for _name, _value in {
    "bool": np.bool_, "int": np.int64, "float": np.float64,
    "complex": np.complex128, "object": np.object_, "str": np.str_,
    "long": np.int64, "unicode": np.str_,
}.items():
    setattr(np, _name, _value)

import torch

from recbole.data import create_dataset, data_preparation
from recbole.data.interaction import Interaction
from recbole.utils import EvaluatorType, get_model, init_seed


def load_checkpoint(path: Path, device: str | torch.device = "cpu") -> dict:
    # QNN-alpha checkpoints contain a serialized RecBole Config whose model
    # class points at this local adapter module.  Import it explicitly before
    # unpickling so resumed subprocesses do not depend on incidental prior
    # imports in the parent process.
    importlib.import_module("recbole.model.context_aware_recommender.qnn_alpha")
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def checkpoint_for(root: Path, dataset: str, model: str, seed: int) -> Path | None:
    directory = root / dataset / model / f"seed_{seed}"
    candidates = sorted(
        path for path in directory.glob("*.pth") if not path.name.endswith("Dataset.pth")
    )
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    scored = []
    for path in candidates:
        payload = load_checkpoint(path)
        score = payload.get("best_valid_score")
        if score is None:
            raise RuntimeError(f"checkpoint lacks best_valid_score: {path}")
        scored.append((float(score), str(path), path))
    return max(scored)[2]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FrozenBackboneAdapter:
    """Load the exact saved checkpoint and expose only immutable score inference."""

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        device: torch.device,
        sample_num: int = 100,
        predict_batch_size: int = 8192,
        data_root: Path | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.checkpoint_sha256 = file_sha256(self.checkpoint_path)
        payload = load_checkpoint(self.checkpoint_path)
        self.config = payload["config"]
        if data_root is not None:
            self.config["data_path"] = str(Path(data_root).resolve() / self.config["dataset"])
        self.config["save_dataset"] = False
        self.config["save_dataloaders"] = False
        self.config["use_gpu"] = device.type == "cuda"
        self.config["device"] = device
        self.config["metrics"] = ["NDCG", "Recall"]
        self.config["topk"] = [10, 20]
        self.config["metric_decimal_place"] = 10
        self.config["eval_type"] = EvaluatorType.RANKING
        self.config["show_progress"] = False
        for phase in ("valid", "test"):
            self.config[f"{phase}_neg_sample_args"] = {
                "distribution": "uniform", "sample_num": sample_num
            }
        init_seed(self.config["seed"], self.config["reproducibility"])
        complete = create_dataset(self.config)
        self.train_data, self.valid_data, self.test_data = data_preparation(
            self.config, complete
        )
        self.dataset = self.train_data._dataset
        self.model = get_model(self.config["model"])(self.config, self.dataset).to(device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.load_other_parameter(payload.get("other_parameter"))
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.device = device
        self.predict_batch_size = predict_batch_size
        self.uid_field = self.config["USER_ID_FIELD"]
        self.iid_field = self.config["ITEM_ID_FIELD"]
        self._initial_state = self.state_digest()

    def state_digest(self) -> str:
        digest = hashlib.sha256()
        for name, value in sorted(self.model.state_dict().items()):
            digest.update(name.encode("utf-8"))
            array = value.detach().cpu().contiguous().numpy()
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes())
        return digest.hexdigest()

    def assert_frozen_and_unchanged(self) -> None:
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise AssertionError("a frozen backbone parameter requires gradients")
        if self.state_digest() != self._initial_state:
            raise AssertionError("backbone parameters or buffers changed")

    @torch.inference_mode()
    def score_interaction(self, interaction: Interaction) -> torch.Tensor:
        output = []
        for start in range(0, len(interaction), self.predict_batch_size):
            end = min(start + self.predict_batch_size, len(interaction))
            output.append(self.model.predict(interaction[start:end].to(self.device)).reshape(-1))
        if not output:
            return torch.empty(0, dtype=torch.float32)
        return torch.cat(output).detach().cpu()

    @torch.inference_mode()
    def score_candidates(
        self,
        user_ids: Iterable[int] | torch.Tensor,
        item_ids: Iterable[int] | torch.Tensor,
        supported_features=None,
    ) -> torch.Tensor:
        del supported_features
        users = torch.as_tensor(user_ids, dtype=torch.long)
        items = torch.as_tensor(item_ids, dtype=torch.long)
        if users.shape != items.shape:
            raise ValueError("user_ids and item_ids must have identical shapes")
        interaction = Interaction({self.uid_field: users, self.iid_field: items})
        return self.score_interaction(self.dataset.join(interaction))

    @property
    def identity(self) -> dict:
        return {
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "dataset": self.config["dataset"],
            "model": self.config["model"],
            "seed": int(self.config["seed"]),
            "state_sha256": self._initial_state,
            "predict_batch_size": self.predict_batch_size,
        }
