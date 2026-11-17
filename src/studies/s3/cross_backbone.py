"""A-only original NFM/BPR models with common fixed-W NDCG@20 selection.

No command starts fitting unless S3_ALLOW_TRAINING=1.  This is a
run-local bridge to the original model, loss, DataLoader and training epoch.
The separate W/F rerankers share this one selected backbone checkpoint.
"""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import logging
import os
import random
import time
from pathlib import Path
import numpy as np
from .cross_common import RUN, ROOT, identity, atomic_json, read_csv, write_csv, load, save, sha, deadline, coord
from .master import prepare_master, tensor_digest
import torch
from recbole.data.interaction import Interaction
from recbole.data.dataloader import TrainDataLoader
from recbole.utils import get_model, init_seed, EvaluatorType, ModelType, InputType
from recbole.trainer import Trainer
from diagnostics.graph_fusion.evaluate import ranking_metrics


def model_state_digest(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode()); h.update(value.numpy().tobytes())
    return h.hexdigest()


def _historical_config(dataset, backbone):
    from common.runtime import restore_config
    from .cross_common import CONFIG, DATA_ROOT
    if backbone != "NFM":
        raise ValueError("Only NFM supplies the shared fixed loader configuration")
    path = CONFIG / 'master' / (dataset + '_config.json')
    return restore_config(json.loads(path.read_text()), DATA_ROOT), {"configuration": str(path), "sha256": sha(path), "weights_reused": False}


def frozen_profile(dataset, backbone):
    from .cross_common import CONFIG
    path = CONFIG / 'backbones' / dataset / backbone / 'seed_2020.json'
    profile = copy.deepcopy(json.loads(path.read_text())['profile'])
    if profile['train_batch_size'] != 4096:
        raise ValueError("Only the recorded R0 backbone profile is permitted")
    return profile


def config_for(dataset, backbone, seed, device):
    c, _ = _historical_config(dataset, "NFM" if backbone == "MF_BPR" else backbone)
    c["seed"] = int(seed); c["device"] = torch.device(device); c["use_gpu"] = torch.device(device).type == "cuda"
    c["checkpoint_dir"] = str(coord(dataset, backbone, seed, area="backbones"))
    c["save_dataset"] = False; c["save_dataloaders"] = False; c["log_wandb"] = False
    c["metrics"] = ["NDCG", "Recall"]; c["topk"] = [10, 20]
    c["eval_type"] = EvaluatorType.RANKING; c["valid_metric"] = "NDCG@20"; c["valid_metric_bigger"] = True
    c["worker"] = 0; c["show_progress"] = False
    if backbone == "MF_BPR":
        profile = frozen_profile(dataset, backbone)
        c["model"] = "BPR"; c["MODEL_TYPE"] = ModelType.GENERAL; c["MODEL_INPUT_TYPE"] = InputType.PAIRWISE
        c["embedding_size"] = 64; c["NEG_PREFIX"] = "neg_"
        c["learner"] = "adam"; c["learning_rate"] = profile["learning_rate"]
        c["weight_decay"] = profile["weight_decay"]; c["train_batch_size"] = 4096
        c["epochs"] = 500; c["eval_step"] = 10; c["stopping_step"] = 10
        c["clip_grad_norm"] = None
        c["train_neg_sample_args"] = {"distribution": "uniform", "sample_num": 1,
                                      "dynamic": False, "candidate_num": 0, "alpha": 1.0}
    return c


class ProbabilityBPR(torch.nn.Module):
    """Frozen BPR probability view for original GraphReranker.predict calls."""
    def __init__(self, native):
        super().__init__(); self.native = native
    def forward(self, interaction):
        return torch.sigmoid(self.native.predict(interaction))
    def predict(self, interaction):
        return self.forward(interaction)
    def full_sort_predict(self, interaction):
        return torch.sigmoid(self.native.full_sort_predict(interaction))
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("native"), name)


class CrossAdapter:
    """One frozen A-trained model. Public pair scoring always returns probability."""
    def __init__(self, model, dataset, config, backbone, checkpoint_path=None,
                 checkpoint_identity=None, freeze=True, predict_batch_size=8192):
        self.raw_model = model
        self.model = ProbabilityBPR(model) if backbone == "MF_BPR" else model
        self.probability_model = self.model
        self.dataset = dataset; self.config = config; self.backbone = backbone
        self.device = torch.device(config["device"])
        self.uid_field = config["USER_ID_FIELD"]; self.iid_field = config["ITEM_ID_FIELD"]
        self.predict_batch_size = predict_batch_size
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.checkpoint_sha256 = sha(self.checkpoint_path) if self.checkpoint_path else None
        self.checkpoint_hash = self.checkpoint_sha256
        self.checkpoint_identity = checkpoint_identity
        self.model.eval()
        if freeze:
            for p in model.parameters():
                p.requires_grad_(False)
        self._initial_state = model_state_digest(self.model) if freeze else None

    def state_digest(self):
        return model_state_digest(self.model)

    def assert_frozen(self):
        if any(p.requires_grad for p in self.model.parameters()):
            raise AssertionError("Frozen backbone has trainable parameters")
        if self.state_digest() != self._initial_state:
            raise AssertionError("Frozen backbone parameters or buffers changed")

    assert_frozen_and_unchanged = assert_frozen

    @property
    def identity(self):
        return {"checkpoint": str(self.checkpoint_path), "checkpoint_sha256": self.checkpoint_sha256,
                "state_sha256": self._initial_state, "dataset": self.config["dataset"],
                "model": self.backbone, "seed": int(self.config["seed"]),
                "score_domain": "sigmoid(raw model logit)", "matched_identity": self.checkpoint_identity}

    @torch.inference_mode()
    def score_interaction(self, interaction):
        result = []
        self.model.eval()
        for start in range(0, len(interaction), self.predict_batch_size):
            batch = interaction[start:start + self.predict_batch_size].to(self.device)
            prediction = self.model.predict(batch).reshape(-1)
            result.append(prediction.detach().cpu())
        return torch.cat(result) if result else torch.empty(0, dtype=torch.float32)

    @torch.inference_mode()
    def score_candidates(self, user_ids, item_ids, supported_features=None):
        del supported_features
        users = torch.as_tensor(user_ids, dtype=torch.long).cpu()
        items = torch.as_tensor(item_ids, dtype=torch.long).cpu()
        if users.shape != items.shape:
            raise ValueError("User/item shape mismatch")
        shape = users.shape; result = []
        users = users.reshape(-1); items = items.reshape(-1)
        for start in range(0, len(users), self.predict_batch_size):
            interaction = Interaction({self.uid_field: users[start:start + self.predict_batch_size],
                                       self.iid_field: items[start:start + self.predict_batch_size]})
            if self.backbone != "MF_BPR":
                interaction = self.dataset.join(interaction)
            result.append(self.score_interaction(interaction))
        return (torch.cat(result) if result else torch.empty(0, dtype=torch.float32)).reshape(shape)

    @torch.inference_mode()
    def score_block(self, user_ids, item_ids):
        """Catalog block scores with the same fixed sigmoid probability domain."""
        users = torch.as_tensor(user_ids, dtype=torch.long).reshape(-1)
        items = torch.as_tensor(item_ids, dtype=torch.long).reshape(-1)
        self.model.eval()
        if self.backbone == "MF_BPR":
            logits = self.raw_model.user_embedding(users.to(self.device)) @ self.raw_model.item_embedding(items.to(self.device)).T
            return torch.sigmoid(logits).cpu()
        return self.score_candidates(users.repeat_interleave(len(items)), items.repeat(len(users))).reshape(len(users), len(items))


def validation_digest(records):
    h = hashlib.sha256()
    for r in records:
        h.update(str(int(r["user"])).encode())
        for key in ("items", "labels", "positive_items"):
            a = np.asarray(r[key]); h.update(key.encode()); h.update(str(a.dtype).encode()); h.update(a.tobytes())
        h.update(str(int(r["positive_count"])).encode())
    return h.hexdigest()


@torch.inference_mode()
def evaluate_fixed_W(adapter, records, group_batch=128):
    """Original common evaluator, identical untruncated positive denominators."""
    aggregate = collections.defaultdict(float)
    n = 0
    for start in range(0, len(records), group_batch):
        deadline()
        batch = records[start:start + group_batch]
        lengths = [len(r["items"]) for r in batch]
        users = np.concatenate([np.full(k, int(r["user"]), np.int64) for k, r in zip(lengths, batch)])
        items = np.concatenate([np.asarray(r["items"], dtype=np.int64) for r in batch])
        raw = adapter.score_candidates(users, items).numpy()
        scores = torch.full((len(batch), 50), -torch.inf)
        ids = torch.zeros((len(batch), 50), dtype=torch.long)
        labels = torch.zeros((len(batch), 50), dtype=torch.bool)
        mask = torch.zeros_like(labels)
        positives = torch.tensor([r["positive_count"] for r in batch], dtype=torch.long)
        offset = 0
        for j, (r, count) in enumerate(zip(batch, lengths)):
            s = raw[offset:offset + count]; it = items[offset:offset + count]
            order = np.lexsort((it, -s))[:50]; k = len(order)
            scores[j, :k] = torch.from_numpy(s[order].copy())
            ids[j, :k] = torch.from_numpy(it[order].copy())
            labels[j, :k] = torch.as_tensor(np.asarray(r["labels"])[order].copy(), dtype=torch.bool)
            mask[j, :k] = True; offset += count
        values = ranking_metrics(scores, {"candidate_ids": ids, "labels": labels,
                                  "valid_mask": mask, "positive_count": positives})
        for k, v in values.items():
            aggregate[k] += v * len(batch)
        n += len(batch)
    if not n:
        raise RuntimeError("No validation users")
    return {k: v / n for k, v in aggregate.items()}


def _rng_state(loader=None):
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "loader": loader.generator.get_state() if loader is not None else None}


def _restore_rng(state, loader=None):
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if loader is not None and state["loader"] is not None:
        loader.generator.set_state(state["loader"])


class DeadlineLoader:
    def __init__(self, loader):
        self.loader = loader; self.shuffle = loader.shuffle; self.sampler = loader.sampler
    def __len__(self):
        return len(self.loader)
    def __iter__(self):
        for batch in self.loader:
            deadline()
            yield batch


def _bpr_epoch(model, bundle, optimizer, device, seed, epoch):
    # This retains the validated legacy BPR epoch, now with A-only positives.
    init_seed(seed * 10000 + epoch, True)
    a = bundle["train"]; positive = a["labels"] > 0
    users = a["user_ids"][positive].to(device); items = a["item_ids"][positive].to(device)
    ni = bundle["item_num"]
    observed = bundle["outer_train"]
    known = (observed["user_ids"] * ni + observed["item_ids"]).unique(sorted=True).to(device)
    known = torch.cat((known, torch.tensor([bundle["user_num"] * ni + ni], device=device)))
    model.train(); order = torch.randperm(len(users), device=device)
    total = 0.; batches = 0
    for start in range(0, len(order), 4096):
        deadline(); ix = order[start:start + 4096]; u = users[ix]; p = items[ix]
        negative = torch.randint(1, ni, (len(ix),), device=device)
        for _ in range(10000):
            encoded = u * ni + negative
            bad = known[torch.searchsorted(known, encoded)] == encoded
            if not bad.any():
                break
            negative[bad] = torch.randint(1, ni, (int(bad.sum()),), device=device)
        else:
            raise RuntimeError("No eligible BPR negative after excluding A+B observations")
        interaction = Interaction({model.USER_ID: u, model.ITEM_ID: p, model.NEG_ITEM_ID: negative})
        optimizer.zero_grad(set_to_none=True); loss = model.calculate_loss(interaction)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite BPR loss")
        loss.backward(); optimizer.step(); total += float(loss.detach()); batches += 1
    return total / max(batches, 1)


def train_backbone(dataset, backbone, seed, device="cuda"):
    if os.environ.get("S3_ALLOW_TRAINING") != "1":
        raise RuntimeError("Backbone fitting requires explicit full reproduction invocation")
    deadline(); bundle = prepare_master(dataset)
    from .cross_candidates import get_pools
    pools = get_pools(dataset); records = pools["valid"]
    data = load(RUN / "master" / dataset / "dataset.pt")
    profile = frozen_profile(dataset, backbone)
    signature = {"dataset": dataset, "backbone": backbone, "training_seed": int(seed),
                 "master_signature": bundle["signature"]["sha256"], "profile": profile,
                 "validation_W_candidate_sha256": validation_digest(records),
                 "validation_pool_signature": pools.get("signature"),
                 "candidate_seed": 2020, "A_dataset_sha256": sha(RUN / "master" / dataset / "dataset.pt"),
                 "score_domain": "sigmoid(raw model logit)", "W_F_checkpoint_shared": True}
    signature["sha256"] = identity(signature)
    out = coord(dataset, backbone, seed, area="backbones"); out.mkdir(parents=True, exist_ok=True)
    if (out / "result.json").is_file():
        result = json.loads((out / "result.json").read_text())
        if result["identity"] != signature or sha(out / "best_checkpoint.pt") != result["checkpoint_sha256"]:
            raise RuntimeError("Existing completed backbone differs from frozen conditions")
        return result
    fixed = out / "fixed_settings.json"
    if fixed.is_file() and json.loads(fixed.read_text()) != signature:
        raise RuntimeError("Existing backbone fitting settings mismatch")
    atomic_json(fixed, signature)
    config = config_for(dataset, backbone, seed, device); data.config = config
    init_seed(int(seed), config["reproducibility"])
    model = get_model(config["model"])(config, data).to(device)
    loader = trainer = None
    if backbone == "MF_BPR":
        optimizer = torch.optim.Adam(model.parameters(), lr=profile["learning_rate"], weight_decay=profile["weight_decay"])
    else:
        loader = TrainDataLoader(config, data, sampler=None, shuffle=config["shuffle"])
        # Original Trainer's SummaryWriter emits only inside this new run.
        previous_cwd = Path.cwd()
        try:
            os.chdir(out)
            trainer = Trainer(config, model)
        finally:
            os.chdir(previous_cwd)
        optimizer = trainer.optimizer
    adapter = CrossAdapter(model, data, config, backbone, freeze=False)
    best = -float("inf"); best_epoch = -1; stale = 0; curves = []; start_epoch = 1; elapsed_before = 0.
    progress = out / "progress.pt"
    if progress.is_file():
        cp = load(progress)
        if cp["identity"] != signature:
            raise RuntimeError("Progress settings mismatch")
        model.load_state_dict(cp["state_dict"]); optimizer.load_state_dict(cp["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        best = cp["best"]; best_epoch = cp["best_epoch"]; stale = cp["stale"]
        curves = cp["curves"]; start_epoch = cp["epoch"] + 1; elapsed_before = cp["elapsed_seconds"]
        _restore_rng(cp["rng"], loader)
    started = time.monotonic(); complete_reason = "MAXIMUM_EPOCH_BUDGET"
    epoch = start_epoch - 1
    patience = profile["patience_validation_checks"]
    already_stopped = stale >= patience if backbone == "MF_BPR" else stale > patience
    already_stopped &= epoch >= profile["minimum_epochs"]
    if already_stopped:
        complete_reason = "EARLY_STOPPED"
    try:
        for epoch in ([] if already_stopped else range(start_epoch, profile["maximum_epochs"] + 1)):
            deadline(); epoch_started = time.monotonic()
            if backbone == "MF_BPR":
                loss = _bpr_epoch(model, bundle, optimizer, device, int(seed), epoch)
            else:
                loss = trainer._train_epoch(DeadlineLoader(loader), epoch - 1, show_progress=False)
                if isinstance(loss, tuple):
                    loss = sum(loss)
            training_seconds = time.monotonic() - epoch_started
            row = {"dataset": dataset, "backbone": backbone, "seed": int(seed), "epoch": epoch,
                   "training_loss": float(loss), "training_seconds": training_seconds,
                   "validation_seconds": 0., "validation_ndcg@20": "", "selected": False}
            if epoch % profile["validation_interval"] == 0:
                eval_started = time.monotonic(); model.eval()
                metrics = evaluate_fixed_W(adapter, records)
                score = metrics["ndcg@20"]
                row.update({"validation_seconds": time.monotonic() - eval_started,
                            "validation_ndcg@20": score, "validation_ndcg@10": metrics["ndcg@10"]})
                if score > best:
                    best = score; best_epoch = epoch; stale = 0; row["selected"] = True
                    save(out / "best_checkpoint.pt", {"identity": signature, "config": config,
                         "state_dict": model.state_dict(), "other_parameter": model.other_parameter(),
                         "selected_epoch": epoch, "validation_metrics": metrics,
                         "training_parameters": sum(p.numel() for p in model.parameters())})
                else:
                    stale += 1
            curves.append(row); write_csv(out / "training_curves.csv", curves)
            save(progress, {"identity": signature, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "epoch": epoch, "best": best, "best_epoch": best_epoch, "stale": stale,
                            "curves": curves, "rng": _rng_state(loader),
                            "elapsed_seconds": elapsed_before + time.monotonic() - started})
            print(json.dumps(row), flush=True)
            patience = profile["patience_validation_checks"]
            stop = stale >= patience if backbone == "MF_BPR" else stale > patience
            if epoch >= profile["minimum_epochs"] and stop:
                complete_reason = "EARLY_STOPPED"
                break
    except BaseException as exc:
        atomic_json(out / "run_status.json", {"status": "INCOMPLETE", "reason": type(exc).__name__ + ": " + str(exc),
                                              "identity": signature, "latest_complete_epoch": curves[-1]["epoch"] if curves else 0})
        raise
    finally:
        if trainer is not None:
            trainer.tensorboard.close()
    cp = load(out / "best_checkpoint.pt")
    result = {"status": "COMPLETE", "completion_reason": complete_reason, "dataset": dataset,
              "backbone": backbone, "training_seed": int(seed), "identity": signature,
              "selected_epoch": best_epoch, "epochs_ran": epoch, "validation_metrics": cp["validation_metrics"],
              "checkpoint": str(out / "best_checkpoint.pt"), "checkpoint_sha256": sha(out / "best_checkpoint.pt"),
              "parameters": sum(p.numel() for p in model.parameters()),
              "wall_seconds": elapsed_before + time.monotonic() - started,
              "training_seconds": sum(x["training_seconds"] for x in curves),
              "validation_seconds": sum(x["validation_seconds"] for x in curves),
              "test_evaluated_during_training": False, "A_only_training": True,
              "W_F_shared_selected_checkpoint": True}
    atomic_json(out / "result.json", result); atomic_json(out / "run_status.json", result)
    return result


def load_adapter(dataset, backbone, seed, device="cuda", predict_batch_size=8192):
    out = coord(dataset, backbone, seed, area="backbones")
    result = json.loads((out / "result.json").read_text())
    if result["status"] != "COMPLETE":
        raise RuntimeError("Backbone fitting incomplete")
    path = out / "best_checkpoint.pt"
    if sha(path) != result["checkpoint_sha256"]:
        raise RuntimeError("Selected backbone checkpoint hash changed")
    bundle = prepare_master(dataset); cp = load(path)
    if cp["identity"]["master_signature"] != bundle["signature"]["sha256"]:
        raise RuntimeError("Backbone master split mismatch")
    if cp["identity"] != result["identity"]:
        raise RuntimeError("Checkpoint/run identity mismatch")
    data = load(RUN / "master" / dataset / "dataset.pt")
    config = copy.deepcopy(cp["config"]); config["device"] = torch.device(device)
    config["use_gpu"] = torch.device(device).type == "cuda"
    model = get_model(config["model"])(config, data).to(device)
    model.load_state_dict(cp["state_dict"]); model.load_other_parameter(cp.get("other_parameter"))
    return CrossAdapter(model, data, config, backbone, path, cp["identity"], predict_batch_size=predict_batch_size)

