"""Training and common-evaluator bridge for PRM baselines."""

from __future__ import annotations
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from common.evaluation import evaluate_cached
from common.io import load_torch
from common.io import atomic_json
from common.batching import batches_for_epoch
from .input_adapter import collate,move_batch,training_scaler
from .model import PRM
from .principled_loss import masked_logloss,prm_o6_loss


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


class EvaluatedPRM(torch.nn.Module):
    def __init__(self,model,scaler): super().__init__(); self.model=model; self.scaler=scaler
    def forward(self,batch):
        batch=dict(batch); batch["score_standardized"]=(batch["base_score_raw"]-self.scaler["score_mean"])/self.scaler["score_std"]
        return self.model(batch)


def build_model(cache,applied,device):
    c=cache["model_config"]
    return PRM(c["user_num"],c["item_num"],max_len=50,embedding_dim=applied["embedding_dim"],d_model=applied["d_model"],
               d_inner=applied["d_inner"],n_head=applied["n_head"],dropout=applied["dropout"]).to(device)


def evaluate(model,cache,split,scaler,applied,device):
    return evaluate_cached(cache["recbole_config"],EvaluatedPRM(model,scaler),cache[f"{split}_batches"],cache["evaluation_histories"],
                           item_num=cache["model_config"]["item_num"],score_transform="identity_probability",device=device,
                           group_batch_size=applied["eval_group_batch_size"],attention_element_budget=applied["attention_element_budget"])


def train(cache_path,applied,run_dir,device,smoke=False):
    cache=load_torch(cache_path); seed_all(applied["seed"]); scaler=training_scaler(cache["train_samples"])
    model=build_model(cache,applied,device); optimizer=torch.optim.Adam(model.parameters(),lr=applied["learning_rate"],weight_decay=applied["weight_decay"])
    epochs=2 if smoke else applied["maximum_epochs"]; samples=cache["train_samples"]
    if smoke: samples=samples[:min(len(samples),2*applied["batch_size"])]
    history=[]; best=-float("inf"); best_epoch=None; stale=0; gradient=False; started=time.perf_counter()
    if device.type=="cuda": torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(epochs):
        model.train(); total=0.; count=0; gen=torch.Generator().manual_seed(applied["seed"]*1000+epoch)
        for records in batches_for_epoch(samples,applied["batch_size"],applied["seed"]+epoch,applied["attention_element_budget"]):
            batch=move_batch(collate(records,cache["model_config"]["item_num"],scaler,True),device); optimizer.zero_grad(set_to_none=True)
            if applied["method"]=="prm_principled": loss,parts=prm_o6_loss(model,batch,gen)
            else: loss=masked_logloss(model(batch),batch["labels"],batch["candidate_mask"]); parts=None
            loss.backward(); gradient|=any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in model.parameters())
            clip_grad_norm_(model.parameters(),applied["gradient_clip_norm"]); optimizer.step(); total+=float(loss.detach()); count+=1
        model.eval(); valid_batches=cache["valid_batches"][:1] if smoke else cache["valid_batches"]
        original=cache["valid_batches"]; cache["valid_batches"]=valid_batches
        metrics,seconds=evaluate(model,cache,"valid",scaler,applied,device); cache["valid_batches"]=original
        improved=metrics["ndcg@10"]>best+applied["selection_min_delta"]
        if improved:
            best=metrics["ndcg@10"]; best_epoch=epoch; stale=0; run_dir.mkdir(parents=True,exist_ok=True)
            torch.save({"variant":"public_benchmark_PRM_O6_port" if applied["method"]=="prm_principled" else "public_benchmark_PRM_port",
                        "method":applied["method"],"state_dict":model.state_dict(),"scaler":scaler,"applied":applied,
                        "epoch":epoch,"validation_metrics":metrics,"source_commits":applied["source_commits"]},run_dir/("smoke_checkpoint.pt" if smoke else "best_checkpoint.pt"))
        else: stale+=1
        history.append({"epoch":epoch,"loss":total/max(count,1),"validation_metrics":metrics,"validation_seconds":seconds})
        atomic_json(run_dir/("smoke_validation_history.json" if smoke else "validation_history.json"),{"epochs":history})
        if not smoke and epoch+1>=applied["minimum_epochs"] and stale>=applied["patience"]: break
    if not gradient: raise AssertionError("PRM received no finite nonzero gradient")
    summary={"method":applied["method"],"epochs_ran":len(history),"selected_epoch":best_epoch,"best_validation_ndcg@10":best,
             "training_wall_seconds":time.perf_counter()-started,"trainable_parameters":sum(p.numel() for p in model.parameters()),
             "peak_gpu_memory_bytes":int(torch.cuda.max_memory_allocated(device)) if device.type=="cuda" else 0,"gradient_verified":True}
    atomic_json(run_dir/("smoke_training_summary.json" if smoke else "training_summary.json"),summary); return summary


def load_model(checkpoint_path,cache,device):
    cp=load_torch(checkpoint_path); model=build_model(cache,cp["applied"],device); model.load_state_dict(cp["state_dict"]); model.eval(); return model,cp
