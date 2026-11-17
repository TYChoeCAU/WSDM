"""Connect existing matched Top-50 DNR caches to PRM."""

import torch
from baselines.dnr.data_adapter import collate_slates, move_batch


def training_scaler(samples):
    values=torch.cat([x["base_score_raw"].float() for x in samples])
    return {"score_mean":float(values.mean()),"score_std":max(float(values.std(unbiased=False)),1e-6)}


def collate(records, item_num, scaler, include_labels=True):
    batch=collate_slates(records,None,item_num=item_num,score_transform="identity_probability",include_labels=include_labels)
    batch["score_standardized"]=(batch["base_score_raw"]-scaler["score_mean"])/scaler["score_std"]
    return batch


__all__=["collate","move_batch","training_scaler"]
