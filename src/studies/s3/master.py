"""Freeze the original NFM-2020 outer split plus an A-only fitting dataset.

This module never fits a model or scores candidates.  Binary labels already
come from the recorded loader's rating >= 4 conversion.  All writes remain in
the matched-cross run.  Original datasets and checkpoints remain read-only.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import collections
import numpy as np

for _n, _v in {"bool": np.bool_, "int": np.int64, "float": np.float64,
               "complex": np.complex128, "object": np.object_, "str": np.str_,
               "long": np.int64, "unicode": np.str_}.items():
    setattr(np, _n, _v)

from .cross_common import RUN, ROOT, CONFIG, DATA_ROOT, FIXED_DATA_SEED, identity, sha, load, save, atomic_json
import torch
from recbole.data import create_dataset
from recbole.utils import init_seed


AB_POLICY = {
    "version": "fixed-user-pair-group-90-10-v1",
    "seed": FIXED_DATA_SEED,
    "unit": "unique observed (user,item) group, all duplicate rows kept together",
    "B_count": "floor(0.1 * unique observed items) per user",
    "sparse_users": "fewer than 10 unique observed items stay entirely in A",
    "relevant_A_protection": "if every relevant pair lands in B, swap the first relevant B pair with the first nonrelevant A pair in fixed shuffled order",
    "pair_relevance": "any row with threshold-derived label 1",
    "row_order": "original outer-train order within each retained partition",
    "candidate_seed": FIXED_DATA_SEED,
    "training_seeds_change": "initialization and training order only",
    "graph_source": "A binary observed edges regardless of threshold label",
    "B_low_rating": "never a positive or a synthetic negative",
}


def tensor_digest(values):
    h = hashlib.sha256()
    for key, value in sorted(values.items()):
        value = value.detach().cpu().contiguous()
        h.update(key.encode()); h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode()); h.update(value.numpy().tobytes())
    return h.hexdigest()


def arrays(ds):
    return {"user_ids": ds.inter_feat[ds.uid_field].long().cpu().clone(),
            "item_ids": ds.inter_feat[ds.iid_field].long().cpu().clone(),
            "labels": ds.inter_feat[ds.label_field].float().cpu().clone()}


def counts(values):
    y = values["labels"]
    return {"rows": len(y), "label0_rows": int((y == 0).sum()),
            "label1_rows": int((y == 1).sum()),
            "users": int(values["user_ids"].unique().numel()),
            "items": int(values["item_ids"].unique().numel())}


def partition_indices(train, dataset, seed=FIXED_DATA_SEED):
    """Row indexes of deterministic grouped 90/10 A/B, before any fitting."""
    users = collections.defaultdict(dict)
    for idx, (u, i, y) in enumerate(zip(train["user_ids"].tolist(),
                                      train["item_ids"].tolist(), train["labels"].tolist())):
        if u <= 0 or i <= 0 or y not in (0, 1):
            raise ValueError("Expected non-padding IDs and threshold binary labels")
        group = users[u].setdefault(i, {"rows": [], "relevant": False})
        group["rows"].append(idx); group["relevant"] |= bool(y)
    in_B = torch.zeros(len(train["labels"]), dtype=torch.bool)
    swapped = sparse = 0
    for u in sorted(users):
        groups = users[u]
        items = np.asarray(list(groups), dtype=np.int64)
        entropy = hashlib.sha256(json.dumps(["A-B", dataset, int(seed), int(u)]).encode()).digest()
        rng = np.random.default_rng(np.frombuffer(entropy, dtype=np.uint32))
        rng.shuffle(items)
        n_B = len(items) // 10
        if n_B == 0:
            sparse += 1
            continue
        B = items[:n_B].tolist(); A = items[n_B:].tolist()
        if any(g["relevant"] for g in groups.values()) and not any(groups[i]["relevant"] for i in A):
            b = next(j for j, i in enumerate(B) if groups[i]["relevant"])
            a = next(j for j, i in enumerate(A) if not groups[i]["relevant"])
            A[a], B[b] = B[b], A[a]
            swapped += 1
        for i in B:
            in_B[groups[i]["rows"]] = True
    ai = torch.where(~in_B)[0]; bi = torch.where(in_B)[0]
    return ai, bi, {"protected_relevant_A_swaps": swapped, "sparse_users_all_in_A": sparse,
                   "unique_user_item_groups": sum(len(x) for x in users.values()),
                   "duplicate_rows": len(in_B) - sum(len(x) for x in users.values())}


def binary_degrees(train, user_num, item_num):
    encoded = (train["user_ids"] * item_num + train["item_ids"]).unique()
    user = torch.bincount(encoded // item_num, minlength=user_num).long()
    item = torch.bincount(encoded % item_num, minlength=item_num).long()
    user[0] = 0; item[0] = 0
    return user, item


def replace_yelp_counts(ds, A):
    """A observation-row review counts, original min-max numerical encoding."""
    provenance = []
    for field, source, ids, num in (
        ("user_review_count", "user", A["user_ids"], ds.user_num),
        ("item_review_count", "item", A["item_ids"], ds.item_num),
    ):
        feat = getattr(ds, source + "_feat")
        old = feat[field]
        raw = torch.bincount(ids, minlength=num).float(); raw[0] = 0
        mn, mx = float(raw[1:].min()), float(raw[1:].max())
        normalized = (raw - mn) / (mx - mn) if mx > mn else torch.ones_like(raw)
        normalized[0] = 0
        if old.ndim == 2:
            encoded = torch.stack((normalized, torch.ones_like(raw)), dim=-1)
            encoded[0] = 0
        else:
            encoded = normalized
        feat[field] = encoded.to(old.dtype)
        provenance.append({"field": field, "source": "A observed row counts, including threshold-negative rows",
                           "duplicate_policy": "count observation rows, unlike binary graph degree",
                           "normalization": "min-max over non-padding mapped entities, computed from A only",
                           "minimum": mn, "maximum": mx, "padding": 0,
                           "raw_count_digest": tensor_digest({field: raw}),
                           "encoded_count_digest": tensor_digest({field: encoded}),
                           "historical_count_values_discarded": True})
    return provenance


def _pair_keys(split, item_num):
    return (split["user_ids"] * item_num + split["item_ids"]).unique()


def prepare_master(dataset):
    out = RUN / "master" / dataset
    destination = out / "bundle.pt"
    if destination.is_file():
        manifest = json.loads((out / "manifest.json").read_text())
        if manifest["policy"] != AB_POLICY:
            raise RuntimeError("Existing frozen A/B policy differs")
        if sha(destination) != manifest["bundle_sha256"] or sha(out / "dataset.pt") != manifest["dataset_sha256"]:
            raise RuntimeError("Frozen master file changed")
        return load(destination)
    from common.runtime import restore_config
    config_path = CONFIG / 'master' / (dataset + '_config.json')
    expected_path = CONFIG / 'master' / (dataset + '_expected.json')
    expected = json.loads(expected_path.read_text())
    config = restore_config(json.loads(config_path.read_text()), DATA_ROOT,
                            checkpoint_dir=RUN/'backbones', device='cpu')
    if config['threshold'] != {'rating': 4}:
        raise RuntimeError('Unexpected relevance threshold')
    init_seed(FIXED_DATA_SEED, config['reproducibility'])
    complete = create_dataset(config)
    outer, valid, test = complete.build()
    actual = {'train': arrays(outer), 'valid': arrays(valid), 'test': arrays(test)}
    for phase, values in actual.items():
        if tensor_digest(values) != expected['outer_split_digests'][phase]:
            raise RuntimeError('Recorded outer split mismatch: '+dataset+'/'+phase)
    mapping = {'user': outer.field2id_token[outer.uid_field], 'item': outer.field2id_token[outer.iid_field]}
    mapping_digests = {k: hashlib.sha256(json.dumps(v.tolist(),sort_keys=True).encode()).hexdigest() for k,v in mapping.items()}
    if mapping_digests != expected['mapping_digests']:
        raise RuntimeError('Recorded mapped ID order differs')
    if (int(outer.user_num),int(outer.item_num)) != (expected['user_num'],expected['item_num']):
        raise RuntimeError('Recorded user/item counts differ')
    old = {'user_num':int(outer.user_num),'item_num':int(outer.item_num),
        'uid_field':outer.uid_field,'iid_field':outer.iid_field,'dataset':dataset,
        'field2id_token':mapping,'mapping_digests':mapping_digests,
        'split_digests':expected['outer_split_digests'],
        'original_checkpoint_sha256':expected['original_checkpoint_sha256']}
    ai, bi, partition_summary = partition_indices(actual["train"], dataset)
    A = {k: v[ai].clone() for k, v in actual["train"].items()}
    B = {k: v[bi].clone() for k, v in actual["train"].items()}
    if torch.isin(_pair_keys(A, old["item_num"]), _pair_keys(B, old["item_num"])).any():
        raise AssertionError("A/B duplicate-pair leakage")
    a_dataset = outer.copy(outer.inter_feat[ai])
    a_dataset.user_feat = copy.deepcopy(outer.user_feat)
    a_dataset.item_feat = copy.deepcopy(outer.item_feat)
    a_dataset.config = config
    side_counts = replace_yelp_counts(a_dataset, A) if dataset == "yelp2018" else []
    user_degree, item_degree = binary_degrees(A, old["user_num"], old["item_num"])
    splits = {"train": A, "refinement": B, "outer_train": actual["train"],
              "valid": actual["valid"], "test": actual["test"]}
    split_digests = {k: tensor_digest(v) for k, v in splits.items()}
    if split_digests != expected['split_digests']:
        raise RuntimeError('Recorded A/B refinement partition differs; refusing to fit')
    fixed_metadata = []
    data_path = Path(config["data_path"])
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    for kind in ("user", "item"):
        path = data_path / f"{config['dataset']}.{kind}"
        fixed_metadata.append({"kind": kind, "path": str(path), "exists": path.is_file(),
                               "sha256": sha(path) if path.is_file() else None,
                               "availability": "recorded dataset snapshot, per-interaction availability timestamps are not established",
                               "fields": config["load_col"].get(kind, []),
                               "A_replaced_fields": [x["field"] for x in side_counts if x["field"].startswith(kind + "_")]})
    signature = {"dataset": dataset, "master_outer_seed": FIXED_DATA_SEED,
                 "recorded_master_bundle_sha256": expected["source_bundle_sha256"],
                 "loader_configuration_sha256": sha(config_path),
                 "original_checkpoint_sha256": old["original_checkpoint_sha256"],
                 "mapping_digests": old["mapping_digests"], "split_digests": split_digests,
                 "policy": AB_POLICY, "side_count_provenance": side_counts,
                 "external_metadata": fixed_metadata,
                 "score_domain": "sigmoid(raw model logit) for every backbone, W/F, external reranker and graph score"}
    signature["sha256"] = identity(signature)
    bundle = {**old, **splits, "seed": FIXED_DATA_SEED, "config": config,
              "signature": signature, "split_digests": split_digests,
              "outer_split_digests": old["split_digests"],
              "item_degree": item_degree, "user_degree": user_degree,
              "A_row_indices": ai, "B_row_indices": bi,
              "graph_source_path": str(destination),
              "target_definition": "rating>=4, binary pair relevance any positive row, B only for refinement"}
    out.mkdir(parents=True, exist_ok=True)
    save(out / "dataset.pt", a_dataset); save(destination, bundle)
    manifest = {"status": "FROZEN_BEFORE_FITTING", "policy": AB_POLICY,
                "signature": signature, "counts": {k: counts(v) for k, v in splits.items()},
                "partition_summary": partition_summary, "item_num": old["item_num"], "user_num": old["user_num"],
                "A_B_pair_overlap": 0, "A_graph_edges": int(item_degree.sum()),
                "A_observed_row_fraction": len(A["labels"]) / len(actual["train"]["labels"]),
                "B_observed_row_fraction": len(B["labels"]) / len(actual["train"]["labels"]),
                "bundle_sha256": sha(destination), "dataset_sha256": sha(out / "dataset.pt"),
                "source_split_hashes_match_actual_original_loader": True,
                "source_mapping_match_actual_original_loader": True,
                "test_labels_enter_A_B_or_side_counts": False,
                "no_training_or_candidate_scoring": True}
    atomic_json(out / "manifest.json", manifest)
    return bundle

