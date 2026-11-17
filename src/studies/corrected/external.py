"""Original corrected PRM evaluation; no DNR model or trainer."""
import time
import torch
from baselines.dnr.data_adapter import collate_slates, move_batch
from diagnostics.revision_evidence.common import ranking_metrics
def check_deadline(applied):
    return None

def _units_and_alignment(eval_batches, split):
    units = []
    for cached in eval_batches:
        for row, record in enumerate(cached["groups"]):
            units.append((record, cached["positive_i"][cached["positive_u"] == row]))
    if len(units) != len(split["s"]): raise ValueError("external evaluator user-group count differs from feature cache")
    for index, (record, positives) in enumerate(units):
        mask = split["valid_mask"][index]
        if int(record["user"]) != int(split["user_ids"][index]): raise ValueError("external user order differs from feature cache")
        ids = record["items"].long()
        if not torch.equal(ids, split["candidate_ids"][index][mask].long()): raise ValueError("external candidate IDs differ from feature cache")
        if not torch.equal(record["base_score_raw"].float(), split["s"][index][mask].float()): raise ValueError("external frozen scores differ from feature cache")
        if not torch.equal(torch.isin(ids, positives.long()), split["labels"][index][mask]): raise ValueError("external labels differ from feature cache")
        if int(split["positive_count"][index]) != len(set(positives.tolist())): raise ValueError("external all-relevant denominator differs")
    return units


@torch.inference_mode()
def _external_scores(model, eval_batches, histories, split, applied, device, item_num, checked_units=None):
    units = _units_and_alignment(eval_batches, split) if checked_units is None else checked_units
    ordered = sorted(((i, unit) for i, unit in enumerate(units) if len(unit[0]["items"])), key=lambda pair: len(pair[1][0]["items"])**2+max(1, len(histories.get(int(pair[1][0]["user"]), [])))**2)
    chunks, current, maximum = [], [], 0
    for entry in ordered:
        record = entry[1][0]; cost = len(record["items"])**2+max(1, len(histories.get(int(record["user"]), [])))**2
        proposed = max(maximum, cost)
        if current and (len(current) >= int(applied["eval_group_batch_size"]) or (len(current)+1)*proposed > int(applied["attention_element_budget"])):
            chunks.append(current); current = []; maximum = 0
        current.append(entry); maximum = max(maximum, cost)
    if current: chunks.append(current)
    scores = torch.zeros_like(split["s"][:len(units)]); model.eval()
    for chunk in chunks:
        check_deadline(applied)
        batch = collate_slates([entry[1][0] for entry in chunk], histories, item_num=item_num, score_transform=applied.get("score_transform", "identity_probability"), include_labels=False)
        output = model(move_batch(batch, device)).detach().cpu()
        for row, (index, (record, _)) in enumerate(chunk): scores[index, split["valid_mask"][index]] = output[row, :len(record["items"])]
    return scores


def external_evaluate_cached(config, model, eval_batches, histories, *, item_num, score_transform, device, group_batch_size, passthrough=False, attention_element_budget=8_000_000):
    """Existing external forward/collate, attached to the common Top50 evaluator."""
    if passthrough: raise ValueError("corrected external training does not request passthrough")
    split = config["corrected_validation_split"]
    applied = {**config["corrected_applied"], "score_transform": score_transform, "eval_group_batch_size": group_batch_size, "attention_element_budget": attention_element_budget}
    started = time.perf_counter()
    original = config["corrected_valid_batches"]
    same_prefix = len(eval_batches) <= len(original) and all(left is right for left, right in zip(eval_batches, original))
    checked = config["corrected_valid_units"][:sum(len(batch["groups"]) for batch in eval_batches)] if same_prefix else None
    scores = _external_scores(model, eval_batches, histories, split, applied, device, item_num, checked)
    limit = len(scores); used = {key: value[:limit] if torch.is_tensor(value) else value for key, value in split.items()}
    return ranking_metrics(scores, used), time.perf_counter()-started


def _external_cache(cache, applied, feature_cache):
    result = dict(cache)
    result["train_samples"] = [record for record in cache["train_samples"] if len(record["items"])]
    if not result["train_samples"]: raise ValueError("external baseline has no nonempty corrected train slate")
    result["empty_candidate_train_groups_excluded"] = len(cache["train_samples"])-len(result["train_samples"])
    result["nonempty_zero_positive_train_groups_retained"] = sum(not bool(record["labels"].any()) for record in result["train_samples"])
    model_config = dict(cache.get("model_config", {}))
    model_config.update(user_num=int(cache.get("user_num", model_config.get("user_num", 0))), item_num=int(cache.get("item_num", model_config.get("item_num", 0))))
    if min(model_config["user_num"], model_config["item_num"]) <= 0: raise ValueError("missing external embedding catalog sizes")
    result["model_config"] = model_config
    by_split = cache.get("evaluation_histories_by_split", cache.get("eval_histories_by_split", {}))
    common = cache.get("evaluation_histories", cache.get("train_histories"))
    if common is None and "valid" not in by_split: raise ValueError("external validation histories are unspecified")
    result["evaluation_histories"] = by_split.get("valid", common)
    result["recbole_config"] = {"corrected_validation_split": feature_cache["valid"], "corrected_applied": applied,
        "corrected_valid_batches": result["valid_batches"], "corrected_valid_units": _units_and_alignment(result["valid_batches"], feature_cache["valid"])}
    return result

