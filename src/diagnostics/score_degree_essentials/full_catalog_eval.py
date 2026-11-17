"""Original streamed full-catalog ranking utilities."""
import torch

def positives_by_user(batches):
    result, order = {}, []
    for batch in batches:
        for row_index, group in enumerate(batch["groups"]):
            user = int(group["user"])
            if user not in result:
                result[user] = set()
                order.append(user)
            result[user].update(int(value) for value in batch["positive_i"][batch["positive_u"] == row_index].tolist())
    return result, order


def stable_rank(scores, items):
    item_order = torch.argsort(items, dim=1, stable=True)
    ordered_scores = torch.gather(scores, 1, item_order)
    score_order = torch.argsort(ordered_scores, dim=1, descending=True, stable=True)
    return torch.gather(item_order, 1, score_order)


def merge_top50(old_scores, old_items, new_scores, new_items):
    scores = torch.cat((old_scores, new_scores), dim=1)
    items = torch.cat((old_items, new_items), dim=1)
    order = stable_rank(scores, items)[:, :50]
    return torch.gather(scores, 1, order), torch.gather(items, 1, order)


def metric_contribution(scores, items, relevant_sets, full_counts, candidate_counts):
    order = stable_rank(scores, items)
    ranked_items = torch.gather(items, 1, order)
    labels = torch.zeros_like(ranked_items, dtype=torch.float64)
    for row, relevant in enumerate(relevant_sets):
        if relevant:
            labels[row] = torch.isin(ranked_items[row], torch.tensor(sorted(relevant), dtype=torch.long)).to(torch.float64)
    output = {}
    full = torch.as_tensor(full_counts, dtype=torch.float64)
    conditional = torch.as_tensor(candidate_counts, dtype=torch.float64)
    for k in (10, 20):
        gain = labels[:, :k]
        hits = gain.sum(1)
        discount = 1.0 / torch.log2(torch.arange(2, k + 2, dtype=torch.float64))
        dcg = (gain * discount).sum(1)
        prefix = torch.cat((torch.zeros(1, dtype=torch.float64), torch.cumsum(discount, 0)))
        full_idcg = prefix[torch.minimum(full.long(), torch.tensor(k))]
        conditional_idcg = prefix[torch.minimum(conditional.long(), torch.tensor(k))]
        output[f"recall@{k}"] = float(torch.where(full > 0, hits / full, torch.zeros_like(hits)).sum())
        output[f"ndcg@{k}"] = float(torch.where(full_idcg > 0, dcg / full_idcg, torch.zeros_like(dcg)).sum())
        output[f"conditional_recall@{k}"] = float(torch.where(conditional > 0, hits / conditional, torch.zeros_like(hits)).sum())
        output[f"conditional_ndcg@{k}"] = float(torch.where(conditional_idcg > 0, dcg / conditional_idcg, torch.zeros_like(dcg)).sum())
    return output

