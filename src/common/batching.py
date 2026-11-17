"""Shared length-aware minibatches used by PRM."""
import random

def batches_for_epoch(
    samples: list[dict], batch_size: int, seed: int, attention_element_budget: int
) -> list[list[dict]]:
    # Length-local batches avoid unnecessary padding without dropping candidates.
    ordered = sorted(
        samples,
        key=lambda sample: len(sample["items"]) ** 2 + max(1, len(sample["history"])) ** 2,
    )
    batches = []
    current = []
    maximum_cost = 0
    for sample in ordered:
        cost = len(sample["items"]) ** 2 + max(1, len(sample["history"])) ** 2
        proposed_max = max(maximum_cost, cost)
        if current and (
            len(current) >= batch_size
            or (len(current) + 1) * proposed_max > attention_element_budget
        ):
            batches.append(current)
            current = []
            maximum_cost = 0
        current.append(sample)
        maximum_cost = max(maximum_cost, cost)
    if current:
        batches.append(current)
    random.Random(seed).shuffle(batches)
    return batches

