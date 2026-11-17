"""Evaluate candidate slates with the project's unchanged RecBole ranking evaluator."""

from __future__ import annotations

import time

import torch

from recbole.evaluator import Collector, Evaluator

from baselines.dnr.data_adapter import collate_slates, move_batch


METRICS = ("ndcg@10", "ndcg@20", "recall@10", "recall@20")


@torch.inference_mode()
def evaluate_cached(
    config,
    model,
    eval_batches: list[dict],
    histories: dict[int, list[int]],
    *,
    item_num: int,
    score_transform: str,
    device: torch.device,
    group_batch_size: int,
    passthrough: bool = False,
    attention_element_budget: int = 8_000_000,
) -> tuple[dict, float]:
    collector = Collector(config)
    evaluator = Evaluator(config)
    if model is not None:
        model.eval()
    started = time.perf_counter()
    # RecBole's sampled loader may emit one user group at a time.  Re-batch
    # those independent groups by sequence cost; this changes neither each
    # user's candidates nor relevance coordinates, but avoids tens of
    # thousands of single-user forward calls per validation epoch.
    units = []
    for cached in eval_batches:
        positive_u = cached["positive_u"]
        positive_i = cached["positive_i"]
        for row, record in enumerate(cached["groups"]):
            units.append((record, positive_i[positive_u == row].clone()))
    units.sort(
        key=lambda unit: len(unit[0]["items"]) ** 2
        + max(1, len(histories.get(int(unit[0]["user"]), []))) ** 2
    )
    chunks, current, maximum_cost = [], [], 0
    for unit in units:
        record = unit[0]
        cost = len(record["items"]) ** 2 + max(
            1, len(histories.get(int(record["user"]), []))
        ) ** 2
        proposed = max(maximum_cost, cost)
        if current and (
            len(current) >= group_batch_size
            or (len(current) + 1) * proposed > attention_element_budget
        ):
            chunks.append(current)
            current, maximum_cost = [], 0
        current.append(unit)
        maximum_cost = max(maximum_cost, cost)
    if current:
        chunks.append(current)

    for chunk in chunks:
        groups = [unit[0] for unit in chunk]
        if passthrough:
            predictions = [record["base_score_raw"].float() for record in groups]
        else:
            batch = collate_slates(
                groups, histories, item_num=item_num,
                score_transform=score_transform, include_labels=False,
            )
            batch = move_batch(batch, device)
            values = model(batch).detach().cpu()
            predictions = [
                values[row, :len(record["items"])]
                for row, record in enumerate(groups)
            ]
        rows = len(groups)
        matrix = torch.full((rows, item_num), -torch.inf, device=device)
        for row, (record, values) in enumerate(zip(groups, predictions)):
            matrix[row, record["items"].long().to(device)] = values.to(device)
        positive_u = torch.cat([
            torch.full((len(unit[1]),), row, dtype=torch.long)
            for row, unit in enumerate(chunk)
        ])
        positive_i = torch.cat([unit[1].long() for unit in chunk])
        collector.eval_batch_collect(
            matrix, None,
            positive_u.to(device), positive_i.to(device),
        )
    metrics = dict(evaluator.evaluate(collector.get_data_struct()))
    return {name: float(metrics[name]) for name in METRICS}, time.perf_counter() - started
