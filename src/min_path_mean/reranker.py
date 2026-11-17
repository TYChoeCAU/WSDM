"""Frozen-backbone graph scorers and independently selectable controllers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch

from recbole.data.interaction import Interaction

from .graph import NeighborTable


def path_average_weight(
    base: torch.Tensor,
    graph: torch.Tensor,
    cutoff: float,
    scale: float,
    *,
    temperature: float = 2.5,
    atol: float = 1e-8,
) -> torch.Tensor:
    """Average the normalized sigmoid sensitivity along base-to-graph paths."""

    if base.shape != graph.shape:
        raise ValueError("path endpoints must have the same shape")
    if scale <= 0 or temperature <= 0 or atol <= 0:
        raise ValueError("scale, temperature and atol must be positive")
    work_base = base.to(torch.float64)
    work_graph = graph.to(torch.float64)
    x = (work_base - cutoff) / (temperature * scale)
    y = (work_graph - cutoff) / (temperature * scale)
    low, high = torch.minimum(x, y), torch.maximum(x, y)
    width = high - low

    # The equal-endpoint limit is 4 sigma(x)(1-sigma(x)).
    tail = torch.exp(-torch.abs(x))
    result = 4.0 * tail / torch.square(1.0 + tail)
    apart = width > atol
    if apart.any():
        lo, hi = low[apart], high[apart]
        # Positive endpoints need the opposite sigmoid tail to avoid 1 - 1.
        delta = torch.where(
            lo >= 0,
            torch.sigmoid(-lo) - torch.sigmoid(-hi),
            torch.sigmoid(hi) - torch.sigmoid(lo),
        )
        result[apart] = 4.0 * delta / width[apart]
    return result.clamp(0.0, 1.0)


def _lower_median(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        raise ValueError("median of an empty tensor")
    return values.kthvalue((values.numel() + 1) // 2).values


def _distinct_candidates(
    positions: torch.Tensor, item_ids: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, List[int]]]:
    """Canonicalize duplicates while making the later scatter unambiguous."""

    occurrence: Dict[int, List[int]] = {}
    for position in positions.detach().cpu().tolist():
        occurrence.setdefault(int(item_ids[position]), []).append(position)
    ordered_items = sorted(occurrence)
    representative = [occurrence[item][-1] for item in ordered_items]
    return (
        torch.tensor(representative, dtype=torch.long, device=positions.device),
        torch.tensor(ordered_items, dtype=torch.long, device=positions.device),
        occurrence,
    )


def apply_gamma_gate(
    base: torch.Tensor,
    graph: torch.Tensor,
    gamma: torch.Tensor,
    gate: torch.Tensor,
    *,
    alpha: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply gamma once, then gate the gamma-controlled residual.

    The returned pair is ``(g, final)`` where
    ``g = base + gamma * (graph - base)`` and
    ``final = base + alpha * gate * (g - base)``.
    """

    if not (base.shape == graph.shape == gamma.shape == gate.shape):
        raise ValueError("base, graph, gamma and gate must have the same shape")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    gamma_graph = base + gamma.to(base.dtype) * (graph - base)
    final = base + alpha * gate.to(base.dtype) * (gamma_graph - base)
    return gamma_graph, final


@dataclass
class BatchDiagnostics:
    users: int = 0
    supported_candidates: int = 0
    neighbour_queries: int = 0
    fallback_candidates: int = 0
    lambda_sum: float = 0.0
    diagnostic_candidates: int = 0
    gamma_sum: float = 0.0
    gate_sum: float = 0.0
    effective_sum: float = 0.0
    gamma_residual_abs_sum: float = 0.0
    final_residual_abs_sum: float = 0.0

    @property
    def mean_lambda(self) -> float:
        return self.lambda_sum / self.users if self.users else float("nan")

    def _candidate_mean(self, value: float) -> float:
        return value / self.diagnostic_candidates if self.diagnostic_candidates else float("nan")

    @property
    def mean_gamma(self) -> float:
        return self._candidate_mean(self.gamma_sum)

    @property
    def mean_gate(self) -> float:
        return self._candidate_mean(self.gate_sum)

    @property
    def mean_effective(self) -> float:
        return self._candidate_mean(self.effective_sum)

    @property
    def mean_abs_gamma_residual(self) -> float:
        return self._candidate_mean(self.gamma_residual_abs_sum)

    @property
    def mean_abs_final_residual(self) -> float:
        return self._candidate_mean(self.final_residual_abs_sum)


class GraphReranker:
    """Apply a selected graph scorer/controller to every eligible candidate."""

    def __init__(
        self,
        model,
        dataset,
        user_neighbors: NeighborTable,
        item_neighbors: NeighborTable,
        *,
        cutoff_k: int = 20,
        candidate_top_m: int | None = None,
        scorer: str = "minimal",
        controller: str = "path_mean",
        evidence_attenuation: bool = False,
        evidence_scale: float = 1.0,
        native_eq12: bool = False,
        alpha: float = 1.0,
        temperature: float = 2.5,
        temperature_above: float = 2.5,
        temperature_below: float = 1.0,
        flat_width: float = 1.0,
        constant: float = 0.75,
        mad_epsilon: float = 1e-8,
        path_atol: float = 1e-8,
        predict_batch_size: int = 8192,
    ) -> None:
        controllers = {
            "uniform", "symmetric", "flat_top", "asymmetric",
            "path_candidate", "path_mean", "constant", "membership",
        }
        if scorer not in {"minimal", "idf", "information", "custom", "nonparamgc"}:
            raise ValueError(f"unknown scorer: {scorer}")
        if controller not in controllers:
            raise ValueError(f"unknown controller: {controller}")
        if cutoff_k < 1 or alpha < 0:
            raise ValueError("cutoff_k must be positive and alpha non-negative")
        if candidate_top_m is not None and candidate_top_m < 1:
            raise ValueError("candidate_top_m must be positive")
        if min(
            temperature, temperature_above, temperature_below,
            mad_epsilon, path_atol, evidence_scale,
        ) <= 0 or predict_batch_size < 1:
            raise ValueError("invalid numerical or batch-size setting")
        if flat_width < 0 or not 0 <= constant <= 1:
            raise ValueError("flat width or constant coefficient is invalid")
        if native_eq12 != (scorer == "nonparamgc"):
            raise ValueError("Eq. 12 must be enabled exactly for native NonParamGC")
        self.model = model
        self.dataset = dataset
        self.user_neighbors = user_neighbors
        self.item_neighbors = item_neighbors
        self.cutoff_k = cutoff_k
        self.candidate_top_m = candidate_top_m
        self.scorer = scorer
        self.controller = controller
        self.evidence_attenuation = evidence_attenuation
        self.evidence_scale = evidence_scale
        self.native_eq12 = native_eq12
        self.alpha = alpha
        self.temperature = temperature
        self.temperature_above = temperature_above
        self.temperature_below = temperature_below
        self.flat_width = flat_width
        self.constant = constant
        self.mad_epsilon = mad_epsilon
        self.path_atol = path_atol
        self.predict_batch_size = predict_batch_size
        self.uid_field = dataset.uid_field
        self.iid_field = dataset.iid_field
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device("cpu")

    def _predict_pairs(self, users: np.ndarray, items: np.ndarray) -> torch.Tensor:
        output: List[torch.Tensor] = []
        for start in range(0, len(users), self.predict_batch_size):
            end = min(start + self.predict_batch_size, len(users))
            pair = Interaction(
                {
                    self.uid_field: torch.as_tensor(users[start:end], dtype=torch.long),
                    self.iid_field: torch.as_tensor(items[start:end], dtype=torch.long),
                }
            )
            pair = self.dataset.join(pair).to(self.device)
            predicted = self.model.predict(pair).reshape(-1)
            if predicted.numel() != end - start:
                raise RuntimeError("backbone returned an unexpected number of scores")
            output.append(predicted)
        if not output:
            return torch.empty(0, dtype=torch.float32, device=self.device)
        return torch.cat(output)

    def _graph_scores(
        self,
        user_id: int,
        items: torch.Tensor,
        base: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        if self.native_eq12 and self.user_neighbors.ids.shape[1] == 1:
            return base.clone(), torch.zeros_like(base), 0, len(items)
        item_array = items.detach().cpu().numpy().astype(np.int64, copy=False)
        user_ids = self.user_neighbors.ids[user_id]
        user_weights = self.user_neighbors.weights[user_id]
        item_ids = self.item_neighbors.ids[item_array]
        item_weights = self.item_neighbors.weights[item_array]

        item_k = item_ids.shape[1]
        pair_users = np.broadcast_to(user_ids[None, :, None], (len(items), len(user_ids), item_k))
        pair_items = np.broadcast_to(item_ids[:, None, :], pair_users.shape)
        pair_weights = user_weights[None, :, None] * item_weights[:, None, :]
        valid = (pair_users >= 0) & (pair_items >= 0) & (pair_weights > 0)

        evidence_mass = np.where(valid, pair_weights, 0.0).sum(axis=(1, 2))
        aggregation_weights = pair_weights.copy()
        if self.native_eq12:
            flattened = aggregation_weights.reshape(len(items), -1)
            flattened[:, 0] = (1.0 - flattened[:, 1:]).sum(axis=1)
            aggregation_weights = flattened.reshape(aggregation_weights.shape)
        active = valid & (aggregation_weights > 0)

        flat_users = pair_users[active].astype(np.int64, copy=False)
        flat_items = pair_items[active].astype(np.int64, copy=False)
        pair_predictions = self._predict_pairs(flat_users, flat_items)
        graph_scores = base.clone()
        cursor = 0
        fallback = 0
        for candidate in range(len(items)):
            candidate_weights = aggregation_weights[candidate][active[candidate]]
            count = len(candidate_weights)
            if count == 0:
                fallback += 1
                continue
            weights = torch.as_tensor(
                candidate_weights, dtype=torch.float64, device=self.device
            )
            scores = pair_predictions[cursor : cursor + count].to(torch.float64)
            graph_scores[candidate] = (weights * scores).sum() / weights.sum()
            cursor += count
        mass = torch.as_tensor(evidence_mass, dtype=torch.float64, device=self.device)
        return graph_scores, mass, int(active.sum()), fallback

    def _controller_weights(
        self,
        base: torch.Tensor,
        graph_scores: torch.Tensor,
        cutoff: float,
        scale: float,
    ) -> Tuple[torch.Tensor, float]:
        z = (base.to(torch.float64) - cutoff) / scale
        if self.controller == "uniform":
            weights = torch.ones_like(z)
        elif self.controller == "constant":
            weights = torch.full_like(z, self.constant)
        elif self.controller == "symmetric":
            point = z / self.temperature
            tail = torch.exp(-torch.abs(point))
            weights = 4.0 * tail / torch.square(1.0 + tail)
        elif self.controller == "membership":
            weights = torch.sigmoid(z / self.temperature)
        elif self.controller == "flat_top":
            point = torch.clamp(torch.abs(z) - self.flat_width, min=0) / self.temperature
            tail = torch.exp(-point)
            weights = 4.0 * tail / torch.square(1.0 + tail)
        elif self.controller == "asymmetric":
            divisor = torch.where(
                z >= 0,
                torch.full_like(z, self.temperature_above),
                torch.full_like(z, self.temperature_below),
            )
            point = z / divisor
            tail = torch.exp(-torch.abs(point))
            weights = 4.0 * tail / torch.square(1.0 + tail)
        else:
            weights = path_average_weight(
                base,
                graph_scores,
                cutoff,
                scale,
                temperature=self.temperature,
                atol=self.path_atol,
            )
            if self.controller == "path_mean":
                weights = torch.full_like(weights, weights.mean())
        return weights, float(weights.mean())

    @torch.no_grad()
    def rerank_sampled_batch(
        self,
        interaction: Interaction,
        row_index: torch.Tensor,
        base_scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, BatchDiagnostics]:
        """Re-rank one RecBole sampled-evaluation batch.

        Candidate sets are defined by ``row_index`` exactly as in RecBole's
        scatter evaluator.  If a sampler emits a duplicate item, every copy is
        assigned the same fused score so scatter order cannot change the rank.
        """

        if row_index is None:
            raise ValueError("graph re-ranking currently requires sampled evaluation")
        row_index = row_index.to(base_scores.device)
        users = interaction[self.uid_field].to(base_scores.device)
        items = interaction[self.iid_field].to(base_scores.device)
        final = base_scores.clone()
        diagnostics = BatchDiagnostics()

        for row in torch.unique(row_index, sorted=True):
            positions = torch.where(row_index == row)[0]
            representatives, distinct_items, occurrence = _distinct_candidates(
                positions, items
            )
            distinct_base = base_scores[representatives]
            user_values = torch.unique(users[positions])
            if user_values.numel() != 1:
                raise ValueError("one sampled ranking row must correspond to one user")
            user_id = int(user_values.item())

            # ID ascending first, then a stable descending score sort gives the
            # documented deterministic tie rule without consulting labels.
            id_order = torch.argsort(distinct_items, stable=True)
            score_order = torch.argsort(
                distinct_base[id_order], descending=True, stable=True
            )
            ranked = id_order[score_order]
            if ranked.numel() == 0:
                continue
            support = (
                ranked[: self.candidate_top_m]
                if self.candidate_top_m is not None else ranked
            )
            support_base = distinct_base[support]
            cutoff = float(
                support_base[min(self.cutoff_k, support.numel()) - 1].to(torch.float64)
            )
            median = _lower_median(support_base.to(torch.float64))
            mad = _lower_median(torch.abs(support_base.to(torch.float64) - median))
            scale = max(float(mad), self.mad_epsilon)
            graph_scores, evidence_mass, query_count, fallback = self._graph_scores(
                user_id, distinct_items[support], support_base
            )
            gate, _ = self._controller_weights(
                support_base,
                graph_scores,
                cutoff,
                scale,
            )
            if self.evidence_attenuation:
                gamma = torch.clamp(
                    evidence_mass / (self.evidence_scale ** 2), max=1.0
                )
            else:
                gamma = torch.ones_like(evidence_mass)
            effective = gamma * gate
            gamma_graph, fused = apply_gamma_gate(
                support_base,
                graph_scores,
                gamma,
                gate,
                alpha=self.alpha,
            )

            for local, candidate_index in enumerate(support.detach().cpu().tolist()):
                item_id = int(distinct_items[candidate_index])
                final[occurrence[item_id]] = fused[local]
            diagnostics.users += 1
            diagnostics.supported_candidates += int(support.numel())
            diagnostics.neighbour_queries += query_count
            diagnostics.fallback_candidates += fallback
            diagnostics.lambda_sum += float(effective.mean())
            diagnostics.diagnostic_candidates += int(support.numel())
            diagnostics.gamma_sum += float(gamma.sum())
            diagnostics.gate_sum += float(gate.sum())
            diagnostics.effective_sum += float(effective.sum())
            diagnostics.gamma_residual_abs_sum += float(
                torch.abs(gamma_graph - support_base).sum()
            )
            diagnostics.final_residual_abs_sum += float(
                torch.abs(fused - support_base).sum()
            )
        return final, diagnostics


class GenericPathReranker(GraphReranker):
    """Compatibility name for full-candidate minimal path-mean re-ranking."""

    def __init__(self, *args, **kwargs):
        kwargs.update(scorer="minimal", controller="path_mean", native_eq12=False)
        super().__init__(*args, **kwargs)
