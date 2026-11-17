"""The deliberately small, shared graph-fusion architecture."""

from __future__ import annotations

import torch
from torch import nn


class FusionMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(6, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def fuse_scores(
    method: str,
    model: nn.Module,
    standardized: torch.Tensor,
    raw_s: torch.Tensor,
    raw_h: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if method == "score_only_residual":
        controlled = torch.zeros_like(standardized)
        controlled[..., 0] = standardized[..., 0]
        output = model(controlled)
        return raw_s + scale * output, None
    output = model(standardized)
    if method == "learned_graph_gate":
        gate = torch.sigmoid(output)
        return raw_s + gate * (raw_h - raw_s), gate
    if method == "learned_graph_residual":
        return raw_s + scale * output, None
    raise ValueError(f"unknown fusion method: {method}")
