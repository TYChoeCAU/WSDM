"""Shared candidate normalization, without DNR model code."""
import torch

def masked_minmax(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    positive_inf = torch.full_like(values, torch.inf)
    negative_inf = torch.full_like(values, -torch.inf)
    minimum = torch.where(mask, values, positive_inf).amin(dim=1, keepdim=True)
    maximum = torch.where(mask, values, negative_inf).amax(dim=1, keepdim=True)
    denominator = (maximum - minimum).clamp_min(eps)
    scaled = (values - minimum) / denominator
    return torch.where(mask, scaled, torch.zeros_like(scaled))

