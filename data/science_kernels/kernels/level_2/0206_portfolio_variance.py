"""Portfolio variance under a covariance model.

Domain: economics / finance (portfolio theory).
Solves: evaluates ``sigma_p^2 = w^T Sigma w`` for a batch of portfolios and
their covariance matrices.
Source: Markowitz, "Portfolio selection" (1952).
Level: 2 (batched matrix-vector products and reductions).
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def forward(self, weights: torch.Tensor, covariance: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bi,bij,bj->b", weights, covariance, weights)


def get_inputs():
    batch, assets = 128, 32
    weights = torch.rand(batch, assets)
    weights = weights / weights.sum(dim=1, keepdim=True)
    factors = torch.randn(batch, assets, assets)
    covariance = factors @ factors.transpose(-1, -2) / assets
    return [weights, covariance]


def get_init_inputs():
    return []
