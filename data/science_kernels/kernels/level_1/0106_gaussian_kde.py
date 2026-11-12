"""Gaussian kernel density estimate at a set of query points.

Domain: statistics / non-parametric density estimation.
Solves: evaluates
``p(x) = 1/(N h^d (2*pi)^(d/2)) * sum_i exp(-||x - x_i||^2 / (2 h^2))``.
Source: Rosenblatt, "Remarks on some nonparametric estimates of a density
function" (1956); Parzen, "On estimation of a probability density function and
mode" (1962).
Level: 1 (pairwise distance and reduction).
"""

import math

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, bandwidth: float = 0.5):
        super().__init__()
        self.bandwidth = float(bandwidth)

    def forward(self, samples: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        dimension = samples.shape[1]
        difference = queries[:, None, :] - samples[None, :, :]
        squared_distance = (difference * difference).sum(dim=-1)
        exponent = -squared_distance / (2.0 * self.bandwidth * self.bandwidth)
        normalizer = samples.shape[0] * self.bandwidth**dimension * (2.0 * math.pi) ** (dimension / 2.0)
        return torch.exp(exponent).sum(dim=1) / normalizer


def get_inputs():
    return [torch.randn(256, 4), torch.randn(128, 4)]


def get_init_inputs():
    return [0.5]
