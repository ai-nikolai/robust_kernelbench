"""Pairwise Gaussian (RBF) kernel matrix.

Domain: statistics / machine learning (kernel methods).
Solves: evaluates ``exp(-gamma * ||x_i - y_j||^2)`` for every pair.
Source: Schölkopf & Smola, *Learning with Kernels* (2002); Rasmussen &
Williams, *Gaussian Processes for Machine Learning* (2006).
Level: 1 (pairwise distance and elementwise exponential).
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """Pairwise Gaussian (RBF) kernel matrix.

    Domain: statistics / machine learning (kernel methods).
    Solves: evaluates ``exp(-gamma * ||x_i - y_j||^2)`` for every pair.
    Source: the Gaussian kernel is classical kernel-method machinery; see
    Schölkopf & Smola, *Learning with Kernels* (2002), and Rasmussen & Williams,
    *Gaussian Processes for Machine Learning* (2006).
    """

    def __init__(self, gamma: float = 0.5):
        super().__init__()
        self.gamma = float(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        squared_distance = torch.cdist(x, y, p=2).pow(2)
        return torch.exp(-self.gamma * squared_distance)


def get_inputs():
    return [torch.randn(32, 16), torch.randn(24, 16)]


def get_init_inputs():
    return [0.5]
