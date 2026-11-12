"""Total Lennard-Jones potential energy of a small particle cluster.

Domain: computational chemistry / molecular dynamics.
Solves: evaluates
``U = 4 * epsilon * sum_{i<j} [(sigma/r_ij)^12 - (sigma/r_ij)^6]``
for all distinct particle pairs.
Source: Lennard-Jones, "On the determination of molecular fields" (1924);
the standard MD formulation is in Allen & Tildesley, *Computer Simulation of
Liquids* (1987).
Level: 1 (pairwise reduction).
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, epsilon: float = 1.0, sigma: float = 1.0):
        super().__init__()
        self.epsilon = float(epsilon)
        self.sigma = float(sigma)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        n = positions.shape[0]
        difference = positions[:, None, :] - positions[None, :, :]
        squared_distance = (difference * difference).sum(dim=-1)
        squared_distance = torch.clamp(squared_distance, min=1e-8)
        off_diagonal = ~torch.eye(n, dtype=torch.bool, device=positions.device)
        squared_distance = squared_distance[off_diagonal]

        sigma2_over_r2 = (self.sigma * self.sigma) / squared_distance
        sr6 = sigma2_over_r2.pow(3)
        sr12 = sr6 * sr6
        # 0.5 corrects the double counting of i<j in the full pair matrix.
        return 2.0 * self.epsilon * (sr12 - sr6).sum()


def get_inputs():
    return [torch.randn(64, 3)]


def get_init_inputs():
    return [1.0, 1.0]
