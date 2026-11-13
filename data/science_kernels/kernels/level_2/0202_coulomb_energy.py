"""Total softened Coulomb energy of a set of point charges.

Domain: computational chemistry / electrostatics.
Solves: evaluates
``E = k * sum_{i<j} q_i q_j / sqrt(||r_i - r_j||^2 + eps^2)``.
Source: Coulomb's law (1785); the direct pairwise sum and softening are
standard in molecular simulation (Frenkel & Smit, *Understanding Molecular
Simulation*, 2002).
Level: 2 (all-pairs interaction with per-particle charges).
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, coulomb_constant: float = 1.0, softening: float = 0.05):
        super().__init__()
        self.coulomb_constant = float(coulomb_constant)
        self.softening = float(softening)

    def forward(self, positions: torch.Tensor, charges: torch.Tensor) -> torch.Tensor:
        difference = positions[:, None, :] - positions[None, :, :]
        squared_distance = (difference * difference).sum(dim=-1)
        squared_distance = squared_distance + self.softening * self.softening
        inverse_distance = squared_distance.rsqrt()
        off_diagonal = ~torch.eye(positions.shape[0], dtype=torch.bool, device=positions.device)
        pair_energy = charges[:, None] * charges[None, :] * inverse_distance
        # 0.5 removes the double counting of the full i,j matrix.
        return 0.5 * self.coulomb_constant * pair_energy[off_diagonal].sum()


def get_inputs():
    return [torch.randn(96, 3), torch.randn(96)]


def get_init_inputs():
    return [1.0, 0.05]
