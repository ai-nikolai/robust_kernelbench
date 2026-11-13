"""Real-space Ewald sum for periodic point charges.

Domain: chemistry / computational electrostatics.
Solves: evaluates the real-space contribution (plus the self correction) of
the Ewald decomposition:
``E_real = 1/2 sum_{i,j,n}' q_i q_j erfc(alpha r_ij,n) / r_ij,n``
with periodic images in a cubic box.  The reciprocal-space term is separate.
Source: Ewald, "Die Berechnung optischer und elektrostatischer Gitterpotentiale"
(1921); Frenkel & Smit, *Understanding Molecular Simulation* (2002).
Level: 3 (periodic images, all-pairs sums and a special-function kernel).
"""

import math

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(
        self,
        alpha: float = 0.3,
        box_length: float = 10.0,
        cutoff: float = 5.0,
        coulomb_constant: float = 1.0,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.box_length = float(box_length)
        self.cutoff = float(cutoff)
        self.coulomb_constant = float(coulomb_constant)

    def forward(self, positions: torch.Tensor, charges: torch.Tensor) -> torch.Tensor:
        shifts = [
            (i, j, k)
            for i in (-1, 0, 1)
            for j in (-1, 0, 1)
            for k in (-1, 0, 1)
        ]
        energy = positions.new_zeros(())
        for shift in shifts:
            offset = positions.new_tensor(shift) * self.box_length
            delta = positions[None, :, :] - positions[:, None, :] + offset
            squared_distance = (delta * delta).sum(dim=-1)
            distance = torch.sqrt(torch.clamp(squared_distance, min=1e-12))
            interaction = torch.erfc(self.alpha * distance) / distance
            within_cutoff = (squared_distance > 1e-12) & (distance < self.cutoff)
            interaction = interaction * within_cutoff
            energy = energy + 0.5 * (charges[:, None] * charges[None, :] * interaction).sum()

        self_energy = (
            -self.coulomb_constant * self.alpha / math.sqrt(math.pi) * charges.square().sum()
        )
        return self.coulomb_constant * energy + self_energy


def get_inputs():
    positions = torch.rand(48, 3) * 10.0
    charges = torch.randn(48)
    return [positions, charges]


def get_init_inputs():
    return [0.3, 10.0, 5.0, 1.0]
