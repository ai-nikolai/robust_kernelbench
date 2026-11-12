"""Direct-sum gravitational acceleration in an N-body system.

Domain: computational physics / astrophysics.
Solves: for every body i,
``a_i = G * sum_{j != i} m_j (r_j - r_i) / (||r_j - r_i||^2 + eps^2)^(3/2)``.
Source: Newton's law of universal gravitation (1687); the direct O(N^2) method
is described in Aarseth, *Gravitational N-Body Simulations* (2003).
Level: 2 (all-pairs interaction with 3D vectors).
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, gravitational_constant: float = 1.0, softening: float = 0.05):
        super().__init__()
        self.gravitational_constant = float(gravitational_constant)
        self.softening = float(softening)

    def forward(self, positions: torch.Tensor, masses: torch.Tensor) -> torch.Tensor:
        # delta[i, j] = r_j - r_i
        delta = positions[None, :, :] - positions[:, None, :]
        squared_distance = (delta * delta).sum(dim=-1) + self.softening * self.softening
        inverse_cube = squared_distance.pow(-1.5)
        weights = masses[None, :] * inverse_cube
        weights = weights * (1.0 - torch.eye(positions.shape[0], device=positions.device))
        return self.gravitational_constant * (delta * weights[..., None]).sum(dim=1)


def get_inputs():
    return [torch.randn(128, 3), torch.rand(128) + 0.1]


def get_init_inputs():
    return [1.0, 0.05]
