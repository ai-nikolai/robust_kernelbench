"""One leapfrog step of the 2D wave equation.

Domain: computational physics / partial differential equations.
Solves: advances ``u_tt = c^2 laplacian(u)`` by one explicit second-order
finite-difference step:
``u_next = 2u - u_prev + (c dt)^2 laplacian(u)``.
Source: standard explicit wave-equation scheme; see LeVeque, *Finite Difference
Methods for Ordinary and Partial Differential Equations* (2007).
Level: 2 (two state fields plus the five-point stencil).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, wave_speed: float = 1.0, dt: float = 0.01):
        super().__init__()
        self.wave_speed = float(wave_speed)
        self.dt = float(dt)

    def forward(self, previous: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        if previous.dim() == 3:
            previous = previous.unsqueeze(1)
        if current.dim() == 3:
            current = current.unsqueeze(1)
        padded = F.pad(current, (1, 1, 1, 1), mode="replicate")
        laplacian = (
            padded[:, :, :-2, 1:-1]
            + padded[:, :, 2:, 1:-1]
            + padded[:, :, 1:-1, :-2]
            + padded[:, :, 1:-1, 2:]
            - 4.0 * current
        )
        coefficient = (self.wave_speed * self.dt) ** 2
        return 2.0 * current - previous + coefficient * laplacian


def get_inputs():
    return [torch.randn(4, 1, 64, 64), torch.randn(4, 1, 64, 64)]


def get_init_inputs():
    return [1.0, 0.01]
