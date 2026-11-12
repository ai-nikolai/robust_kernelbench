"""Explicit heat-equation step on a 2D field.

Domain: computational physics / partial differential equations.
Solves: advances the heat equation ``u_t = alpha * laplacian(u)`` by one
explicit finite-difference timestep with zero-flux (replicate) boundaries.
Source: Fourier's heat equation (1822); the explicit scheme is the textbook
method in LeVeque, *Finite Difference Methods for Ordinary and Partial
Differential Equations* (2007), and Press et al., *Numerical Recipes* (2007).
Level: 1 (single stencil application, elementwise arithmetic).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        if u.dim() == 3:
            u = u.unsqueeze(1)
        padded = F.pad(u, (1, 1, 1, 1), mode="replicate")
        laplacian = (
            padded[:, :, :-2, 1:-1]
            + padded[:, :, 2:, 1:-1]
            + padded[:, :, 1:-1, :-2]
            + padded[:, :, 1:-1, 2:]
            - 4.0 * u
        )
        return u + self.alpha * laplacian


def get_inputs():
    return [torch.randn(4, 1, 64, 64)]


def get_init_inputs():
    return [0.1]
