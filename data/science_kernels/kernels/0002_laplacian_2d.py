import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """Five-point Laplacian stencil on a 2D field.

    This is the core stencil in diffusion, Poisson solvers, and PDE
    preconditioners.  A custom kernel can use shared-memory tiling instead of
    the four separate shifted reads used by the reference implementation.
    """

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        if u.dim() == 3:
            u = u.unsqueeze(1)
        padded = F.pad(u, (1, 1, 1, 1), mode="replicate")
        return (
            padded[:, :, :-2, 1:-1]
            + padded[:, :, 2:, 1:-1]
            + padded[:, :, 1:-1, :-2]
            + padded[:, :, 1:-1, 2:]
            - 4.0 * u
        )


def get_inputs():
    return [torch.randn(4, 1, 64, 64)]


def get_init_inputs():
    return []
