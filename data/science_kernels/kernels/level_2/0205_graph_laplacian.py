"""Combinatorial graph Laplacian of a weighted undirected graph.

Domain: mathematics / spectral graph theory.
Solves: computes ``L = D - A``, where ``A`` is a symmetric non-negative
adjacency matrix and ``D`` is its diagonal degree matrix.
Source: the Laplacian is central to Fiedler's spectral theory (1973); see
Chung, *Spectral Graph Theory* (1997).
Level: 2 (dense matrix construction and reduction).
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def forward(self, adjacency: torch.Tensor) -> torch.Tensor:
        degree = adjacency.sum(dim=-1)
        return torch.diag_embed(degree) - adjacency


def get_inputs():
    factors = torch.rand(256, 256)
    adjacency = (factors + factors.transpose(-1, -2)) * 0.5
    adjacency = adjacency * (1.0 - torch.eye(256))
    return [adjacency]


def get_init_inputs():
    return []
