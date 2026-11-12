"""All-pairs shortest paths with the Floyd-Warshall dynamic program.

Domain: mathematics / graph algorithms.
Solves: for every source-target pair and every allowed intermediate vertex k,
updates ``D[i,j] = min(D[i,j], D[i,k] + D[k,j])`` until all intermediate
vertices have been considered.
Source: Floyd, "Algorithm 97: Shortest path" (1962); Warshall, "A theorem on
Boolean matrices" (1962).
Level: 3 (N sequential updates over a dense distance matrix).
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def forward(self, adjacency: torch.Tensor) -> torch.Tensor:
        vertices = adjacency.shape[-1]
        distance = adjacency.clone()
        distance.diagonal(dim1=-2, dim2=-1).zero_()

        for intermediate in range(vertices):
            through_k = (
                distance[:, :, intermediate].unsqueeze(-1)
                + distance[:, intermediate, :].unsqueeze(-2)
            )
            distance = torch.minimum(distance, through_k)
        return distance


def get_inputs():
    batch, vertices = 4, 64
    adjacency = torch.rand(batch, vertices, vertices) + 0.1
    adjacency.diagonal(dim1=-2, dim2=-1).zero_()
    return [adjacency]


def get_init_inputs():
    return []
