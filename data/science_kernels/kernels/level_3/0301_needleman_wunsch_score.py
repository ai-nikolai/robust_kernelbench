"""Global sequence-alignment score (Needleman-Wunsch dynamic programming).

Domain: biology / bioinformatics.
Solves: computes the optimal global alignment score between batches of
integer-encoded sequences using match/mismatch/gap scores:
``D[i,j] = max(D[i-1,j-1] + score(a_i,b_j), D[i-1,j] + gap, D[i,j-1] + gap)``.
Source: Needleman & Wunsch, "A general method applicable to the search for
similarities in the amino acid sequence of two proteins" (1970).
Level: 3 (long sequential dynamic program; the dependency chain is the
optimisation target).
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, match_score: float = 1.0, mismatch_score: float = -1.0, gap_score: float = -1.0):
        super().__init__()
        self.match_score = float(match_score)
        self.mismatch_score = float(mismatch_score)
        self.gap_score = float(gap_score)

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        batch, first_length = first.shape
        second_length = second.shape[1]

        row = torch.arange(second_length + 1, device=first.device, dtype=torch.float32)
        row = row * self.gap_score
        previous = row.unsqueeze(0).expand(batch, -1).clone()

        for i in range(1, first_length + 1):
            current = torch.empty_like(previous)
            current[:, 0] = i * self.gap_score
            for j in range(1, second_length + 1):
                substitution = torch.where(
                    first[:, i - 1] == second[:, j - 1],
                    torch.full_like(previous[:, j - 1], self.match_score),
                    torch.full_like(previous[:, j - 1], self.mismatch_score),
                )
                diagonal = previous[:, j - 1] + substitution
                up = previous[:, j] + self.gap_score
                left = current[:, j - 1] + self.gap_score
                current[:, j] = torch.maximum(diagonal, torch.maximum(up, left))
            previous = current

        return previous[:, -1]


def get_inputs():
    return [
        torch.randint(0, 4, (16, 48)),
        torch.randint(0, 4, (16, 40)),
    ]


def get_init_inputs():
    return [1.0, -1.0, -1.0]
