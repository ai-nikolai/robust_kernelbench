"""Horner evaluation of a polynomial at many points.

Domain: numerical mathematics / scientific computing.
Solves: evaluates
``p(x) = c_0 + c_1 x + ... + c_{K-1} x^{K-1}``
using Horner's recurrence, which requires K multiplications and additions
instead of computing every power separately.
Source: Horner, "A new method of solving numerical equations of all orders"
(1819); see also Higham, *Accuracy and Stability of Numerical Algorithms*
(2nd ed., 2002).
Level: 1 (short sequential recurrence over coefficients).
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def forward(self, coefficients: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        result = torch.full_like(x, float(coefficients[-1]))
        for index in range(coefficients.shape[0] - 2, -1, -1):
            result = result * x + coefficients[index]
        return result


def get_inputs():
    return [torch.randn(16), torch.randn(4096)]


def get_init_inputs():
    return []
