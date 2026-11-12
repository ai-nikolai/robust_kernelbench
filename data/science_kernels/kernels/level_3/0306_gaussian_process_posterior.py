"""Gaussian-process posterior mean with an RBF kernel.

Domain: statistics / Bayesian machine learning.
Solves: evaluates
``mean(X*) = K(X*,X) (K(X,X) + sigma_n^2 I)^{-1} y``
for batched training/test sets and an isotropic squared-exponential kernel.
Source: Rasmussen & Williams, *Gaussian Processes for Machine Learning* (2006),
Chapter 2.
Level: 3 (kernel matrix construction and batched linear solves).
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, lengthscale: float = 0.7, signal_variance: float = 1.0, noise: float = 0.1):
        super().__init__()
        self.lengthscale = float(lengthscale)
        self.signal_variance = float(signal_variance)
        self.noise = float(noise)

    def _kernel(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        squared_distance = torch.cdist(left, right, p=2).pow(2)
        return self.signal_variance * torch.exp(
            -0.5 * squared_distance / (self.lengthscale * self.lengthscale)
        )

    def forward(
        self,
        train_inputs: torch.Tensor,
        train_targets: torch.Tensor,
        test_inputs: torch.Tensor,
    ) -> torch.Tensor:
        kernel_train = self._kernel(train_inputs, train_inputs)
        identity = torch.eye(
            kernel_train.shape[-1], device=kernel_train.device, dtype=kernel_train.dtype
        )
        kernel_train = kernel_train + (self.noise * self.noise) * identity
        kernel_test = self._kernel(test_inputs, train_inputs)

        weights = torch.linalg.solve(kernel_train, train_targets.unsqueeze(-1)).squeeze(-1)
        return kernel_test @ weights.unsqueeze(-1)


def get_inputs():
    batch, train_size, test_size, dimension = 8, 48, 24, 4
    return [
        torch.randn(batch, train_size, dimension),
        torch.randn(batch, train_size),
        torch.randn(batch, test_size, dimension),
    ]


def get_init_inputs():
    return [0.7, 1.0, 0.1]
