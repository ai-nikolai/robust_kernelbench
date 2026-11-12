"""One Kalman-filter measurement update for a linear-Gaussian model.

Domain: statistics / signal processing / control.
Solves: given a prior ``(x, P)`` and a measurement ``z = Hx + noise``, computes
``S = HPH^T + R``, ``K = PH^T S^{-1}``, ``x' = x + K(z - Hx)`` and
``P' = (I - KH)P``.
Source: Kalman, "A new approach to linear filtering and prediction problems"
(1960).
Level: 2 (batched small-matrix algebra and a linear solve).
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def forward(
        self,
        state: torch.Tensor,
        covariance: torch.Tensor,
        measurement: torch.Tensor,
        observation_matrix: torch.Tensor,
        measurement_noise: torch.Tensor,
    ):
        predicted_measurement = state @ observation_matrix.transpose(-1, -2)
        innovation = measurement - predicted_measurement

        innovation_covariance = (
            observation_matrix @ covariance @ observation_matrix.transpose(-1, -2)
            + measurement_noise
        )
        state_measurement_covariance = covariance @ observation_matrix.transpose(-1, -2)
        gain = torch.linalg.solve(
            innovation_covariance,
            state_measurement_covariance.transpose(-1, -2),
        ).transpose(-1, -2)

        updated_state = state + (gain @ innovation.unsqueeze(-1)).squeeze(-1)
        identity = torch.eye(covariance.shape[-1], device=covariance.device, dtype=covariance.dtype)
        updated_covariance = (identity - gain @ observation_matrix) @ covariance
        return updated_state, updated_covariance


def get_inputs():
    batch, state_dim, obs_dim = 64, 8, 4
    state = torch.randn(batch, state_dim)
    factors = torch.randn(batch, state_dim, state_dim)
    covariance = factors @ factors.transpose(-1, -2) + 0.1 * torch.eye(state_dim)
    measurement = torch.randn(batch, obs_dim)
    observation_matrix = torch.randn(obs_dim, state_dim)
    measurement_noise = torch.diag(torch.rand(obs_dim) + 0.1)
    return [state, covariance, measurement, observation_matrix, measurement_noise]


def get_init_inputs():
    return []
