"""One explicit step of the Hodgkin-Huxley squid-axon model.

Domain: biology / computational neuroscience.
Solves: advances the membrane potential and the sodium/potassium gating
variables ``(V, m, h, n)`` by one explicit Euler step:
``C dV/dt = I_ext - gNa m^3 h (V-ENa) - gK n^4 (V-EK) - gL (V-EL)``
plus the standard first-order gating kinetics.
Source: Hodgkin & Huxley, "A quantitative description of membrane current and
its application to conduction and excitation in nerve" (1952).
Level: 2 (nonlinear gating functions and several state variables).
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(
        self,
        capacitance: float = 1.0,
        g_na: float = 120.0,
        g_k: float = 36.0,
        g_l: float = 0.3,
        e_na: float = 50.0,
        e_k: float = -77.0,
        e_l: float = -54.387,
    ):
        super().__init__()
        self.capacitance = capacitance
        self.g_na = g_na
        self.g_k = g_k
        self.g_l = g_l
        self.e_na = e_na
        self.e_k = e_k
        self.e_l = e_l

    @staticmethod
    def _alpha_x(voltage: torch.Tensor, offset: float) -> torch.Tensor:
        x = (voltage + offset) / 10.0
        safe = torch.where(torch.abs(x) < 1e-6, torch.ones_like(x), x / (-torch.expm1(-x)))
        return safe

    def forward(self, state: torch.Tensor, external_current: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        voltage = torch.clamp(state[:, 0], -100.0, 100.0)
        m, h, n = state[:, 1], state[:, 2], state[:, 3]

        alpha_m = self._alpha_x(voltage, 40.0)
        beta_m = 4.0 * torch.exp(-(voltage + 65.0) / 18.0)
        alpha_h = 0.07 * torch.exp(-(voltage + 65.0) / 20.0)
        beta_h = 1.0 / (1.0 + torch.exp(-(voltage + 35.0) / 10.0))
        alpha_n = self._alpha_x(voltage, 55.0)
        beta_n = 0.125 * torch.exp(-(voltage + 65.0) / 80.0)

        d_m = alpha_m * (1.0 - m) - beta_m * m
        d_h = alpha_h * (1.0 - h) - beta_h * h
        d_n = alpha_n * (1.0 - n) - beta_n * n

        ionic_current = (
            self.g_na * m.pow(3) * h * (voltage - self.e_na)
            + self.g_k * n.pow(4) * (voltage - self.e_k)
            + self.g_l * (voltage - self.e_l)
        )
        d_v = (external_current - ionic_current) / self.capacitance
        return torch.stack(
            (
                state[:, 0] + dt * d_v,
                m + dt * d_m,
                h + dt * d_h,
                n + dt * d_n,
            ),
            dim=1,
        )


def get_inputs():
    state = torch.stack(
        (
            torch.randn(128) * 5.0 - 65.0,
            torch.rand(128) * 0.2 + 0.05,
            torch.rand(128) * 0.2 + 0.4,
            torch.rand(128) * 0.2 + 0.25,
        ),
        dim=1,
    )
    return [state, torch.randn(128) * 2.0, torch.full((128,), 0.01)]


def get_init_inputs():
    return []
