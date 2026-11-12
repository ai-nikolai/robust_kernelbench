"""One classical Runge-Kutta step of the Lotka-Volterra predator-prey system.

Domain: mathematical biology / population ecology.
Solves: advances the ODE
``dx/dt = alpha*x - beta*x*y`` and ``dy/dt = delta*x*y - gamma*y``
by one RK4 step for a batch of population states.
Source: Lotka, *Elements of Physical Biology* (1925); Volterra (1926).
Level: 1 (few elementwise operations per step).
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, alpha: float = 1.1, beta: float = 0.4, delta: float = 0.1, gamma: float = 0.4):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.delta = float(delta)
        self.gamma = float(gamma)

    def _rhs(self, state: torch.Tensor) -> torch.Tensor:
        prey = state[:, 0]
        predator = state[:, 1]
        d_prey = self.alpha * prey - self.beta * prey * predator
        d_predator = self.delta * prey * predator - self.gamma * predator
        return torch.stack((d_prey, d_predator), dim=1)

    def forward(self, state: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        k1 = self._rhs(state)
        k2 = self._rhs(state + 0.5 * dt[:, None] * k1)
        k3 = self._rhs(state + 0.5 * dt[:, None] * k2)
        k4 = self._rhs(state + dt[:, None] * k3)
        return state + (dt[:, None] / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def get_inputs():
    return [torch.rand(128, 2) + 0.1, torch.full((128,), 0.01)]


def get_init_inputs():
    return [1.1, 0.4, 0.1, 0.4]
