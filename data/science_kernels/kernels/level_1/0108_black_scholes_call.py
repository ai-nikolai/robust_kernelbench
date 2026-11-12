"""Vectorised Black-Scholes price of a European call option.

Domain: economics / quantitative finance.
Solves: evaluates
``C = S * Phi(d1) - K * exp(-rT) * Phi(d2)``,
``d1 = (log(S/K) + (r + sigma^2/2)T) / (sigma sqrt(T))``, ``d2 = d1 - sigma sqrt(T)``,
where ``Phi`` is the standard normal CDF.
Source: Black & Scholes, "The pricing of options and corporate liabilities"
(1973).
Level: 1 (elementwise transcendental functions).
"""

import math

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, risk_free_rate: float = 0.05):
        super().__init__()
        self.risk_free_rate = float(risk_free_rate)

    @staticmethod
    def _normal_cdf(value: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1.0 + torch.erf(value / math.sqrt(2.0)))

    def forward(
        self,
        spot: torch.Tensor,
        strike: torch.Tensor,
        maturity: torch.Tensor,
        volatility: torch.Tensor,
    ) -> torch.Tensor:
        sqrt_t = torch.sqrt(maturity)
        d1 = (
            torch.log(spot / strike)
            + (self.risk_free_rate + 0.5 * volatility * volatility) * maturity
        ) / (volatility * sqrt_t)
        d2 = d1 - volatility * sqrt_t
        return spot * self._normal_cdf(d1) - strike * torch.exp(
            -self.risk_free_rate * maturity
        ) * self._normal_cdf(d2)


def get_inputs():
    return [
        torch.rand(1024) * 100.0 + 50.0,
        torch.rand(1024) * 100.0 + 50.0,
        torch.rand(1024) + 0.1,
        torch.rand(1024) * 0.5 + 0.1,
    ]


def get_init_inputs():
    return [0.05]
