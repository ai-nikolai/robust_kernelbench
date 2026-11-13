"""Cox-Ross-Rubinstein binomial price of a European call option.

Domain: economics / quantitative finance.
Solves: builds a recombining binomial tree with
``u = exp(sigma sqrt(dt))``, ``d = 1/u`` and
``p = (exp(r dt) - d) / (u - d)``, then discounts the terminal payoff back
through the tree.
Source: Cox, Ross & Rubinstein, "Option pricing: a simplified approach" (1979).
Level: 3 (sequential backward induction over tree levels).
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, steps: int = 64):
        super().__init__()
        self.steps = int(steps)

    def forward(
        self,
        spot: torch.Tensor,
        strike: torch.Tensor,
        maturity: torch.Tensor,
        rate: torch.Tensor,
        volatility: torch.Tensor,
    ) -> torch.Tensor:
        steps = self.steps
        dt = maturity / steps
        up = torch.exp(volatility * torch.sqrt(dt))
        down = 1.0 / up
        probability = (torch.exp(rate * dt) - down) / (up - down)
        discount = torch.exp(-rate * dt)

        nodes = torch.arange(steps + 1, device=spot.device, dtype=spot.dtype)
        log_terminal = (
            torch.log(spot)[:, None]
            + nodes[None, :] * torch.log(up)[:, None]
            + (steps - nodes)[None, :] * torch.log(down)[:, None]
        )
        terminal = torch.exp(log_terminal)
        values = torch.clamp(terminal - strike[:, None], min=0.0)

        for _ in range(steps):
            values = discount[:, None] * (
                probability[:, None] * values[:, 1:]
                + (1.0 - probability[:, None]) * values[:, :-1]
            )
        return values[:, 0]


def get_inputs():
    batch = 64
    return [
        torch.rand(batch) * 100.0 + 50.0,
        torch.rand(batch) * 100.0 + 50.0,
        torch.rand(batch) * 1.5 + 0.5,
        torch.rand(batch) * 0.07 + 0.01,
        torch.rand(batch) * 0.4 + 0.1,
    ]


def get_init_inputs():
    return [64]
