"""One split-step Fourier step of the 2D time-dependent Schrodinger equation.

Domain: physics / quantum mechanics.
Solves: advances
``i hbar dpsi/dt = -hbar^2/(2m) laplacian(psi) + V(x,y) psi``
by a Strang splitting: half kinetic step in Fourier space, full potential
phase, half kinetic step in Fourier space.
Source: Feit, Fleck & Steiger, "Solution of the Schrodinger equation by a
spectral method" (1982); Strang splitting (1968).
Level: 3 (complex FFTs and non-local kinetic operator).
"""

import math

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dt: float = 0.01, mass: float = 1.0, hbar: float = 1.0):
        super().__init__()
        self.dt = float(dt)
        self.mass = float(mass)
        self.hbar = float(hbar)

    def _kinetic_phase(self, size: int, device, dtype) -> torch.Tensor:
        frequencies = 2.0 * math.pi * torch.fft.fftfreq(size, device=device)
        squared = frequencies[:, None].pow(2) + frequencies[None, :].pow(2)
        return torch.exp(-1j * self.hbar * self.dt / (2.0 * self.mass) * squared).to(dtype)

    def forward(self, wavefunction: torch.Tensor, potential: torch.Tensor) -> torch.Tensor:
        phase = self._kinetic_phase(wavefunction.shape[-1], wavefunction.device, wavefunction.dtype)

        kinetic_half = torch.fft.ifft2(torch.fft.fft2(wavefunction, dim=(-2, -1)) * phase)
        potential_step = torch.exp(-1j * potential * self.dt / self.hbar) * kinetic_half
        return torch.fft.ifft2(torch.fft.fft2(potential_step, dim=(-2, -1)) * phase)


def get_inputs():
    wavefunction = torch.randn(4, 32, 32, dtype=torch.complex64)
    potential = torch.randn(4, 32, 32)
    return [wavefunction, potential]


def get_init_inputs():
    return [0.01, 1.0, 1.0]
