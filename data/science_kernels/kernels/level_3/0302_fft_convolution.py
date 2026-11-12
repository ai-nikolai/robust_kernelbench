"""Circular convolution of batched signals via the FFT.

Domain: mathematics / signal processing.
Solves: computes the circular convolution of two length-N signals using the
convolution theorem: ``ifft(fft(a) * fft(b))``.
Source: Cooley & Tukey, "An algorithm for the machine calculation of complex
Fourier series" (1965); the convolution theorem is classical Fourier analysis.
Level: 3 (complex FFT/IFFT and pointwise multiplication).
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(first, dim=-1) * torch.fft.rfft(second, dim=-1)
        return torch.fft.irfft(spectrum, n=first.shape[-1], dim=-1)


def get_inputs():
    return [torch.randn(16, 512), torch.randn(16, 512)]


def get_init_inputs():
    return []
