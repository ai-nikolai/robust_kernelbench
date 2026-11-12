import torch
import torch.nn as nn


class Model(nn.Module):
    """Pairwise Gaussian (RBF) kernel matrix.

    Scientific workloads (Gaussian processes, kernel methods, molecular ML)
    spend most of their time in this operation.  A custom kernel can fuse the
    squared-distance computation with the exponential and avoid materialising
    the full distance matrix.
    """

    def __init__(self, gamma: float = 0.5):
        super().__init__()
        self.gamma = float(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        squared_distance = torch.cdist(x, y, p=2).pow(2)
        return torch.exp(-self.gamma * squared_distance)


def get_inputs():
    return [torch.randn(32, 16), torch.randn(24, 16)]


def get_init_inputs():
    return [0.5]
