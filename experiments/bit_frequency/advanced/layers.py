"""Small convolutional layers and deterministic, aligned two-scale resizing."""
import torch
from torch import nn
from torch.nn import functional as F


class LocalMix(nn.Module):
    """Pointwise cross-bit/frequency mixing followed by local spatial mixing."""
    def __init__(self, in_channels, width):
        super().__init__()
        self.pointwise = nn.Conv2d(in_channels, width, 1)
        self.spatial = nn.Conv2d(width, width, 3, padding=1, groups=width)
        self.output = nn.Conv2d(width, width, 1)

    def forward(self, x):
        x = F.gelu(self.pointwise(x))
        return self.output(F.gelu(self.spatial(x)))


def up_two(x, hw):
    """Nearest x2 with an explicit crop, including odd parent sizes."""
    if any(not 2*n-1 <= m <= 2*n for n, m in zip(x.shape[-2:], hw)):
        raise ValueError("Cross-scale parent/child sizes must differ by Haar x2")
    return x.repeat_interleave(2, -2).repeat_interleave(2, -1)[..., :hw[0], :hw[1]]


def down_two(x):
    if x.shape[-2] % 2:
        x = torch.cat((x, x[..., -1:, :]), dim=-2)
    if x.shape[-1] % 2:
        x = torch.cat((x, x[..., -1:]), dim=-1)
    return F.avg_pool2d(x, 2)


def zero_output(layer):
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)


def check_planes(planes):
    if (planes.ndim != 4 or planes.shape[1] != 24 or planes.dtype != torch.float32
            or min(planes.shape[-2:]) < 1):
        raise ValueError("Expected native FP32 [N,24,H,W] planes, RGB-major B7..B0")
