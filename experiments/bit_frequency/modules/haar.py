"""Orthonormal one-level Haar; RGB-major, MSB-first native Binary channels.

For [[a,b],[c,d]], LH=(-a-b+c+d)/2 (high along rows),
HL=(-a+b-c+d)/2 (high along columns). Other naming conventions exist;
these definitions, signs, and axes are fixed for every BF artifact.
"""

import torch
from torch import nn

BANDS = ("LL", "LH", "HL", "HH")
CHANNEL_MAP = {f"B{k}": [c * 8 + 7 - k for c in range(3)] for k in range(7, -1, -1)}


def haar_dwt(x):
    """Return (LL,LH,HL,HH), original HW; replicate bottom/right if odd.

    Concatenating boundary slices avoids CUDA replicate_pad backward's
    nondeterminism. No convolution, quantization, thresholding, or rebinarizing.
    """
    if x.ndim != 4 or not x.is_floating_point() or min(x.shape[-2:]) < 1:
        raise ValueError("Haar requires nonempty floating BCHW")
    hw = tuple(x.shape[-2:])
    if hw[0] % 2:
        x = torch.cat((x, x[..., -1:, :]), dim=-2)
    if hw[1] % 2:
        x = torch.cat((x, x[..., -1:]), dim=-1)
    a, b = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
    c, d = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
    return ((a + b + c + d) * 0.5, (-a - b + c + d) * 0.5,
            (-a + b - c + d) * 0.5, (a - b - c + d) * 0.5), hw


def haar_idwt(bands, hw):
    if len(bands) != 4 or any(t.shape != bands[0].shape for t in bands):
        raise ValueError("Four equally shaped Haar subbands required")
    ll, lh, hl, hh = bands
    if not all(2 * n - 1 <= m <= 2 * n for n, m in zip(ll.shape[-2:], hw)):
        raise ValueError("Invalid original shape for subbands")
    a, b = (ll - lh - hl + hh) * 0.5, (ll - lh + hl - hh) * 0.5
    c, d = (ll + lh - hl - hh) * 0.5, (ll + lh + hl + hh) * 0.5
    top = torch.stack((a, b), dim=-1).flatten(-2)
    bottom = torch.stack((c, d), dim=-1).flatten(-2)
    output = torch.stack((top, bottom), dim=-2).flatten(-3, -2)
    return output[..., :hw[0], :hw[1]]


class BitFrequencyModulation(nn.Module):
    """BF-01: fixed B0/B1/B2 highs × .75; BF-05: 24 shared RGB weights.

    theta row k always means physical bit Bk, so plots explicitly reorder
    rows B7..B0. Wavelet arithmetic uses native float32 outside AMP;
    the original backbone and its precision policy are unchanged.
    """

    def __init__(self, experiment):
        super().__init__()
        if experiment not in ("BF-01", "BF-05"):
            raise ValueError("Only BF-01 and BF-05 are authorized")
        self.experiment = experiment
        if experiment == "BF-05":
            self.theta = nn.Parameter(torch.zeros(8, 3))
        else:
            weights = torch.ones(8, 3)
            weights[:3] = 0.75
            self.register_buffer("fixed_weights", weights)

    def weights(self):
        return 1 + self.theta.tanh() if self.experiment == "BF-05" else self.fixed_weights

    def forward(self, planes):
        if planes.ndim != 4 or planes.shape[1] != 24 or planes.dtype != torch.float32:
            raise ValueError("Native float32 [B,24,H,W] Binary planes required")
        with torch.autocast(planes.device.type, enabled=False):
            (ll, lh, hl, hh), hw = haar_dwt(planes)
            # Repeat MSB..LSB for R, then G, then B; LL remains untouched.
            w = self.weights().flip(0).repeat(3, 1)
            modulated = [ll] + [band * w[:, f].view(1, 24, 1, 1)
                                for f, band in enumerate((lh, hl, hh))]
            return haar_idwt(modulated, hw)

    def arithmetic_flops(self, height, width, batch=1):
        """Actual scalar +,-,× counts for this implementation, excluding tanh.

        Per padded 2×2/channel: DWT=18 (including 2 unary negations),
        IDWT=16, modulation=3 => 37. Without unary sign operations: 35.
        Tanh and the 24 lambda additions are reported separately for BF-05.
        Memory copies, indexing, padding, and cropping are not FLOPs.
        """
        return batch * 24 * ((height + 1) // 2) * ((width + 1) // 2) * 37
