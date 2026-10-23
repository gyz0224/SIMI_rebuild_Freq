"""BF-07: one-level, dual-representation RGB-conditioned bit-frequency fusion.

Unlike BF-06 this model has no second wavelet scale. Four bit-band branches
exchange information through a shared cross-frequency context. A separate
RGB wavelet/intensity branch predicts soft affine conditioning and gates;
RGB never replaces the bit residual, and no fixed edge/noise mask is used.
"""
import torch
from torch import nn
from torch.nn import functional as F

from experiments.bit_frequency.advanced.layers import LocalMix, zero_output, check_planes, down_two
from experiments.bit_frequency.modules.haar import BANDS, haar_dwt, haar_idwt


class RGBGuidedBitFrequency(nn.Module):
    def __init__(self, width=32, rgb_width=24):
        super().__init__()
        self.bit_stems = nn.ModuleDict({b: LocalMix(24, width) for b in BANDS})
        self.cross_frequency = LocalMix(4*width, width)
        # [RGB LL,LH,HL,HH, broader local intensity], aligned at H/2 x W/2.
        self.rgb_context = LocalMix(15, rgb_width)
        self.rgb_conditions = nn.ModuleDict({b: nn.Conv2d(rgb_width, 3*width, 1) for b in BANDS})
        self.bit_gates = nn.ModuleDict({b: nn.Conv2d(2*width, width, 1) for b in BANDS})
        self.refiners = nn.ModuleDict({b: LocalMix(width, width) for b in BANDS})
        self.residual_heads = nn.ModuleDict({b: nn.Conv2d(width, 24, 1) for b in BANDS})
        for layer in self.residual_heads.values():
            zero_output(layer)

    def _forward(self, planes, rgb, capture):
        check_planes(planes)
        if (rgb.ndim != 4 or rgb.shape[1] != 3 or rgb.dtype != torch.float32
                or rgb.shape[0] != planes.shape[0] or rgb.shape[-2:] != planes.shape[-2:]
                or rgb.device != planes.device):
            raise ValueError("RGB guidance must be the aligned native FP32 input")
        with torch.autocast(planes.device.type, enabled=False):
            bit_bands, hw = haar_dwt(planes)
            rgb_bands, _ = haar_dwt(rgb)
            # A 5x5 neighborhood before x2 averaging supplies intensity context
            # beyond the 2x2 Haar cell instead of duplicating a scaled LL band.
            intensity = down_two(F.avg_pool2d(rgb, 5, stride=1, padding=2, count_include_pad=False))
            rgb_input = torch.cat((*rgb_bands, intensity), 1)
        bit = {b: self.bit_stems[b](v) for b, v in zip(BANDS, bit_bands)}
        cross = self.cross_frequency(torch.cat([bit[b] for b in BANDS], 1))
        guidance = self.rgb_context(rgb_input)
        residuals, maps = [], {}
        for b in BANDS:
            gamma, beta, rgb_logit = self.rgb_conditions[b](guidance).chunk(3, 1)
            gate = torch.sigmoid(self.bit_gates[b](torch.cat((bit[b], cross), 1)) + rgb_logit)
            conditioned = (1+torch.tanh(gamma))*(bit[b]+cross) + torch.tanh(beta)
            fused = bit[b] + gate*conditioned
            fused = fused + self.refiners[b](fused)
            residuals.append(self.residual_heads[b](fused).float())
            if capture:
                maps.update({f"guide_gate_{b}": gate, f"guide_gain_{b}": 1+torch.tanh(gamma),
                             f"guide_bias_{b}": torch.tanh(beta), f"RGB_{b}": rgb_bands[BANDS.index(b)],
                             f"residual_L1_{b}": residuals[-1]})
        with torch.autocast(planes.device.type, enabled=False):
            delta = haar_idwt(residuals, hw)
            output = planes + delta
        if not capture:
            return output
        return output, {**maps, "RGB_context": guidance, "delta_planes": delta}

    def forward(self, planes, rgb):
        return self._forward(planes, rgb, False)

    def inspect(self, planes, rgb):
        return self._forward(planes, rgb, True)

    @staticmethod
    def wavelet_scalar_ops(height, width, batch=1):
        blocks = ((height+1)//2)*((width+1)//2)
        # RGB intensity pooling is excluded, as stated in the FLOPs report.
        return batch*(24*34*blocks + 3*18*blocks + 24*height*width)
