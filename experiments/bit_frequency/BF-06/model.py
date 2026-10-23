"""BF-06: bit-only, two-level adaptive cross-bit/cross-scale interaction.

There is no RGB guidance and no prescribed bit hierarchy/noise mask. Dense
pointwise convolutions mix all RGB/bit/directional channels at each scale.
All seven independent wavelet residuals are learned, including LL2; this is
a new full mechanism, not the LL-preserving scalar BF-05 experiment.
"""
import torch
from torch import nn

from experiments.bit_frequency.advanced.layers import (
    LocalMix, up_two, down_two, zero_output, check_planes,
)
from experiments.bit_frequency.modules.haar import haar_dwt, haar_idwt


class MultiScaleBitFrequency(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.level1 = LocalMix(96, width)
        self.level2 = LocalMix(96, width)
        self.coarse_to_fine = nn.Conv2d(width, width, 1)
        self.fine_to_coarse = nn.Conv2d(width, width, 1)
        self.gate1 = nn.Conv2d(2*width, width, 1)
        self.gate2 = nn.Conv2d(2*width, width, 1)
        self.refine1 = LocalMix(width, width)
        self.refine2 = LocalMix(width, width)
        self.residual_high1 = nn.Conv2d(width, 72, 1)
        self.residual_bands2 = nn.Conv2d(width, 96, 1)
        zero_output(self.residual_high1)
        zero_output(self.residual_bands2)

    def _forward(self, planes, capture):
        check_planes(planes)
        # Keep binary decomposition/reconstruction FP32 under native FP16 AMP.
        with torch.autocast(planes.device.type, enabled=False):
            bands1, hw1 = haar_dwt(planes)
            bands2, hw2 = haar_dwt(bands1[0])
        fine = self.level1(torch.cat(bands1, 1))
        coarse = self.level2(torch.cat(bands2, 1))
        context1 = up_two(self.coarse_to_fine(coarse), fine.shape[-2:])
        context2 = self.fine_to_coarse(down_two(fine))
        gate1 = torch.sigmoid(self.gate1(torch.cat((fine, context1), 1)))
        gate2 = torch.sigmoid(self.gate2(torch.cat((coarse, context2), 1)))
        fused1 = fine + gate1*context1
        fused2 = coarse + gate2*context2
        fused1 = fused1 + self.refine1(fused1)
        fused2 = fused2 + self.refine2(fused2)
        delta_high1 = self.residual_high1(fused1)
        delta_bands2 = self.residual_bands2(fused2)
        with torch.autocast(planes.device.type, enabled=False):
            residual2 = delta_bands2.float().chunk(4, 1)
            delta_ll1 = haar_idwt(residual2, hw2)
            residual1 = (delta_ll1, *delta_high1.float().chunk(3, 1))
            delta = haar_idwt(residual1, hw1)
            output = planes + delta
        if not capture:
            return output
        return output, {
            "gate_level1": gate1, "gate_level2": gate2,
            "features_level1": fused1, "features_level2": fused2,
            "residual_L1_LH": residual1[1], "residual_L1_HL": residual1[2],
            "residual_L1_HH": residual1[3],
            **{f"residual_L2_{b}": v for b, v in zip(("LL", "LH", "HL", "HH"), residual2)},
            "delta_planes": delta,
        }

    def forward(self, planes):
        return self._forward(planes, False)

    def inspect(self, planes, rgb=None):
        # Common diagnostic interface; RGB is deliberately unused in BF-06.
        return self._forward(planes, True)

    @staticmethod
    def wavelet_scalar_ops(height, width, batch=1):
        n1 = ((height+1)//2)*((width+1)//2)
        n2 = (((height+1)//2+1)//2)*(((width+1)//2+1)//2)
        return batch*24*(34*(n1+n2)+height*width)
