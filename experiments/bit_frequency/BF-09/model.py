"""BF-09: bit-guided, LL-preserving repair of continuous enhanced RGB.

This module is inside the denoising branch AFTER the original detach. Its
output enters the unchanged denoiser; the original detached enhancement
remains the native denoising target. No wavelet transform of Binary planes.
"""
import torch
from torch import nn
from experiments.bit_frequency.advanced.layers import LocalMix, down_two, zero_output
from experiments.bit_frequency.modules.haar import haar_dwt, haar_idwt


class BitGuidedWaveletRestoration(nn.Module):
    def __init__(self, width=24, high_frequency_bound=0.1):
        super().__init__()
        self.high_frequency_bound = high_frequency_bound
        self.image_features = LocalMix(12, width)
        self.bit_features = LocalMix(24, width)
        self.cross_gate = nn.Conv2d(2*width, width, 1)
        self.restoration = LocalMix(3*width, width)
        self.high_residual = nn.Conv2d(width, 9, 1)
        zero_output(self.high_residual)
        self.capture = False
        self.observe = False
        self.maps = {}
        self.statistics = {}

    def forward(self, enhanced_detached, smoothed_bits_detached):
        if enhanced_detached.shape[1] != 3 or smoothed_bits_detached.shape[1] != 24:
            raise ValueError("Expected continuous enhanced RGB and native smoothed 24-channel bit features")
        if enhanced_detached.requires_grad or smoothed_bits_detached.requires_grad:
            raise ValueError("BF09 must receive the native detached enhancement and detached bit features")
        self.maps = {}
        self.statistics = {}
        with torch.autocast(enhanced_detached.device.type, enabled=False):
            enhanced = enhanced_detached.float()
            bands, hw = haar_dwt(enhanced)
            image = self.image_features(torch.cat(bands, 1))
            bit = self.bit_features(down_two(smoothed_bits_detached.float()))
            gate = torch.sigmoid(self.cross_gate(torch.cat((image, bit), 1)))
            fused = self.restoration(torch.cat((image, gate*bit, image*bit), 1))
            residuals = (self.high_frequency_bound*torch.tanh(self.high_residual(fused))).chunk(3, 1)
            h, w = hw
            sh, sw = bands[0].shape[-2:]
            row = (torch.arange(sh, device=enhanced.device) < sh-(h % 2)).to(enhanced.dtype).view(1, 1, sh, 1)
            col = (torch.arange(sw, device=enhanced.device) < sw-(w % 2)).to(enhanced.dtype).view(1, 1, 1, sw)
            # Project odd-size edge residuals to the valid replicate-boundary
            # Haar subspace. This is boundary consistency, not a noise mask.
            residuals = (residuals[0]*row, residuals[1]*col, residuals[2]*row*col)
            delta = haar_idwt((torch.zeros_like(bands[0]), *residuals), hw)
            repaired = enhanced+delta
        if self.capture:
            self.maps = {"original_enhancement": enhanced.detach(), "repaired_enhancement": repaired.detach(),
                "LL": bands[0].detach(), "bit_frequency_gate": gate.detach(), "delta_image": delta.detach(),
                **{f"high_residual_{n}": v.detach() for n, v in zip(("LH", "HL", "HH"), residuals)}}
        if self.observe:
            new_ll = haar_dwt(repaired.detach())[0][0]
            self.statistics = {"high_residual_abs_mean": torch.cat(residuals, 1).detach().abs().mean(),
                "high_residual_max_abs": torch.cat(residuals, 1).detach().abs().amax(),
                "LL_reanalysis_max_error": (new_ll-bands[0].detach()).abs().amax(),
                "gate_mean": gate.detach().mean(), "gate_std": gate.detach().std(unbiased=False),
                "repaired_min": repaired.detach().amin(), "repaired_max": repaired.detach().amax()}
        return repaired
