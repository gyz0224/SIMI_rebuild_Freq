"""BF-08: luminance Fourier amplitude x native smoothed bits -> curve logits.

Native Binary, smoothing, CBAM/fusion, DEA features, activations and seven
curve recurrences are untouched. Zero-initialized heads predict bounded
residuals before the native tanh(A) and .5+.5*sigmoid(B) activations.
"""
import torch
from torch import nn
from experiments.bit_frequency.advanced.layers import LocalMix, down_two, up_two, zero_output


class FourierBitCurveGuidance(nn.Module):
    def __init__(self, width=16, amplitude_bound=0.1, logit_bound=0.1):
        super().__init__()
        self.amplitude_bound = amplitude_bound
        self.logit_bound = logit_bound
        self.amplitude = nn.Sequential(nn.Conv2d(1, 8, 1), nn.GELU(), nn.Conv2d(8, 1, 1))
        self.bit_features = LocalMix(24, width)
        self.frequency_features = LocalMix(3, width)
        self.cross_gate = nn.Conv2d(2*width, width, 1)
        self.cross_features = LocalMix(3*width, width)
        self.stage_features = nn.ModuleList([LocalMix(32+width, width) for _ in range(7)])
        self.logit_heads = nn.ModuleList([nn.Conv2d(width, 6, 1) for _ in range(7)])
        for head in self.logit_heads:
            zero_output(head)
        self.capture = False
        self.observe = False
        self.maps = {}
        self.statistics = {}

    def frequency_prior(self, rgb):
        with torch.autocast(rgb.device.type, enabled=False):
            rgb = rgb.float()
            y = .299*rgb[:, :1] + .587*rgb[:, 1:2] + .114*rgb[:, 2:3]
            spectrum = torch.fft.rfft2(y, norm="ortho")
            log_amplitude = torch.log1p(spectrum.abs())
            log_gain = self.amplitude_bound*torch.tanh(self.amplitude(log_amplitude))
            # Enforce conjugate-pair equality on the self-conjugate columns.
            # Positive real gain preserves phase without angle/atan2 gradients.
            paired = torch.roll(log_gain.flip(-2), 1, -2)
            first = .5*(log_gain[..., :1]+paired[..., :1])
            if rgb.shape[-1] % 2 == 0 and log_gain.shape[-1] > 1:
                last = .5*(log_gain[..., -1:]+paired[..., -1:])
                log_gain = torch.cat((first, log_gain[..., 1:-1], last), -1)
            else:
                log_gain = torch.cat((first, log_gain[..., 1:]), -1)
            modified = spectrum*torch.exp(log_gain)
            prior = torch.fft.irfft2(modified, s=y.shape[-2:], norm="ortho")
        return y, prior, log_amplitude, log_gain

    def prepare(self, rgb, smoothed_bits):
        if rgb.ndim != 4 or rgb.shape[1] != 3 or smoothed_bits.shape[1] != 24:
            raise ValueError("Expected original RGB and native 24-channel smoothed Binary features")
        self.maps = {}
        self.statistics = {}
        with torch.autocast(rgb.device.type, enabled=False):
            y, prior, amplitude, gain = self.frequency_prior(rgb)
            bit = self.bit_features(down_two(smoothed_bits.float()))
            frequency = self.frequency_features(down_two(torch.cat((y, prior, prior-y), 1)))
            gate = torch.sigmoid(self.cross_gate(torch.cat((bit, frequency), 1)))
            context = self.cross_features(torch.cat((bit, gate*frequency, bit*frequency), 1))
        if self.capture:
            self.maps.update({k: v.detach() for k, v in {
                "luminance": y, "fourier_prior": prior, "log_amplitude": amplitude,
                "amplitude_log_gain": gain, "bit_frequency_gate": gate}.items()})
        if self.observe:
            self.statistics.update({"amplitude_log_gain_abs_mean": gain.detach().abs().mean(),
                "prior_minus_luminance_abs_mean": (prior-y).detach().abs().mean(),
                "bit_frequency_gate_mean": gate.detach().mean(),
                "bit_frequency_gate_std": gate.detach().std(unbiased=False)})
        return context

    def delta_logits(self, stage, native_stage_features, context):
        with torch.autocast(native_stage_features.device.type, enabled=False):
            x = torch.cat((down_two(native_stage_features.float()), context), 1)
            delta = self.logit_bound*torch.tanh(self.logit_heads[stage](self.stage_features[stage](x)))
            delta = up_two(delta, native_stage_features.shape[-2:])
        if self.capture:
            self.maps[f"stage_{stage+1}_delta_logits"] = delta.detach()
        if self.observe:
            self.statistics[f"stage_{stage+1}_delta_abs_mean"] = delta.detach().abs().mean()
        return delta

    def observe_curves(self, a, b):
        if self.observe:
            self.statistics.update({"curve_a_min": a.detach().amin(), "curve_a_max": a.detach().amax(),
                "curve_a_saturation_fraction": (a.detach().abs() >= .99).float().mean(),
                "curve_b_min": b.detach().amin(), "curve_b_max": b.detach().amax(),
                "curve_b_near_boundary_fraction": ((b.detach() <= .505) | (b.detach() >= .995)).float().mean()})
        if self.capture:
            self.maps.update(curve_a=a.detach(), curve_b=b.detach())
