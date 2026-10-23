"""Attach to existing modules without rewriting native SIMI calculations."""
import importlib.util
from pathlib import Path
import sys
import torch

EXP = Path(__file__).resolve().parents[1]
VARIANTS = ("BF-08", "BF-09")


def model_class(variant):
    if variant not in VARIANTS:
        raise ValueError("Only BF08/BF09 are supported; no M0 training")
    key = "_simi_stage_"+variant.replace("-", "_")
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, EXP/variant/"model.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[key] = module
    module = sys.modules[key]
    return module.FourierBitCurveGuidance if variant == "BF-08" else module.BitGuidedWaveletRestoration


def attach(model, spec):
    v = spec["experiment"]
    if hasattr(model, "_stage_frequency_state"):
        raise ValueError("Stage frequency module already attached")
    with torch.inference_mode(False), torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(spec["initialization_seed"])
        options = {"width": spec["width"]}
        if v == "BF-08":
            options.update(amplitude_bound=spec["amplitude_bound"], logit_bound=spec["logit_bound"])
        else:
            options.update(high_frequency_bound=spec["high_frequency_bound"])
        module = model_class(v)(**options)
        module.to(next(model.parameters()).device)
    owner = model.light_net if v == "BF-08" else model.denoise
    owner.add_module("stage_frequency", module)
    module.train(owner.training)
    state, handles = {}, []
    model._stage_frequency_state = state

    def before_encoder(encoder, inputs):
        rgb, raw = inputs
        state.clear()
        state.update(rgb=rgb, raw_planes=raw, stage_deltas={})

    def after_smoothing(layer, inputs, output):
        # The native extract_features API calls encoder.extract_features directly,
        # bypassing Module.__call__ and its pre-hook. Preserve that readout path.
        if "rgb" not in state:
            return
        if v == "BF-08":
            state["context"] = module.prepare(state["rgb"], output)
        else:
            state["bit_features"] = output.detach()

    handles.append(model.light_net.bit_encoder.register_forward_pre_hook(before_encoder))
    handles.append(model.light_net.bit_encoder.smoothing[-1].register_forward_hook(after_smoothing))
    if v == "BF-08":
        def make_head_hook(stage, kind):
            def hook(head, inputs, output):
                if kind == "a":
                    state["stage_deltas"][stage] = module.delta_logits(stage, inputs[0], state["context"])
                    delta = state["stage_deltas"][stage][:, :3]
                else:
                    delta = state["stage_deltas"].pop(stage)[:, 3:]
                return output+delta.to(output.dtype)
            return hook
        for stage in range(7):
            for kind in ("a", "b"):
                head = getattr(model.light_net.curve_network, f"e_conv{stage+1}_{kind}")
                handles.append(head.register_forward_hook(make_head_hook(stage, kind)))

        def after_light(light, inputs, output):
            module.observe_curves(output[1], output[2])
            state.clear()
        handles.append(model.light_net.register_forward_hook(after_light))
    else:
        def before_denoise(denoise, inputs):
            enhanced, features = inputs
            if "bit_features" not in state:
                raise RuntimeError("BF09 denoising needs the current native bit encoder forward")
            bits = state.pop("bit_features")
            repaired = module(enhanced, bits)
            state.clear()
            return repaired, features
        handles.append(model.denoise.register_forward_pre_hook(before_denoise))
    model._stage_frequency_handles = handles
    return module


def make_model(config, device="cpu"):
    from models import SIMI
    model = SIMI(bit_mode=config["bit_mode"], enhancement_guidance=config["enhancement_guidance"],
                 enhancement_backbone=config["enhancement_backbone"],
                 activation_checkpointing=config["train"]["activation_checkpointing"]).to(device)
    attach(model, config["stage_frequency"])
    return model
