"""Attach after native initialization; register before native bit smoothing."""
import importlib.util
from pathlib import Path
import sys

import torch

EXP = Path(__file__).resolve().parents[1]


def model_class(experiment):
    if experiment not in ("BF-06", "BF-07"):
        raise ValueError("Only BF-06/BF-07; M0 cannot be trained here")
    name = "_simi_" + experiment.replace("-", "_")
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, EXP/experiment/"model.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[name] = module
    module = sys.modules[name]
    return module.MultiScaleBitFrequency if experiment == "BF-06" else module.RGBGuidedBitFrequency


def attach(model, mechanism):
    if hasattr(model.light_net, "bit_frequency"):
        raise ValueError("A bit-frequency module is already attached")
    experiment = mechanism["experiment"]
    # New random initialization must not consume the native training RNG stream.
    # Construct on CPU, after the complete native backbone was initialized.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(mechanism["initialization_seed"])
        options = {"width": mechanism["width"]}
        if experiment == "BF-07":
            options["rgb_width"] = mechanism["rgb_width"]
        module = model_class(experiment)(**options)
    module.to(next(model.parameters()).device)
    module.train(model.light_net.training)
    model.light_net.add_module("bit_frequency", module)

    def before_smoothing(encoder, inputs):
        rgb, planes = inputs
        planes = module(planes) if experiment == "BF-06" else module(planes, rgb)
        return rgb, planes

    model._bf_advanced_hook = model.light_net.bit_encoder.register_forward_pre_hook(before_smoothing)
    return module


def make_model(config, device="cpu"):
    from models import SIMI
    model = SIMI(bit_mode=config["bit_mode"], enhancement_guidance=config["enhancement_guidance"],
                 enhancement_backbone=config["enhancement_backbone"],
                 activation_checkpointing=config["train"]["activation_checkpointing"]).to(device)
    attach(model, config["bit_frequency_mechanism"])
    return model
