"""Use the verified native model and all its original state-dict keys."""

from models import SIMI
from .haar import BitFrequencyModulation


def attach_bit_frequency(model, experiment):
    if hasattr(model.light_net, "bit_frequency"):
        raise ValueError("Frequency module already attached")
    module = BitFrequencyModulation(experiment).to(next(model.parameters()).device)
    # Registration under light_net makes theta part of enhancement parameters,
    # grad clipping, state_dict, eval/train transitions and the real optimizer.
    model.light_net.add_module("bit_frequency", module)

    def before_encoder(encoder, inputs):
        rgb, planes = inputs
        return rgb, module(planes)

    # Native encoder.forward calls native smoothing. All training/validation/
    # inference paths pass here, whether explicit planes or internal Binary.
    model._bf_hook_handle = model.light_net.bit_encoder.register_forward_pre_hook(before_encoder)
    return module


def make_model(config, *, device="cpu"):
    model = SIMI(bit_mode=config["bit_mode"],
                 enhancement_guidance=config["enhancement_guidance"],
                 enhancement_backbone=config["enhancement_backbone"],
                 activation_checkpointing=config["train"]["activation_checkpointing"])
    model.to(device)
    attach_bit_frequency(model, config["bit_frequency"]["experiment"])
    return model
