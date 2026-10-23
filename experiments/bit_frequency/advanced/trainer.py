"""Keep native losses, optimizers, AMP, schedulers, sampler and selection intact."""
from copy import deepcopy
import json
from pathlib import Path
import time

import torch

from trainers.trainer import Trainer
from .integration import attach
from .protocol import BUDGET, VARIANTS, write_json


class MechanismTrainer(Trainer):
    def __init__(self, config):
        self.mechanism = deepcopy(config["bit_frequency_mechanism"])
        self.experiment = self.mechanism["experiment"]
        if self.experiment not in VARIANTS or config["train"]["load_pretrain"]:
            raise ValueError("Only independent BF06/BF07 runs; no M0/other-branch warm start")
        super().__init__(config)
        self.bf = attach(self.model, self.mechanism)
        self.optimizer_light.param_groups[0]["params"].extend(self.bf.parameters())
        self.gradient_stats = {"observations": 0, "nonfinite_observations": 0,
            "first_effective_nonzero_step": {}, "nonzero_observation_counts": {n: 0 for n, p in self.bf.named_parameters()}}
        self.log_every_step = False
        self.assert_optimizer()

    def assert_optimizer(self):
        light = [p for g in self.optimizer_light.param_groups for p in g["params"]]
        noise = [p for g in self.optimizer_denoise.param_groups for p in g["params"]]
        if len({id(p) for p in light}) != len(light):
            raise RuntimeError("Duplicate enhancement optimizer parameters")
        for name, p in self.bf.named_parameters():
            if not p.requires_grad or sum(p is q for q in light) != 1 or any(p is q for q in noise):
                raise RuntimeError(f"New parameter is not uniquely managed by optimizer_light: {name}")
        return {"passed": True, "extra_parameters": sum(p.numel() for p in self.bf.parameters()),
            "total_parameters": sum(p.numel() for p in self.model.parameters()),
            "new_parameter_tensors": len(list(self.bf.parameters())), "optimizer": "native enhancement group",
            "optimizer_groups": len(self.optimizer_light.param_groups), "same_learning_rate_and_weight_decay": True}

    def _resume_signature(self):
        return {**super()._resume_signature(), "bit_frequency_mechanism": deepcopy(self.mechanism)}

    def train_step(self, *args, **kwargs):
        started = time.perf_counter()
        metrics = super().train_step(*args, **kwargs)
        observe = (self.log_every_step or self.iteration <= 5
                   or self.iteration % self.mechanism["gradient_log_interval"] == 0 or self.iteration == BUDGET)
        # Validate losses each batch; expected AMP overflows retain native skips.
        values = torch.stack(list(metrics.values())).float().detach()
        if not bool(torch.isfinite(values).all()):
            raise RuntimeError(f"Nonfinite loss at step {self.iteration}; formal training stopped")
        if observe:
            names, tensors = [], []
            for name, p in self.bf.named_parameters():
                names.append(name)
                g = p.grad
                tensors.append(torch.stack((p.detach().norm(), p.new_tensor(0) if g is None else g.detach().norm())))
            states = torch.stack(tensors).detach().cpu().tolist()
            self.gradient_stats["observations"] += 1
            gradients = {}
            import math
            for name, (pnorm, gnorm) in zip(names, states):
                finite = math.isfinite(gnorm)
                gradients[name] = gnorm if finite else None
                if not finite:
                    self.gradient_stats["nonfinite_observations"] += 1
                if finite and gnorm > 0 and self.last_step_updates["light"]:
                    self.gradient_stats["nonzero_observation_counts"][name] += 1
                    self.gradient_stats["first_effective_nonzero_step"].setdefault(name, self.iteration)
            directory = Path(self.config["output"]["root"])/"logs"
            directory.mkdir(parents=True, exist_ok=True)
            row = {"step": self.iteration, **dict(zip(metrics, values.cpu().tolist())),
                   "lr_light": self.optimizer_light.param_groups[0]["lr"],
                   "lr_denoise": self.optimizer_denoise.param_groups[0]["lr"],
                   "loss_scale": self.scaler.get_scale(), **self.last_step_updates,
                   "module_grad_norms_after_native_clip": gradients,
                   "batch_compute_seconds": time.perf_counter()-started}
            with (directory/"training_metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False)+"\n")
        return metrics

    def _save_epoch_checkpoint(self, epoch, snapshots_folder):
        super()._save_epoch_checkpoint(epoch, snapshots_folder)
        if epoch+1 in self.mechanism["observation_epochs"]:
            self.save_checkpoint(snapshots_folder/f"epoch_{epoch+1:03d}_step_{self.iteration:06d}.pth")
        self.export_statistics()

    def save_checkpoint(self, path):
        path = super().save_checkpoint(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        checkpoint.update(mechanism=deepcopy(self.mechanism), module_gradient_statistics=deepcopy(self.gradient_stats))
        temporary = path.with_suffix(path.suffix+".mechanism.tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(path)
        return path

    def load_checkpoint(self, path, *args, **kwargs):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("mechanism") != self.mechanism:
            raise ValueError("Resume requires an identical BF mechanism/configuration")
        if kwargs.get("resume_mode", "full") != "full":
            raise ValueError("Only exact full-state resume is supported")
        super().load_checkpoint(path, *args, **kwargs)
        self.gradient_stats = deepcopy(checkpoint["module_gradient_statistics"])
        self.assert_optimizer()
        log = Path(self.config["output"]["root"])/"logs/training_metrics.jsonl"
        if log.exists():
            lines = [s for s in log.read_text().splitlines() if json.loads(s)["step"] <= self.iteration]
            log.write_text("\n".join(lines)+("\n" if lines else ""))

    def export_statistics(self):
        missing = [n for n, count in self.gradient_stats["nonzero_observation_counts"].items() if count == 0]
        report = {**self.assert_optimizer(), "step": self.iteration, "gradient_statistics": self.gradient_stats,
                  "parameters_without_observed_effective_gradient": missing,
                  "zero_heads_note": "Upstream gradients are zero until a residual head first updates; this is expected."}
        write_json(Path(self.config["output"]["root"])/"module_statistics.json", report)
        return report
