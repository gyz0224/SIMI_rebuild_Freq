"""Native Trainer extension: same losses, update order, schedulers and AMP."""
from copy import deepcopy
import csv
import json
from pathlib import Path

import torch

from trainers.trainer import Trainer
from .common import BUDGET, write_json, heatmap
from .modules import attach_bit_frequency


class BFTrainer(Trainer):
    def __init__(self, config):
        self.experiment = config["bit_frequency"]["experiment"]
        if self.experiment not in ("BF-01", "BF-05"):
            raise ValueError("M0 and other variants cannot be trained here")
        if config["train"]["load_pretrain"]:
            raise ValueError("Use explicit full BF resume; no warm-start from M0 or other branches")
        # Native constructor seeds and initializes the entire backbone in its
        # original order BEFORE the zero-initialized, nonrandom BF module.
        super().__init__(config)
        self.bf = attach_bit_frequency(self.model, self.experiment)
        self.optimizer_light.param_groups[0]["params"].extend(list(self.bf.parameters()))
        self.bf_stats = {"steps": 0, "successful_finite_nonzero_gradient_steps": 0,
                         "nonfinite_gradient_steps": 0, "nonzero_gradient_counts": [[0]*3 for _ in range(8)]}
        self._bf_logging = True
        self.assert_optimizer()

    def assert_optimizer(self):
        light = [p for group in self.optimizer_light.param_groups for p in group["params"]]
        noise = [p for group in self.optimizer_denoise.param_groups for p in group["params"]]
        if len({id(p) for p in light}) != len(light):
            raise RuntimeError("Duplicate enhancement optimizer parameters")
        for p in self.bf.parameters():
            if sum(q is p for q in light) != 1 or any(q is p for q in noise) or not p.requires_grad:
                raise RuntimeError("Frequency theta is not uniquely managed by active enhancement optimizer")
        return {"experiment": self.experiment, "frequency_parameters": sum(p.numel() for p in self.bf.parameters()),
                "model_parameters": sum(p.numel() for p in self.model.parameters()),
                "optimizer": "optimizer_light, original group, original hyperparameters",
                "groups": len(self.optimizer_light.param_groups), "passed": True}

    def _resume_signature(self):
        return {**super()._resume_signature(), "bit_frequency": deepcopy(self.config["bit_frequency"])}

    def train_step(self, *args, **kwargs):
        metrics = super().train_step(*args, **kwargs)
        self.bf_stats["steps"] += 1
        grad = self.bf.theta.grad if hasattr(self.bf, "theta") else None
        grad_norm = None
        if grad is not None:
            finite = bool(torch.isfinite(grad).all())
            if finite:
                grad_norm = float(grad.norm())
                if self.last_step_updates["light"] and grad.abs().sum() > 0:
                    self.bf_stats["successful_finite_nonzero_gradient_steps"] += 1
                    old = torch.tensor(self.bf_stats["nonzero_gradient_counts"], device=grad.device)
                    self.bf_stats["nonzero_gradient_counts"] = (old+(grad != 0)).cpu().tolist()
            else:
                self.bf_stats["nonfinite_gradient_steps"] += 1
        if self._bf_logging:
            directory = Path(self.config["output"]["root"]) / "logs"
            directory.mkdir(parents=True, exist_ok=True)
            row = {"step": self.iteration, **{k: float(v) for k, v in metrics.items()},
                   "lr_light": self.optimizer_light.param_groups[0]["lr"],
                   "lr_denoise": self.optimizer_denoise.param_groups[0]["lr"],
                   "light_updated": self.last_step_updates["light"], "denoise_updated": self.last_step_updates["denoise"],
                   "loss_scale": self.scaler.get_scale(), "theta_gradient_norm_after_native_clip": grad_norm}
            # Native AMP may skip overflowed gradients; never serialize Inf/NaN as valid stats.
            if not all(torch.isfinite(torch.tensor(float(v))) for v in metrics.values()):
                raise RuntimeError("Nonfinite loss encountered; see checkpoint/logs")
            with (directory / "training_metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False)+"\n")
            interval = self.config["bit_frequency"]["weight_log_interval"]
            if self.iteration <= 3 or self.iteration % interval == 0 or self.iteration == BUDGET:
                weights = self.bf.weights().detach().cpu()
                weight_row = {"step": self.iteration, "lambda_B0_to_B7_LH_HL_HH": weights.tolist(),
                              "saturation_fraction_abs_tanh_ge_0.99": float(((weights-1).abs()>=.99).float().mean()),
                              "gradient_norm": grad_norm, "light_updated": self.last_step_updates["light"]}
                with (directory/"bit_frequency_weights.jsonl").open("a") as stream:
                    stream.write(json.dumps(weight_row, allow_nan=False)+"\n")
        return metrics

    def _save_epoch_checkpoint(self, epoch, snapshots_folder):
        super()._save_epoch_checkpoint(epoch, snapshots_folder)
        if epoch+1 in self.config["bit_frequency"]["observation_epochs"]:
            self.save_checkpoint(snapshots_folder / f"epoch_{epoch+1:03d}_step_{self.iteration:06d}.pth")

    def save_checkpoint(self, path):
        path = super().save_checkpoint(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint["bit_frequency"] = deepcopy(self.config["bit_frequency"])
        checkpoint["bf_training_stats"] = deepcopy(self.bf_stats)
        temp = path.with_suffix(path.suffix+".bf.tmp")
        torch.save(checkpoint, temp)
        temp.replace(path)
        return path

    def load_checkpoint(self, path, *args, **kwargs):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("bit_frequency") != self.config["bit_frequency"]:
            raise ValueError("Only an identical BF experiment can be resumed")
        super().load_checkpoint(path, *args, **kwargs)
        self.bf_stats = deepcopy(checkpoint["bf_training_stats"])
        self.assert_optimizer()
        # Remove records ahead of restored state, e.g. crash after log append.
        directory = Path(self.config["output"]["root"]) / "logs"
        for name in ("training_metrics.jsonl", "bit_frequency_weights.jsonl"):
            target = directory/name
            if target.exists():
                rows = [r for r in target.read_text().splitlines() if json.loads(r)["step"] <= self.iteration]
                target.write_text("\n".join(rows)+("\n" if rows else ""))

    def export_frequency(self):
        out = Path(self.config["output"]["root"])
        weights = self.bf.weights().detach().cpu().numpy()
        write_json(out/"module_statistics.json", {**self.assert_optimizer(), "gradient_checks": self.bf_stats,
                   "weights_B0_to_B7": weights.tolist(),
                   "saturated": int(((abs(weights-1)) >= .99).sum()),
                   "high_frequency_weight_range": [float(weights.min()), float(weights.max())]})
        if self.experiment == "BF-05":
            heatmap(weights[::-1], ["LH", "HL", "HH"], f"{self.experiment} learned high-frequency weights, step {self.iteration}",
                    out/"learned_weights.png", cmap="coolwarm", fmt=".6f")
