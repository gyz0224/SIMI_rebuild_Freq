"""Native training with correctly owned new parameters and actual update counts."""
from copy import deepcopy
import json
import math
from pathlib import Path
import time
import torch

from trainers.trainer import Trainer
from .integration import attach
from .protocol import BUDGET, VARIANTS, write_json


class StageTrainer(Trainer):
    def __init__(self, config):
        self.spec = deepcopy(config["stage_frequency"])
        self.variant = self.spec["experiment"]
        self.branch = self.spec["optimizer_branch"]
        if self.variant not in VARIANTS or config["train"]["load_pretrain"]:
            raise ValueError("Only independent BF08/BF09; no warm starts from any baseline/candidate")
        super().__init__(config)
        self.module = attach(self.model, self.spec)
        optimizer = self.optimizer_light if self.branch == "light" else self.optimizer_denoise
        optimizer.param_groups[0]["params"].extend(self.module.parameters())
        self.stage_stats = {"optimizer_updates": {"light": 0, "denoise": 0}, "gradient_observations": 0,
            "nonfinite_gradient_observations": 0,
            "first_effective_gradient_step": {}, "nonzero_gradient_observations": {n: 0 for n, p in self.module.named_parameters()}}
        self.log_every_step = False
        self.assert_optimizer()

    def assert_optimizer(self):
        groups = {"light": [p for g in self.optimizer_light.param_groups for p in g["params"]],
                  "denoise": [p for g in self.optimizer_denoise.param_groups for p in g["params"]]}
        for group in groups.values():
            assert len(group) == len({id(p) for p in group})
        other = "denoise" if self.branch == "light" else "light"
        for name, p in self.module.named_parameters():
            assert p.requires_grad and sum(p is q for q in groups[self.branch]) == 1
            assert not any(p is q for q in groups[other]), name
        return {"passed": True, "experiment": self.variant, "optimizer_branch": self.branch,
            "total_parameters": sum(p.numel() for p in self.model.parameters()),
            "new_parameters": sum(p.numel() for p in self.module.parameters()),
            "parameter_tensors": len(list(self.module.parameters())), "native_optimizer_group_extended": True}

    def _resume_signature(self):
        return {**super()._resume_signature(), "stage_frequency": self.spec}

    def train_step(self, *args, **kwargs):
        t = time.perf_counter()
        next_step = self.iteration+1
        observe = self.log_every_step or next_step <= 5 or next_step % self.spec["log_interval"] == 0 or next_step == BUDGET
        self.module.observe = observe
        result = super().train_step(*args, **kwargs)
        for n in self.stage_stats["optimizer_updates"]:
            self.stage_stats["optimizer_updates"][n] += int(self.last_step_updates[n])
        values = torch.stack(list(result.values())).detach().float()
        if not bool(torch.isfinite(values).all()):
            raise RuntimeError(f"Nonfinite native loss at iteration {self.iteration}")
        if observe:
            names, tensors = [], []
            for n, p in self.module.named_parameters():
                names.append(n)
                g = p.grad
                tensors.append(p.new_tensor(0) if g is None else g.detach().abs().amax())
            gradients = dict(zip(names, torch.stack(tensors).cpu().tolist()))
            self.stage_stats["gradient_observations"] += 1
            for n, value in list(gradients.items()):
                if not math.isfinite(value):
                    self.stage_stats["nonfinite_gradient_observations"] += 1
                    gradients[n] = None
                elif value > 0 and self.last_step_updates[self.branch]:
                    self.stage_stats["nonzero_gradient_observations"][n] += 1
                    self.stage_stats["first_effective_gradient_step"].setdefault(n, self.iteration)
            diagnostics = {k: float(v) for k, v in self.module.statistics.items()}
            if not all(math.isfinite(x) for x in diagnostics.values()):
                raise RuntimeError("Nonfinite stage-module diagnostics")
            row = {"step": self.iteration, **dict(zip(result, values.cpu().tolist())),
                "optimizer_updates": deepcopy(self.stage_stats["optimizer_updates"]),
                "branch_updates_this_batch": deepcopy(self.last_step_updates),
                "lr_light": self.optimizer_light.param_groups[0]["lr"],
                "lr_denoise": self.optimizer_denoise.param_groups[0]["lr"],
                "AMP_scale": self.scaler.get_scale(), "new_parameter_max_abs_grad_after_native_clip": gradients,
                "module_diagnostics": diagnostics, "batch_compute_seconds": time.perf_counter()-t}
            directory = Path(self.config["output"]["root"])/"logs"
            directory.mkdir(parents=True, exist_ok=True)
            with (directory/"training_metrics.jsonl").open("a") as f:
                f.write(json.dumps(row, allow_nan=False)+"\n")
        return result

    def _save_epoch_checkpoint(self, epoch, folder):
        super()._save_epoch_checkpoint(epoch, folder)
        if epoch+1 in self.spec["observation_epochs"]:
            self.save_checkpoint(folder/f"epoch_{epoch+1:03d}_step_{self.iteration:06d}.pth")
        self.export_statistics()

    def save_checkpoint(self, path):
        path = super().save_checkpoint(path)
        cp = torch.load(path, map_location="cpu", weights_only=True)
        cp.update(stage_frequency=deepcopy(self.spec), stage_training_statistics=deepcopy(self.stage_stats))
        temporary = path.with_suffix(path.suffix+".stage.tmp")
        torch.save(cp, temporary)
        temporary.replace(path)
        return path

    def load_checkpoint(self, path, *args, **kwargs):
        cp = torch.load(path, map_location="cpu", weights_only=True)
        if cp.get("stage_frequency") != self.spec or kwargs.get("resume_mode", "full") != "full":
            raise ValueError("Only exact same-experiment full-state continuation is supported")
        super().load_checkpoint(path, *args, **kwargs)
        self.stage_stats = deepcopy(cp["stage_training_statistics"])
        assert all(self.stage_stats["optimizer_updates"][n] == self.iteration-self.amp_skipped_steps[n]
                   for n in self.stage_stats["optimizer_updates"])
        self.assert_optimizer()
        log = Path(self.config["output"]["root"])/"logs/training_metrics.jsonl"
        if log.exists():
            rows = [s for s in log.read_text().splitlines() if json.loads(s)["step"] <= self.iteration]
            log.write_text("\n".join(rows)+("\n" if rows else ""))

    def export_statistics(self):
        unseen = [n for n, c in self.stage_stats["nonzero_gradient_observations"].items() if c == 0]
        r = {**self.assert_optimizer(), "global_step": self.iteration, "statistics": self.stage_stats,
             "parameters_without_observed_effective_gradient": unseen,
             "zero_head_note": "Upstream gradients are zero before the residual output heads first update."}
        write_json(Path(self.config["output"]["root"])/"module_statistics.json", r)
        return r
