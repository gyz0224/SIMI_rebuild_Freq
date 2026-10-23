"""Train BF08/BF09 with native M0 settings; long runs are started by the user."""
import _bootstrap
import argparse
import json
from pathlib import Path
import time
import torch
import yaml

from experiments.bit_frequency.stage_frequency.protocol import (
    EXP, REPORT, VARIANTS, BUDGET, load_config, audit, source_manifest, archive_source,
    verify_source, backbone_hash, environment, sha, write_json, gpu_lock,
)
from experiments.bit_frequency.stage_frequency.trainer import StageTrainer
from utils.logging import tee_training_log


def run(args):
    torch.set_num_threads(8)
    config = load_config(args.experiment)
    smoke = args.smoke_steps is not None
    if smoke:
        if args.resume or not 1 <= args.smoke_steps <= 5:
            raise ValueError("Short engineering run must be 1..5 steps, with no resume")
        out = EXP/args.experiment/"engineering/native_batch"
        config["output"]["root"] = str(out)
        config["train"]["snapshots_folder"] = str(out/"weights")
        config["validation"]["enabled"] = False
    else:
        out = EXP/args.experiment/"run"
    if not args.resume and ((out/"weights/latest.pth").exists() or (out/"logs/training_metrics.jsonl").exists()):
        raise FileExistsError(f"Existing run protected: {out}; use --resume for continuation")
    if args.resume and not (out/"weights/latest.pth").is_file():
        raise FileNotFoundError("No latest.pth for this experiment")
    if args.resume and (out/"completion.json").is_file() and json.loads((out/"completion.json").read_text()).get("formal_complete"):
        raise RuntimeError("Run already completed; proceed to evaluation")
    out.mkdir(parents=True, exist_ok=True)
    audit(output=out/"preflight.json", check_lpips=not smoke)
    manifest = source_manifest()
    if not smoke:
        verified = json.loads((REPORT/"verification.json").read_text())
        if not verified.get("passed") or verified["source_id"] != manifest["source_id"]:
            raise RuntimeError("Current model/training code must pass engineering verification first")
    if args.resume:
        verify_source(out)
        if json.loads((out/"config.json").read_text()) != config:
            raise RuntimeError("Resolved configuration changed")
    else:
        write_json(out/"source_manifest.json", manifest)
        archive_source(out/"source_snapshot.tar.gz", manifest)
        write_json(out/"config.json", config)
        (out/"resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        write_json(out/"environment.json", environment())
        write_json(out/"M0_reference.json", json.loads((EXP/"evaluation/M0_historical.json").read_text()))
    started = time.perf_counter()
    with tee_training_log(out):
        print(f"{args.experiment}: {'ENGINEERING ONLY' if smoke else 'FORMAL 19656 native joint iterations'}; seed1143, batch8, 512, native AMP FP16", flush=True)
        trainer = StageTrainer(config)
        trainer.log_every_step = smoke
        initial = backbone_hash(trainer.model)
        expected = json.loads((EXP/"runs/BF-05/initialization.json").read_text())["backbone_initial_sha256"]
        if initial != expected:
            raise RuntimeError("Original SIMI initialization differs from verified seed1143 runs")
        if args.resume:
            trainer.load_checkpoint(out/"weights/latest.pth", resume_mode="full")
        else:
            write_json(out/"initialization.json", {"backbone_initial_sha256": initial,
                "historical_M0_initial_hash_available": False, "M0_or_candidate_weights_loaded": False,
                "new_module_randomness_isolated": True, "optimizer": trainer.assert_optimizer()})
            trainer.save_checkpoint(out/"weights/latest.pth")
        torch.cuda.reset_peak_memory_stats()
        trainer.fit(max_steps=args.smoke_steps if smoke else BUDGET)
        stats = trainer.export_statistics()
        if not smoke and (trainer.iteration != BUDGET or stats["parameters_without_observed_effective_gradient"]):
            raise RuntimeError("Incomplete budget or unseen effective gradients; no formal completion declaration")
        counts = trainer.stage_stats["optimizer_updates"]
        assert all(counts[n] == trainer.iteration-trainer.amp_skipped_steps[n] for n in counts)
        write_json(out/"completion.json", {"experiment": args.experiment, "formal_complete": not smoke,
            "smoke_only": smoke, "global_step": trainer.iteration, "joint_iteration_budget": BUDGET,
            "actual_successful_optimizer_updates": counts, "amp_skipped_steps": trainer.amp_skipped_steps,
            "budget_definition": config["stage_frequency"]["budget_definition"],
            "elapsed_seconds_this_invocation": time.perf_counter()-started,
            "timing_state": trainer.timing_state, "source_id": manifest["source_id"],
            "checkpoint_sha256": sha(out/"weights/latest.pth"),
            "peak_allocated_bytes_this_invocation": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes_this_invocation": torch.cuda.max_memory_reserved(),
            "quality_metrics_pending": True})
        print(f"Saved {out}; iterations={trainer.iteration}; successful optimizer updates={counts}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", choices=VARIANTS, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke-steps", type=int)
    args = p.parse_args()
    with gpu_lock():
        run(args)
