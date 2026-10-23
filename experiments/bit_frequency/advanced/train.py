"""Train only BF-06/BF-07; formal runs always finish all 19656 steps."""
import _bootstrap
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import torch
import yaml

from experiments.bit_frequency.advanced.protocol import (
    EXP, REPORT, VARIANTS, BUDGET, load_config, audit, source_manifest, archive_source,
    environment, write_json, sha, verify_source, backbone_hash, gpu_lock,
)
from experiments.bit_frequency.advanced.trainer import MechanismTrainer
from utils.logging import tee_training_log


def run(args):
    torch.set_num_threads(8)
    cfg = load_config(args.experiment)
    smoke = args.smoke_steps is not None
    if smoke:
        if args.resume or not 1 <= args.smoke_steps <= 5:
            raise ValueError("Engineering smoke: 1..5 steps, never a formal result")
        out = EXP/args.experiment/"engineering/native_batch_smoke"
        cfg["output"]["root"] = str(out)
        cfg["train"]["snapshots_folder"] = str(out/"weights")
        cfg["validation"]["enabled"] = False
    else:
        out = EXP/args.experiment/"run"
    out.mkdir(parents=True, exist_ok=True)
    if not args.resume and ((out/"weights/latest.pth").exists() or (out/"logs/training_metrics.jsonl").exists()):
        raise FileExistsError(f"Existing run protected: {out}; use --resume for formal continuation")
    if args.resume and not (out/"weights/latest.pth").is_file():
        raise FileNotFoundError(f"No own-experiment latest.pth: {out}")
    if args.resume and (out/"completion.json").is_file():
        done = json.loads((out/"completion.json").read_text())
        if done.get("formal_complete"):
            raise RuntimeError("Formal run already completed; proceed to evaluation")
    checked = audit(output=out/"preflight.json", check_lpips=not smoke)
    if not smoke:
        verification_path = REPORT/"engineering_verification.json"
        if not verification_path.is_file():
            raise RuntimeError("Run advanced/verify.py and native-batch smokes before formal training")
        verified = json.loads(verification_path.read_text())
        if not verified.get("passed") or verified["source_id"] != source_manifest()["source_id"]:
            raise RuntimeError("Engineering verification is missing/outdated for the frozen source")
    manifest = source_manifest()
    if args.resume:
        verify_source(out)
        if yaml.safe_load((out/"resolved_config.yaml").read_text()) != cfg:
            raise RuntimeError("Resolved training configuration changed")
    else:
        write_json(out/"source_manifest.json", manifest)
        archive_source(out/"source_snapshot.tar.gz", manifest)
        write_json(out/"environment.json", environment())
        (out/"resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
        write_json(out/"config.json", cfg)
    started = time.perf_counter()
    with tee_training_log(out):
        print(f"{args.experiment}: {'engineering only' if smoke else 'FORMAL'}; seed=1143, batch=8, image=512, AMP=FP16", flush=True)
        trainer = MechanismTrainer(cfg)
        trainer.log_every_step = smoke
        initial = backbone_hash(trainer.model)
        previous = json.loads((EXP/"runs/BF-05/initialization.json").read_text())
        if initial != previous["backbone_initial_sha256"]:
            raise RuntimeError("Native backbone initialization differs from verified seed-1143 BF runs")
        if args.resume:
            trainer.load_checkpoint(out/"weights/latest.pth", resume_mode="full")
        else:
            write_json(out/"initialization.json", {
                "backbone_initial_sha256": initial, "same_as_BF01_BF05": True,
                "historical_M0_initial_hash_available": False, "M0_loaded_for_training": False,
                "new_module_rng_isolated": True, "optimizer": trainer.assert_optimizer(),
                "residual_output_init": "all output weights and biases zero; identity verified separately",
            })
            # A first-epoch interruption can resume even before epoch 1 ends.
            trainer.save_checkpoint(out/"weights/latest.pth")
            write_json(out/"M0_reference.json", json.loads((EXP/"evaluation/M0_historical.json").read_text()))
        torch.cuda.reset_peak_memory_stats()
        trainer.fit(max_steps=args.smoke_steps if smoke else BUDGET)
        stats = trainer.export_statistics()
        if not smoke and trainer.iteration != BUDGET:
            raise RuntimeError("Incomplete budget; scientific comparison is not permitted")
        if not smoke and stats["parameters_without_observed_effective_gradient"]:
            raise RuntimeError("Some module parameters never received observed effective gradients")
        timing = trainer.timing_state
        write_json(out/"completion.json", {
            "experiment": args.experiment, "formal_complete": not smoke, "smoke_only": smoke,
            "global_step": trainer.iteration, "budget": BUDGET,
            "elapsed_seconds_this_invocation": time.perf_counter()-started,
            "checkpoint_sha256": sha(out/"weights/latest.pth"), "source_id": manifest["source_id"],
            "amp_skipped_steps": trainer.amp_skipped_steps, "timing_state": timing,
            "peak_cuda_allocated_bytes_this_invocation": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes_this_invocation": torch.cuda.max_memory_reserved(),
            "metrics_evaluated": False, "scientific_conclusion": "pending full-budget best-PSNR evaluation",
        })
        print(f"Saved {out}; step={trainer.iteration}; all historical outputs untouched", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", choices=VARIANTS, required=True)
    p.add_argument("--resume", action="store_true", help="Exact continuation from this experiment's latest.pth")
    p.add_argument("--smoke-steps", type=int)
    with gpu_lock():
        run(p.parse_args())
