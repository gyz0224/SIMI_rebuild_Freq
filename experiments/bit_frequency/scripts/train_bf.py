"""Train exactly BF-01/BF-05 for 19656 steps, or an isolated <=5-step smoke."""
import _bootstrap
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
import yaml
import torch

from experiments.bit_frequency.common import EXP, BUDGET, environment, load_config, source_manifest, archive_source, write_json, sha
from experiments.bit_frequency.training import BFTrainer
from utils.logging import tee_training_log
from preflight import audit


def state_hash(model):
    h = hashlib.sha256()
    for key, tensor in model.state_dict().items():
        if ".bit_frequency." not in key:
            h.update(key.encode())
            h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def run(args):
    torch.set_num_threads(8)
    checked = audit(check_lpips=False)
    if not checked["passed"]:
        raise RuntimeError("M0 protocol audit failed; no training started")
    for path, field in (("diagnostics/BF-D0/report.json", "passed"), ("diagnostics/BF-D1/summary.json", "complete")):
        if not json.loads((EXP/path).read_text()).get(field):
            raise RuntimeError(f"Required earlier stage not completed: {path}")
    config = load_config(args.experiment)
    smoke = args.smoke_steps is not None
    if smoke:
        if args.resume or not 1 <= args.smoke_steps <= 5:
            raise ValueError("Smoke must be 1..5 steps, no resume; formal training always uses 19656")
        out = EXP/"diagnostics/engineering_smoke"/args.experiment
        config["output"]["root"] = str(out)
        config["train"]["snapshots_folder"] = str(out/"weights")
        config["validation"]["enabled"] = False
    else:
        out = EXP/"runs"/args.experiment
        if Path(config["output"]["root"]).resolve() != out.resolve():
            raise ValueError("Formal outputs must use the experiment's isolated run directory")
        if config["bit_frequency"]["max_steps"] != BUDGET or config["train"]["epochs"] != 52:
            raise ValueError("Formal budget must be exactly 19656 batches/52 epochs")
    if not args.resume and ((out/"weights/latest.pth").exists() or (out/"logs/training_metrics.jsonl").exists()):
        raise FileExistsError(f"Existing run protected: {out}; use --resume for formal continuation")
    if args.resume and args.resume.resolve() != (out/"weights/latest.pth").resolve():
        raise ValueError("Resume only from this experiment's latest.pth")
    out.mkdir(parents=True, exist_ok=True)
    manifest = source_manifest()
    if args.resume:
        prior = json.loads((out/"source_manifest.json").read_text())
        if prior["source_id"] != manifest["source_id"]:
            raise RuntimeError("Source/config changed since training began; exact continuation rejected")
    else:
        write_json(out/"source_manifest.json", manifest)
        archive_source(out/"source_snapshot.tar.gz", manifest)
        write_json(out/"environment.json", environment())
        with (out/"resolved_config.yaml").open("w") as stream:
            yaml.safe_dump(config, stream, sort_keys=False)
    started = time.perf_counter()
    with tee_training_log(out):
        print(f"{args.experiment}: {'isolated engineering smoke' if smoke else 'formal 19656-step run'}, seed=1143", flush=True)
        trainer = BFTrainer(config)
        initial_hash = state_hash(trainer.model)
        if args.resume:
            trainer.load_checkpoint(args.resume, resume_mode="full")
        else:
            write_json(out/"initialization.json", {"backbone_initial_sha256": initial_hash,
                       "random_seed": 1143, "historical_initial_hash_proven": False,
                       "M0_loaded_for_training": False, "optimizer": trainer.assert_optimizer(),
                       "lambda_initial_B0_to_B7": trainer.bf.weights().detach().cpu().tolist()})
        trainer.fit(max_steps=args.smoke_steps if smoke else BUDGET)
        trainer.export_frequency()
        if not smoke and trainer.iteration != BUDGET:
            raise RuntimeError("Incomplete budget; formal result cannot be assessed")
        if args.experiment == "BF-05" and trainer.bf_stats["successful_finite_nonzero_gradient_steps"] == 0:
            raise RuntimeError("BF-05 frequency parameters never received an effective nonzero gradient")
        write_json(out/"completion.json", {"experiment": args.experiment, "formal_complete": not smoke,
                   "global_step": trainer.iteration, "budget": BUDGET, "smoke_only": smoke,
                   "elapsed_seconds_this_invocation": time.perf_counter()-started,
                   "checkpoint_sha256": sha(out/"weights/latest.pth"),
                   "source_id": manifest["source_id"], "amp_skipped_steps": trainer.amp_skipped_steps,
                   "metrics_evaluated": False, "scientific_conclusion": "pending full-budget best-PSNR evaluation"})
        print(f"Saved {out}; step={trainer.iteration}; M0 untouched", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("BF-01", "BF-05"), required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke-steps", type=int)
    run(parser.parse_args())
