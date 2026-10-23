"""Read-only baseline audit, exact configuration inheritance and data checks."""
import _bootstrap
import argparse
import csv
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

import torch
import yaml
from PIL import Image, __version__ as pillow_version

from experiments.bit_frequency.common import (EXP, ROOT, OLD, COUNTS, BUDGET, METRICS,
    environment, native_configuration, bf_spec, sha, write_json, write_csv)
from trainers.validator import EpochValidator, _paired_image_paths, _create_lpips_model


def audit(write_configs=False, check_lpips=False):
    config = native_configuration()
    checkpoint = torch.load(OLD / "outputs/weights/latest.pth", map_location="cpu", weights_only=False)
    historical = checkpoint["training_config"]
    blockers, notes = [], []
    comparisons = {
        "model": {k: config[k] for k in ("bit_mode", "enhancement_guidance", "enhancement_backbone")},
        "mode": config["train"]["mode"], "loss": config["loss"],
        "optimizer": {k: config["train"][k] for k in historical["optimizer"]},
        "scheduler": config["scheduler"], "data": {k: config["data"][k] for k in historical["data"]},
        "batch_size": config["train"]["batch_size"], "activation_checkpointing": config["train"]["activation_checkpointing"],
        "seed": config["runtime"]["seed"], "deterministic": config["runtime"]["deterministic"],
        "amp": {"enabled": config["amp"]["enabled"], "dtype": config["amp"]["dtype"],
                "scaler": {k: config["amp"][k] for k in historical["amp"]["scaler"]}},
        "validation": {"enabled": config["validation"]["enabled"], "lpips_net": config["validation"]["lpips_net"],
                       "datasets": list(config["validation"]["datasets"])},
    }
    for key, value in comparisons.items():
        if value != historical[key]:
            blockers.append(f"M0 configuration mismatch: {key}")
    if pillow_version != historical["environment"]["pillow"]:
        blockers.append(f"Pillow preprocessing mismatch: {pillow_version} vs {historical['environment']['pillow']}")
    if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name(0):
        blockers.append("An NVIDIA A100 is required for formal training")
    original_sources = {}
    for directory in ("models", "datasets", "losses", "trainers", "utils"):
        for old in (OLD / directory).glob("*.py"):
            local = ROOT / old.relative_to(OLD)
            original_sources[str(old.relative_to(OLD))] = sha(old)
            if not local.exists() or sha(local) != sha(old):
                blockers.append(f"Native source differs from verified M0: {old.relative_to(OLD)}")
    train_root = Path(config["data"]["root"])
    train_files = sorted(train_root.glob("*/*.jpg"))
    train_records = [(str(p.relative_to(train_root)), p.stat().st_size, p.stat().st_mtime_ns) for p in train_files]
    fingerprint = hashlib.sha256(repr(sorted(train_records)).encode()).hexdigest()
    saved_data = checkpoint["training_state"]["dataset"]
    if fingerprint != saved_data["fingerprint"] or [str(p.relative_to(train_root)) for p in train_files] != saved_data["paths"]:
        blockers.append("Training file list/size/mtime fingerprint differs from historical M0")
    steps_per_epoch = (len(train_files) + config["train"]["batch_size"] - 1) // config["train"]["batch_size"]
    if len(train_files) != 3021 or steps_per_epoch != 378 or checkpoint["training_state"]["global_step"] != BUDGET:
        blockers.append("Historical budget/train count not matched")
    dataset_rows, timestamp_differences = [], []
    # These are provenance records only; no SNR/Gray model code is imported.
    prior_manifest = ROOT.parent / "SIMI_rebuild_SNR/analysis_outputs/bit_interaction_phase4/phase4_dataset_manifest.csv"
    historical_files = {(r["dataset"], r["image"]): r for r in csv.DictReader(prior_manifest.open())}
    for name, count in COUNTS.items():
        dirs = config["validation"]["datasets"][name]
        pairs = _paired_image_paths(Path(dirs["low_dir"]), Path(dirs["high_dir"]))
        # Construct the signature without loading an evaluator or running inference.
        obj = EpochValidator.__new__(EpochValidator)
        obj.pairs, obj.lpips_net = pairs, config["validation"]["lpips_net"]
        signature = obj._build_signature()
        reference = checkpoint["selection_state"]["dataset_signature"][name]
        if signature["metrics"] != reference["metrics"]:
            blockers.append(f"Test metric signature differs from historical M0: {name}")
        if signature["pairs_fingerprint"] != reference["pairs_fingerprint"]:
            timestamp_differences.append(name)
        if len(pairs) != count:
            blockers.append(f"Incorrect test image count: {name}")
        for low, high in pairs:
            with Image.open(low) as image:
                width, height = image.size
            low_hash, high_hash = sha(low), sha(high)
            reference_file = historical_files.get((name, low.name))
            content_matches = reference_file is not None and low_hash == reference_file["input_sha256"] and high_hash == reference_file["gt_sha256"]
            if not content_matches:
                blockers.append(f"Historical test image content mismatch: {name}/{low.name}")
            dataset_rows.append({"dataset": name, "low": str(low), "high": str(high),
                                 "width": width, "height": height, "low_sha256": low_hash, "high_sha256": high_hash,
                                 "historical_content_hashes_match": content_matches})
    if len(historical_files) != len(dataset_rows):
        blockers.append("Historical test manifest image set/count differs")
    if check_lpips:
        _create_lpips_model("alex", torch.device("cuda:0"))
    history = list(csv.DictReader((OLD / "outputs/logs/validation_metrics.csv").open()))
    baseline = {}
    for name, count in COUNTS.items():
        rows = [r for r in history if r["dataset"] == name and int(r["global_step"]) <= BUDGET]
        if len(rows) != 52 or any(int(r["num_images"]) != count for r in rows):
            blockers.append(f"Incomplete M0 epoch history: {name}")
        best = max(rows, key=lambda r: float(r["psnr"]))
        payload = torch.load(OLD / f"outputs/weights/{name}/best_psnr.pth", map_location="cpu", weights_only=False)
        record = payload["selection_state"]["best_records"][name]["psnr"]
        if payload["training_state"]["global_step"] != int(best["global_step"]) or any(abs(record[k] - float(best[k])) > 1e-10 for k in METRICS):
            blockers.append(f"M0 best-PSNR checkpoint and CSV disagree: {name}")
        baseline[name] = {k: float(best[k]) for k in METRICS}
        baseline[name].update(num_images=count, epoch=int(best["epoch"]), global_step=int(best["global_step"]),
                              checkpoint=str(OLD / f"outputs/weights/{name}/best_psnr.pth"),
                              checkpoint_sha256=sha(OLD / f"outputs/weights/{name}/best_psnr.pth"))
    baseline_summary = {"experiment": "M0", "origin": "historical_server_only", "total_training_steps": BUDGET,
                        "selection": "best_psnr_per_dataset; all metrics from that checkpoint",
                        "datasets": baseline,
                        "macro": {k: sum(r[k] for r in baseline.values()) / 6 for k in METRICS},
                        "image_weighted": {k: sum(r[k]*r["num_images"] for r in baseline.values()) / 515 for k in METRICS},
                        "source_csv": str(OLD / "outputs/logs/validation_metrics.csv"),
                        "source_csv_sha256": sha(OLD / "outputs/logs/validation_metrics.csv")}
    write_json(EXP / "evaluation/M0_historical.json", baseline_summary)
    write_csv(EXP / "evaluation/M0_historical.csv", [{"dataset": n, **r} for n, r in baseline.items()])
    write_csv(EXP / "reports/test_manifest.csv", dataset_rows)
    write_csv(EXP / "reports/train_manifest.csv", [{"relative_path": n, "bytes": s, "mtime_ns": t} for n, s, t in train_records])
    for key in ("gpu", "cuda", "torch_version", "pillow"):
        current = {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                   "cuda": torch.version.cuda, "torch_version": str(torch.__version__), "pillow": pillow_version}[key]
        oldvalue = historical.get(key, historical["environment"].get(key))
        if current != oldvalue:
            notes.append({"field": key, "historical": oldvalue, "current": current})
    report = {"passed": not blockers, "blockers": blockers, "environment_changes": notes,
              "environment": environment(), "historical_training_config": historical,
              "training_count": len(train_files), "steps_per_epoch": steps_per_epoch, "budget": BUDGET,
              "train_fingerprint": fingerprint, "historical_train_fingerprint": saved_data["fingerprint"],
              "test_counts": COUNTS, "test_total": len(dataset_rows), "test_metric_signatures_match": not any("metric signature" in b for b in blockers),
              "test_metadata_fingerprint_differences": timestamp_differences,
              "test_content_hashes_match": all(r["historical_content_hashes_match"] for r in dataset_rows),
              "historical_test_manifest": str(prior_manifest), "historical_test_manifest_sha256": sha(prior_manifest),
              "source_hashes": original_sources, "native_source_unchanged": not any("source" in b for b in blockers),
              "M0_latest_sha256": sha(OLD / "outputs/weights/latest.pth"),
              "M0_never_trained": True, "baseline_logic_repairs": [],
              "missing_historical_fields": ["Python/torchvision/NumPy versions and initial model hash not stored in M0 checkpoint"],
              "content_provenance": "Training list/size/mtime match the M0 checkpoint. All test input/GT SHA256 values checked against the old-server Phase4 manifest. Test attribute fingerprint differences are retained, never rewritten. No historical train content hashes are available.",
              "lpips_constructed": check_lpips,
              "nvidia_smi": subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout}
    write_json(EXP / "reports/environment_audit.json", report)
    if write_configs:
        for experiment in ("BF-01", "BF-05"):
            candidate = deepcopy(config)
            out = EXP / "runs" / experiment
            candidate["output"]["root"] = str(out)
            candidate["train"]["snapshots_folder"] = str(out / "weights")
            # Administrative stopping budget; no schedule/batch/sampling changes.
            candidate["train"]["epochs"] = BUDGET // steps_per_epoch
            candidate["bit_frequency"] = bf_spec(experiment)
            with (EXP / "configs" / f"{experiment}.yaml").open("w") as stream:
                yaml.safe_dump(candidate, stream, sort_keys=False, allow_unicode=True)
    print(json.dumps({"passed": report["passed"], "blockers": blockers, "train": len(train_files),
                      "test": len(dataset_rows), "M0_macro": baseline_summary["macro"], "environment_changes": notes}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-configs", action="store_true")
    parser.add_argument("--check-lpips", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    result = audit(args.write_configs, args.check_lpips)
    if not result["passed"]:
        raise SystemExit("Formal experiments blocked; see reports/environment_audit.json")
