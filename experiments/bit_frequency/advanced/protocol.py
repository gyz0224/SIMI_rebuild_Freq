"""Recovered M0 protocol, new scoped provenance, and read-only baseline audits."""
from contextlib import contextmanager
from copy import deepcopy
import csv
import fcntl
import hashlib
import json
from pathlib import Path
import os
import tarfile

import torch
import yaml

from experiments.bit_frequency.common import (
    EXP, ROOT, OLD, BUDGET, COUNTS, METRICS, sha, write_json, write_csv,
    environment, native_configuration,
)

REPORT = EXP/"reports/BF06_BF07"
VARIANTS = ("BF-06", "BF-07")


def mechanism(experiment):
    if experiment not in VARIANTS:
        raise ValueError("Only independent BF-06/BF-07 experiments are supported")
    spec = {"experiment": experiment, "version": 1, "width": 32,
            "initialization_seed": 1143, "max_steps": BUDGET,
            "gradient_log_interval": 100, "observation_epochs": [10, 26, 40, 52],
            "channel_order": "R7..R0,G7..G0,B7..B0", "residual_output_init": "zero",
            "residual_rebin_or_clamp": False, "wavelet_precision": "float32"}
    if experiment == "BF-06":
        spec.update(wavelet_levels=2, conditioning="bit_only", low_frequency="learned_LL2_residual")
    else:
        spec.update(wavelet_levels=1, rgb_width=24, conditioning="RGB_affine_and_soft_spatial_gate",
                    low_frequency="learned_LL1_residual")
    return spec


def expected_config(experiment):
    cfg = deepcopy(native_configuration())
    out = EXP/experiment/"run"
    cfg["output"]["root"] = str(out)
    cfg["train"]["snapshots_folder"] = str(out/"weights")
    cfg["train"]["epochs"] = 52
    cfg["bit_frequency_mechanism"] = mechanism(experiment)
    from utils.config import normalize_training_paths
    return normalize_training_paths(cfg)


def load_config(experiment):
    expected = expected_config(experiment)
    with (EXP/experiment/"config.yaml").open() as stream:
        actual = yaml.safe_load(stream)
    if actual != expected:
        raise ValueError(f"{experiment} config differs from the frozen mechanism/M0 protocol")
    return actual


def source_manifest():
    # Deliberately scope source versions to the actual new engine and both
    # mechanisms. Old logs, figures and unrelated future code do not change it.
    files = []
    for d in ("models", "datasets", "losses", "trainers", "utils"):
        files.extend((ROOT/d).rglob("*.py"))
    files.extend((EXP/"advanced").glob("*.py"))
    files.extend((EXP/"advanced").glob("*.sh"))
    files.extend(EXP/"modules"/name for name in ("__init__.py", "haar.py", "integration.py"))
    files.extend((EXP/"__init__.py", EXP/"common.py"))
    for variant in VARIANTS:
        files.extend(EXP/variant/name for name in ("model.py", "config.yaml", "train.sh"))
    files.extend((EXP/"advanced/README.txt", EXP/"requirements.txt"))
    records = {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(files)) if p.is_file()}
    analysis_only = {"evaluate.py", "visuals.py", "summarize.py", "profile_models.py",
                     "package.py", "verify.py", "after_training.sh", "README.txt"}
    training = {p: h for p, h in records.items()
                if not (Path(p).parent == Path("experiments/bit_frequency/advanced") and Path(p).name in analysis_only)}
    digest = lambda values: hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
    return {"files": records, "training_files": training, "source_id": digest(training),
            "bundle_source_id": digest(records),
            "scope": "source_id freezes training engine/models/configs; bundle_source_id additionally versions analysis/packaging code"}


def verify_source(run):
    saved = json.loads((Path(run)/"source_manifest.json").read_text())
    if saved["source_id"] != source_manifest()["source_id"]:
        raise RuntimeError("Source or frozen configuration changed; exact resume/evaluation rejected")
    return saved


def archive_source(path, manifest):
    with tarfile.open(path, "w:gz") as archive:
        for relative in manifest["files"]:
            archive.add(ROOT/relative, arcname=relative)


def backbone_hash(model):
    h = hashlib.sha256()
    for key, tensor in model.state_dict().items():
        if ".bit_frequency." not in key:
            h.update(key.encode())
            h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@contextmanager
def gpu_lock():
    # All new training/evaluation/profile entrypoints share this single-GPU lock.
    lock = EXP/"advanced/.gpu.lock"
    with lock.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another BF06/BF07 GPU job is running; run sequentially") from exc
        stream.seek(0)
        stream.truncate()
        stream.write(f"pid={os.getpid()}\n")
        stream.flush()
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def check_preservation(write=False):
    record = json.loads((REPORT/"protected_before.json").read_text())
    changed = [p for p, h in record["files"].items() if not Path(p).is_file() or sha(p) != h]
    result = {"passed": not changed, "protected_file_count": len(record["files"]), "changed": changed}
    if write:
        write_json(REPORT/"preservation_check.json", result)
    if changed:
        raise RuntimeError(f"Protected historical artifacts changed: {changed[:5]}")
    return result


def audit(output=None, check_lpips=False):
    """Read real M0 weights/config/history and all fixed test content hashes.

    GT is only read for identity/provenance and metric-interface checks. It is
    never passed to the training loss, model, or a module design/selection step.
    """
    from PIL import __version__ as pillow_version
    from trainers.validator import EpochValidator, _paired_image_paths, _create_lpips_model
    cfg = expected_config("BF-06")
    cp = torch.load(OLD/"outputs/weights/latest.pth", map_location="cpu", weights_only=True)
    historical = cp["training_config"]
    failures = []
    current = {
        "model": {k: cfg[k] for k in ("bit_mode", "enhancement_guidance", "enhancement_backbone")},
        "mode": cfg["train"]["mode"], "loss": cfg["loss"],
        "optimizer": {k: cfg["train"][k] for k in historical["optimizer"]},
        "scheduler": cfg["scheduler"], "data": {k: cfg["data"][k] for k in historical["data"]},
        "batch_size": cfg["train"]["batch_size"], "seed": cfg["runtime"]["seed"],
        "deterministic": cfg["runtime"]["deterministic"],
        "activation_checkpointing": cfg["train"]["activation_checkpointing"],
        "amp": {"enabled": cfg["amp"]["enabled"], "dtype": cfg["amp"]["dtype"],
                "scaler": {k: cfg["amp"][k] for k in historical["amp"]["scaler"]}},
        "validation": {"enabled": cfg["validation"]["enabled"], "lpips_net": cfg["validation"]["lpips_net"],
                       "datasets": list(cfg["validation"]["datasets"])},
    }
    for key, value in current.items():
        if historical[key] != value:
            failures.append(f"M0 actual training configuration mismatch: {key}")
    if pillow_version != historical["environment"]["pillow"]:
        failures.append("Historical Pillow preprocessing version mismatch")
    if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name(0):
        failures.append("The formal experiments require an NVIDIA A100")
    for directory in ("models", "datasets", "losses", "trainers", "utils"):
        for original in (OLD/directory).glob("*.py"):
            copy = ROOT/original.relative_to(OLD)
            if not copy.is_file() or sha(copy) != sha(original):
                failures.append(f"Native M0 engine changed: {original.relative_to(OLD)}")
    root = Path(cfg["data"]["root"])
    files = sorted(root.glob("*/*.jpg"))
    entries = [(str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns) for p in files]
    fingerprint = hashlib.sha256(repr(sorted(entries)).encode()).hexdigest()
    old_data = cp["training_state"]["dataset"]
    if (len(files) != 3021 or fingerprint != old_data["fingerprint"]
            or [p[0] for p in entries] != old_data["paths"]):
        failures.append("Training paths/count/file fingerprint differs from historical M0")
    content = list(csv.DictReader((EXP/"reports/test_manifest.csv").open()))
    expected_files = {(r["dataset"], Path(r["low"]).name): r for r in content}
    if len(content) != 515 or len(expected_files) != 515:
        failures.append("Historical test content manifest is incomplete")
    signatures, examples = {}, {}
    for name, count in COUNTS.items():
        d = cfg["validation"]["datasets"][name]
        pairs = _paired_image_paths(Path(d["low_dir"]), Path(d["high_dir"]))
        obj = EpochValidator.__new__(EpochValidator)
        obj.pairs, obj.lpips_net = pairs, cfg["validation"]["lpips_net"]
        signatures[name] = obj._build_signature()
        if len(pairs) != count or signatures[name]["metrics"] != cp["selection_state"]["dataset_signature"][name]["metrics"]:
            failures.append(f"Test count/metric protocol differs: {name}")
        examples[name] = [str(low) for low, high in pairs[:2]]
        for low, high in pairs:
            reference = expected_files.get((name, low.name), {})
            if sha(low) != reference.get("low_sha256") or sha(high) != reference.get("high_sha256"):
                failures.append(f"Test content changed: {name}/{low.name}")
    base = json.loads((EXP/"evaluation/M0_historical.json").read_text())
    if sha(base["source_csv"]) != base["source_csv_sha256"]:
        failures.append("Historical M0 metric CSV changed")
    rows = list(csv.DictReader(Path(base["source_csv"]).open()))
    for name, reference in base["datasets"].items():
        path = Path(reference["checkpoint"])
        saved = torch.load(path, map_location="cpu", weights_only=True)
        subset = [r for r in rows if r["dataset"] == name]
        best = max(subset, key=lambda r: float(r["psnr"]))
        record = saved["selection_state"]["best_records"][name]["psnr"]
        if (len(subset) != 52 or sha(path) != reference["checkpoint_sha256"]
                or saved["training_state"]["global_step"] != reference["global_step"]
                or int(best["global_step"]) != reference["global_step"]
                or any(float(best[k]) != reference[k] or record[k] != reference[k] for k in METRICS)):
            failures.append(f"M0 best-PSNR reference/history/weight mismatch: {name}")
    if cp["training_state"]["global_step"] != BUDGET:
        failures.append("Historical M0 training budget not verified")
    if check_lpips:
        _create_lpips_model("alex", "cuda:0")
    report = {
        "passed": not failures, "blockers": failures, "environment": environment(),
        "historical_training_config": historical, "actual_inherited_config": current,
        "train_count": len(files), "steps_per_epoch": 378, "budget": BUDGET,
        "train_fingerprint": fingerprint, "test_counts": COUNTS, "test_total": 515,
        "test_signatures": signatures, "test_content_sha256_verified": not any("content" in f for f in failures),
        "visualization_examples_fixed_before_training": examples,
        "M0_latest_sha256": sha(OLD/"outputs/weights/latest.pth"), "M0_never_trained": True,
        "source_logic_repairs": [], "M0_macro": base["macro"],
        "known_environment_differences": {"M0_GPU": historical["environment"]["gpu"],
            "M0_CUDA": historical["environment"]["cuda"], "M0_torch": historical["torch_version"]},
        "historical_missing_fields": ["Python/torchvision/NumPy versions", "initial backbone hash", "training image content hashes"],
        "lpips_constructed": check_lpips,
    }
    if output:
        write_json(output, report)
    if failures:
        raise RuntimeError("M0 fairness audit failed: " + "; ".join(failures))
    return report


def require_full_run(experiment):
    cfg = load_config(experiment)
    run = EXP/experiment/"run"
    done_path = run/"completion.json"
    if not done_path.is_file():
        raise RuntimeError(f"{experiment}: no completed 19656-step formal run")
    done = json.loads(done_path.read_text())
    if not done.get("formal_complete") or done.get("smoke_only") or done.get("global_step") != BUDGET:
        raise RuntimeError("A complete 19656-step formal run is required")
    verify_source(run)
    latest_path = run/"weights/latest.pth"
    if sha(latest_path) != done["checkpoint_sha256"]:
        raise RuntimeError("Completed checkpoint content changed")
    latest = torch.load(latest_path, map_location="cpu", weights_only=True)
    if (latest.get("mechanism") != cfg["bit_frequency_mechanism"]
            or latest["training_state"]["global_step"] != BUDGET):
        raise RuntimeError("Checkpoint experiment/budget mismatch")
    history = latest["selection_state"]["validation_history"]
    if len(history) != 52 or [r["global_step"] for r in history] != list(range(378, BUDGET+1, 378)):
        raise RuntimeError("All 52 native validations are required")
    for row in history:
        if set(row["datasets"]) != set(COUNTS) or any(row["datasets"][n]["num_images"] != c for n, c in COUNTS.items()):
            raise RuntimeError("Incomplete fixed-six-dataset validation history")
    return cfg, run, latest
