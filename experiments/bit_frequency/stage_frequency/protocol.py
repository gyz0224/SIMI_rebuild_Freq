"""Independent stage experiments with the actual historical M0 protocol."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tarfile
import torch

from experiments.bit_frequency.common import (
    EXP, ROOT, OLD, BUDGET, COUNTS, METRICS, sha, write_json, write_csv,
    environment, native_configuration,
)
from experiments.bit_frequency.advanced.protocol import audit as baseline_audit, gpu_lock

VARIANTS = ("BF-08", "BF-09")
REPORT = EXP/"reports/BF08_BF09"


def backbone_hash(model):
    """Hash only original SIMI tensors, excluding this branch's new module."""
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        if ".stage_frequency." in name:
            continue
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def specification(v):
    if v not in VARIANTS:
        raise ValueError("Only independent BF08/BF09 runs; no M0/previous-variant training")
    s = {"experiment": v, "version": 1, "initialization_seed": 1143,
        "max_steps": BUDGET, "log_interval": 100, "observation_epochs": [19, 29, 40, 52],
        "budget_definition": "historical_joint_batch_iterations; successful_optimizer_updates_logged_separately",
        "native_binary_order": "R7..R0,G7..G0,B7..B0", "new_module_precision": "float32",
        "residual_head_initialization": "zero", "GT_used_in_training_or_module_selection": False}
    if v == "BF-08":
        s.update(width=16, amplitude_bound=.1, logit_bound=.1, fft_norm="ortho",
            placement="native_seven_curve_heads_before_activation", optimizer_branch="light")
    else:
        s.update(width=24, high_frequency_bound=.1, LL="unchanged",
            placement="after_native_detach_before_native_denoising", optimizer_branch="denoise")
    return s


def expected_config(v):
    c = deepcopy(native_configuration())
    out = EXP/v/"run"
    c["output"]["root"] = str(out)
    c["train"]["snapshots_folder"] = str(out/"weights")
    c["train"]["epochs"] = 52
    c["stage_frequency"] = specification(v)
    from utils.config import normalize_training_paths
    return normalize_training_paths(c)


def load_config(v):
    expected = expected_config(v)
    actual = json.loads((EXP/v/"config.json").read_text())
    if actual != expected:
        raise ValueError("Config differs from recovered M0 plus this frozen stage mechanism")
    return actual


def source_manifest():
    files = []
    for d in ("models", "datasets", "losses", "trainers", "utils"):
        files.extend((ROOT/d).rglob("*.py"))
    files.extend((EXP/"stage_frequency").glob("*.py"))
    files.extend((EXP/"stage_frequency").glob("*.sh"))
    files.extend([EXP/"stage_frequency/README.txt", EXP/"common.py", EXP/"__init__.py",
        EXP/"modules/haar.py", EXP/"modules/__init__.py", EXP/"modules/integration.py",
        EXP/"advanced/__init__.py", EXP/"advanced/layers.py", EXP/"advanced/protocol.py"])
    for v in VARIANTS:
        files.extend(EXP/v/n for n in ("model.py", "config.json", "train.sh"))
    all_files = {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(files)) if p.is_file()}
    analysis = {"evaluate.py", "results.py", "verify.py", "after_training.sh", "README.txt"}
    training = {p: h for p, h in all_files.items()
        if not (Path(p).parent == Path("experiments/bit_frequency/stage_frequency") and Path(p).name in analysis)}
    digest = lambda x: hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()
    return {"source_id": digest(training), "bundle_source_id": digest(all_files),
        "files": all_files, "training_files": training,
        "scope": "new independent mechanisms, native M0, and explicitly used numerical/audit helpers"}


def archive_source(path, manifest):
    with tarfile.open(path, "w:gz") as a:
        for p in manifest["files"]:
            a.add(ROOT/p, arcname=p)


def verify_source(out):
    old = json.loads((Path(out)/"source_manifest.json").read_text())
    if old["source_id"] != source_manifest()["source_id"]:
        raise RuntimeError("Actual training source/config changed; exact continuation rejected")
    return old


def audit(output=None, check_lpips=False):
    r = baseline_audit(check_lpips=check_lpips)
    m0 = torch.load(OLD/"outputs/weights/latest.pth", map_location="cpu", weights_only=True)
    baseline = json.loads((EXP/"evaluation/M0_historical.json").read_text())
    assert abs(baseline["macro"]["psnr"]-18.813698429107667) < 1e-12
    assert abs(baseline["macro"]["ssim"]-.6701409923450815) < 1e-12
    assert abs(baseline["macro"]["lpips"]-.29649439226008123) < 1e-12
    r.update(historical_M0_optimizer_updates={n: BUDGET-m0["amp_skipped_steps"][n] for n in ("light", "denoise")},
        budget_definition=specification("BF-08")["budget_definition"],
        native_interfaces={"planes": "24 RGB-major MSB-first, unchanged",
            "smoothing": "two ConvSiLU24, then concatRGB+CBAM27+three fusion convolutions",
            "curves": "seven native DEA stages; A=tanh(logit), B=.5+.5*sigmoid(logit)",
            "enhancement_loss": "native exposure+curve smoothness+local/global color",
            "denoise": "native detached enhancement+detached 32-channel features",
            "denoising_loss": "native SSIM+gradient+smoothness; target=original enhanced.detach()"},
        BF09_adaptation="Bit-guided repair is inside the denoising branch after both native detach operations. Input features detached; original target and losses unchanged.")
    if output:
        write_json(output, r)
    return r


def preservation(write=True):
    before = json.loads((REPORT/"protected_before.json").read_text())["files"]
    changed = [p for p, h in before.items() if not Path(p).is_file() or sha(p) != h]
    r = {"passed": not changed, "protected_files": len(before), "changed": changed}
    if write:
        write_json(REPORT/"preservation_check.json", r)
    if changed:
        raise RuntimeError(f"Historical sources/results changed: {changed[:5]}")
    return r


def require_full_run(v):
    cfg = load_config(v)
    out = EXP/v/"run"
    if not (out/"completion.json").is_file():
        raise RuntimeError("A complete 19656-step run is required")
    done = json.loads((out/"completion.json").read_text())
    if not done.get("formal_complete") or done.get("smoke_only") or done.get("global_step") != BUDGET:
        raise RuntimeError("Incomplete or engineering-only run; no formal candidate metrics")
    verify_source(out)
    if sha(out/"weights/latest.pth") != done["checkpoint_sha256"]:
        raise RuntimeError("Completed weights changed")
    cp = torch.load(out/"weights/latest.pth", map_location="cpu", weights_only=True)
    h = cp["selection_state"]["validation_history"]
    assert cp["stage_frequency"] == cfg["stage_frequency"]
    assert cp["training_state"]["global_step"] == BUDGET
    assert len(h) == 52 and [r["global_step"] for r in h] == list(range(378, BUDGET+1, 378))
    assert all(r["datasets"][n]["num_images"] == c for r in h for n, c in COUNTS.items())
    counts = cp["stage_training_statistics"]["optimizer_updates"]
    assert all(counts[n] == BUDGET-cp["amp_skipped_steps"][n] for n in counts)
    return cfg, out, cp
