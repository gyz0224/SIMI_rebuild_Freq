"""Protocol, provenance and small artifact helpers for this branch only."""
from copy import deepcopy
import csv
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

import torch
import yaml

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
OLD = ROOT.parent / "SIMI_rebuild"
COUNTS = {"LOLv1": 15, "LOLv2real": 100, "LOLv2syn": 100,
          "LSRWhuawei": 30, "LSRWNikon": 20, "SICE": 250}
METRICS = ("psnr", "ssim", "lpips")
BUDGET = 19656


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def write_csv(path, rows, fields=None):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def environment():
    names = ["torch", "torchvision", "numpy", "Pillow", "pytorch-msssim", "lpips", "matplotlib"]
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "packages": versions, "torch_build": str(torch.__version__),
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
            "threads": torch.get_num_threads(), "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32}


def source_manifest():
    files = [p for directory in ("models", "datasets", "trainers", "losses", "utils", "experiments/bit_frequency")
             for p in (ROOT / directory).rglob("*.py")
             if not {"runtime_vendor", "__pycache__"}.intersection(p.parts)]
    files += list((EXP / "configs").glob("*.yaml"))
    files += list((EXP / "scripts").glob("*.sh"))
    files += [p for p in (EXP/"README.md", EXP/"requirements.txt") if p.exists()]
    records = {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)}
    return {"files": records, "source_id": hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()}


def archive_source(path, manifest):
    import tarfile
    with tarfile.open(path, "w:gz") as archive:
        for relative in manifest["files"]:
            archive.add(ROOT/relative, arcname=relative)


def load_config(experiment):
    if experiment not in ("BF-01", "BF-05"):
        raise ValueError("No M0 training or other experiments supported")
    with (EXP / "configs" / f"{experiment}.yaml").open() as stream:
        config = yaml.safe_load(stream)
    from utils.config import normalize_training_paths
    config = normalize_training_paths(config)
    expected = normalize_training_paths(native_configuration())
    out = EXP/"runs"/experiment
    expected["output"]["root"] = str(out)
    expected["train"]["snapshots_folder"] = str(out/"weights")
    expected["train"]["epochs"] = 52
    expected["bit_frequency"] = bf_spec(experiment)
    if config != expected:
        raise ValueError("Candidate YAML differs from the recovered M0 protocol plus the prescribed BF change")
    return config


def bf_spec(experiment):
    if experiment not in ("BF-01", "BF-05"):
        raise ValueError("Only BF-01/BF-05 are authorized")
    return {"experiment": experiment, "max_steps": BUDGET,
            "weight_log_interval": 100, "observation_epochs": [10, 26, 40, 52],
            "fixed_low_bits": [0, 1, 2], "fixed_high_frequency_scale": 0.75,
            "LL": "identity", "parameterization": "1+tanh(theta)", "theta_init": 0.0,
            "theta_rows": "B0..B7", "directions": ["LH", "HL", "HH"], "RGB": "shared"}


def native_configuration():
    """Use local verified M0 YAML, remap dataset root only; audit against checkpoint."""
    with (OLD / "configs/default.yaml").open() as stream:
        config = yaml.safe_load(stream)
    config = deepcopy(config)
    data_root = ROOT.parents[1] / "model/dataset"
    for values in [config["data"], *config["validation"]["datasets"].values()]:
        for name in ("root", "low_dir", "high_dir"):
            if name in values:
                relative = values[name].split("model/dataset/", 1)[1]
                values[name] = str(data_root / relative)
    return config


def heatmap(matrix, columns, title, output, *, rows=None, fmt=".3f", cmap="viridis"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(columns)), columns)
    ax.set_yticks(range(len(matrix)), rows or [f"B{k}" for k in range(7, -1, -1)])
    ax.set_title(title)
    for i in range(len(matrix)):
        for j in range(len(columns)):
            ax.text(j, i, format(float(matrix[i][j]), fmt), ha="center", va="center", fontsize=8,
                    color="white" if float(matrix[i][j]) < (im.get_clim()[0]+im.get_clim()[1])/2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
