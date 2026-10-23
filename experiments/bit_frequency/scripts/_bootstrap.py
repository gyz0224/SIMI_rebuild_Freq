"""Local entry-point paths; pin historical Pillow without modifying any env."""
from pathlib import Path
import sys

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
VENDOR = EXP / "runtime_vendor"
sys.path.insert(0, str(ROOT))
if VENDOR.is_dir():
    sys.path.insert(0, str(VENDOR))
