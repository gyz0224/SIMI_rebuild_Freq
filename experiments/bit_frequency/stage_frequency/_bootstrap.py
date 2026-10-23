from pathlib import Path
import os
import sys

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP/"runtime_vendor"))
