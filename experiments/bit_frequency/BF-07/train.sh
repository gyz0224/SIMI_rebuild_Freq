#!/usr/bin/env bash
set -euo pipefail
bf_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$bf_repo"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8
exec /home/u104754253089/.conda/envs/ICLR/bin/python -B -u experiments/bit_frequency/advanced/train.py --experiment BF-07 "$@"
