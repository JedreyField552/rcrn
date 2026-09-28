#!/usr/bin/env bash
# Launch a training run. The package is not pip-installed in this repository, so the
# script points PYTHONPATH at its own src tree and never assumes a `python` binary.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG="${1:-configs/experiment/main.yaml}"
OUT="${2:-}"

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"

# Single process.
"${PYTHON_BIN}" -m rcrn.cli.train --config "${CONFIG}" ${OUT:+--out "${OUT}"}

# Multi-process: uncomment and set NPROC_PER_NODE, then run
#   torchrun --nproc_per_node="${NPROC_PER_NODE}" -m rcrn.cli.train --config "${CONFIG}" \
#     --set train.world_size="${NPROC_PER_NODE}"
# The trainer computes the effective batch as batch_size x grad_accum x world_size.
