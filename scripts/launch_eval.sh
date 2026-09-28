#!/usr/bin/env bash
# Launch the evaluation, the certification and the transport report for one configuration.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG="${1:-configs/experiment/main.yaml}"
OUT="${2:-${ROOT}/runs/evaluation}"

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"

"${PYTHON_BIN}" -m rcrn.cli.evaluate --config "${CONFIG}" --out "${OUT}"
"${PYTHON_BIN}" -m rcrn.cli.certify --config "${CONFIG}" --out "${OUT}/certification"
"${PYTHON_BIN}" -m rcrn.cli.transport --config "${CONFIG}" --out "${OUT}/transport"
