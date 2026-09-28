#!/usr/bin/env bash
# Prepare the data roots the pipeline reads. Nothing is downloaded here: the private
# cohort and the public releases are obtained under their own terms, and this script
# only checks that the expected files are in place and prints what is missing.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "${DATA_ROOT}/cohort" "${DATA_ROOT}/assay" "${DATA_ROOT}/candidates"

check() {
  if [ -f "$1" ]; then
    echo "present  $1"
  else
    echo "missing  $1"
  fi
}

check "${DATA_ROOT}/cohort/derived_records.csv"
check "${DATA_ROOT}/cohort/implied_order.csv"
check "${DATA_ROOT}/candidates/candidate_list.csv"
check "${DATA_ROOT}/assay/permeability.csv"

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_BIN}" - <<'PY'
from rcrn.data.molecules import ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from rcrn.data.schemas import ACQUISITION_CHANNELS

print(f"atom feature width     : {ATOM_FEATURE_DIM}")
print(f"bond feature width     : {BOND_FEATURE_DIM}")
print(f"contrast channel order : {','.join(ACQUISITION_CHANNELS)}")
print("the schema-compatible cohort fixture needs no file and is built from configuration")
PY
