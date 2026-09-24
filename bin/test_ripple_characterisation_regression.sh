#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORK_DIR="${WORK_DIR:-$HOME/DATA/gmrt_40_014/work}"
PYTHON_CMD="${PYTHON_CMD:-python}"

FITS="$WORK_DIR/split/3c468.1/3c468.1_primary_secondary_calibrated_flagged.uvfits"
INDEX="$FITS.row_index_cache.npz"
SOURCE=3C468.1

RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
TEST_ROOT="${TEST_ROOT:-$REPO_ROOT/work/ripple_characterisation_regression_${RUN_TS}}"
REPORT_FILE="$TEST_ROOT/regression_report.txt"

[[ -f "$FITS" ]] || { echo "Real 3C468.1 FITS not found: $FITS" >&2; exit 2; }
[[ -f "$INDEX" ]] || { echo "Real 3C468.1 row-index cache not found: $INDEX" >&2; exit 2; }

mkdir -p "$TEST_ROOT"

run_mode() {
  local mode="$1"
  local outdir="$TEST_ROOT/$mode"
  mkdir -p "$outdir"
  set +e
  "$PYTHON_CMD" "$REPO_ROOT/src/characterise_ripple.py" \
    --fits "$FITS" \
    --index-cache "$INDEX" \
    --source "$SOURCE" \
    --chan-range 0 127 \
    --elevation-min 25 \
    --physical-model-mode "$mode" \
    --outfile-prefix "run_$mode" \
    --outdir "$outdir" \
    --provenance-dir "$TEST_ROOT/provenance" \
    > "$TEST_ROOT/run_${mode}.log" 2>&1
  local status=$?
  set -e
  return $status
}

RUN_OK=PASS
run_mode fit || RUN_OK=FAIL
run_mode known || RUN_OK=FAIL

OUTPUTS_PRESENT=PASS
JSON_VALID=PASS
KNOWN_MODEL_CROSSCHECK_PRESENT=PASS
for mode in fit known; do
  outdir="$TEST_ROOT/$mode"
  png="$outdir/run_${mode}.png"
  json="$outdir/run_${mode}_summary.json"
  csv="$outdir/run_${mode}_components.csv"
  md="$outdir/run_${mode}_engineer_report.md"
  for f in "$png" "$json" "$csv" "$md"; do
    if [[ ! -s "$f" ]]; then
      echo "[ripple-regression] missing/empty output: $f" >&2
      OUTPUTS_PRESENT=FAIL
    fi
  done

  if [[ -f "$json" ]]; then
    if ! "$PYTHON_CMD" - "$json" <<'PY'
import json
import sys

path = sys.argv[1]


def _reject_nonfinite(token):
    raise ValueError(f"raw non-finite JSON token found: {token!r}")


with open(path) as f:
    data = json.load(f, parse_constant=_reject_nonfinite)

assert data.get('schema') == 'gmrt-ripple-characterisation-v1'
PY
    then
      echo "[ripple-regression] invalid/non-finite-token JSON: $json" >&2
      JSON_VALID=FAIL
    fi

    crosscheck_present="$("$PYTHON_CMD" - "$json" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    data = json.load(f)
print('yes' if data.get('known_model_local_alpha_crosscheck') is not None else 'no')
PY
)"
    if [[ "$crosscheck_present" != "yes" ]]; then
      echo "[ripple-regression] known_model_local_alpha_crosscheck missing/null in: $json" >&2
      KNOWN_MODEL_CROSSCHECK_PRESENT=FAIL
    fi
  else
    JSON_VALID=FAIL
    KNOWN_MODEL_CROSSCHECK_PRESENT=FAIL
  fi
done

{
  echo "RUN_TS=$RUN_TS"
  echo "FITS=$FITS"
  echo "INDEX=$INDEX"
  echo "SOURCE=$SOURCE"
  echo "TEST_ROOT=$TEST_ROOT"
  echo "RUN_OK=$RUN_OK"
  echo "OUTPUTS_PRESENT=$OUTPUTS_PRESENT"
  echo "JSON_VALID=$JSON_VALID"
  echo "KNOWN_MODEL_CROSSCHECK_PRESENT=$KNOWN_MODEL_CROSSCHECK_PRESENT"
} > "$REPORT_FILE"

cat "$REPORT_FILE"

if [[ "$RUN_OK" != "PASS" || "$OUTPUTS_PRESENT" != "PASS" || "$JSON_VALID" != "PASS" || "$KNOWN_MODEL_CROSSCHECK_PRESENT" != "PASS" ]]; then
  echo "[ripple-regression] one or more markers FAILed; see $REPORT_FILE and $TEST_ROOT/run_*.log" >&2
  exit 1
fi

echo "[ripple-regression] PASS: $REPORT_FILE"
