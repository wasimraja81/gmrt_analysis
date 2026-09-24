#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORK_DIR="${WORK_DIR:-$HOME/DATA/gmrt_40_014/work}"
STRICT_DATA="${STRICT_DATA:-1}"

FITS="$WORK_DIR/split/3c468.1/3c468.1_primary_secondary_calibrated_flagged.uvfits"
INDEX="$FITS.row_index_cache.npz"

if [[ ! -f "$FITS" || ! -f "$INDEX" ]]; then
  if [[ "$STRICT_DATA" == "0" ]]; then
    echo "[ripple-regression-ci] SKIP: real 3C468.1 data not found under \$WORK_DIR=$WORK_DIR"
    exit 0
  fi
  echo "[ripple-regression-ci] FAIL: real 3C468.1 data missing (set STRICT_DATA=0 to allow skip)" >&2
  echo "  FITS=$FITS" >&2
  echo "  INDEX=$INDEX" >&2
  exit 2
fi

RUN_TS="$(date +%Y%m%d_%H%M%S)"
TEST_ROOT="${TEST_ROOT:-$REPO_ROOT/work/ripple_characterisation_regression_${RUN_TS}}"
REPORT_FILE="$TEST_ROOT/regression_report.txt"

echo "[ripple-regression-ci] RUN_TS=$RUN_TS"
echo "[ripple-regression-ci] TEST_ROOT=$TEST_ROOT"

env WORK_DIR="$WORK_DIR" RUN_TS="$RUN_TS" TEST_ROOT="$TEST_ROOT" \
    bash "$REPO_ROOT/bin/test_ripple_characterisation_regression.sh"

[[ -f "$REPORT_FILE" ]] || {
  echo "[ripple-regression-ci] FAIL: report not found: $REPORT_FILE" >&2
  exit 2
}

grep -q '^RUN_OK=PASS$' "$REPORT_FILE" || { echo "[ripple-regression-ci] FAIL: RUN_OK not PASS" >&2; exit 1; }
grep -q '^OUTPUTS_PRESENT=PASS$' "$REPORT_FILE" || { echo "[ripple-regression-ci] FAIL: OUTPUTS_PRESENT not PASS" >&2; exit 1; }
grep -q '^JSON_VALID=PASS$' "$REPORT_FILE" || { echo "[ripple-regression-ci] FAIL: JSON_VALID not PASS" >&2; exit 1; }
grep -q '^KNOWN_MODEL_CROSSCHECK_PRESENT=PASS$' "$REPORT_FILE" || { echo "[ripple-regression-ci] FAIL: KNOWN_MODEL_CROSSCHECK_PRESENT not PASS" >&2; exit 1; }

echo "[ripple-regression-ci] PASS: ripple characterisation regression gate passed"
echo "[ripple-regression-ci] REPORT=$REPORT_FILE"
