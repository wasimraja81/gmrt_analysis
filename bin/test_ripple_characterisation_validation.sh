#!/usr/bin/env bash
set -euo pipefail

# Validates the fresh characterise_ripple() implementation against two real
# JSON outputs already on disk from the prior-art prototype
# (experimental/ripple_calibration_test.py), used here purely as historical
# reference baselines — see RC-16 in ripple_characterisation_tickets.md.
#
# Two checks, run per (source, polarisation):
#   ALPHA_TOLERANCE_CHECK  — locally-fit alpha_nu0 (quad/curved power-law fit)
#     agrees with the baseline's quad alpha_nu0 to within 1% relative. Both
#     implementations run the same plain OLS fit with no heuristic freedom,
#     so this is expected to be a tight, near-exact match.
#   PERIOD_TOLERANCE_CHECK — the baseline's own maximum-amplitude ripple
#     component period is found (within 20% relative) *somewhere* in the
#     fresh implementation's candidate list for that (source, pol) — not
#     necessarily as the fresh implementation's own top/"primary" candidate.
#     This is deliberately looser than a "same dominant component" check:
#     the two implementations use genuinely different multi-component
#     fitting strategies (joint least-squares here vs. the prototype's
#     heuristic sequential selection), so which candidate ends up with the
#     largest *fitted amplitude* can legitimately differ even when both
#     implementations detect the same underlying spectral features.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORK_DIR="${WORK_DIR:-$HOME/DATA/gmrt_40_014/work}"
PYTHON_CMD="${PYTHON_CMD:-python}"
STRICT_DATA="${STRICT_DATA:-1}"

BASELINE_3C468="$REPO_ROOT/experimental/out/ripple_characterisation_src-3c468-1_fit_summary.json"
BASELINE_3C48="$REPO_ROOT/experimental/out/ripple_characterisation_src-3c48_fit_summary.json"
FITS_3C468="$WORK_DIR/split/3c468.1/3c468.1_primary_secondary_calibrated_flagged.uvfits"
INDEX_3C468="$FITS_3C468.row_index_cache.npz"
FITS_3C48="$WORK_DIR/split/3c48/3c48_calibrated_flagged.uvfits"
INDEX_3C48="$FITS_3C48.row_index_cache.npz"

# The baseline JSONs (like the real UVFITS/row-index-cache data below) are
# untracked local artifacts under experimental/out/, not committed to git —
# their absence is treated the same way as missing real data: a clean skip
# under STRICT_DATA=0, a hard fail otherwise.
if [[ ! -f "$BASELINE_3C468" || ! -f "$BASELINE_3C48" \
      || ! -f "$FITS_3C468" || ! -f "$INDEX_3C468" || ! -f "$FITS_3C48" || ! -f "$INDEX_3C48" ]]; then
  if [[ "$STRICT_DATA" == "0" ]]; then
    echo "[ripple-validation] SKIP: baseline JSONs and/or real 3C468.1/3C48 data not found"
    exit 0
  fi
  echo "[ripple-validation] FAIL: baseline JSONs and/or real 3C468.1/3C48 data missing (set STRICT_DATA=0 to allow skip)" >&2
  echo "  BASELINE_3C468=$BASELINE_3C468" >&2
  echo "  BASELINE_3C48=$BASELINE_3C48" >&2
  echo "  FITS_3C468=$FITS_3C468" >&2
  echo "  FITS_3C48=$FITS_3C48" >&2
  exit 2
fi

RUN_TS="$(date +%Y%m%d_%H%M%S)"
TEST_ROOT="${TEST_ROOT:-$REPO_ROOT/work/ripple_characterisation_validation_${RUN_TS}}"
REPORT_FILE="$TEST_ROOT/validation_report.txt"
mkdir -p "$TEST_ROOT"

run_fresh() {
  local label="$1" fits="$2" index="$3" source="$4" outdir="$5"
  mkdir -p "$outdir"
  "$PYTHON_CMD" "$REPO_ROOT/src/characterise_ripple.py" \
    --fits "$fits" \
    --index-cache "$index" \
    --source "$source" \
    --chan-range 0 127 \
    --elevation-min 25 \
    --physical-model-mode fit \
    --peak-snr-threshold 3.0 \
    --outfile-prefix "$label" \
    --outdir "$outdir" \
    --provenance-dir "$TEST_ROOT/provenance" \
    > "$TEST_ROOT/run_${label}.log" 2>&1
}

run_fresh check_3c468 "$FITS_3C468" "$INDEX_3C468" 3C468.1 "$TEST_ROOT/3c468"
run_fresh check_3c48 "$FITS_3C48" "$INDEX_3C48" 3C48 "$TEST_ROOT/3c48"

compare_source() {
  local baseline_json="$1" fresh_json="$2"
  "$PYTHON_CMD" - "$baseline_json" "$fresh_json" <<'PY'
import json
import sys

baseline_path, fresh_path = sys.argv[1], sys.argv[2]

with open(baseline_path) as f:
    baseline = json.load(f, parse_constant=lambda tok: float('nan'))
with open(fresh_path) as f:
    fresh = json.load(f)

ALPHA_REL_TOL = 0.01
PERIOD_REL_TOL = 0.20

overall_alpha_ok = True
overall_period_ok = True

for pol in ('rr', 'll'):
    pol_upper = pol.upper()
    baseline_alpha = baseline[pol]['quad']['alpha_nu0']
    fresh_alpha = fresh['per_pol'][pol_upper]['power_law_fit']['alpha_nu0']
    alpha_rel_diff = abs(fresh_alpha - baseline_alpha) / abs(baseline_alpha)
    alpha_ok = alpha_rel_diff <= ALPHA_REL_TOL
    overall_alpha_ok &= alpha_ok
    print(f"{pol_upper}_ALPHA_BASELINE={baseline_alpha:.6f}")
    print(f"{pol_upper}_ALPHA_FRESH={fresh_alpha:.6f}")
    print(f"{pol_upper}_ALPHA_REL_DIFF={alpha_rel_diff:.6f}")
    print(f"{pol_upper}_ALPHA_CHECK={'PASS' if alpha_ok else 'FAIL'}")

    baseline_components = baseline[f'ripple_{pol}']['beta0']['components']
    baseline_dominant = max(
        (c for c in baseline_components if c['amp1'] == c['amp1']),  # drop NaN amp1, if any
        key=lambda c: c['amp1'],
    )
    baseline_period = baseline_dominant['period_mhz']

    fresh_periods = [c['period_mhz'] for c in fresh['per_pol'][pol_upper]['components']]
    if fresh_periods:
        closest = min(fresh_periods, key=lambda p: abs(p - baseline_period) / baseline_period)
        period_rel_diff = abs(closest - baseline_period) / baseline_period
    else:
        closest = float('nan')
        period_rel_diff = float('inf')
    period_ok = period_rel_diff <= PERIOD_REL_TOL
    overall_period_ok &= period_ok
    print(f"{pol_upper}_PERIOD_BASELINE_DOMINANT_MHZ={baseline_period:.4f}")
    print(f"{pol_upper}_PERIOD_FRESH_CLOSEST_MHZ={closest:.4f}")
    print(f"{pol_upper}_PERIOD_REL_DIFF={period_rel_diff:.4f}")
    print(f"{pol_upper}_PERIOD_CHECK={'PASS' if period_ok else 'FAIL'}")

print(f"SOURCE_ALPHA_OK={'PASS' if overall_alpha_ok else 'FAIL'}")
print(f"SOURCE_PERIOD_OK={'PASS' if overall_period_ok else 'FAIL'}")
PY
}

{
  echo "RUN_TS=$RUN_TS"
  echo "=== 3C468.1 ==="
  compare_source "$BASELINE_3C468" "$TEST_ROOT/3c468/check_3c468_summary.json"
  echo "=== 3C48 ==="
  compare_source "$BASELINE_3C48" "$TEST_ROOT/3c48/check_3c48_summary.json"
} | tee "$TEST_ROOT/comparison.txt"

ALPHA_TOLERANCE_CHECK=PASS
PERIOD_TOLERANCE_CHECK=PASS
grep -q '^SOURCE_ALPHA_OK=FAIL$' "$TEST_ROOT/comparison.txt" && ALPHA_TOLERANCE_CHECK=FAIL
grep -q '^SOURCE_PERIOD_OK=FAIL$' "$TEST_ROOT/comparison.txt" && PERIOD_TOLERANCE_CHECK=FAIL

{
  echo "RUN_TS=$RUN_TS"
  echo "TEST_ROOT=$TEST_ROOT"
  cat "$TEST_ROOT/comparison.txt"
  echo "ALPHA_TOLERANCE_CHECK=$ALPHA_TOLERANCE_CHECK"
  echo "PERIOD_TOLERANCE_CHECK=$PERIOD_TOLERANCE_CHECK"
} > "$REPORT_FILE"

echo
echo "[ripple-validation] ALPHA_TOLERANCE_CHECK=$ALPHA_TOLERANCE_CHECK"
echo "[ripple-validation] PERIOD_TOLERANCE_CHECK=$PERIOD_TOLERANCE_CHECK"
echo "[ripple-validation] report: $REPORT_FILE"

if [[ "$ALPHA_TOLERANCE_CHECK" != "PASS" || "$PERIOD_TOLERANCE_CHECK" != "PASS" ]]; then
  echo "[ripple-validation] FAIL: see $REPORT_FILE for per-(source,pol) detail. A period-only" >&2
  echo "  disagreement is expected/documented for 3C468.1 RR — see RC-16 in" >&2
  echo "  ripple_characterisation_tickets.md before investigating further." >&2
  exit 1
fi

echo "[ripple-validation] PASS"
