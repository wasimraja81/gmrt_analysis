#!/usr/bin/env bash
# End-to-end stripe/baseline diagnostic wrapper.
#
# Usage:
#   bash stripe_baseline_diagnostic_example.sh
#   CUBE=./casa_out/moon/moon0520_uvmin0.000kl_moontrack_cube_recovered.image.fits bash stripe_baseline_diagnostic_example.sh

set -euo pipefail

PYTHON_CMD=${PYTHON_CMD:-python}
MS=${MS:-./casa_out/moon/moon0520.ms}
CUBE=${CUBE:-./casa_out/moon/moon0520_uvmin0.000kl_moontrack_cube_recovered.image.fits}
OUTDIR=${OUTDIR:-./diagnostics_out/stripe_baseline}
TAG=${TAG:-moon0520}
CELL_ARCSEC=${CELL_ARCSEC:-2.0}
TOP_N=${TOP_N:-10}

$PYTHON_CMD stripe_baseline_diagnostic.py \
  --ms "$MS" \
  --cube "$CUBE" \
  --cell-arcsec "$CELL_ARCSEC" \
  --top-n-baselines "$TOP_N" \
  --outdir "$OUTDIR" \
  --tag "$TAG"

echo "[stripe-diag] done. See:"
echo "  $OUTDIR/${TAG}_summary.txt"
echo "  $OUTDIR/${TAG}_report.json"
