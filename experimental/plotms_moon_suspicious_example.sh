#!/usr/bin/env bash
# Example wrapper for CASA plotms diagnostics on suspicious Moon baselines.
#
# Run with either:
#   bash plotms_moon_suspicious_example.sh
#   CASA_CMD="casa --nogui --nologger -c" bash plotms_moon_suspicious_example.sh

set -euo pipefail

CASA_CMD=${CASA_CMD:-python}
VIS=${VIS:-./casa_out/moon/moon0520.ms}
OUTDIR=${OUTDIR:-./diagnostics_out/moon_plotms}
SUSPECTS=${SUSPECTS:-'29&30;1&25;18&30;15&27'}
MAKE_EXCLUDED_COPY=${MAKE_EXCLUDED_COPY:-1}

CMD=(
  $CASA_CMD casa_plotms_suspicious_baselines.py
  --vis "$VIS"
  --outdir "$OUTDIR"
  --suspicious-baselines "$SUSPECTS"
)

if [[ "$MAKE_EXCLUDED_COPY" == "1" ]]; then
  CMD+=(--make-excluded-copy)
fi

echo "[plotms-moon] vis=$VIS"
echo "[plotms-moon] suspicious baselines=$SUSPECTS"
echo "[plotms-moon] outdir=$OUTDIR"
"${CMD[@]}"
