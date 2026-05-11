#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_CMD="${PYTHON_CMD:-python}"

cd "$REPO_ROOT"

# Plot diagnostics for 3C468.1 primary-only calibrated split data
# Run this after visSplit_3c468.1_primary_example.sh

FITS=~/DATA/gmrt_40_014/work/split/3c468.1/3c468.1_primary_calibrated.uvfits
INDEX=~/DATA/gmrt_40_014/work/split/3c468.1/3c468.1_primary_calibrated.uvfits.row_index_cache.npz
SOURCE=3C468.1
CONFIG="$REPO_ROOT/preprocess_ugmrt.cfg"

CHAN_START=0
CHAN_END=127
OUTROOT=~/DATA/gmrt_40_014/work/diagnostics_out/secondary/3c468.1/transfer

LOG_DIR=~/DATA/gmrt_40_014/work/logs
RUN_TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTROOT" "$LOG_DIR"

if [[ ! -f "$FITS" ]]; then
  echo "Primary-only split file not found: $FITS"
  echo "Run visSplit_3c468.1_primary_example.sh first."
  exit 1
fi

if (( CHAN_START > CHAN_END )); then
  echo "Invalid channel range: CHAN_START ($CHAN_START) > CHAN_END ($CHAN_END)"
  exit 1
fi

if [[ -f "$OUTROOT/plotvis_3c468.1.pdf" ]]; then
  echo "[plotVis-3c468.1-primary] WARNING: existing plot outputs will be overwritten under: $OUTROOT"
fi

echo "[plotVis-3c468.1-primary] run config:"
echo "  FITS=$FITS"
echo "  INDEX=$INDEX"
echo "  SOURCE=$SOURCE"
echo "  CHAN_RANGE=$CHAN_START..$CHAN_END"
echo "  OUTROOT=$OUTROOT"
echo "[plotVis-3c468.1-primary] mode=calibrated: skipping --bpcal/--flag"

CMD=(
  "$PYTHON_CMD" "$REPO_ROOT/src/plotVis.py"
  --config "$CONFIG"
  --fits "$FITS"
  --outdir "$OUTROOT"
  --index-cache "$INDEX"
  --source "$SOURCE"
  --chan-range "$CHAN_START" "$CHAN_END"
  --elevation-min 25
  --products RR,LL
  --panels amp_uvdist,phase_uvdist,real_uvdist,imag_uvdist,amp_time,phase_time,real_time,imag_time,az_time,el_time,amp_freq,phase_freq,real_freq,imag_freq,ri_scatter,vector_avg,uv_sampling
  --sample-frac 0.01
  --overlay-flags
  --multipage both
)

CMD_FILE="$LOG_DIR/plotVis_3c468.1_primary_example_${RUN_TS}.cmd"
LOG_FILE="$LOG_DIR/plotVis_3c468.1_primary_example_${RUN_TS}.log"
{
  echo "# timestamp=$RUN_TS"
  echo "# cwd=$PWD"
  printf '%q ' "${CMD[@]}"
  printf '\n'
} > "$CMD_FILE"

echo "[plotVis-3c468.1-primary] cmd manifest: $CMD_FILE"
echo "[plotVis-3c468.1-primary] log: $LOG_FILE"

"${CMD[@]}" 2>&1 | tee "$LOG_FILE"

echo "[plotVis-3c468.1-primary] done. outputs under: $OUTROOT"
