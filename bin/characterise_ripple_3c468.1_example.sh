#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_CMD="${PYTHON_CMD:-python}"

cd "$REPO_ROOT"

# Uses the already-calibrated 3C468.1 split (same data as
# bin/plotVis_3c468.1_example.sh's DATA_MODE=calibrated) — no
# --bandpass-solution is passed, since this data has already had a bandpass
# solution applied upstream (see "Explicitly out of scope" in
# ripple_characterisation_tickets.md: this script characterises, it does not
# apply or build a correction).
FITS=~/DATA/gmrt_40_014/work/split/3c468.1/3c468.1_primary_secondary_calibrated_flagged.uvfits
INDEX=~/DATA/gmrt_40_014/work/split/3c468.1/3c468.1_primary_secondary_calibrated_flagged.uvfits.row_index_cache.npz
SOURCE=3C468.1
CHAN_START=0
CHAN_END=127
OUTDIR=~/DATA/gmrt_40_014/work/diagnostics_out/secondary/3c468.1/ripple_characterisation

mkdir -p "$OUTDIR"

echo "[characterise_ripple-3c468.1] run config:"
echo "  FITS=$FITS"
echo "  INDEX=$INDEX"
echo "  SOURCE=$SOURCE"
echo "  CHAN_RANGE=$CHAN_START..$CHAN_END"
echo "  OUTDIR=$OUTDIR"

"$PYTHON_CMD" "$REPO_ROOT/src/characterise_ripple.py" \
  --fits "$FITS" \
  --index-cache "$INDEX" \
  --source "$SOURCE" \
  --chan-range "$CHAN_START" "$CHAN_END" \
  --elevation-min 25 \
  --physical-model-mode fit \
  --outdir "$OUTDIR"

echo "[characterise_ripple-3c468.1] done. outputs under: $OUTDIR"
