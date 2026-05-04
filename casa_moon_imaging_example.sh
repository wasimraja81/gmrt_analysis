#!/usr/bin/env bash
# CASA imaging example for split Moon UVFITS files.
# Images each Moon scan independently using a minimum uv cut of 0.12 kλ.
#
# Run with either:
#   bash casa_moon_imaging_example.sh            (if your active python has casatasks), or
#   CASA_CMD="casa --nogui --nologger -c" bash casa_moon_imaging_example.sh

set -euo pipefail

if [[ -z "${BASH_VERSION:-}" ]]; then
    echo "Please run this script with bash, not sh."
    exit 1
fi

# ── CASA launcher ─────────────────────────────────────────────────────────────
# Default: run with current python.
# If you have a CASA launcher, override like:
#   CASA_CMD="casa --nogui --nologger -c"
CASA_CMD=${CASA_CMD:-python}

# ── Input UVFITS files ────────────────────────────────────────────────────────
# Per-source subdirectory: split/<sourceName>/<sourceName>*_calibrated.uvfits
MOON_DIR=~/DATA/gmrt_40_014/work/split/moon
MOON_FITS=(
  ${MOON_DIR}/moon0520_calibrated.uvfits
  ${MOON_DIR}/moon0545_calibrated.uvfits
  ${MOON_DIR}/moon0605_calibrated.uvfits
  ${MOON_DIR}/moon0625_calibrated.uvfits
  ${MOON_DIR}/moon0635_calibrated.uvfits
)

# ── Imaging controls ──────────────────────────────────────────────────────────
OUTDIR=./casa_out/moon
UVMIN_KL=0.0
# UVMAX_KL unset => no upper uv cut (all baselines included)
CELL=2arcsec
IMSIZE=4096
NITER=6000
THRESHOLD=5mJy
SCALES=0,5,20,80,300,900
WEIGHTING=briggs
ROBUST=0.5
# Moon-tracking is the default (--moon-track-per-integration is True by
# default). Add --no-moon-track-per-integration to image as a static pointing.
# Add --wproject to enable W-projection gridder (slow, rarely needed for Moon).
STOKES=I
EXPORT_FITS=1   # 1 => add --export-fits, 0 => skip FITS export
CYCLENITER=250
CYCLES_PER_REPORT=1

mkdir -p "$OUTDIR"

echo "[casa-moon] imaging ${#MOON_FITS[@]} Moon scans with uvmin=${UVMIN_KL} kλ"

CMD=(
  $CASA_CMD casa_moon_imaging.py
  --fits "${MOON_FITS[@]}"
  --outdir "$OUTDIR"
  --uvmin-klambda "$UVMIN_KL"
  --cell "$CELL"
  --imsize "$IMSIZE"
  --niter "$NITER"
  --cycleniter "$CYCLENITER"
  --cycles-per-report "$CYCLES_PER_REPORT"
  --threshold "$THRESHOLD"
  --scales "$SCALES"
  --weighting "$WEIGHTING"
  --robust "$ROBUST"
  --stokes "$STOKES"
  --keep-ms
  --overwrite
)

if [[ "$EXPORT_FITS" == "1" ]]; then
  CMD+=(--export-fits)
  echo "[casa-moon] FITS export enabled"
else
  echo "[casa-moon] FITS export disabled"
fi

"${CMD[@]}"

echo "[casa-moon] done. outputs: $OUTDIR"
