#!/usr/bin/env bash
# CASA imaging example for split Moon UVFITS files.
# Images each Moon scan independently with uvmin=0.0 kλ (all baselines).
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
SCALES=0,5,20,80,300,400  # multi-scale CLEAN with these scales in pixels (0=point source, then increasing sizes)
SMALLSCALEBIAS=0.0
WEIGHTING=briggs
ROBUST=0.5
# Moon-tracking is the default (--moon-track-per-integration is True by
# default). Add --no-moon-track-per-integration to image as a static pointing.
# Add --wproject to enable W-projection gridder (slow, rarely needed for Moon).
STOKES=I
EXPORT_FITS=1   # 1 => add --export-fits, 0 => skip FITS export
INTEGRATION_NITER=800  # total budget of minor CLEAN iterations per integration/snapshot in moon-track mode (review/tune as needed)
INTEGRATION_NMAJOR=6   # major cycles per snapshot (default was 2)
CYCLENITER=50   # minor CLEAN iterations per major cycle per snapshot (maps to --integration-cycleniter; default was 100)
# If ALL cycles are fully used with no early stop, then:
#   INTEGRATION_NITER = CYCLENITER * INTEGRATION_NMAJOR
# In real runs, early threshold/convergence/divergence stops can end earlier, so this is often an upper bound.
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
  --integration-niter "$INTEGRATION_NITER"
  --integration-nmajor "$INTEGRATION_NMAJOR"\
  # Moon-track path uses --integration-cycleniter; --cycleniter is only for non-moon-track mode.
  --integration-cycleniter "$CYCLENITER"
  --cycles-per-report "$CYCLES_PER_REPORT"
  --threshold "$THRESHOLD"
  --scales "$SCALES"
  --smallscalebias "$SMALLSCALEBIAS"
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
