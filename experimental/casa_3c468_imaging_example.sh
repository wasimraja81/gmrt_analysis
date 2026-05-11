#!/usr/bin/env bash
# CASA imaging example for calibrated 3c468.1 UVFITS.
# Uses casa_moon_imaging.py (which already includes robust import fallback and
# major-cycle progress reporting).

set -euo pipefail

if [[ -z "${BASH_VERSION:-}" ]]; then
    echo "Please run this script with bash, not sh."
    exit 1
fi

# Activate your project venv (contains casatasks/casatools/pyuvdata)
source /Users/raj030/github-wasimraja81/gmrt_analysis/gmrt/bin/activate

# CASA launcher (optional override):
#   CASA_CMD="casa --nogui --nologger -c" bash casa_3c468_imaging_example.sh
CASA_CMD=${CASA_CMD:-python}

# Input calibrated UVFITS
CAL_FITS=~/DATA/gmrt_40_014/work/split/3c468.1/3c468.1_calibrated.uvfits

# Imaging controls
OUTDIR=./casa_out/3c468.1
UVMIN_KL=0.0
CELL=2arcsec
IMSIZE=1024
NITER=2000
THRESHOLD=2mJy
DECONVOLVER=hogbom
SCALES=0
WEIGHTING=briggs
ROBUST=0.0
STOKES=I
EXPORT_FITS=1   # 1 => write .image.fits, 0 => skip FITS export

mkdir -p "$OUTDIR"

echo "[casa-3c468] imaging 3c468.1 from: $CAL_FITS"

CMD=(
  $CASA_CMD casa_moon_imaging.py
  --fits "$CAL_FITS"
  --outdir "$OUTDIR"
  --uvmin-klambda "$UVMIN_KL"
  --cell "$CELL"
  --imsize "$IMSIZE"
  --niter "$NITER"
  --threshold "$THRESHOLD"
  --deconvolver "$DECONVOLVER"
  --scales "$SCALES"
  --weighting "$WEIGHTING"
  --robust "$ROBUST"
  --stokes "$STOKES"
  --keep-ms
  --overwrite
)

if [[ "$EXPORT_FITS" == "1" ]]; then
  CMD+=(--export-fits)
  echo "[casa-3c468] FITS export enabled"
else
  echo "[casa-3c468] FITS export disabled"
fi

"${CMD[@]}"

echo "[casa-3c468] done. outputs: $OUTDIR"
