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
# Per-source subdirectory: split/moon/moon*_*.uvfits
MOON_DIR=~/DATA/gmrt_40_014/work/split/moon
MOON_CAL_MODE_TAG=primaryonly
MOON_FITS=(
  ${MOON_DIR}/moon0520_${MOON_CAL_MODE_TAG}_calibrated.uvfits
  ${MOON_DIR}/moon0545_${MOON_CAL_MODE_TAG}_calibrated.uvfits
  ${MOON_DIR}/moon0605_${MOON_CAL_MODE_TAG}_calibrated.uvfits
  ${MOON_DIR}/moon0625_${MOON_CAL_MODE_TAG}_calibrated.uvfits
  ${MOON_DIR}/moon0635_${MOON_CAL_MODE_TAG}_calibrated.uvfits
)
# Select subset to image:
#   all    -> image all listed scans
#   1..N   -> image one scan by 1-based index in MOON_FITS
#   token  -> image entries whose filename contains token (e.g. 0520)
MOON_SCAN_SELECTOR=all

# ── Imaging controls ──────────────────────────────────────────────────────────
OUTDIR=./casa_out/moon
UVMIN_KL=0.015 # kλ (set to 0.0 to include all baselines, or e.g. 0.075 to exclude shortest ~100m baselines)
UVMAX_KL='' # unset => no upper uv cut (all baselines included)
CELL=2arcsec
IMSIZE=4096
SCALES=0,5,15,45,135  # multi-scale CLEAN with these scales in pixels (0=point source, then increasing sizes)
SMALLSCALEBIAS=0.3
LOOP_GAIN=0.1        # CLEAN loop gain: fraction of peak subtracted per minor-cycle iter (CASA default 0.1)
CYCLEFACTOR=1.5      # major-cycle trigger threshold scale: lower → fewer major cycles (CASA default 1.5)
WEIGHTING=briggs
ROBUST=0.5
UVTAPER='30arcsec'     # Gaussian uv-taper, e.g. '60arcsec' or '90arcsec,60arcsec,45deg' (blank = no taper)
# Moon-tracking is the default (--moon-track-per-integration is True by
# default). Add --no-moon-track-per-integration to image as a static pointing.
# Add --wproject to enable W-projection gridder (slow, rarely needed for Moon).
STOKES=I
TIMERANGE="" #"2021/07/26/00:09:02~2021/07/26/00:10:22"  # last 10 integrations of moon0520 (integ 104-113, 00:09:06~00:10:18, cadence~8s)
EXPORT_FITS=1   # 1 => add --export-fits, 0 => skip FITS export
# Unified CLEAN cycle controls (used in both native whole-scan and per-integration modes)
NMAJOR=10        # maximum major cycles
CYCLENITER=50   # minor-cycle iterations per major cycle
# niter is derived for both modes to keep behavior consistent and simple.
NITER=$((NMAJOR * CYCLENITER))
INTEGRATION_JOBS=1  # parallel per-integration workers (scan loop remains serial)
DECONVOLVER=hogbom  # CASA deconvolver: multiscale (default), hogbom, or clark
# Imaging mode selector. This is the parent switch that decides which child
# parameters are active:
#   native_moon_track        -> whole-scan native CASA moving-source tracking
#   whole_scan_static        -> whole-scan imaging at the observed phase centre
#   per_integration_moon     -> per-integration Moon-centred snapshots
#   per_integration_observed -> per-integration snapshots at the observed phase centre
IMAGING_MODE=per_integration_observed
#IMAGING_MODE=native_moon_track

# ── Child params [native_moon_track only] ─────────────────────────────────────
NATIVE_TRACK_PHASECENTER=MOON   # CASA source name or J2000 direction used as the fixed tracking phasecenter

# ── Child params [per_integration_moon only] ──────────────────────────────────
PHASE_STRATEGY=tclean           # tclean (shift internally, low-I/O) or phaseshift (explicit shifted MS)

# ── Child params [per_integration_moon | per_integration_observed] ────────────
KEEP_INTEGRATION_PRODUCTS=0    # 1 => keep CASA per-integration image products on disk
KEEP_INTEGRATION_FITS=0        # 1 => keep exported per-integration FITS snapshots on disk
WRITE_INTEGRATION_CUBE=1       # 1 => stack snapshots into a per-integration FITS cube

# ── Shared clean thresholds (selected automatically by IMAGING_MODE) ──────────
THRESHOLD_PER_INTEGRATION=5mJy  # threshold used for per_integration_* modes
THRESHOLD_WHOLE_SCAN=1mJy       # threshold used for native_moon_track / whole_scan_static

# ── Misc ──────────────────────────────────────────────────────────────────────
EXCLUDE_BASELINES=''  # e.g. '29&30;1&25;18&30;15&27;11&12;9&25;19&30'
CYCLES_PER_REPORT=1

mkdir -p "$OUTDIR"

SELECTED_MOON_FITS=()
case "$MOON_SCAN_SELECTOR" in
  all)
    SELECTED_MOON_FITS=("${MOON_FITS[@]}")
    ;;
  ''|*[!0-9]*)
    for f in "${MOON_FITS[@]}"; do
      if [[ "$(basename "$f")" == *"$MOON_SCAN_SELECTOR"* ]]; then
        SELECTED_MOON_FITS+=("$f")
      fi
    done
    ;;
  *)
    idx=$((MOON_SCAN_SELECTOR - 1))
    if (( idx < 0 || idx >= ${#MOON_FITS[@]} )); then
      echo "[casa-moon] ERROR: MOON_SCAN_SELECTOR=$MOON_SCAN_SELECTOR out of range 1..${#MOON_FITS[@]}"
      exit 2
    fi
    SELECTED_MOON_FITS=("${MOON_FITS[$idx]}")
    ;;
esac

if (( ${#SELECTED_MOON_FITS[@]} == 0 )); then
  echo "[casa-moon] ERROR: MOON_SCAN_SELECTOR='$MOON_SCAN_SELECTOR' matched no scans"
  exit 2
fi

MOON_TRACK_PER_INTEGRATION=0
INTEGRATION_PHASECENTER_MODE=observed
EFFECTIVE_NATIVE_TRACK_PHASECENTER=""

case "$IMAGING_MODE" in
  native_moon_track)
    MOON_TRACK_PER_INTEGRATION=0
    EFFECTIVE_NATIVE_TRACK_PHASECENTER="$NATIVE_TRACK_PHASECENTER"
    CLEAN_THRESHOLD="$THRESHOLD_WHOLE_SCAN"
    ;;
  whole_scan_static)
    MOON_TRACK_PER_INTEGRATION=0
    EFFECTIVE_NATIVE_TRACK_PHASECENTER=""
    CLEAN_THRESHOLD="$THRESHOLD_WHOLE_SCAN"
    ;;
  per_integration_moon)
    MOON_TRACK_PER_INTEGRATION=1
    INTEGRATION_PHASECENTER_MODE=moon
    EFFECTIVE_NATIVE_TRACK_PHASECENTER=""
    CLEAN_THRESHOLD="$THRESHOLD_PER_INTEGRATION"
    ;;
  per_integration_observed)
    MOON_TRACK_PER_INTEGRATION=1
    INTEGRATION_PHASECENTER_MODE=observed
    EFFECTIVE_NATIVE_TRACK_PHASECENTER=""
    CLEAN_THRESHOLD="$THRESHOLD_PER_INTEGRATION"
    ;;
  *)
    echo "[casa-moon] ERROR: unsupported IMAGING_MODE='$IMAGING_MODE'"
    echo "[casa-moon] valid modes: native_moon_track | whole_scan_static | per_integration_moon | per_integration_observed"
    exit 2
    ;;
esac

echo "[casa-moon] cal-mode tag: ${MOON_CAL_MODE_TAG}"
echo "[casa-moon] imaging ${#SELECTED_MOON_FITS[@]} Moon scans with uvmin=${UVMIN_KL} kλ, uvmax=${UVMAX_KL:-none} kλ (selector=${MOON_SCAN_SELECTOR})"
for f in "${SELECTED_MOON_FITS[@]}"; do
  echo "[casa-moon] selected scan: $(basename "$f")"
done
echo "[casa-moon] imaging mode: ${IMAGING_MODE}"
echo "[casa-moon] stopping threshold: ${CLEAN_THRESHOLD}"

CMD=(
  $CASA_CMD casa_moon_imaging.py
  --fits "${SELECTED_MOON_FITS[@]}"
  --outdir "$OUTDIR"
  --uvmin-klambda "$UVMIN_KL"
  --cell "$CELL"
  --imsize "$IMSIZE"
  --niter "$NITER"
  --nmajor "$NMAJOR"
  --integration-niter "$NITER"
  --integration-nmajor "$NMAJOR"
  --integration-jobs "$INTEGRATION_JOBS"
  --integration-phasecenter-mode "$INTEGRATION_PHASECENTER_MODE"
  --native-track-phasecenter "$EFFECTIVE_NATIVE_TRACK_PHASECENTER"
  --cycleniter "$CYCLENITER"
  --integration-cycleniter "$CYCLENITER"
  --cycles-per-report "$CYCLES_PER_REPORT"
  --threshold "$CLEAN_THRESHOLD"
  --scales "$SCALES"
  --smallscalebias "$SMALLSCALEBIAS"
  --gain "$LOOP_GAIN"
  --cyclefactor "$CYCLEFACTOR"
  --deconvolver "$DECONVOLVER"
  --weighting "$WEIGHTING"
  --robust "$ROBUST"
  --stokes "$STOKES"
  --keep-ms
  --overwrite
)

if [[ -n "$EXCLUDE_BASELINES" ]]; then
  CMD+=(--exclude-baselines "$EXCLUDE_BASELINES")
  echo "[casa-moon] excluding baselines: $EXCLUDE_BASELINES"
fi

if [[ -n "$UVMAX_KL" ]]; then
  CMD+=(--uvmax-klambda "$UVMAX_KL")
  echo "[casa-moon] uvmax: ${UVMAX_KL} kλ"
fi

if [[ -n "$UVTAPER" ]]; then
  CMD+=(--uvtaper "$UVTAPER")
  echo "[casa-moon] uvtaper: ${UVTAPER}"
fi

if [[ -n "$TIMERANGE" ]]; then
  CMD+=(--timerange "$TIMERANGE")
  echo "[casa-moon] timerange filter enabled: $TIMERANGE"
fi

if [[ "$MOON_TRACK_PER_INTEGRATION" == "0" ]]; then
  CMD+=(--no-moon-track-per-integration)
  if [[ -n "$EFFECTIVE_NATIVE_TRACK_PHASECENTER" ]]; then
    echo "[casa-moon] whole-scan native tracking enabled (phasecenter=${EFFECTIVE_NATIVE_TRACK_PHASECENTER})"
  else
    echo "[casa-moon] whole-scan static observed-phase-centre imaging enabled"
  fi
else
  if [[ "$IMAGING_MODE" == "per_integration_moon" ]]; then
    CMD+=(--phase-strategy "$PHASE_STRATEGY")
    echo "[casa-moon] per-integration Moon-centred imaging enabled (phase-strategy=${PHASE_STRATEGY})"
  else
    echo "[casa-moon] per-integration static observed-phase-centre imaging enabled"
  fi

  if [[ "$KEEP_INTEGRATION_PRODUCTS" == "1" ]]; then
    CMD+=(--keep-integration-products)
    echo "[casa-moon] keeping per-integration CASA image products"
  fi

  if [[ "$KEEP_INTEGRATION_FITS" == "1" ]]; then
    CMD+=(--keep-integration-fits)
    echo "[casa-moon] keeping per-integration FITS snapshots"
  fi

  if [[ "$WRITE_INTEGRATION_CUBE" == "0" ]]; then
    CMD+=(--no-write-moontrack-cube)
    echo "[casa-moon] per-integration FITS cube disabled"
  fi
fi

if [[ "$EXPORT_FITS" == "1" ]]; then
  CMD+=(--export-fits)
  echo "[casa-moon] FITS export enabled"
else
  echo "[casa-moon] FITS export disabled"
fi

"${CMD[@]}"

echo "[casa-moon] done. outputs: $OUTDIR"
