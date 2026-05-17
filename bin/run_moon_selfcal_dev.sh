#!/usr/bin/env bash
# run_moon_selfcal_dev.sh
#
# Tuning script for moon per-integration imaging + phase-only selfcal.
# Edit the parameters below and re-run to experiment.
#
# Usage:
#   bash bin/run_moon_selfcal_dev.sh
#
# To test a different scan, change SCAN and UVFITS below.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_GMRT_PYTHON="$REPO_ROOT/gmrt/bin/python"
if [[ -x "$DEFAULT_GMRT_PYTHON" ]]; then
    PYTHON="${PYTHON_CMD:-$DEFAULT_GMRT_PYTHON}"
else
    PYTHON="${PYTHON_CMD:-python}"
fi

# ── Data ──────────────────────────────────────────────────────────────────────
SCAN="MOON0520"
SCAN_LOWER="$(printf '%s' "$SCAN" | tr '[:upper:]' '[:lower:]')"

# Use primary+secondary calibrated data:
# Last 3C468.1 secondary scan ended ~15 min before MOON0520 — making
# primary+secondary the best-calibrated starting point for all moon scans.
UVFITS=~/DATA/gmrt_40_014/work/split/moon/moon0520_primary_secondary_calibrated_flagged.uvfits

# Row index cache is derived automatically from UVFITS path
# (<uvfits>.row_index_cache.npz). Override with --index if needed.
# INDEX=~/DATA/gmrt_40_014/work/split/moon/moon0520_primary_secondary_calibrated_flagged.uvfits.row_index_cache.npz

# ── Optional extra clustering flag table(s) (moon0520 dev) ───────────────────
# Leave empty ("") to skip the per-baseline clustering flags.
#
# Backward-compatible single-path variable:
#   EXTRA_FLAG_TABLE=/path/to/one.json
#
# Preferred multi-table variable (comma- or space-separated):
#   EXTRA_FLAG_TABLES="/path/to/uv1_5kl.json,/path/to/uv5_10kl.json"
#
# If one or more flag tables are found and the pre-flagged UVFITS does not yet
# exist, visSplit.py is run automatically to produce it; subsequent runs reuse
# the cached file for that exact flag-table combination.
EXTRA_FLAG_TABLE="${EXTRA_FLAG_TABLE:-~/DATA/gmrt_40_014/work/diagnostics_out/target/moon0520/flagging/moon0520_dev_flag_table.json}"
EXTRA_FLAG_TABLES="${EXTRA_FLAG_TABLES:-$EXTRA_FLAG_TABLE}"

# Derived output path base for the moon-clustering-flagged UVFITS (auto-named;
# do not edit). The original _flagged suffix already reflects primary+secondary
# flags.  Additional clustering tables are layered on top of that.
_UVFITS_BASE="${UVFITS%.uvfits}"
_UVFITS_BASE="${_UVFITS_BASE%.FITS}"

OUTDIR="${OUTDIR:-$HOME/DATA/gmrt_40_014/work/casa_selfcal/moon0520_stk10}"
NO_MASK="${NO_MASK:-0}"
USE_TCLEAN_PHASECENTER="${USE_TCLEAN_PHASECENTER:-0}"
LOG_DIR=~/DATA/gmrt_40_014/work/logs
RUN_TS="$(date +%Y%m%d_%H%M%S)"
MAKE_MOVIE_AFTER_SELFCAL="${MAKE_MOVIE_AFTER_SELFCAL:-1}"
MOVIE_FPS="${MOVIE_FPS:-4}"
MOVIE_CMAP="${MOVIE_CMAP:-magma}"
MOVIE_PERCENTILE_LOW="${MOVIE_PERCENTILE_LOW:-5}"
MOVIE_PERCENTILE_HIGH="${MOVIE_PERCENTILE_HIGH:-99.5}"
RUN_DESTRIPE_AFTER_SELFCAL="${RUN_DESTRIPE_AFTER_SELFCAL:-0}"
MAKE_DESTRIPED_MOVIE="${MAKE_DESTRIPED_MOVIE:-1}"
MAKE_COMPARE_MOVIE="${MAKE_COMPARE_MOVIE:-1}"
DESTRIPE_ITERS="${DESTRIPE_ITERS:-20}"
DESTRIPE_GLOB_PATTERN="${DESTRIPE_GLOB_PATTERN:-*_final.fits}"
DESTRIPE_MOON_MASK_CENTER="${DESTRIPE_MOON_MASK_CENTER:-ephem}"
DESTRIPE_BAND_MODE="${DESTRIPE_BAND_MODE:-1}"
DESTRIPE_BAND_SIGMA_PIX="${DESTRIPE_BAND_SIGMA_PIX:-6.0}"
DESTRIPE_BAND_DEPTH="${DESTRIPE_BAND_DEPTH:-0.35}"
DESTRIPE_DEBUG_DIR="${DESTRIPE_DEBUG_DIR:-$OUTDIR/destripe_debug}"
DESTRIPE_WRITE_DIR="${DESTRIPE_WRITE_DIR:-$OUTDIR/destriped_finals}"
DESTRIPE_STACK_OUTPUT="${DESTRIPE_STACK_OUTPUT:-$OUTDIR/${SCAN_LOWER}_destriped_stack.fits}"

# ── Integration selection ────────────────────────────────────────────────────────
# All 113 integrations (0-based, 0..112). Grouped into stacks of STACK_SIZE below.
# Override from environment: INTEGRATIONS="10 11 12 13 14 15 16 17 18 19" bash run_moon_selfcal_dev.sh
INTEGRATIONS="${INTEGRATIONS:-$(seq 0 112)}"

# ── Stacking ───────────────────────────────────────────────────────────────────
# 10 integrations × 8s = 80s per stack. Moon moves ~3" in 80s → well within 8" beam.
# 113 integrations / 10 = ~11 stacks → ~11 output images for the movie.
STACK_SIZE="${STACK_SIZE:-10}"

# ── Image geometry ────────────────────────────────────────────────────────────
CELL="1.5arcsec"          # beam ~7.4 arcsec, cell = beam/5
IMSIZE=3072               # 3072 * 1.5" = 76.8 arcmin — covers primary beam + drift with margin
STOKES="I"
WEIGHTING="briggs"
ROBUST=0.5
DECONVOLVER="multiscale"
# ── Moon mask ─────────────────────────────────────────────────────────────────
# Moon angular diameter ~30' — mask radius 20' gives disk + margin.
MASK_RADIUS_ARCMIN=20.0

# ── Per-cycle selfcal parameters (comma-separated; one value per cycle) ────────
# Centre freq = 322.67 MHz  →  1 kλ = 929 m
# Moon is ~1000 Jy at 322 MHz → high SNR on ALL baselines including 25 km arm baselines.
# Use all baselines for both imaging and gaincal so arm antennas receive phase corrections.
# A progressive uvmax cut would leave arm antennas unsolved in early cycles, then
# those uncorrected antennas degrade the full-resolution image.
NITER_PER_CYCLE="100,200,400,600,1000"  # increase niter each cycle as the model improves and can support deeper clean
# Imaging uvrange: all baselines
UVMIN_PER_CYCLE=",,,,"                        # no lower limit — include short baselines for Moon disk
UVMAX_PER_CYCLE=",,,,"                        # no upper limit
# Calibration uvrange: also all baselines (Moon SNR is sufficient on arm baselines)
UVMIN_CAL_PER_CYCLE=",,,,"
UVMAX_CAL_PER_CYCLE=",,,,"                    # no upper limit — solve for all antennas each cycle
SCALES_PER_CYCLE="0,5,15|0,5,15|0,5,15,45|0,5,15,45|0,5,15,45,90"  # add larger scales in later cycles to capture extended emission as it improves

# ── Final (post-selfcal) image ─────────────────────────────────────────────────
NITER_FINAL=1000    # final image: deeper clean (uses last cycle's uv/scale params)
THRESHOLD="0mJy"
CYCLENITER=50       # low cycleniter → more major cycles → more conservative CLEAN

# ── Selfcal ───────────────────────────────────────────────────────────────────
MINSNR="${MINSNR:-1.0}"          # gaincal minsnr — lowered: Moon model is weak (~0.2 Jy per cycle 1), data already well-calibrated
LOOP_GAIN="${LOOP_GAIN:-0.05}"      # CLEAN loop gain — 0.05 is conservative for extended Moon emission
NEGATIVETHRESHOLD="${NEGATIVETHRESHOLD:-0.001}"  # Jy/beam: restrict to positive-only cleaning in selfcal cycles
SOLMODE="${SOLMODE:-L1}"       # robust solver: L1R = iterative re-weighted LS, robust against outlier baselines
                     # Safe now that all baselines are used (~27 antennas, ~350 baselines)
REFANT="${REFANT:-1}"          # reference antenna NAME in CASA MS (phys ID 1 = C00:01, central arm, 100% BP valid)
REFANTMODE="${REFANTMODE:-flex}"   # flex = switch to next-best antenna if refant drops out

# ── Misc ──────────────────────────────────────────────────────────────────────
OVERWRITE="${OVERWRITE:---overwrite}"   # set to "" to reuse existing MS / skip already-done ints

# ─────────────────────────────────────────────────────────────────────────────
cd "$REPO_ROOT"
mkdir -p "$LOG_DIR"

# ── Optional pre-flagging step via visSplit.py ────────────────────────────────
# Expand ~ in paths (bash doesn't expand ~ inside variables in all contexts)
UVFITS_EXPANDED="${UVFITS/#\~/$HOME}"

UVFITS_FOR_SELFCAL="$UVFITS_EXPANDED"
declare -a EXTRA_FLAG_PATHS=()
if [[ -n "${EXTRA_FLAG_TABLES:-}" ]]; then
    EXTRA_FLAG_TABLES_EXPANDED="${EXTRA_FLAG_TABLES//,/ }"
    for extra_flag_path in $EXTRA_FLAG_TABLES_EXPANDED; do
        extra_flag_path="${extra_flag_path/#\~/$HOME}"
        [[ -z "$extra_flag_path" ]] && continue
        if [[ -f "$extra_flag_path" ]]; then
            EXTRA_FLAG_PATHS+=("$extra_flag_path")
        else
            echo "[run-moon-selfcal] WARNING: extra flag table not found, skipping: $extra_flag_path"
        fi
    done
fi

if (( ${#EXTRA_FLAG_PATHS[@]} > 0 )); then
    if (( ${#EXTRA_FLAG_PATHS[@]} == 1 )); then
        FLAGGED_UVFITS="${_UVFITS_BASE}_additional_flags_from_clustering.uvfits"
    else
        FLAG_TAG="$($PYTHON - "${EXTRA_FLAG_PATHS[@]}" <<'PY'
import re
import sys
from pathlib import Path

tags = []
for p in sys.argv[1:]:
    s = Path(p).stem.lower()
    s = re.sub(r'[^a-z0-9._-]+', '_', s)
    s = re.sub(r'_+', '_', s).strip('_')
    if s:
        tags.append(s)
joined = '__'.join(tags) if tags else 'flags'
if len(joined) > 120:
    joined = joined[:120].rstrip('_')
print(joined)
PY
        )"
        FLAGGED_UVFITS="${_UVFITS_BASE}_additional_flags_from_${FLAG_TAG}.uvfits"
    fi

    if [[ -f "$FLAGGED_UVFITS" ]]; then
        echo "[run-moon-selfcal] Reusing existing pre-flagged UVFITS: $FLAGGED_UVFITS"
        UVFITS_FOR_SELFCAL="$FLAGGED_UVFITS"
    else
        echo "[run-moon-selfcal] Producing pre-flagged UVFITS via visSplit.py ..."
        echo "[run-moon-selfcal]   input : $UVFITS_EXPANDED"
        echo "[run-moon-selfcal]   flags :"
        for extra_flag_path in "${EXTRA_FLAG_PATHS[@]}"; do
            echo "[run-moon-selfcal]     - $extra_flag_path"
        done
        echo "[run-moon-selfcal]   output: $FLAGGED_UVFITS"
        "$PYTHON" src/visSplit.py \
            --config   preprocess_ugmrt.cfg \
            --fits     "$UVFITS_EXPANDED" \
            --source   "$SCAN" \
            --chan-range 0 127 \
            --stokes   RR LL \
            --flag-tables "${EXTRA_FLAG_PATHS[@]}" \
            --out      "$FLAGGED_UVFITS" \
            --overwrite
        echo "[run-moon-selfcal] Pre-flagged UVFITS written: $FLAGGED_UVFITS"
        UVFITS_FOR_SELFCAL="$FLAGGED_UVFITS"
    fi
fi

CMD_FILE="$LOG_DIR/run_moon_selfcal_${SCAN_LOWER}_${RUN_TS}.cmd"
LOG_FILE="$LOG_DIR/run_moon_selfcal_${SCAN_LOWER}_${RUN_TS}.log"

# shellcheck disable=SC2206
INTEGRATION_ARGS=( $INTEGRATIONS )

CMD=(
    "$PYTHON" "$REPO_ROOT/src/moon_selfcal_dev.py"
    --scan "$SCAN"
    --uvfits "$UVFITS_FOR_SELFCAL"
    --build-index-if-missing
    --outdir "$OUTDIR"
    --integrations "${INTEGRATION_ARGS[@]}"
    --stack-size "$STACK_SIZE"
    --cell "$CELL"
    --imsize "$IMSIZE"
    --stokes "$STOKES"
    --weighting "$WEIGHTING"
    --robust "$ROBUST"
    --deconvolver "$DECONVOLVER"
    --mask-radius-arcmin "$MASK_RADIUS_ARCMIN"
    --niter-per-cycle "$NITER_PER_CYCLE"
    --uvmin-per-cycle "$UVMIN_PER_CYCLE"
    --uvmax-per-cycle "$UVMAX_PER_CYCLE"
    --uvmin-cal-per-cycle "$UVMIN_CAL_PER_CYCLE"
    --uvmax-cal-per-cycle "$UVMAX_CAL_PER_CYCLE"
    --scales-per-cycle "$SCALES_PER_CYCLE"
    --niter-final "$NITER_FINAL"
    --threshold "$THRESHOLD"
    --cycleniter "$CYCLENITER"
    --minsnr "$MINSNR"
    --loop-gain "$LOOP_GAIN"
    --negativethreshold "$NEGATIVETHRESHOLD"
    --solmode "$SOLMODE"
    --refant "$REFANT"
    --refantmode "$REFANTMODE"
)

if [[ "$NO_MASK" == "1" ]]; then
    CMD+=("--no-mask")
fi

if [[ "$USE_TCLEAN_PHASECENTER" == "1" ]]; then
    CMD+=("--use-tclean-phasecenter")
fi

if [[ -n "$OVERWRITE" ]]; then
    CMD+=("$OVERWRITE")
fi

{
    echo "# timestamp=$RUN_TS"
    echo "# cwd=$PWD"
    printf '%q ' "${CMD[@]}"
    printf '\n'
} > "$CMD_FILE"

echo "[run-moon-selfcal] cmd       : $CMD_FILE"
echo "[run-moon-selfcal] log       : $LOG_FILE"

"${CMD[@]}" 2>&1 | tee "$LOG_FILE"

DESTRIPE_RAN=0
if [[ "$RUN_DESTRIPE_AFTER_SELFCAL" == "1" ]]; then
    echo "[run-moon-selfcal] Running optional destriping stage..."
    D_CMD=(
        "$PYTHON" "$REPO_ROOT/src/stack_moon_snapshots.py"
        --selfcal-dir "$OUTDIR"
        --output "$DESTRIPE_STACK_OUTPUT"
        --glob "$DESTRIPE_GLOB_PATTERN"
        --destripe-iters "$DESTRIPE_ITERS"
        --destripe-debug-dir "$DESTRIPE_DEBUG_DIR"
        --moon-mask-center "$DESTRIPE_MOON_MASK_CENTER"
        --destripe-write-dir "$DESTRIPE_WRITE_DIR"
        --destripe-band-sigma-pix "$DESTRIPE_BAND_SIGMA_PIX"
        --destripe-band-depth "$DESTRIPE_BAND_DEPTH"
    )
    if [[ "$DESTRIPE_BAND_MODE" == "1" ]]; then
        D_CMD+=(--destripe-band-mode)
    fi

    D_CMD_FILE="$LOG_DIR/run_moon_destripe_${SCAN_LOWER}_${RUN_TS}.cmd"
    D_LOG_FILE="$LOG_DIR/run_moon_destripe_${SCAN_LOWER}_${RUN_TS}.log"
    {
        echo "# timestamp=$RUN_TS"
        echo "# cwd=$PWD"
        printf '%q ' "${D_CMD[@]}"
        printf '\n'
    } > "$D_CMD_FILE"

    echo "[run-moon-selfcal] destripe cmd : $D_CMD_FILE"
    echo "[run-moon-selfcal] destripe log : $D_LOG_FILE"

    if "${D_CMD[@]}" 2>&1 | tee "$D_LOG_FILE"; then
        DESTRIPE_RAN=1
        echo "[run-moon-selfcal] Optional destriping completed."
    else
        echo "[run-moon-selfcal] WARNING: optional destriping failed; continuing without destriped products."
    fi
else
    echo "[run-moon-selfcal] Optional destriping disabled (RUN_DESTRIPE_AFTER_SELFCAL=$RUN_DESTRIPE_AFTER_SELFCAL)."
fi

if [[ "$MAKE_MOVIE_AFTER_SELFCAL" == "1" ]]; then
    echo "[run-moon-selfcal] Running optional movie generation stage..."
    MOVIE_SCALE_DIRS="$OUTDIR"
    if [[ "$DESTRIPE_RAN" == "1" ]]; then
        MOVIE_SCALE_DIRS="$OUTDIR,$DESTRIPE_WRITE_DIR"
    fi

    if SCAN="$SCAN" \
       SELFCAL_DIR="$OUTDIR" \
       LOG_DIR="$LOG_DIR" \
       FPS="$MOVIE_FPS" \
       CMAP="$MOVIE_CMAP" \
       PERCENTILE_LOW="$MOVIE_PERCENTILE_LOW" \
       PERCENTILE_HIGH="$MOVIE_PERCENTILE_HIGH" \
       SCALE_DIRS="$MOVIE_SCALE_DIRS" \
       PYTHON_CMD="$PYTHON" \
    bash "$REPO_ROOT/bin/run_moon_selfcal_movie_dev.sh"; then
        echo "[run-moon-selfcal] Optional movie generation completed."
    else
        echo "[run-moon-selfcal] WARNING: optional movie generation failed; continuing with selfcal outputs only."
    fi
else
    echo "[run-moon-selfcal] Optional movie generation disabled (MAKE_MOVIE_AFTER_SELFCAL=$MAKE_MOVIE_AFTER_SELFCAL)."
fi

if [[ "$DESTRIPE_RAN" == "1" && "$MAKE_DESTRIPED_MOVIE" == "1" ]]; then
    echo "[run-moon-selfcal] Running optional destriped movie generation stage..."
    if SCAN="$SCAN" \
       SELFCAL_DIR="$DESTRIPE_WRITE_DIR" \
       LOG_DIR="$LOG_DIR" \
       FPS="$MOVIE_FPS" \
       CMAP="$MOVIE_CMAP" \
       PERCENTILE_LOW="$MOVIE_PERCENTILE_LOW" \
       PERCENTILE_HIGH="$MOVIE_PERCENTILE_HIGH" \
       SCALE_DIRS="$OUTDIR,$DESTRIPE_WRITE_DIR" \
       FRAMES_DIR="$OUTDIR/movie_frames_destriped" \
       OUT_MP4="$OUTDIR/${SCAN_LOWER}_selfcal_movie_destriped.mp4" \
       OUT_GIF="$OUTDIR/${SCAN_LOWER}_selfcal_movie_destriped.gif" \
       OUT_MOV="$OUTDIR/${SCAN_LOWER}_selfcal_movie_destriped.mov" \
       PYTHON_CMD="$PYTHON" \
    bash "$REPO_ROOT/bin/run_moon_selfcal_movie_dev.sh"; then
        echo "[run-moon-selfcal] Optional destriped movie generation completed."
    else
        echo "[run-moon-selfcal] WARNING: optional destriped movie generation failed."
    fi
fi

if [[ "$DESTRIPE_RAN" == "1" && "$MAKE_COMPARE_MOVIE" == "1" ]]; then
    echo "[run-moon-selfcal] Running optional compare movie generation stage..."
    if SCAN="$SCAN" \
       SELFCAL_DIR="$OUTDIR" \
       COMPARE_DIR="$DESTRIPE_WRITE_DIR" \
       LEFT_LABEL="Before" \
       RIGHT_LABEL="Destriped" \
       LOG_DIR="$LOG_DIR" \
       FPS="$MOVIE_FPS" \
       CMAP="$MOVIE_CMAP" \
       PERCENTILE_LOW="$MOVIE_PERCENTILE_LOW" \
       PERCENTILE_HIGH="$MOVIE_PERCENTILE_HIGH" \
       SCALE_DIRS="$OUTDIR,$DESTRIPE_WRITE_DIR" \
       FRAMES_DIR="$OUTDIR/movie_frames_compare" \
       OUT_MP4="$OUTDIR/${SCAN_LOWER}_selfcal_movie_before_after_compare.mp4" \
       OUT_GIF="$OUTDIR/${SCAN_LOWER}_selfcal_movie_before_after_compare.gif" \
       OUT_MOV="$OUTDIR/${SCAN_LOWER}_selfcal_movie_before_after_compare.mov" \
       PYTHON_CMD="$PYTHON" \
    bash "$REPO_ROOT/bin/run_moon_selfcal_movie_dev.sh"; then
        echo "[run-moon-selfcal] Optional compare movie generation completed."
    else
        echo "[run-moon-selfcal] WARNING: optional compare movie generation failed."
    fi
fi

echo ""
echo "Log: $LOG_FILE"
echo "Cmd: $CMD_FILE"
echo "Outputs: $OUTDIR"
