#!/usr/bin/env bash
# run_3c468.1_selfcal_dev.sh
#
# Per-scan imaging + phase-only selfcal for 3C468.1 (fixed-target mode).
#
# 3C468.1 has one source name but nine separate scan blocks separated by time
# gaps >24 s.  Run one imaging pass per scan block by supplying the correct
# integration range via INTEGRATIONS and SCAN_TAG below.
#
# Scan blocks (0-based row indices in the 321-integration UVFITS):
#   scan01: 0..12    (13 integrations)
#   scan02: 13..51   (39 integrations)
#   scan03: 52..90   (39 integrations)
#   scan04: 91..129  (39 integrations)
#   scan05: 130..167 (38 integrations)
#   scan06: 168..206 (39 integrations)
#   scan07: 207..244 (38 integrations)
#   scan08: 245..282 (38 integrations)
#   scan09: 283..320 (38 integrations)
#
# Usage (single scan):
#   bash bin/run_3c468.1_selfcal_dev.sh
#
# Select a scan by overriding SCAN_TAG and INTEGRATIONS from the environment:
#   SCAN_TAG=scan02 INTEGRATIONS="$(seq 13 51)" bash bin/run_3c468.1_selfcal_dev.sh
#
# Run all nine scans sequentially:
#   for block in "scan01:0:12" "scan02:13:51" "scan03:52:90" \
#                "scan04:91:129" "scan05:130:167" "scan06:168:206" \
#                "scan07:207:244" "scan08:245:282" "scan09:283:320"; do
#       tag="${block%%:*}"; rest="${block#*:}"; lo="${rest%%:*}"; hi="${rest#*:}"
#       SCAN_TAG="$tag" INTEGRATIONS="$(seq "$lo" "$hi")" \
#           OUTDIR=~/DATA/gmrt_40_014/work/casa_selfcal/3c468.1_"$tag" \
#           bash bin/run_3c468.1_selfcal_dev.sh
#   done

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_GMRT_PYTHON="$REPO_ROOT/gmrt/bin/python"
if [[ -x "$DEFAULT_GMRT_PYTHON" ]]; then
    PYTHON="${PYTHON_CMD:-$DEFAULT_GMRT_PYTHON}"
else
    PYTHON="${PYTHON_CMD:-python}"
fi

DRY_RUN=0

usage() {
    cat <<'USAGE'
Usage: bash bin/run_3c468.1_selfcal_dev.sh [--dry-run]

Environment overrides:
  SCAN_TAG=scan01..scan09   Select scan block (default: scan01)
  INTEGRATIONS="..."        Override the integration list
  STACK_SIZE=N|auto         Stack size per output image (default: auto)
  OUTDIR=...                Override output directory
  UVFITS=...                Override input UVFITS path

Examples:
  SCAN_TAG=scan05 STACK_SIZE=1 bash bin/run_3c468.1_selfcal_dev.sh
  bash bin/run_3c468.1_selfcal_dev.sh --dry-run
USAGE
}

case "${1:-}" in
    --dry-run)
        DRY_RUN=1
        shift
        ;;
    --help|-h)
        usage
        exit 0
        ;;
esac

# ── Source / scan ──────────────────────────────────────────────────────────────
SCAN="3C468.1"
# Human-readable tag used in output directory and log names.
# Override to select a different scan block.
SCAN_TAG="${SCAN_TAG:-scan01}"

case "$SCAN_TAG" in
    scan0[1-9])
        :
        ;;
    scan[1-9])
        SCAN_TAG="scan0${SCAN_TAG#scan}"
        ;;
    [1-9])
        SCAN_TAG="scan0${SCAN_TAG}"
        ;;
    0[1-9])
        SCAN_TAG="scan${SCAN_TAG}"
        ;;
    *)
        ;;
esac

# ── UVFITS (single branch) ───────────────────────────────────────────────────
# Default input for current runs.
UVFITS="${UVFITS:-$HOME/DATA/gmrt_40_014/work/split/3c468.1/3c468.1_primary_secondary_calibrated_flagged.uvfits}"
UVFITS_EXPANDED="${UVFITS/#\~/$HOME}"
if [[ ! -f "$UVFITS_EXPANDED" ]]; then
    echo "[run-3c468.1-selfcal] ERROR: UVFITS not found: $UVFITS_EXPANDED" >&2
    exit 1
fi

# ── Output / logging ──────────────────────────────────────────────────────────
OUTDIR="${OUTDIR:-}"
OUTDIR_BASE="${OUTDIR_BASE:-$HOME/DATA/gmrt_40_014/work/casa_selfcal}"
LOG_DIR="${LOG_DIR:-$HOME/DATA/gmrt_40_014/work/logs}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${RUN_ID:-${RUN_TS}_p$$_r${RANDOM}}"
MAKE_MOVIE_AFTER_SELFCAL="${MAKE_MOVIE_AFTER_SELFCAL:-1}"
MOVIE_FPS="${MOVIE_FPS:-4}"
MOVIE_CMAP="${MOVIE_CMAP:-magma}"
MOVIE_PERCENTILE_LOW="${MOVIE_PERCENTILE_LOW:-5}"
MOVIE_PERCENTILE_HIGH="${MOVIE_PERCENTILE_HIGH:-99.5}"

# ── Integration selection ──────────────────────────────────────────────────────
# Default: derive integrations from SCAN_TAG (one of scan01..scan09).
# You can still override explicitly via INTEGRATIONS.
INTEGRATIONS="${INTEGRATIONS:-}"

if [[ -z "$INTEGRATIONS" ]]; then
    case "$SCAN_TAG" in
        scan01) INTEGRATIONS="$(seq 0 12)" ;;
        scan02) INTEGRATIONS="$(seq 13 51)" ;;
        scan03) INTEGRATIONS="$(seq 52 90)" ;;
        scan04) INTEGRATIONS="$(seq 91 129)" ;;
        scan05) INTEGRATIONS="$(seq 130 167)" ;;
        scan06) INTEGRATIONS="$(seq 168 206)" ;;
        scan07) INTEGRATIONS="$(seq 207 244)" ;;
        scan08) INTEGRATIONS="$(seq 245 282)" ;;
        scan09) INTEGRATIONS="$(seq 283 320)" ;;
        *)
            echo "[run-3c468.1-selfcal] ERROR: unknown SCAN_TAG='$SCAN_TAG'. " \
                 "Expected one of scan01..scan09 or pass INTEGRATIONS explicitly." >&2
            exit 2
            ;;
    esac
fi

# ── Stacking ───────────────────────────────────────────────────────────────────
# Default: one image per scan (stack-size = all selected integrations).
# Set STACK_SIZE=10 for time-evolution stacks, or any positive integer.
STACK_SIZE="${STACK_SIZE:-auto}"

# ── Image geometry ────────────────────────────────────────────────────────────
CELL="${CELL:-1.5arcsec}"       # beam ~7.4 arcsec; cell = beam/5
IMSIZE="${IMSIZE:-1024}"        # 1024 × 1.5" = 25.6 arcmin
STOKES="${STOKES:-I}"
WEIGHTING="${WEIGHTING:-briggs}"
ROBUST="${ROBUST:--0.5}"
DECONVOLVER="${DECONVOLVER:-hogbom}"

# ── Mask ─────────────────────────────────────────────────────────────────────
# 3C468.1 is a compact calibrator sitting at the phase centre.
# No CLEAN mask needed for the current 3C468.1 workflow.
MASK_RADIUS_ARCMIN="${MASK_RADIUS_ARCMIN:-5.0}"
NO_MASK="${NO_MASK:-1}"
USE_TCLEAN_PHASECENTER="${USE_TCLEAN_PHASECENTER:-0}"

# ── Per-cycle selfcal parameters (comma-separated) ────────────────────────────
# Centre freq 322.67 MHz → 1 kλ = 929 m.
# 3C468.1 is bright (~10 Jy); good SNR on all baselines — use full uv range.
NITER_PER_CYCLE="${NITER_PER_CYCLE:-300,1000,2000,6000}"
UVMIN_PER_CYCLE="${UVMIN_PER_CYCLE:-,,,}"          # no lower limit
UVMAX_PER_CYCLE="${UVMAX_PER_CYCLE:-,,,}"          # no upper limit
UVMIN_CAL_PER_CYCLE="${UVMIN_CAL_PER_CYCLE:-,,,}"
UVMAX_CAL_PER_CYCLE="${UVMAX_CAL_PER_CYCLE:-,,,}"
SCALES_PER_CYCLE="${SCALES_PER_CYCLE:-0|0|0|0}"

# ── Final image ────────────────────────────────────────────────────────────────
NITER_FINAL="${NITER_FINAL:-10000}"
THRESHOLD="${THRESHOLD:-0mJy}"
CYCLENITER="${CYCLENITER:-50}"
NEGATIVETHRESHOLD="${NEGATIVETHRESHOLD:-0.0}"

# ── Selfcal ────────────────────────────────────────────────────────────────────
MINSNR="${MINSNR:-3.0}"
LOOP_GAIN="${LOOP_GAIN:-0.1}"
SOLMODE="${SOLMODE:-L1}"
REFANT="${REFANT:-1}"
REFANTMODE="${REFANTMODE:-flex}"

# ── Misc ──────────────────────────────────────────────────────────────────────
OVERWRITE="${OVERWRITE:---overwrite}"

# ─────────────────────────────────────────────────────────────────────────────
cd "$REPO_ROOT"
mkdir -p "$LOG_DIR"

CMD_FILE="$LOG_DIR/run_3c468.1_selfcal_${SCAN_TAG}_${RUN_ID}.cmd"
LOG_FILE="$LOG_DIR/run_3c468.1_selfcal_${SCAN_TAG}_${RUN_ID}.log"

# shellcheck disable=SC2206
INTEGRATION_ARGS=( $INTEGRATIONS )

if [[ "$STACK_SIZE" == "auto" ]]; then
    STACK_SIZE="${#INTEGRATION_ARGS[@]}"
fi

if ! [[ "$STACK_SIZE" =~ ^[0-9]+$ ]] || [[ "$STACK_SIZE" == "0" ]]; then
    echo "[run-3c468.1-selfcal] ERROR: STACK_SIZE must be a positive integer or 'auto'. Got: $STACK_SIZE" >&2
    exit 2
fi

if [[ -z "$OUTDIR" ]]; then
    OUTDIR="$OUTDIR_BASE/3c468.1_${SCAN_TAG}_stk${STACK_SIZE}"
fi

INDEX_PATH="${INDEX:-$UVFITS_EXPANDED.row_index_cache.npz}"
INTEGRATIONS_COMPACT="${INTEGRATION_ARGS[*]}"

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[run-3c468.1-selfcal] DRY RUN"
    echo "[run-3c468.1-selfcal] scan_tag      : $SCAN_TAG"
    echo "[run-3c468.1-selfcal] integrations  : ${#INTEGRATION_ARGS[@]} values -> $INTEGRATIONS_COMPACT"
    echo "[run-3c468.1-selfcal] stack_size    : $STACK_SIZE"
    echo "[run-3c468.1-selfcal] uvfits        : $UVFITS_EXPANDED"
    echo "[run-3c468.1-selfcal] outdir        : $OUTDIR"
    echo "[run-3c468.1-selfcal] log_dir       : $LOG_DIR"
    echo "[run-3c468.1-selfcal] movie         : ${MAKE_MOVIE_AFTER_SELFCAL}"
    echo "[run-3c468.1-selfcal] image size    : ${IMSIZE}"
    echo "[run-3c468.1-selfcal] cell          : ${CELL}"
    echo "[run-3c468.1-selfcal] loop_gain     : ${LOOP_GAIN}"
    echo "[run-3c468.1-selfcal] no_mask       : ${NO_MASK}"
    echo "[run-3c468.1-selfcal] tclean_phase  : $( [[ "$USE_TCLEAN_PHASECENTER" == "1" ]] && echo enabled || echo disabled ) (Force tclean to image at specified direction, otherwise use MS/native phase center direction)"
    echo "[run-3c468.1-selfcal] exists uvfits : $( [[ -f "$UVFITS_EXPANDED" ]] && echo yes || echo no )"
    echo "[run-3c468.1-selfcal] index cache   : $INDEX_PATH"
    echo "[run-3c468.1-selfcal] exists index  : $( [[ -f "$INDEX_PATH" ]] && echo yes || echo no )"
    echo "[run-3c468.1-selfcal] runner exists : $( [[ -f "$SCRIPT_DIR/run_3c468.1_selfcal_dev.sh" ]] && echo yes || echo no )"
    echo "[run-3c468.1-selfcal] engine exists  : $( [[ -f "$REPO_ROOT/src/gmrt_selfcal_dev.py" ]] && echo yes || echo no )"
    echo "[run-3c468.1-selfcal] outdir exists : $( [[ -d "$OUTDIR" ]] && echo yes || echo no ) (will be created if missing)"
    exit 0
fi

CMD=(
    "$PYTHON" "$REPO_ROOT/src/gmrt_selfcal_dev.py"
    --target-mode  fixed-target
    --scan         "$SCAN"
    --uvfits       "$UVFITS_EXPANDED"
    --build-index-if-missing
    --run-id       "$RUN_ID"
    --launcher-log-path "$LOG_FILE"
    --launcher-cmd-path "$CMD_FILE"
    --outdir       "$OUTDIR"
    --integrations "${INTEGRATION_ARGS[@]}"
    --stack-size   "$STACK_SIZE"
    --cell         "$CELL"
    --imsize       "$IMSIZE"
    --stokes       "$STOKES"
    --weighting    "$WEIGHTING"
    --robust       "$ROBUST"
    --deconvolver  "$DECONVOLVER"
    --mask-radius-arcmin "$MASK_RADIUS_ARCMIN"
    --niter-per-cycle    "$NITER_PER_CYCLE"
    --uvmin-per-cycle    "$UVMIN_PER_CYCLE"
    --uvmax-per-cycle    "$UVMAX_PER_CYCLE"
    --uvmin-cal-per-cycle "$UVMIN_CAL_PER_CYCLE"
    --uvmax-cal-per-cycle "$UVMAX_CAL_PER_CYCLE"
    --scales-per-cycle   "$SCALES_PER_CYCLE"
    --niter-final        "$NITER_FINAL"
    --threshold          "$THRESHOLD"
    --cycleniter         "$CYCLENITER"
    --negativethreshold  "$NEGATIVETHRESHOLD"
    --minsnr             "$MINSNR"
    --loop-gain          "$LOOP_GAIN"
    --solmode            "$SOLMODE"
    --refant             "$REFANT"
    --refantmode         "$REFANTMODE"
)

[[ "$NO_MASK"              == "1" ]] && CMD+=("--no-mask")
[[ "$USE_TCLEAN_PHASECENTER" == "1" ]] && CMD+=("--use-tclean-phasecenter")
[[ -n "$OVERWRITE" ]] && CMD+=("$OVERWRITE")

{
    echo "# timestamp=$RUN_TS"
    echo "# run_id=$RUN_ID"
    echo "# scan_tag=$SCAN_TAG"
    echo "# cwd=$PWD"
    printf '%q ' "${CMD[@]}"
    printf '\n'
} > "$CMD_FILE"

echo "[run-3c468.1-selfcal] scan_tag : $SCAN_TAG"
echo "[run-3c468.1-selfcal] run_id   : $RUN_ID"
echo "[run-3c468.1-selfcal] uvfits   : $UVFITS_EXPANDED"
echo "[run-3c468.1-selfcal] outdir   : $OUTDIR"
echo "[run-3c468.1-selfcal] cmd      : $CMD_FILE"
echo "[run-3c468.1-selfcal] log      : $LOG_FILE"

"${CMD[@]}" 2>&1 | tee "$LOG_FILE"

# ── Optional movie generation ─────────────────────────────────────────────────
if [[ "$MAKE_MOVIE_AFTER_SELFCAL" == "1" ]]; then
    echo "[run-3c468.1-selfcal] Running optional movie generation stage..."
    if SCAN="$SCAN" \
       SELFCAL_DIR="$OUTDIR" \
       LOG_DIR="$LOG_DIR" \
       FPS="$MOVIE_FPS" \
       CMAP="$MOVIE_CMAP" \
       PERCENTILE_LOW="$MOVIE_PERCENTILE_LOW" \
       PERCENTILE_HIGH="$MOVIE_PERCENTILE_HIGH" \
       SCALE_DIRS="$OUTDIR" \
       PYTHON_CMD="$PYTHON" \
    bash "$REPO_ROOT/bin/run_moon_selfcal_movie_dev.sh"; then
        echo "[run-3c468.1-selfcal] Optional movie generation completed."
    else
        echo "[run-3c468.1-selfcal] WARNING: optional movie generation failed; continuing with selfcal outputs only."
    fi
else
    echo "[run-3c468.1-selfcal] Optional movie generation disabled (MAKE_MOVIE_AFTER_SELFCAL=$MAKE_MOVIE_AFTER_SELFCAL)."
fi

echo ""
echo "Log: $LOG_FILE"
echo "Cmd: $CMD_FILE"
echo "Outputs: $OUTDIR"
