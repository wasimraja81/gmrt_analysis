#!/usr/bin/env bash
set -euo pipefail

# test_radial_shell_stk0000.sh
#
# Single-stack radial-shell destriping trial for moon0520_stk0000_n10_final.fits.
# Edit the variables below if you want to try different parameters.
#
# Usage:
#   bash experimental/test_radial_shell_stk0000.sh

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python}"

SELFCAL_DIR="${SELFCAL_DIR:-$HOME/DATA/gmrt_40_014/work/casa_selfcal/moon0520_stk10/before_finals}"
RUN_ALL_STACKS="${RUN_ALL_STACKS:-1}"
STACK_ID="${STACK_ID:-0010}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/DATA/gmrt_40_014/work/casa_selfcal/moon0520_stk10}"

if [[ "$RUN_ALL_STACKS" == "1" ]]; then
    STACK_LABEL="${STACK_LABEL:-allstacks}"
    STACK_GLOB="${STACK_GLOB:-moon0520_stk*_final.fits}"
    STACK_BASENAME="${STACK_BASENAME:-moon0520_${STACK_LABEL}_shell_single_trial}"
    OUTPUT_FITS="${OUTPUT_FITS:-${OUTPUT_ROOT}/${STACK_BASENAME}.fits}"
    DEBUG_DIR="${DEBUG_DIR:-${OUTPUT_ROOT}/destripe_debug_shell_${STACK_LABEL}_single_trial}"
    WRITE_DIR="${WRITE_DIR:-${OUTPUT_ROOT}/destriped_finals_shell_${STACK_LABEL}_single_trial}"
else
    STACK_BASENAME="${STACK_BASENAME:-moon0520_stk${STACK_ID}_n10_final}"
    STACK_GLOB="${STACK_GLOB:-${STACK_BASENAME}.fits}"
    OUTPUT_FITS="${OUTPUT_FITS:-${OUTPUT_ROOT}/${STACK_BASENAME%_final}_shell_single_trial.fits}"
    DEBUG_DIR="${DEBUG_DIR:-${OUTPUT_ROOT}/destripe_debug_shell_stk${STACK_ID}_single_trial}"
    WRITE_DIR="${WRITE_DIR:-${OUTPUT_ROOT}/destriped_finals_shell_stk${STACK_ID}_single_trial}"
fi

DESTRIPE_ITERS="${DESTRIPE_ITERS:-20}"
SHELL_RADIUS_PIX="${SHELL_RADIUS_PIX:-2.56}"
SHELL_SIGMA_PIX="${SHELL_SIGMA_PIX:-20.0}"
MOON_MASK_RADIUS_ARCMIN="${MOON_MASK_RADIUS_ARCMIN:-20.0}"
FFT_MAX_KLAMBDA="${FFT_MAX_KLAMBDA:-3}"
FFT_SCALE_LOW="${FFT_SCALE_LOW:-5}"
FFT_SCALE_HIGH="${FFT_SCALE_HIGH:-99.5}"
ANIMATION_FPS="${ANIMATION_FPS:-2.0}"
SAVE_NPZ="${SAVE_NPZ:-0}"
REGISTRATION_MODE="${REGISTRATION_MODE:-derive}"

mkdir -p "$(dirname "$OUTPUT_FITS")" "$DEBUG_DIR" "$WRITE_DIR"

CMD=(
    "$PYTHON" "$REPO_ROOT/experimental/stack_moon_snapshots.py"
    --selfcal-dir "$SELFCAL_DIR"
    --glob "$STACK_GLOB"
    --output "$OUTPUT_FITS"
    --registration-mode "$REGISTRATION_MODE"
    --destripe-iters "$DESTRIPE_ITERS"
    --destripe-strategy radial-shell
    --destripe-shell-radius-pix "$SHELL_RADIUS_PIX"
    --destripe-shell-sigma-pix "$SHELL_SIGMA_PIX"
    --moon-mask-center ephem
    --moon-mask-radius-arcmin "$MOON_MASK_RADIUS_ARCMIN"
    --destripe-debug-dir "$DEBUG_DIR"
    --destripe-debug-fft-axis klambda
    --destripe-debug-fft-max-klambda "$FFT_MAX_KLAMBDA"
    --destripe-debug-fft-scale-mode per-iter
    --destripe-debug-fft-scale-low "$FFT_SCALE_LOW"
    --destripe-debug-fft-scale-high "$FFT_SCALE_HIGH"
    --destripe-debug-animation-fps "$ANIMATION_FPS"
    --destripe-write-dir "$WRITE_DIR"
)

if [[ "$SAVE_NPZ" == "1" ]]; then
    CMD+=(--destripe-save-npz)
fi

echo "[test-radial-shell] Running trial for $STACK_GLOB"
if [[ "$RUN_ALL_STACKS" == "1" ]]; then
    echo "[test-radial-shell] Mode      : all stacks"
else
    echo "[test-radial-shell] Mode      : single stack"
fi
echo "[test-radial-shell] STACK_ID   : $STACK_ID"
echo "[test-radial-shell] REG_MODE   : $REGISTRATION_MODE"
printf '[test-radial-shell] cmd: '
printf '%q ' "${CMD[@]}"
printf '\n'

cd "$REPO_ROOT"
"${CMD[@]}"

echo "[test-radial-shell] Final FITS: $OUTPUT_FITS"
echo "[test-radial-shell] Debug dir  : $DEBUG_DIR"
echo "[test-radial-shell] Write dir  : $WRITE_DIR"
