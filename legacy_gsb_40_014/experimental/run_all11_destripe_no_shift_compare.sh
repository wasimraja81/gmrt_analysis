#!/usr/bin/env bash
set -euo pipefail

# Reproducible workflow for moon0520 (all 11 stacks):
# 1) Destripe per-frame with radial-shell
# 2) Stack destriped frames with direct co-add (no shifts)
# 3) Compare post-destripe stack vs pre-destripe no-shift stack
#
# Usage:
#   bash experimental/run_all11_destripe_no_shift_compare.sh
#
# Optional overrides:
#   PYTHON=... OUTPUT_ROOT=... DESTRIPE_ITERS=... SHELL_RADIUS_PIX=... SHELL_SIGMA_PIX=...

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-$REPO_ROOT/gmrt/bin/python}"

# Input: 11 pre-destriped final FITS images
SELFCAL_DIR="${SELFCAL_DIR:-/Users/raj030/DATA/gmrt_40_014/work/casa_selfcal/moon0520_stk10_phasecenter}"
STACK_GLOB="${STACK_GLOB:-moon0520_stk*_n*_final.fits}"

# Output root for this experiment
OUTPUT_ROOT="${OUTPUT_ROOT:-/Users/raj030/DATA/gmrt_40_014/work/casa_selfcal/moon0520_stk10_stack_compare}"

# Pre-destripe no-shift stack (baseline for comparison)
PRE_STACK_FITS="${PRE_STACK_FITS:-$OUTPUT_ROOT/no_shift_direct_coadd/moon0520_stack_no_shift_direct_coadd.fits}"
PRE_PROV_DIR="${PRE_PROV_DIR:-$OUTPUT_ROOT/no_shift_direct_coadd/provenance}"

# Post-destripe no-shift stack (new output)
POST_STACK_FITS="${POST_STACK_FITS:-$OUTPUT_ROOT/moon0520_all11_phasecenter_no_shift_shell_single_trial.fits}"
POST_DEBUG_DIR="${POST_DEBUG_DIR:-$OUTPUT_ROOT/destripe_debug_shell_all11_phasecenter_no_shift_single_trial}"
POST_WRITE_DIR="${POST_WRITE_DIR:-$OUTPUT_ROOT/destriped_finals_shell_all11_phasecenter_no_shift_single_trial}"

# Radial-shell parameters (copied from your preferred trial)
DESTRIPE_ITERS="${DESTRIPE_ITERS:-20}"
SHELL_RADIUS_PIX="${SHELL_RADIUS_PIX:-2.56}"
SHELL_SIGMA_PIX="${SHELL_SIGMA_PIX:-20.0}"
MOON_MASK_RADIUS_ARCMIN="${MOON_MASK_RADIUS_ARCMIN:-20.0}"
FFT_MAX_KLAMBDA="${FFT_MAX_KLAMBDA:-3}"
FFT_SCALE_LOW="${FFT_SCALE_LOW:-5}"
FFT_SCALE_HIGH="${FFT_SCALE_HIGH:-99.5}"
ANIMATION_FPS="${ANIMATION_FPS:-2.0}"
SAVE_NPZ="${SAVE_NPZ:-0}"

mkdir -p "$(dirname "$POST_STACK_FITS")" "$PRE_PROV_DIR" "$POST_DEBUG_DIR" "$POST_WRITE_DIR"

echo "[workflow] repo          : $REPO_ROOT"
echo "[workflow] python        : $PYTHON"
echo "[workflow] selfcal_dir   : $SELFCAL_DIR"
echo "[workflow] stack_glob    : $STACK_GLOB"
echo "[workflow] pre_stack     : $PRE_STACK_FITS"
echo "[workflow] post_stack    : $POST_STACK_FITS"

if [[ ! -f "$PRE_STACK_FITS" ]]; then
  echo "[step 0] Pre-destripe baseline stack missing; creating it (no-shift mean)."
  "$PYTHON" "$REPO_ROOT/experimental/stack_moon_snapshots.py" \
    --selfcal-dir "$SELFCAL_DIR" \
    --glob "$STACK_GLOB" \
    --output "$PRE_STACK_FITS" \
    --registration-mode none \
    --method mean \
    --provenance-dir "$PRE_PROV_DIR"
else
  echo "[step 0] Using existing pre-destripe baseline stack."
fi

echo "[step 1+2] Destripe all 11 + no-shift mean stack"
CMD=(
  "$PYTHON" "$REPO_ROOT/experimental/stack_moon_snapshots.py"
  --selfcal-dir "$SELFCAL_DIR"
  --glob "$STACK_GLOB"
  --output "$POST_STACK_FITS"
  --registration-mode none
  --method mean
  --destripe-iters "$DESTRIPE_ITERS"
  --destripe-strategy radial-shell
  --destripe-shell-radius-pix "$SHELL_RADIUS_PIX"
  --destripe-shell-sigma-pix "$SHELL_SIGMA_PIX"
  --moon-mask-center ephem
  --moon-mask-radius-arcmin "$MOON_MASK_RADIUS_ARCMIN"
  --destripe-debug-dir "$POST_DEBUG_DIR"
  --destripe-debug-fft-axis klambda
  --destripe-debug-fft-max-klambda "$FFT_MAX_KLAMBDA"
  --destripe-debug-fft-scale-mode per-iter
  --destripe-debug-fft-scale-low "$FFT_SCALE_LOW"
  --destripe-debug-fft-scale-high "$FFT_SCALE_HIGH"
  --destripe-debug-animation-fps "$ANIMATION_FPS"
  --destripe-write-dir "$POST_WRITE_DIR"
  --provenance-dir "$(dirname "$POST_STACK_FITS")/provenance"
)
if [[ "$SAVE_NPZ" == "1" ]]; then
  CMD+=(--destripe-save-npz)
fi
printf '[workflow] cmd: '
printf '%q ' "${CMD[@]}"
printf '\n'
"${CMD[@]}"

echo "[step 3] Compare post-destripe vs pre-destripe co-add"
"$PYTHON" - <<'PY' "$PRE_STACK_FITS" "$POST_STACK_FITS"
import sys
from pathlib import Path
import numpy as np
from astropy.io import fits

pre_path = Path(sys.argv[1])
post_path = Path(sys.argv[2])

with fits.open(pre_path) as h:
    pre = np.squeeze(h[0].data).astype(np.float64)
with fits.open(post_path) as h:
    post = np.squeeze(h[0].data).astype(np.float64)

if pre.shape != post.shape:
    raise SystemExit(f"ERROR: shape mismatch pre={pre.shape} post={post.shape}")

diff = post - pre
pre_rms = float(np.sqrt(np.mean(pre**2)))
post_rms = float(np.sqrt(np.mean(post**2)))
diff_rms = float(np.sqrt(np.mean(diff**2)))
max_abs_diff = float(np.max(np.abs(diff)))
frac = diff_rms / (pre_rms + 1e-30)

print('[compare] pre_stack :', pre_path)
print('[compare] post_stack:', post_path)
print('[compare] shape     :', pre.shape)
print('[compare] pre_rms   : %.6e' % pre_rms)
print('[compare] post_rms  : %.6e' % post_rms)
print('[compare] diff_rms  : %.6e' % diff_rms)
print('[compare] max|diff| : %.6e' % max_abs_diff)
print('[compare] frac_diff : %.6e  (diff_rms/pre_rms)' % frac)
PY

echo "[done] post-destripe stack : $POST_STACK_FITS"
echo "[done] pre-destripe stack  : $PRE_STACK_FITS"
echo "[done] debug dir           : $POST_DEBUG_DIR"
echo "[done] destriped frames    : $POST_WRITE_DIR"
