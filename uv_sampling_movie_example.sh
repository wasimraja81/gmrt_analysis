#!/usr/bin/env bash

set -euo pipefail

FITS_DIR=~/DATA/gmrt_40_014/work/split/moon
PATTERN="moon*_calibrated.uvfits"  # change to any glob, e.g. "3c468*_calibrated.uvfits"

OUTDIR=./diagnostics_out/uv_sampling_from_scratch
PRODUCTS=RR,LL
SAMPLE_FRAC=0.3
FPS=8
TIME_STEP=1
UV_GOOD_MARKER_SIZE=1.35
UV_FLAG_MARKER_SIZE=0.06

# Use all channels in this split Moon file (central 128 channels): 0..127.
CHAN_START=0
CHAN_END=127

mkdir -p "$OUTDIR"

python uv_sampling.py batch \
  --fits-dir "$FITS_DIR" \
  --pattern "$PATTERN" \
  --products "$PRODUCTS" \
  --sample-frac "$SAMPLE_FRAC" \
  --overlay-flags \
  --uv-good-marker-size "$UV_GOOD_MARKER_SIZE" \
  --uv-flag-marker-size "$UV_FLAG_MARKER_SIZE" \
  --chan-range "$CHAN_START" "$CHAN_END" \
  --outdir "$OUTDIR" \
  --fps "$FPS" \
  --time-step "$TIME_STEP" \
  --make-montage

echo "[uv-sampling-example] done. outputs under: $OUTDIR"
