#!/usr/bin/env bash
# run_moon_selfcal_dev.sh
#
# Tuning script for moon per-integration imaging + phase-only selfcal.
# Edit the parameters below and re-run to experiment.
#
# Usage:
#   bash experimental/run_moon_selfcal_dev.sh
#
# To test a different scan, change SCAN and UVFITS below.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON_CMD:-python}"

# ── Data ──────────────────────────────────────────────────────────────────────
SCAN="MOON0520"

# Use primary+secondary calibrated data:
# Last 3C468.1 secondary scan ended ~15 min before MOON0520 — making
# primary+secondary the best-calibrated starting point for all moon scans.
UVFITS=~/DATA/gmrt_40_014/work/split/moon/moon0520_primary_secondary_calibrated_flagged.uvfits

# Row index cache is derived automatically from UVFITS path
# (<uvfits>.row_index_cache.npz). Override with --index if needed.
# INDEX=~/DATA/gmrt_40_014/work/split/moon/moon0520_primary_secondary_calibrated_flagged.uvfits.row_index_cache.npz

OUTDIR=~/DATA/gmrt_40_014/work/casa_selfcal/moon0520_dev

# ── Integration selection ─────────────────────────────────────────────────────
# 0-based indices into the scan (MOON0520 has 113 integrations, 0..112).
# 5 evenly spaced integrations including both endpoints: 0, 28, 56, 84, 112.
INTEGRATIONS="0 28 56 84 112"

# ── Image geometry ────────────────────────────────────────────────────────────
CELL="1.5arcsec"          # beam ~7.4 arcsec, cell = beam/5
IMSIZE=2048               # 2048 * 1.5" = 51 arcmin — covers primary beam + drift
STOKES="I"
WEIGHTING="briggs"
ROBUST=0.0
DECONVOLVER="multiscale"
# ── Moon mask ─────────────────────────────────────────────────────────────────
# Moon angular diameter ~30' — mask radius 20' gives disk + margin.
MASK_RADIUS_ARCMIN=20.0

# ── Per-cycle selfcal parameters (comma-separated; one value per cycle) ────────
# Cycle 1: restricted uv-range to build initial model from bright, compact emission
#   0.5 kλ = 465 m min baseline — avoids RFI/diffuse confusion.
# Cycle 2: relaxed uv-range to include large-scale lunar structure
#   0.12 kλ = ~112 m min baseline.
NITER_PER_CYCLE="100,300"       # tclean niter per selfcal cycle
UVMIN_PER_CYCLE="0.5,0.12"     # kλ uvmin per cycle (empty element = no limit)
UVMAX_PER_CYCLE=","            # kλ uvmax per cycle (empty = no upper limit)
SCALES_PER_CYCLE="0,5,15|0,5,15,45"  # multiscale pixels per cycle (pipe-separated)

# ── Final (post-selfcal) image ─────────────────────────────────────────────────
NITER_FINAL=500     # final image: deeper clean (uses last cycle's uv/scale params)
THRESHOLD="0mJy"
CYCLENITER=100

# ── Selfcal ───────────────────────────────────────────────────────────────────
MINSNR=3.0          # gaincal minsnr

# ── Misc ──────────────────────────────────────────────────────────────────────
OVERWRITE="--overwrite"   # set to "" to reuse existing MS / skip already-done ints

# ─────────────────────────────────────────────────────────────────────────────
cd "$REPO_ROOT"

# shellcheck disable=SC2086
"$PYTHON" experimental/moon_selfcal_dev.py \
    --scan        "$SCAN" \
    --uvfits      "$UVFITS" \
    --outdir      "$OUTDIR" \
    --integrations $INTEGRATIONS \
    --cell        "$CELL" \
    --imsize      "$IMSIZE" \
    --stokes      "$STOKES" \
    --weighting   "$WEIGHTING" \
    --robust      "$ROBUST" \
    --deconvolver "$DECONVOLVER" \
    --mask-radius-arcmin  "$MASK_RADIUS_ARCMIN" \
    --niter-per-cycle     "$NITER_PER_CYCLE" \
    --uvmin-per-cycle     "$UVMIN_PER_CYCLE" \
    --uvmax-per-cycle     "$UVMAX_PER_CYCLE" \
    --scales-per-cycle    "$SCALES_PER_CYCLE" \
    --niter-final         "$NITER_FINAL" \
    --threshold           "$THRESHOLD" \
    --cycleniter          "$CYCLENITER" \
    --minsnr              "$MINSNR" \
    $OVERWRITE \
    2>&1 | tee /tmp/moon_selfcal_dev.log

echo ""
echo "Log: /tmp/moon_selfcal_dev.log"
echo "Outputs: $OUTDIR"
