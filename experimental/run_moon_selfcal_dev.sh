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
INDEX=~/DATA/gmrt_40_014/work/40_014_25jul2021_gsb.index.npz

OUTDIR=~/DATA/gmrt_40_014/work/casa_selfcal/moon0520_dev

# ── Integration selection ─────────────────────────────────────────────────────
# 0-based indices into the scan (MOON0520 has 113 integrations, 0..112).
# Start with a small spread to tune parameters quickly.
INTEGRATIONS="0 20 50 80 112"

# ── Image geometry ────────────────────────────────────────────────────────────
CELL="1.5arcsec"          # beam ~7.4 arcsec, cell = beam/5
IMSIZE=2048               # 2048 * 1.5" = 51 arcmin — covers primary beam + drift
STOKES="I"
WEIGHTING="briggs"
ROBUST=0.0
DECONVOLVER="multiscale"
SCALES="0,5,15,45"        # pixels: point, 1×beam, 2×beam, 6×beam

# ── Moon mask ─────────────────────────────────────────────────────────────────
# Moon angular diameter ~30' — mask radius 20' gives disk + margin.
MASK_RADIUS_ARCMIN=20.0

# ── UV range: cycle 1 (restricted — bright, well-behaved baselines only) ──────
# 0.5 kλ at 322 MHz = 465 m minimum baseline.
# Avoids short spacings dominated by RFI/diffuse confusion for first model.
UVMIN1_KLAMBDA=0.5
UVMAX1_KLAMBDA=""          # leave blank for no upper limit

# ── UV range: cycle 2 + final (relaxed — include shorter baselines) ───────────
# 0.12 kλ = ~112 m: keeps baselines sensitive to large-scale lunar emission.
UVMIN2_KLAMBDA=0.12
UVMAX2_KLAMBDA=""

# ── Clean iterations ──────────────────────────────────────────────────────────
NITER1=100          # shallow: just enough to build a starting model
NITER2=300          # after first selfcal: go a bit deeper
NITER_FINAL=500     # final image: deeper clean
THRESHOLD="0mJy"
CYCLENITER=100

# ── Selfcal ───────────────────────────────────────────────────────────────────
SELFCAL_CYCLES=2    # 1 or 2 phase-only selfcal rounds
MINSNR=3.0          # gaincal minsnr

# ── Misc ──────────────────────────────────────────────────────────────────────
OVERWRITE="--overwrite"   # set to "" to reuse existing MS / skip already-done ints

# ─────────────────────────────────────────────────────────────────────────────
# Build optional uvmax flags (only pass if non-empty)
uv1_max_flag=""
[[ -n "$UVMAX1_KLAMBDA" ]] && uv1_max_flag="--uvmax1-klambda $UVMAX1_KLAMBDA"
uv2_max_flag=""
[[ -n "$UVMAX2_KLAMBDA" ]] && uv2_max_flag="--uvmax2-klambda $UVMAX2_KLAMBDA"

# ─────────────────────────────────────────────────────────────────────────────
cd "$REPO_ROOT"

# shellcheck disable=SC2086
"$PYTHON" experimental/moon_selfcal_dev.py \
    --scan        "$SCAN" \
    --uvfits      "$UVFITS" \
    --index       "$INDEX" \
    --outdir      "$OUTDIR" \
    --integrations $INTEGRATIONS \
    --cell        "$CELL" \
    --imsize      "$IMSIZE" \
    --stokes      "$STOKES" \
    --weighting   "$WEIGHTING" \
    --robust      "$ROBUST" \
    --deconvolver "$DECONVOLVER" \
    --scales      "$SCALES" \
    --mask-radius-arcmin "$MASK_RADIUS_ARCMIN" \
    --uvmin1-klambda "$UVMIN1_KLAMBDA" \
    $uv1_max_flag \
    --uvmin2-klambda "$UVMIN2_KLAMBDA" \
    $uv2_max_flag \
    --niter1          "$NITER1" \
    --niter2          "$NITER2" \
    --niter-final     "$NITER_FINAL" \
    --threshold       "$THRESHOLD" \
    --cycleniter      "$CYCLENITER" \
    --selfcal-cycles  "$SELFCAL_CYCLES" \
    --minsnr          "$MINSNR" \
    $OVERWRITE \
    2>&1 | tee /tmp/moon_selfcal_dev.log

echo ""
echo "Log: /tmp/moon_selfcal_dev.log"
echo "Outputs: $OUTDIR"
