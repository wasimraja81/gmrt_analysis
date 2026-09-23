#!/usr/bin/env bash
# run_moon_selfcal_nomask_dev.sh
#
# Runs the full moon selfcal pipeline WITHOUT any CLEAN mask.
# All imaging parameters are identical to run_moon_selfcal_dev.sh.
#
# Output goes to a clearly tagged directory:
#   ~/DATA/gmrt_40_014/work/casa_selfcal/moon0520_stk10_nomask
#
# Purpose: compare against the masked run (moon0520_stk10) to check
# whether imposing a circular mask at the expected Moon position is
# biasing the reconstructed brightness distribution or the selfcal
# phase solutions.
#
# Usage:
#   bash experimental/run_moon_selfcal_nomask_dev.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

NO_MASK=1 \
OUTDIR="$HOME/DATA/gmrt_40_014/work/casa_selfcal/moon0520_stk10_nomask" \
bash "$SCRIPT_DIR/run_moon_selfcal_dev.sh" "$@"
