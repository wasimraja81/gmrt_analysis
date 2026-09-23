#!/usr/bin/env bash
# run_bandpass.sh — Unified bandpass derivation & audit workflow
#
# Three phases:
#   --phase=derive           Phase 1: V-based iterative flagging loop.
#                            Produces a converged bandpass solution and
#                            cumulative flag table on disk (with --no-dry-run).
#
#   --phase=audit            Phase 2: Clustering detection on the saved Phase-1
#                            outputs.  Shows six BEFORE/AFTER plots.  Safe to
#                            repeat with different thresholds (dry-run by default).
#
#   --phase=derive-and-audit Both phases in sequence: derive runs to convergence,
#                            then audit runs immediately on the result.
#                            Most convenient for a first-time run.
#
# ─────────────────────────────────────────────────────────────────────────────
# QUICK START
#
#   First time — full pipeline, commit everything:
#     ./run_bandpass.sh \
#         --phase=derive-and-audit \
#         --no-dry-run \
#         --auto \
#         --n-iters 100 \
#         --refit \
#         --set "SOURCE='3C48'" \
#         --set "FLAG_WHAT_TO_FLAG='baselines'" \
#         --set "CLUSTERING_THRESHOLD_JY=3.5" \
#         --set "CONVERGENCE_EPSILON=0.005" \
#         --set "SOLVE_ELEVATION_MIN_DEG=25.0"
#
#   Re-audit with a tighter threshold (Phase 1 not repeated):
#     ./run_bandpass.sh \
#         --phase=audit \
#         --set "SOURCE='3C48'" \
#         --set "CLUSTERING_THRESHOLD_JY=2.5" \
#         --set "SOLVE_ELEVATION_MIN_DEG=25.0"
#
#   Happy with threshold — commit clustering flags + re-fit:
#     ./run_bandpass.sh \
#         --phase=audit \
#         --no-dry-run \
#         --refit \
#         --save-plots \
#         --set "SOURCE='3C48'" \
#         --set "CLUSTERING_THRESHOLD_JY=2.5" \
#         --set "SOLVE_ELEVATION_MIN_DEG=25.0"
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORCH="${SCRIPT_DIR}/pipeline_cli.py"

# ── activate venv (mirrors run_clustering.sh) ─────────────────────────────────
VENV_ACTIVATE="${SCRIPT_DIR}/gmrt/bin/activate"
if [[ -f "${VENV_ACTIVATE}" ]]; then
    # shellcheck source=/dev/null
    source "${VENV_ACTIVATE}"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Delegate to unified Python orchestrator
# ─────────────────────────────────────────────────────────────────────────────
exec python "${ORCH}" bandpass "$@"
