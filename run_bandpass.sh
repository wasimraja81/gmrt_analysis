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
DRIVER="${SCRIPT_DIR}/run_preprocess.sh"
CLUSTERING="${SCRIPT_DIR}/run_clustering.py"

# ── activate venv (mirrors run_clustering.sh) ─────────────────────────────────
VENV_ACTIVATE="${SCRIPT_DIR}/gmrt/bin/activate"
if [[ -f "${VENV_ACTIVATE}" ]]; then
    # shellcheck source=/dev/null
    source "${VENV_ACTIVATE}"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Help
# ─────────────────────────────────────────────────────────────────────────────
for _a in "$@"; do
    if [[ "${_a}" == "-h" || "${_a}" == "--help" ]]; then
        cat <<'EOF'

run_bandpass.sh — Unified bandpass derivation & audit workflow
══════════════════════════════════════════════════════════════════

Phases:
  --phase=derive           V-based iterative flagging loop (Phase 1).
                           Runs until convergence; writes bandpass +
                           cumulative flag table to disk (--no-dry-run).

  --phase=audit            Clustering detection (Phase 2).
                           Loads Phase-1 outputs, shows six BEFORE/AFTER
                           plots.  Safe to repeat to tune threshold.
                           Default: dry-run — nothing written to disk.

  --phase=derive-and-audit Both phases sequentially.
                           Derive runs first; audit follows immediately.
                           Useful for a clean first-time run.

Flags consumed by DERIVE only:
  --auto                   Non-interactive: flags auto-accepted each iteration.
  --n-iters N              Iteration cap (default: 15).
  --start-iter N           Label runs from iterN.

Flags consumed by AUDIT only:
  --refit                  Re-solve bandpass with clustering flags active.
  --save-plots             Save the six BEFORE/AFTER plots as PNGs.

Flags shared by BOTH phases:
  --dry-run / --no-dry-run Default: dry-run. --no-dry-run writes to disk.
  --config FILE            Alternate config file.
  --set KEY=expr           Override any config key, e.g.:
                             --set "SOURCE='3C48'"
                             --set "CLUSTERING_THRESHOLD_JY=3.5"
                             --set "FLAG_WHAT_TO_FLAG='baselines'"
                             --set "CONVERGENCE_EPSILON=0.005"
                             --set "SOLVE_ELEVATION_MIN_DEG=25.0"
  --log-level LEVEL        DEBUG | INFO | WARNING | ERROR

Derive presets (edit this script to change):
  OUTLIER_METRIC                = V
  OUTLIER_METRIC_MERGE_STRATEGY = union
  ANTENNA_FLAG_THRESHOLD_JY     = {V: 5.0}
  BASELINE_FLAG_THRESHOLD_JY    = {V: 5.0}
  CONVERGENCE_COMBINE_STRATEGY  = all
  COMPARE_METRICS_FOR_CONVERGENCE = [V, Model]
  RUN_ITER0_DIAGNOSTIC           = True
  n-iters (default)             = 15

Examples:
  # Full run — derive + audit, commit everything:
  ./run_bandpass.sh \
      --phase=derive-and-audit \
      --no-dry-run \
      --auto \
      --n-iters 100 \
      --refit \
      --set "SOURCE='3C48'" \
      --set "FLAG_WHAT_TO_FLAG='baselines'" \
      --set "CLUSTERING_THRESHOLD_JY=3.5" \
      --set "CONVERGENCE_EPSILON=0.005" \
      --set "SOLVE_ELEVATION_MIN_DEG=25.0"

  # Tune clustering threshold interactively (Phase 1 not re-run):
  ./run_bandpass.sh \
      --phase=audit \
      --set "SOURCE='3C48'" \
      --set "CLUSTERING_THRESHOLD_JY=2.5"

  # Commit clustering flags + re-fit:
  ./run_bandpass.sh \
      --phase=audit \
      --no-dry-run \
      --refit \
      --save-plots \
      --set "SOURCE='3C48'" \
      --set "CLUSTERING_THRESHOLD_JY=2.5"

  # Derive only (e.g. before deciding on audit threshold):
  ./run_bandpass.sh \
      --phase=derive \
      --no-dry-run \
      --auto \
      --n-iters 100 \
      --set "SOURCE='3C48'" \
      --set "FLAG_WHAT_TO_FLAG='baselines'" \
      --set "CONVERGENCE_EPSILON=0.005" \
      --set "SOLVE_ELEVATION_MIN_DEG=25.0"

For the full driver reference:
  ./run_preprocess.sh --help
  ./run_clustering.sh --help

EOF
        exit 0
    fi
done

# ─────────────────────────────────────────────────────────────────────────────
# Argument bucketing
#
# Flags are split into three buckets so each sub-script only receives flags
# it understands.  argparse aborts on unknown flags, so passing --refit to
# run_preprocess.sh (which doesn't know it) would crash the derive phase.
#
# Derive-only  : --auto  --n-iters  --start-iter
# Audit-only   : --refit  --save-plots
# Dry-run      : --dry-run / --no-dry-run captured separately — never passed
#                through SHARED_ARGS to avoid duplicate/conflicting flags.
# Shared       : everything else (--set, --config, --log-level, ...)
# ─────────────────────────────────────────────────────────────────────────────
PHASE=""
SHARED_ARGS=()
DERIVE_ARGS=()
AUDIT_ARGS=()
DRY_RUN_ARG=--dry-run          # default: safe dry-run; overridden below

while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase=*)      PHASE="${1#--phase=}"; shift ;;
        --phase)        PHASE="$2"; shift 2 ;;

        # dry-run: captured separately, never forwarded via SHARED_ARGS
        --dry-run)      DRY_RUN_ARG=--dry-run; shift ;;
        --no-dry-run)   DRY_RUN_ARG=--no-dry-run; shift ;;

        # derive-only flags
        --auto)             DERIVE_ARGS+=("$1"); shift ;;
        --n-iters)          DERIVE_ARGS+=("$1" "$2"); shift 2 ;;
        --n-iters=*)        DERIVE_ARGS+=("$1"); shift ;;
        --start-iter)       DERIVE_ARGS+=("$1" "$2"); shift 2 ;;
        --start-iter=*)     DERIVE_ARGS+=("$1"); shift ;;

        # audit-only flags
        --refit)            AUDIT_ARGS+=("$1"); shift ;;
        --save-plots)       AUDIT_ARGS+=("$1"); shift ;;

        # shared (everything else: --set, --config, --log-level, ...)
        *)                  SHARED_ARGS+=("$1"); shift ;;
    esac
done

# Derive and audit each get their own dry-run flag derived from the single
# DRY_RUN_ARG the user provided — never duplicated.
_DERIVE_DRY_RUN_FLAG="${DRY_RUN_ARG}"
_AUDIT_DRY_RUN_FLAG="${DRY_RUN_ARG}"

# ── Phase validation ──────────────────────────────────────────────────────────
if [[ -z "${PHASE}" ]]; then
    echo "ERROR: --phase is required.  Use one of: derive, audit, derive-and-audit" >&2
    echo "       Run ./run_bandpass.sh --help for full usage." >&2
    exit 1
fi
case "${PHASE}" in
    derive|audit|derive-and-audit) ;;
    *)
        echo "ERROR: unknown --phase '${PHASE}'.  Must be: derive, audit, or derive-and-audit" >&2
        exit 1
        ;;
esac

# ── Warn if derive-and-audit is run without --no-dry-run ─────────────────────
if [[ "${PHASE}" == "derive-and-audit" && "${DRY_RUN_ARG}" != "--no-dry-run" ]]; then
    echo ""
    echo "  WARNING: --phase=derive-and-audit without --no-dry-run."
    echo "  The derive phase will be a dry-run (nothing written to disk)."
    echo "  The audit phase will then fail to load the bandpass solution."
    echo "  Add --no-dry-run to write Phase-1 outputs before auditing."
    echo ""
fi

# ── MPLBACKEND ────────────────────────────────────────────────────────────────
# Derive: Agg if --auto is present (non-blocking batch loop).
# Audit:  MacOSX (interactive plots for threshold inspection).
# Caller-set MPLBACKEND always wins.
_DERIVE_MPLBACKEND="${MPLBACKEND:-}"
_AUDIT_MPLBACKEND="${MPLBACKEND:-}"

if [[ -z "${MPLBACKEND:-}" ]]; then
    _IS_AUTO=false
    for _a in "${DERIVE_ARGS[@]+"${DERIVE_ARGS[@]}"}"; do
        [[ "${_a}" == '--auto' ]] && { _IS_AUTO=true; break; }
    done
    ${_IS_AUTO} && _DERIVE_MPLBACKEND=Agg || _DERIVE_MPLBACKEND=MacOSX
    _AUDIT_MPLBACKEND=MacOSX
fi

# ── Derive preset args ────────────────────────────────────────────────────────
# These mirror the presets hardcoded in v-based-outlier-detection.sh.
# The caller can override any of them with --set.
#
DERIVE_PRESETS=(
    --step all
    --n-iters 15
    --set "OUTLIER_METRIC='V'"
    --set "OUTLIER_METRIC_MERGE_STRATEGY='union'"
    --set "ANTENNA_FLAG_THRESHOLD_JY={'V': 5.0}"
    --set "BASELINE_FLAG_THRESHOLD_JY={'V': 5.0}"
    --set "CONVERGENCE_COMBINE_STRATEGY='all'"
    --set "COMPARE_METRICS_FOR_CONVERGENCE=['V', 'Model']"
    --set "RUN_ITER0_DIAGNOSTIC=True"
)

# ─────────────────────────────────────────────────────────────────────────────
# Phase runner helpers
# ─────────────────────────────────────────────────────────────────────────────
run_derive() {
    echo ""
    echo "════════════════════════════════════════════════════════════════════════"
    echo "  PHASE 1 — V-based iterative flagging (derive)  [${_DERIVE_DRY_RUN_FLAG}]"
    echo "════════════════════════════════════════════════════════════════════════"
    echo ""
    MPLBACKEND="${_DERIVE_MPLBACKEND}" \
        "${DRIVER}" \
            "${DERIVE_PRESETS[@]}" \
            "${_DERIVE_DRY_RUN_FLAG}" \
            "${SHARED_ARGS[@]+"${SHARED_ARGS[@]}"}" \
            "${DERIVE_ARGS[@]+"${DERIVE_ARGS[@]}"}"
}

run_audit() {
    echo ""
    echo "════════════════════════════════════════════════════════════════════════"
    echo "  PHASE 2 — Clustering detection (audit)"
    echo "════════════════════════════════════════════════════════════════════════"
    echo ""
    MPLBACKEND="${_AUDIT_MPLBACKEND}" \
        python "${CLUSTERING}" \
            "${_AUDIT_DRY_RUN_FLAG}" \
            "${SHARED_ARGS[@]+"${SHARED_ARGS[@]}"}" \
            "${AUDIT_ARGS[@]+"${AUDIT_ARGS[@]}"}"
}

# ─────────────────────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────────────────────
case "${PHASE}" in
    derive)
        run_derive
        EXIT_CODE=$?
        ;;
    audit)
        run_audit
        EXIT_CODE=$?
        ;;
    derive-and-audit)
        run_derive
        DERIVE_EXIT=$?
        if [[ ${DERIVE_EXIT} -ne 0 ]]; then
            echo "ERROR: derive phase failed (exit ${DERIVE_EXIT}) — skipping audit." >&2
            exit ${DERIVE_EXIT}
        fi
        run_audit
        EXIT_CODE=$?
        ;;
esac

# ─────────────────────────────────────────────────────────────────────────────
# Post-run banner
# ─────────────────────────────────────────────────────────────────────────────
_IS_REAL_RUN=false
for _a in "${SHARED_ARGS[@]+"${SHARED_ARGS[@]}"}"; do
    [[ "${_a}" == '--no-dry-run' ]] && { _IS_REAL_RUN=true; break; }
done

echo ""
echo "════════════════════════════════════════════════════════════════════════"
case "${PHASE}" in
    derive)
        if ${_IS_REAL_RUN}; then
            echo "  Phase 1 complete — bandpass + flags written to disk."
            echo ""
            echo "  Next: run Phase 2 to tune the clustering threshold:"
            echo ""
            echo "    ./run_bandpass.sh \\"
            echo "        --phase=audit \\"
            echo "        --set \"CLUSTERING_THRESHOLD_JY=3.5\""
        else
            echo "  Phase 1 complete — DRY-RUN, nothing written to disk."
            echo ""
            echo "  Add --no-dry-run to commit bandpass + flags:"
            echo ""
            echo "    ./run_bandpass.sh --phase=derive --no-dry-run --auto ..."
        fi
        ;;
    audit)
        if ${_IS_REAL_RUN}; then
            echo "  Phase 2 complete — clustering flags written to disk."
        else
            echo "  Phase 2 complete — DRY-RUN, nothing written to disk."
            echo ""
            echo "  Add --no-dry-run to commit clustering flags:"
            echo ""
            echo "    ./run_bandpass.sh --phase=audit --no-dry-run --refit \\"
            echo "        --set \"CLUSTERING_THRESHOLD_JY=<chosen_value>\""
        fi
        ;;
    derive-and-audit)
        if ${_IS_REAL_RUN}; then
            echo "  Full pipeline complete — bandpass + flags written to disk."
            echo ""
            echo "  To re-tune the clustering threshold without re-running Phase 1:"
            echo ""
            echo "    ./run_bandpass.sh --phase=audit --no-dry-run --refit \\"
            echo "        --set \"CLUSTERING_THRESHOLD_JY=<new_value>\""
        else
            echo "  Full pipeline complete — DRY-RUN, nothing written to disk."
        fi
        ;;
esac
echo "════════════════════════════════════════════════════════════════════════"
echo ""

exit ${EXIT_CODE}
