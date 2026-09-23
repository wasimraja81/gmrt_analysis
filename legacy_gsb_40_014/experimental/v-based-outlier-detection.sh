#!/usr/bin/env bash
# v-based-outlier-detection.sh
#
# Strategy : Flag using the Stokes-V proxy metric.
#            Model-free — works for any calibrator source.
#
# Mode     : Interactive loop.  A blocking diagnostic plot is shown after each
#            iteration.  Accepted flags accumulate IN MEMORY across iterations
#            within this run, so each solve incorporates all previously accepted
#            flags even in dry-run mode.
#
# Default  : DRY-RUN.  Nothing is written to disk (bandpass solutions and flag
#            tables are computed but discarded).  This is a safe scratch pad for
#            exploring thresholds and inspecting plots before committing.
#
# ─────────────────────────────────────────────────────────────────────────────
# TO MAKE A REAL (COMMITTING) RUN:
#   Remove the --dry-run line from the CMD block below, then re-run.
#   Accepted flags will be written to FLAG_TABLE_SESSION after each iteration.
# ─────────────────────────────────────────────────────────────────────────────
#
# Usage:
#   ./v-based-outlier-detection.sh                  # dry-run, 10 iterations (antennas + baselines)
#   ./v-based-outlier-detection.sh --set "FLAG_WHAT_TO_FLAG='baselines'"   # baselines only
#   ./v-based-outlier-detection.sh --set "FLAG_WHAT_TO_FLAG='antennas'"    # antennas only
#   ./v-based-outlier-detection.sh --n-iters 5      # fewer iterations
#   ./v-based-outlier-detection.sh --start-iter 4   # label runs from iter04
#   ./v-based-outlier-detection.sh --config /path/to/other.cfg
#   ./v-based-outlier-detection.sh \
#       --set "CAL_FITS=Path('/data/3c147.FITS')" \
#       --set "WORK_DIR=Path('/data/3c147/work')" \
#       --set "SOURCE='3C147'"

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVER="${SCRIPT_DIR}/run_preprocess.sh"

# matplotlib backend selection:
#   --auto in args → force Agg (non-interactive) so the unattended loop never
#                    blocks on a popup window.
#   otherwise      → MacOSX so each iteration opens a blocking plot window.
# In both cases, an explicit MPLBACKEND set by the caller takes precedence.
if [[ -z "${MPLBACKEND:-}" ]]; then
    if [[ " $* " =~ " --auto " ]]; then
        export MPLBACKEND=Agg
    else
        export MPLBACKEND=MacOSX
    fi
fi

# ── Help intercept ────────────────────────────────────────────────────────────
for arg in "$@"; do
    if [[ "${arg}" == "-h" || "${arg}" == "--help" ]]; then
        cat <<'EOF'

v-based-outlier-detection.sh  —  Stokes-V outlier flagging strategy
═══════════════════════════════════════════════════════════════════════

Strategy  : Flag antennas/baselines using the Stokes-V proxy metric (|RR-LL|).
            Model-free — no flux calibrator model required.

Mode      : Interactive loop (--step all --n-iters 10).
            After each iteration a diagnostic plot window opens (blocking)
            and you are prompted to accept or reject the proposed flags.
            Accepted flags accumulate in memory and feed the next solve.
            Plots are also saved to disk as PNG files in WORK_DIR.

            Default plot behaviour: popup windows ON (MPLBACKEND=MacOSX).
            To suppress popup windows and review PNGs instead:
              MPLBACKEND=Agg ./v-based-outlier-detection.sh

            To run fully non-interactively (no prompts, no blocking plots):
              ./v-based-outlier-detection.sh --auto
            --auto automatically forces MPLBACKEND=Agg so the loop never
            blocks on a window, unless you override MPLBACKEND explicitly.

            NOTE: --auto and MPLBACKEND are independent axes.
              --auto        controls decision mode: flags are auto-accepted
                            each iteration with no y/n/q prompt.
              MPLBACKEND    controls display: whether popup windows open.
            Combining them (MPLBACKEND=MacOSX + --auto) means popup windows
            open but you have no decision power — closing the window simply
            lets the automated loop continue.

Default   : DRY-RUN — nothing is written to disk.
            Safe for exploring thresholds and tuning parameters.

Preset parameters (edit this script to change):
    OUTLIER_METRIC                = V
    OUTLIER_METRIC_MERGE_STRATEGY = union
    ANTENNA_FLAG_THRESHOLD_JY     = {V: 5.0}
    BASELINE_FLAG_THRESHOLD_JY    = {V: 5.0}
    n-iters                       = 10
    FLAG_WHAT_TO_FLAG             = both  (from config; override with --set)
    COMPARE_METRICS_FOR_CONVERGENCE  = ['V', 'Model']
    CONVERGENCE_COMBINE_STRATEGY     = all
    RUN_ITER0_DIAGNOSTIC              = True

Flagging mode trade-offs:
    'both'      : flag antennas AND baselines each iteration.  Most aggressive.
                  A bad antenna is removed immediately, which may be too broad
                  if only some of its baselines are bad.
    'baselines' : flag only baselines.  A bad antenna will accumulate flags
                  across all its baselines over successive iterations and is
                  effectively quarantined without an explicit antenna flag.
                  Gentler — recommended for exploratory runs with more iterations.
    'antennas'  : flag only whole antennas.  Least granular.

    To switch mode at runtime (no config edit needed):
      ./v-based-outlier-detection.sh --set "FLAG_WHAT_TO_FLAG='baselines'"
      ./v-based-outlier-detection.sh --set "FLAG_WHAT_TO_FLAG='antennas'"
      ./v-based-outlier-detection.sh --set "FLAG_WHAT_TO_FLAG='both'"

TO MAKE A REAL (COMMITTING) RUN:
    Remove the --dry-run line from the CMD block in this script,
    then re-run.  Accepted flags will be written to FLAG_TABLE_SESSION
    after each accepted iteration.

Convergence / early stopping:
    C1 (always active): stops when the cumulative flag set is identical to the
    previous iteration.  Another solve would produce the same result.

    iter00 raw diagnostic (RUN_ITER0_DIAGNOSTIC=True, default):
      Before iter01 a diagnostic plot is produced from raw (uncorrected)
      visibilities tagged 'iter00'.  This is the pre-treatment baseline:
      it shows how high Stokes-V is and how far RR/LL deviate from the
      flux model BEFORE any bandpass correction or flagging is applied.
      Set to False to skip this step and start directly from iter01.

    C3 and C4 (epsilon-based): stop when spectral rms metrics stop changing.
      C3 tracks coherent Re⟨RR−LL⟩ rms at the phase centre (model-free).
      C4 tracks RR and LL residual rms independently (requires flux model).

      CONVERGENCE_EPSILON            fractional change threshold (default 0.01 = 1%).
                                     Set to 0.0 to disable all epsilon criteria.
      CONVERGENCE_MIN_ITERS          minimum iterations before C3/C4 are evaluated (default 3).
      COMPARE_METRICS_FOR_CONVERGENCE  which metrics to use; default ['V', 'Model'].
                                     Use ['V'] for C3 only, ['Model'] for C4 only, [] to disable.
      CONVERGENCE_COMBINE_STRATEGY   'any' = stop if any criterion fires.
                                     'all' = stop only if all criteria fire
                                             simultaneously (default for this script).

    Examples:
      # Default for this script: both C3 and C4 must stall at 1% before stopping.
      # (COMPARE_METRICS_FOR_CONVERGENCE and CONVERGENCE_COMBINE_STRATEGY are
      #  already set in the CMD block — no extra --set needed.)
      --set "CONVERGENCE_EPSILON=0.01"

      # Looser: stop as soon as either V-floor or model-residual stalls:
      --set "CONVERGENCE_EPSILON=0.01" \
      --set "CONVERGENCE_COMBINE_STRATEGY='any'"

      # V-floor only (model-free), e.g. if source has no PB2017 entry:
      --set "CONVERGENCE_EPSILON=0.005" \
      --set "COMPARE_METRICS_FOR_CONVERGENCE=['V']"

      # Disable all epsilon criteria — rely on C1 (flag-set unchanged) only:
      --set "CONVERGENCE_EPSILON=0.0"

Passthrough arguments (appended after the preset):
    --no-dry-run         REAL RUN — write bandpass + flags to disk each iteration.
                         The script always passes --dry-run for safety; adding
                         --no-dry-run overrides it.  Required for Phase-2
                         (run_clustering.sh) to find outputs.
    --auto               fully unattended batch loop (no prompts, no blocking plots)
    --n-iters N          override iteration count
    --start-iter N       start labelling from iterN
    --config FILE        use a different base config
    --set KEY=expr       override any config key, e.g.:
                           --set "FLAG_WHAT_TO_FLAG='baselines'"  (gentler strategy)
                           --set "FLAG_WHAT_TO_FLAG='antennas'"   (antennas only)
                           --set "ANTENNA_FLAG_THRESHOLD_JY={'V': 8.0}"  (loosen)
                           --set "BASELINE_FLAG_THRESHOLD_JY={'V': 3.0}" (tighten)
                           --set "SOURCE='3C147'"
                           --set "CAL_FITS=Path('/data/3c147.FITS')"
                           --set "WORK_DIR=Path('/data/3c147/work')"
                           # ── data-selection (applied to every solve + diagnostic):
                           --set "SOLVE_ELEVATION_MIN_DEG=25.0"        (drop below 25°)
                           --set "SOLVE_UVRANGE_KLAMBDA=(2.0, 50.0)"   (UV range kλ)
                           --set "SOLVE_UVRANGE_M=(200.0, 50000.0)"    (UV range metres)
                           --set "SOLVE_TIMERANGE=('2021-07-25 19:00','2021-07-25 23:00')"

For the full driver reference:
    ./run_preprocess.sh --help

EOF
        exit 0
    fi
done

# ── Run ───────────────────────────────────────────────────────────────────────
# --dry-run is hardcoded for safety.  Pass --no-dry-run to make a real run:
#   ./v-based-outlier-detection.sh --no-dry-run --auto ...
# argparse last-wins: --no-dry-run after --dry-run overrides it.
"${DRIVER}" \
    --step all \
    --n-iters 15 \
    --dry-run \
    --set "OUTLIER_METRIC='V'" \
    --set "OUTLIER_METRIC_MERGE_STRATEGY='union'" \
    --set "ANTENNA_FLAG_THRESHOLD_JY={'V': 5.0}" \
    --set "BASELINE_FLAG_THRESHOLD_JY={'V': 5.0}" \
    --set "CONVERGENCE_COMBINE_STRATEGY='all'" \
    --set "COMPARE_METRICS_FOR_CONVERGENCE=['V', 'Model']" \
    --set "RUN_ITER0_DIAGNOSTIC=True" \
    "$@"
EXIT_CODE=$?

# ── Post-run reminder ─────────────────────────────────────────────────────────
# Detect whether this was a real run (--no-dry-run present in original args).
_IS_REAL_RUN=false
for _arg in "$@"; do
    [[ "${_arg}" == '--no-dry-run' ]] && { _IS_REAL_RUN=true; break; }
done

echo ""
if ${_IS_REAL_RUN}; then
    echo "════════════════════════════════════════════════════════════════════════"
    echo "  REAL RUN COMPLETE — bandpass + flags written to disk."
    echo ""
    echo "  Preferred: use run_bandpass.sh for the full derive→audit workflow:"
    echo ""
    echo "    ./run_bandpass.sh \\"
    echo "        --phase=audit \\"
    echo "        --set \"CLUSTERING_THRESHOLD_JY=3.5\""
    echo ""
    echo "  Or commit clustering flags once threshold is chosen:"
    echo ""
    echo "    ./run_bandpass.sh \\"
    echo "        --phase=audit \\"
    echo "        --no-dry-run --refit \\"
    echo "        --set \"CLUSTERING_THRESHOLD_JY=3.5\""
    echo "════════════════════════════════════════════════════════════════════════"
else
    echo "════════════════════════════════════════════════════════════════════════"
    echo "  DRY-RUN COMPLETE — nothing was written to disk."
    echo ""
    echo "  Preferred: use run_bandpass.sh for the full workflow:"
    echo ""
    echo "    ./run_bandpass.sh \\"
    echo "        --phase=derive-and-audit \\"
    echo "        --no-dry-run --auto \\"
    echo "        --set \"SOURCE='3C48'\" ..."
    echo ""
    echo "  Or to commit this derive phase only:"
    echo ""
    echo "    ./v-based-outlier-detection.sh --no-dry-run --auto ..."
    echo "════════════════════════════════════════════════════════════════════════"
fi
echo ""

exit ${EXIT_CODE}
