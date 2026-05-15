#!/usr/bin/env bash
# clustering_moon0520_dev.sh
#
# Clustering outlier-detection diagnostic for moon0520 primary+secondary
# calibrated split UVFITS.
#
# Run AFTER plotVis_moon0520_dev.sh — inspect the Stokes-V amp_uvdist and
# amp_time plots to choose CLUSTERING_THRESHOLD_JY for V, then set it below.
#
# Input  : split/moon/moon0520_primary_secondary_calibrated_flagged.uvfits  (READ-ONLY)
# Outputs:
#   diagnostics_out/target/moon0520/flagging/<series>/   ← clustering PNGs/PDF
#   diagnostics_out/target/moon0520/flagging/moon0520_<series>_flag_table.json
#
# Lineage controls:
#   FLAG_SERIES_TAG=dev             # default series / lineage name
#   FLAG_MODE=progressive           # progressive | fresh
#
# Meaning:
#   progressive  → apply the existing session table for this series (if any),
#                  detect residual flags, and merge by UNION into the same
#                  session table; versioned snapshots remain cumulative within
#                  that series.
#   fresh        → start a new lineage for this series name. If the session
#                  table already exists, refuse to proceed and exit cleanly.
#
# The new flag table is written to a series-specific dev path and does NOT
# touch the baked-in flags in the split UVFITS or any existing production flag
# tables.
#
# Usage:
#   bash experimental/clustering_moon0520_dev.sh
#
# Reproducibility:
#   Each run writes:
#     - full command line: ~/DATA/gmrt_40_014/work/logs/clustering_moon0520_<series>_<mode>_<ts>.cmd
#     - full stdout/stderr: ~/DATA/gmrt_40_014/work/logs/clustering_moon0520_<series>_<mode>_<ts>.log
#
# Recommended residual-pass reminder (example):
#   FLAG_MODE=progressive FLAG_SERIES_TAG=dev \
#   UVRANGE_KLAMBDA="0.5 4" CLUSTERING_THRESHOLD_RR_LL_JY=5.0 \
#   bash experimental/clustering_moon0520_dev.sh
#
# Note on naming:
#   The key name SOLVE_UVRANGE_KLAMBDA is historical config naming shared with
#   the pipeline; in run_clustering.py it is the correct selection key passed
#   into load_vis_for_source(..., uvrange_klambda=...).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_CMD="${PYTHON_CMD:-python}"

cd "$REPO_ROOT"

# ── Input (read-only) ─────────────────────────────────────────────────────────
FITS=~/DATA/gmrt_40_014/work/split/moon/moon0520_primary_secondary_calibrated_flagged.uvfits
INDEX="${FITS}.row_index_cache.npz"
SOURCE="MOON0520"

# ── Output directories / flag lineage controls ───────────────────────────────
CLUSTER_DIR=~/DATA/gmrt_40_014/work/diagnostics_out/target/moon0520/flagging
FLAG_SERIES_TAG="${FLAG_SERIES_TAG:-dev}"
FLAG_MODE="${FLAG_MODE:-progressive}"

FLAG_SERIES_TAG_SANITIZED="$(printf '%s' "$FLAG_SERIES_TAG" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9._-]+/_/g; s/_+/_/g; s/^_+|_+$//g')"
if [[ -z "$FLAG_SERIES_TAG_SANITIZED" ]]; then
    echo "ERROR: FLAG_SERIES_TAG resolved to empty after sanitization"
    exit 1
fi

case "$FLAG_MODE" in
    progressive|fresh) ;;
    *)
        echo "ERROR: invalid FLAG_MODE=$FLAG_MODE (expected: progressive or fresh)"
        exit 1
        ;;
esac

FLAG_TABLE="$CLUSTER_DIR/moon0520_${FLAG_SERIES_TAG_SANITIZED}_flag_table.json"
PLOT_DIR="$CLUSTER_DIR/${FLAG_SERIES_TAG_SANITIZED}"

LOG_DIR=~/DATA/gmrt_40_014/work/logs
RUN_TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$CLUSTER_DIR" "$PLOT_DIR" "$LOG_DIR"

# ── Clustering thresholds ─────────────────────────────────────────────────────
# RR/LL: 50 Jy — based on visual inspection of visibility plots showing outliers
#        above this level.
# V: SET THIS after inspecting plotVis_moon0520_dev.sh output. Stokes-V should
#    be near-zero for the Moon (thermally emitting, nearly unpolarised at 322 MHz).
#    Any baseline consistently above ~5 Jy in V is suspicious RFI/cross-talk.
#    Adjust CLUSTERING_THRESHOLD_V_JY based on what you see in the V plots.
#
# Optional env overrides:
#   CLUSTERING_THRESHOLD_RR_LL_JY=5.0
#   CLUSTERING_THRESHOLD_V_JY=2.5
#   UVRANGE_KLAMBDA="0.5 4"   or   UVRANGE_KLAMBDA="0.5,4"
CLUSTERING_THRESHOLD_RR_LL_JY="${CLUSTERING_THRESHOLD_RR_LL_JY:-50.0}"
CLUSTERING_THRESHOLD_V_JY="${CLUSTERING_THRESHOLD_V_JY:-2.5}"
UVRANGE_KLAMBDA="${UVRANGE_KLAMBDA:-}"

UVRANGE_KLAMBDA_EXPR="None"
if [[ -n "$UVRANGE_KLAMBDA" ]]; then
    _uv_kl_raw="${UVRANGE_KLAMBDA//,/ }"
    read -r _uv_kl_min _uv_kl_max _uv_kl_extra <<<"$_uv_kl_raw"
    if [[ -z "${_uv_kl_min:-}" || -z "${_uv_kl_max:-}" || -n "${_uv_kl_extra:-}" ]]; then
        echo "ERROR: UVRANGE_KLAMBDA must contain exactly two values, got: $UVRANGE_KLAMBDA"
        exit 1
    fi
    UVRANGE_KLAMBDA_EXPR="(${_uv_kl_min},${_uv_kl_max})"
fi

if [[ ! -f "$FITS" ]]; then
    echo "ERROR: FITS not found: $FITS"
    exit 1
fi
if [[ ! -f "$INDEX" ]]; then
    echo "ERROR: index cache not found: $INDEX"
    exit 1
fi

# Determine whether this run should start fresh or update an existing lineage.
DOFLAG_ARG="off"
if [[ "$FLAG_MODE" == "progressive" ]]; then
    DOFLAG_ARG="on"
elif [[ -f "$FLAG_TABLE" ]]; then
    echo "[clustering-moon0520-dev] WARNING: FLAG_MODE=fresh but session file already exists:"
    echo "[clustering-moon0520-dev]   $FLAG_TABLE"
    echo "[clustering-moon0520-dev]"
    echo "[clustering-moon0520-dev] Refusing to overwrite or fork implicitly."
    echo "[clustering-moon0520-dev] Valid next actions:"
    echo "[clustering-moon0520-dev]   1) Continue this lineage: FLAG_MODE=progressive bash experimental/clustering_moon0520_dev.sh"
    echo "[clustering-moon0520-dev]   2) Start a new lineage:  FLAG_MODE=fresh FLAG_SERIES_TAG=<new_tag> bash experimental/clustering_moon0520_dev.sh"
    echo "[clustering-moon0520-dev]   3) Remove this lineage manually if you truly want to restart it from scratch"
    echo "[clustering-moon0520-dev]      rm -f '$FLAG_TABLE' '${FLAG_TABLE%.json}'_v*.json"
    exit 0
fi

CMD=(
    "$PYTHON_CMD" "$REPO_ROOT/src/pipeline_cli.py" clustering
    --config preprocess_ugmrt.cfg
    --no-dry-run
    --docal off
    --doflag "$DOFLAG_ARG"
    --save-plots
    --set "WORK_DIR=Path('${PLOT_DIR}')"
    --set "CAL_FITS=Path('${FITS}')"
    --set "INDEX_CACHE=Path('${INDEX}')"
    --set "SOURCE='${SOURCE}'"
    --set "BANDPASS_OUT=None"
    --set "FLAG_TABLE_PATHS=[]"
    --set "FLAG_TABLE_SESSION=Path('${FLAG_TABLE}')"
    --set "MAX_ROWS_SOLVE=None"
    --set "SOLVE_ELEVATION_MIN_DEG=None"
    # Historical config naming: this is the correct key consumed by run_clustering.py.
    --set "SOLVE_UVRANGE_KLAMBDA=${UVRANGE_KLAMBDA_EXPR}"
    --set "CLUSTERING_CORR=['V','RR','LL']"
    --set "CLUSTERING_THRESHOLD_JY={'V':${CLUSTERING_THRESHOLD_V_JY},'RR':${CLUSTERING_THRESHOLD_RR_LL_JY},'LL':${CLUSTERING_THRESHOLD_RR_LL_JY}}"
    --set "CLUSTERING_THRESHOLD_LOW_JY=None"
    --set "CLUSTERING_GLOBAL_ANT_FLAG_FRACTION=0.80"
    --set "CHAN_RANGE=(0,127)"
    --set "PLOT_CHAN_RANGE=(0,127)"
)

CMD_FILE="$LOG_DIR/clustering_moon0520_${FLAG_SERIES_TAG_SANITIZED}_${FLAG_MODE}_${RUN_TS}.cmd"
LOG_FILE="$LOG_DIR/clustering_moon0520_${FLAG_SERIES_TAG_SANITIZED}_${FLAG_MODE}_${RUN_TS}.log"

{
    echo "# timestamp=$RUN_TS"
    echo "# cwd=$PWD"
    printf '%q ' "${CMD[@]}"
    printf '\n'
} > "$CMD_FILE"

echo "[clustering-moon0520-dev] FITS      : $FITS"
echo "[clustering-moon0520-dev] source    : $SOURCE"
echo "[clustering-moon0520-dev] plot dir   : $PLOT_DIR"
echo "[clustering-moon0520-dev] series    : $FLAG_SERIES_TAG_SANITIZED"
echo "[clustering-moon0520-dev] mode      : $FLAG_MODE"
echo "[clustering-moon0520-dev] doflag    : $DOFLAG_ARG"
echo "[clustering-moon0520-dev] flag out  : $FLAG_TABLE  (series session table; does NOT touch split UVFITS)"
echo "[clustering-moon0520-dev] thresholds: V=${CLUSTERING_THRESHOLD_V_JY} Jy  RR/LL=${CLUSTERING_THRESHOLD_RR_LL_JY} Jy"
echo "[clustering-moon0520-dev] uvrange kλ: ${UVRANGE_KLAMBDA_EXPR}"
echo "[clustering-moon0520-dev] cmd       : $CMD_FILE"
echo "[clustering-moon0520-dev] log       : $LOG_FILE"
if [[ -f "$FLAG_TABLE" ]]; then
    echo "[clustering-moon0520-dev] existing session table found and will be used as prior flags"
fi
echo ""
echo "[clustering-moon0520-dev] NOTE: CLUSTERING_THRESHOLD_V_JY=${CLUSTERING_THRESHOLD_V_JY} Jy is a placeholder."
echo "                               Review V plots from plotVis_moon0520_dev.sh first."
echo ""

"${CMD[@]}" 2>&1 | tee "$LOG_FILE"

echo ""
echo "[clustering-moon0520-dev] Done."
echo "  Flagging plots : $PLOT_DIR"
echo "  Session table  : $FLAG_TABLE"
echo ""
echo "  Next: review the flagging plots and flag table, then decide whether to"
echo "        pass this merged session table lineage to run_moon_selfcal_dev.sh."
