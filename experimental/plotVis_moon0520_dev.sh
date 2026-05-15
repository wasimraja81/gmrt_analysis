#!/usr/bin/env bash
# plotVis_moon0520_dev.sh
#
# Visibility diagnostic plots for moon0520 primary+secondary calibrated split,
# including Stokes-V panels. Can be run before clustering to choose an initial
# threshold, or after clustering with the session flag table applied to inspect
# residuals under additional data-selection filters.
#
# Input  : split/moon/moon0520_primary_secondary_calibrated_flagged.uvfits  (READ-ONLY)
# Outputs: diagnostics_out/target/moon0520/vis_<series>/   (PNGs + PDF)
#
# Usage:
#   bash experimental/plotVis_moon0520_dev.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_CMD="${PYTHON_CMD:-python}"

cd "$REPO_ROOT"

FITS=~/DATA/gmrt_40_014/work/split/moon/moon0520_primary_secondary_calibrated_flagged.uvfits
INDEX="${FITS}.row_index_cache.npz"
SOURCE="MOON0520"
CHAN_START=0
CHAN_END=127

# Series-aware optional application of the clustering session flag table.
FLAG_SERIES_TAG="${FLAG_SERIES_TAG:-dev}"
APPLY_SESSION_FLAG="${APPLY_SESSION_FLAG:-on}"
FLAG_DIR=~/DATA/gmrt_40_014/work/diagnostics_out/target/moon0520/flagging
FLAG_SERIES_TAG_SANITIZED="$(printf '%s' "$FLAG_SERIES_TAG" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9._-]+/_/g; s/_+/_/g; s/^_+|_+$//g')"
if [[ -z "$FLAG_SERIES_TAG_SANITIZED" ]]; then
    echo "ERROR: FLAG_SERIES_TAG resolved to empty after sanitization"
    exit 1
fi
SESSION_FLAG_TABLE="$FLAG_DIR/moon0520_${FLAG_SERIES_TAG_SANITIZED}_flag_table.json"

# Optional selection filters (leave empty to disable).
# Examples:
#   UVRANGE_KLAMBDA="1 5"  or  UVRANGE_KLAMBDA="1,5"
#   TIME_RANGE="2021/07/25/23:56:37 2021/07/26/00:11:37"
TIME_RANGE="${TIME_RANGE:-}"
UVRANGE_M="${UVRANGE_M:-}"
UVRANGE_KLAMBDA="${UVRANGE_KLAMBDA:-}"
ELEVATION_MIN="${ELEVATION_MIN:-}"
ELEVATION_MAX="${ELEVATION_MAX:-}"

# Dev diagnostic output — completely separate from the production final_qa outputs
OUTROOT=~/DATA/gmrt_40_014/work/diagnostics_out/target/moon0520/vis_${FLAG_SERIES_TAG_SANITIZED}

CONFIG="$REPO_ROOT/preprocess_ugmrt.cfg"

LOG_DIR=~/DATA/gmrt_40_014/work/logs
RUN_TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTROOT" "$LOG_DIR"

if [[ ! -f "$FITS" ]]; then
    echo "ERROR: FITS not found: $FITS"
    exit 1
fi
if [[ ! -f "$INDEX" ]]; then
    echo "ERROR: index cache not found: $INDEX"
    exit 1
fi

case "$APPLY_SESSION_FLAG" in
    on|off) ;;
    *)
        echo "ERROR: APPLY_SESSION_FLAG must be on or off (got: $APPLY_SESSION_FLAG)"
        exit 1
        ;;
esac

append_pair_args() {
    local opt="$1"
    local raw="$2"
    [[ -z "$raw" ]] && return 0
    raw="${raw//,/ }"
    # shellcheck disable=SC2206
    local parts=( $raw )
    if (( ${#parts[@]} != 2 )); then
        echo "ERROR: $opt expects exactly two values, got: $raw"
        exit 1
    fi
    CMD+=("$opt" "${parts[0]}" "${parts[1]}")
}

CMD=(
    "$PYTHON_CMD" "$REPO_ROOT/src/plotVis.py"
    --config "$CONFIG"
    --fits "$FITS"
    --outdir "$OUTROOT"
    --index-cache "$INDEX"
    --source "$SOURCE"
    --chan-range "$CHAN_START" "$CHAN_END"
    --products RR,LL
    --panels amp_uvdist,phase_uvdist,amp_time,phase_time,amp_freq,phase_freq,ri_scatter,uv_sampling
    --sample-frac 0.05
    --overlay-flags
    --multipage both
)

if [[ "$APPLY_SESSION_FLAG" == "on" ]]; then
    if [[ -f "$SESSION_FLAG_TABLE" ]]; then
        CMD+=(--flag "$SESSION_FLAG_TABLE")
    else
        echo "[plotVis-moon0520-dev] WARNING: session flag table not found, plotting without extra clustering flags: $SESSION_FLAG_TABLE"
    fi
fi

append_pair_args --time-range "$TIME_RANGE"
append_pair_args --uvrange-m "$UVRANGE_M"
append_pair_args --uvrange-klambda "$UVRANGE_KLAMBDA"
if [[ -n "$ELEVATION_MIN" ]]; then
    CMD+=(--elevation-min "$ELEVATION_MIN")
fi
if [[ -n "$ELEVATION_MAX" ]]; then
    CMD+=(--elevation-max "$ELEVATION_MAX")
fi

CMD_FILE="$LOG_DIR/plotVis_moon0520_${FLAG_SERIES_TAG_SANITIZED}_${RUN_TS}.cmd"
LOG_FILE="$LOG_DIR/plotVis_moon0520_${FLAG_SERIES_TAG_SANITIZED}_${RUN_TS}.log"

{
    echo "# timestamp=$RUN_TS"
    echo "# cwd=$PWD"
    printf '%q ' "${CMD[@]}"
    printf '\n'
} > "$CMD_FILE"

echo "[plotVis-moon0520-dev] FITS    : $FITS"
echo "[plotVis-moon0520-dev] source  : $SOURCE"
echo "[plotVis-moon0520-dev] outroot : $OUTROOT"
echo "[plotVis-moon0520-dev] series  : $FLAG_SERIES_TAG_SANITIZED"
echo "[plotVis-moon0520-dev] products: V"
echo "[plotVis-moon0520-dev] apply clustering session flag: $APPLY_SESSION_FLAG"
if [[ "$APPLY_SESSION_FLAG" == "on" ]]; then
    echo "[plotVis-moon0520-dev] session flag table         : $SESSION_FLAG_TABLE"
fi
if [[ -n "$TIME_RANGE" ]]; then
    echo "[plotVis-moon0520-dev] time-range                 : $TIME_RANGE"
fi
if [[ -n "$UVRANGE_M" ]]; then
    echo "[plotVis-moon0520-dev] uvrange-m                  : $UVRANGE_M"
fi
if [[ -n "$UVRANGE_KLAMBDA" ]]; then
    echo "[plotVis-moon0520-dev] uvrange-klambda            : $UVRANGE_KLAMBDA"
fi
if [[ -n "$ELEVATION_MIN$ELEVATION_MAX" ]]; then
    echo "[plotVis-moon0520-dev] elevation min/max          : ${ELEVATION_MIN:-none} / ${ELEVATION_MAX:-none}"
fi
echo "[plotVis-moon0520-dev] cmd     : $CMD_FILE"
echo "[plotVis-moon0520-dev] log     : $LOG_FILE"

"${CMD[@]}" 2>&1 | tee "$LOG_FILE"

echo ""
echo "[plotVis-moon0520-dev] Done. Review V amp_uvdist and amp_time plots to set"
echo "                               CLUSTERING_THRESHOLD_JY for V in clustering_moon0520_dev.sh"
echo "                               Outputs: $OUTROOT"
