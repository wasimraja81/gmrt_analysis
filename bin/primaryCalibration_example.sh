#!/usr/bin/env bash

# Primary bandpass calibration example (pipeline_cli preprocess interface)
# - Uses config defaults from preprocess_ugmrt.cfg
# - Reproduces the Apr 29 "happy run" behavior in ./tmp via explicit --set overrides
# - Runs Phase-1 derive workflow with iterative flagging + final clustering
# - Starts clean by removing prior session flag table in ./tmp

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# PYTHON_CMD is set in the machine-specific block below

cd "$REPO_ROOT"

SOURCE="3C48"
SRC_TAG="$(printf '%s' "$SOURCE" | tr '[:upper:]' '[:lower:]')"

# ── Machine/data-profile settings ───────────────────────────────────────
# Data profile can be forced on any machine:
#   GMRT_DATA_PROFILE=gsb  or  GMRT_DATA_PROFILE=gwb
# Optional path overrides:
#   GMRT_BASE_DIR, GMRT_DATA_DIR, GMRT_WORK_DIR, GMRT_CAL_FITS, GMRT_INDEX_CACHE
_HOSTNAME="$(hostname -s)"
if [[ "$_HOSTNAME" == "wasim-desktop" ]]; then
    _BASE_DIR_DEFAULT=/data1/gmrt/40_014
    _DATA_DIR_DEFAULT=/data1/gmrt/40_014_25JUL2021
    PYTHON_CMD="${PYTHON_CMD:-${REPO_ROOT}/gmrt/bin/python}"
else
    _BASE_DIR_DEFAULT="$HOME/DATA/gmrt_40_014"
    _DATA_DIR_DEFAULT="${_BASE_DIR_DEFAULT}/data"
    PYTHON_CMD="${PYTHON_CMD:-python}"
fi

BASE_DIR="${GMRT_BASE_DIR:-${_BASE_DIR_DEFAULT}}"
DATA_DIR="${GMRT_DATA_DIR:-${_DATA_DIR_DEFAULT}}"
WORK_DIR="${GMRT_WORK_DIR:-${BASE_DIR}/work}"

_GMRT_DATA_PROFILE="${GMRT_DATA_PROFILE:-}"
if [[ -z "$_GMRT_DATA_PROFILE" ]]; then
    if [[ "$_HOSTNAME" == "wasim-desktop" ]]; then
        _GMRT_DATA_PROFILE=gwb
    else
        _GMRT_DATA_PROFILE=gsb
    fi
fi
_GMRT_DATA_PROFILE="$(printf '%s' "$_GMRT_DATA_PROFILE" | tr '[:upper:]' '[:lower:]')"
if [[ "$_GMRT_DATA_PROFILE" != "gwb" && "$_GMRT_DATA_PROFILE" != "gsb" ]]; then
    echo "Invalid GMRT_DATA_PROFILE='$_GMRT_DATA_PROFILE' (expected 'gwb' or 'gsb')" >&2
    exit 2
fi

if [[ "$_GMRT_DATA_PROFILE" == "gwb" ]]; then
    _fits_stem='40_014_25jul2021_2.6s_gwb'
    CHAN_START=1731; CHAN_END=1901
else
    _fits_stem='40_014_25jul2021_gsb'
    CHAN_START=64; CHAN_END=191
fi

_DATA_TAG="$_GMRT_DATA_PROFILE"
FITS="${GMRT_CAL_FITS:-${DATA_DIR}/${_fits_stem}.FITS}"
_fits_root="$FITS"
_fits_root="${_fits_root%.FITS}"
_fits_root="${_fits_root%.fits}"
INDEX="${GMRT_INDEX_CACHE:-${_fits_root}.index.npz}"
_RAW_INDEX="$INDEX"
_CHAN_START_RAW="$CHAN_START"
_CHAN_END_RAW="$CHAN_END"
export GMRT_DATA_PROFILE="$_GMRT_DATA_PROFILE"
# ───────────────────────────────────────────────────────────────────────────
PC_DIR="$WORK_DIR/primary_calibration"    # all primary-cal outputs live here
BP_DIR="$PC_DIR/bandpass"                  # bandpass/gain table npz files
FLAG_DIR="$PC_DIR/flag"                    # flag table json files
DIAG_DIR="$WORK_DIR/diagnostics_out/primary/3c48/solve"  # primary-cal diagnostics (PNG/PDF)
RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$WORK_DIR/logs"
MPLBACKEND="Agg"  # Headless plotting (no live popups). Set to MacOSX (macOS) or TkAgg (Linux) to show live plot windows.

mkdir -p "$WORK_DIR" "$BP_DIR" "$FLAG_DIR" "$DIAG_DIR"
mkdir -p "$LOG_DIR"

if [[ -f "${BP_DIR}/${SRC_TAG}_bandpass_25jul_${_DATA_TAG}.npz" ]]; then
	echo "[primaryCalibration-example] WARNING: existing primary bandpass output will be replaced: ${BP_DIR}/${SRC_TAG}_bandpass_25jul_${_DATA_TAG}.npz"
fi

if compgen -G "${DIAG_DIR}/${SRC_TAG}_bandpass_*" > /dev/null; then
	echo "[primaryCalibration-example] WARNING: existing primary diagnostics will be replaced under: $DIAG_DIR"
fi

rm -f "${FLAG_DIR}/${SRC_TAG}_flag_table_session.json"

CMD=(
"$PYTHON_CMD" "$REPO_ROOT/src/pipeline_cli.py" preprocess --step all --no-dry-run --auto --n-iters 100
--docal on --doflag on --calver latest --flagver latest
--config preprocess_ugmrt.cfg
--set "WORK_DIR=Path('${WORK_DIR}')"
--set "SOURCE='${SOURCE}'"
--set "BANDPASS_OUT=Path('${BP_DIR}/${SRC_TAG}_bandpass_25jul_${_DATA_TAG}.npz')"
--set "DIAG_PLOT_BASE=Path('${DIAG_DIR}/${SRC_TAG}_bandpass_diagnostics.png')"
--set "DIAG_PLOT_UNFLAGGED=Path('${DIAG_DIR}/${SRC_TAG}_bandpass_diagnostics_unflagged.png')"
--set "GAIN_PLOT_BASE=Path('${DIAG_DIR}/${SRC_TAG}_bandpass_gains.png')"
--set "FLAG_TABLE_SESSION=Path('${FLAG_DIR}/${SRC_TAG}_flag_table_session.json')"
--set "SOLVE_ELEVATION_MIN_DEG=25.0"
--set "OUTLIER_METRIC='V'"
--set "OUTLIER_METRIC_MERGE_STRATEGY='union'"
--set "ANTENNA_FLAG_THRESHOLD_JY={'V': 5.0}"
--set "BASELINE_FLAG_THRESHOLD_JY={'V': 5.0}"
--set "COMPARE_METRICS_FOR_CONVERGENCE=['V','Model']"
--set "CONVERGENCE_MIN_ITERS=3"
--set "CONVERGENCE_EPSILON=0.005"
--set "CONVERGENCE_COMBINE_STRATEGY='any'"
--set "CLUSTERING_GLOBAL_ANT_FLAG_FRACTION=0.8"
--set "RUN_ITER0_DIAGNOSTIC=True"
--set "RUN_FINAL_CLUSTERING=True"
)

CMD_FILE="$LOG_DIR/primaryCalibration_example_${RUN_TS}.cmd"
LOG_FILE="$LOG_DIR/primaryCalibration_example_${RUN_TS}.log"
{
	echo "# timestamp=$RUN_TS"
	echo "# cwd=$PWD"
	echo "# MPLBACKEND=$MPLBACKEND"
	printf '%q ' "${CMD[@]}"
	printf '\n'
} > "$CMD_FILE"

echo "[primaryCalibration-example] cmd manifest: $CMD_FILE"
echo "[primaryCalibration-example] log: $LOG_FILE"
echo "[primaryCalibration-example] MPLBACKEND=$MPLBACKEND (headless; no live plot popups)"

MPLBACKEND="$MPLBACKEND" "${CMD[@]}" 2>&1 | tee "$LOG_FILE"

# Optional audit command for the same run outputs:
# python src/pipeline_cli.py bandpass --phase=audit --no-dry-run \
# --docal on --doflag on --calver latest --flagver latest \
# --config preprocess_ugmrt.cfg \
# --set "WORK_DIR=Path('${WORK_DIR}')" \
# --set "SOURCE='${SOURCE}'" \
# --set "BANDPASS_OUT=Path('${BP_DIR}/${SRC_TAG}_bandpass_25jul_gsb.npz')" \
# --set "FLAG_TABLE_SESSION=Path('${FLAG_DIR}/${SRC_TAG}_flag_table_session.json')"
