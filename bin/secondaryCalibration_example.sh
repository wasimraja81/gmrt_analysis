#!/usr/bin/env bash

# Secondary calibration example (pipeline_cli interface)
# - Uses config defaults from preprocess_ugmrt.cfg
# - Updates key params through CLI + --set (same style as clustering_example.sh)
# - Uses split-clustering-derived flag table via --flagver latest
# - Uses input primary bandpass table via --bpcal
# - No row cap: --max-rows 0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# PYTHON_CMD is set in the machine-specific block below

cd "$REPO_ROOT"

SOURCE="3C468.1"
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
SC_DIR="$WORK_DIR/secondary_calibration"  # all secondary-cal outputs live here
FLAG_DIR="$SC_DIR/flag"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$WORK_DIR/logs"
mkdir -p "$SC_DIR" "$FLAG_DIR" "$LOG_DIR"

if compgen -G "$SC_DIR/${SRC_TAG}_secondary_phase_only_scan*.npz" > /dev/null; then
	echo "[secondaryCalibration-example] WARNING: existing secondary scan tables will be replaced under: $SC_DIR"
fi

# Primary table is an INPUT transfer table (can remain 3C48-derived if desired)
PRIMARY_BPCAL="$WORK_DIR/primary_calibration/bandpass/3c48_bandpass_25jul_${_DATA_TAG}_iterfinal_clustering.npz"

CMD=(
"$PYTHON_CMD" "$REPO_ROOT/src/pipeline_cli.py" secondary
--config preprocess_ugmrt.cfg
--mode phase_only
--flagver latest
--solint-mode scan
--max-rows 0
--set "WORK_DIR=Path('$WORK_DIR')"
--set "SOURCE='$SOURCE'"
--set "FLAG_TABLE_SESSION=Path('$FLAG_DIR/${SRC_TAG}_split_clustering_flag_table_session.json')"
--bpcal "$PRIMARY_BPCAL"
--out "$SC_DIR/${SRC_TAG}_secondary_phase_only.npz"
)

CMD_FILE="$LOG_DIR/secondaryCalibration_example_${RUN_TS}.cmd"
LOG_FILE="$LOG_DIR/secondaryCalibration_example_${RUN_TS}.log"
{
	echo "# timestamp=$RUN_TS"
	echo "# cwd=$PWD"
	printf '%q ' "${CMD[@]}"
	printf '\n'
} > "$CMD_FILE"

echo "[secondaryCalibration-example] cmd manifest: $CMD_FILE"
echo "[secondaryCalibration-example] log: $LOG_FILE"

"${CMD[@]}" 2>&1 | tee "$LOG_FILE"

# Alternate examples:
# --mode delay_phase --solint scan --out ~/DATA/gmrt_40_014/work/3c468.1_secondary_delay_phase.npz
# --solint all (single solution over full selected timerange)
# --solint-mode minutes --solint-minutes 10  (fixed-width bins; can span scan boundaries)
# --scan-gap-minutes 2  (optional override for --solint-mode scan; default auto-derived)
