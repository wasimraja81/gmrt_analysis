#!/usr/bin/env bash

# Secondary calibration example (pipeline_cli interface)
# - Uses config defaults from preprocess_ugmrt.cfg
# - Updates key params through CLI + --set (same style as clustering_example.sh)
# - Uses split-clustering-derived flag table via --flagver latest
# - Uses input primary bandpass table via --bpcal
# - No row cap: --max-rows 0

SOURCE="3C468.1"
SRC_TAG="$(printf '%s' "$SOURCE" | tr '[:upper:]' '[:lower:]')"
WORK_DIR="$HOME/DATA/gmrt_40_014/work"
SC_DIR="$WORK_DIR/secondary_calibration"  # all secondary-cal outputs live here
FLAG_DIR="$SC_DIR/flag"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$WORK_DIR/logs"
mkdir -p "$SC_DIR" "$FLAG_DIR" "$LOG_DIR"

if compgen -G "$SC_DIR/${SRC_TAG}_secondary_phase_only_scan*.npz" > /dev/null; then
	echo "[secondaryCalibration-example] WARNING: existing secondary scan tables will be replaced under: $SC_DIR"
fi

# Primary table is an INPUT transfer table (can remain 3C48-derived if desired)
PRIMARY_BPCAL=~/DATA/gmrt_40_014/work/primary_calibration/bandpass/3c48_bandpass_25jul_gsb_iterfinal_clustering.npz

CMD=(
python pipeline_cli.py secondary
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
