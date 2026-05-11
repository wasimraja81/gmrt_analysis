#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_CMD="${PYTHON_CMD:-python}"

cd "$REPO_ROOT"

# Clustering for 3C468.1 on raw multi-source FITS using on-the-fly primary calibration
# Run this after primaryCalibration_example.sh and before secondaryCalibration_example.sh

SOURCE="3C468.1"
SRC_TAG="$(printf '%s' "$SOURCE" | tr '[:upper:]' '[:lower:]')"

WORK_DIR="$HOME/DATA/gmrt_40_014/work"
RAW_FITS="$HOME/DATA/gmrt_40_014/data/40_014_25jul2021_gsb.FITS"
PRIMARY_BPCAL="$WORK_DIR/primary_calibration/bandpass/3c48_bandpass_25jul_gsb_iterfinal_clustering.npz"
PRIMARY_FLAG_TABLE="$WORK_DIR/primary_calibration/flag/3c48_flag_table_session.json"

SC_DIR="$WORK_DIR/secondary_calibration"
CLUSTER_DIR="$SC_DIR/clustering"
FLAG_DIR="$SC_DIR/flag"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$WORK_DIR/logs"
MPLBACKEND="Agg"

mkdir -p "$CLUSTER_DIR" "$FLAG_DIR" "$LOG_DIR"

FLAG_TABLE_SESSION="$FLAG_DIR/${SRC_TAG}_split_clustering_flag_table_session.json"

if [[ -f "$FLAG_TABLE_SESSION" ]]; then
  echo "[clustering-3c468.1-split] WARNING: existing split-clustering flag table will be overwritten: $FLAG_TABLE_SESSION"
fi

if [[ ! -f "$RAW_FITS" ]]; then
  echo "Raw FITS not found: $RAW_FITS"
  exit 1
fi

if [[ ! -f "$PRIMARY_BPCAL" ]]; then
  echo "Primary bandpass table not found: $PRIMARY_BPCAL"
  exit 1
fi

if [[ ! -f "$PRIMARY_FLAG_TABLE" ]]; then
  echo "Primary flag table not found: $PRIMARY_FLAG_TABLE"
  exit 1
fi

CMD=(
"$PYTHON_CMD" "$REPO_ROOT/src/pipeline_cli.py" clustering
--config preprocess_ugmrt.cfg
--no-dry-run
--docal on
--doflag on
--save-plots
--set "WORK_DIR=Path('${CLUSTER_DIR}')"
--set "CAL_FITS=Path('${RAW_FITS}')"
--set "SOURCE='${SOURCE}'"
--set "BANDPASS_OUT=Path('${PRIMARY_BPCAL}')"
--set "FLAG_TABLE_PATHS=[Path('${PRIMARY_FLAG_TABLE}')]"
--set "FLAG_TABLE_SESSION=Path('${FLAG_TABLE_SESSION}')"
--set "MAX_ROWS_SOLVE=None"
--set "SOLVE_ELEVATION_MIN_DEG=25.0"
--set "CLUSTERING_CORR=['V','RR','LL']"
--set "CLUSTERING_THRESHOLD_JY={'V':5.0,'RR':50.0,'LL':50.0}"
--set "CLUSTERING_THRESHOLD_LOW_JY={'RR':10.0,'LL':10.0}"
--set "CLUSTERING_GLOBAL_ANT_FLAG_FRACTION=0.80"
--set "CHAN_RANGE=(64,191)"
--set "PLOT_CHAN_RANGE=(64,191)"
)

CMD_FILE="$LOG_DIR/clustering_3c468.1_split_example_${RUN_TS}.cmd"
LOG_FILE="$LOG_DIR/clustering_3c468.1_split_example_${RUN_TS}.log"
{
  echo "# timestamp=$RUN_TS"
  echo "# cwd=$PWD"
  echo "# MPLBACKEND=$MPLBACKEND"
  printf '%q ' "${CMD[@]}"
  printf '\n'
} > "$CMD_FILE"

echo "[clustering-3c468.1-split] cmd manifest: $CMD_FILE"
echo "[clustering-3c468.1-split] log: $LOG_FILE"
echo "[clustering-3c468.1-split] raw fits input: $RAW_FITS"
echo "[clustering-3c468.1-split] flag table out: $FLAG_TABLE_SESSION"

MPLBACKEND="$MPLBACKEND" "${CMD[@]}" 2>&1 | tee "$LOG_FILE"

echo "[clustering-3c468.1-split] done"
