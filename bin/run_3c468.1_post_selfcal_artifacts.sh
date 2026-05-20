#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-$REPO_ROOT/gmrt/bin/python}"
WORK_SELFCAL_ROOT="${WORK_SELFCAL_ROOT:-$HOME/DATA/gmrt_40_014/work/casa_selfcal}"
SELFCAL_GLOB="${SELFCAL_GLOB:-3c468.1_scan*_stk*}"
RUN_ID="${RUN_ID:-}"
MOVIE_FPS="${MOVIE_FPS:-4}"
MOVIE_CMAP="${MOVIE_CMAP:-magma}"
MOVIE_PERCENTILE_LOW="${MOVIE_PERCENTILE_LOW:-5}"
MOVIE_PERCENTILE_HIGH="${MOVIE_PERCENTILE_HIGH:-99.5}"
FORCE_3C468_POST_SELFCAL_REBUILD="${FORCE_3C468_POST_SELFCAL_REBUILD:-${FORCE_3C468_POST_REBUILD:-0}}"

if [[ -n "${FORCE_3C468_POST_REBUILD:-}" && -z "${FORCE_3C468_POST_SELFCAL_REBUILD:-}" ]]; then
  echo "[3c468.1-post] INFO: FORCE_3C468_POST_REBUILD is deprecated; use FORCE_3C468_POST_SELFCAL_REBUILD"
fi

echo "[3c468.1-post] python            : $PYTHON"
echo "[3c468.1-post] selfcal root       : $WORK_SELFCAL_ROOT"
echo "[3c468.1-post] selfcal glob       : $SELFCAL_GLOB"
echo "[3c468.1-post] run id            : ${RUN_ID:-<unset>}"
echo "[3c468.1-post] force rebuild      : $FORCE_3C468_POST_SELFCAL_REBUILD"

shopt -s nullglob
DIRS=( "$WORK_SELFCAL_ROOT"/$SELFCAL_GLOB )
shopt -u nullglob

if (( ${#DIRS[@]} == 0 )); then
  echo "[3c468.1-post] WARNING: no selfcal directories matched: $WORK_SELFCAL_ROOT/$SELFCAL_GLOB"
  exit 0
fi

for d in "${DIRS[@]}"; do
  [[ -d "$d" ]] || continue
  echo "[3c468.1-post] --- processing: $d"

  shopt -s nullglob
  fits=( "$d"/*_final.fits "$d"/*/*_final.fits )
  shopt -u nullglob
  if (( ${#fits[@]} == 0 )); then
    echo "[3c468.1-post] WARNING: no *_final.fits found in $d, skipping"
    continue
  fi

  mp4="$d/3c468.1_selfcal_movie.mp4"
  gif="$d/3c468.1_selfcal_movie.gif"
  mov="$d/3c468.1_selfcal_movie.mov"

  if [[ "$FORCE_3C468_POST_SELFCAL_REBUILD" != "1" && -f "$mp4" && -f "$gif" && -f "$mov" ]]; then
    echo "[3c468.1-post] SKIP movie (already exists): $mp4"
  else
    SCAN="3C468.1" \
    SELFCAL_DIR="$d" \
    FPS="$MOVIE_FPS" \
    CMAP="$MOVIE_CMAP" \
    PERCENTILE_LOW="$MOVIE_PERCENTILE_LOW" \
    PERCENTILE_HIGH="$MOVIE_PERCENTILE_HIGH" \
    SCALE_DIRS="$d" \
    GLOB_PATTERN="*_final.fits" \
    PYTHON_CMD="$PYTHON" \
    bash "$REPO_ROOT/bin/run_moon_selfcal_movie_dev.sh"
  fi

  stack_out="$d/3c468.1_stack_mean.fits"
  if [[ "$FORCE_3C468_POST_SELFCAL_REBUILD" != "1" && -f "$stack_out" ]]; then
    echo "[3c468.1-post] SKIP stack (already exists): $stack_out"
  else
    "$PYTHON" "$REPO_ROOT/src/stack_moon_snapshots.py" \
      --selfcal-dir "$d" \
      --glob "*_final.fits" \
      --registration-mode none \
      --method mean \
      --output "$stack_out"
  fi

  rms_png="$d/3c468.1_cumulative_rms_evolution.png"
  rms_csv="$d/3c468.1_cumulative_rms_evolution.csv"
  if [[ "$FORCE_3C468_POST_SELFCAL_REBUILD" != "1" && -f "$rms_png" && -f "$rms_csv" ]]; then
    echo "[3c468.1-post] SKIP RMS progression (already exists): $rms_png"
  else
    "$PYTHON" "$REPO_ROOT/src/moon_cumulative_coadd_movie.py" \
      --selfcal-dir "$d" \
      --glob "*_final.fits" \
      --registration-mode none \
      --out-rms-png "$rms_png" \
      --out-rms-csv "$rms_csv" \
      --source-name "3C468.1" \
      --rms-strategy image-center \
      --source-mask-radius-arcmin 10.0 \
      2>&1 | sed 's/^/[3c468.1-rms] /'
  fi

  # Generic diagnostics for the exact active run (JSON-only, no CASA log parsing).
  if [[ -n "$RUN_ID" ]]; then
    scan_tag=""
    if [[ "$d" =~ (scan[0-9][0-9]) ]]; then
      scan_tag="${BASH_REMATCH[1]}"
    fi

    diag_csv="$d/scan05_clean_cycle_metrics_per_integration.csv"
    diag_stats="$d/scan05_clean_cycle_metrics_summary_stats.csv"
    diag_png="$d/scan05_clean_cycle_metrics_panel_5x5.png"

    "$PYTHON" "$REPO_ROOT/src/selfcal_diagnostics.py" plot-micro \
      --root "$WORK_SELFCAL_ROOT" \
      --run-id "$RUN_ID" \
      ${scan_tag:+--scan-tag "$scan_tag"} \
      --selfcal-name "$(basename "$d")" \
      --require-single-integration \
      --out-panel-png "$diag_png" \
      --out-stats-csv "$diag_stats" \
      --out-csv "$diag_csv" \
      --title-prefix "$(basename "$d")" \
      2>&1 | sed 's/^/[3c468.1-diag] /'
  fi
done

echo "[3c468.1-post] done"
