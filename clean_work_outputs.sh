#!/usr/bin/env bash

set -euo pipefail

WORK_DIR="$HOME/DATA/gmrt_40_014/work"
KEEP_LOGS=1
CONFIRM=0

usage() {
  cat <<'EOF'
Usage: clean_work_outputs.sh --yes [--include-logs]

Safely resets generated outputs under ~/DATA/gmrt_40_014/work.

Default behavior:
  - Removes generated science products and caches
  - Preserves work/logs/ (all .cmd/.log history)

Options:
  --yes           Required confirmation flag
  --include-logs  Also delete work/logs/ (dangerous)
  -h, --help      Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --yes)
      CONFIRM=1
      shift
      ;;
    --include-logs)
      KEEP_LOGS=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 2
      ;;
  esac
done

if [[ $CONFIRM -ne 1 ]]; then
  echo "Refusing to clean without explicit confirmation. Use: --yes"
  exit 2
fi

echo "[clean-work] target: $WORK_DIR"
if [[ $KEEP_LOGS -eq 1 ]]; then
  echo "[clean-work] preserving logs: $WORK_DIR/logs"
else
  echo "[clean-work] WARNING: logs will also be removed"
fi

mkdir -p "$WORK_DIR"

rm -rf "$WORK_DIR/primary_calibration"
rm -rf "$WORK_DIR/secondary_calibration"
rm -rf "$WORK_DIR/split"
rm -rf "$WORK_DIR/diagnostics_out"
rm -f "$WORK_DIR/40_014_25jul2021_gsb.index.npz"

if [[ $KEEP_LOGS -eq 0 ]]; then
  rm -rf "$WORK_DIR/logs"
fi

mkdir -p "$WORK_DIR"
mkdir -p "$WORK_DIR/logs"

echo "[clean-work] done"
