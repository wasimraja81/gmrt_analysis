#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORK_DIR="${WORK_DIR:-$HOME/DATA/gmrt_40_014/work}"
MANIFEST_PATH="${1:-}"
STRICT_DATA="${STRICT_DATA:-1}"
STRICT_LAYOUT_COMPAT="${STRICT_LAYOUT_COMPAT:-0}"
LAYOUT_COMPAT_REF="${LAYOUT_COMPAT_REF:-HEAD~1}"

find_latest_manifest() {
  ls -1t "$WORK_DIR"/logs/run_gmrt_40_014_products_*.txt 2>/dev/null | head -1 || true
}

if [[ -z "$MANIFEST_PATH" ]]; then
  MANIFEST_PATH="$(find_latest_manifest)"
fi

if [[ -z "$MANIFEST_PATH" || ! -f "$MANIFEST_PATH" ]]; then
  if [[ "$STRICT_DATA" == "0" ]]; then
    echo "[moon-manifest-ci] SKIP: no workflow manifest found under $WORK_DIR/logs"
    exit 0
  fi
  echo "[moon-manifest-ci] FAIL: workflow manifest missing (set STRICT_DATA=0 to allow skip)" >&2
  exit 2
fi

RUN_TS="$(basename "$MANIFEST_PATH" | sed -E 's/^run_gmrt_40_014_products_([0-9]{8}_[0-9]{6})\.txt$/\1/')"
[[ "$RUN_TS" =~ ^[0-9]{8}_[0-9]{6}$ ]] || {
  echo "[moon-manifest-ci] FAIL: could not parse RUN_TS from $MANIFEST_PATH" >&2
  exit 2
}

REPORT_FILE="${TEST_ROOT:-$REPO_ROOT/work/manifest_equivalence_${RUN_TS}}/equivalence_report.txt"

echo "[moon-manifest-ci] RUN_TS=$RUN_TS"
echo "[moon-manifest-ci] MANIFEST_PATH=$MANIFEST_PATH"
echo "[moon-manifest-ci] STRICT_LAYOUT_COMPAT=$STRICT_LAYOUT_COMPAT"
if [[ "$STRICT_LAYOUT_COMPAT" == "1" ]]; then
  echo "[moon-manifest-ci] LAYOUT_COMPAT_REF=$LAYOUT_COMPAT_REF"
fi

env WORK_DIR="$WORK_DIR" \
    STRICT_LAYOUT_COMPAT="$STRICT_LAYOUT_COMPAT" \
    LAYOUT_COMPAT_REF="$LAYOUT_COMPAT_REF" \
    bash "$REPO_ROOT/bin/test_moon_manifest_equivalence.sh" "$MANIFEST_PATH"

[[ -f "$REPORT_FILE" ]] || {
  echo "[moon-manifest-ci] FAIL: report not found: $REPORT_FILE" >&2
  exit 2
}

grep -q '^BYTE_EQ=PASS$' "$REPORT_FILE" || { echo "[moon-manifest-ci] FAIL: BYTE_EQ not PASS" >&2; exit 1; }
grep -q '^NEGATIVE_CONTROL=PASS$' "$REPORT_FILE" || { echo "[moon-manifest-ci] FAIL: NEGATIVE_CONTROL not PASS" >&2; exit 1; }
grep -q '^SABOTAGE_CHECK=PASS$' "$REPORT_FILE" || { echo "[moon-manifest-ci] FAIL: SABOTAGE_CHECK not PASS" >&2; exit 1; }
if [[ "$STRICT_LAYOUT_COMPAT" == "1" ]]; then
  grep -q '^LAYOUT_COMPAT=PASS$' "$REPORT_FILE" || { echo "[moon-manifest-ci] FAIL: LAYOUT_COMPAT not PASS" >&2; exit 1; }
fi

echo "[moon-manifest-ci] PASS: Moon manifest regression gate passed"
echo "[moon-manifest-ci] REPORT=$REPORT_FILE"
