#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORK_DIR="${WORK_DIR:-$HOME/DATA/gmrt_40_014/work}"
MANIFEST_PATH="${1:-}"
STRICT_LAYOUT_COMPAT="${STRICT_LAYOUT_COMPAT:-0}"
LAYOUT_COMPAT_REF="${LAYOUT_COMPAT_REF:-HEAD~1}"

if [[ -z "$MANIFEST_PATH" ]]; then
  MANIFEST_PATH="$(ls -1t "$WORK_DIR"/logs/run_gmrt_40_014_products_*.txt 2>/dev/null | head -1 || true)"
fi

[[ -n "$MANIFEST_PATH" ]] || { echo "No workflow manifest found" >&2; exit 2; }
[[ -f "$MANIFEST_PATH" ]] || { echo "Workflow manifest missing: $MANIFEST_PATH" >&2; exit 2; }

RUN_TS="$(basename "$MANIFEST_PATH" | sed -E 's/^run_gmrt_40_014_products_([0-9]{8}_[0-9]{6})\.txt$/\1/')"
[[ "$RUN_TS" =~ ^[0-9]{8}_[0-9]{6}$ ]] || { echo "Could not parse RUN_TS from $MANIFEST_PATH" >&2; exit 2; }

TEST_ROOT="${TEST_ROOT:-$REPO_ROOT/work/manifest_equivalence_${RUN_TS}}"
BASELINE_SCRIPT="$TEST_ROOT/publish_baseline.sh"
LAYOUT_MANIFEST="$TEST_ROOT/moon_layout_manifest.json"
TRACE_FILE="$TEST_ROOT/manifest_trace.jsonl"
PAGES_BASELINE="$TEST_ROOT/pages_baseline"
PAGES_MANIFEST="$TEST_ROOT/pages_manifest"
PAGES_MANIFEST_SABOTAGE="$TEST_ROOT/pages_manifest_sabotage"
REPORT_FILE="$TEST_ROOT/equivalence_report.txt"
FAKEBIN="$TEST_ROOT/fakebin"
FIXED_PUBLISHED_AT="$(${DATE_BIN:-/bin/date} '+%Y-%m-%d %H:%M:%S %Z')"

mkdir -p "$TEST_ROOT"
mkdir -p "$FAKEBIN"

cat > "$FAKEBIN/date" <<EOF
#!/usr/bin/env bash
if [[ "\$#" -eq 1 && "\$1" == "+%Y-%m-%d %H:%M:%S %Z" ]]; then
  echo "$FIXED_PUBLISHED_AT"
  exit 0
fi
exec /bin/date "\$@"
EOF
chmod +x "$FAKEBIN/date"

if [[ "$STRICT_LAYOUT_COMPAT" == "1" ]]; then
  if ! git -C "$REPO_ROOT" cat-file -e "${LAYOUT_COMPAT_REF}:bin/publish_gh_pages.sh" 2>/dev/null; then
    echo "Layout compat ref missing script: ${LAYOUT_COMPAT_REF}:bin/publish_gh_pages.sh" >&2
    exit 2
  fi
  git -C "$REPO_ROOT" show "${LAYOUT_COMPAT_REF}:bin/publish_gh_pages.sh" > "$BASELINE_SCRIPT"
  BASELINE_SOURCE="$LAYOUT_COMPAT_REF"
else
  cp -f "$REPO_ROOT/bin/publish_gh_pages.sh" "$BASELINE_SCRIPT"
  BASELINE_SOURCE="WORKTREE"
fi
chmod +x "$BASELINE_SCRIPT"
python3 "$REPO_ROOT/bin/build_moon_layout_manifest.py" --work-dir "$WORK_DIR" --run-ts "$RUN_TS" --output "$LAYOUT_MANIFEST"

BASELINE_BRANCH="gh-pages-baseline-${RUN_TS//_/}-$$"
MANIFEST_BRANCH="gh-pages-manifest-${RUN_TS//_/}-$$"
MANIFEST_SAB_BRANCH="gh-pages-manifest-sab-${RUN_TS//_/}-$$"

GHPAGES_LAYOUT_MANIFEST="$LAYOUT_MANIFEST" \
PATH="$FAKEBIN:$PATH" bash "$BASELINE_SCRIPT" \
  --manifest "$MANIFEST_PATH" \
  --work-dir "$WORK_DIR" \
  --pages-dir "$PAGES_BASELINE" \
  --branch "$BASELINE_BRANCH" \
  --no-open

MANIFEST_ONLY=1 \
GHPAGES_LAYOUT_MANIFEST="$LAYOUT_MANIFEST" \
MANIFEST_TRACE_PATH="$TRACE_FILE" \
PATH="$FAKEBIN:$PATH" bash "$REPO_ROOT/bin/publish_gh_pages.sh" \
  --manifest "$MANIFEST_PATH" \
  --work-dir "$WORK_DIR" \
  --pages-dir "$PAGES_MANIFEST" \
  --branch "$MANIFEST_BRANCH" \
  --no-open

BASELINE_INDEX="$PAGES_BASELINE/40_014/latest/index.html"
MANIFEST_INDEX="$PAGES_MANIFEST/40_014/latest/index.html"
[[ -f "$BASELINE_INDEX" ]] || { echo "Baseline index missing: $BASELINE_INDEX" >&2; exit 2; }
[[ -f "$MANIFEST_INDEX" ]] || { echo "Manifest index missing: $MANIFEST_INDEX" >&2; exit 2; }

BASE_SHA="$(shasum -a 256 "$BASELINE_INDEX" | awk '{print $1}')"
MAN_SHA="$(shasum -a 256 "$MANIFEST_INDEX" | awk '{print $1}')"

set +e
cmp -s "$BASELINE_INDEX" "$MANIFEST_INDEX"
CMP_STATUS=$?
set -e

{
  echo "RUN_TS=$RUN_TS"
  echo "MANIFEST_PATH=$MANIFEST_PATH"
  echo "STRICT_LAYOUT_COMPAT=$STRICT_LAYOUT_COMPAT"
  echo "LAYOUT_COMPAT_REF=$LAYOUT_COMPAT_REF"
  echo "BASELINE_SOURCE=$BASELINE_SOURCE"
  echo "LAYOUT_MANIFEST=$LAYOUT_MANIFEST"
  echo "TRACE_FILE=$TRACE_FILE"
  echo "BASELINE_INDEX=$BASELINE_INDEX"
  echo "MANIFEST_INDEX=$MANIFEST_INDEX"
  echo "BASELINE_SHA256=$BASE_SHA"
  echo "MANIFEST_SHA256=$MAN_SHA"
  echo "CMP_STATUS=$CMP_STATUS"
} > "$REPORT_FILE"

if [[ "$CMP_STATUS" -ne 0 ]]; then
  diff -u "$BASELINE_INDEX" "$MANIFEST_INDEX" > "$TEST_ROOT/index.diff" || true
  if [[ "$STRICT_LAYOUT_COMPAT" == "1" ]]; then
    echo "LAYOUT_COMPAT=FAIL" >> "$REPORT_FILE"
  fi
  echo "BYTE_EQ=FAIL" >> "$REPORT_FILE"
  echo "Diff written: $TEST_ROOT/index.diff" >> "$REPORT_FILE"
  cat "$REPORT_FILE"
  exit 1
fi

if [[ "$STRICT_LAYOUT_COMPAT" == "1" ]]; then
  echo "LAYOUT_COMPAT=PASS" >> "$REPORT_FILE"
else
  echo "LAYOUT_COMPAT=SKIP" >> "$REPORT_FILE"
fi

echo "BYTE_EQ=PASS" >> "$REPORT_FILE"

# Negative control: mutate one byte and prove detector fails.
NEG_FILE="$TEST_ROOT/index.mutated.html"
cp "$MANIFEST_INDEX" "$NEG_FILE"
python3 - "$NEG_FILE" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
b = bytearray(p.read_bytes())
if not b:
    raise SystemExit(2)
b[0] = (b[0] + 1) % 256
p.write_bytes(bytes(b))
PY

set +e
cmp -s "$BASELINE_INDEX" "$NEG_FILE"
NEG_CMP_STATUS=$?
set -e
if [[ "$NEG_CMP_STATUS" -eq 0 ]]; then
  echo "NEGATIVE_CONTROL=FAIL" >> "$REPORT_FILE"
  cat "$REPORT_FILE"
  exit 1
fi
echo "NEGATIVE_CONTROL=PASS" >> "$REPORT_FILE"

# Sabotage check: hide legacy paths if present and rerun MANIFEST_ONLY renderer.
LEGACY_NO="$WORK_DIR/casa_selfcal/ghpages_products/no_phasecenter"
LEGACY_PC="$WORK_DIR/casa_selfcal/ghpages_products/phasecenter"
HIDDEN_NO=""
HIDDEN_PC=""
restore_legacy() {
  if [[ -n "$HIDDEN_NO" && -d "$HIDDEN_NO" ]]; then mv "$HIDDEN_NO" "$LEGACY_NO"; fi
  if [[ -n "$HIDDEN_PC" && -d "$HIDDEN_PC" ]]; then mv "$HIDDEN_PC" "$LEGACY_PC"; fi
}
trap restore_legacy EXIT

if [[ -d "$LEGACY_NO" ]]; then
  HIDDEN_NO="$LEGACY_NO.sabotage.$$"
  mv "$LEGACY_NO" "$HIDDEN_NO"
fi
if [[ -d "$LEGACY_PC" ]]; then
  HIDDEN_PC="$LEGACY_PC.sabotage.$$"
  mv "$LEGACY_PC" "$HIDDEN_PC"
fi

MANIFEST_ONLY=1 \
GHPAGES_LAYOUT_MANIFEST="$LAYOUT_MANIFEST" \
MANIFEST_TRACE_PATH="$TEST_ROOT/manifest_trace_sabotage.jsonl" \
PATH="$FAKEBIN:$PATH" bash "$REPO_ROOT/bin/publish_gh_pages.sh" \
  --manifest "$MANIFEST_PATH" \
  --work-dir "$WORK_DIR" \
  --pages-dir "$PAGES_MANIFEST_SABOTAGE" \
  --branch "$MANIFEST_SAB_BRANCH" \
  --no-open

restore_legacy
trap - EXIT

SAB_INDEX="$PAGES_MANIFEST_SABOTAGE/40_014/latest/index.html"
[[ -f "$SAB_INDEX" ]] || { echo "Sabotage index missing: $SAB_INDEX" >&2; exit 1; }

set +e
cmp -s "$BASELINE_INDEX" "$SAB_INDEX"
SAB_CMP_STATUS=$?
set -e
if [[ "$SAB_CMP_STATUS" -ne 0 ]]; then
  echo "SABOTAGE_CHECK=FAIL" >> "$REPORT_FILE"
  diff -u "$BASELINE_INDEX" "$SAB_INDEX" > "$TEST_ROOT/index_sabotage.diff" || true
  cat "$REPORT_FILE"
  exit 1
fi

echo "SABOTAGE_CHECK=PASS" >> "$REPORT_FILE"
cat "$REPORT_FILE"
