#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

WORK_DIR="$HOME/DATA/gmrt_40_014/work"
LOG_DIR="$WORK_DIR/logs"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
CHAIN_LOG="$LOG_DIR/run_gmrt_40_014_calibration_workflow_${RUN_TS}.log"

PRIMARY_BANDPASS="$WORK_DIR/primary_calibration/bandpass/3c48_bandpass_25jul_gsb_iterfinal_clustering.npz"
PRIMARY_FLAGS="$WORK_DIR/primary_calibration/flag/3c48_flag_table_session.json"
SPLIT_3C48="$WORK_DIR/split/3c48/3c48_calibrated_flagged.uvfits"
PLOT_3C48_PDF="$WORK_DIR/diagnostics_out/primary/3c48/selfcheck/plotvis_3c48.pdf"
SPLIT_3C468_PRIMARY="$WORK_DIR/split/3c468.1/3c468.1_primary_calibrated.uvfits"
PLOT_3C468_PRIMARY_PDF="$WORK_DIR/diagnostics_out/secondary/3c468.1/transfer/plotvis_3c468.1.pdf"
CLUSTER_FLAGS="$WORK_DIR/secondary_calibration/flag/3c468.1_split_clustering_flag_table_session.json"
CLUSTER_PRODUCTS_GLOB="$WORK_DIR/diagnostics_out/secondary/3c468.1/flagging/3c468.1_clustering_*"
SECONDARY_TABLES_GLOB="$WORK_DIR/secondary_calibration/3c468.1_secondary_phase_only_scan*.npz"
SPLIT_3C468_FINAL="$WORK_DIR/split/3c468.1/3c468.1_primary_secondary_calibrated_flagged.uvfits"
PLOT_3C468_FINAL_PDF="$WORK_DIR/diagnostics_out/secondary/3c468.1/final_qa/plotvis_3c468.1.pdf"

AUDIT_ONLY=false

usage() {
  cat <<'USAGE'
Usage:
  bash bin/run_gmrt_40_014_calibration_workflow.sh            # run full 10-step workflow
  bash bin/run_gmrt_40_014_calibration_workflow.sh --audit-only

Options:
  --audit-only   Do not run calibration; only verify and print expected output products.
USAGE
}

if (( $# > 1 )); then
  usage
  exit 2
fi

if (( $# == 1 )); then
  case "$1" in
    --audit-only) AUDIT_ONLY=true ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 2
      ;;
  esac
fi

mkdir -p "$LOG_DIR"

log() {
  echo "[$(date +%F' '%T)] $*" | tee -a "$CHAIN_LOG"
}

on_error() {
  local exit_code=$?
  log "ABORT: current step failed (exit=${exit_code}). No further steps will run."
  exit "$exit_code"
}

trap on_error ERR

run_step() {
  local step_no="$1"
  local step_name="$2"
  shift 2

  log "STEP ${step_no}/10 START: ${step_name}"
  log "COMMAND: $*"
  "$@" 2>&1 | tee -a "$CHAIN_LOG"
  log "STEP ${step_no}/10 DONE : ${step_name}"
}

assert_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    log "CHECK FAILED: expected file not found: $path"
    return 1
  fi
  log "CHECK OK: $path"
}

assert_glob() {
  local pattern="$1"
  shopt -s nullglob
  local matches=( $pattern )
  shopt -u nullglob

  if (( ${#matches[@]} == 0 )); then
    log "CHECK FAILED: no files matched pattern: $pattern"
    return 1
  fi

  log "CHECK OK: ${#matches[@]} file(s) matched: $pattern"
}

print_products_to_audit() {
  local manifest="$LOG_DIR/run_gmrt_40_014_products_${RUN_TS}.txt"

  {
    echo "GMRT 40_014 products to audit"
    echo "Generated: $(date '+%F %T')"
    echo
    echo "Workflow log:"
    echo "$CHAIN_LOG"
    echo
    echo "Step 2: primary calibration"
    echo "$PRIMARY_BANDPASS"
    echo "$PRIMARY_FLAGS"
    echo
    echo "Step 3: primary split"
    echo "$SPLIT_3C48"
    echo
    echo "Step 4: primary vis plot"
    echo "$PLOT_3C48_PDF"
    echo
    echo "Step 5: 3C468.1 primary-only split"
    echo "$SPLIT_3C468_PRIMARY"
    echo
    echo "Step 6: 3C468.1 primary-only plot"
    echo "$PLOT_3C468_PRIMARY_PDF"
    echo
    echo "Step 7: clustering flags + products"
    echo "$CLUSTER_FLAGS"
    ls -1 $CLUSTER_PRODUCTS_GLOB 2>/dev/null || true
    echo
    echo "Step 8: secondary calibration tables"
    ls -1 $SECONDARY_TABLES_GLOB 2>/dev/null || true
    echo
    echo "Step 9: 3C468.1 primary+secondary split"
    echo "$SPLIT_3C468_FINAL"
    echo
    echo "Step 10: final 3C468.1 plot"
    echo "$PLOT_3C468_FINAL_PDF"
  } > "$manifest"

  log "PRODUCTS TO AUDIT (full paths):"
  while IFS= read -r line; do
    log "  $line"
  done < "$manifest"
  log "Product manifest file: $manifest"
}

log "GMRT 40_014 calibration workflow start"
log "cwd=$PWD"
log "chain_log=$CHAIN_LOG"

if [[ "$AUDIT_ONLY" == true ]]; then
  log "Running in --audit-only mode (no calibration steps will be executed)."
  assert_file "$PRIMARY_BANDPASS"
  assert_file "$PRIMARY_FLAGS"
  assert_file "$SPLIT_3C48"
  assert_file "$PLOT_3C48_PDF"
  assert_file "$SPLIT_3C468_PRIMARY"
  assert_file "$PLOT_3C468_PRIMARY_PDF"
  assert_file "$CLUSTER_FLAGS"
  assert_glob "$CLUSTER_PRODUCTS_GLOB"
  assert_glob "$SECONDARY_TABLES_GLOB"
  assert_file "$SPLIT_3C468_FINAL"
  assert_file "$PLOT_3C468_FINAL_PDF"
  print_products_to_audit
  log "Audit-only check complete"
  exit 0
fi

# 1) clean
run_step 1 "clean work outputs" bash "$SCRIPT_DIR/clean_work_outputs.sh" --yes

# 2) primary cal
run_step 2 "primary calibration" bash "$SCRIPT_DIR/primaryCalibration_example.sh"
assert_file "$PRIMARY_BANDPASS"
assert_file "$PRIMARY_FLAGS"

# 3) split primary itself (3C48 with primary corrections + primary flags)
run_step 3 "split 3C48 with primary calibration+flags" bash "$SCRIPT_DIR/visSplit_3c48_example.sh"
assert_file "$SPLIT_3C48"

# 4) plot visibilities for primary cal
run_step 4 "plot visibilities for primary calibration" bash "$SCRIPT_DIR/plotVis_3c48_example.sh"
assert_file "$PLOT_3C48_PDF"

# 5) split 3c468.1 (applying primary-only corrections and flags)
run_step 5 "split 3C468.1 with primary-only corrections+flags" bash "$SCRIPT_DIR/visSplit_3c468.1_primary_example.sh"
assert_file "$SPLIT_3C468_PRIMARY"

# 6) plot vis for step-5
run_step 6 "plot visibilities for primary-only 3C468.1 split" bash "$SCRIPT_DIR/plotVis_3c468.1_primary_example.sh"
assert_file "$PLOT_3C468_PRIMARY_PDF"

# 7) derive new flags for primary-cal applied 3c468.1 using clustering
#    (script already generates before/after plots + summary PDF)
run_step 7 "derive clustering flags for 3C468.1" bash "$SCRIPT_DIR/clustering_3c468.1_split_example.sh"
assert_file "$CLUSTER_FLAGS"
assert_glob "$CLUSTER_PRODUCTS_GLOB"

# 8) derive secondary calibration using new clustering flags
run_step 8 "secondary calibration using clustering flags" bash "$SCRIPT_DIR/secondaryCalibration_example.sh"
assert_glob "$SECONDARY_TABLES_GLOB"

# 9) generate primary+secondary calibrated 3c468.1
run_step 9 "split 3C468.1 with primary+secondary calibration" bash "$SCRIPT_DIR/visSplit_3c468.1_example.sh"
assert_file "$SPLIT_3C468_FINAL"

# 10) plot vis on step-9 output
run_step 10 "plot visibilities for primary+secondary 3C468.1 split" bash "$SCRIPT_DIR/plotVis_3c468.1_example.sh"
assert_file "$PLOT_3C468_FINAL_PDF"

log "GMRT 40_014 calibration workflow complete"
print_products_to_audit
