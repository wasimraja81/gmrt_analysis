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
MOON_SOURCES=(MOON0520 MOON0545 MOON0605 MOON0625 MOON0635)
MOON_SPLIT_GLOB="$WORK_DIR/split/moon/moon*_primary_secondary_calibrated_flagged.uvfits"
MOON_PLOTS_GLOB="$WORK_DIR/diagnostics_out/target/moon*/final_qa/plotvis_*.pdf"
RUN_MOON_POSTSELFCAL_ARTIFACTS="${RUN_MOON_POSTSELFCAL_ARTIFACTS:-0}"
RUN_MOON_SELFCAL_STAGE="${RUN_MOON_SELFCAL_STAGE:-0}"
RUN_MOON_SELFCAL_NO_PHASECENTER="${RUN_MOON_SELFCAL_NO_PHASECENTER:-0}"
RUN_CLEAN_STAGE="${RUN_CLEAN_STAGE:-0}"
MOON_GHPAGES_PRODUCTS_DIR="$WORK_DIR/casa_selfcal/ghpages_products"
MOON_SELFCAL_NC_DIR="$WORK_DIR/casa_selfcal/moon0520_stk10"
MOON_SELFCAL_PC_DIR="$WORK_DIR/casa_selfcal/moon0520_stk10_phasecenter"
SMART_RESUME="${SMART_RESUME:-1}"
FORCE_FROM_STEP="${FORCE_FROM_STEP:-}"

AUDIT_ONLY=false
DRY_RUN=false

usage() {
  cat <<'USAGE'
Usage:
  bash bin/run_gmrt_40_014_calibration_workflow.sh            # smart-resume workflow
  bash bin/run_gmrt_40_014_calibration_workflow.sh --audit-only
  bash bin/run_gmrt_40_014_calibration_workflow.sh --dry-run

Options:
  --audit-only   Do not run calibration; only verify and print expected output products.
  --dry-run      Do not execute; print formatted plan of stages to RUN/SKIP.

Stages:
  --------------------------------------------------------------------------
  | Step | Flag (set=1 to enable)              | Stage                    |
  --------------------------------------------------------------------------
  |    1 | RUN_CLEAN_STAGE                     | clean work outputs       |
  |    2 | (always enabled)                    | primary calibration      |
  |    3 | (always enabled)                    | split 3C48 primary       |
  |    4 | (always enabled)                    | plot 3C48 primary        |
  |    5 | (always enabled)                    | split 3C468.1 primary    |
  |    6 | (always enabled)                    | plot 3C468.1 primary     |
  |    7 | (always enabled)                    | derive clustering flags  |
  |    8 | (always enabled)                    | secondary calibration    |
  |    9 | (always enabled)                    | split 3C468.1 final      |
  |   10 | (always enabled)                    | plot 3C468.1 final       |
  |   11 | (always enabled)                    | split moon scans         |
  |   12 | (always enabled)                    | plot moon scans          |
  |   13 | RUN_MOON_SELFCAL_STAGE              | moon selfcal imaging     |
  |   14 | RUN_MOON_SELFCAL_STAGE              | moon trajectory plot     |
  |   15 | RUN_MOON_POSTSELFCAL_ARTIFACTS      | moon post-selfcal arts   |
  --------------------------------------------------------------------------
  Step 13 sub-mode: RUN_MOON_SELFCAL_NO_PHASECENTER=1 adds no-phasecenter branch.
  Steps 2-12 skip automatically when outputs exist (SMART_RESUME=1, default).
  Use --dry-run to see the exact RUN/SKIP plan before committing.

Environment controls:
  SMART_RESUME=1|0    Stage auto-skip based on output existence (default: 1)
  FORCE_FROM_STEP=N   Force rerun from step N through all downstream steps

Common intents:
  Resume from where you left off (default):
    bash bin/run_gmrt_40_014_calibration_workflow.sh

  Preview what will run without executing:
    RUN_MOON_SELFCAL_STAGE=1 RUN_MOON_POSTSELFCAL_ARTIFACTS=1 \
      bash bin/run_gmrt_40_014_calibration_workflow.sh --dry-run

  Clean everything and rerun all stages from scratch:
    RUN_CLEAN_STAGE=1 bash bin/run_gmrt_40_014_calibration_workflow.sh

  Rerun from a specific step N and all downstream:
    FORCE_FROM_STEP=N bash bin/run_gmrt_40_014_calibration_workflow.sh

  Run full workflow including selfcal + post-selfcal artifacts:
    RUN_MOON_SELFCAL_STAGE=1 RUN_MOON_POSTSELFCAL_ARTIFACTS=1 \
      bash bin/run_gmrt_40_014_calibration_workflow.sh

  Run post-selfcal artifact stage only (steps 2-14 already done):
    RUN_MOON_POSTSELFCAL_ARTIFACTS=1 bash bin/run_gmrt_40_014_calibration_workflow.sh
USAGE
}

while (( $# > 0 )); do
  case "$1" in
    --audit-only) AUDIT_ONLY=true ;;
    --dry-run) DRY_RUN=true ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 2
      ;;
  esac
  shift
done

if [[ "$AUDIT_ONLY" == true && "$DRY_RUN" == true ]]; then
  echo "ERROR: --audit-only and --dry-run are mutually exclusive"
  exit 2
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

  log "STEP ${step_no} START: ${step_name}"
  log "COMMAND: $*"
  "$@" 2>&1 | tee -a "$CHAIN_LOG"
  log "STEP ${step_no} DONE : ${step_name}"
}

run_optional_step() {
  local step_no="$1"
  local step_name="$2"
  shift 2

  log "STEP ${step_no} START (optional): ${step_name}"
  log "COMMAND: $*"
  if "$@" 2>&1 | tee -a "$CHAIN_LOG"; then
    log "STEP ${step_no} DONE  (optional): ${step_name}"
  else
    local exit_code=$?
    log "WARNING: optional step failed (exit=${exit_code}): ${step_name}"
    log "WARNING: continuing workflow so pre-selfcal products remain publishable"
  fi
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

glob_exists() {
  local pattern="$1"
  shopt -s nullglob
  local matches=( $pattern )
  shopt -u nullglob
  (( ${#matches[@]} > 0 ))
}

selfcal_final_exists() {
  local root_dir="$1"
  [[ -d "$root_dir" ]] || return 1
  find "$root_dir" -mindepth 1 -maxdepth 3 -type f -name '*_final.fits' 2>/dev/null | grep -q .
}

assert_selfcal_final_outputs() {
  local root_dir="$1"
  if ! selfcal_final_exists "$root_dir"; then
    log "CHECK FAILED: no nested *_final.fits found under: $root_dir"
    return 1
  fi
  log "CHECK OK: nested *_final.fits found under: $root_dir"
}

stage_enabled() {
  local step="$1"
  case "$step" in
    13)
      [[ "$RUN_MOON_SELFCAL_STAGE" == "1" ]]
      ;;
    14)
      [[ -d "$WORK_DIR/casa_selfcal/moon0520_dev/moon0520_full.ms" ]]
      ;;
    15)
      [[ "$RUN_MOON_POSTSELFCAL_ARTIFACTS" == "1" ]]
      ;;
    *)
      return 0
      ;;
  esac
}

stage_done() {
  local step="$1"
  case "$step" in
    1) return 1 ;;
    2) [[ -f "$PRIMARY_BANDPASS" && -f "$PRIMARY_FLAGS" ]] ;;
    3) [[ -f "$SPLIT_3C48" ]] ;;
    4) [[ -f "$PLOT_3C48_PDF" ]] ;;
    5) [[ -f "$SPLIT_3C468_PRIMARY" ]] ;;
    6) [[ -f "$PLOT_3C468_PRIMARY_PDF" ]] ;;
    7) [[ -f "$CLUSTER_FLAGS" ]] && glob_exists "$CLUSTER_PRODUCTS_GLOB" ;;
    8) glob_exists "$SECONDARY_TABLES_GLOB" ;;
    9) [[ -f "$SPLIT_3C468_FINAL" ]] ;;
    10) [[ -f "$PLOT_3C468_FINAL_PDF" ]] ;;
    11) glob_exists "$MOON_SPLIT_GLOB" ;;
    12) glob_exists "$MOON_PLOTS_GLOB" ;;
    13)
      if selfcal_no_phase_enabled; then
        selfcal_final_exists "$MOON_SELFCAL_PC_DIR" && selfcal_final_exists "$MOON_SELFCAL_NC_DIR"
      else
        selfcal_final_exists "$MOON_SELFCAL_PC_DIR"
      fi
      ;;
    14) [[ -f "$WORK_DIR/diagnostics_out/moon_imaging/moon0520_trajectory.png" ]] ;;
    15) glob_exists "$MOON_GHPAGES_PRODUCTS_DIR/no_phasecenter/*" && glob_exists "$MOON_GHPAGES_PRODUCTS_DIR/phasecenter/*" ;;
    *) return 1 ;;
  esac
}

determine_run_from_step() {
  if [[ "$RUN_CLEAN_STAGE" == "1" ]]; then
    echo 1
    return 0
  fi

  if [[ -n "$FORCE_FROM_STEP" ]]; then
    if ! [[ "$FORCE_FROM_STEP" =~ ^[0-9]+$ ]]; then
      log "ERROR: FORCE_FROM_STEP must be an integer, got '$FORCE_FROM_STEP'"
      exit 2
    fi
    echo "$FORCE_FROM_STEP"
    return 0
  fi

  if [[ "$SMART_RESUME" != "1" ]]; then
    echo 1
    return 0
  fi

  local step
  for step in $(seq 2 15); do
    if ! stage_enabled "$step"; then
      continue
    fi
    if ! stage_done "$step"; then
      echo "$step"
      return 0
    fi
  done

  echo 0
}

stage_name() {
  local step="$1"
  case "$step" in
    1) echo "clean work outputs" ;;
    2) echo "primary calibration" ;;
    3) echo "split 3C48 primary" ;;
    4) echo "plot 3C48 primary" ;;
    5) echo "split 3C468.1 primary" ;;
    6) echo "plot 3C468.1 primary" ;;
    7) echo "derive clustering flags" ;;
    8) echo "secondary calibration" ;;
    9) echo "split 3C468.1 final" ;;
    10) echo "plot 3C468.1 final" ;;
    11) echo "split moon scans" ;;
    12) echo "plot moon scans" ;;
    13) echo "moon selfcal imaging (optional)" ;;
    14) echo "moon trajectory plot (optional)" ;;
    15) echo "moon post-selfcal artifacts (optional)" ;;
    *) echo "unknown" ;;
  esac
}

selfcal_no_phase_enabled() {
  [[ "$RUN_MOON_SELFCAL_NO_PHASECENTER" == "1" ]]
}

print_execution_plan() {
  local run_from="$1"
  log ""
  log "Execution plan (dry-run)"
  log "--------------------------------------------------------------------------"
  log "| Step | Action | Reason                                 | Stage              |"
  log "--------------------------------------------------------------------------"

  local step action reason name
  for step in $(seq 1 15); do
    name="$(stage_name "$step")"

    if ! stage_enabled "$step"; then
      action="SKIP"
      reason="disabled by configuration"
    elif [[ "$run_from" == "0" ]]; then
      action="SKIP"
      reason="all enabled stages already complete"
    elif (( step < run_from )); then
      if stage_done "$step"; then
        action="SKIP"
        reason="already complete (strict downstream rerun)"
      else
        action="SKIP"
        reason="upstream of rerun boundary"
      fi
    else
      action="RUN"
      reason="selected by strict rerun policy"
    fi

    log "| $(printf '%4d' "$step") | $(printf '%6s' "$action") | $(printf '%-38s' "$reason") | $(printf '%-18s' "$name") |"
  done
  log "--------------------------------------------------------------------------"
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
    echo
    echo "Step 11: moon split (all variants, all sources)"
    ls -1 $MOON_SPLIT_GLOB 2>/dev/null || true
    echo
    echo "Step 12: moon visibility plots"
    ls -1 $MOON_PLOTS_GLOB 2>/dev/null || true
    echo
    echo "Step 13: moon selfcal imaging outputs"
    find "$MOON_SELFCAL_NC_DIR" -mindepth 1 -maxdepth 3 -type f -name '*_final.fits' 2>/dev/null | sort || true
    find "$MOON_SELFCAL_PC_DIR" -mindepth 1 -maxdepth 3 -type f -name '*_final.fits' 2>/dev/null | sort || true
    echo
    echo "Step 14: moon imaging diagnostics (trajectory plot)"
    ls -1 "$WORK_DIR/diagnostics_out/moon_imaging/"*.png 2>/dev/null || true
    echo
    echo "Step 15: moon post-selfcal artifacts (destripe + movies)"
    ls -1 "$MOON_GHPAGES_PRODUCTS_DIR"/no_phasecenter/* 2>/dev/null || true
    ls -1 "$MOON_GHPAGES_PRODUCTS_DIR"/phasecenter/* 2>/dev/null || true
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
  assert_glob "$MOON_SPLIT_GLOB"
  assert_glob "$MOON_PLOTS_GLOB"
  if [[ "$RUN_MOON_SELFCAL_STAGE" == "1" ]]; then
    assert_selfcal_final_outputs "$MOON_SELFCAL_PC_DIR"
    if selfcal_no_phase_enabled; then
      assert_selfcal_final_outputs "$MOON_SELFCAL_NC_DIR"
    fi
  fi
  if [[ "$RUN_MOON_POSTSELFCAL_ARTIFACTS" == "1" ]]; then
    assert_glob "$MOON_GHPAGES_PRODUCTS_DIR/no_phasecenter/*"
    assert_glob "$MOON_GHPAGES_PRODUCTS_DIR/phasecenter/*"
  fi
  print_products_to_audit
  log "Audit-only check complete"
  exit 0
fi

RUN_FROM_STEP="$(determine_run_from_step)"
if [[ "$DRY_RUN" == true ]]; then
  if [[ "$RUN_FROM_STEP" == "0" ]]; then
    log "SMART_RESUME: all enabled stages already complete."
  else
    log "SMART_RESUME: strict downstream rerun would start at step $RUN_FROM_STEP."
  fi
  print_execution_plan "$RUN_FROM_STEP"
  log "DRY-RUN complete (no commands executed)."
  log ""
  log "Next: run without --dry-run to execute, then publish diagnostics:"
  log "  bash bin/publish_gh_pages.sh          # commit to local gh-pages worktree"
  log "  bash bin/publish_gh_pages.sh --open   # commit + open index.html in browser"
  log "  bash bin/publish_gh_pages.sh --push   # commit + push to GitHub Pages"
  exit 0
fi

if [[ "$RUN_FROM_STEP" == "0" ]]; then
  log "SMART_RESUME: all enabled stages already complete; nothing to rerun."
  print_products_to_audit
  exit 0
fi

log "SMART_RESUME: executing strict downstream rerun from step $RUN_FROM_STEP"
if [[ "$RUN_FROM_STEP" != "1" ]]; then
  log "SMART_RESUME: skipping clean step to preserve existing products"
fi

# 1) clean
if (( RUN_FROM_STEP <= 1 )); then
  run_step 1 "clean work outputs" bash "$SCRIPT_DIR/clean_work_outputs.sh" --yes
fi

# 2) primary cal
if (( RUN_FROM_STEP <= 2 )); then
  run_step 2 "primary calibration" bash "$SCRIPT_DIR/primaryCalibration_example.sh"
  assert_file "$PRIMARY_BANDPASS"
  assert_file "$PRIMARY_FLAGS"
fi

# 3) split primary itself (3C48 with primary corrections + primary flags)
if (( RUN_FROM_STEP <= 3 )); then
  run_step 3 "split 3C48 with primary calibration+flags" bash "$SCRIPT_DIR/visSplit_3c48_example.sh"
  assert_file "$SPLIT_3C48"
fi

# 4) plot visibilities for primary cal
if (( RUN_FROM_STEP <= 4 )); then
  run_step 4 "plot visibilities for primary calibration" bash "$SCRIPT_DIR/plotVis_3c48_example.sh"
  assert_file "$PLOT_3C48_PDF"
fi

# 5) split 3c468.1 (applying primary-only corrections and flags)
if (( RUN_FROM_STEP <= 5 )); then
  run_step 5 "split 3C468.1 with primary-only corrections+flags" bash "$SCRIPT_DIR/visSplit_3c468.1_primary_example.sh"
  assert_file "$SPLIT_3C468_PRIMARY"
fi

# 6) plot vis for step-5
if (( RUN_FROM_STEP <= 6 )); then
  run_step 6 "plot visibilities for primary-only 3C468.1 split" bash "$SCRIPT_DIR/plotVis_3c468.1_primary_example.sh"
  assert_file "$PLOT_3C468_PRIMARY_PDF"
fi

# 7) derive new flags for primary-cal applied 3c468.1 using clustering
#    (script already generates before/after plots + summary PDF)
if (( RUN_FROM_STEP <= 7 )); then
  run_step 7 "derive clustering flags for 3C468.1" bash "$SCRIPT_DIR/clustering_3c468.1_split_example.sh"
  assert_file "$CLUSTER_FLAGS"
  assert_glob "$CLUSTER_PRODUCTS_GLOB"
fi

# 8) derive secondary calibration using new clustering flags
if (( RUN_FROM_STEP <= 8 )); then
  run_step 8 "secondary calibration using clustering flags" bash "$SCRIPT_DIR/secondaryCalibration_example.sh"
  assert_glob "$SECONDARY_TABLES_GLOB"
fi

# 9) generate primary+secondary calibrated 3c468.1
if (( RUN_FROM_STEP <= 9 )); then
  run_step 9 "split 3C468.1 with primary+secondary calibration" bash "$SCRIPT_DIR/visSplit_3c468.1_example.sh"
  assert_file "$SPLIT_3C468_FINAL"
fi

# 10) plot vis on step-9 output
if (( RUN_FROM_STEP <= 10 )); then
  run_step 10 "plot visibilities for primary+secondary 3C468.1 split" bash "$SCRIPT_DIR/plotVis_3c468.1_example.sh"
  assert_file "$PLOT_3C468_FINAL_PDF"
fi

# 11) split all moon (target) scans — raw, primary, primary+secondary variants
if (( RUN_FROM_STEP <= 11 )); then
  run_step 11 "split moon (target) scans — raw, primary, primary+secondary" bash "$SCRIPT_DIR/visSplit_moon_example.sh"
  assert_glob "$MOON_SPLIT_GLOB"
fi

# 12) plot vis for all moon variants
if (( RUN_FROM_STEP <= 12 )); then
  run_step 12 "plot visibilities for moon (target) scans" bash "$SCRIPT_DIR/plotVis_moon_example.sh"
  assert_glob "$MOON_PLOTS_GLOB"
fi

# 13) moon imaging diagnostics — trajectory plot for MOON0520
# 13) optional moon selfcal stage (long-running)
if [[ "$RUN_MOON_SELFCAL_STAGE" == "1" ]]; then
  if (( RUN_FROM_STEP <= 13 )) && stage_enabled 13; then
    run_optional_step 13 "moon selfcal imaging (tClean phasecentre)" \
      env OUTDIR="$MOON_SELFCAL_PC_DIR" \
      USE_TCLEAN_PHASECENTER=1 \
      MAKE_MOVIE_AFTER_SELFCAL=0 \
      RUN_DESTRIPE_AFTER_SELFCAL=0 \
      bash "$SCRIPT_DIR/run_moon_selfcal_dev.sh"

    if selfcal_no_phase_enabled; then
      run_optional_step 13 "moon selfcal imaging (no-phasecentre)" \
        env OUTDIR="$MOON_SELFCAL_NC_DIR" \
        USE_TCLEAN_PHASECENTER=0 \
        MAKE_MOVIE_AFTER_SELFCAL=0 \
        RUN_DESTRIPE_AFTER_SELFCAL=0 \
        bash "$SCRIPT_DIR/run_moon_selfcal_dev.sh"
    else
      log "INFO: no-phasecenter moon selfcal disabled (RUN_MOON_SELFCAL_NO_PHASECENTER=$RUN_MOON_SELFCAL_NO_PHASECENTER)"
    fi
  else
    log "SMART_RESUME: step 13 already complete; skipping"
  fi
else
  log "INFO: optional moon selfcal stage disabled (RUN_MOON_SELFCAL_STAGE=$RUN_MOON_SELFCAL_STAGE)"
fi

# 14) moon imaging diagnostics — trajectory plot for MOON0520
#     Requires moon0520_full.ms to exist in $WORK_DIR/casa_selfcal/moon0520_dev/
#     Produces: diagnostics_out/moon_imaging/moon0520_trajectory.png
if [ -d "$WORK_DIR/casa_selfcal/moon0520_dev/moon0520_full.ms" ]; then
  if (( RUN_FROM_STEP <= 14 )) && stage_enabled 14; then
    run_optional_step 14 "moon imaging diagnostics (trajectory plot)" \
      env WORK_DIR="$WORK_DIR" bash "$SCRIPT_DIR/../experimental/run_moon_trajectory_plot.sh"
  else
    log "SMART_RESUME: step 14 already complete; skipping"
  fi
else
  log "WARNING: optional moon trajectory step skipped; MS not found: $WORK_DIR/casa_selfcal/moon0520_dev/moon0520_full.ms"
  log "WARNING: continuing workflow so pre-selfcal products remain publishable"
fi

# 15) optional moon post-selfcal artifacts for GH Pages
#     Includes destriping, stacked products, before/after movies, and cumulative co-add RMS movies
if [[ "$RUN_MOON_POSTSELFCAL_ARTIFACTS" == "1" ]]; then
  if (( RUN_FROM_STEP <= 15 )) && stage_enabled 15; then
    run_optional_step 15 "moon post-selfcal artifacts (destripe + movies)" \
      env WORK_SELFCAL_ROOT="$WORK_DIR/casa_selfcal" \
      OUTPUT_ROOT="$MOON_GHPAGES_PRODUCTS_DIR" \
      bash "$SCRIPT_DIR/run_moon_ghpages_products.sh"
  else
    log "SMART_RESUME: step 15 already complete; skipping"
  fi
else
  log "INFO: optional moon post-selfcal artifacts step disabled (RUN_MOON_POSTSELFCAL_ARTIFACTS=$RUN_MOON_POSTSELFCAL_ARTIFACTS)"
fi

log "GMRT 40_014 calibration workflow complete"
print_products_to_audit
