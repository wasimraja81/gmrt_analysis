#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-$REPO_ROOT/gmrt/bin/python}"

WORK_SELFCAL_ROOT="${WORK_SELFCAL_ROOT:-/Users/raj030/DATA/gmrt_40_014/work/casa_selfcal}"
NO_PHASECENTER_DIR="${NO_PHASECENTER_DIR:-$WORK_SELFCAL_ROOT/moon0520_stk10}"
PHASECENTER_DIR="${PHASECENTER_DIR:-$WORK_SELFCAL_ROOT/moon0520_stk10_phasecenter}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$WORK_SELFCAL_ROOT/ghpages_products}"
NO_OUT="${NO_OUT:-$OUTPUT_ROOT/no_phasecenter}"
PC_OUT="${PC_OUT:-$OUTPUT_ROOT/phasecenter}"

SCAN="${SCAN:-MOON0520}"
STACK_GLOB="${STACK_GLOB:-moon0520_stk*_n*/moon0520_stk*_n*_final.fits}"

DESTRIPE_ITERS="${DESTRIPE_ITERS:-20}"
SHELL_RADIUS_PIX="${SHELL_RADIUS_PIX:-2.56}"
SHELL_SIGMA_PIX="${SHELL_SIGMA_PIX:-20.0}"
MOON_MASK_RADIUS_ARCMIN="${MOON_MASK_RADIUS_ARCMIN:-20.0}"
FFT_MAX_KLAMBDA="${FFT_MAX_KLAMBDA:-3}"
FFT_SCALE_LOW="${FFT_SCALE_LOW:-5}"
FFT_SCALE_HIGH="${FFT_SCALE_HIGH:-99.5}"
ANIMATION_FPS="${ANIMATION_FPS:-2.0}"

MOVIE_FPS="${MOVIE_FPS:-4}"
MOVIE_CMAP="${MOVIE_CMAP:-magma}"
MOVIE_PERCENTILE_LOW="${MOVIE_PERCENTILE_LOW:-5}"
MOVIE_PERCENTILE_HIGH="${MOVIE_PERCENTILE_HIGH:-99.5}"
FORCE_MOON_GHPAGES_REBUILD="${FORCE_MOON_GHPAGES_REBUILD:-0}"
KEEP_DESTRIPED_FINALS="${KEEP_DESTRIPED_FINALS:-0}"

mkdir -p "$NO_OUT" "$PC_OUT"

have_file() {
  local path="$1"
  [[ -f "$path" ]]
}

have_any_fits() {
  local dir="$1"
  [[ -d "$dir" ]] && find "$dir" -maxdepth 1 -type f -name '*.fits' | grep -q .
}

copy_if_exists() {
  local src="$1"
  local dst="$2"
  if [[ -f "$src" ]]; then
    cp -f "$src" "$dst"
  fi
}

write_stack_aliases() {
  local out_dir="$1"
  local stack_tag="$2"
  local preview_base="$out_dir/${stack_tag}_destriped_stack"
  copy_if_exists "${preview_base}_original.png" "$out_dir/${stack_tag}_raw_selfcal_stack.png"
  copy_if_exists "${preview_base}_destriped.png" "$out_dir/${stack_tag}_destriped_stack.png"
}

run_movie() {
    local src_dir="$1"
    local frames_dir="$2"
    local out_mp4="$3"
    local out_gif="$4"
    local out_mov="$5"
    local compare_dir="${6:-}"
  local fits_glob="${7:-*_final.fits}"

    if [[ "$FORCE_MOON_GHPAGES_REBUILD" != "1" ]] && have_file "$out_mp4" && have_file "$out_gif" && have_file "$out_mov"; then
      echo "[ghpages-products] SKIP movie (already exists): $out_mp4"
      return 0
    fi

    if [[ -n "$compare_dir" ]]; then
        SCAN="$SCAN" \
        SELFCAL_DIR="$src_dir" \
        COMPARE_DIR="$compare_dir" \
        LEFT_LABEL="Before" \
        RIGHT_LABEL="Destriped" \
        FPS="$MOVIE_FPS" \
        CMAP="$MOVIE_CMAP" \
        PERCENTILE_LOW="$MOVIE_PERCENTILE_LOW" \
        PERCENTILE_HIGH="$MOVIE_PERCENTILE_HIGH" \
        SCALE_DIRS="$src_dir,$compare_dir" \
        GLOB_PATTERN="$fits_glob" \
        FRAMES_DIR="$frames_dir" \
        OUT_MP4="$out_mp4" \
        OUT_GIF="$out_gif" \
        OUT_MOV="$out_mov" \
        PYTHON_CMD="$PYTHON" \
        bash "$REPO_ROOT/bin/run_moon_selfcal_movie_dev.sh"
    else
        SCAN="$SCAN" \
        SELFCAL_DIR="$src_dir" \
        FPS="$MOVIE_FPS" \
        CMAP="$MOVIE_CMAP" \
        PERCENTILE_LOW="$MOVIE_PERCENTILE_LOW" \
        PERCENTILE_HIGH="$MOVIE_PERCENTILE_HIGH" \
        SCALE_DIRS="$src_dir" \
        GLOB_PATTERN="$fits_glob" \
        FRAMES_DIR="$frames_dir" \
        OUT_MP4="$out_mp4" \
        OUT_GIF="$out_gif" \
        OUT_MOV="$out_mov" \
        PYTHON_CMD="$PYTHON" \
        bash "$REPO_ROOT/bin/run_moon_selfcal_movie_dev.sh"
    fi
}

run_stack_destripe() {
    local src_dir="$1"
    local out_dir="$2"
    local reg_mode="$3"
    local stack_tag="$4"

    local stack_out="$out_dir/${stack_tag}_destriped_stack.fits"
    local debug_dir="$out_dir/destripe_debug"
    local write_dir="$out_dir/destriped_finals"
    local prov_dir="$out_dir/provenance"

    mkdir -p "$debug_dir" "$write_dir" "$prov_dir"

    if [[ "$FORCE_MOON_GHPAGES_REBUILD" != "1" ]] && have_file "$stack_out" && have_any_fits "$write_dir"; then
      write_stack_aliases "$out_dir" "$stack_tag"
      echo "[ghpages-products] SKIP destripe+stack (already exists): $stack_out"
      return 0
    fi

    "$PYTHON" "$REPO_ROOT/src/stack_moon_snapshots.py" \
      --selfcal-dir "$src_dir" \
      --glob "$STACK_GLOB" \
      --output "$stack_out" \
      --registration-mode "$reg_mode" \
      --registration-method phase-correlation \
      --upsample 10 \
      --method mean \
      --destripe-iters "$DESTRIPE_ITERS" \
      --destripe-strategy radial-shell \
      --destripe-shell-radius-pix "$SHELL_RADIUS_PIX" \
      --destripe-shell-sigma-pix "$SHELL_SIGMA_PIX" \
      --moon-mask-center ephem \
      --moon-mask-radius-arcmin "$MOON_MASK_RADIUS_ARCMIN" \
      --destripe-debug-dir "$debug_dir" \
      --destripe-debug-fft-axis klambda \
      --destripe-debug-fft-max-klambda "$FFT_MAX_KLAMBDA" \
      --destripe-debug-fft-scale-mode per-iter \
      --destripe-debug-fft-scale-low "$FFT_SCALE_LOW" \
      --destripe-debug-fft-scale-high "$FFT_SCALE_HIGH" \
      --destripe-debug-animation-fps "$ANIMATION_FPS" \
      --destripe-write-dir "$write_dir" \
      --provenance-dir "$prov_dir"

    write_stack_aliases "$out_dir" "$stack_tag"
}

run_cumulative_movie() {
    local src_dir="$1"
    local out_dir="$2"
    local reg_mode="$3"
    local tag="$4"
    local fits_glob="${5:-$STACK_GLOB}"

    local out_mp4="$out_dir/${tag}_cumulative_coadd.mp4"
    local out_gif="$out_dir/${tag}_cumulative_coadd.gif"
    local out_mov="$out_dir/${tag}_cumulative_coadd.mov"
    local out_rms_png="$out_dir/${tag}_cumulative_rms_evolution.png"
    local out_rms_csv="$out_dir/${tag}_cumulative_rms_evolution.csv"

    if [[ "$FORCE_MOON_GHPAGES_REBUILD" != "1" ]] && have_file "$out_mp4" && have_file "$out_gif" && have_file "$out_mov" && have_file "$out_rms_png" && have_file "$out_rms_csv"; then
      echo "[ghpages-products] SKIP cumulative coadd (already exists): $out_mp4"
      return 0
    fi

    "$PYTHON" "$REPO_ROOT/src/moon_cumulative_coadd_movie.py" \
      --selfcal-dir "$src_dir" \
      --glob "$fits_glob" \
      --registration-mode "$reg_mode" \
      --moon-mask-radius-arcmin "$MOON_MASK_RADIUS_ARCMIN" \
      --fps "$MOVIE_FPS" \
      --cmap "$MOVIE_CMAP" \
      --percentile-low "$MOVIE_PERCENTILE_LOW" \
      --percentile-high "$MOVIE_PERCENTILE_HIGH" \
      --frames-dir "$out_dir/movie_frames_cumulative" \
      --out-mp4 "$out_mp4" \
      --out-gif "$out_gif" \
      --out-mov "$out_mov" \
      --out-rms-png "$out_rms_png" \
      --out-rms-csv "$out_rms_csv"
}

cleanup_destriped_finals() {
  local out_dir="$1"
  local write_dir="$out_dir/destriped_finals"
  if [[ "$KEEP_DESTRIPED_FINALS" == "1" ]]; then
    return 0
  fi
  if [[ -d "$write_dir" ]]; then
    echo "[ghpages-products] CLEANUP intermediate destriped FITS: $write_dir"
    rm -rf "$write_dir"
  fi
}

echo "[ghpages-products] python           : $PYTHON"
echo "[ghpages-products] no-phasecenter   : $NO_PHASECENTER_DIR"
echo "[ghpages-products] phasecenter      : $PHASECENTER_DIR"
echo "[ghpages-products] output root      : $OUTPUT_ROOT"
echo "[ghpages-products] force rebuild    : $FORCE_MOON_GHPAGES_REBUILD"
echo "[ghpages-products] keep destriped   : $KEEP_DESTRIPED_FINALS"

echo "[ghpages-products] === No-phasecentre case ==="
mkdir -p "$NO_OUT"

run_movie "$NO_PHASECENTER_DIR" \
  "$NO_OUT/movie_frames_raw_selfcal" \
  "$NO_OUT/moon0520_no_phasecenter_raw_selfcal.mp4" \
  "$NO_OUT/moon0520_no_phasecenter_raw_selfcal.gif" \
  "$NO_OUT/moon0520_no_phasecenter_raw_selfcal.mov" \
  "" \
  "$STACK_GLOB"

run_stack_destripe "$NO_PHASECENTER_DIR" "$NO_OUT" "derive" "moon0520_no_phasecenter_phasecorr"

run_movie "$NO_OUT/destriped_finals" \
  "$NO_OUT/movie_frames_destriped" \
  "$NO_OUT/moon0520_no_phasecenter_destriped.mp4" \
  "$NO_OUT/moon0520_no_phasecenter_destriped.gif" \
  "$NO_OUT/moon0520_no_phasecenter_destriped.mov" \
  "" \
  "*_final.fits"

run_movie "$NO_PHASECENTER_DIR" \
  "$NO_OUT/movie_frames_compare" \
  "$NO_OUT/moon0520_no_phasecenter_before_after_compare.mp4" \
  "$NO_OUT/moon0520_no_phasecenter_before_after_compare.gif" \
  "$NO_OUT/moon0520_no_phasecenter_before_after_compare.mov" \
  "$NO_OUT/destriped_finals" \
  "*_final.fits"

run_cumulative_movie "$NO_PHASECENTER_DIR" "$NO_OUT" "derive" "moon0520_no_phasecenter_phasecorr_raw_selfcal" "$STACK_GLOB"
run_cumulative_movie "$NO_OUT/destriped_finals" "$NO_OUT" "derive" "moon0520_no_phasecenter_phasecorr_destriped" "*_final.fits"
cleanup_destriped_finals "$NO_OUT"

echo "[ghpages-products] === tClean phase-centre case ==="
mkdir -p "$PC_OUT"

run_movie "$PHASECENTER_DIR" \
  "$PC_OUT/movie_frames_raw_selfcal" \
  "$PC_OUT/moon0520_phasecenter_raw_selfcal.mp4" \
  "$PC_OUT/moon0520_phasecenter_raw_selfcal.gif" \
  "$PC_OUT/moon0520_phasecenter_raw_selfcal.mov" \
  "" \
  "$STACK_GLOB"

run_stack_destripe "$PHASECENTER_DIR" "$PC_OUT" "none" "moon0520_phasecenter_noshift"

run_movie "$PC_OUT/destriped_finals" \
  "$PC_OUT/movie_frames_destriped" \
  "$PC_OUT/moon0520_phasecenter_destriped.mp4" \
  "$PC_OUT/moon0520_phasecenter_destriped.gif" \
  "$PC_OUT/moon0520_phasecenter_destriped.mov" \
  "" \
  "*_final.fits"

run_movie "$PHASECENTER_DIR" \
  "$PC_OUT/movie_frames_compare" \
  "$PC_OUT/moon0520_phasecenter_before_after_compare.mp4" \
  "$PC_OUT/moon0520_phasecenter_before_after_compare.gif" \
  "$PC_OUT/moon0520_phasecenter_before_after_compare.mov" \
  "$PC_OUT/destriped_finals" \
  "*_final.fits"

run_cumulative_movie "$PHASECENTER_DIR" "$PC_OUT" "none" "moon0520_phasecenter_noshift_raw_selfcal" "$STACK_GLOB"
run_cumulative_movie "$PC_OUT/destriped_finals" "$PC_OUT" "none" "moon0520_phasecenter_noshift_destriped" "*_final.fits"
cleanup_destriped_finals "$PC_OUT"

echo "[ghpages-products] done"
echo "[ghpages-products] no-phasecenter products: $NO_OUT"
echo "[ghpages-products] phasecenter products   : $PC_OUT"
