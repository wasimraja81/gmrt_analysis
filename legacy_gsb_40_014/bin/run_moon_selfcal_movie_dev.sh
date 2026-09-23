#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INVOKE_CWD="$PWD"
PYTHON="${PYTHON_CMD:-python}"

SCAN="${SCAN:-MOON0520}"
SCAN_LOWER="$(printf '%s' "$SCAN" | tr '[:upper:]' '[:lower:]')"
SELFCAL_DIR="${SELFCAL_DIR:-$HOME/DATA/gmrt_40_014/work/casa_selfcal/moon0520_stk10}"
LOG_DIR="${LOG_DIR:-$HOME/DATA/gmrt_40_014/work/logs}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"

FPS="${FPS:-4}"
CMAP="${CMAP:-magma}"
P_LOW="${PERCENTILE_LOW:-5}"
P_HIGH="${PERCENTILE_HIGH:-99.5}"
COMPARE_DIR="${COMPARE_DIR:-}"
LEFT_LABEL="${LEFT_LABEL:-Before}"
RIGHT_LABEL="${RIGHT_LABEL:-After}"
SCALE_DIRS="${SCALE_DIRS:-}"
ADD_CLEAN_FRAMES=0
GLOB_PATTERN="${GLOB_PATTERN:-*_final.fits}"
FRAMES_DIR="${FRAMES_DIR:-}"
OUT_MP4="${OUT_MP4:-}"
OUT_GIF="${OUT_GIF:-}"
OUT_MOV="${OUT_MOV:-}"

resolve_path() {
    local raw="$1"
    local expanded="${raw/#\~/$HOME}"
    if [[ "$expanded" = /* ]]; then
        printf '%s\n' "$expanded"
    else
        printf '%s/%s\n' "$INVOKE_CWD" "$expanded"
    fi
}

POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --selfcal-dir)
            SELFCAL_DIR="$2"
            shift 2
            ;;
        --glob)
            GLOB_PATTERN="$2"
            shift 2
            ;;
        --frames-dir)
            FRAMES_DIR="$2"
            shift 2
            ;;
        --out-mp4)
            OUT_MP4="$2"
            shift 2
            ;;
        --out-gif)
            OUT_GIF="$2"
            shift 2
            ;;
        --out-mov)
            OUT_MOV="$2"
            shift 2
            ;;
        --fps)
            FPS="$2"
            shift 2
            ;;
        --cmap)
            CMAP="$2"
            shift 2
            ;;
        --percentile-low)
            P_LOW="$2"
            shift 2
            ;;
        --percentile-high)
            P_HIGH="$2"
            shift 2
            ;;
        --compare-dir)
            COMPARE_DIR="$2"
            shift 2
            ;;
        --left-label)
            LEFT_LABEL="$2"
            shift 2
            ;;
        --right-label)
            RIGHT_LABEL="$2"
            shift 2
            ;;
        --scale-dir)
            shift
            scale_values=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                scale_values+=("$1")
                shift
            done
            if [[ ${#scale_values[@]} -gt 0 ]]; then
                SCALE_DIRS="${scale_values[*]}"
            fi
            ;;
        --clean-frames)
            ADD_CLEAN_FRAMES=1
            shift
            ;;
        --help|-h)
            cat <<'USAGE'
Usage: run_moon_selfcal_movie_dev.sh [options] [selfcal_dir]

Options:
  --selfcal-dir DIR
  --compare-dir DIR
  --left-label LABEL
  --right-label LABEL
  --scale-dir DIR [DIR ...]
  --glob PATTERN
  --frames-dir DIR
  --out-mp4 PATH
  --out-gif PATH
  --out-mov PATH
  --fps N
  --cmap NAME
  --percentile-low VALUE
  --percentile-high VALUE
  --clean-frames
USAGE
            exit 0
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

if [[ ${#POSITIONAL[@]} -gt 0 ]]; then
    SELFCAL_DIR="${POSITIONAL[0]}"
fi

SELFCAL_DIR="$(resolve_path "$SELFCAL_DIR")"
if [[ -n "$COMPARE_DIR" ]]; then
    COMPARE_DIR="$(resolve_path "$COMPARE_DIR")"
fi

if [[ -n "$COMPARE_DIR" ]]; then
    sel_base="$(basename "$SELFCAL_DIR" | tr '[:upper:]' '[:lower:]')"
    cmp_base="$(basename "$COMPARE_DIR" | tr '[:upper:]' '[:lower:]')"
    left_lc="$(printf '%s' "$LEFT_LABEL" | tr '[:upper:]' '[:lower:]')"
    right_lc="$(printf '%s' "$RIGHT_LABEL" | tr '[:upper:]' '[:lower:]')"

    if [[ "$sel_base" == *destrip* && "$cmp_base" == *before* && "$left_lc" == *before* ]]; then
        tmp_dir="$SELFCAL_DIR"
        SELFCAL_DIR="$COMPARE_DIR"
        COMPARE_DIR="$tmp_dir"
    fi

    if [[ -z "$FRAMES_DIR" || -z "$OUT_MP4" || -z "$OUT_GIF" || -z "$OUT_MOV" ]]; then
        sel_parent="$(dirname "$SELFCAL_DIR")"
        cmp_parent="$(dirname "$COMPARE_DIR")"
        if [[ "$sel_parent" == "$cmp_parent" ]]; then
            compare_out_root="$sel_parent"
            if [[ -z "$FRAMES_DIR" ]]; then
                FRAMES_DIR="$compare_out_root/movie_frames_compare"
            fi
            if [[ -z "$OUT_MP4" ]]; then
                OUT_MP4="$compare_out_root/${SCAN_LOWER}_selfcal_movie_before_after_compare.mp4"
            fi
            if [[ -z "$OUT_GIF" ]]; then
                OUT_GIF="$compare_out_root/${SCAN_LOWER}_selfcal_movie_before_after_compare.gif"
            fi
            if [[ -z "$OUT_MOV" ]]; then
                OUT_MOV="$compare_out_root/${SCAN_LOWER}_selfcal_movie_before_after_compare.mov"
            fi
        fi
    fi
fi

if [[ -z "$FRAMES_DIR" ]]; then
    FRAMES_DIR="$SELFCAL_DIR/movie_frames"
else
    FRAMES_DIR="$(resolve_path "$FRAMES_DIR")"
fi

if [[ -z "$OUT_MP4" ]]; then
    OUT_MP4="$SELFCAL_DIR/${SCAN_LOWER}_selfcal_movie.mp4"
else
    OUT_MP4="$(resolve_path "$OUT_MP4")"
fi
if [[ -z "$OUT_GIF" ]]; then
    OUT_GIF="$SELFCAL_DIR/${SCAN_LOWER}_selfcal_movie.gif"
else
    OUT_GIF="$(resolve_path "$OUT_GIF")"
fi
if [[ -z "$OUT_MOV" ]]; then
    OUT_MOV="$SELFCAL_DIR/${SCAN_LOWER}_selfcal_movie.mov"
else
    OUT_MOV="$(resolve_path "$OUT_MOV")"
fi

default_mp4="$SELFCAL_DIR/${SCAN_LOWER}_selfcal_movie.mp4"
default_gif="$SELFCAL_DIR/${SCAN_LOWER}_selfcal_movie.gif"
default_mov="$SELFCAL_DIR/${SCAN_LOWER}_selfcal_movie.mov"
if [[ -n "$COMPARE_DIR" ]]; then
    [[ "$OUT_MP4" == "$default_mp4" ]] && OUT_MP4="$SELFCAL_DIR/${SCAN_LOWER}_selfcal_movie_before_after_compare.mp4"
    [[ "$OUT_GIF" == "$default_gif" ]] && OUT_GIF="$SELFCAL_DIR/${SCAN_LOWER}_selfcal_movie_before_after_compare.gif"
    [[ "$OUT_MOV" == "$default_mov" ]] && OUT_MOV="$SELFCAL_DIR/${SCAN_LOWER}_selfcal_movie_before_after_compare.mov"
fi

mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

CMD_FILE="$LOG_DIR/run_moon_selfcal_movie_${SCAN_LOWER}_${RUN_TS}.cmd"
LOG_FILE="$LOG_DIR/run_moon_selfcal_movie_${SCAN_LOWER}_${RUN_TS}.log"

CMD=(
    "$PYTHON" "$REPO_ROOT/src/moon_selfcal_movie.py"
    --selfcal-dir "$SELFCAL_DIR"
    --glob "$GLOB_PATTERN"
    --frames-dir "$FRAMES_DIR"
    --out-mp4 "$OUT_MP4"
    --out-gif "$OUT_GIF"
    --out-mov "$OUT_MOV"
    --fps "$FPS"
    --cmap "$CMAP"
    --percentile-low "$P_LOW"
    --percentile-high "$P_HIGH"
)

if [[ -n "$COMPARE_DIR" ]]; then
    CMD+=(--compare-dir "$COMPARE_DIR" --left-label "$LEFT_LABEL" --right-label "$RIGHT_LABEL")
fi

if [[ -n "$SCALE_DIRS" ]]; then
    SCALE_DIRS_EXPANDED="${SCALE_DIRS//,/ }"
    for scale_dir in $SCALE_DIRS_EXPANDED; do
        scale_dir="$(resolve_path "$scale_dir")"
        [[ -z "$scale_dir" ]] && continue
        CMD+=(--scale-dir "$scale_dir")
    done
fi

if [[ "$ADD_CLEAN_FRAMES" -eq 1 ]]; then
    CMD+=(--clean-frames)
fi

{
    echo "# timestamp=$RUN_TS"
    echo "# cwd=$PWD"
    printf '%q ' "${CMD[@]}"
    printf '\n'
} > "$CMD_FILE"

echo "[run-moon-selfcal-movie] cmd       : $CMD_FILE"
echo "[run-moon-selfcal-movie] log       : $LOG_FILE"
echo "[run-moon-selfcal-movie] selfcal   : $SELFCAL_DIR"
echo "[run-moon-selfcal-movie] compare   : ${COMPARE_DIR:-<none>}"
echo "[run-moon-selfcal-movie] out_mp4   : $OUT_MP4"
echo "[run-moon-selfcal-movie] out_gif   : $OUT_GIF"
echo "[run-moon-selfcal-movie] out_mov   : $OUT_MOV"

"${CMD[@]}" 2>&1 | tee "$LOG_FILE"

echo ""
echo "Log: $LOG_FILE"
echo "Cmd: $CMD_FILE"
echo "Movie MP4: $OUT_MP4"
echo "Movie GIF: $OUT_GIF"
echo "Movie MOV: $OUT_MOV"
