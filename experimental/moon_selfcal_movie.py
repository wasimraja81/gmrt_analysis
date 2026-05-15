#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

if not matplotlib.get_backend().lower().startswith('agg'):
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Create Moon selfcal movie artifacts from *_final.fits frames.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--selfcal-dir', required=True,
                        help='Directory containing stacked selfcal outputs and *_final.fits files')
    parser.add_argument('--glob', default='*_final.fits',
                        help='Glob used to discover frame FITS files recursively')
    parser.add_argument('--compare-dir', default=None,
                        help='Optional second directory with matching FITS frames for 2-panel before/after movie')
    parser.add_argument('--left-label', default='Before',
                        help='Label for left panel when --compare-dir is provided')
    parser.add_argument('--right-label', default='After',
                        help='Label for right panel when --compare-dir is provided')
    parser.add_argument('--scale-dir', action='append', default=[],
                        help='Directory to include when computing display scale; may be repeated')
    parser.add_argument('--frames-dir', default=None,
                        help='Output directory for rendered PNG frames (default: <selfcal-dir>/movie_frames)')
    parser.add_argument('--out-mp4', default=None,
                        help='MP4 output path (default: <selfcal-dir>/<selfcal-dir-name>_movie.mp4)')
    parser.add_argument('--out-gif', default=None,
                        help='GIF output path (default: <selfcal-dir>/<selfcal-dir-name>_movie.gif)')
    parser.add_argument('--out-mov', default=None,
                        help='MOV output path (default: <selfcal-dir>/<selfcal-dir-name>_movie.mov)')
    parser.add_argument('--fps', type=int, default=4,
                        help='Frame rate for MP4/GIF output')
    parser.add_argument('--dpi', type=int, default=120,
                        help='DPI for rendered frame PNGs')
    parser.add_argument('--percentile-low', type=float, default=5.0,
                        help='Lower percentile for robust display scaling')
    parser.add_argument('--percentile-high', type=float, default=99.5,
                        help='Upper percentile for robust display scaling')
    parser.add_argument('--vmin', type=float, default=None,
                        help='Explicit lower display scale override (if set, requires --vmax)')
    parser.add_argument('--vmax', type=float, default=None,
                        help='Explicit upper display scale override (if set, requires --vmin)')
    parser.add_argument('--cmap', default='magma',
                        help='Matplotlib colormap')
    parser.add_argument('--clean-frames', action='store_true',
                        help='Delete existing frame PNGs before rendering')
    return parser.parse_args()


def _frame_key(path: Path) -> tuple[int, str]:
    stem = path.stem.lower()
    match = re.search(r'(?:stk|int)(\d+)', stem)
    if match:
        return (int(match.group(1)), stem)
    nums = re.findall(r'(\d+)', stem)
    if nums:
        return (int(nums[-1]), stem)
    return (sys.maxsize, stem)


def _discover_frames(root: Path, glob_pat: str) -> list[Path]:
    return sorted(root.rglob(glob_pat), key=_frame_key)


def _read_2d_fits(path: Path) -> np.ndarray:
    with fits.open(path, memmap=False) as hdul:
        arr = np.asarray(hdul[0].data, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f'FITS is not 2D after squeeze: {path}')
    return arr


def _compute_scale(frame_paths: list[Path], p_low: float, p_high: float) -> tuple[float, float]:
    lows: list[float] = []
    highs: list[float] = []
    for frame_path in frame_paths:
        arr = _read_2d_fits(frame_path)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            continue
        lows.append(float(np.percentile(finite, p_low)))
        highs.append(float(np.percentile(finite, p_high)))
    if not lows or not highs:
        return 0.0, 1.0
    vmin = float(np.min(lows))
    vmax = float(np.max(highs))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def _render_frames(
    frame_paths: list[Path],
    frames_dir: Path,
    vmin: float,
    vmax: float,
    cmap: str,
    dpi: int,
) -> list[Path]:
    rendered: list[Path] = []
    total = len(frame_paths)
    for idx, frame_path in enumerate(frame_paths):
        arr = _read_2d_fits(frame_path)
        fig, ax = plt.subplots(figsize=(6.0, 6.0))
        image = ax.imshow(arr, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f'{frame_path.stem} ({idx + 1}/{total})')
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label='Jy/beam')

        out_png = frames_dir / f'frame_{idx:04d}.png'
        fig.tight_layout()
        fig.savefig(out_png, dpi=dpi)
        plt.close(fig)
        rendered.append(out_png)
    return rendered


def _render_compare_frames(
    paired_paths: list[tuple[Path, Path]],
    frames_dir: Path,
    vmin: float,
    vmax: float,
    cmap: str,
    dpi: int,
    left_label: str,
    right_label: str,
) -> list[Path]:
    rendered: list[Path] = []
    total = len(paired_paths)
    for idx, (left_path, right_path) in enumerate(paired_paths):
        left_arr = _read_2d_fits(left_path)
        right_arr = _read_2d_fits(right_path)

        fig = plt.figure(figsize=(12.8, 5.0))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.045], wspace=0.08)
        ax_left = fig.add_subplot(gs[0, 0])
        ax_right = fig.add_subplot(gs[0, 1])
        cax = fig.add_subplot(gs[0, 2])

        im_left = ax_left.imshow(left_arr, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        im_right = ax_right.imshow(right_arr, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)

        ax_left.set_title(f'{left_label}: {left_path.stem}')
        ax_right.set_title(f'{right_label}: {right_path.stem}')
        for ax in (ax_left, ax_right):
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(f'Frame {idx + 1}/{total}')
        fig.colorbar(im_right, cax=cax, label='Jy/beam')

        out_png = frames_dir / f'frame_{idx:04d}.png'
        fig.subplots_adjust(left=0.04, right=0.96, top=0.93, bottom=0.05)
        fig.savefig(out_png, dpi=dpi)
        plt.close(fig)
        rendered.append(out_png)
    return rendered


def _check_ffmpeg() -> str:
    ffmpeg_path = shutil.which('ffmpeg')
    if not ffmpeg_path:
        raise RuntimeError('ffmpeg not found in PATH')
    return ffmpeg_path


def _build_mp4(ffmpeg_bin: str, frames_dir: Path, fps: int, out_mp4: Path) -> None:
    pattern = str(frames_dir / 'frame_%04d.png')
    cmd = [
        ffmpeg_bin,
        '-y',
        '-framerate',
        str(int(fps)),
        '-i',
        pattern,
        '-vf',
        'scale=trunc(iw/2)*2:trunc(ih/2)*2',
        '-c:v',
        'libx264',
        '-pix_fmt',
        'yuv420p',
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True)


def _build_gif(ffmpeg_bin: str, frames_dir: Path, fps: int, out_gif: Path) -> None:
    pattern = str(frames_dir / 'frame_%04d.png')
    palette = frames_dir / 'palette.png'

    cmd_palette = [
        ffmpeg_bin,
        '-y',
        '-framerate',
        str(int(fps)),
        '-i',
        pattern,
        '-vf',
        f'fps={int(fps)},palettegen',
        '-frames:v',
        '1',
        str(palette),
    ]
    cmd_gif = [
        ffmpeg_bin,
        '-y',
        '-framerate',
        str(int(fps)),
        '-i',
        pattern,
        '-i',
        str(palette),
        '-lavfi',
        f'fps={int(fps)} [x]; [x][1:v] paletteuse',
        str(out_gif),
    ]
    subprocess.run(cmd_palette, check=True)
    subprocess.run(cmd_gif, check=True)
    if palette.exists():
        palette.unlink()


def _build_mov(ffmpeg_bin: str, frames_dir: Path, fps: int, out_mov: Path) -> None:
    pattern = str(frames_dir / 'frame_%04d.png')
    cmd = [
        ffmpeg_bin,
        '-y',
        '-framerate',
        str(int(fps)),
        '-i',
        pattern,
        '-vf',
        'scale=trunc(iw/2)*2:trunc(ih/2)*2',
        '-c:v',
        'prores_ks',
        '-profile:v',
        '3',
        '-pix_fmt',
        'yuv422p10le',
        str(out_mov),
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    args = parse_args()

    selfcal_dir = Path(args.selfcal_dir).expanduser().resolve()
    if not selfcal_dir.exists():
        raise FileNotFoundError(f'selfcal dir not found: {selfcal_dir}')

    compare_dir = None
    if args.compare_dir:
        compare_dir = Path(args.compare_dir).expanduser().resolve()
        if not compare_dir.exists():
            raise FileNotFoundError(f'compare dir not found: {compare_dir}')

    frames_dir = Path(args.frames_dir).expanduser() if args.frames_dir else selfcal_dir / 'movie_frames'
    frames_dir = frames_dir.resolve()

    out_mp4 = Path(args.out_mp4).expanduser() if args.out_mp4 else selfcal_dir / f'{selfcal_dir.name}_movie.mp4'
    out_gif = Path(args.out_gif).expanduser() if args.out_gif else selfcal_dir / f'{selfcal_dir.name}_movie.gif'
    out_mov = Path(args.out_mov).expanduser() if args.out_mov else selfcal_dir / f'{selfcal_dir.name}_movie.mov'
    out_mp4 = out_mp4.resolve()
    out_gif = out_gif.resolve()
    out_mov = out_mov.resolve()

    frame_paths = _discover_frames(selfcal_dir, args.glob)
    if not frame_paths:
        raise RuntimeError(f'no frames found under {selfcal_dir} with glob {args.glob!r}')

    paired_paths: list[tuple[Path, Path]] = []
    if compare_dir is not None:
        compare_paths = _discover_frames(compare_dir, args.glob)

        # Primary strategy: match by identical relative path under each root.
        compare_by_rel = {
            str(path.relative_to(compare_dir)): path
            for path in compare_paths
        }
        for left_path in frame_paths:
            rel = str(left_path.relative_to(selfcal_dir))
            right_path = compare_by_rel.get(rel)
            if right_path is not None:
                paired_paths.append((left_path, right_path))

        # Fallback strategy: if no relative matches, match by basename.
        # Useful when one directory is nested and the other is flat.
        if not paired_paths:
            compare_by_name: dict[str, Path] = {}
            duplicate_names: set[str] = set()
            for path in compare_paths:
                name = path.name
                if name in compare_by_name:
                    duplicate_names.add(name)
                else:
                    compare_by_name[name] = path

            if duplicate_names:
                print(f'[moon-selfcal-movie] WARNING: duplicate compare basenames detected; '
                      f'basename fallback may skip ambiguous files: {sorted(list(duplicate_names))[:5]}')

            for left_path in frame_paths:
                right_path = compare_by_name.get(left_path.name)
                if right_path is not None and left_path.name not in duplicate_names:
                    paired_paths.append((left_path, right_path))

        if not paired_paths:
            raise RuntimeError(
                f'no matching frame pairs found between {selfcal_dir} and {compare_dir} '
                f'with glob {args.glob!r}'
            )
        if len(paired_paths) != len(frame_paths):
            print(f'[moon-selfcal-movie] WARNING: matched {len(paired_paths)}/{len(frame_paths)} frames for compare mode')

    if args.percentile_high <= args.percentile_low:
        raise ValueError('percentile-high must be greater than percentile-low')
    if args.fps <= 0:
        raise ValueError('fps must be > 0')
    if (args.vmin is None) ^ (args.vmax is None):
        raise ValueError('provide both --vmin and --vmax, or neither')
    if args.vmin is not None and args.vmax <= args.vmin:
        raise ValueError('--vmax must be greater than --vmin')

    if args.clean_frames and frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    out_mov.parent.mkdir(parents=True, exist_ok=True)

    if args.vmin is not None:
        vmin, vmax = float(args.vmin), float(args.vmax)
    else:
        scale_paths: list[Path] = []
        if args.scale_dir:
            for scale_dir_raw in args.scale_dir:
                scale_dir = Path(scale_dir_raw).expanduser().resolve()
                if not scale_dir.exists():
                    raise FileNotFoundError(f'scale dir not found: {scale_dir}')
                scale_paths.extend(_discover_frames(scale_dir, args.glob))
        elif compare_dir is not None:
            scale_paths.extend([left for left, _ in paired_paths])
            scale_paths.extend([right for _, right in paired_paths])
        else:
            scale_paths = frame_paths

        if not scale_paths:
            raise RuntimeError('no frames available to compute display scale')
        vmin, vmax = _compute_scale(scale_paths, args.percentile_low, args.percentile_high)

    print(f'[moon-selfcal-movie] selfcal_dir : {selfcal_dir}')
    print(f'[moon-selfcal-movie] frames     : {len(frame_paths)}')
    if compare_dir is not None:
        print(f'[moon-selfcal-movie] compare_dir: {compare_dir}')
        print(f'[moon-selfcal-movie] pairs      : {len(paired_paths)}')
    print(f'[moon-selfcal-movie] frames_dir : {frames_dir}')
    print(f'[moon-selfcal-movie] scale      : vmin={vmin:.6g}, vmax={vmax:.6g}')
    print(f'[moon-selfcal-movie] out_mp4    : {out_mp4}')
    print(f'[moon-selfcal-movie] out_gif    : {out_gif}')
    print(f'[moon-selfcal-movie] out_mov    : {out_mov}')

    if compare_dir is not None:
        rendered = _render_compare_frames(
            paired_paths,
            frames_dir,
            vmin=vmin,
            vmax=vmax,
            cmap=args.cmap,
            dpi=args.dpi,
            left_label=args.left_label,
            right_label=args.right_label,
        )
    else:
        rendered = _render_frames(frame_paths, frames_dir, vmin=vmin, vmax=vmax,
                                  cmap=args.cmap, dpi=args.dpi)
    if not rendered:
        raise RuntimeError('frame render produced no PNG files')

    ffmpeg_bin = _check_ffmpeg()
    _build_mp4(ffmpeg_bin, frames_dir, args.fps, out_mp4)
    _build_gif(ffmpeg_bin, frames_dir, args.fps, out_gif)
    _build_mov(ffmpeg_bin, frames_dir, args.fps, out_mov)

    duration_s = len(rendered) / float(args.fps)
    print(f'[moon-selfcal-movie] done. duration≈{duration_s:.2f}s')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
