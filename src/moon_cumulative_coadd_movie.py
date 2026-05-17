#!/usr/bin/env python3

from __future__ import annotations

import argparse
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
from modules.moon_registration import (
    align_images as shared_align_images,
    choose_reference_index_mean_jd,
    off_moon_rms as shared_off_moon_rms,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build cumulative co-add movie with off-moon RMS inset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--selfcal-dir', required=True,
                        help='Directory containing input FITS frames')
    parser.add_argument('--glob', default='*_final.fits',
                        help='Glob used to discover FITS frames recursively')
    parser.add_argument('--registration-mode', choices=['derive', 'none'], default='none',
                        help='Registration before cumulative co-add')
    parser.add_argument('--upsample', type=int, default=10,
                        help='Sub-pixel upsample factor for derive registration')
    parser.add_argument('--moon-mask-radius-arcmin', type=float, default=20.0,
                        help='Radius used for off-moon RMS mask')
    parser.add_argument('--frames-dir', default=None,
                        help='Directory to render movie frame PNGs')
    parser.add_argument('--out-mp4', default=None,
                        help='Output MP4 path')
    parser.add_argument('--out-gif', default=None,
                        help='Output GIF path')
    parser.add_argument('--out-mov', default=None,
                        help='Output MOV path')
    parser.add_argument('--out-rms-png', default=None,
                        help='Output PNG path for RMS evolution')
    parser.add_argument('--out-rms-csv', default=None,
                        help='Output CSV path for RMS evolution values')
    parser.add_argument('--fps', type=int, default=4,
                        help='Frame rate for MP4/GIF/MOV')
    parser.add_argument('--dpi', type=int, default=120,
                        help='DPI for rendered frame PNGs')
    parser.add_argument('--percentile-low', type=float, default=5.0,
                        help='Lower percentile for robust display scaling')
    parser.add_argument('--percentile-high', type=float, default=99.5,
                        help='Upper percentile for robust display scaling')
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


def _read_2d_fits(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=False) as hdul:
        primary_hdu = hdul[0]
        data = getattr(primary_hdu, 'data', None)
        header = getattr(primary_hdu, 'header', None)
        if data is None or header is None:
            raise ValueError(f'FITS primary HDU missing data/header: {path}')
        arr = np.asarray(data, dtype=np.float32)
        hdr = fits.Header(header)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f'FITS is not 2D after squeeze: {path}')
    return arr, hdr


def _robust_scale(arrays: list[np.ndarray], p_low: float, p_high: float) -> tuple[float, float]:
    lows: list[float] = []
    highs: list[float] = []
    for arr in arrays:
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


def _off_moon_rms(image: np.ndarray, mask_radius_pix: float,
                  moon_center: tuple[float, float] | None = None) -> float:
    return shared_off_moon_rms(image, mask_radius_pix, moon_center=moon_center)


def _find_moon_center(image: np.ndarray, bright_frac: float = 0.005) -> tuple[float, float]:
    """Estimate moon centre as weighted centroid of brightest ~0.5 % of pixels."""
    arr = np.asarray(image, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return (arr.shape[0] - 1) / 2.0, (arr.shape[1] - 1) / 2.0
    threshold = np.percentile(finite, 100.0 * (1.0 - bright_frac))
    mask = np.isfinite(arr) & (arr >= threshold)
    weights = np.where(mask, arr, 0.0)
    total = float(weights.sum())
    if total <= 0:
        return (arr.shape[0] - 1) / 2.0, (arr.shape[1] - 1) / 2.0
    yy, xx = np.indices(arr.shape, dtype=np.float64)
    cy = float((weights * yy).sum() / total)
    cx = float((weights * xx).sum() / total)
    return cy, cx


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


def _save_rms_outputs(rms_vals: list[float], out_png: Path, out_csv: Path) -> None:
    n = len(rms_vals)
    x = np.arange(1, n + 1)
    y = np.array(rms_vals, dtype=float)

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(x, y * 1000.0, '-o', color='tab:cyan', linewidth=2.0, markersize=5)
    ax.set_xlabel('Number of co-added frames')
    ax.set_ylabel('Off-moon RMS (mJy/beam, robust)')
    ax.set_title('Cumulative co-add RMS evolution')
    ax.set_xlim(1, n)
    y_mJy = y * 1000.0
    y_finite = y_mJy[np.isfinite(y_mJy)]
    if y_finite.size > 0:
        ymin = float(np.min(y_finite))
        ymax = float(np.max(y_finite))
        if ymax <= ymin:
            ymax = ymin + 1e-6
        pad = 0.08 * (ymax - ymin)
        ax.set_ylim(ymin - pad, ymax + pad)
    ax.grid(alpha=0.3, linestyle='--')
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    out_csv.write_text('n_coadd,offmoon_rms_mJy_per_beam\n' + '\n'.join(
        f'{idx},{val * 1000.0:.6f}' for idx, val in enumerate(rms_vals, start=1)
    ))


def main() -> int:
    args = parse_args()

    if args.percentile_high <= args.percentile_low:
        raise ValueError('percentile-high must be greater than percentile-low')
    if args.fps <= 0:
        raise ValueError('fps must be > 0')

    selfcal_dir = Path(args.selfcal_dir).expanduser().resolve()
    if not selfcal_dir.exists():
        raise FileNotFoundError(f'selfcal dir not found: {selfcal_dir}')

    frames_dir = Path(args.frames_dir).expanduser().resolve() if args.frames_dir else selfcal_dir / 'cumulative_frames'
    out_mp4 = Path(args.out_mp4).expanduser().resolve() if args.out_mp4 else selfcal_dir / f'{selfcal_dir.name}_cumulative_coadd.mp4'
    out_gif = Path(args.out_gif).expanduser().resolve() if args.out_gif else selfcal_dir / f'{selfcal_dir.name}_cumulative_coadd.gif'
    out_mov = Path(args.out_mov).expanduser().resolve() if args.out_mov else selfcal_dir / f'{selfcal_dir.name}_cumulative_coadd.mov'
    out_rms_png = Path(args.out_rms_png).expanduser().resolve() if args.out_rms_png else selfcal_dir / f'{selfcal_dir.name}_cumulative_rms_evolution.png'
    out_rms_csv = Path(args.out_rms_csv).expanduser().resolve() if args.out_rms_csv else selfcal_dir / f'{selfcal_dir.name}_cumulative_rms_evolution.csv'

    frame_paths = _discover_frames(selfcal_dir, args.glob)
    if not frame_paths:
        raise RuntimeError(f'no frames found under {selfcal_dir} with glob {args.glob!r}')

    arrays: list[np.ndarray] = []
    headers: list[fits.Header] = []
    for path in frame_paths:
        arr, hdr = _read_2d_fits(path)
        arrays.append(arr)
        headers.append(hdr)

    ref_idx, jd_arr, jd_mean = choose_reference_index_mean_jd(headers)
    aligned_arrays, shifts, backend = shared_align_images(
        images=arrays,
        ref_idx=ref_idx,
        registration_mode=args.registration_mode,
        registration_method='phase-correlation',
        upsample=args.upsample,
    )

    # Log per-frame shifts immediately after alignment so they appear in the run log.
    for _i, (_dr, _dc) in enumerate(shifts):
        print(f'[cum-coadd] frame_shift  {_i:02d} ({frame_paths[_i].name}): '
              f'dr={_dr:+.3f}px  dc={_dc:+.3f}px')

    # Find moon centre in the aligned reference frame.
    # For no_phasecenter data the moon is NOT at the image centre; using the
    # centroid of the brightest pixels in the reference aligned image ensures the
    # off-moon RMS mask is placed correctly for all geometries.
    moon_cy, moon_cx = _find_moon_center(aligned_arrays[ref_idx])
    print(f'[cum-coadd] moon_center  : cy={moon_cy:.1f}px  cx={moon_cx:.1f}px  '
          f'(image centre: cy={(aligned_arrays[ref_idx].shape[0]-1)/2.0:.1f}  '
          f'cx={(aligned_arrays[ref_idx].shape[1]-1)/2.0:.1f})')

    vmin, vmax = _robust_scale(aligned_arrays, args.percentile_low, args.percentile_high)

    cdelt2_raw = headers[0].get('CDELT2', 1.0 / 3600.0)
    if isinstance(cdelt2_raw, (int, float, np.floating)):
        cdelt2_deg = abs(float(cdelt2_raw))
    else:
        cdelt2_deg = 1.0 / 3600.0
    pix_per_arcmin = 1.0 / (cdelt2_deg * 60.0)
    moon_mask_radius_pix = max(1.0, args.moon_mask_radius_arcmin * pix_per_arcmin)

    if args.clean_frames and frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    out_mov.parent.mkdir(parents=True, exist_ok=True)
    out_rms_png.parent.mkdir(parents=True, exist_ok=True)
    out_rms_csv.parent.mkdir(parents=True, exist_ok=True)

    rms_vals: list[float] = []
    cumulative_sum = np.zeros_like(aligned_arrays[0], dtype=np.float64)
    n_total = len(aligned_arrays)

    for idx, arr in enumerate(aligned_arrays, start=1):
        cumulative_sum += np.asarray(arr, dtype=np.float64)
        stack = cumulative_sum / float(idx)
        rms_val = _off_moon_rms(stack, moon_mask_radius_pix, moon_center=(moon_cy, moon_cx))
        rms_vals.append(rms_val)

    cumulative_sum = np.zeros_like(aligned_arrays[0], dtype=np.float64)
    for idx, arr in enumerate(aligned_arrays, start=1):
        cumulative_sum += np.asarray(arr, dtype=np.float64)
        stack = cumulative_sum / float(idx)

        fig, ax = plt.subplots(figsize=(7.2, 7.2))
        im = ax.imshow(stack, origin='lower', cmap=args.cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f'Cumulative co-add ({args.registration_mode})  n={idx}/{n_total}')
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Jy/beam')  # noqa: colourbar keeps Jy/beam (raw pixel units)
        cbar.ax.tick_params(labelsize=8)

        if args.registration_mode == 'derive':
            dr, dc = shifts[idx - 1]
            ax.text(0.02, 0.02,
                f'shift applied: Δr={dr:+.3f}px  Δc={dc:+.3f}px',
                    transform=ax.transAxes, fontsize=8,
                    color='white', bbox=dict(facecolor='black', alpha=0.45, pad=3))

        inset = ax.inset_axes((0.58, 0.58, 0.38, 0.35))
        x_full = np.arange(1, n_total + 1)
        inset.plot(x_full, [np.nan] * n_total, alpha=0.0)
        # Inset background is white → curve must be dark; labels/ticks/spines
        # overhang onto the (dark) image region so they stay yellow.
        rms_mJy = [v * 1000.0 for v in rms_vals]
        inset.plot(np.arange(1, idx + 1), rms_mJy[:idx], '-o', color='navy', linewidth=1.7, markersize=3)
        inset.set_xlim(1, n_total)
        y_finite = np.array(rms_mJy, dtype=float)
        y_finite = y_finite[np.isfinite(y_finite)]
        if y_finite.size > 0:
            ymin = float(np.min(y_finite))
            ymax = float(np.max(y_finite))
            if ymax <= ymin:
                ymax = ymin + 1e-6
            pad = 0.1 * (ymax - ymin)
            inset.set_ylim(ymin - pad, ymax + pad)
        inset.set_title('Off-moon RMS', fontsize=8, color='yellow')
        inset.set_xlabel('N', fontsize=7, color='yellow')
        inset.set_ylabel('mJy/beam', fontsize=7, color='yellow')
        inset.tick_params(axis='both', labelsize=7, colors='yellow')
        for spine in inset.spines.values():
            spine.set_color('yellow')
        inset.grid(alpha=0.3, linestyle='--')

        frame_png = frames_dir / f'frame_{idx - 1:04d}.png'
        fig.tight_layout()
        fig.savefig(frame_png, dpi=args.dpi)
        plt.close(fig)

    _save_rms_outputs(rms_vals, out_rms_png, out_rms_csv)

    ffmpeg_bin = _check_ffmpeg()
    _build_mp4(ffmpeg_bin, frames_dir, args.fps, out_mp4)
    _build_gif(ffmpeg_bin, frames_dir, args.fps, out_gif)
    _build_mov(ffmpeg_bin, frames_dir, args.fps, out_mov)

    print(f'[cum-coadd] selfcal_dir : {selfcal_dir}')
    print(f'[cum-coadd] frames      : {n_total}')
    print(f'[cum-coadd] reg_mode    : {args.registration_mode}')
    print(f'[cum-coadd] reg_backend : {backend}')
    print(f'[cum-coadd] upsample    : {args.upsample}')
    print(f'[cum-coadd] ref_idx     : {ref_idx}')
    print(f'[cum-coadd] ref_file    : {frame_paths[ref_idx].name}')
    print(f'[cum-coadd] ref_jd      : {jd_arr[ref_idx]:.8f} (mean={jd_mean:.8f})')
    print(f'[cum-coadd] frame_dir   : {frames_dir}')
    print(f'[cum-coadd] out_mp4     : {out_mp4}')
    print(f'[cum-coadd] out_gif     : {out_gif}')
    print(f'[cum-coadd] out_mov     : {out_mov}')
    print(f'[cum-coadd] out_rms_png : {out_rms_png}')
    print(f'[cum-coadd] out_rms_csv : {out_rms_csv}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
