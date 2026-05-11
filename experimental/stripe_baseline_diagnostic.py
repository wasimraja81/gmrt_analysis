#!/usr/bin/env python
"""Stripe-pattern baseline diagnostic for radio images + MeasurementSet.

Workflow:
1) Read one image, many images, or a cube.
2) Detect dominant stripe Fourier mode per plane.
3) Map mode to likely baseline(s) using MS UVW per integration time.
4) Rank recurring suspicious baselines.
5) Generate diagnostic plots and a machine-readable JSON report.
6) Emit CASA selectors and example commands for include/exclude decisions.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

try:
    from casacore.tables import table as _casacore_table
except Exception:
    _casacore_table = None


def _read_table_col(table_path: str, col: str):
    if _casacore_table is not None:
        with _casacore_table(table_path, readonly=True) as t:
            return t.getcol(col)

    try:
        from casatools import table as casa_table  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            'Neither python-casacore nor casatools.table is available to read MeasurementSet tables.'
        ) from exc

    tb = casa_table()
    tb.open(table_path)
    try:
        return tb.getcol(col)
    finally:
        tb.close()


@dataclass
class PlaneResult:
    plane_index: int
    source_file: str
    du_pix: int
    dv_pix: int
    uv_lambda: float
    stripe_angle_deg: float
    best_pair: tuple[int, int]
    best_pair_hits: int


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Diagnose image striping and likely bad baselines from MS UVW')
    p.add_argument('--ms', required=True, help='Input MeasurementSet path')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--cube', help='FITS cube path (time, y, x)')
    g.add_argument('--images', nargs='+', help='One or more 2D FITS images')
    p.add_argument('--cell-arcsec', type=float, default=2.0, help='Image cell size in arcsec/pixel (default: 2.0)')
    p.add_argument('--top-k-rows', type=int, default=50, help='Number of nearest UV rows per plane for pair voting')
    p.add_argument('--max-radius-frac', type=float, default=0.2, help='Max FFT radius fraction for stripe search (default: 0.2)')
    p.add_argument('--min-radius-pix', type=float, default=8.0, help='Min FFT radius in pixels to suppress DC (default: 8)')
    p.add_argument('--top-n-baselines', type=int, default=10, help='Top recurring suspicious baselines to report')
    p.add_argument('--outdir', default='./diagnostics_out/stripe_baseline', help='Output directory')
    p.add_argument('--tag', default='stripe_diag', help='Tag for output filenames')
    p.add_argument('--ignore-selector', default='', help='CASA one-based baseline selector to ignore (e.g. 1&25;18&30)')
    return p.parse_args()


def _parse_selector_pairs(selector: str) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    selector = (selector or '').strip()
    if not selector:
        return out
    for tok in selector.split(';'):
        tok = tok.strip().lstrip('!')
        if not tok or '&' not in tok:
            continue
        try:
            a, b = tok.split('&', 1)
            i = int(a) - 1
            j = int(b) - 1
        except Exception:
            continue
        out.add(tuple(sorted((i, j))))
    return out


def _load_planes(args: argparse.Namespace) -> list[tuple[str, int, np.ndarray]]:
    planes: list[tuple[str, int, np.ndarray]] = []
    if args.cube:
        cube = Path(args.cube).expanduser().resolve()
        with fits.open(cube, memmap=True) as h:
            arr = np.asarray(h[0].data, dtype=np.float32)
        if arr.ndim != 3:
            raise SystemExit(f'Expected 3D cube for --cube, got shape={arr.shape}')
        for i in range(arr.shape[0]):
            planes.append((cube.name, i + 1, arr[i]))
        return planes

    for fp in args.images:
        p = Path(fp).expanduser().resolve()
        with fits.open(p, memmap=True) as h:
            arr = np.asarray(h[0].data, dtype=np.float32)
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise SystemExit(f'Expected 2D image for {p}, got shape={arr.shape}')
        planes.append((p.name, 1, arr))
    return planes


def _load_ms(ms_path: Path) -> dict[str, Any]:
    uvw = np.asarray(_read_table_col(str(ms_path), 'UVW'), dtype=np.float64)
    a1 = _read_table_col(str(ms_path), 'ANTENNA1')
    a2 = _read_table_col(str(ms_path), 'ANTENNA2')
    times = _read_table_col(str(ms_path), 'TIME')

    chan_freq = _read_table_col(str(ms_path / 'SPECTRAL_WINDOW'), 'CHAN_FREQ')[0]
    names = _read_table_col(str(ms_path / 'ANTENNA'), 'NAME')
    stations = _read_table_col(str(ms_path / 'ANTENNA'), 'STATION')

    if uvw.ndim != 2:
        raise RuntimeError(f'Unexpected UVW shape {uvw.shape}; expected 2D array')
    if uvw.shape[1] == 3:
        uvw_nx3 = uvw
    elif uvw.shape[0] == 3:
        uvw_nx3 = uvw.T
    else:
        raise RuntimeError(f'Unexpected UVW shape {uvw.shape}; expected (nrow,3) or (3,nrow)')

    lam = 299792458.0 / float(np.mean(chan_freq))
    u = uvw_nx3[:, 0] / lam
    v = uvw_nx3[:, 1] / lam
    uniq_times = np.unique(times)

    return {
        'u': u,
        'v': v,
        'a1': a1,
        'a2': a2,
        'times': times,
        'uniq_times': uniq_times,
        'names': names,
        'stations': stations,
        'mean_freq_mhz': float(np.mean(chan_freq) / 1e6),
    }


def _dominant_fft_mode(img: np.ndarray, min_radius_pix: float, max_radius_frac: float) -> tuple[int, int, float, float]:
    img2 = np.nan_to_num(img, nan=0.0)
    if not np.isfinite(img2).any():
        return (0, 0, 0.0, 0.0)
    ny, nx = img2.shape
    img2 = img2 - np.median(img2)
    win = np.hanning(ny)[:, None] * np.hanning(nx)[None, :]
    F = np.fft.fftshift(np.fft.fft2(img2 * win))
    P = np.abs(F) ** 2

    cy, cx = ny // 2, nx // 2
    yy, xx = np.indices((ny, nx))
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    P[(rr < min_radius_pix) | (rr > max_radius_frac * min(nx, ny))] = 0.0

    py, px = np.unravel_index(np.argmax(P), P.shape)
    du = int(px - cx)
    dv = int(py - cy)
    stripe_angle = (np.degrees(np.arctan2(dv, du)) + 90.0) % 180.0 if (du != 0 or dv != 0) else 0.0
    peak_power = float(P[py, px])
    return du, dv, stripe_angle, peak_power


def _pair_to_selector(pair_list: list[tuple[int, int]], one_based: bool = True) -> str:
    if one_based:
        return ';'.join([f'{i + 1}&{j + 1}' for i, j in pair_list])
    return ';'.join([f'{i}&{j}' for i, j in pair_list])


def main() -> int:
    args = _parse_args()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    ms_path = Path(args.ms).expanduser().resolve()
    if not ms_path.exists():
        raise SystemExit(f'MS not found: {ms_path}')

    planes = _load_planes(args)
    ms = _load_ms(ms_path)
    ignore_pairs = _parse_selector_pairs(args.ignore_selector)

    cell_rad = np.deg2rad(float(args.cell_arcsec) / 3600.0)
    pair_votes: dict[tuple[int, int], int] = {}
    pair_rowsum: dict[tuple[int, int], int] = {}
    results: list[PlaneResult] = []

    uniq_times = ms['uniq_times']
    ntime = len(uniq_times)

    for k, (src, plane_idx, img) in enumerate(planes, start=1):
        du, dv, stripe_angle, peak_power = _dominant_fft_mode(img, float(args.min_radius_pix), float(args.max_radius_frac))
        if du == 0 and dv == 0:
            continue

        ny, nx = img.shape
        u0 = du / (nx * cell_rad)
        v0 = dv / (ny * cell_rad)
        uvmag = float(np.hypot(u0, v0))

        if len(planes) == ntime:
            tm = uniq_times[k - 1]
            m = ms['times'] == tm
        else:
            m = np.ones_like(ms['a1'], dtype=bool)

        u = ms['u'][m]
        v = ms['v'][m]
        a1 = ms['a1'][m]
        a2 = ms['a2'][m]
        if u.size == 0:
            continue

        d = np.minimum(np.hypot(u - u0, v - v0), np.hypot(u + u0, v + v0))
        sel = np.argsort(d)[: min(int(args.top_k_rows), d.size)]

        local: dict[tuple[int, int], int] = {}
        for j in sel:
            p = tuple(sorted((int(a1[j]), int(a2[j]))))
            if p[0] == p[1]:
                continue
            if p in ignore_pairs:
                continue
            local[p] = local.get(p, 0) + 1
        if not local:
            continue

        best_pair, best_hits = sorted(local.items(), key=lambda kv: kv[1], reverse=True)[0]
        pair_votes[best_pair] = pair_votes.get(best_pair, 0) + 1
        pair_rowsum[best_pair] = pair_rowsum.get(best_pair, 0) + int(best_hits)

        results.append(
            PlaneResult(
                plane_index=plane_idx,
                source_file=src,
                du_pix=du,
                dv_pix=dv,
                uv_lambda=uvmag,
                stripe_angle_deg=float(stripe_angle),
                best_pair=best_pair,
                best_pair_hits=int(best_hits),
            )
        )

    ranked = sorted(pair_votes.items(), key=lambda kv: (-kv[1], -pair_rowsum[kv[0]]))
    top = ranked[: int(args.top_n_baselines)]
    top_pairs = [p for p, _ in top]
    selector = _pair_to_selector(top_pairs, one_based=True)

    names = ms['names']
    stations = ms['stations']

    # --- plots ---
    if results:
        angles = np.array([r.stripe_angle_deg for r in results], dtype=float)
        mags = np.array([r.uv_lambda for r in results], dtype=float)
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        ang = angles[np.isfinite(angles)]
        ang_bins = 24 if (ang.size > 1 and float(np.nanmax(ang) - np.nanmin(ang)) > 0.0) else 1
        try:
            plt.hist(ang, bins=ang_bins, color='tab:blue', alpha=0.8)
        except Exception:
            plt.hist(np.asarray(ang, dtype=np.float64).ravel(), bins=1, color='tab:blue', alpha=0.8)
        plt.xlabel('Stripe angle (deg)')
        plt.ylabel('Count')
        plt.title('Stripe angle distribution')
        plt.subplot(1, 2, 2)
        mm = mags[np.isfinite(mags)]
        mag_bins = 24 if (mm.size > 1 and float(np.nanmax(mm) - np.nanmin(mm)) > 0.0) else 1
        try:
            plt.hist(mm, bins=mag_bins, color='tab:orange', alpha=0.8)
        except Exception:
            plt.hist(np.asarray(mm, dtype=np.float64).ravel(), bins=1, color='tab:orange', alpha=0.8)
        plt.xlabel('|uv| of dominant mode (lambda)')
        plt.ylabel('Count')
        plt.title('Dominant stripe spatial frequency')
        plt.tight_layout()
        plt.savefig(outdir / f'{args.tag}_stripe_mode_hist.png', dpi=140)
        plt.close()

    if top:
        labels = [f"{names[i]}-{names[j]}" for (i, j), _ in top]
        votes = [pair_votes[p] for p, _ in top]
        plt.figure(figsize=(max(8, 0.8 * len(labels)), 4.5))
        x = np.arange(len(labels))
        plt.bar(x, votes, color='tab:red', alpha=0.85)
        plt.xticks(x, labels, rotation=45, ha='right')
        plt.ylabel('Planes matched')
        plt.title('Recurring suspicious baselines')
        plt.tight_layout()
        plt.savefig(outdir / f'{args.tag}_baseline_votes.png', dpi=140)
        plt.close()

    # --- report ---
    report: dict[str, Any] = {
        'inputs': {
            'ms': str(ms_path),
            'cube': str(args.cube) if args.cube else None,
            'images': [str(x) for x in args.images] if args.images else None,
            'cell_arcsec': float(args.cell_arcsec),
            'mean_freq_mhz': ms['mean_freq_mhz'],
            'planes_loaded': len(planes),
            'planes_analyzed': len(results),
            'ignore_selector': args.ignore_selector,
        },
        'top_suspicious_baselines': [
            {
                'ant_pair': [int(i), int(j)],
                'name_pair': f'{names[i]}-{names[j]}',
                'station_pair': f'{stations[i]}-{stations[j]}',
                'votes': int(pair_votes[(i, j)]),
                'rowsum': int(pair_rowsum[(i, j)]),
            }
            for (i, j), _ in top
        ],
        'casa_selector': selector,
        'casa_actions': {
            'plot_suspects_only': f"plotms(vis='YOUR.ms', xaxis='time', yaxis='amp', antenna='{selector}', coloraxis='baseline')",
            'exclude_via_flagcopy': [
                "split(vis='YOUR.ms', outputvis='YOUR_suspects_flagged.ms', datacolumn='all')",
                f"flagdata(vis='YOUR_suspects_flagged.ms', mode='manual', antenna='{selector}', flagbackup=False)",
                "tclean(vis='YOUR_suspects_flagged.ms', ...)",
            ],
            'direct_tclean_antenna_exclude_hint': f"tclean(..., antenna='!{selector}')",
        },
        'plane_results': [
            {
                'source_file': r.source_file,
                'plane_index': int(r.plane_index),
                'du_pix': int(r.du_pix),
                'dv_pix': int(r.dv_pix),
                'uv_lambda': float(r.uv_lambda),
                'stripe_angle_deg': float(r.stripe_angle_deg),
                'best_pair': [int(r.best_pair[0]), int(r.best_pair[1])],
                'best_pair_name': f"{names[r.best_pair[0]]}-{names[r.best_pair[1]]}",
                'best_pair_hits': int(r.best_pair_hits),
            }
            for r in results
        ],
    }

    out_json = outdir / f'{args.tag}_report.json'
    out_txt = outdir / f'{args.tag}_summary.txt'
    out_json.write_text(json.dumps(report, indent=2))

    lines = []
    lines.append(f"[diag] planes analyzed: {len(results)} / {len(planes)}")
    lines.append(f"[diag] mean frequency: {ms['mean_freq_mhz']:.3f} MHz")
    lines.append('[diag] top suspicious baselines:')
    for item in report['top_suspicious_baselines']:
        lines.append(
            f"  - {item['name_pair']} ({item['station_pair']}): votes={item['votes']} rowsum={item['rowsum']}"
        )
    lines.append(f"[diag] CASA selector: {selector}")
    lines.append('[diag] outputs:')
    lines.append(f"  - {out_json}")
    lines.append(f"  - {out_txt}")
    lines.append(f"  - {outdir / (args.tag + '_stripe_mode_hist.png')}")
    lines.append(f"  - {outdir / (args.tag + '_baseline_votes.png')}")
    out_txt.write_text('\n'.join(lines) + '\n')

    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
