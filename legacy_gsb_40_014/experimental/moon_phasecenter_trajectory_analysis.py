#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import EarthLocation, FK5, SkyCoord, get_body
from astropy.io import fits
from astropy.time import Time
import astropy.units as u
from pyuvdata import UVData


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Analyze Moon trajectory relative to UVFITS phase-tracking center.')
    p.add_argument('--moon-dir', default='/Users/raj030/DATA/gmrt_40_014/work/split/moon', help='Directory with moon*_calibrated.uvfits')
    p.add_argument('--pattern', default='moon*_calibrated.uvfits', help='Glob pattern for scan files')
    p.add_argument('--outdir', default='./work/moon_trajectory', help='Output directory')
    p.add_argument('--make-gif', action='store_true', help='Attempt to write animated GIF per scan')
    return p.parse_args()


def read_uvfits_with_minimal_fix(path: Path) -> tuple[UVData, dict[str, Any]]:
    uv = UVData()
    try:
        uv.read_uvfits(str(path), run_check=False)
        return uv, {'used_fix': False, 'patches': []}
    except Exception as exc:
        tempdir = Path(tempfile.mkdtemp(prefix='moontraj_'))
        patched = tempdir / path.name
        shutil.copy2(path, patched)
        patches: list[dict[str, Any]] = []
        with fits.open(patched, mode='update') as hdul:
            ph = hdul[0].header
            if ph.get('RADESYS') == 'FK5':
                ph['RADESYS'] = 'fk5'
                patches.append({'hdu': 'PRIMARY', 'key': 'RADESYS', 'old': 'FK5', 'new': 'fk5'})
            if ph.get('TIMESYS') is None:
                ph['TIMESYS'] = 'UTC'
                patches.append({'hdu': 'PRIMARY', 'key': 'TIMESYS', 'old': None, 'new': 'UTC'})
            if 'AIPS AN' in hdul:
                ah = hdul['AIPS AN'].header
                if ah.get('FRAME') is None:
                    ah['FRAME'] = 'ITRF'
                    patches.append({'hdu': 'AIPS AN', 'key': 'FRAME', 'old': None, 'new': 'ITRF'})
                if ah.get('XYZHAND') is None:
                    ah['XYZHAND'] = 'RIGHT'
                    patches.append({'hdu': 'AIPS AN', 'key': 'XYZHAND', 'old': None, 'new': 'RIGHT'})
            hdul.flush(output_verify='ignore')
        uv.read_uvfits(str(patched), run_check=False)
        return uv, {'used_fix': True, 'patches': patches, 'read_error': str(exc), 'temporary_copy': str(patched)}


def get_su_phasecenter(path: Path) -> dict[str, Any]:
    with fits.open(path) as hdul:
        su = hdul['AIPS SU'].data
        row = su[0]
        src = row['SOURCE'].strip() if hasattr(row['SOURCE'], 'strip') else str(row['SOURCE'])
        raepo = float(row['RAEPO'])
        decepo = float(row['DECEPO'])
        epoch = float(row['EPOCH']) if 'EPOCH' in su.columns.names else 2000.0
    coord = SkyCoord(raepo * u.deg, decepo * u.deg, frame=FK5(equinox='J2000'))
    return {
        'source': src,
        'ra_deg': raepo,
        'dec_deg': decepo,
        'epoch': epoch,
        'coord': coord,
    }


def unwrap_center_ra(coord: SkyCoord) -> float:
    ra = float(coord.ra.rad)
    if ra < 0:
        ra += 2.0 * np.pi
    return ra


def moon_j2000_via_casa(t: Time, gmrt: EarthLocation) -> SkyCoord:
    from casatools import measures, quanta  # type: ignore

    me = measures()
    qa = quanta()
    me.doframe(me.position('WGS84', '74.0497deg', '19.0965deg', '650m'))

    ra_deg: list[float] = []
    dec_deg: list[float] = []
    for ti in t:
        moon_app = get_body('moon', ti, location=gmrt)
        app_ra = moon_app.ra.to_string(unit=u.hour, sep='hms', precision=6, pad=True)
        app_dec = moon_app.dec.to_string(unit=u.deg, sep='dms', precision=6, alwayssign=True, pad=True)

        me.doframe(me.epoch('UTC', ti.isot))
        app_dir = me.direction('APP', app_ra, app_dec)
        j2000_dir = me.measure(app_dir, 'J2000')

        ra_deg.append(float(qa.convert(j2000_dir['m0'], 'deg')['value']))
        dec_deg.append(float(qa.convert(j2000_dir['m1'], 'deg')['value']))

    return SkyCoord(np.asarray(ra_deg) * u.deg, np.asarray(dec_deg) * u.deg, frame=FK5(equinox='J2000'))


def analyze_one(path: Path, outdir: Path, make_gif: bool) -> dict[str, Any]:
    uv, read_info = read_uvfits_with_minimal_fix(path)
    su_center = get_su_phasecenter(path)

    cat = uv.phase_center_catalog or {}
    cat_entry = next(iter(cat.values()))
    uv_ra = float(cat_entry['cat_lon'])
    uv_dec = float(cat_entry['cat_lat'])
    if uv_ra < 0:
        uv_ra += 2.0 * np.pi
    uv_center = SkyCoord(uv_ra * u.rad, uv_dec * u.rad, frame=FK5(equinox='J2000'))

    su_center_coord = su_center['coord']
    su_ra = unwrap_center_ra(su_center_coord)

    times_jd = np.unique(np.round(np.asarray(uv.time_array, dtype=np.float64), 12))
    t = Time(times_jd, format='jd', scale='utc')

    gmrt = EarthLocation.from_geodetic(lon=74.0497 * u.deg, lat=19.0965 * u.deg, height=650 * u.m)
    moon_j2000 = moon_j2000_via_casa(t, gmrt)

    # offsets wrt UVData phase center (the center used by imaging workflow)
    dra_arcmin = ((moon_j2000.ra - uv_center.ra).wrap_at(180 * u.deg).to(u.rad).value * np.cos(uv_center.dec.rad)) * (180 / np.pi) * 60
    ddec_arcmin = (moon_j2000.dec - uv_center.dec).to(u.deg).value * 60
    sep_arcmin = uv_center.separation(moon_j2000).to(u.arcmin).value

    idx_min = int(np.argmin(sep_arcmin))

    # plotting
    tag = path.stem.replace('_calibrated', '')
    fig = plt.figure(figsize=(11, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    mins_from_start = (t - t[0]).to_value(u.s) / 60.0
    sc = ax1.scatter(dra_arcmin, ddec_arcmin, c=mins_from_start, cmap='viridis', s=18)
    ax1.axhline(0, color='k', lw=0.8, alpha=0.5)
    ax1.axvline(0, color='k', lw=0.8, alpha=0.5)
    ax1.scatter([0], [0], marker='+', s=100, color='red', label='Phase center')
    ax1.scatter([dra_arcmin[idx_min]], [ddec_arcmin[idx_min]], marker='x', s=70, color='orange', label='Closest moon')
    ax1.set_xlabel('ΔRA cos(dec) [arcmin]')
    ax1.set_ylabel('ΔDec [arcmin]')
    ax1.set_title(f'{tag}: Moon trajectory around phase center')
    ax1.legend(loc='best', fontsize=8)
    cb = fig.colorbar(sc, ax=ax1)
    cb.set_label('Minutes from scan start')

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot_date(t.datetime, sep_arcmin, '-', lw=1.2)
    ax2.axvline(t[idx_min].datetime, color='orange', ls='--', lw=1)
    ax2.set_ylabel('Moon-center separation [arcmin]')
    ax2.set_title('Separation vs time')
    ax2.tick_params(axis='x', rotation=30)

    fig.tight_layout()
    png_path = outdir / f'{tag}_trajectory.png'
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    gif_path = None
    gif_error = None
    if make_gif:
        try:
            from matplotlib import animation

            fig2, ax = plt.subplots(figsize=(5.5, 5.0))
            ax.axhline(0, color='k', lw=0.8, alpha=0.5)
            ax.axvline(0, color='k', lw=0.8, alpha=0.5)
            ax.scatter([0], [0], marker='+', s=100, color='red')
            span = max(10.0, float(np.nanmax(np.abs(np.r_[dra_arcmin, ddec_arcmin])) * 1.2))
            ax.set_xlim(-span, span)
            ax.set_ylim(-span, span)
            ax.set_xlabel('ΔRA cos(dec) [arcmin]')
            ax.set_ylabel('ΔDec [arcmin]')
            ax.set_title(f'{tag}: Moon trajectory')
            line, = ax.plot([], [], '-', color='tab:blue', lw=1)
            point, = ax.plot([], [], 'o', color='tab:orange', ms=5)

            def _init():
                line.set_data([], [])
                point.set_data([], [])
                return line, point

            def _update(i: int):
                line.set_data(dra_arcmin[: i + 1], ddec_arcmin[: i + 1])
                point.set_data([dra_arcmin[i]], [ddec_arcmin[i]])
                ax.set_title(f"{tag}: {t[i].isot}")
                return line, point

            anim = animation.FuncAnimation(fig2, _update, init_func=_init, frames=len(t), interval=80, blit=True)
            gif_path = outdir / f'{tag}_trajectory.gif'
            anim.save(gif_path, writer='pillow', fps=10)
            plt.close(fig2)
        except Exception as exc:
            gif_error = str(exc)

    return {
        'scan': tag,
        'file': str(path),
        'read_info': read_info,
        'phasecenter_source': su_center['source'],
        'phasecenter_su_ra_deg': su_center['ra_deg'],
        'phasecenter_su_dec_deg': su_center['dec_deg'],
        'phasecenter_uv_ra_deg': float(np.degrees(uv_ra)),
        'phasecenter_uv_dec_deg': float(np.degrees(uv_dec)),
        'phasecenter_ra_hms_dec_dms': uv_center.to_string('hmsdms'),
        'scan_start_utc': t.min().isot,
        'scan_end_utc': t.max().isot,
        'n_times': int(len(t)),
        'closest_time_utc': t[idx_min].isot,
        'min_sep_arcmin': float(sep_arcmin[idx_min]),
        'min_sep_arcsec': float(sep_arcmin[idx_min] * 60.0),
        'start_sep_arcmin': float(sep_arcmin[0]),
        'end_sep_arcmin': float(sep_arcmin[-1]),
        'trajectory_png': str(png_path),
        'trajectory_gif': str(gif_path) if gif_path else None,
        'trajectory_gif_error': gif_error,
    }


def main() -> int:
    args = parse_args()
    moon_dir = Path(args.moon_dir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    scans = sorted(moon_dir.glob(args.pattern))
    if not scans:
        raise SystemExit(f'No scan files found: {moon_dir}/{args.pattern}')

    reports = [analyze_one(path, outdir, make_gif=bool(args.make_gif)) for path in scans]

    json_path = outdir / 'moon_scan_phasecenter_trajectory_report.json'
    json_path.write_text(json.dumps(reports, indent=2))

    csv_path = outdir / 'moon_scan_phasecenter_trajectory_summary.csv'
    fields = [
        'scan', 'phasecenter_source', 'phasecenter_ra_hms_dec_dms',
        'scan_start_utc', 'scan_end_utc', 'n_times', 'closest_time_utc',
        'min_sep_arcmin', 'min_sep_arcsec', 'start_sep_arcmin', 'end_sep_arcmin',
        'trajectory_png', 'trajectory_gif',
    ]
    with csv_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in reports:
            w.writerow({k: r.get(k) for k in fields})

    print(f'[traj] scans analyzed: {len(reports)}')
    print(f'[traj] wrote {json_path}')
    print(f'[traj] wrote {csv_path}')
    print('[traj] minimum separations:')
    for r in reports:
        print(f"  - {r['scan']}: min_sep={r['min_sep_arcmin']:.2f} arcmin at {r['closest_time_utc']}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
