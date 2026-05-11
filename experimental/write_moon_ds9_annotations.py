#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import CIRS, EarthLocation, FK5, SkyCoord, get_body, solar_system_ephemeris
from astropy.io import fits
from astropy.time import Time


def _load_cube_times_mjd(cube_path: Path) -> np.ndarray:
    with fits.open(cube_path) as hdul:
        if 'TIMEAXIS' in hdul and 'TIME_MJD' in hdul['TIMEAXIS'].columns.names:
            arr = np.asarray(hdul['TIMEAXIS'].data['TIME_MJD'], dtype=np.float64)
            return arr

        hdr = hdul[0].header
        n3 = int(hdr.get('NAXIS3', 0))
        crval3 = float(hdr.get('CRVAL3', 0.0))
        cdelt3 = float(hdr.get('CDELT3', 0.0))
        crpix3 = float(hdr.get('CRPIX3', 1.0))
        if n3 <= 0:
            raise RuntimeError(f'Could not infer time axis from {cube_path}')
        idx = np.arange(1, n3 + 1, dtype=np.float64)
        return crval3 + (idx - crpix3) * cdelt3


def _moon_j2000_via_casa(times_utc: Time, gmrt: EarthLocation) -> SkyCoord:
    from casatools import measures, quanta  # type: ignore

    me = measures()
    qa = quanta()
    me.doframe(me.position('WGS84', '74.0497deg', '19.0965deg', '650m'))

    ra_deg: list[float] = []
    dec_deg: list[float] = []
    for ti in times_utc:
        moon_app = get_body('moon', ti, location=gmrt)
        app_ra = moon_app.ra.to_string(unit=u.hour, sep='hms', precision=6, pad=True)
        app_dec = moon_app.dec.to_string(unit=u.deg, sep='dms', precision=6, alwayssign=True, pad=True)

        me.doframe(me.epoch('UTC', ti.isot))
        app_dir = me.direction('APP', app_ra, app_dec)
        j2000_dir = me.measure(app_dir, 'J2000')

        ra_deg.append(float(qa.convert(j2000_dir['m0'], 'deg')['value']))
        dec_deg.append(float(qa.convert(j2000_dir['m1'], 'deg')['value']))

    return SkyCoord(np.asarray(ra_deg) * u.deg, np.asarray(dec_deg) * u.deg, frame=FK5(equinox='J2000'))


def _moon_j2000_astropy(times_utc: Time, gmrt: EarthLocation, ephemeris: str = 'de440s') -> tuple[SkyCoord, float]:
    with solar_system_ephemeris.set(ephemeris):
        moon_gcrs = get_body('moon', times_utc, location=gmrt)

    # Equivalent routes should agree to sub-arcsecond when invoked correctly.
    moon_fk5_direct = moon_gcrs.transform_to(FK5(equinox='J2000'))
    moon_fk5_via_cirs = moon_gcrs.transform_to(
        CIRS(obstime=times_utc, location=gmrt)
    ).transform_to(FK5(equinox='J2000'))

    max_internal_sep_arcsec = float(np.max(moon_fk5_direct.separation(moon_fk5_via_cirs).to(u.arcsec).value))
    return moon_fk5_via_cirs, max_internal_sep_arcsec


def cast_to_fk5(coord: SkyCoord) -> SkyCoord:
    return coord.transform_to(FK5(equinox='J2000'))


def format_ra_dec(ra_deg: float, dec_deg: float) -> tuple[str, str]:
    c = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame=FK5(equinox='J2000'))
    ra = c.ra.to_string(unit=u.hour, sep=':', precision=3, pad=True)
    dec = c.dec.to_string(unit=u.deg, sep=':', precision=3, alwayssign=True, pad=True)
    return ra, dec


def main() -> int:
    p = argparse.ArgumentParser(description='Write DS9 Moon annotation regions per integration for a cube.')
    p.add_argument('--cube', required=True, help='Path to *_moontrack_cube.image.fits')
    p.add_argument('--outdir', default='', help='Output directory (default: <cube_dir>/<cube_stem>_regions)')
    p.add_argument('--moon-radius-arcmin', type=float, default=15.5, help='Lunar disk radius in arcmin (default: 15.5)')
    p.add_argument('--color', default='cyan', help='DS9 color for circle/point (default: cyan)')
    p.add_argument('--time-scale', choices=['tai', 'utc'], default='tai',
                   help='Scale of TIME_MJD in cube TIMEAXIS (default: tai; MS TIME is typically TAI).')
    p.add_argument('--engine', choices=['auto', 'casa', 'astropy'], default='auto',
                   help='Moon ephemeris engine: auto (try CASA then Astropy), casa, or astropy.')
    p.add_argument('--astropy-ephemeris', default='de440s',
                   help='Astropy solar-system ephemeris (default: de440s; requires jplephem).')
    args = p.parse_args()

    cube_path = Path(args.cube).resolve()
    if not cube_path.exists():
        raise SystemExit(f'Cube not found: {cube_path}')

    outdir = Path(args.outdir).resolve() if args.outdir else cube_path.parent / f'{cube_path.stem}_regions'
    outdir.mkdir(parents=True, exist_ok=True)

    times_mjd = _load_cube_times_mjd(cube_path)
    times_obs = Time(times_mjd, format='mjd', scale=str(args.time_scale).lower())
    times_utc = times_obs.utc

    gmrt = EarthLocation.from_geodetic(lon=74.0497 * u.deg, lat=19.0965 * u.deg, height=650 * u.m)

    astropy_internal_sep_arcsec: float | None = None
    engine = str(args.engine).lower()

    if engine == 'casa':
        moon_j2000 = _moon_j2000_via_casa(times_utc, gmrt)
        ephem_method = 'casa_app_to_j2000'
    elif engine == 'astropy':
        moon_j2000, astropy_internal_sep_arcsec = _moon_j2000_astropy(
            times_utc,
            gmrt,
            ephemeris=str(args.astropy_ephemeris),
        )
        ephem_method = f'astropy_{args.astropy_ephemeris}_cirs_to_fk5_j2000'
    else:
        try:
            moon_j2000 = _moon_j2000_via_casa(times_utc, gmrt)
            ephem_method = 'casa_app_to_j2000'
        except Exception:
            moon_j2000, astropy_internal_sep_arcsec = _moon_j2000_astropy(
                times_utc,
                gmrt,
                ephemeris=str(args.astropy_ephemeris),
            )
            ephem_method = f'astropy_{args.astropy_ephemeris}_cirs_to_fk5_j2000'

    radius_arcsec = float(args.moon_radius_arcmin) * 60.0

    index_csv = outdir / 'index.csv'
    with index_csv.open('w', newline='') as fcsv:
        w = csv.writer(fcsv)
        w.writerow(['integration_index_1based', 'mjd_utc', 'utc_iso', 'ra_deg_j2000', 'dec_deg_j2000', 'region_file'])

        for i in range(len(times_mjd)):
            idx = i + 1
            ra_deg = float(moon_j2000.ra.deg[i])
            dec_deg = float(moon_j2000.dec.deg[i])
            ra_hms, dec_dms = format_ra_dec(ra_deg, dec_deg)
            utc_iso = times_utc[i].isot

            reg_name = f'int{idx:04d}.reg'
            reg_path = outdir / reg_name

            region_text = (
                '# Region file format: DS9 version 4.1\n'
                f'global color={args.color} dashlist=8 3 width=2 font="helvetica 10 normal roman" '
                'select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n'
                'fk5\n'
                f'circle({ra_hms},{dec_dms},{radius_arcsec:.3f}\") # text={{Moon disk int{idx:04d} {utc_iso}}}\n'
                f'point({ra_hms},{dec_dms}) # point=x color={args.color} text={{Moon ctr int{idx:04d}}}\n'
            )
            reg_path.write_text(region_text)

            w.writerow([idx, f'{times_mjd[i]:.10f}', utc_iso, f'{ra_deg:.10f}', f'{dec_deg:.10f}', reg_name])

    all_points = outdir / 'all_positions.reg'
    lines = [
        '# Region file format: DS9 version 4.1',
        f'global color={args.color} dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1',
        'fk5',
    ]
    for i in range(len(times_mjd)):
        idx = i + 1
        ra_hms, dec_dms = format_ra_dec(float(moon_j2000.ra.deg[i]), float(moon_j2000.dec.deg[i]))
        lines.append(f'point({ra_hms},{dec_dms}) # point=cross text={{int{idx:04d}}}')
    all_points.write_text('\n'.join(lines) + '\n')

    print(f'Wrote regions to: {outdir}')
    print(f'Index CSV: {index_csv}')
    print(f'Combined track: {all_points}')
    print(f'Integrations: {len(times_mjd)}')
    print(f'Ephemeris method: {ephem_method}')
    print(f'Time scale interpreted: {args.time_scale} (converted to UTC for ephemeris)')
    if astropy_internal_sep_arcsec is not None:
        print(f'Astropy internal frame-consistency max separation: {astropy_internal_sep_arcsec:.6f} arcsec')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
