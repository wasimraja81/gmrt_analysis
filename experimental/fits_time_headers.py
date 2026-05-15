#!/usr/bin/env python3
"""Utilities for writing time bookkeeping metadata into FITS headers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from astropy.io import fits
from astropy.time import Time


def _iso_utc(jd_utc: float) -> str:
    return Time(jd_utc, format='jd', scale='utc').isot


def _mjd_utc(jd_utc: float) -> float:
    return float(Time(jd_utc, format='jd', scale='utc').mjd)


def write_extra_header_to_fits_image(
    fits_path: str | Path,
    *,
    jd_start: float,
    jd_end: float,
    jd_avg: float,
    jd_mid: Optional[float] = None,
    jd_mean: Optional[float] = None,
    nint: Optional[int] = None,
    inttime_sec: Optional[float] = None,
    start_integration: Optional[int] = None,
    moon_ra_deg: Optional[float] = None,
    moon_dec_deg: Optional[float] = None,
    source_tag: str = 'selfcal',
) -> None:
    """Write/patch FITS header timing metadata in-place.

    - DATE-OBS/MJD-OBS: stack start
    - DATE-END/MJD-END: stack end
    - DATE-AVG/MJD-AVG: representative image time
    - JDSTART/JDEND/JDAVG/JDMID/JDMEAN: explicit JD values for reproducibility
    - NINT/INTTIME/INTSTART: integration bookkeeping
    """
    path = Path(fits_path)
    if not path.exists():
        raise FileNotFoundError(f'FITS not found: {path}')

    with fits.open(path, mode='update') as hdul:
        hdr = hdul[0].header

        hdr['TIMESYS'] = ('UTC', 'Time scale for DATE/JD/MJD keywords')

        hdr['DATE-OBS'] = (_iso_utc(jd_start), 'Observation start (UTC)')
        hdr['MJD-OBS'] = (_mjd_utc(jd_start), 'Observation start (MJD, UTC)')

        hdr['DATE-END'] = (_iso_utc(jd_end), 'Observation end (UTC)')
        hdr['MJD-END'] = (_mjd_utc(jd_end), 'Observation end (MJD, UTC)')

        hdr['DATE-AVG'] = (_iso_utc(jd_avg), 'Representative image time (UTC)')
        hdr['MJD-AVG'] = (_mjd_utc(jd_avg), 'Representative image time (MJD, UTC)')

        hdr['JDSTART'] = (float(jd_start), 'Observation start (JD, UTC)')
        hdr['JDEND'] = (float(jd_end), 'Observation end (JD, UTC)')
        hdr['JDAVG'] = (float(jd_avg), 'Representative image time (JD, UTC)')

        if jd_mid is not None:
            hdr['JDMID'] = (
                float(jd_mid),
                'Sample-centred JD (n//2 dump) of contributing data',
            )
        # Ensure legacy alias is removed if present.
        if 'JDMASK' in hdr:
            del hdr['JDMASK']
        if jd_mean is not None:
            hdr['JDMEAN'] = (float(jd_mean), 'Arithmetic mean JD of contributing dumps')

        telapse_sec = max(0.0, (jd_end - jd_start) * 86400.0)
        hdr['TELAPSE'] = (float(telapse_sec), 'Elapsed time [s] between DATE-OBS and DATE-END')

        if nint is not None:
            hdr['NINT'] = (int(nint), 'Integrations combined in this image')
        if inttime_sec is not None:
            hdr['INTTIME'] = (float(inttime_sec), 'Integration cadence [s]')
        if start_integration is not None:
            hdr['INTSTART'] = (int(start_integration), '0-based start integration index')

        if moon_ra_deg is not None:
            hdr['MOONRA'] = (float(moon_ra_deg), 'Moon RA used for mask [deg, apparent topocentric]')
        if moon_dec_deg is not None:
            hdr['MOONDEC'] = (float(moon_dec_deg), 'Moon Dec used for mask [deg, apparent topocentric]')

        hdr['HISTORY'] = (
            f'time-header patch ({source_tag}): start/end/avg + NINT/INTTIME metadata written'
        )

        hdul.flush()
