#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, get_body
from astropy.time import Time
from casatools import table

MS_PATH = 'casa_out/moon/moon0520.ms'


def main() -> int:
    tb = table()
    tb.open(MS_PATH)
    times_sec = np.unique(np.round(np.asarray(tb.getcol('TIME'), dtype=float), 6))
    tb.close()

    if times_sec.size == 0:
        raise RuntimeError(f'No TIME rows found in {MS_PATH}')

    location = EarthLocation.from_geodetic(lon=74.0497 * u.deg, lat=19.0965 * u.deg, height=650 * u.m)
    times = Time(times_sec / 86400.0, format='mjd', scale='utc', location=location)

    moon = get_body('moon', times, location=location)
    altaz_frame = AltAz(obstime=times, location=location)
    moon_alt = moon.transform_to(altaz_frame).alt.deg

    min_el = float(np.min(moon_alt))
    med_el = float(np.median(moon_alt))
    max_el = float(np.max(moon_alt))

    frac_lt_20 = float(np.mean(moon_alt < 20.0))
    frac_lt_30 = float(np.mean(moon_alt < 30.0))
    frac_lt_40 = float(np.mean(moon_alt < 40.0))

    t_start = times.min().utc.isot
    t_end = times.max().utc.isot

    print(f'ms={MS_PATH}')
    print(f'integrations={times_sec.size}')
    print(f'utc_range={t_start} -> {t_end}')
    print(f'elevation_deg min/med/max = {min_el:.3f} / {med_el:.3f} / {max_el:.3f}')
    print(f'fraction_below_20deg = {frac_lt_20:.3f}')
    print(f'fraction_below_30deg = {frac_lt_30:.3f}')
    print(f'fraction_below_40deg = {frac_lt_40:.3f}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
