"""Observing-geometry computations: local sidereal time, hour angle,
azimuth/elevation, and parallactic angle, from Julian Date, a sky position,
and an array location.

Generic: works for any equatorial position and any `EarthLocation`, not
specific to GMRT. Callers should pass apparent (not epoch/J2000) RA/Dec --
`source_table.Source.ra_apparent_deg`/`dec_apparent_deg` -- since local
sidereal time here is computed in the apparent frame; mixing an epoch
position with apparent LST would introduce a precession-sized error.
"""

from __future__ import annotations

import erfa
import numpy as np
from astropy.coordinates import EarthLocation
from astropy.time import Time


def local_sidereal_time_hours(jd, location: EarthLocation) -> np.ndarray:
    """Apparent local sidereal time, in hours, at the given Julian Date(s)."""
    time = Time(np.asarray(jd, dtype=float), format="jd", scale="utc")
    return time.sidereal_time("apparent", longitude=location.lon).hour


def hour_angle_hours(jd, ra_deg, location: EarthLocation) -> np.ndarray:
    """Hour angle, in hours, wrapped to [-12, 12)."""
    ha_hours = local_sidereal_time_hours(jd, location) - np.asarray(ra_deg) / 15.0
    return ((ha_hours + 12.0) % 24.0) - 12.0


def altaz_deg(jd, ra_deg, dec_deg, location: EarthLocation) -> tuple[np.ndarray, np.ndarray]:
    """Azimuth (from North, through East) and elevation, in degrees, of a
    sky position as seen from `location` at the given Julian Date(s).

    Computed directly from hour angle, declination, and latitude -- not via
    a separate ICRS-to-apparent frame transform, since the apparent
    declination passed in already accounts for precession/nutation/
    aberration, and mixing it with a frame transform that re-applies those
    corrections from ICRS would double-count them.
    """
    ha_rad = np.radians(hour_angle_hours(jd, ra_deg, location) * 15.0)
    dec_rad = np.radians(dec_deg)
    lat_rad = location.lat.rad

    el_rad = np.arcsin(np.sin(dec_rad) * np.sin(lat_rad) + np.cos(dec_rad) * np.cos(lat_rad) * np.cos(ha_rad))
    az_from_south_rad = np.arctan2(
        np.sin(ha_rad),
        np.cos(ha_rad) * np.sin(lat_rad) - np.tan(dec_rad) * np.cos(lat_rad),
    )
    az_deg = (np.degrees(az_from_south_rad) + 180.0) % 360.0
    return az_deg, np.degrees(el_rad)


def parallactic_angle_deg(jd, ra_deg, dec_deg, location: EarthLocation) -> np.ndarray:
    """Parallactic angle, in degrees, via ERFA's `hd2pa` (hour angle,
    declination, latitude -> parallactic angle)."""
    ha_rad = np.radians(hour_angle_hours(jd, ra_deg, location) * 15.0)
    dec_rad = np.radians(dec_deg)
    return np.degrees(erfa.hd2pa(ha_rad, dec_rad, location.lat.rad))
