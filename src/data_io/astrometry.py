"""Observing-geometry computations: local sidereal time, hour angle,
azimuth/elevation, and parallactic angle, from Julian Date, a sky position,
and an array location.

Generic: works for any equatorial position and any `EarthLocation`, not
specific to GMRT. Callers should pass apparent (not epoch/J2000) RA/Dec --
`source_table.Source.ra_apparent_deg`/`dec_apparent_deg` -- since local
sidereal time here is computed in the apparent frame; mixing an epoch
position with apparent LST would introduce a precession-sized error.

Sidereal time needs UT1 - UTC (Earth-rotation data from IERS tables). It
comes from `Ut1Provider`, which works without a network: astropy's bundled
IERS-B tables (final values) when they cover the dates, the online IERS
tables for later dates, and otherwise UT1 = UTC with a warning -- hour angle
is then off by at most 0.9 s of time (|UT1 - UTC| < 0.9 s by definition).
Sidereal time is taken relative to the terrestrial intermediate origin plus
the site longitude, which needs no further table lookup; it agrees with
astropy's `sidereal_time` at the site longitude to 0.001 ms of time
(measured 2026-09-28).
"""

from __future__ import annotations

import warnings

import erfa
import numpy as np
from astropy.coordinates import EarthLocation
from astropy.time import Time
from astropy.utils import iers

# Bound on |UT1 - UTC| kept by IERS (leap seconds are inserted to keep it so).
MAX_UT1_MINUS_UTC_S = 0.9
ONLINE_TIMEOUT_S = 5.0


class AstrometryWarning(UserWarning):
    pass


class Ut1Provider:
    """UT1 - UTC in seconds for given Julian Dates, from the best source
    available without failing: bundled IERS-B, then the online IERS tables,
    then 0 (UT1 = UTC) with a warning. Records which sources it used, so a
    caller can report them (e.g. on a plot)."""

    def __init__(self, allow_online: bool = True):
        self.allow_online = allow_online
        self.sources_used: set[str] = set()
        self.fallback_used = False

    def ut1_minus_utc_s(self, jd) -> np.ndarray:
        jd = np.atleast_1d(np.asarray(jd, dtype=float))
        time = Time(jd, format="jd", scale="utc")
        bundled = iers.IERS_B.open()
        mjd = jd - 2400000.5
        if mjd.min() >= bundled["MJD"][0].value and mjd.max() <= bundled["MJD"][-1].value:
            self.sources_used.add("IERS-B (bundled with astropy)")
            return bundled.ut1_utc(time).to_value("s")
        if self.allow_online:
            try:
                with iers.conf.set_temp("remote_timeout", ONLINE_TIMEOUT_S):
                    table = iers.IERS_Auto.open()
                    values, status = table.ut1_utc(time, return_status=True)
                if np.all(status >= 0):  # negative: date outside the table
                    self.sources_used.add("IERS-A (online)")
                    return values.to_value("s")
            except Exception:  # no network, server down, stale or partial table
                pass
        self.fallback_used = True
        self.sources_used.add("none: UT1 = UTC assumed")
        warnings.warn(fallback_message(), AstrometryWarning, stacklevel=2)
        return np.zeros_like(jd)


def fallback_message() -> str:
    return (f"UT1 - UTC unavailable for these dates (not in the bundled IERS tables, and the online "
            f"tables could not be used); computed with UT1 = UTC, so hour angle may be off by up to "
            f"{MAX_UT1_MINUS_UTC_S} s of time ({MAX_UT1_MINUS_UTC_S / 3600:.5f} h), and azimuth, "
            f"elevation and parallactic angle correspondingly")


DEFAULT_UT1 = Ut1Provider()


def local_sidereal_time_hours(jd, location: EarthLocation, ut1: Ut1Provider | None = None) -> np.ndarray:
    """Apparent local sidereal time, in hours, at the given Julian Date(s).
    UT1 - UTC comes from `ut1` (default: the shared `DEFAULT_UT1`), so no
    table lookup or download happens inside astropy."""
    jd = np.asarray(jd, dtype=float)
    time = Time(jd, format="jd", scale="utc")
    time.delta_ut1_utc = (ut1 or DEFAULT_UT1).ut1_minus_utc_s(jd).reshape(jd.shape)
    # Greenwich apparent sidereal time relative to the TIO (no polar-motion
    # lookup), plus the site's longitude.
    gast_hours = time.sidereal_time("apparent", longitude="tio").hour
    return np.mod(gast_hours + location.lon.to_value("hourangle"), 24.0)


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
