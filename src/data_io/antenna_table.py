"""Generic AIPS antenna-table (AN) reading: antenna positions and names, for
any random-groups UVFITS file following the AIPS convention -- not specific
to any one telescope.

`STABXYZ` is antenna position relative to the array's own reference position
(`ARRAYX`/`ARRAYY`/`ARRAYZ` in the same table's header), but *not* in the
same (Greenwich-referenced) ECEF frame as `ARRAYX/Y/Z` itself: per AIPS Memo
117 (Greisen), STABXYZ is expressed in ECEF rotated so its own X-axis runs
through the array center's local meridian, not through Greenwich. Adding it
to `ARRAYX/Y/Z` directly (as an earlier version of this module did) is
wrong -- confirmed directly against the real GWB file (2026-09-25): doing
so put antennas up to 12km underground or 12km in the air (checked via
independent geodetic height, height should be ~640m for every antenna on
this site), while rotating STABXYZ's x/y by the array center's own
longitude first brings every antenna to within ~35m of that. `Antenna.x_m`/
`y_m`/`z_m` here are the correctly-rotated absolute ECEF position, not the
raw relative offset.

`name` is read as whatever string the table holds, with no assumption about
its format -- GMRT's own "<code>:<station number>" naming convention (and
the DUD-antenna prefix matching built on it) is a telescope-specific
interpretation of that string, not a fact about the AIPS format, and lives
in `instruments/gmrt/antenna_table.py` instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.coordinates import EarthLocation

from data_io.raw_data_access import open_fits_readonly


@dataclass(frozen=True)
class Antenna:
    station_number: int  # AIPS NOSTA, 1-indexed
    name: str  # ANNAME, exactly as stored
    x_m: float  # absolute ECEF X, metres
    y_m: float  # absolute ECEF Y, metres
    z_m: float  # absolute ECEF Z, metres


def _rotate_stabxyz_to_ecef(dx: float, dy: float, array_x: float, array_y: float) -> tuple[float, float]:
    """Rotate a STABXYZ (dx, dy) from the local-meridian-referenced frame
    AIPS Memo 117 defines it in to the true (Greenwich-referenced) ECEF
    frame `ARRAYX/Y/Z` is expressed in -- a rotation by the array center's
    own longitude, computed directly from `ARRAYX/Y/Z` (longitude is the
    same for geodetic and geocentric coordinates, so this needs no ellipsoid
    conversion)."""
    lon_rad = np.arctan2(array_y, array_x)
    cos_lon, sin_lon = np.cos(lon_rad), np.sin(lon_rad)
    return dx * cos_lon - dy * sin_lon, dx * sin_lon + dy * cos_lon


def read_antenna_table(fits_path: Path | str) -> list[Antenna]:
    """Read every antenna in the AIPS AN table, in table order, with absolute
    ECEF positions."""
    with open_fits_readonly(fits_path) as hdul:
        an = hdul["AIPS AN"]
        array_x, array_y, array_z = _array_reference_position_m(an.header)
        antennas = []
        for row in an.data:
            dx, dy = _rotate_stabxyz_to_ecef(float(row["STABXYZ"][0]), float(row["STABXYZ"][1]), array_x, array_y)
            antennas.append(Antenna(
                station_number=int(row["NOSTA"]),
                name=str(row["ANNAME"]).strip(),
                x_m=array_x + dx,
                y_m=array_y + dy,
                z_m=array_z + float(row["STABXYZ"][2]),
            ))
        return antennas


def read_array_reference_position_m(fits_path: Path | str) -> tuple[float, float, float]:
    """The array's own reference position (absolute ECEF, metres) -- needed
    directly by observing-geometry code (hour angle, Az/El, parallactic
    angle), separately from any one antenna's position."""
    with open_fits_readonly(fits_path) as hdul:
        return _array_reference_position_m(hdul["AIPS AN"].header)


def read_array_earth_location(fits_path: Path | str) -> EarthLocation:
    """The array's reference position as an `EarthLocation` -- what sanity
    checks and observing-geometry code (hour angle, Az/El, parallactic
    angle) both need, built from the same `ARRAYX/Y/Z` this module already
    reads."""
    x, y, z = read_array_reference_position_m(fits_path)
    return EarthLocation.from_geocentric(x, y, z, unit=u.m)


def _array_reference_position_m(an_header) -> tuple[float, float, float]:
    return (
        float(an_header.get("ARRAYX", 0.0)),
        float(an_header.get("ARRAYY", 0.0)),
        float(an_header.get("ARRAYZ", 0.0)),
    )
