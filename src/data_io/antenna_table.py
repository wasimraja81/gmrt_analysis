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


# AN table MNTSTA codes, per AIPS Memo 117.
MOUNT_TYPES = {0: "alt-azimuth", 1: "equatorial", 2: "orbiting", 3: "X-Y", 4: "Naismith (right-handed)",
               5: "Naismith (left-handed)"}


@dataclass(frozen=True)
class AntennaFeeds:
    """An antenna's mount and feeds, from the AN table (T20; AIPS Memo 117):
    `mount_type` MNTSTA (`MOUNT_TYPES`), `axis_offset_m` STAXOF, and for
    feeds A and B their type (POLTYA/POLTYB: 'R', 'L', 'X' or 'Y') and
    position angle (POLAA/POLAB, degrees)."""

    station_number: int
    mount_type: int | None
    axis_offset_m: float | None
    pol_type_a: str
    pol_angle_a_deg: float | None
    pol_type_b: str
    pol_angle_b_deg: float | None


def read_antenna_feeds(fits_path: Path | str) -> dict[int, AntennaFeeds]:
    """Every antenna's mount and feeds, keyed by station number; a column the
    table lacks gives None (or "" for a feed type)."""
    with open_fits_readonly(fits_path) as hdul:
        an = hdul["AIPS AN"]
        cols = set(an.columns.names)

        def value(row, column, kind):
            return kind(row[column]) if column in cols else None

        feeds = {}
        for row in an.data:
            station = int(row["NOSTA"])
            feeds[station] = AntennaFeeds(
                station_number=station, mount_type=value(row, "MNTSTA", int), axis_offset_m=value(row, "STAXOF", float),
                pol_type_a=str(row["POLTYA"]).strip() if "POLTYA" in cols else "",
                pol_angle_a_deg=value(row, "POLAA", float),
                pol_type_b=str(row["POLTYB"]).strip() if "POLTYB" in cols else "",
                pol_angle_b_deg=value(row, "POLAB", float),
            )
        return feeds


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


@dataclass(frozen=True)
class TimeReference:
    """The file's time keywords, None where absent: `reference_date` (AN
    `RDATE`, else the primary header's `DATE-OBS`, as "YYYY-MM-DD"), the date
    AIPS counts days from; `time_system` (AN `TIMSYS`, "IAT" or "UTC");
    `iat_minus_utc_s` (AN `IATUTC`) and `data_minus_utc_s` (AN `DATUTC`).

    AIPS Memo 117 defines DATUTC as the data's time system minus UTC and
    IATUTC as IAT - UTC on RDATE. GMRT GWB files write TIMSYS = 'IAT' with
    DATUTC = 0 and IATUTC = 35 (the leap-second count of 2012-2015; 37 s on
    the 2021 file's date), so the keywords disagree with each other; their
    u, v, w put the timestamps 34.07 s after UTC (`data_io.timestamp_check`,
    2026-09-28). This project's rule (the user's decision, 2026-09-28):
    UTC = recorded time - IATUTC when TIMSYS is 'IAT', recorded time
    otherwise, with that check warning when the u, v, w disagree."""

    reference_date: str | None
    time_system: str | None
    iat_minus_utc_s: float | None = None
    data_minus_utc_s: float | None = None

    @property
    def recorded_minus_utc_s(self) -> float:
        """Seconds to subtract from a recorded time to get UTC (see the
        class docstring for the rule). Raises ValueError if TIMSYS is 'IAT'
        without IATUTC."""
        if (self.time_system or "").upper() == "IAT":
            if self.iat_minus_utc_s is None:
                raise ValueError("TIMSYS is 'IAT' but the antenna table has no IATUTC: UTC cannot be derived")
            return self.iat_minus_utc_s
        return 0.0

    def describe(self) -> str:
        """The keywords and the rule's result, for a log or terminal line."""
        system = self.time_system or "not declared"
        keywords = f"TIMSYS {system}, IATUTC {self.iat_minus_utc_s}, DATUTC {self.data_minus_utc_s}"
        offset = self.recorded_minus_utc_s
        rule = f"UTC = recorded - {offset:g} s (IATUTC)" if offset else "UTC = recorded"
        return f"{keywords}: {rule}"


def read_time_reference(fits_path: Path | str) -> TimeReference:
    with open_fits_readonly(fits_path) as hdul:
        date_obs = str(hdul[0].header.get("DATE-OBS", "")).strip()
        an_header = hdul["AIPS AN"].header if "AIPS AN" in hdul else {}
        rdate = str(an_header.get("RDATE", "")).strip()
        time_system = str(an_header.get("TIMSYS", "")).strip()
        iatutc = an_header.get("IATUTC")
        datutc = an_header.get("DATUTC")
    date = (rdate or date_obs)[:10]
    return TimeReference(
        reference_date=date or None,
        time_system=time_system or None,
        iat_minus_utc_s=float(iatutc) if iatutc is not None else None,
        data_minus_utc_s=float(datutc) if datutc is not None else None,
    )
