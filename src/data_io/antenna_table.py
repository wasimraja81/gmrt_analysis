"""Generic AIPS antenna-table (AN) reading: antenna positions and names, for
any random-groups UVFITS file following the AIPS convention -- not specific
to any one telescope.

`STABXYZ` is antenna position relative to the array's own reference position
(`ARRAYX`/`ARRAYY`/`ARRAYZ` in the same table's header) -- confirmed
directly against the real GWB file (2026-09-25): `STABXYZ` values are
metres-scale (tens to hundreds), `ARRAYX/Y/Z` is Earth-radius-scale
(~10^6 m), so `Antenna.x_m`/`y_m`/`z_m` here are already the absolute ECEF
position (`ARRAYX/Y/Z + STABXYZ`), not the raw relative offset.

`name` is read as whatever string the table holds, with no assumption about
its format -- GMRT's own "<code>:<station number>" naming convention (and
the DUD-antenna prefix matching built on it) is a telescope-specific
interpretation of that string, not a fact about the AIPS format, and lives
in `instruments/gmrt/antenna_table.py` instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from data_io.raw_data_access import open_fits_readonly


@dataclass(frozen=True)
class Antenna:
    station_number: int  # AIPS NOSTA, 1-indexed
    name: str  # ANNAME, exactly as stored
    x_m: float  # absolute ECEF X, metres
    y_m: float  # absolute ECEF Y, metres
    z_m: float  # absolute ECEF Z, metres


def read_antenna_table(fits_path: Path | str) -> list[Antenna]:
    """Read every antenna in the AIPS AN table, in table order, with absolute
    ECEF positions."""
    with open_fits_readonly(fits_path) as hdul:
        an = hdul["AIPS AN"]
        array_x, array_y, array_z = _array_reference_position_m(an.header)
        return [
            Antenna(
                station_number=int(row["NOSTA"]),
                name=str(row["ANNAME"]).strip(),
                x_m=array_x + float(row["STABXYZ"][0]),
                y_m=array_y + float(row["STABXYZ"][1]),
                z_m=array_z + float(row["STABXYZ"][2]),
            )
            for row in an.data
        ]


def read_array_reference_position_m(fits_path: Path | str) -> tuple[float, float, float]:
    """The array's own reference position (absolute ECEF, metres) -- needed
    directly by observing-geometry code (hour angle, Az/El, parallactic
    angle), separately from any one antenna's position."""
    with open_fits_readonly(fits_path) as hdul:
        return _array_reference_position_m(hdul["AIPS AN"].header)


def _array_reference_position_m(an_header) -> tuple[float, float, float]:
    return (
        float(an_header.get("ARRAYX", 0.0)),
        float(an_header.get("ARRAYY", 0.0)),
        float(an_header.get("ARRAYZ", 0.0)),
    )
