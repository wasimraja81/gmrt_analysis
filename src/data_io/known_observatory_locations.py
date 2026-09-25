"""Reference observatory locations, for sanity-checking a file's own recorded
array position -- not a source of truth anything computes from. Every real
geometry computation (hour angle, Az/El, parallactic angle) uses the
position recorded in the file itself, via
`antenna_table.read_array_reference_position_m`. This table exists only to
catch a corrupted, wrong, or unexpectedly different array position in a
file that claims to be from a known observatory -- values here are
approximate (public, published figures), with a generous tolerance expected
of anything checking against them, not treated as precise geodesy.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KnownObservatoryLocation:
    name: str
    latitude_deg: float
    longitude_deg: float
    height_m: float


# GMRT: ~19.093 N, 74.050 E, near Khodad, Maharashtra, India -- confirmed
# directly (2026-09-25) by converting the real GWB/GSB files' own ARRAYX/Y/Z
# to geodetic coordinates and checking the result against GMRT's published
# location; the two agree.
KNOWN_OBSERVATORY_LOCATIONS: dict[str, KnownObservatoryLocation] = {
    "GMRT": KnownObservatoryLocation(name="GMRT", latitude_deg=19.093, longitude_deg=74.050, height_m=640.0),
}
