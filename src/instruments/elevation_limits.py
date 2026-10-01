"""Each telescope's elevation limits, by the FITS TELESCOP value: the
antennas' lowest and highest elevation, in degrees.

GMRT's are its hardware limits, from the user (2026-10-01): 15 degrees low,
110 high (the elevation axis travels past the zenith). A plot of elevation
warns of rows outside them (T47). The elevations it draws are computed from
each source's position and the row's time, so they lie between -90 and 90
degrees and only the low limit can mark one; these files carry no antenna
pointing.
"""

from __future__ import annotations

ELEVATION_LIMITS_DEG = {
    "GMRT": (15.0, 110.0),
}
HORIZON_DEG = (0.0, 90.0)  # a telescope whose limits are not in the table: the horizon, the zenith


def known_elevation_limits_deg(telescop: str | None) -> tuple[float, float] | None:
    """(low, high) elevation limits in degrees of the telescope named by
    TELESCOP, or None for one not in the table."""
    return ELEVATION_LIMITS_DEG.get((telescop or "").strip().upper())


def elevation_limits_deg(telescop: str | None) -> tuple[float, float]:
    """The telescope's known limits, else the horizon and the zenith."""
    return known_elevation_limits_deg(telescop) or HORIZON_DEG
