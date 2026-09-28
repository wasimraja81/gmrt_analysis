"""The local time zone of each observatory, by the FITS TELESCOP value.

UVFITS files carry no time zone, so local observatory time needs it from
here (or from the user). Zones are IANA names, so `zoneinfo` gives each
date's UTC offset, daylight saving included where the zone has it.
"""

from __future__ import annotations

OBSERVATORY_TIME_ZONES = {
    "GMRT": "Asia/Kolkata",  # IST, UTC+05:30, no daylight saving
}


def observatory_time_zone(telescop: str | None) -> str | None:
    """The IANA time zone of the observatory named by TELESCOP, or None if
    it is not in the table."""
    return OBSERVATORY_TIME_ZONES.get((telescop or "").strip().upper())
