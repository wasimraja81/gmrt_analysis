"""Resolving an antenna range spec into concrete station numbers -- a
comma-separated mix of numeric ids/ranges ("1:5,10") and antenna names
("W01:25"), matched against a specific file's own AN table. Mirrors how
`data_io.row_selection.select_rows`'s `sources` parameter already accepts a
name or an id for the same reason: a station number alone means nothing to
someone looking at the file, but the name printed right next to it does.
"""

from __future__ import annotations

from data_io.antenna_table import Antenna
from visplot.range_spec import expand_int_ranges


def resolve_antenna_selection(spec: str, antennas: list[Antenna]) -> list[int]:
    """Station numbers `spec` selects. Each comma-separated term is tried
    as an antenna name first (case-insensitive, exact match against
    `antennas`' own `name`), then as a numeric id or "lo:hi" range -- a
    name is checked first because a real antenna name (e.g. "C00:01")
    contains a colon too, and would otherwise be misread as a numeric
    range. Name *ranges* ("C00:C05") are not supported, matching how
    `select_rows`'s own `sources` parameter also only matches names
    exactly, not by range.

    A numeric range expands to every station number in that range, not
    only ones present in `antennas` -- a caller that only wants antennas
    actually in this file can intersect the result with
    `{a.station_number for a in antennas}`."""
    name_to_station = {a.name.strip().lower(): a.station_number for a in antennas}
    station_numbers: set[int] = set()
    for term in (t.strip() for t in spec.split(",") if t.strip()):
        if term.lower() in name_to_station:
            station_numbers.add(name_to_station[term.lower()])
            continue
        try:
            station_numbers.update(expand_int_ranges(term))
        except ValueError:
            raise ValueError(
                f"{term!r} is neither a known antenna name nor a valid numeric id/range; "
                f"known names: {sorted(a.name for a in antennas)}"
            ) from None
    return sorted(station_numbers)
