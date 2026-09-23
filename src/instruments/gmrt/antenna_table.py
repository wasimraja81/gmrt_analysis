"""GMRT antenna table reading and DUD-antenna resolution.

GMRT's AIPS AN table names antennas as "<code>:<station number>", e.g.
"C00:01" -- the code identifies the physical antenna, the station number
is its AIPS-internal index (NOSTA). DUD-antenna configuration (which
antennas were hardware-dead, never correlating, for a given observation)
is given as bare codes, e.g. "C07", matched against the table by prefix.

DUD antennas are not a data-quality/flagging concern -- they contribute
zero rows to the raw file at all. Excluding them here, once, before
anything downstream computes an antenna count or baseline count, is what
keeps every later calculation (the solver, flagging-percentage
denominators, coverage statistics) consistent with each other. See
docs/dev/GWB_PIPELINE_REFACTOR_PLAN.md, standing rule 8 and T5b.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from data_io.raw_data_access import open_fits_readonly


@dataclass(frozen=True)
class Antenna:
    station_number: int  # AIPS NOSTA, 1-indexed
    name: str  # e.g. "C00:01"


@dataclass
class ActiveAntennaResolution:
    active_antennas: list[Antenna]  # antenna table minus DUDs, table order preserved
    dud_antennas: list[Antenna]  # the excluded ones
    unmatched_dud_names: list[str]  # configured DUD names that matched no antenna


def read_antenna_table(fits_path: Path | str) -> list[Antenna]:
    """Read every antenna in the AIPS AN table, in table order."""
    with open_fits_readonly(fits_path) as hdul:
        an = hdul["AIPS AN"]
        return [
            Antenna(station_number=int(row["NOSTA"]), name=str(row["ANNAME"]).strip())
            for row in an.data
        ]


def resolve_active_antennas(antennas: list[Antenna], dud_names: list[str]) -> ActiveAntennaResolution:
    """Split an antenna table into active vs. DUD, by configured name.

    Matching: a configured name matches a table entry if it equals the
    entry's name exactly, or if the entry's name starts with
    "<configured_name>:" -- the GMRT AN-table convention of code, colon,
    station number (e.g. "C07" matches "C07:08"). Deliberately narrower
    than the archived GSB engine's own matcher, which also accepted a bare
    prefix with no colon required -- that's loose enough to match more
    than one antenna for a short configured name, silently over-excluding
    them. A configured name that doesn't match anything is reported in
    `unmatched_dud_names` rather than silently ignored, so a typo in
    config surfaces immediately instead of quietly leaving a dead antenna
    in the active set.
    """
    dud_station_numbers: set[int] = set()
    dud_antennas: list[Antenna] = []
    unmatched: list[str] = []

    for wanted in dud_names:
        matches = [a for a in antennas if a.name == wanted or a.name.startswith(f"{wanted}:")]
        if not matches:
            unmatched.append(wanted)
            continue
        for match in matches:
            if match.station_number not in dud_station_numbers:
                dud_station_numbers.add(match.station_number)
                dud_antennas.append(match)

    active_antennas = [a for a in antennas if a.station_number not in dud_station_numbers]

    return ActiveAntennaResolution(
        active_antennas=active_antennas,
        dud_antennas=dud_antennas,
        unmatched_dud_names=unmatched,
    )
