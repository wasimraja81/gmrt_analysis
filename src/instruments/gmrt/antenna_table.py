"""GMRT-specific antenna-naming convention: DUD-antenna resolution.

GMRT's AIPS AN table names antennas as "<code>:<station number>", e.g.
"C00:01" -- the code identifies the physical antenna, the station number
is its AIPS-internal index (NOSTA). Antenna names to exclude are given as
bare codes, e.g. "C07", matched against the table by prefix. Reading the
antenna table itself (names, positions) is generic AIPS format knowledge,
not a GMRT fact -- see `data_io.antenna_table` for that; only the naming
convention and DUD list here are GMRT-specific (split 2026-09-25, at the
user's request: "what is generic should remain generic").

"DUD" here means specifically the two antennas with a permanent, structural
quirk in GMRT's own AN-table conventions (see `GMRT_STRUCTURAL_DUD_NAMES`)
-- not any antenna that happens to be inactive for a given observation.
Excluding them, once, before anything downstream computes an antenna count
or baseline count, is what keeps every later calculation (the solver,
flagging-percentage denominators, coverage statistics) consistent with
each other. See docs/dev/GWB_PIPELINE_REFACTOR_PLAN.md, standing rule 8
and T5b. An antenna that's merely dead for one particular observation --
different from a structural DUD, see `instruments.gmrt.row_index` for the
distinction -- is a separate, per-observation concern, not hardcoded here.
"""

from __future__ import annotations

from dataclasses import dataclass

from data_io.antenna_table import Antenna

# The two antennas GMRT's own AN-table conventions carry with permanently
# invalid/placeholder positions -- confirmed directly (2026-09-24): GSB's AN
# table lists them as placeholder entries "C07:31"/"S05:32" appended after
# the 30 real stations, with positions that aren't real geodetic coordinates;
# GWB's AN table omits them by name entirely. A structural fact about the
# antenna table format, true regardless of which observation is being read
# -- not something to pass in per run. Confirmed by the user (2026-09-24):
# not to be confused with an antenna that's merely dead for one observation
# (which has a valid, real position, and will have data again once
# repaired) -- see `instruments.gmrt.row_index.build_gmrt_row_index`'s
# separate `dead_this_observation_names` parameter for that.
GMRT_STRUCTURAL_DUD_NAMES = ["C07", "S05"]


@dataclass
class ActiveAntennaResolution:
    active_antennas: list[Antenna]  # antenna table minus DUDs, table order preserved
    dud_antennas: list[Antenna]  # the excluded ones
    unmatched_dud_names: list[str]  # configured DUD names that matched no antenna


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
