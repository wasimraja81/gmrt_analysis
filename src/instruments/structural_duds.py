"""The structural DUD entries of each telescope's AIPS AN table, by the FITS
TELESCOP value: placeholder rows that are no antenna (GMRT's C07 and S05,
see `instruments.gmrt.antenna_table`). A reader of the file leaves them out
once, before anything counts, names or draws antennas (standing rule 8).
"""

from __future__ import annotations

from data_io.antenna_table import Antenna
from instruments.gmrt.antenna_table import GMRT_STRUCTURAL_DUD_NAMES, ActiveAntennaResolution, resolve_active_antennas

STRUCTURAL_DUD_NAMES = {
    "GMRT": GMRT_STRUCTURAL_DUD_NAMES,
}


def structural_dud_names(telescop: str | None) -> list[str]:
    """The structural DUD names of the telescope named by TELESCOP; none for
    a telescope not in the table."""
    return list(STRUCTURAL_DUD_NAMES.get((telescop or "").strip().upper(), []))


def without_structural_duds(antennas: list[Antenna], telescop: str | None) -> ActiveAntennaResolution:
    """`antennas` split into the antennas and the telescope's structural DUD
    entries (a DUD name the table lacks, e.g. GWB's, is in
    `unmatched_dud_names`)."""
    return resolve_active_antennas(antennas, structural_dud_names(telescop))
