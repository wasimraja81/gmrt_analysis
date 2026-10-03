"""Generic AIPS frequency-table (FQ) reading (T20): a file's frequency
setups, for any random-groups UVFITS file following the AIPS convention.

Per AIPS Memo 117: `IF FREQ` is each IF's frequency offset (Hz) from the
reference frequency (the FREQ axis's CRVAL), `CH WIDTH` the separation of
its channels (Hz), `TOTAL BANDWIDTH` its width, and `SIDEBAND` -1 for lower
and +1 for upper sideband, the frequency's increment per channel being CH
WIDTH times SIDEBAND. Each row is one setup, numbered by `FRQSEL`, which
each visibility row names in its own FREQSEL random parameter.

The rest of this codebase takes channel frequencies from the primary
header's FREQ axis alone, right for a file of one setup (both archival 40_014
files: the GWB file's one row has CH WIDTH -97656.25 Hz, its header's CDELT
the same).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from data_io.raw_data_access import open_fits_readonly


@dataclass(frozen=True)
class FrequencySetup:
    id: int  # FRQSEL
    if_offset_hz: tuple[float, ...]  # IF FREQ, one per IF
    channel_width_hz: tuple[float, ...]  # CH WIDTH
    total_bandwidth_hz: tuple[float, ...]  # TOTAL BANDWIDTH
    sideband: tuple[int, ...]  # SIDEBAND: -1 lower, +1 upper


def _per_if(value) -> tuple:
    return tuple(np.atleast_1d(np.asarray(value)).tolist())


def read_frequency_setups(fits_path: Path | str) -> list[FrequencySetup]:
    """Every row of the AIPS FQ table, in table order; [] for a file
    without one."""
    with open_fits_readonly(fits_path) as hdul:
        try:
            fq = hdul["AIPS FQ"]
        except KeyError:
            return []
        cols = set(fq.columns.names)
        setups = []
        for i, row in enumerate(fq.data):
            setups.append(FrequencySetup(
                id=int(row["FRQSEL"]) if "FRQSEL" in cols else i + 1,
                if_offset_hz=tuple(float(v) for v in _per_if(row["IF FREQ"])) if "IF FREQ" in cols else (),
                channel_width_hz=tuple(float(v) for v in _per_if(row["CH WIDTH"])) if "CH WIDTH" in cols else (),
                total_bandwidth_hz=(tuple(float(v) for v in _per_if(row["TOTAL BANDWIDTH"]))
                                    if "TOTAL BANDWIDTH" in cols else ()),
                sideband=tuple(int(v) for v in _per_if(row["SIDEBAND"])) if "SIDEBAND" in cols else (),
            ))
        return setups
