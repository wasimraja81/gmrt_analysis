"""Structural sanity checks on a GMRT observation.

Independent, single-purpose checks -- each answers one narrow question
("is this really a GMRT file", "is the antenna count plausible", "does the
row count match what the active-antenna set predicts") -- rather than one
monolithic validator, so a caller can run only the ones relevant to what
it's about to do. See docs/dev/GWB_PIPELINE_REFACTOR_PLAN.md, T5c.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from data_io.antenna_table import Antenna, read_array_earth_location
from data_io.known_observatory_locations import KNOWN_OBSERVATORY_LOCATIONS
from data_io.raw_data_access import open_fits_readonly
from data_io.row_index import RowIndex

# The AIPS extended baseline encoding's own ceiling (see decode_baseline in
# data_io/uvfits_group_params.py) -- an antenna count above this could never
# be represented by this file format at all, so a count above it means
# something failed to parse, not a real antenna table. Not a GMRT-specific
# number: it's the format's own limit.
MAX_REPRESENTABLE_ANTENNA_COUNT = 2047


class GmrtValidationError(Exception):
    """A structural sanity check on a GMRT observation failed."""


def check_telescope_is_gmrt(fits_path: Path | str) -> None:
    """Raise if the primary header's TELESCOP isn't 'GMRT' -- catches
    pointing the pipeline at the wrong file immediately, not three stages
    downstream. Cheap: reads the header only.
    """
    with open_fits_readonly(fits_path) as hdul:
        telescop = str(hdul[0].header.get("TELESCOP", "")).strip()
    if telescop != "GMRT":
        raise GmrtValidationError(
            f"{fits_path}: TELESCOP={telescop!r}, expected 'GMRT' -- this doesn't look "
            f"like a GMRT observation."
        )


def check_array_position_matches_known_location(fits_path: Path | str, tolerance_deg: float = 0.5) -> None:
    """Raise if this file's own recorded array position (`ARRAYX/Y/Z`) isn't
    close to GMRT's known, published location.

    `tolerance_deg` is generous on purpose (~55 km at this latitude) -- this
    is a sanity check for a corrupted or wrong-telescope array position, not
    a precise geodesy check; `check_telescope_is_gmrt` already catches an
    honestly-mislabeled file, this catches an internally inconsistent one
    (TELESCOP says GMRT, but the recorded position doesn't match).
    """
    location = read_array_earth_location(fits_path)
    known = KNOWN_OBSERVATORY_LOCATIONS["GMRT"]

    lat_diff = abs(location.lat.deg - known.latitude_deg)
    lon_diff = abs(location.lon.deg - known.longitude_deg)
    if lat_diff > tolerance_deg or lon_diff > tolerance_deg:
        raise GmrtValidationError(
            f"{fits_path}: array position resolves to lat={location.lat.deg:.3f}, "
            f"lon={location.lon.deg:.3f}, more than {tolerance_deg} deg from GMRT's known "
            f"location (lat={known.latitude_deg}, lon={known.longitude_deg}) -- "
            f"ARRAYX/Y/Z may be corrupted, or this isn't actually a GMRT file."
        )


def check_antenna_count_is_plausible(antennas: list[Antenna]) -> None:
    """Raise if the antenna table is empty, or has more entries than the
    baseline encoding could ever represent -- either means a parsing
    problem, not a real antenna table.
    """
    if len(antennas) == 0:
        raise GmrtValidationError("antenna table has zero entries")
    if len(antennas) > MAX_REPRESENTABLE_ANTENNA_COUNT:
        raise GmrtValidationError(
            f"antenna table has {len(antennas)} entries, more than the baseline encoding "
            f"can represent ({MAX_REPRESENTABLE_ANTENNA_COUNT}) -- likely a parsing problem."
        )


@dataclass(frozen=True)
class RowCountConsistencyReport:
    n_integrations: int
    uniform: bool  # do all integrations have the same row count?
    common_row_count: int | None  # the shared count, if uniform; else None
    n_active_antennas: int
    autocorrelations_present: bool  # anywhere in the whole file
    expected_row_count: int  # cross-baselines, plus n_active if autos are present
    matches_expected: bool  # only meaningful when uniform


def check_row_count_consistency(index: RowIndex, active_antennas: list[Antenna]) -> RowCountConsistencyReport:
    """Compare each integration's actual row count against what the active
    antenna set predicts.

    Does not raise on its own -- per-integration antenna participation is
    allowed to vary (standing rule 9), so non-uniformity is reported, not
    assumed to be an error. A caller (e.g. `build_gmrt_row_index`) decides
    what to do with a `matches_expected=False` result.
    """
    counts = np.diff(index.integration_boundaries)
    n_integrations = len(counts)
    uniform = bool(np.all(counts == counts[0])) if n_integrations > 0 else True
    common_row_count = int(counts[0]) if uniform and n_integrations > 0 else None

    n_active = len(active_antennas)
    autocorrelations_present = bool(np.any(index.ant1 == index.ant2))
    expected = n_active * (n_active - 1) // 2 + (n_active if autocorrelations_present else 0)

    return RowCountConsistencyReport(
        n_integrations=n_integrations,
        uniform=uniform,
        common_row_count=common_row_count,
        n_active_antennas=n_active,
        autocorrelations_present=autocorrelations_present,
        expected_row_count=expected,
        matches_expected=(common_row_count == expected) if uniform else False,
    )
