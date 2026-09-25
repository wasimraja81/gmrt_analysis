"""GMRT-specific row-index orchestration: the generic index plus antenna resolution.

Combines two already-built, independently-tested pieces -- `data_io.row_index`'s
generic full-file scan and `antenna_table`'s antenna resolution -- and adds
the one genuinely new thing: figuring out, from the data itself, which
nominal antennas actually have rows.

That's not a hypothetical concern: for the 40_014_25JUL2021 GWB observation,
two antennas (`C03:04`, `C10:10`) have zero rows anywhere in the raw file --
deliberately omitted at the correlator to reduce data size, not flagged or
padded. An earlier version of this module required the caller to name them
up front (`dead_this_observation_names`), which just recreated the exact
stale-config risk that caused this project's own DUD list (`C07`/`S05`) to be
wrong for GWB in the first place. Corrected (2026-09-24, at the user's
prompting): once the row index is built, which antennas have data is already
known -- there's nothing to ask the caller to predict.

Two exclusion concepts, kept conceptually separate even though only one is
still a parameter:

- Structural DUDs (`antenna_table.GMRT_STRUCTURAL_DUD_NAMES`) -- a permanent
  fact about GMRT's AN-table conventions (invalid/placeholder positions),
  true for every observation regardless of which antennas have data, hence
  still hardcoded rather than derived.
- Antennas dead for this observation only -- valid AN-table positions, will
  have real data again once repaired, deliberately omitted from this file at
  the correlator. Derived automatically from the built index: whichever
  nominal, non-structural-DUD antennas have zero rows.

The returned `RowIndex` is the plain, unmodified object from `data_io.row_index`
-- every downstream consumer (`select_rows`, `read_visibility_data`) works on it
directly, with no GMRT-specific wrapper needed for selection or reading.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from data_io.row_index import RowIndex, build_row_index
from instruments.gmrt.antenna_table import (
    GMRT_STRUCTURAL_DUD_NAMES,
    Antenna,
    read_antenna_table,
    resolve_active_antennas,
)
from instruments.gmrt.sanity_checks import (
    RowCountConsistencyReport,
    check_antenna_count_is_plausible,
    check_row_count_consistency,
    check_telescope_is_gmrt,
)


class GmrtRowCountMismatch(Exception):
    """Every integration has the same row count, but it doesn't match what the
    active antenna set predicts -- a strong signal the antenna resolution
    itself is wrong, not just an oddity of this observation."""


@dataclass(frozen=True)
class GmrtAntennaResolution:
    nominal_antennas: list[Antenna]  # every antenna in the AN table
    active_antennas: list[Antenna]  # nominal minus structural DUDs minus dead-this-observation
    structural_dud_antennas: list[Antenna]  # matched permanent AN-table-quirk exclusions
    unmatched_structural_dud_names: list[str]  # structural DUD names matching nothing (e.g. GWB's table)
    dead_this_observation_antennas: list[Antenna]  # derived: nominal, non-structural-DUD, zero rows
    row_count_consistency: RowCountConsistencyReport  # actual vs. predicted rows/integration


def build_gmrt_row_index(
    fits_path: Path | str,
    structural_dud_names: list[str] | None = None,
    strict: bool = True,
    max_chunk_bytes: int | None = None,
    verbose: bool = False,
) -> tuple[RowIndex, GmrtAntennaResolution]:
    """Build the row index and resolve GMRT antenna state together.

    `structural_dud_names` defaults to `GMRT_STRUCTURAL_DUD_NAMES` (the
    permanent `C07`/`S05` AN-table quirk); overriding it is for tests, not
    normal use. Which antennas are merely dead for this one observation is
    not a parameter -- it's derived from the row index itself, once built.

    Runs every sanity check scoped for T5c: `TELESCOP == 'GMRT'` and antenna-
    count plausibility first (cheap, header-only, so a wrong file fails
    before the expensive full-file scan even starts), then, once the index
    is built, row-count consistency.

    `strict=True` (the default) raises `GmrtRowCountMismatch` if every
    integration has the same row count but it doesn't match what the
    resulting active-antenna set predicts. Pass `strict=False` to get the
    result anyway when you already know about a mismatch and need to carry
    on -- it's recorded in the returned `GmrtAntennaResolution` either way.
    """
    structural_dud_names = list(GMRT_STRUCTURAL_DUD_NAMES if structural_dud_names is None else structural_dud_names)

    check_telescope_is_gmrt(fits_path)
    antennas = read_antenna_table(fits_path)
    check_antenna_count_is_plausible(antennas)

    index = build_row_index(fits_path, max_chunk_bytes=max_chunk_bytes, verbose=verbose)

    structural = resolve_active_antennas(antennas, structural_dud_names)

    observed_stations = set(index.ant1.tolist()) | set(index.ant2.tolist())
    dead_this_observation = [a for a in structural.active_antennas if a.station_number not in observed_stations]
    active_antennas = [a for a in structural.active_antennas if a.station_number in observed_stations]

    row_count_report = check_row_count_consistency(index, active_antennas)

    result = GmrtAntennaResolution(
        nominal_antennas=antennas,
        active_antennas=active_antennas,
        structural_dud_antennas=structural.dud_antennas,
        unmatched_structural_dud_names=structural.unmatched_dud_names,
        dead_this_observation_antennas=dead_this_observation,
        row_count_consistency=row_count_report,
    )

    if row_count_report.uniform and not row_count_report.matches_expected and strict:
        raise GmrtRowCountMismatch(
            f"every integration has {row_count_report.common_row_count} rows, but "
            f"{row_count_report.n_active_antennas} active antennas "
            f"({'with' if row_count_report.autocorrelations_present else 'without'} autocorrelations) "
            f"predicts {row_count_report.expected_row_count}. Dead-this-observation antennas: "
            f"{[a.name for a in dead_this_observation]}. Pass strict=False to proceed anyway -- "
            f"recorded in GmrtAntennaResolution.row_count_consistency either way."
        )

    return index, result
