"""Selecting which rows of a UVFITS file to load, from an already-built row index.

Deliberately separate from actually reading visibility data: everything
here operates on the small, in-memory arrays `RowIndex` already holds
(`source_id`, `ant1`/`ant2`, `jd`, `uu_sec`/`vv_sec`), so a selection is
cheap to compute -- no raw-file I/O at all. The result is a set of
absolute row indices; a separate reader (see `read_visibility_data`) is
what actually touches the file, for exactly those rows.

Telescope-agnostic. "Source" is one optional filter among several, not a
required entry point -- `sources=None` selects from every source in the
file, subject to whatever other criteria are given.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from data_io.row_index import RowIndex

SPEED_OF_LIGHT_M_PER_S = 299_792_458.0

CorrelationType = Literal["cross", "auto", "both"]


@dataclass(frozen=True)
class RowSelection:
    row_indices: np.ndarray  # absolute row indices into the file, sorted ascending
    n_rows: int
    sources: dict[int, str]  # source id -> name, for sources actually present in the selection


def _resolve_source_ids(index: RowIndex, sources) -> list[int]:
    if sources is None:
        return []
    if isinstance(sources, (str, int)):
        sources = [sources]

    wanted: list[int] = []
    for s in sources:
        if isinstance(s, str):
            matches = [sid for sid, name in index.id_to_name.items() if name.strip().lower() == s.strip().lower()]
            if not matches:
                raise ValueError(f'source "{s}" not found; available: {sorted(index.id_to_name.values())}')
            wanted.extend(matches)
        else:
            wanted.append(int(s))
    return wanted


def _normalized_pair_keys(pairs) -> set[int]:
    keys = set()
    for a, b in pairs:
        lo, hi = (a, b) if a <= b else (b, a)
        keys.add(int(lo) * 100_000 + int(hi))
    return keys


def select_rows(
    index: RowIndex,
    sources: str | int | list[str | int] | None = None,
    correlation_type: CorrelationType = "cross",
    integration_range: tuple[int, int] | None = None,
    jd_range: tuple[float, float] | None = None,
    antennas: list[int] | None = None,
    exclude_antennas: list[int] | None = None,
    baselines: list[tuple[int, int]] | None = None,
    exclude_baselines: list[tuple[int, int]] | None = None,
    uvdist_range_m: tuple[float, float] | None = None,
    every_nth: int | None = None,
    random_subset_n: int | None = None,
    random_seed: int | None = None,
) -> RowSelection:
    """Select rows by identity-based criteria, never by assumed position (standing rule 9).

    `sources=None` (the default) applies no source filter at all -- every
    source in the file is a candidate, subject to whatever other criteria
    are given. Pass a name, an id, or a list of either to filter.

    `correlation_type` defaults to "cross" (excluding autocorrelations),
    matching the pipeline-wide default; pass "auto" or "both" explicitly
    to include them.

    No silent downsampling: `every_nth` and `random_subset_n` are the only
    ways to reduce the result below what the filters select, both explicit
    and named, never a default. `random_subset_n` requires `random_seed`
    (reproducibility, standing rule 4) -- there is no seed-less path.
    """
    if every_nth is not None and random_subset_n is not None:
        raise ValueError("pass only one of every_nth, random_subset_n, not both")
    if random_subset_n is not None and random_seed is None:
        raise ValueError("random_subset_n requires an explicit random_seed, for reproducibility")

    mask = np.ones(index.gcount, dtype=bool)

    wanted_source_ids = _resolve_source_ids(index, sources)
    if wanted_source_ids:
        mask &= np.isin(index.source_id, wanted_source_ids)

    if correlation_type == "cross":
        mask &= index.ant1 != index.ant2
    elif correlation_type == "auto":
        mask &= index.ant1 == index.ant2
    elif correlation_type != "both":
        raise ValueError(f'correlation_type must be "cross", "auto", or "both", got {correlation_type!r}')

    if integration_range is not None:
        first, last = integration_range
        n_integrations = len(index.integration_boundaries) - 1
        if not (0 <= first <= last < n_integrations):
            raise ValueError(f"integration_range {integration_range} out of bounds for {n_integrations} integrations")
        integ_mask = np.zeros(index.gcount, dtype=bool)
        integ_mask[index.integration_boundaries[first] : index.integration_boundaries[last + 1]] = True
        mask &= integ_mask

    if jd_range is not None:
        lo, hi = jd_range
        mask &= (index.jd >= lo) & (index.jd <= hi)

    if antennas is not None:
        mask &= np.isin(index.ant1, antennas) & np.isin(index.ant2, antennas)

    if exclude_antennas is not None:
        mask &= ~(np.isin(index.ant1, exclude_antennas) | np.isin(index.ant2, exclude_antennas))

    if baselines is not None or exclude_baselines is not None:
        row_lo = np.minimum(index.ant1, index.ant2).astype(np.int64)
        row_hi = np.maximum(index.ant1, index.ant2).astype(np.int64)
        row_keys = row_lo * 100_000 + row_hi
        if baselines is not None:
            mask &= np.isin(row_keys, list(_normalized_pair_keys(baselines)))
        if exclude_baselines is not None:
            mask &= ~np.isin(row_keys, list(_normalized_pair_keys(exclude_baselines)))

    if uvdist_range_m is not None:
        lo, hi = uvdist_range_m
        uvdist_m = np.sqrt(index.uu_sec.astype(np.float64) ** 2 + index.vv_sec.astype(np.float64) ** 2) * SPEED_OF_LIGHT_M_PER_S
        mask &= (uvdist_m >= lo) & (uvdist_m <= hi)

    row_indices = np.where(mask)[0]

    if every_nth is not None:
        row_indices = row_indices[::every_nth]
    elif random_subset_n is not None:
        rng = np.random.default_rng(random_seed)
        if random_subset_n < len(row_indices):
            row_indices = np.sort(rng.choice(row_indices, size=random_subset_n, replace=False))

    present_source_ids = np.unique(index.source_id[row_indices]).tolist()
    sources_present = {sid: index.id_to_name[sid] for sid in present_source_ids if sid in index.id_to_name}

    return RowSelection(row_indices=row_indices, n_rows=len(row_indices), sources=sources_present)
