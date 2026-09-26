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
from astropy.coordinates import EarthLocation

from data_io.astrometry import altaz_deg, hour_angle_hours, parallactic_angle_deg
from data_io.row_index import RowIndex
from data_io.source_table import Source

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


def _cyclic_range_mask(values: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    """A range mask over a cyclic quantity already normalized to its own
    canonical range (hour angle to [-12, 12), azimuth to [0, 360), etc.) --
    `lo > hi` wraps through that range's own boundary rather than matching
    nothing."""
    lo, hi = bounds
    if lo <= hi:
        return (values >= lo) & (values <= hi)
    return (values >= lo) | (values <= hi)


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
    ha_range_hours: tuple[float, float] | None = None,
    az_range_deg: tuple[float, float] | None = None,
    el_range_deg: tuple[float, float] | None = None,
    parallactic_angle_range_deg: tuple[float, float] | None = None,
    source_table: dict[int, Source] | None = None,
    array_location: EarthLocation | None = None,
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

    `ha_range_hours`/`az_range_deg`/`el_range_deg`/`parallactic_angle_range_deg`
    filter by each row's own source, at that row's own JD, as seen from
    `array_location` -- computed via `data_io.astrometry`, not read from the
    file (UVFITS carries neither). Using any of them requires `source_table`
    (`data_io.source_table.read_source_table`'s result) and `array_location`
    (`data_io.antenna_table.read_array_earth_location`'s result); both are
    read once by the caller and passed in, keeping this function's own
    contract (operates on `RowIndex`'s in-memory arrays, no file I/O here)
    intact. `ha_range_hours`/`az_range_deg`/`parallactic_angle_range_deg` are
    all cyclic; a tuple where the low bound exceeds the high bound (e.g.
    `ha_range_hours=(10, -10)`, spanning lower culmination) wraps through
    the cycle's own boundary instead of matching nothing.

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

    geometry_filters = (ha_range_hours, az_range_deg, el_range_deg, parallactic_angle_range_deg)
    if any(f is not None for f in geometry_filters):
        if source_table is None or array_location is None:
            raise ValueError(
                "ha_range_hours/az_range_deg/el_range_deg/parallactic_angle_range_deg "
                "require both source_table and array_location"
            )
        missing_ids = sorted(set(index.source_id.tolist()) - set(source_table.keys()))
        if missing_ids:
            raise ValueError(f"source_table is missing source id(s) {missing_ids} present in this index")

        unique_ids, inverse = np.unique(index.source_id, return_inverse=True)
        ra_deg = np.array([source_table[int(sid)].ra_apparent_deg for sid in unique_ids])[inverse]
        dec_deg = np.array([source_table[int(sid)].dec_apparent_deg for sid in unique_ids])[inverse]

        if ha_range_hours is not None:
            mask &= _cyclic_range_mask(hour_angle_hours(index.jd, ra_deg, array_location), ha_range_hours)
        if az_range_deg is not None or el_range_deg is not None:
            az_deg, el_deg = altaz_deg(index.jd, ra_deg, dec_deg, array_location)
            if az_range_deg is not None:
                mask &= _cyclic_range_mask(az_deg, az_range_deg)
            if el_range_deg is not None:
                lo, hi = el_range_deg
                mask &= (el_deg >= lo) & (el_deg <= hi)
        if parallactic_angle_range_deg is not None:
            pa_deg = parallactic_angle_deg(index.jd, ra_deg, dec_deg, array_location)
            mask &= _cyclic_range_mask(pa_deg, parallactic_angle_range_deg)

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
