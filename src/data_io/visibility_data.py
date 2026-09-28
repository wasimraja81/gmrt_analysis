"""Reading visibility data for a row selection.

Layer 2 of the selection/read split: `select_rows` (row_selection.py) decides
*which* rows, cheaply, from the row index alone; this module reads the
actual visibility bytes for exactly those rows -- the only part of this
pair that touches the (large) data portion of the file.

Telescope-agnostic, like `row_index.py`. Every axis is found and selected
by its CTYPE label, never assumed position or assumed trivial (length 1).
COMPLEX is the one axis handled specially, because the FITS convention
itself defines it as (real, imaginary, weight) -- every other axis
(STOKES, FREQ, IF, RA, DEC, or anything else a file declares) goes through
the same generic selection mechanism, defaulting to "select everything"
when not named. A file with more than one IF, or a genuine multi-pointing
RA/DEC axis, is read correctly by this same code -- nothing here assumes
those axes are trivial the way GWB's happen to be.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from data_io.raw_data_access import open_raw_memmap
from data_io.row_index import RowIndex, find_axis
from data_io.uvfits_group_params import DEFAULT_RAM_FRACTION_TO_USE, host_total_memory_bytes


@dataclass(frozen=True)
class VisibilityBlock:
    row_indices: np.ndarray  # absolute row indices, sorted ascending
    # complex128, shape (n_rows, *one length per axis in axis_types order); None
    # for a metadata-only chunk (iter_visibility_chunks(read_data=False))
    data: np.ndarray | None
    weight: np.ndarray | None  # float64, same shape as data -- AIPS convention: <=0 means flagged
    axis_types: list[str]  # CTYPE label for each of data.shape[1:], in order
    axis_indices: dict[str, np.ndarray]  # axis type -> the pixel indices selected along it
    ant1: np.ndarray
    ant2: np.ndarray
    source_id: np.ndarray
    jd: np.ndarray
    uu_sec: np.ndarray
    vv_sec: np.ndarray
    ww_sec: np.ndarray
    chan_freqs_hz: np.ndarray | None  # physical frequencies for the selected FREQ indices, if a FREQ axis exists
    stokes_labels: list[str] | None  # physical Stokes labels for the selected STOKES indices, if a STOKES axis exists


def _contiguous_runs(sorted_unique_indices: np.ndarray) -> list[tuple[int, int]]:
    """Group sorted, unique row indices into contiguous half-open [start, stop) runs."""
    if len(sorted_unique_indices) == 0:
        return []
    breaks = np.where(np.diff(sorted_unique_indices) > 1)[0] + 1
    run_start_positions = np.concatenate([[0], breaks])
    run_stop_positions = np.concatenate([breaks, [len(sorted_unique_indices)]])
    return [
        (int(sorted_unique_indices[s]), int(sorted_unique_indices[e - 1]) + 1)
        for s, e in zip(run_start_positions, run_stop_positions)
    ]


@dataclass(frozen=True)
class _ReadPlan:
    """What every chunk read needs, worked out once from the index and the
    axis selection."""
    axis_types: list[str]
    selected_indices: list[np.ndarray]
    selected_shape: tuple[int, ...]
    row_bytes: int
    data_floats_per_group: int
    reshape_dims: list[int]
    src_positions: list[int]


def _read_plan(index: RowIndex, axis_selection: dict[str, np.ndarray] | None) -> _ReadPlan:
    axis_selection = axis_selection or {}
    complex_axis = find_axis(index.data_axis_types, "COMPLEX")
    if complex_axis is None:
        raise ValueError(f"file has no COMPLEX axis: {index.data_axis_types}")
    if index.data_axis_lengths[complex_axis] != 3:
        raise NotImplementedError(
            f"expected COMPLEX axis of length 3 (real, imag, weight), "
            f"got {index.data_axis_lengths[complex_axis]}"
        )

    n_axes = len(index.data_axis_lengths)
    other_axis_nums = [i for i in range(n_axes) if i != complex_axis]
    axis_types = [index.data_axis_types[i] for i in other_axis_nums]
    axis_full_lengths = [index.data_axis_lengths[i] for i in other_axis_nums]

    unknown_keys = set(axis_selection) - set(axis_types)
    if unknown_keys:
        raise ValueError(f"axis_selection names {sorted(unknown_keys)} not present in this file's axes {axis_types}")

    selected_indices = [
        np.asarray(axis_selection[ctype]) if ctype in axis_selection else np.arange(full_len)
        for ctype, full_len in zip(axis_types, axis_full_lengths)
    ]
    data_floats_per_group = int(np.prod(index.data_axis_lengths))

    # Reshape order matches numpy's natural order for a GroupsHDU: dims run from the
    # highest axis number (slowest) to axis 2 (fastest) -- confirmed directly against
    # astropy's own construction (row_index.py's module docstring / _read_data_axes).
    def _position_in_reshaped(axis_num: int) -> int:
        return 1 + (n_axes - 1 - axis_num)  # +1 for the leading row axis

    return _ReadPlan(
        axis_types=axis_types,
        selected_indices=selected_indices,
        selected_shape=tuple(len(idx) for idx in selected_indices),
        row_bytes=(index.pcount + data_floats_per_group) * 4,
        data_floats_per_group=data_floats_per_group,
        reshape_dims=list(reversed(index.data_axis_lengths)),
        src_positions=[_position_in_reshaped(complex_axis)] + [_position_in_reshaped(i) for i in other_axis_nums],
    )


def _read_run(fits_path, index: RowIndex, plan: _ReadPlan, start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
    """Data and weight for the contiguous rows [start, stop), axis selection applied."""
    n = stop - start
    raw = open_raw_memmap(
        fits_path,
        dtype=">f4",
        shape=(n, index.pcount + plan.data_floats_per_group),
        offset=index.data_offset + start * plan.row_bytes,
    )
    vis = np.array(raw[:, index.pcount :], dtype=np.float32)
    del raw
    vis = vis.reshape((n, *plan.reshape_dims))

    # Move COMPLEX to position 1 and every other axis to positions 2.. in
    # axis_types order -- generic for however many axes this file has.
    vis = np.moveaxis(vis, plan.src_positions, list(range(1, len(plan.src_positions) + 1)))
    for i, idx in enumerate(plan.selected_indices):
        vis = np.take(vis, idx, axis=2 + i)

    data = vis[:, 0].astype(np.float64) + 1j * vis[:, 1].astype(np.float64)
    weight = vis[:, 2].astype(np.float64)
    return data, weight


def _block(index: RowIndex, plan: _ReadPlan, row_indices: np.ndarray, data, weight) -> VisibilityBlock:
    axis_indices = dict(zip(plan.axis_types, plan.selected_indices))
    chan_freqs_hz = index.chan_freqs_hz[axis_indices["FREQ"]] if "FREQ" in axis_indices else None
    stokes_labels = (
        [index.stokes_labels[i] for i in axis_indices["STOKES"]] if "STOKES" in axis_indices else None
    )
    return VisibilityBlock(
        row_indices=row_indices,
        data=data,
        weight=weight,
        axis_types=plan.axis_types,
        axis_indices=axis_indices,
        ant1=index.ant1[row_indices],
        ant2=index.ant2[row_indices],
        source_id=index.source_id[row_indices],
        jd=index.jd[row_indices],
        uu_sec=index.uu_sec[row_indices],
        vv_sec=index.vv_sec[row_indices],
        ww_sec=index.ww_sec[row_indices],
        chan_freqs_hz=chan_freqs_hz,
        stokes_labels=stokes_labels,
    )


def iter_visibility_chunks(
    fits_path: Path | str,
    index: RowIndex,
    row_indices: np.ndarray,
    axis_selection: dict[str, np.ndarray] | None = None,
    max_chunk_bytes: int | None = None,
    ram_fraction: float = DEFAULT_RAM_FRACTION_TO_USE,
    read_data: bool = True,
) -> Iterator[VisibilityBlock]:
    """Yield the visibility data for the given rows as `VisibilityBlock`s,
    in ascending row order, one chunk at a time.

    `axis_selection` maps a CTYPE name (e.g. "FREQ", "STOKES", "IF") to the
    pixel indices wanted along that axis; an axis not named is selected in
    full, whatever its length -- there is no assumption that any axis
    besides COMPLEX has length 1.

    Each chunk covers at most `max_chunk_bytes` of full rows on disk (every
    channel and Stokes, before `axis_selection` is applied), or `ram_fraction`
    of the host's total RAM if not given -- the same budget convention as
    `read_all_param_columns` (uvfits_group_params.py). Nothing is held
    between chunks, so memory is set by the chunk size, whatever the number
    of rows; a caller that wants every row at once passes a chunk size large
    enough for the whole selection, or concatenates the chunks.

    A chunk collects rows from one or more contiguous runs, so a sparse
    selection still yields full-sized chunks; each run is read sequentially.

    `read_data=False` yields the same chunks with `data` and `weight` set to
    None and nothing read from disk -- for work that needs only per-row
    metadata (time, u/v/w, source) and the axis selection."""
    row_indices = np.unique(np.asarray(row_indices, dtype=np.int64))
    plan = _read_plan(index, axis_selection)
    if max_chunk_bytes is None:
        max_chunk_bytes = int(host_total_memory_bytes() * ram_fraction)
    rows_per_chunk = max(1, max_chunk_bytes // plan.row_bytes)

    pending_rows, pending_data, pending_weight = [], [], []
    n_pending = 0
    for run_start, run_stop in _contiguous_runs(row_indices):
        start = run_start
        while start < run_stop:
            stop = min(run_stop, start + rows_per_chunk - n_pending)
            data, weight = _read_run(fits_path, index, plan, start, stop) if read_data else (None, None)
            pending_rows.append(np.arange(start, stop))
            pending_data.append(data)
            pending_weight.append(weight)
            n_pending += stop - start
            start = stop
            if n_pending == rows_per_chunk:
                yield _pending_block(index, plan, pending_rows, pending_data, pending_weight)
                pending_rows, pending_data, pending_weight = [], [], []
                n_pending = 0
    if n_pending:
        yield _pending_block(index, plan, pending_rows, pending_data, pending_weight)


def _pending_block(index, plan, pending_rows, pending_data, pending_weight) -> VisibilityBlock:
    rows = np.concatenate(pending_rows)
    if pending_data[0] is None:
        return _block(index, plan, rows, None, None)
    return _block(index, plan, rows, np.concatenate(pending_data), np.concatenate(pending_weight))

