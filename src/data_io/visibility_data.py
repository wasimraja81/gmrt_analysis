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

# `iter_visibility_chunks(spread=True)` reorders units of consecutive integrations holding at least this many
# bytes of the selection's rows: one seek per unit, a few percent of its read at a spinning disk's ~10 ms seek and
# ~180 MB/s (a GWB integration, 378 rows, is about 37 MB; a GSB one about 2.3 MB).
SPREAD_UNIT_BYTES = 32 * 1024**2


@dataclass(frozen=True)
class VisibilityBlock:
    row_indices: np.ndarray  # absolute row indices, sorted ascending
    # complex64 (the file's float32 real/imag), shape (n_rows, *one length per axis in
    # axis_types order); None for a metadata-only chunk (iter_visibility_chunks(read_data=False))
    data: np.ndarray | None
    weight: np.ndarray | None  # float32, same shape as data -- AIPS convention: <=0 means flagged
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
    # per axis: a slice (a contiguous run of indices, so selecting is a view) or an index array
    selectors: list
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
        selectors=[_selector(idx) for idx in selected_indices],
        selected_shape=tuple(len(idx) for idx in selected_indices),
        row_bytes=(index.pcount + data_floats_per_group) * 4,
        data_floats_per_group=data_floats_per_group,
        reshape_dims=list(reversed(index.data_axis_lengths)),
        src_positions=[_position_in_reshaped(complex_axis)] + [_position_in_reshaped(i) for i in other_axis_nums],
    )


def _selector(idx: np.ndarray):
    """A slice for a contiguous ascending run of indices (selecting is then a
    view, no copy), otherwise the index array itself."""
    idx = np.asarray(idx)
    if idx.size and np.all(np.diff(idx) == 1):
        return slice(int(idx[0]), int(idx[-1]) + 1)
    return idx


def _read_run_into(fits_path, index: RowIndex, plan: _ReadPlan, start: int, stop: int,
                   data_out: np.ndarray, weight_out: np.ndarray) -> None:
    """Copy the contiguous rows [start, stop), axis selection applied, into
    `data_out` (complex64) and `weight_out` (float32). The file's rows are
    viewed in place, the selection taken as views where it is contiguous, and
    each value converted (big-endian to native) once, directly into the output."""
    n = stop - start
    raw = open_raw_memmap(
        fits_path,
        dtype=">f4",
        shape=(n, index.pcount + plan.data_floats_per_group),
        offset=index.data_offset + start * plan.row_bytes,
    )
    vis = raw[:, index.pcount :].reshape((n, *plan.reshape_dims))

    # Move COMPLEX to position 1 and every other axis to positions 2.. in
    # axis_types order -- generic for however many axes this file has.
    vis = np.moveaxis(vis, plan.src_positions, list(range(1, len(plan.src_positions) + 1)))
    for i, sel in enumerate(plan.selectors):
        if isinstance(sel, slice):
            vis = vis[(slice(None),) * (2 + i) + (sel,)]
        else:
            vis = np.take(vis, sel, axis=2 + i)

    data_out.real = vis[:, 0]
    data_out.imag = vis[:, 1]
    weight_out[...] = vis[:, 2]
    del vis, raw


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
    spread: bool = False,
) -> Iterator[VisibilityBlock]:
    """Yield the visibility data for the given rows as `VisibilityBlock`s,
    one chunk at a time: in ascending row order, or with `spread` in an order
    covering the time range early (each chunk's rows ascending).

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
    metadata (time, u/v/w, source) and the axis selection.

    `spread=True` yields the chunks in an order that covers the selection's
    time range early (for a window drawing as it reads): the selection is cut
    into units of consecutive integrations of at least SPREAD_UNIT_BYTES each
    (one sequential read), taken in bit-reversed order (`spread_order`); a
    chunk gathers units in that order and holds its rows ascending. Every row
    is still read once; only the chunks' order, and which rows share one,
    change."""
    row_indices = np.unique(np.asarray(row_indices, dtype=np.int64))
    plan = _read_plan(index, axis_selection)
    if max_chunk_bytes is None:
        max_chunk_bytes = int(host_total_memory_bytes() * ram_fraction)
    rows_per_chunk = max(1, max_chunk_bytes // plan.row_bytes)
    chunks = (_spread_chunk_pieces(index, row_indices, rows_per_chunk, plan.row_bytes) if spread
              else _chunk_pieces(row_indices, rows_per_chunk))

    for pieces in chunks:
        rows = np.concatenate([np.arange(a, b) for a, b in pieces])
        if not read_data:
            yield _block(index, plan, rows, None, None)
            continue
        data = np.empty((len(rows), *plan.selected_shape), dtype=np.complex64)
        weight = np.empty((len(rows), *plan.selected_shape), dtype=np.float32)
        pos = 0
        for a, b in pieces:
            _read_run_into(fits_path, index, plan, a, b, data[pos : pos + b - a], weight[pos : pos + b - a])
            pos += b - a
        yield _block(index, plan, rows, data, weight)


def _chunk_pieces(row_indices: np.ndarray, rows_per_chunk: int) -> list[list[tuple[int, int]]]:
    """The selection's contiguous runs, split and grouped into chunks of at
    most `rows_per_chunk` rows: each chunk a list of (start, stop) pieces."""
    chunks, current, n_current = [], [], 0
    for run_start, run_stop in _contiguous_runs(row_indices):
        start = run_start
        while start < run_stop:
            stop = min(run_stop, start + rows_per_chunk - n_current)
            current.append((start, stop))
            n_current += stop - start
            start = stop
            if n_current == rows_per_chunk:
                chunks.append(current)
                current, n_current = [], 0
    if current:
        chunks.append(current)
    return chunks


def spread_order(n: int) -> np.ndarray:
    """0 .. n-1 in bit-reversed order (0, n/2, n/4, 3n/4, ...), so that every
    prefix of it is spread evenly over the range."""
    if n <= 1:
        return np.arange(n)
    bits = int(np.ceil(np.log2(n)))
    i = np.arange(1 << bits)
    reversed_i = np.zeros_like(i)
    for b in range(bits):
        reversed_i |= ((i >> b) & 1) << (bits - 1 - b)
    return reversed_i[reversed_i < n]


def _spread_chunk_pieces(index: RowIndex, row_indices: np.ndarray, rows_per_chunk: int,
                         row_bytes: int) -> list[list[tuple[int, int]]]:
    """The chunks of `iter_visibility_chunks(spread=True)`: units of
    consecutive integrations, each holding at least SPREAD_UNIT_BYTES of the
    selection's rows, taken in `spread_order`, gathered into chunks of at
    most `rows_per_chunk` rows (a unit split across chunks where it does not
    fit), each chunk's rows ascending as (start, stop) pieces."""
    if row_indices.size == 0:
        return []
    integration = np.searchsorted(index.integration_boundaries, row_indices, side="right") - 1
    starts = np.flatnonzero(np.r_[True, integration[1:] != integration[:-1]])  # each integration's first row
    sizes = np.diff(np.r_[starts, row_indices.size])
    rows_per_unit = max(1, SPREAD_UNIT_BYTES // row_bytes)
    unit_starts, held = [], rows_per_unit
    for start, size in zip(starts, sizes):
        if held >= rows_per_unit:
            unit_starts.append(start)
            held = 0
        held += size
    bounds = np.r_[unit_starts, row_indices.size]
    chunks, current = [], []
    room = rows_per_chunk
    for unit in spread_order(len(unit_starts)):
        rows = row_indices[bounds[unit]:bounds[unit + 1]]
        while rows.size:
            take, rows = rows[:room], rows[room:]
            current.append(take)
            room -= take.size
            if room == 0:
                chunks.append(_contiguous_runs(np.sort(np.concatenate(current))))
                current, room = [], rows_per_chunk
    if current:
        chunks.append(_contiguous_runs(np.sort(np.concatenate(current))))
    return chunks


def slice_block(block: VisibilityBlock, start: int, stop: int) -> VisibilityBlock:
    """Rows [start, stop) of a block, as views -- for splitting a chunk
    across worker threads."""
    rows = slice(start, stop)
    return VisibilityBlock(
        row_indices=block.row_indices[rows],
        data=block.data[rows] if block.data is not None else None,
        weight=block.weight[rows] if block.weight is not None else None,
        axis_types=block.axis_types,
        axis_indices=block.axis_indices,
        ant1=block.ant1[rows],
        ant2=block.ant2[rows],
        source_id=block.source_id[rows],
        jd=block.jd[rows],
        uu_sec=block.uu_sec[rows],
        vv_sec=block.vv_sec[rows],
        ww_sec=block.ww_sec[rows],
        chan_freqs_hz=block.chan_freqs_hz,
        stokes_labels=block.stokes_labels,
    )


def narrow_block(block: VisibilityBlock, rows: np.ndarray | None = None, axis: str | None = None,
                 positions=None) -> VisibilityBlock:
    """A block narrowed to `rows` (a boolean mask or positions; None: all)
    and, along data axis `axis` (e.g. "STOKES"), to `positions` within the
    block's selection along it -- as copies."""
    def take(a):
        return a if rows is None or a is None else a[rows]

    data, weight = take(block.data), take(block.weight)
    axis_indices, chan_freqs_hz, stokes_labels = block.axis_indices, block.chan_freqs_hz, block.stokes_labels
    if axis is not None:
        positions = np.asarray(positions)
        dim = block.axis_types.index(axis) + 1  # data's first dimension is the row
        data = np.take(data, positions, axis=dim) if data is not None else None
        weight = np.take(weight, positions, axis=dim) if weight is not None else None
        axis_indices = {**axis_indices, axis: np.asarray(axis_indices[axis])[positions]}
        if axis == "FREQ" and chan_freqs_hz is not None:
            chan_freqs_hz = np.asarray(chan_freqs_hz)[positions]
        if axis == "STOKES" and stokes_labels is not None:
            stokes_labels = [stokes_labels[p] for p in positions]
    return VisibilityBlock(
        row_indices=take(block.row_indices), data=data, weight=weight, axis_types=block.axis_types,
        axis_indices=axis_indices, ant1=take(block.ant1), ant2=take(block.ant2), source_id=take(block.source_id),
        jd=take(block.jd), uu_sec=take(block.uu_sec), vv_sec=take(block.vv_sec), ww_sec=take(block.ww_sec),
        chan_freqs_hz=chan_freqs_hz, stokes_labels=stokes_labels,
    )
