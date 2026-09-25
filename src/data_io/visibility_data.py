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

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from data_io.raw_data_access import open_raw_memmap
from data_io.row_index import RowIndex, find_axis
from data_io.uvfits_group_params import DEFAULT_RAM_FRACTION_TO_USE, host_total_memory_bytes


@dataclass(frozen=True)
class VisibilityBlock:
    row_indices: np.ndarray  # absolute row indices, sorted ascending
    data: np.ndarray  # complex128, shape (n_rows, *one length per axis in axis_types order)
    weight: np.ndarray  # float64, same shape as data -- AIPS convention: <=0 means flagged
    axis_types: list[str]  # CTYPE label for each of data.shape[1:], in order
    axis_indices: dict[str, np.ndarray]  # axis type -> the pixel indices selected along it
    ant1: np.ndarray
    ant2: np.ndarray
    jd: np.ndarray
    uu_sec: np.ndarray
    vv_sec: np.ndarray
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


def read_visibility_data(
    fits_path: Path | str,
    index: RowIndex,
    row_indices: np.ndarray,
    axis_selection: dict[str, np.ndarray] | None = None,
    max_bytes: int | None = None,
    ram_fraction: float = DEFAULT_RAM_FRACTION_TO_USE,
) -> VisibilityBlock:
    """Read visibility data for exactly the given rows.

    `axis_selection` maps a CTYPE name (e.g. "FREQ", "STOKES", "IF") to the
    pixel indices wanted along that axis; an axis not named is selected in
    full, whatever its length -- there is no assumption that any axis
    besides COMPLEX has length 1. No silent truncation: if the read would
    need more than `max_bytes` (default: `ram_fraction` of host RAM), this
    raises with the estimated size and the budget, rather than partially
    reading.
    """
    row_indices = np.unique(np.asarray(row_indices, dtype=np.int64))
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
    selected_shape = tuple(len(idx) for idx in selected_indices)

    n_rows = len(row_indices)
    data_floats_per_group = int(np.prod(index.data_axis_lengths))
    row_bytes = (index.pcount + data_floats_per_group) * 4

    estimated_bytes = n_rows * int(np.prod(selected_shape)) * 3 * 8  # complex128 + float64 weight, worst case
    if max_bytes is None:
        max_bytes = int(host_total_memory_bytes() * ram_fraction)
    if estimated_bytes > max_bytes:
        raise MemoryError(
            f"requested visibility read needs ~{estimated_bytes / 1e9:.2f}GB "
            f"({n_rows:,} rows x shape {selected_shape}), exceeding the "
            f"{max_bytes / 1e9:.2f}GB budget ({ram_fraction:.0%} of host RAM). "
            f"Narrow the selection, or pass max_bytes explicitly."
        )

    # Reshape order matches numpy's natural order for a GroupsHDU: dims run from the
    # highest axis number (slowest) to axis 2 (fastest) -- confirmed directly against
    # astropy's own construction (row_index.py's module docstring / _read_data_axes).
    reshape_dims = list(reversed(index.data_axis_lengths))

    def _position_in_reshaped(axis_num: int) -> int:
        return 1 + (n_axes - 1 - axis_num)  # +1 for the leading row axis

    complex_pos = _position_in_reshaped(complex_axis)
    other_positions = [_position_in_reshaped(i) for i in other_axis_nums]

    out_data = np.empty((n_rows, *selected_shape), dtype=np.complex128)
    out_weight = np.empty((n_rows, *selected_shape), dtype=np.float64)

    write_pos = 0
    for start, stop in _contiguous_runs(row_indices):
        n = stop - start
        block = open_raw_memmap(
            fits_path,
            dtype=">f4",
            shape=(n, index.pcount + data_floats_per_group),
            offset=index.data_offset + start * row_bytes,
        )
        vis = np.array(block[:, index.pcount :], dtype=np.float32)
        del block
        vis = vis.reshape((n, *reshape_dims))

        # Move COMPLEX to position 1 and every other axis to positions 2.. in
        # axis_types order -- generic for however many axes this file has.
        src_positions = [complex_pos] + other_positions
        dest_positions = list(range(1, len(src_positions) + 1))
        vis = np.moveaxis(vis, src_positions, dest_positions)

        for i, idx in enumerate(selected_indices):
            vis = np.take(vis, idx, axis=2 + i)

        out_data[write_pos : write_pos + n] = vis[:, 0].astype(np.float64) + 1j * vis[:, 1].astype(np.float64)
        out_weight[write_pos : write_pos + n] = vis[:, 2].astype(np.float64)
        write_pos += n

    axis_indices = dict(zip(axis_types, selected_indices))
    chan_freqs_hz = index.chan_freqs_hz[axis_indices["FREQ"]] if "FREQ" in axis_indices else None
    stokes_labels = (
        [index.stokes_labels[i] for i in axis_indices["STOKES"]] if "STOKES" in axis_indices else None
    )

    return VisibilityBlock(
        row_indices=row_indices,
        data=out_data,
        weight=out_weight,
        axis_types=axis_types,
        axis_indices=axis_indices,
        ant1=index.ant1[row_indices],
        ant2=index.ant2[row_indices],
        jd=index.jd[row_indices],
        uu_sec=index.uu_sec[row_indices],
        vv_sec=index.vv_sec[row_indices],
        chan_freqs_hz=chan_freqs_hz,
        stokes_labels=stokes_labels,
    )
