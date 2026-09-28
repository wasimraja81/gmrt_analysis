"""Streaming generic plots over a row selection: the chunk size, the range
pre-pass, and the plotting pass -- all through `stream.run_stream`.

Memory is set by the chunk size and the plots' pixel grids, never by the
number of rows or samples selected.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from data_io.row_index import RowIndex
from data_io.uvfits_group_params import DEFAULT_RAM_FRACTION_TO_USE, host_total_memory_bytes
from data_io.visibility_data import iter_visibility_chunks
from visplot.plot_spec import PlotSpec
from visplot.quantities import QUANTITIES, QuantityContext
from visplot.stream import GridReducer, RangeReducer, run_stream

# Peak working memory of the streaming plot path per byte of full rows in a
# chunk: the chunk being processed (selected samples as complex64 and float32
# weight, and the per-sample arrays its worker threads build), plus the next
# chunk waiting in the prefetch queue. Measured 2026-09-28 at the worst case
# (all four Stokes, colored by Stokes, 6 threads, 256 MiB chunks): 2.66 GB peak
# against 0.20 GB before the pass, ~9.2 per chunk byte.
STREAM_MEMORY_PER_CHUNK_BYTE = 10
# Largest chunk whatever the RAM: an interactive window's first image appears
# after one chunk, and later chunks refresh it.
MAX_STREAM_CHUNK_BYTES = 256 * 1024**2
# Worker threads per chunk: the project's convention of 6 (physical cores, no
# hyperthreading), or fewer on a smaller machine.
DEFAULT_STREAM_THREADS = min(6, os.cpu_count() or 1)


def stream_chunk_bytes(ram_fraction: float = DEFAULT_RAM_FRACTION_TO_USE) -> int:
    """Chunk size (bytes of full rows) keeping the plot's working memory
    within `ram_fraction` of this host's RAM -- the same budget as index
    building (`read_all_param_columns`)."""
    budget = host_total_memory_bytes() * ram_fraction
    return int(min(MAX_STREAM_CHUNK_BYTES, budget / STREAM_MEMORY_PER_CHUNK_BYTE))


@dataclass
class XYSource:
    """The selection a session streams: which rows, which channels/Stokes,
    and the context quantities are evaluated in."""

    fits_path: Path | str
    index: RowIndex
    row_indices: np.ndarray
    axis_selection: dict[str, np.ndarray] | None
    ctx: QuantityContext
    chunk_bytes: int
    threads: int = 1

    @property
    def n_rows(self) -> int:
        return len(self.row_indices)

    @property
    def row_bytes(self) -> int:
        """Bytes read from disk per row when visibility data is read: the full
        row (every channel and Stokes), whatever the axis selection."""
        return (self.index.pcount + int(np.prod(self.index.data_axis_lengths))) * 4

    @property
    def samples_per_row(self) -> int:
        """Samples per row under the axis selection (every data axis but COMPLEX)."""
        selection = self.axis_selection or {}
        n = 1
        for ctype, length in zip(self.index.data_axis_types, self.index.data_axis_lengths):
            if ctype != "COMPLEX":
                n *= len(selection[ctype]) if ctype in selection else length
        return n

    def stream(self, reducers: list, read_data: bool, on_chunk: Callable[[int], bool] | None = None) -> bool:
        chunks = iter_visibility_chunks(
            self.fits_path, self.index, self.row_indices, self.axis_selection,
            max_chunk_bytes=self.chunk_bytes, read_data=read_data,
        )
        return run_stream(chunks, self.ctx, reducers, on_chunk, threads=self.threads)


def resolve_extents(source: XYSource, plots: list[PlotSpec], on_chunk=None) -> dict[PlotSpec, tuple]:
    """(x_extent, y_extent) per plot: a given range as is, otherwise the
    data's min/max from one pre-pass over the selection. The pre-pass reads
    visibility data only if an axis being ranged is a visibility quantity
    (amp, real, imag, phase); a pre-pass over metadata alone covers every
    selected sample, flagged or not."""
    reducers = {(plot, axis): RangeReducer(plot, axis) for plot, axis in range_pass_axes(plots)}
    if reducers:
        source.stream(list(reducers.values()), read_data=range_pass_reads_data(plots), on_chunk=on_chunk)

    extents = {}
    for plot in plots:
        x = plot.x_range if plot.x_range is not None else reducers[(plot, "x")].extent()
        y = plot.y_range if plot.y_range is not None else reducers[(plot, "y")].extent()
        extents[plot] = (tuple(x), tuple(y))
    return extents


def plot_grids(source: XYSource, plots: list[PlotSpec], extents, shapes, on_chunk=None) -> tuple[dict, bool]:
    """One plotting pass over the selection, filling a `GridReducer` per plot
    (`shapes[plot]` = (height, width)). Returns the grids and whether the pass
    ran to the end."""
    grids = {p: GridReducer(p, *extents[p], *shapes[p]) for p in plots}
    completed = source.stream(list(grids.values()), read_data=any(p.needs_data for p in plots), on_chunk=on_chunk)
    return grids, completed


def range_pass_axes(plots: list[PlotSpec]) -> list[tuple[PlotSpec, str]]:
    """The (plot, axis) pairs whose range comes from the data (no range given)."""
    return [(p, axis) for p in plots for axis, fixed in (("x", p.x_range), ("y", p.y_range)) if fixed is None]


def range_pass_reads_data(plots: list[PlotSpec]) -> bool:
    """Whether the range pass reads visibility data: only if a ranged axis is
    a visibility quantity (amp, real, imag, phase)."""
    return any(QUANTITIES[p.x if axis == "x" else p.y].needs_data for p, axis in range_pass_axes(plots))


def describe_passes(source: XYSource, plots: list[PlotSpec]) -> list[str]:
    """What streaming these plots will do, before it starts:
    how much is selected, and what each pass over the selection reads."""
    from visplot.quantities import quantity_label

    if any(p.needs_data for p in plots):
        n_samples = source.n_rows * source.samples_per_row
        size = (f"{source.n_rows:,} rows x {source.samples_per_row:,} visibility samples per row "
                f"= {n_samples:,} samples")
    else:
        size = f"{source.n_rows:,} rows (one value per row; no visibility data needed)"
    lines = [size, "plots: " + "; ".join(p.title for p in plots)]
    data_gb = source.n_rows * source.row_bytes / 1e9

    def reads(read_data: bool) -> str:
        return f"reads visibility data, {data_gb:.1f} GB" if read_data else "row metadata only, no visibility data read"

    passes = []
    ranged = range_pass_axes(plots)
    if ranged:
        names = sorted({
            quantity_label(p.x if axis == "x" else p.y, source.ctx)
            + (f" (percentiles {p.range_percentiles[0]:g}-{p.range_percentiles[1]:g})"
               if p.range_mode(axis) == "percentile" else "")
            for p, axis in ranged
        })
        passes.append(f"find the data range of {', '.join(names)} ({reads(range_pass_reads_data(plots))})")
    passes.append(f"draw the plots ({reads(any(p.needs_data for p in plots))})")
    for i, text in enumerate(passes, start=1):
        lines.append(f"pass {i} of {len(passes)}: {text}")
    return lines


class PassProgress:
    """Progress text for one pass over the selection: rows done, percentage,
    data read and elapsed time -- all measured, no estimate of time left."""

    def __init__(self, label: str, total_rows: int, bytes_per_row: int):
        self.label = label
        self.total_rows = total_rows
        self.bytes_per_row = bytes_per_row
        self.start = time.monotonic()

    def text(self, rows_done: int) -> str:
        pct = 100.0 * rows_done / self.total_rows if self.total_rows else 100.0
        read = f", {rows_done * self.bytes_per_row / 1e9:.1f} GB read" if self.bytes_per_row else ""
        elapsed = int(time.monotonic() - self.start)
        return (f"{self.label}: {rows_done:,} / {self.total_rows:,} rows ({pct:.0f}%){read}, "
                f"{elapsed // 60}:{elapsed % 60:02d} elapsed")


def pass_progress(source: XYSource, label: str, read_data: bool) -> PassProgress:
    return PassProgress(label, source.n_rows, source.row_bytes if read_data else 0)
