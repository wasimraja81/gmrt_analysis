"""Iterations (T26): one plot per baseline, antenna, source or Stokes product
of the selection (--one-plot-per), as CASA plotms's `iteraxis` and AIPS
VPLOT's one baseline per plot. An iteration holds the selection's rows, or
its Stokes products, that pass one more filter; there is one for each value
the selection holds, in antenna, source and Stokes order. A page holds a
grid of iterations (--page-grid), as plotms's gridrows x gridcols and
VPLOT's NPLOTS."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace

import numpy as np

ITERATION_KINDS = ("baseline", "antenna", "source", "stokes")


@dataclass(frozen=True)
class Iteration:
    by: str  # one of ITERATION_KINDS
    value: object  # (ant1, ant2), an antenna number, a source id, or a Stokes label
    label: str  # as the plot names it, e.g. "C00:01-C01:02"

    @property
    def text(self) -> str:
        """What the iteration holds, e.g. "baseline C00:01-C01:02", "Stokes RR"."""
        return f"{'Stokes' if self.by == 'stokes' else self.by} {self.label}"

    @property
    def file_label(self) -> str:
        """`label` in a filename: characters other than letters, digits, '.',
        '+' and '-' become '_' (C00:01-C01:02: C00_01-C01_02)."""
        return re.sub(r"[^A-Za-z0-9.+-]", "_", self.label)

    def row_mask(self, ant1, ant2, source_id) -> np.ndarray | None:
        """Which of these rows the iteration holds; None for a Stokes
        iteration, which holds every row."""
        if self.by == "baseline":
            a, b = self.value
            return (np.asarray(ant1) == a) & (np.asarray(ant2) == b)
        if self.by == "antenna":
            return (np.asarray(ant1) == self.value) | (np.asarray(ant2) == self.value)
        if self.by == "source":
            return np.asarray(source_id) == self.value
        return None

    def context(self, ctx):
        """The quantity context of the iteration's samples: a Stokes
        iteration's holds its one Stokes product, as a selection of it alone
        would."""
        return replace(ctx, stokes_labels=(self.value,)) if self.by == "stokes" else ctx


def list_iterations(by: str, index, row_indices, stokes_labels, ctx) -> list[Iteration]:
    """The iterations of a selection (`row_indices` of `index`, its Stokes
    `stokes_labels`), one per value of `by` it holds, named as `ctx` names
    antennas and sources."""
    rows = np.asarray(row_indices)
    antenna_names = ctx.antenna_names or {}
    source_names = ctx.source_names or {}

    def antenna(number) -> str:
        return str(antenna_names.get(int(number), int(number)))

    if by == "baseline":
        ant1, ant2 = np.asarray(index.ant1)[rows], np.asarray(index.ant2)[rows]
        pairs = np.unique(ant1.astype(np.int64) * 65536 + ant2)  # one integer per pair, as `panel_facts`
        return [Iteration(by, (int(p // 65536), int(p % 65536)), f"{antenna(p // 65536)}-{antenna(p % 65536)}")
                for p in pairs]
    if by == "antenna":
        numbers = np.union1d(np.asarray(index.ant1)[rows], np.asarray(index.ant2)[rows])
        return [Iteration(by, int(a), antenna(a)) for a in numbers]
    if by == "source":
        ids = np.unique(np.asarray(index.source_id)[rows])
        return [Iteration(by, int(s), str(source_names.get(int(s), int(s)))) for s in ids]
    if by == "stokes":
        return [Iteration(by, label, label) for label in stokes_labels]
    raise ValueError(f"one plot per one of {', '.join(ITERATION_KINDS)}; got {by!r}")


def page_layout(n_plots: int, grid: tuple[int, int] | None, default: tuple[int, int]) -> tuple[int, int]:
    """(rows, columns) of plots on a page: `grid` as given; otherwise the
    `default` grid, or, for fewer plots than it holds, the smallest grid
    within it that holds them (the user, 2026-10-02), the one nearest the
    default's shape among equals (4 in 5 x 6: 2 x 2; 13: 3 x 5)."""
    if grid is not None:
        return grid
    rows, cols = default
    if n_plots >= rows * cols:
        return rows, cols
    n = max(1, n_plots)
    shape = cols / rows
    candidates = [(r, c) for r in range(1, rows + 1) for c in range(1, cols + 1) if r * c >= n]
    return min(candidates, key=lambda rc: (rc[0] * rc[1], abs(math.log((rc[1] / rc[0]) / shape))))


def iteration_source(source, iteration: Iteration):
    """`source` (an `XYSource`) narrowed to `iteration`: its rows there, or
    its one Stokes product, with the iteration's quantity context."""
    if iteration.by == "stokes":
        pixel = list(source.index.stokes_labels).index(iteration.value)
        axis_selection = {**(source.axis_selection or {}), "STOKES": np.array([pixel])}
        return replace(source, axis_selection=axis_selection, ctx=iteration.context(source.ctx))
    rows = source.row_indices
    index = source.index
    mask = iteration.row_mask(np.asarray(index.ant1)[rows], np.asarray(index.ant2)[rows],
                              np.asarray(index.source_id)[rows])
    return source.subset(rows[mask])
