"""One loop that feeds chunks of a selection to reducers.

`run_stream` evaluates quantities on each chunk once (`ChunkValues` caches
them, however many reducers ask) and hands the chunk to every reducer; the
chunk is then dropped. A reducer keeps only a small, fixed-size summary:

- `RangeReducer`: the min/max of one plot axis (the range pre-pass).
- `GridReducer`: which pixels of a fixed grid a plot's samples fall in.

Memory is one chunk plus the reducers' summaries, whatever the size of the
selection. Saving plots, filling an interactive window, and re-streaming a
zoomed region all run through `run_stream`; they differ only in the
reducers passed and in the `on_chunk` callback.
"""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np

from data_io.visibility_data import VisibilityBlock
from visplot.plot_spec import PlotSpec
from visplot.quantities import QUANTITIES, QuantityContext

EMPTY_LAYER = 0
FLAGGED_LAYER = np.iinfo(np.int16).max  # flagged samples draw above every category


class ChunkValues:
    """Quantity values for one chunk, each evaluated once."""

    def __init__(self, block: VisibilityBlock, ctx: QuantityContext):
        self.block = block
        self.ctx = ctx
        self._quantities: dict[str, np.ndarray] = {}
        self._samples: dict[PlotSpec, tuple] = {}

    def __getitem__(self, name: str) -> np.ndarray:
        if name not in self._quantities:
            self._quantities[name] = QUANTITIES[name].evaluate(self.block, self.ctx)
        return self._quantities[name]

    def samples(self, plot: PlotSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The samples `plot` draws from this chunk, as flat arrays (x, y,
        category code, flagged): the quantities broadcast to the shape the
        pair needs, flagged samples dropped or marked per the plot, non-finite
        values dropped, and (-x, -y) appended when mirrored.

        Flags come from the weights, so they apply only when the chunk was
        read with data; a metadata-only chunk treats every sample as unflagged."""
        if plot not in self._samples:
            self._samples[plot] = self._compute_samples(plot)
        return self._samples[plot]

    def _compute_samples(self, plot: PlotSpec):
        x = self[plot.x]
        y = self[plot.y]
        code = self[plot.colorize_by] if plot.colorize_by else np.zeros(1, dtype=np.int64)
        use_flags = plot.apply_flags and self.block.weight is not None
        good = self.block.weight > 0 if use_flags else np.ones(1, dtype=bool)
        x, y, code, good = (a.reshape(-1) for a in np.broadcast_arrays(x, y, code, good))

        keep = np.isfinite(x) & np.isfinite(y)
        if not plot.show_flagged:
            keep &= good
        x, y, code, flagged = x[keep], y[keep], code[keep].astype(np.int64), ~good[keep]
        if plot.mirror:
            x, y = np.concatenate([x, -x]), np.concatenate([y, -y])
            code, flagged = np.concatenate([code, code]), np.concatenate([flagged, flagged])
        return x.astype(np.float64), y.astype(np.float64), code, flagged


def run_stream(
    chunks: Iterable[VisibilityBlock],
    ctx: QuantityContext,
    reducers: list,
    on_chunk: Callable[[int], bool] | None = None,
) -> bool:
    """Feed every chunk to every reducer. `on_chunk(rows_done)` is called
    after each chunk; returning False stops the stream early. Returns True if
    every chunk was consumed."""
    rows_done = 0
    for block in chunks:
        values = ChunkValues(block, ctx)
        for reducer in reducers:
            reducer.update(values)
        rows_done += len(block.row_indices)
        if on_chunk is not None and on_chunk(rows_done) is False:
            return False
    return True


class RangeReducer:
    """Min/max of one axis ("x" or "y") of a plot. Evaluates only that axis's
    quantity, so a range pass reads visibility data only when that quantity
    needs it; flagged samples are left out when the chunk carries weights
    (and the plot applies flags), and a mirrored plot's range covers the
    negated values too."""

    def __init__(self, plot: PlotSpec, axis: str):
        self.plot = plot
        self.axis = axis
        self.quantity = plot.x if axis == "x" else plot.y
        self.lo = np.inf
        self.hi = -np.inf

    def update(self, values: ChunkValues) -> None:
        v = values[self.quantity]
        weight = values.block.weight
        if self.plot.apply_flags and not self.plot.show_flagged and weight is not None:
            v, good = np.broadcast_arrays(v, weight > 0)
            v = v[good]
        v = v[np.isfinite(v)]
        if v.size:
            self.lo = min(self.lo, float(v.min()))
            self.hi = max(self.hi, float(v.max()))
            if self.plot.mirror:
                self.lo, self.hi = min(self.lo, -self.hi), max(self.hi, -self.lo)

    def extent(self, margin: float = 0.03) -> tuple[float, float]:
        """The range with a margin on each side, so edge samples are not cut
        by the axes frame; a category axis gets half a slot each side."""
        if not np.isfinite(self.lo):
            return (0.0, 1.0)
        name = self.plot.x if self.axis == "x" else self.plot.y
        if QUANTITIES[name].categorical:
            return (self.lo - 0.5, self.hi + 0.5)
        span = self.hi - self.lo
        pad = span * margin if span > 0 else max(abs(self.lo) * margin, 0.5)
        return (self.lo - pad, self.hi + pad)


class GridReducer:
    """Which pixels of a `height` x `width` grid over (x_extent, y_extent) a
    plot's samples fall in, and which layer drew there last in paint order:
    category code + 1 (ascending codes paint later), flagged samples on top.
    One int16 per pixel, whatever the number of categories; adding the same
    samples twice leaves it unchanged."""

    def __init__(self, plot: PlotSpec, x_extent, y_extent, height: int, width: int):
        self.plot = plot
        self.x_extent = tuple(float(v) for v in x_extent)
        self.y_extent = tuple(float(v) for v in y_extent)
        self.height = height
        self.width = width
        self.layers = np.zeros(height * width, dtype=np.int16)
        self.seen_codes: set[int] = set()
        self.n_samples = 0

    def update(self, values: ChunkValues) -> None:
        x, y, code, flagged = values.samples(self.plot)
        (x0, x1), (y0, y1) = self.x_extent, self.y_extent
        inside = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        if not inside.any():
            return
        x, y, code, flagged = x[inside], y[inside], code[inside], flagged[inside]
        ix = np.minimum(((x - x0) / (x1 - x0) * self.width).astype(np.int64), self.width - 1)
        iy = np.minimum(((y - y0) / (y1 - y0) * self.height).astype(np.int64), self.height - 1)
        pixel = iy * self.width + ix

        if code.size and code.max() + 1 >= FLAGGED_LAYER:
            raise ValueError(f"category code {code.max()} too large for the layer grid")
        layer = (code + 1).astype(np.int16)
        layer[flagged] = FLAGGED_LAYER
        for value in np.unique(layer):
            sel = pixel[layer == value]
            self.layers[sel] = np.maximum(self.layers[sel], value)

        self.seen_codes.update(int(c) for c in np.unique(code[~flagged]))
        self.n_samples += int(x.size)

    def layers_2d(self) -> np.ndarray:
        """The layer grid as (height, width), row 0 at the bottom (y0)."""
        return self.layers.reshape(self.height, self.width)
