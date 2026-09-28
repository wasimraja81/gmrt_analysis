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

import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable

import numpy as np

from data_io.visibility_data import VisibilityBlock, slice_block
from visplot.plot_spec import PlotSpec
from visplot.quantities import QUANTITIES, QuantityContext
from visplot.value_histogram import ValueHistogram

EMPTY_LAYER = 0
# Smallest block a worker thread gets: below this, splitting a chunk costs
# more in per-block overhead than it saves.
MIN_ROWS_PER_THREAD = 512
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

    def samples(self, plot: PlotSpec):
        """The samples `plot` draws from this chunk, as flat arrays
        (x, y, code, flagged): the quantities broadcast to the shape the pair
        needs, flagged samples dropped or marked per the plot, non-finite
        values dropped, and (-x, -y) appended when mirrored. `code` is None
        for a plot without categories and `flagged` None when no sample is
        flagged, so the common case carries no per-sample arrays for either.

        Flags come from the weights, so they apply only when the chunk was
        read with data; a metadata-only chunk treats every sample as unflagged."""
        if plot not in self._samples:
            self._samples[plot] = self._compute_samples(plot)
        return self._samples[plot]

    def release(self, plot: PlotSpec) -> None:
        """Drop `plot`'s samples, once every reducer of that plot has seen them."""
        self._samples.pop(plot, None)

    def _compute_samples(self, plot: PlotSpec):
        arrays = [self[plot.x], self[plot.y]]
        if plot.colorize_by:
            arrays.append(self[plot.colorize_by])
        use_flags = plot.apply_flags and self.block.weight is not None
        if use_flags:
            arrays.append(self.block.weight > 0)
        arrays = [a.reshape(-1) for a in np.broadcast_arrays(*arrays)]
        x = arrays[0].astype(np.float64, copy=False)
        y = arrays[1].astype(np.float64, copy=False)
        code = arrays[2].astype(np.int64, copy=False) if plot.colorize_by else None
        good = arrays[-1] if use_flags else None

        keep = np.isfinite(x)
        keep &= np.isfinite(y)
        if good is not None and not plot.show_flagged:
            keep &= good
        flagged = None
        if good is not None and plot.show_flagged and not good.all():
            flagged = ~good
        if not keep.all():
            x, y = x[keep], y[keep]
            code = code[keep] if code is not None else None
            flagged = flagged[keep] if flagged is not None else None
        if plot.mirror:
            x, y = np.concatenate([x, -x]), np.concatenate([y, -y])
            code = np.concatenate([code, code]) if code is not None else None
            flagged = np.concatenate([flagged, flagged]) if flagged is not None else None
        return x, y, code, flagged


def run_stream(
    chunks: Iterable[VisibilityBlock],
    ctx: QuantityContext,
    reducers: list,
    on_chunk: Callable[[int], bool] | None = None,
    threads: int = 1,
    prefetch: bool = True,
) -> bool:
    """Feed every chunk to every reducer. `on_chunk(rows_done)` is called
    after each chunk; returning False stops the stream early. Returns True if
    every chunk was consumed.

    `prefetch` reads the next chunk in a background thread while the current
    one is processed (at most one chunk waits, in memory). With `threads` > 1
    each chunk is split into row blocks whose quantities and pixel indices are
    computed by worker threads; every reducer's results are then applied by
    this thread, in row order, so the outcome is the same for any thread
    count. A plot's samples are released as soon as its reducers have them."""
    source = _prefetched(chunks) if prefetch else iter(chunks)
    pool = ThreadPoolExecutor(max_workers=threads) if threads > 1 else None
    rows_done = 0
    try:
        for block in source:
            n = len(block.row_indices)
            parts = max(1, min(threads, n // MIN_ROWS_PER_THREAD))
            bounds = [(n * i // parts, n * (i + 1) // parts) for i in range(parts)]
            blocks = [slice_block(block, lo, hi) for lo, hi in bounds] if parts > 1 else [block]
            if pool is not None and len(blocks) > 1:
                results = list(pool.map(lambda b: _compute_all(b, ctx, reducers), blocks))
            else:
                results = [_compute_all(b, ctx, reducers) for b in blocks]
            for block_results in results:
                for reducer, result in zip(reducers, block_results):
                    reducer.apply(result)
            del results, blocks
            rows_done += n
            if on_chunk is not None and on_chunk(rows_done) is False:
                return False
        return True
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
        if prefetch:
            source.close()


def _compute_all(block: VisibilityBlock, ctx: QuantityContext, reducers: list) -> list:
    """Every reducer's result for one block, releasing each plot's samples
    after its last reducer (reducers of one plot are adjacent or not; the
    release happens once none of the remaining reducers needs them)."""
    values = ChunkValues(block, ctx)
    remaining = {}
    for reducer in reducers:
        plot = getattr(reducer, "plot", None)
        remaining[plot] = remaining.get(plot, 0) + 1
    results = []
    for reducer in reducers:
        results.append(reducer.compute(values))
        plot = getattr(reducer, "plot", None)
        remaining[plot] -= 1
        if plot is not None and remaining[plot] == 0:
            values.release(plot)
    return results


def _prefetched(chunks: Iterable[VisibilityBlock]):
    """Iterate `chunks` from a background thread, one chunk ahead. Errors in
    the reader are raised here; closing the generator stops the reader."""
    q: queue.Queue = queue.Queue(maxsize=1)
    stop = threading.Event()
    done = object()

    def reader():
        try:
            for chunk in chunks:
                while not stop.is_set():
                    try:
                        q.put(chunk, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                if stop.is_set():
                    return
            q.put(done)
        except BaseException as err:  # re-raised in the consuming thread
            q.put(err)

    thread = threading.Thread(target=reader, name="visplot-reader", daemon=True)
    thread.start()
    try:
        while True:
            item = q.get()
            if item is done:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
        while thread.is_alive():  # unblock a reader waiting on a full queue
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            thread.join(timeout=0.1)


class RangeReducer:
    """The range of one axis ("x" or "y") of a plot: min/max, and in
    percentile mode a histogram of values for the percentiles. Evaluates
    only that axis's quantity, so a range pass reads visibility data only
    when that quantity needs it; flagged samples are left out when the chunk
    carries weights (and the plot applies flags); values the axis scale
    cannot show (log: non-positive) are left out; a mirrored plot's range
    covers the negated values too."""

    def __init__(self, plot: PlotSpec, axis: str, with_histogram: bool = False):
        self.plot = plot
        self.axis = axis
        self.quantity = plot.x if axis == "x" else plot.y
        self.scale = plot.axis_scale(axis)
        self.mode = plot.range_mode(axis)
        self.lo = np.inf
        self.hi = -np.inf
        # kept in percentile mode, or when asked (e.g. to cache it for later percentile runs)
        self.histogram = ValueHistogram() if self.mode == "percentile" or with_histogram else None

    def load(self, lo: float, hi: float, histogram_counts: np.ndarray) -> None:
        """Take a result computed earlier (e.g. from a cache) in place of a pass."""
        self.lo, self.hi = lo, hi
        self.histogram = ValueHistogram()
        self.histogram.counts[:] = histogram_counts

    def compute(self, values: ChunkValues):
        """This chunk's (lo, hi, histogram or None), or None if it has no
        value to show. Safe to call from worker threads."""
        v = values[self.quantity]
        weight = values.block.weight
        if self.plot.apply_flags and not self.plot.show_flagged and weight is not None:
            v, good = np.broadcast_arrays(v, weight > 0)
            v = v[good]
        v = v[np.isfinite(v)]
        valid = self.scale.valid(v)
        if valid is not None:
            v = v[valid]
        if not v.size:
            return None
        lo, hi = float(v.min()), float(v.max())
        histogram = None
        if self.histogram is not None:
            histogram = ValueHistogram()
            histogram.add(np.asarray(v, dtype=np.float64).ravel())
            if self.plot.mirror:
                histogram.add(-np.asarray(v, dtype=np.float64).ravel())
        if self.plot.mirror:
            lo, hi = min(lo, -hi), max(hi, -lo)
        return lo, hi, histogram

    def apply(self, result) -> None:
        if result is not None:
            self.lo = min(self.lo, result[0])
            self.hi = max(self.hi, result[1])
            if result[2] is not None:
                self.histogram.merge(result[2])

    def update(self, values: ChunkValues) -> None:
        self.apply(self.compute(values))

    def extent(self, margin: float = 0.03) -> tuple[float, float]:
        """The range with a margin on each side (in the axis scale's
        coordinate), so edge samples are not cut by the axes frame; a
        category axis gets half a slot each side. Percentile mode narrows
        the range to `range_percentiles` of the values."""
        if not np.isfinite(self.lo):
            return (1.0, 10.0) if self.scale.name == "log" else (0.0, 1.0)
        if QUANTITIES[self.quantity].categorical:
            return (self.lo - 0.5, self.hi + 0.5)
        lo, hi = self.lo, self.hi
        if self.mode == "percentile" and self.histogram is not None and self.histogram.counts.any():
            p_lo, p_hi = self.plot.range_percentiles
            lo = max(lo, self.histogram.percentile(p_lo, upper=False))
            hi = min(hi, self.histogram.percentile(p_hi, upper=True))
            if self.plot.mirror:
                lo, hi = min(lo, -hi), max(hi, -lo)
        t_lo, t_hi = (float(t) for t in self.scale.forward(np.array([lo, hi], dtype=np.float64)))
        span = t_hi - t_lo
        pad = span * margin if span > 0 else max(abs(t_lo) * margin, 0.5)
        lo, hi = (float(v) for v in self.scale.inverse(np.array([t_lo - pad, t_hi + pad])))
        return (lo, hi)


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
        self.x_scale = plot.axis_scale("x")
        self.y_scale = plot.axis_scale("y")
        # grid edges in each axis scale's coordinate (log10 of the value for a log axis, ...)
        self._tx = tuple(float(t) for t in self.x_scale.forward(np.array(self.x_extent)))
        self._ty = tuple(float(t) for t in self.y_scale.forward(np.array(self.y_extent)))
        self.layers = np.zeros(height * width, dtype=np.int16)
        self.seen_codes: set[int] = set()
        self.n_samples = 0
        self.n_outside = 0  # samples outside the extent, or not showable on the axis scale

    def compute(self, values: ChunkValues):
        """This chunk's samples as (pixel, code, flagged) inside the grid, or
        None if none fall inside. Safe to call from worker threads."""
        x, y, code, flagged = values.samples(self.plot)
        if x.size == 0:
            return None
        (x0, x1), (y0, y1) = self._tx, self._ty
        # Pixel coordinates in place: one temporary per axis (after the
        # scale's transform, for a non-linear axis).
        with np.errstate(invalid="ignore", divide="ignore"):  # log of non-positive: NaN, outside
            tx = x - x0 if self.x_scale.is_linear else self.x_scale.forward(x) - x0
            ty = y - y0 if self.y_scale.is_linear else self.y_scale.forward(y) - y0
            tx *= self.width / (x1 - x0)
            ty *= self.height / (y1 - y0)
            inside = tx >= 0
            inside &= tx <= self.width
            inside &= ty >= 0
            inside &= ty <= self.height
            n_outside = int(inside.size - np.count_nonzero(inside))
            pixel = ty.astype(np.int64)
            np.minimum(pixel, self.height - 1, out=pixel)
            pixel *= self.width
            ix = tx.astype(np.int64)
            np.minimum(ix, self.width - 1, out=ix)
            pixel += ix
        del tx, ty, ix
        if not inside.all():
            pixel = pixel[inside]
            code = code[inside] if code is not None else None
            flagged = flagged[inside] if flagged is not None else None
        if pixel.size == 0:
            return None if not n_outside else (pixel, code, flagged, n_outside)
        if code is not None and code.max() + 1 >= FLAGGED_LAYER:
            raise ValueError(f"category code {code.max()} too large for the layer grid")
        return pixel, code, flagged, n_outside

    def apply(self, result) -> None:
        """Write one block's samples into the grid (one thread only)."""
        if result is None:
            return
        pixel, code, flagged, n_outside = result
        self.n_outside += n_outside
        if pixel.size == 0:
            return
        if code is None and flagged is None and not self.plot.show_flagged:
            self.layers[pixel] = 1  # one layer, nothing above it: direct writes
            self.seen_codes.add(0)
        else:
            good_pixel = pixel if flagged is None else pixel[~flagged]
            good_code = None if code is None else (code if flagged is None else code[~flagged])
            if code is None:
                present = [0] if good_pixel.size else []
            else:
                present = np.flatnonzero(np.bincount(good_code)) if good_code.size else []
            for c in present:  # ascending, so higher codes paint over lower ones
                sel = good_pixel if code is None else good_pixel[good_code == c]
                self.layers[sel] = np.maximum(self.layers[sel], c + 1)
            if flagged is not None and flagged.any():
                self.layers[pixel[flagged]] = FLAGGED_LAYER
            self.seen_codes.update(int(c) for c in present)
        self.n_samples += int(pixel.size)

    def update(self, values: ChunkValues) -> None:
        self.apply(self.compute(values))

    def layers_2d(self) -> np.ndarray:
        """The layer grid as (height, width), row 0 at the bottom (y0)."""
        return self.layers.reshape(self.height, self.width)
