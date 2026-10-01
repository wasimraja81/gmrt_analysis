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

import copy as _copy
import queue
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable

import numpy as np

from data_io.visibility_data import VisibilityBlock, slice_block
from visplot.plot_spec import PlotSpec
from visplot.quantities import QUANTITIES, QuantityContext, Unit, resolve_unit
from visplot.value_histogram import ValueHistogram

EMPTY_LAYER = 0
# Smallest block a worker thread gets: below this, splitting a chunk costs
# more in per-block overhead than it saves.
MIN_ROWS_PER_THREAD = 512
FLAGGED_LAYER = np.iinfo(np.int16).max  # flagged samples draw above every category


def combine_flags(good: np.ndarray, pair_shape: tuple, axis_types, rule: str = "any") -> np.ndarray:
    """The flags of the points a plot draws, from each sample's `good`
    (weight > 0, shaped like the data: (rows, *axes)) and `pair_shape`, the
    shape its quantities vary along (1 on an axis they do not vary along).
    A point whose quantities do not vary along Stokes (e.g. u-v) combines
    the selected Stokes of its visibility: by the user's rule (2026-09-29,
    `rule` "any", the default) it is flagged if any of them is flagged; by
    `rule` "all" (T38), only if every one is, so it shows wherever one
    Stokes product has data. One Stokes selected, either is exactly that
    product's flag. Every other axis keeps the data's own flags: a
    visibility is its own time, baseline and channel, so a per-row quantity
    (e.g. u in m) is one point per channel's visibility."""
    if rule not in ("any", "all"):
        raise ValueError(f"the flag rule is 'any' or 'all', got {rule!r}")
    for k, axis_type in enumerate(axis_types, start=1):
        if axis_type == "STOKES" and pair_shape[k] == 1 and good.shape[k] > 1:
            good = good.all(axis=k, keepdims=True) if rule == "any" else good.any(axis=k, keepdims=True)
    return good


class ChunkValues:
    """Quantity values for one chunk, each evaluated once per base and once
    per unit."""

    def __init__(self, block: VisibilityBlock, ctx: QuantityContext):
        self.block = block
        self.ctx = ctx
        self._bases: dict[tuple[str, str], np.ndarray] = {}
        self._scaled: dict[tuple[str, str], np.ndarray] = {}
        self._samples: dict[PlotSpec, tuple] = {}

    def base(self, name: str, base: str) -> np.ndarray:
        """Quantity `name` in its base `base` (a category: its codes)."""
        key = (name, base)
        if key not in self._bases:
            self._bases[key] = QUANTITIES[name].evaluate_base(base, self.block, self.ctx)
        return self._bases[key]

    def in_unit(self, name: str, unit: Unit) -> np.ndarray:
        values = self.base(name, unit.base)
        if unit.factor == 1.0:
            return values
        key = (name, unit.name)
        if key not in self._scaled:
            self._scaled[key] = values * unit.factor
        return self._scaled[key]

    def axis(self, plot: PlotSpec, axis: str) -> np.ndarray:
        """The values on `plot`'s axis "x" or "y", in the axis's unit."""
        return self.in_unit(plot.x if axis == "x" else plot.y, plot.unit(axis, self.ctx))

    def __getitem__(self, name: str) -> np.ndarray:
        """Quantity `name` in its default unit (a category: its codes)."""
        return self.in_unit(name, resolve_unit(name, None, self.ctx))

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

    def good(self, plot: PlotSpec, *arrays) -> np.ndarray | None:
        """Which of `plot`'s points are unflagged (`combine_flags` over the
        shape of `arrays`, its quantities), or None when flags do not apply
        (no weights read, or the plot ignores flags)."""
        if not plot.apply_flags or self.block.weight is None:
            return None
        pair_shape = np.broadcast_shapes(*(a.shape for a in arrays))
        return combine_flags(self.block.weight > 0, pair_shape, self.block.axis_types, plot.combine_flags)

    def _compute_samples(self, plot: PlotSpec):
        arrays = [self.axis(plot, "x"), self.axis(plot, "y")]
        if plot.colorize_by:
            arrays.append(self[plot.colorize_by])
        good = self.good(plot, *arrays)
        use_flags = good is not None
        if use_flags:
            arrays.append(good)
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
    covers the negated values too.

    Values are taken in the unit's base (`base`) and converted by the unit's
    factor in `extent`, so a result found for one unit (e.g. from a cache)
    serves every unit of the same base; the factor is positive, so min, max
    and percentiles convert exactly and log's positive values stay positive."""

    def __init__(self, plot: PlotSpec, axis: str, ctx: QuantityContext | None = None, with_histogram: bool = False):
        self.plot = plot
        self.axis = axis
        self.quantity = plot.x if axis == "x" else plot.y
        self.unit = plot.unit(axis, ctx)
        self.base = self.unit.base
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
        v = values.base(self.quantity, self.base)
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
        """The range in the axis's unit, with a margin on each side (in the
        axis scale's coordinate), so edge samples are not cut by the axes frame; a
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
        lo, hi = lo * self.unit.factor, hi * self.unit.factor
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
    samples twice leaves it unchanged.

    A plot with `limits` (T47) keeps its unflagged samples beyond them in
    two more grids of the same kind, `below` and `above` (drawn as ▼ and ▲),
    placed on the grid's edge where they lie beyond it, and counts them by
    category with their lowest (below) or highest (above) value:
    `beyond[kind][code] = [count, extreme]`."""

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
        self.below = np.zeros(height * width, dtype=np.int16) if plot.limits else None
        self.above = np.zeros(height * width, dtype=np.int16) if plot.limits else None
        self.beyond: dict[str, dict[int, list]] = {"below": {}, "above": {}}
        self.seen_codes: set[int] = set()
        self.n_samples = 0
        self.n_outside = 0  # samples outside the extent, or not showable on the axis scale
        self._lock = threading.Lock()  # apply() and snapshot() may run on different threads

    def compute(self, values: ChunkValues):
        """This chunk's samples as (pixel, code, flagged, n_outside, kind,
        beyond) inside the grid (`kind`: 0, or 1 below / 2 above the plot's
        limits, None without limits; `beyond`: their counts and extreme
        values by category, as `self.beyond` holds them), or None if none
        fall inside. Safe to call from worker threads."""
        x, y, code, flagged = values.samples(self.plot)
        if x.size == 0:
            return None
        kind = None
        if self.plot.limits:
            axis, low, high = self.plot.limits
            v = y if axis == "y" else x
            kind = np.where(v < low, 1, np.where(v > high, 2, 0)).astype(np.int8)
            if flagged is not None:
                kind[flagged] = 0  # a flagged sample is drawn as flagged
        (x0, x1), (y0, y1) = self._tx, self._ty
        # Pixel coordinates in place: one temporary per axis (after the
        # scale's transform, for a non-linear axis).
        with np.errstate(invalid="ignore", divide="ignore"):  # log of non-positive: NaN, outside
            tx = x - x0 if self.x_scale.is_linear else self.x_scale.forward(x) - x0
            ty = y - y0 if self.y_scale.is_linear else self.y_scale.forward(y) - y0
            tx *= self.width / (x1 - x0)
            ty *= self.height / (y1 - y0)
            if kind is not None and kind.any():  # beyond a limit and the grid's edge: on the edge
                t, size = (ty, self.height) if self.plot.limits[0] == "y" else (tx, self.width)
                t[kind == 1] = np.maximum(t[kind == 1], 0.0)
                t[kind == 2] = np.minimum(t[kind == 2], float(size))
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
        values_beyond = None
        if kind is not None:
            values_beyond = y if self.plot.limits[0] == "y" else x
        if not inside.all():
            pixel = pixel[inside]
            code = code[inside] if code is not None else None
            flagged = flagged[inside] if flagged is not None else None
            kind = kind[inside] if kind is not None else None
            values_beyond = values_beyond[inside] if values_beyond is not None else None
        if pixel.size == 0:
            return None if not n_outside else (pixel, code, flagged, n_outside, None, None)
        if code is not None and code.max() + 1 >= FLAGGED_LAYER:
            raise ValueError(f"category code {code.max()} too large for the layer grid")
        beyond = None
        if kind is not None and kind.any():
            codes = code if code is not None else np.zeros(kind.size, dtype=np.int64)
            beyond = {}
            for k, name, extreme in ((1, "below", np.min), (2, "above", np.max)):
                for c in np.unique(codes[kind == k]):
                    chosen = (kind == k) & (codes == c)
                    beyond.setdefault(name, {})[int(c)] = [int(chosen.sum()), float(extreme(values_beyond[chosen]))]
        return pixel, code, flagged, n_outside, kind, beyond

    def apply(self, result) -> None:
        """Write one block's samples into the grid (one thread only)."""
        if result is None:
            return
        with self._lock:
            self._apply(result)

    def _apply(self, result) -> None:
        pixel, code, flagged, n_outside, kind, beyond = result
        self.n_outside += n_outside
        if pixel.size == 0:
            return
        if kind is not None and kind.any():
            for k, grid in ((1, self.below), (2, self.above)):
                chosen = kind == k
                for c in (np.unique(code[chosen]) if code is not None else [0]):
                    sel = pixel[chosen] if code is None else pixel[chosen & (code == c)]
                    grid[sel] = np.maximum(grid[sel], c + 1)
            for name, counts in beyond.items():
                for c, (n, extreme) in counts.items():
                    seen = self.beyond[name].setdefault(c, [0, extreme])
                    seen[0] += n
                    seen[1] = min(seen[1], extreme) if name == "below" else max(seen[1], extreme)
            self.seen_codes.update(int(c) for c in (np.unique(code[kind > 0]) if code is not None else [0]))
            self.n_samples += int((kind > 0).sum())
            keep = kind == 0
            pixel = pixel[keep]
            code = code[keep] if code is not None else None
            flagged = flagged[keep] if flagged is not None else None
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

    def snapshot(self) -> "GridReducer":
        """A copy of the grid's current state, taken under the lock, for
        drawing from another thread while this one keeps streaming."""
        with self._lock:
            copy = _copy.copy(self)
            copy.layers = self.layers.copy()
            copy.below = self.below.copy() if self.below is not None else None
            copy.above = self.above.copy() if self.above is not None else None
            copy.beyond = {name: {c: list(v) for c, v in counts.items()} for name, counts in self.beyond.items()}
            copy.seen_codes = set(self.seen_codes)
        return copy

    def layers_2d(self, grid: np.ndarray | None = None) -> np.ndarray:
        """The layer grid (or `grid`: `below`, `above`) as (height, width),
        row 0 at the bottom (y0)."""
        return (self.layers if grid is None else grid).reshape(self.height, self.width)


class ViewRowsReducer:
    """Which rows of a selection (`row_indices`, ascending) can hold a point
    of `plot` inside (x_extent, y_extent): the rows a pass reading
    visibility data over that view needs -- a draw, an export, a Locate box
    (T19 point D). Only the axes whose
    quantity needs no visibility data are evaluated, so a pass of it reads
    none; a row is kept when, on each of them, its values over the channels
    and Stokes reach the extent (for a mirrored plot, or their negation on
    both axes). A row left out holds no point inside; a row kept may hold
    none."""

    def __init__(self, plot: PlotSpec, x_extent, y_extent, row_indices: np.ndarray):
        self.plot = plot
        self.extents = {"x": sorted(float(v) for v in x_extent), "y": sorted(float(v) for v in y_extent)}
        self.row_indices = np.asarray(row_indices)
        self.keep = np.zeros(len(self.row_indices), dtype=bool)

    def compute(self, values: ChunkValues) -> np.ndarray:
        """Positions, in `row_indices`, of this chunk's rows to keep."""
        n = len(values.block.row_indices)
        plain = np.ones(n, dtype=bool)
        mirrored = np.ones(n, dtype=bool)
        for axis in ("x", "y"):
            if QUANTITIES[self.plot.x if axis == "x" else self.plot.y].needs_data:
                continue
            v = np.asarray(values.axis(self.plot, axis), dtype=np.float64)
            flat = v.reshape(v.shape[0], -1) if v.ndim > 1 else v.reshape(-1, 1)
            finite = np.isfinite(flat)
            lo = np.where(finite, flat, np.inf).min(axis=1)
            hi = np.where(finite, flat, -np.inf).max(axis=1)
            if lo.size == 1 and n != 1:  # the same values for every row
                lo, hi = np.full(n, lo[0]), np.full(n, hi[0])
            a, b = self.extents[axis]
            plain &= (hi >= a) & (lo <= b)
            mirrored &= (-lo >= a) & (-hi <= b)
        keep = plain | mirrored if self.plot.mirror else plain
        return np.searchsorted(self.row_indices, values.block.row_indices)[keep]

    def apply(self, result) -> None:
        self.keep[result] = True

    def update(self, values: ChunkValues) -> None:
        self.apply(self.compute(values))


LOCATE_FIELDS = ("row", "ant1", "ant2", "jd", "source_id", "channel", "freq_hz", "stokes",
                 "x", "y", "weight", "flagged", "mirrored")


class LocateReducer:
    """The samples of a plot inside a box (x_box, y_box in the axes' units), with
    where each comes from: the first `limit` kept (for a table), every one
    counted by baseline, and every one passed to `sink` if given (e.g. a CSV
    writer), in row order. A plot that does not vary along an axis (e.g. hour
    angle vs time, one value per row) reports that axis as "all" (channel -1,
    Stokes "all").

    Samples are handled as columns (`LOCATE_FIELDS`), one array per field per
    chunk, so memory is bounded by the chunk however many samples fall in
    the box."""

    def __init__(self, plot: PlotSpec, x_box, y_box, limit: int = 10_000, sink=None):
        self.plot = plot
        self.x_box = tuple(sorted(float(v) for v in x_box))
        self.y_box = tuple(sorted(float(v) for v in y_box))
        self.limit = limit
        self.sink = sink
        self._kept: list[dict] = []  # column dicts, up to `limit` samples in total
        self._n_kept = 0
        self.n_found = 0
        self.by_baseline: Counter = Counter()

    @property
    def kept_all(self) -> bool:
        """Whether every located sample was kept (none beyond `limit`)."""
        return self.n_found <= self.limit

    def kept_columns(self) -> list[dict]:
        """The kept samples, as the column dicts `sink` receives."""
        return list(self._kept)

    @property
    def records(self) -> list[dict]:
        """The kept samples, one dict per sample."""
        out = []
        for columns in self._kept:
            for i in range(len(columns["row"])):
                out.append({f: _plain(columns[f][i]) for f in LOCATE_FIELDS})
        for r in out:
            r["channel"] = None if r["channel"] < 0 else r["channel"]
            r["stokes"] = None if r["stokes"] == "all" else r["stokes"]
        return out

    def compute(self, values: ChunkValues):
        block = values.block
        # The points the plot draws, flagged by the plot's own rule (`combine_flags`): a u-v
        # point is one per row and channel, its Stokes combined; without flags, a per-row
        # plot stays one point per row.
        parts = [values.axis(self.plot, "x"), values.axis(self.plot, "y")]
        if self.plot.colorize_by:
            parts.append(values[self.plot.colorize_by])
        good = values.good(self.plot, *parts)
        arrays = np.broadcast_arrays(*parts, *([good] if good is not None else []))
        x, y = arrays[0], arrays[1]
        good = arrays[-1] if good is not None else None
        # a sample's own weight where each point is one sample; none where a point combines several
        weight = block.weight if block.weight is not None and block.weight.shape == x.shape else None
        (x0, x1), (y0, y1) = self.x_box, self.y_box

        pieces, counts, n_found = [], Counter(), 0
        for sign in ((1.0, -1.0) if self.plot.mirror else (1.0,)):
            with np.errstate(invalid="ignore"):
                sx, sy = sign * x, sign * y
                inside = (sx >= x0) & (sx <= x1) & (sy >= y0) & (sy <= y1)
            if good is not None and not self.plot.show_flagged:
                inside &= good
            index = np.nonzero(inside)
            n = len(index[0])
            if not n:
                continue
            n_found += n
            columns = self._columns(block, index, sx, sy, weight, good, sign < 0)
            pairs = np.stack([columns["ant1"], columns["ant2"]], axis=1)
            unique, pair_counts = np.unique(pairs, axis=0, return_counts=True)
            counts.update({(int(a), int(b)): int(c) for (a, b), c in zip(unique, pair_counts)})
            pieces.append(columns)
        return n_found, pieces, counts

    def _columns(self, block, index, sx, sy, weight, good, mirrored: bool) -> dict:
        rows = index[0]
        n = len(rows)
        columns = {
            "row": block.row_indices[rows].astype(np.int64),
            "ant1": block.ant1[rows].astype(np.int64),
            "ant2": block.ant2[rows].astype(np.int64),
            "jd": block.jd[rows].astype(np.float64),
            "source_id": block.source_id[rows].astype(np.int64),
            "channel": np.full(n, -1, dtype=np.int64),
            "freq_hz": np.full(n, np.nan),
            "stokes": np.full(n, "all", dtype=object),
            "x": np.asarray(sx[index], dtype=np.float64),
            "y": np.asarray(sy[index], dtype=np.float64),
            "weight": np.asarray(weight[index], dtype=np.float64) if weight is not None else np.full(n, np.nan),
            # by the plot's rule (`combine_flags`); "" where flags do not apply
            "flagged": (np.where(good[index], "no", "yes").astype(object) if good is not None
                        else np.full(n, "", dtype=object)),
            "mirrored": np.full(n, mirrored),
        }
        for k, axis_type in enumerate(block.axis_types, start=1):
            full = len(block.axis_indices[axis_type])
            if sx.shape[k] != full or full == 0:
                continue  # the plot does not vary along this axis: "all"
            if axis_type == "FREQ":
                columns["channel"] = np.asarray(block.axis_indices["FREQ"])[index[k]].astype(np.int64)
                columns["freq_hz"] = np.asarray(block.chan_freqs_hz)[index[k]].astype(np.float64)
            elif axis_type == "STOKES" and block.stokes_labels:
                columns["stokes"] = np.asarray(block.stokes_labels, dtype=object)[index[k]]
        return columns

    def apply(self, result) -> None:
        """Count, keep the first `limit`, and pass every sample to `sink`
        (one thread only, in row order)."""
        n_found, pieces, counts = result
        self.n_found += n_found
        self.by_baseline.update(counts)
        for columns in pieces:
            if self.sink is not None:
                self.sink(columns)
            room = self.limit - self._n_kept
            if room > 0:
                kept = {f: v[:room] for f, v in columns.items()}
                self._kept.append(kept)
                self._n_kept += len(kept["row"])

    def update(self, values: ChunkValues) -> None:
        self.apply(self.compute(values))


def _plain(value):
    """A numpy scalar as the native Python value (for records and tables)."""
    return value.item() if hasattr(value, "item") else value
