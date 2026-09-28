"""Drawing a `GridReducer`'s layer grid as a plot: the grid becomes an image
on ordinary matplotlib axes, so labels, units, titles, legend and ticks stay
vector while the samples, however many, are one image.

The same code serves a saved page (grid at 600 dpi, shown at 600 or summed
down to 150) and an interactive window (grid at the window's own pixels).
"""

from __future__ import annotations

import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from visplot.clock_axis import ClockFormatter, ClockLocator
from visplot.plot_spec import PlotSpec
from visplot.plot_title import build_plot_title
from visplot.quantities import QUANTITIES, QuantityContext, category_label, quantity_label
from visplot.stream import FLAGGED_LAYER, GridReducer

# Marker area (points^2) by number of samples drawn: (max_samples, size),
# first match wins. Denser plots get smaller markers so samples don't paint
# over each other; 0.25 points^2 is a ~0.5pt marker, one pixel at 150 dpi.
AUTO_POINT_SIZE_STEPS = ((1_000, 20.0), (100_000, 4.0), (1_000_000, 1.0))
AUTO_POINT_SIZE_FLOOR = 0.25
# Above this many samples markers are squares (circles below it): at the
# sizes used there a square reads as a point.
DENSE_MARKER_THRESHOLD = 100_000
CATEGORY_COLORMAP = "tab10"
FLAGGED_COLOR = "lightcoral"
# Occupied pixels stamped per batch when drawing markers, bounding the
# coordinate arrays however dense the grid.
_STAMP_BATCH = 1_000_000


def auto_point_size(n_samples: int) -> float:
    for max_samples, size in AUTO_POINT_SIZE_STEPS:
        if n_samples <= max_samples:
            return size
    return AUTO_POINT_SIZE_FLOOR


def auto_square_marker(n_samples: int) -> bool:
    return n_samples > DENSE_MARKER_THRESHOLD


def marker_radius_px(point_size: float, dpi: float) -> float:
    """Radius in pixels of a marker of area `point_size` points^2 (matplotlib's
    scatter convention: diameter sqrt(point_size) points)."""
    return math.sqrt(point_size) / 2 * dpi / 72


def marker_offsets(radius_px: float, square: bool) -> list[tuple[int, int]]:
    """Pixel offsets a marker covers around its centre pixel."""
    r = int(math.floor(radius_px))
    return [
        (dy, dx)
        for dy in range(-r, r + 1)
        for dx in range(-r, r + 1)
        if square or dy * dy + dx * dx <= radius_px * radius_px
    ] or [(0, 0)]


def draw_markers(layers: np.ndarray, offsets) -> np.ndarray:
    """Every occupied pixel stamped with the marker shape; where markers
    overlap, the higher layer wins, as in paint order."""
    if offsets == [(0, 0)]:
        return layers
    out = layers.copy()
    height, width = layers.shape
    flat = np.flatnonzero(layers)
    for start in range(0, flat.size, _STAMP_BATCH):
        batch = flat[start : start + _STAMP_BATCH]
        ys, xs = np.divmod(batch, width)
        values = layers.ravel()[batch]
        for dy, dx in offsets:
            ny, nx = ys + dy, xs + dx
            ok = (ny >= 0) & (ny < height) & (nx >= 0) & (nx < width)
            ny, nx, v = ny[ok], nx[ok], values[ok]
            out[ny, nx] = np.maximum(out[ny, nx], v)
    return out


def downsample_layers(layers: np.ndarray, factor: int) -> np.ndarray:
    """A grid `factor` times coarser, each pixel the top layer of its block --
    exactly what binning directly at the coarser resolution would give."""
    if factor == 1:
        return layers
    h, w = layers.shape
    return layers.reshape(h // factor, factor, w // factor, factor).max(axis=(1, 3))


def layer_colors(plot: PlotSpec, seen_codes) -> dict[int, tuple]:
    if plot.colorize_by:
        cmap = plt.get_cmap(CATEGORY_COLORMAP)
        colors = {code + 1: cmap(rank % cmap.N) for rank, code in enumerate(sorted(seen_codes))}
    else:
        colors = {1: to_rgba(plot.color)}
    colors[FLAGGED_LAYER] = to_rgba(FLAGGED_COLOR)
    return colors


def layers_to_rgba(layers: np.ndarray, colors: dict[int, tuple]) -> np.ndarray:
    lut = np.zeros((int(FLAGGED_LAYER) + 1, 4), dtype=np.uint8)
    for value, rgba in colors.items():
        lut[value] = np.round(np.asarray(rgba) * 255)
    return lut[layers]


class XYFigure:
    """One plot's figure: vector axes, title and labels, with the samples
    drawn from a `GridReducer` as an image."""

    def __init__(self, plot: PlotSpec, ctx: QuantityContext, sources=None, telescope=None, source_path=None,
                 figsize=(8, 6)):
        self.plot = plot
        self.ctx = ctx
        # A bare Figure (no pyplot): saved with savefig, or embedded in a Qt window.
        self.fig = Figure(figsize=figsize)
        self.ax = self.fig.add_subplot()
        self.ax.set_xlabel(quantity_label(plot.x, ctx, plot.x_unit))
        self.ax.set_ylabel(quantity_label(plot.y, ctx, plot.y_unit))
        for axis, mpl_axis in (("x", self.ax.xaxis), ("y", self.ax.yaxis)):
            if plot.unit(axis, ctx).clock:
                mpl_axis.set_major_locator(ClockLocator())
                mpl_axis.set_major_formatter(ClockFormatter())
        self.ax.set_title(build_plot_title(plot.title, sources, telescope, source_path))
        self.ax.grid(True, alpha=0.3)
        for axis, value in plot.reference_lines:
            line = self.ax.axhline if axis == "y" else self.ax.axvline
            line(value, color="0.5", lw=0.8, ls="--", zorder=1)
        self.image = None
        self._scales_set = False
        self.equal_override: bool | None = None  # set from a window's aspect toggle
        self.view_request: tuple | None = None  # the extents last asked of set_view, before any widening
        self.status = self.fig.text(0.99, 0.005, "", ha="right", va="bottom", fontsize=8, color="0.4")
        self.note = self.fig.text(0.01, 0.005, "", ha="left", va="bottom", fontsize=7, color="darkred", wrap=True)

    def set_view(self, x_extent, y_extent) -> tuple[tuple, tuple]:
        """Set the axes' scales, limits and aspect for these extents, and return
        the limits the axes end up with: with equal aspect the axes box is
        square and the shorter range widens about its centre to the longer's
        span, so a unit is the same length on both axes and both show the same
        span (a mirrored u-v plot: both +-R); a grid binned over the returned
        limits fills the axes. The extents asked for are kept
        (`view_request`), so turning equal scale off again returns to them."""
        self.view_request = (tuple(x_extent), tuple(y_extent))
        self._set_scales()
        equal = self.plot.equal_aspect if self.equal_override is None else self.equal_override
        # Equal scale is the square box and equal spans; matplotlib's own data aspect stays
        # "auto" (a window re-applies set_view after a zoom, keeping the scale equal).
        self.ax.set_box_aspect(1.0 if equal and self.linear else None)
        self.ax.set_aspect("auto")
        self.ax.apply_aspect()  # the box's shape settles before its size is read below
        if equal and self.linear:
            x_extent, y_extent = self._widened_for_equal_scale(x_extent, y_extent)
        self.ax.set_xlim(x_extent)
        self.ax.set_ylim(y_extent)
        return tuple(self.ax.get_xlim()), tuple(self.ax.get_ylim())

    def _widened_for_equal_scale(self, x_extent, y_extent):
        """Widen one axis's range about its centre, never narrow either, so a
        unit spans the same number of pixels on both axes."""
        pos = self.ax.get_position()
        fig_w, fig_h = self.fig.get_size_inches()
        width, height = pos.width * fig_w, pos.height * fig_h
        (x0, x1), (y0, y1) = x_extent, y_extent
        per_inch = max((x1 - x0) / width, (y1 - y0) / height)
        xc, yc = (x0 + x1) / 2, (y0 + y1) / 2
        half_x, half_y = per_inch * width / 2, per_inch * height / 2
        return (xc - half_x, xc + half_x), (yc - half_y, yc + half_y)

    def grid_shape(self, dpi: float) -> tuple[int, int]:
        """(height, width) in pixels of the axes area at `dpi`."""
        pos = self.ax.get_position()
        fig_w, fig_h = self.fig.get_size_inches()
        return max(1, round(pos.height * fig_h * dpi)), max(1, round(pos.width * fig_w * dpi))

    def show(self, grid: GridReducer, display_dpi: float, downsample: int = 1) -> None:
        """Draw `grid` (binned at `downsample` x `display_dpi`) at `display_dpi`."""
        n = grid.n_samples
        size = self.plot.point_size if self.plot.point_size is not None else auto_point_size(n)
        offsets = marker_offsets(marker_radius_px(size, display_dpi), auto_square_marker(n))
        layers = draw_markers(downsample_layers(grid.layers_2d(), downsample), offsets)
        rgba = layers_to_rgba(layers, layer_colors(self.plot, grid.seen_codes))
        if self.image is None:
            self._set_scales()
            if not self.linear:
                # limits first, autoscaling off: a pinned image's 0-1 extent must not reach the limits
                self.ax.set_xlim(grid.x_extent)
                self.ax.set_ylim(grid.y_extent)
                self.ax.set_autoscale_on(False)
        if self.linear:
            # pixels uniform in data space: placed by data extent, so they follow zoom and pan
            extent, transform = (*grid.x_extent, *grid.y_extent), self.ax.transData
        else:
            # pixels uniform in the axis scale's coordinate: placed over the axes area, which
            # matches the scale when the axis limits equal the grid's extent (set below)
            extent, transform = (0.0, 1.0, 0.0, 1.0), self.ax.transAxes
        if self.image is None:
            self.image = self.ax.imshow(rgba, origin="lower", extent=extent, interpolation="nearest",
                                        aspect="auto", zorder=0, transform=transform)
            self.ax.set_xlim(grid.x_extent)
            self.ax.set_ylim(grid.y_extent)
        else:
            self.image.set_data(rgba)
            self.image.set_extent(extent)
            if not self.linear:
                self.ax.set_xlim(grid.x_extent)
                self.ax.set_ylim(grid.y_extent)
        self.image.set_visible(True)
        self._legend(grid)
        self._category_ticks()

    @property
    def linear(self) -> bool:
        return self.plot.axis_scale("x").is_linear and self.plot.axis_scale("y").is_linear

    def _set_scales(self) -> None:
        if self._scales_set:
            return
        self._scales_set = True
        if not self.plot.axis_scale("x").is_linear:
            self.ax.set_xscale(**self.plot.axis_scale("x").mpl_kwargs())
        if not self.plot.axis_scale("y").is_linear:
            self.ax.set_yscale(**self.plot.axis_scale("y").mpl_kwargs())

    def hide_if_view_moved(self, grid: GridReducer) -> None:
        """On a non-linear axis the image is pinned to the axes area, so after
        a zoom or pan it no longer lines up; hide it until the view is redrawn."""
        if not self.linear and self.image is not None:
            in_place = (tuple(self.ax.get_xlim()), tuple(self.ax.get_ylim())) == (grid.x_extent, grid.y_extent)
            self.image.set_visible(in_place)

    def _legend(self, grid: GridReducer) -> None:
        handles = []
        if self.plot.colorize_by and len(grid.seen_codes) > 1:
            colors = layer_colors(self.plot, grid.seen_codes)
            handles = [
                Line2D([], [], marker="s", linestyle="", color=colors[code + 1],
                       label=category_label(self.plot.colorize_by, code, self.ctx))
                for code in sorted(grid.seen_codes)
            ]
        if self.plot.show_flagged and (grid.layers == FLAGGED_LAYER).any():
            handles.append(Line2D([], [], marker="s", linestyle="", color=FLAGGED_COLOR, label="flagged"))
        legend = self.ax.get_legend()
        if legend is not None:
            legend.remove()
        if handles:
            self.ax.legend(handles=handles, fontsize=8, loc="best")

    def _category_ticks(self) -> None:
        for name, axis, limits in ((self.plot.x, self.ax.xaxis, self.ax.get_xlim()),
                                   (self.plot.y, self.ax.yaxis, self.ax.get_ylim())):
            if QUANTITIES[name].categorical:
                codes = range(math.ceil(min(limits)), math.floor(max(limits)) + 1)
                axis.set_ticks(list(codes), [category_label(name, c, self.ctx) for c in codes])

    def set_status(self, text: str) -> None:
        self.status.set_text(text)

    def set_note(self, text: str) -> None:
        """A caveat shown on the plot itself (bottom left), e.g. a warning
        about how a quantity was computed."""
        self.note.set_text(text)


def grid_summary(grid: GridReducer, n_rows: int) -> str:
    """What a finished plot shows: samples drawn, rows read, and samples
    left out because they fall outside the axis ranges (or, on a log axis,
    are not positive)."""
    text = f"{grid.n_samples:,} samples from {n_rows:,} rows"
    if grid.n_outside:
        text += f"; {grid.n_outside:,} outside the axis ranges, left out"
    return text
