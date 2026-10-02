"""Drawing a `GridReducer`'s layer grid as a plot: the grid becomes an image
on ordinary matplotlib axes, so labels, units, titles, the panel under the
plot (`visplot.plot_panel`: color key and what the plot shows) and ticks
stay vector while the samples, however many, are one image.

The same code serves a saved page (grid at 600 dpi, shown at 600 or summed
down to 150) and an interactive window (grid at the window's own pixels).
Its colors are those of its --plot-theme (`visplot.plot_theme`), on screen
and saved.
"""

from __future__ import annotations

import datetime
import math

import numpy as np
from astropy.time import Time
from matplotlib import rcParams
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath

from visplot.clock_axis import DEFAULT_TIME_FORMAT, ClockFormatter, ClockLocator, clock_text
from visplot.fonts import DEFAULT_PANEL_FONT
from visplot.plot_panel import DENSITY_MIN_ALPHA, DensityScale, PanelFacts, PlotPanel
from visplot.plot_spec import PlotSpec
from visplot.plot_theme import DEFAULT_PLOT_THEME, color_axes, plot_theme
from visplot.plot_title import build_plot_title
from visplot.quantities import QUANTITIES, QuantityContext, category_label, day_origin_jd, quantity_label
from visplot.stream import FLAGGED_LAYER, GridReducer

# Marker area (points^2) by number of samples drawn: (max_samples, size),
# first match wins. Denser plots get smaller markers so samples don't paint
# over each other; 0.25 points^2 is a ~0.5pt marker, one pixel at 150 dpi.
AUTO_POINT_SIZE_STEPS = ((1_000, 20.0), (100_000, 4.0), (1_000_000, 1.0))
AUTO_POINT_SIZE_FLOOR = 0.25
# Above this many samples markers are squares (circles below it): at the
# sizes used there a square reads as a point.
DENSE_MARKER_THRESHOLD = 100_000
# Flagged samples are crosses the size of the markers (the user: "the same/similar size as the
# plot markers"), and at least this many pixels from centre to tip: 3 x 3 px, the smallest "x".
FLAGGED_CROSS_MIN_PX = 1
# Points beyond a plot's limits (T47: elevation) are triangles at least this many pixels tall.
LIMIT_MARKER_MIN_PX = 5
# Occupied pixels stamped per batch when drawing markers, bounding the
# coordinate arrays however dense the grid.
_STAMP_BATCH = 1_000_000
# The axes' place in the figure: left and right as fractions of its width; above them the
# title, below them the tick labels and x label and then the panel, in inches, so the
# panel keeps its size however tall the figure.
AXES_LEFT, AXES_RIGHT = 0.125, 0.9
TITLE_IN = 0.72
XLABEL_IN = 0.55
# Clock times on the x axis: tick labels tilted (the user, 2026-10-01), so up to this many fit.
CLOCK_TILT_DEG = 30.0
CLOCK_MAX_TICKS_TILTED = 10


def tilted_label_extra_in(time_format: str, day0: datetime.date) -> float:
    """How much taller, in inches, the widest clock tick label `time_format`
    writes (two digits of seconds, as a zoom may need) stands tilted by
    CLOCK_TILT_DEG than level: the room the axes leave for it."""
    font = FontProperties(size=rcParams["xtick.labelsize"])
    sample = clock_text(9 * 24 + 23.9999, 2, time_format, day0)
    width = TextPath((0, 0), sample, prop=font).get_extents().width
    size = font.get_size_in_points()
    angle = math.radians(CLOCK_TILT_DEG)
    return max(0.0, width * math.sin(angle) + size * math.cos(angle) - size) / 72.0


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


def triangle_offsets(radius_px: float, pointing_down: bool) -> list[tuple[int, int]]:
    """Pixel offsets of a filled triangle with its apex on the centre pixel,
    standing above it (▼, pointing down at the value) or hanging below it
    (▲), as tall as a marker of `radius_px` is wide and at least
    LIMIT_MARKER_MIN_PX (rows count up from the bottom, as the grid's do)."""
    height = max(LIMIT_MARKER_MIN_PX, int(round(2 * radius_px)))
    return [((k if pointing_down else -k), dx)
            for k in range(height + 1) for dx in range(-round(k * 0.58), round(k * 0.58) + 1)]


def cross_offsets(radius_px: float) -> list[tuple[int, int]]:
    """Pixel offsets of an "x" around its centre pixel, as far out as a
    marker of `radius_px` reaches, and at least FLAGGED_CROSS_MIN_PX."""
    r = max(FLAGGED_CROSS_MIN_PX, int(math.floor(radius_px)))
    return sorted({(d, d) for d in range(-r, r + 1)} | {(d, -d) for d in range(-r, r + 1)})


def draw_markers(layers: np.ndarray, offsets, flagged_offsets=None) -> np.ndarray:
    """Every occupied pixel stamped with the marker shape -- a flagged one
    with `flagged_offsets` (default: the same shape); where markers overlap,
    the higher layer wins, as in paint order (flagged on top)."""
    flagged_offsets = offsets if flagged_offsets is None else flagged_offsets
    if offsets == [(0, 0)] and flagged_offsets == [(0, 0)]:
        return layers
    out = layers.copy()
    height, width = layers.shape
    flat = np.flatnonzero(layers)
    for start in range(0, flat.size, _STAMP_BATCH):
        batch = flat[start : start + _STAMP_BATCH]
        ys, xs = np.divmod(batch, width)
        values = layers.ravel()[batch]
        flagged = values == FLAGGED_LAYER
        for chosen, shape in ((~flagged, offsets), (flagged, flagged_offsets)):
            if shape == [(0, 0)] or not chosen.any():
                continue
            cy, cx, cv = ys[chosen], xs[chosen], values[chosen]
            for dy, dx in shape:
                ny, nx = cy + dy, cx + dx
                ok = (ny >= 0) & (ny < height) & (nx >= 0) & (nx < width)
                ny, nx, v = ny[ok], nx[ok], cv[ok]
                out[ny, nx] = np.maximum(out[ny, nx], v)
    return out


def downsample_layers(layers: np.ndarray, factor: int) -> np.ndarray:
    """A grid `factor` times coarser, each pixel the top layer of its block --
    exactly what binning directly at the coarser resolution would give."""
    if factor == 1:
        return layers
    h, w = layers.shape
    return layers.reshape(h // factor, factor, w // factor, factor).max(axis=(1, 3))


def downsample_counts(counts: np.ndarray, factor: int) -> np.ndarray:
    """A count grid `factor` times coarser, each pixel the sum of its block --
    exactly what counting directly at the coarser resolution would give."""
    if factor == 1:
        return counts
    h, w = counts.shape
    return counts.reshape(h // factor, factor, w // factor, factor).sum(axis=(1, 3))


def density_opacity(count: np.ndarray, scale: str = "log", top_percentile: float = 95.0
                    ) -> tuple[np.ndarray, DensityScale]:
    """Opacity of each pixel holding `count` samples (0 for none), from
    DENSITY_MIN_ALPHA at the fewest up to 1, and the scale it followed:
    "log", rising with log(count) to the `top_percentile` percentile of the
    occupied pixels' counts (rounded up), full beyond it; "histogram", by
    the pixel's rank among the occupied pixels (histogram equalization)."""
    count = np.asarray(count, dtype=np.float64)
    occupied = count[count > 0]
    if not occupied.size:
        return np.zeros(count.shape), DensityScale(scale, 0)
    most = int(occupied.max())
    if scale == "histogram":
        ranked = np.sort(occupied)
        below = np.searchsorted(ranked, count, side="right") / ranked.size  # share of pixels holding no more
        lowest = np.searchsorted(ranked, ranked[0], side="right") / ranked.size
        rise = (below - lowest) / (1.0 - lowest) if lowest < 1.0 else np.ones(count.shape)
        described = DensityScale("histogram", most, median=int(np.median(occupied)),
                                 p90=int(np.ceil(np.percentile(occupied, 90))))
    else:
        top = int(np.ceil(np.percentile(occupied, top_percentile)))
        rise = np.minimum(np.log(np.maximum(count, 1)) / np.log(top), 1.0) if top > 1 else np.ones(count.shape)
        described = DensityScale("log", most, top=top, percentile=top_percentile)
    return np.where(count > 0, DENSITY_MIN_ALPHA + (1.0 - DENSITY_MIN_ALPHA) * rise, 0.0), described


def density_rgba(shape, counts: dict[int, np.ndarray], colors: dict[int, tuple], flagged: np.ndarray | None = None,
                 flagged_color: tuple | None = None, scale: str = "log", top_percentile: float = 95.0
                 ) -> tuple[np.ndarray, DensityScale]:
    """RGBA (uint8) of a density plot and the scale its opacity followed:
    each pixel's hue the mix of its categories' colors weighted by their
    counts (Datashader's way, the user's choice), its opacity by
    `density_opacity`; flagged samples over them in their color, on a scale
    of their own counts."""
    total = np.zeros(shape, dtype=np.float64)
    rgb = np.zeros(shape + (3,), dtype=np.float64)
    for code, count in counts.items():
        total += count
        rgb += count[..., None] * np.asarray(colors[code][:3])
    rgb /= np.maximum(total, 1.0)[..., None]
    alpha, described = density_opacity(total, scale, top_percentile)
    if flagged is not None and flagged.any():
        f_alpha, _ = density_opacity(flagged, scale, top_percentile)
        rgb = f_alpha[..., None] * np.asarray(flagged_color[:3]) + (1 - f_alpha[..., None]) * rgb
        alpha = f_alpha + (1 - f_alpha) * alpha
    rgba = np.concatenate([rgb, alpha[..., None]], axis=-1)
    return np.round(rgba * 255).astype(np.uint8), described


def layers_to_rgba(layers: np.ndarray, colors: dict[int, tuple]) -> np.ndarray:
    lut = np.zeros((int(FLAGGED_LAYER) + 1, 4), dtype=np.uint8)
    for value, rgba in colors.items():
        lut[value] = np.round(np.asarray(rgba) * 255)
    return lut[layers]


class XYFigure:
    """One plot's figure: vector axes, title and labels, with the samples
    drawn from a `GridReducer` as an image, and the panel under it
    (`facts`: what the run selected, from `visplot.plot_panel.panel_facts`;
    without them the panel shows what was drawn and the record)."""

    def __init__(self, plot: PlotSpec, ctx: QuantityContext, sources=None, telescope=None, source_path=None,
                 figsize=(8, 7), facts: PanelFacts | None = None, panel_font: str = DEFAULT_PANEL_FONT,
                 theme: str = DEFAULT_PLOT_THEME, time_format: str = DEFAULT_TIME_FORMAT):
        self.plot = plot
        self.ctx = ctx
        self.theme = plot_theme(theme)
        # A bare Figure (no pyplot): saved with savefig, or embedded in a Qt window.
        self.fig = Figure(figsize=figsize, facecolor=self.theme.figure_face)
        self.ax = self.fig.add_subplot()
        self.panel = PlotPanel(self.fig, plot, ctx, facts, sources, font=panel_font, theme=theme)
        self.status = self.panel.status  # what was drawn (get_text / set_text)
        self._seen_codes: tuple = ()
        self.ax.set_xlabel(quantity_label(plot.x, ctx, plot.x_unit))
        self.ax.set_ylabel(quantity_label(plot.y, ctx, plot.y_unit))
        self._xtick_extra_in = 0.0  # room for tilted clock tick labels on the x axis
        clock_axes = [axis for axis in ("x", "y") if plot.unit(axis, ctx).clock]
        day0 = (datetime.date.fromisoformat(Time(day_origin_jd(ctx), format="jd", scale="utc").iso[:10])
                if clock_axes else None)
        for axis in clock_axes:
            mpl_axis = self.ax.xaxis if axis == "x" else self.ax.yaxis
            mpl_axis.set_major_locator(ClockLocator(CLOCK_MAX_TICKS_TILTED if axis == "x" else 7))
            mpl_axis.set_major_formatter(ClockFormatter(time_format, day0))
            if axis == "x":
                self.ax.tick_params(axis="x", labelrotation=CLOCK_TILT_DEG, labelrotation_mode="xtick")
                self._xtick_extra_in = tilted_label_extra_in(time_format, day0)
        self.ax.set_title(build_plot_title(plot.title, sources, telescope, source_path, page=plot.page))
        color_axes(self.ax, self.theme)
        self.ax.grid(True, alpha=0.3, color=self.theme.grid)
        for axis, value in plot.reference_lines:
            line = self.ax.axhline if axis == "y" else self.ax.axvline
            line(value, color=self.theme.reference, lw=0.8, ls="--", zorder=1)
        # A fixed range holds from the start: the empty axes do not stretch to a reference line outside it
        # (an elevation limit of 110 degrees) before the first drawing.
        if plot.x_range is not None and plot.axis_scale("x").is_linear:
            self.ax.set_xlim(plot.x_range)
        if plot.y_range is not None and plot.axis_scale("y").is_linear:
            self.ax.set_ylim(plot.y_range)
        self.image = None
        self.preview = None  # the last complete image, kept under a redraw until it completes (`begin_redraw`)
        self._scales_set = False
        self.equal_override: bool | None = None  # set from a window's aspect toggle
        self.view_request: tuple | None = None  # the extents last asked of set_view, before any widening
        self.relayout()

    def relayout(self) -> None:
        """Place the panel and the axes for the figure's current size: the
        panel at the bottom, the axes between it and the title (in inches,
        so a window resizing the figure keeps the panel's size). A window
        calls this when it resizes the figure."""
        width_in, height_in = self.fig.get_size_inches()
        self.panel.build(width_in, self._seen_codes)
        bottom = min((self.panel.height_in + XLABEL_IN + self._xtick_extra_in) / height_in, 0.6)
        top = max(1.0 - TITLE_IN / height_in, bottom + 0.1)
        self.ax.set_position([AXES_LEFT, bottom, AXES_RIGHT - AXES_LEFT, top - bottom])
        self.ax.apply_aspect()

    def set_view(self, x_extent, y_extent) -> tuple[tuple, tuple]:
        """Set the axes' scales, limits and aspect for these extents, and return
        the limits the axes end up with: with equal aspect the axes box is
        square and the shorter range widens about its centre to the longer's
        span, so a unit is the same length on both axes and both show the same
        span (a mirrored u-v plot: both +-R); a grid binned over the returned
        limits fills the axes. The extents asked for are kept
        (`view_request`), so turning equal scale off again returns to them."""
        self.view_request = (tuple(x_extent), tuple(y_extent))
        self.relayout()  # the figure's size may have changed (a window)
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
        if self.plot.limits:
            self.panel.set_beyond(grid.beyond)
        seen = tuple(sorted(grid.seen_codes)) if self.plot.colorize_by else ()
        if not set(seen) <= set(self.panel.category_codes()) and seen != self._seen_codes:
            self._seen_codes = seen  # a category the selection did not list: key it too
            self.relayout()
        if grid.density:
            rgba = self._density_image(grid, downsample)
        else:
            rgba = self._points_image(grid, display_dpi, downsample)
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
        self._category_ticks()

    def _points_image(self, grid: GridReducer, display_dpi: float, downsample: int) -> np.ndarray:
        """RGBA of a points plot: a marker on every occupied pixel, ▼ / ▲
        beyond its limits, flagged crosses on top."""
        n = grid.n_samples
        size = self.plot.point_size if self.plot.point_size is not None else auto_point_size(n)
        radius = marker_radius_px(size, display_dpi)
        layers = draw_markers(downsample_layers(grid.layers_2d(), downsample),
                              marker_offsets(radius, auto_square_marker(n)), cross_offsets(radius))
        for beyond, pointing_down in ((grid.below, True), (grid.above, False)):
            if beyond is not None and beyond.any():  # over the points, under flagged crosses
                marks = draw_markers(downsample_layers(grid.layers_2d(beyond), downsample),
                                     triangle_offsets(radius, pointing_down))
                over = (marks > 0) & (layers != FLAGGED_LAYER)
                layers = np.where(over, marks, layers)
        if self.plot.show_flagged:
            self.panel.set_flagged_drawn(bool((layers == FLAGGED_LAYER).any()))
        return layers_to_rgba(layers, self.panel.colors(self._seen_codes))

    def _density_image(self, grid: GridReducer, downsample: int) -> np.ndarray:
        """RGBA of a density plot (T21), the panel told its largest count."""
        colors = self.panel.colors(self._seen_codes)
        counts = {code: downsample_counts(grid.layers_2d(c), downsample) for code, c in grid.counts.items()}
        flagged = (downsample_counts(grid.layers_2d(grid.flagged_counts), downsample)
                   if grid.flagged_counts is not None else None)
        shape = (grid.height // downsample, grid.width // downsample)
        rgba, described = density_rgba(shape, counts, {code: colors[code + 1] for code in counts}, flagged,
                                       colors[FLAGGED_LAYER], self.plot.density_scale, self.plot.density_top)
        self.panel.set_density_scale(described)
        if self.plot.show_flagged:
            self.panel.set_flagged_drawn(flagged is not None and bool(flagged.any()))
        return rgba

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

    def begin_redraw(self) -> None:
        """A window draws the view again on a new grid (a zoom, a pan, a
        resize; T19 point D): on linear axes the last complete image stays
        under the new one as a preview, enlarged with the view, until
        `end_redraw`, so the view never empties while the new grid fills in.
        A partial image of a redraw stopped meanwhile is dropped, the preview
        kept. On a non-linear axis the image is pinned to the axes area and
        cannot follow the view (`hide_if_view_moved`): unchanged."""
        if self.image is None or not self.linear:
            return
        if self.preview is None:
            self.preview = self.image
            self.preview.set_zorder(-1)
        else:
            self.image.remove()
        self.image = None

    def end_redraw(self) -> None:
        """The redraw is complete: the preview goes."""
        if self.preview is not None:
            self.preview.remove()
            self.preview = None

    def hide_if_view_moved(self, grid: GridReducer) -> None:
        """On a non-linear axis the image is pinned to the axes area, so after
        a zoom or pan it no longer lines up; hide it until the view is redrawn."""
        if not self.linear and self.image is not None:
            in_place = (tuple(self.ax.get_xlim()), tuple(self.ax.get_ylim())) == (grid.x_extent, grid.y_extent)
            self.image.set_visible(in_place)

    def _category_ticks(self) -> None:
        for name, axis, limits in ((self.plot.x, self.ax.xaxis, self.ax.get_xlim()),
                                   (self.plot.y, self.ax.yaxis, self.ax.get_ylim())):
            if QUANTITIES[name].categorical:
                codes = range(math.ceil(min(limits)), math.floor(max(limits)) + 1)
                axis.set_ticks(list(codes), [category_label(name, c, self.ctx) for c in codes])

    def set_status(self, text: str) -> None:
        """What was drawn (or the drawing's progress), in the panel."""
        self.status.set_text(text)

    def set_note(self, text: str) -> None:
        """A caveat shown on the plot itself (under the panel), e.g. a
        warning about how a quantity was computed."""
        self.panel.set_note(text)

    @property
    def limits_warning(self) -> str | None:
        """What the last drawing found beyond the plot's limits (the panel's
        Limits line), or None."""
        return self.panel.limits_warning

    @property
    def record_id(self) -> str:
        return self.panel.record.get_text()

    def set_record(self, run_id: str | None) -> None:
        """The provenance record of the run the figure is shown or saved by."""
        self.panel.record.set_text(run_id or "")


def grid_summary(grid: GridReducer, n_rows: int, of_rows: int | None = None) -> str:
    """What a finished plot shows: samples drawn, rows read (of the
    selection's `of_rows`, when the pass read only the rows that can reach
    the view), and samples left out because they fall outside the axis
    ranges (or, on a log axis, are not positive)."""
    text = f"{grid.n_samples:,} samples from {n_rows:,} rows"
    if of_rows is not None and of_rows != n_rows:
        text += f" of {of_rows:,} (the others lie outside the view)"
    if grid.n_outside:
        text += f"; {grid.n_outside:,} outside the axis ranges, left out"
    return text
