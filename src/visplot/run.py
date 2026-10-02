"""Running a plot request: the steps the command line (`cli/run_visplot.py`)
and the GUI (`visplot/gui/`) share, so both run exactly the same code.

- `check_request`: what can be checked without the file (plot names,
  units, ranges, option combinations).
- `open_file`: the file's row index, headers and tables, read once (the GUI
  keeps them while the file is open); the telescope's structural DUD entries
  of the AN table left out of its antennas.
- `prepare`: the units that depend on the file, the time system, the row
  selection, the quantity context, the streaming source and the figures.
- `run_locate`, `save_outputs`: the batch work, for the command line or a
  GUI "save" action.

A request that cannot run raises `RequestError`, whose message says why;
everything worth telling the user goes to a `report(level, text)` callback
("info" or "warning"), which the command line prints and the GUI lists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from functools import cached_property
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
from astropy.io import fits
from astropy.time import Time

from data_io.antenna_table import read_antenna_table, read_array_earth_location, read_time_reference
from data_io.astrometry import DEFAULT_UT1, fallback_message
from data_io.row_index import default_row_index_path, load_row_index
from data_io.row_selection import select_rows
from data_io.source_table import read_source_table
from data_io.timestamp_check import check_timestamps
from data_io.uvfits_group_params import DEFAULT_RAM_FRACTION_TO_USE, host_total_memory_bytes
from instruments.elevation_limits import elevation_limits_deg, known_elevation_limits_deg
from instruments.observatory_time_zones import OBSERVATORY_TIME_ZONES, observatory_time_zone
from instruments.structural_duds import without_structural_duds
from visplot.antenna_layout import antenna_layout
from visplot.fonts import font_file
from visplot.locate_csv import LocateCsvWriter
from visplot.iterations import Iteration, iteration_source, list_iterations, page_layout
from visplot.plot_panel import panel_facts
from visplot.plot_spec import PlotSpec, expand_plot_name
from visplot.quantities import QUANTITIES, QuantityContext, context_from_source_table, convert, local_time_zone, \
    utc_jd, utc_offset_text
from visplot.range_cache import RangeCache
from visplot.request import PlotRequest
from visplot.request_args import (
    DEFAULT_FIGURE_SIZE,
    DEFAULT_PAGE_FIGURE_SIZE,
    DEFAULT_PAGE_GRID,
    TABLE_PLOTS,
    parse_plot_names,
    resolve_antennas_arg,
    resolve_channels_arg,
    resolve_deg_range_arg,
    resolve_dpi_arg,
    resolve_figure_size_arg,
    resolve_ha_range_arg,
    resolve_klambda_range_arg,
    resolve_locate_box_arg,
    resolve_page_grid_arg,
    resolve_percentiles_arg,
    resolve_plain_range_arg,
    resolve_stokes_axis_selection,
    resolve_time_range_arg,
    resolve_uvdist_range_arg,
    validate_colorize_by,
)
from visplot.source_listing import source_listing
from visplot.stream import GridReducer, LocateReducer
from visplot.xy_figure import PageLayout, XYFigure, grid_summary, grids_summary
from visplot.xy_session import (
    PassProgress,
    XYSource,
    cached_ranges,
    describe_passes,
    pass_progress,
    plot_grids,
    range_pass_axes,
    range_pass_reads_data,
    resolve_axis_ranges,
    resolve_extents,
    source_in_views,
    stream_chunk_bytes,
)

GEOMETRY_QUANTITIES = {"ha", "az", "el", "pa"}
HIGHRES_MIN_DPI = 600  # the high-resolution PDF's least dpi

Report = Callable[[str, str], None]  # (level: "info" or "warning", text)
# Makes the on_chunk callback of one pass from its progress (e.g. printing every 10%).
ProgressFactory = Callable[[PassProgress], Callable[[int], bool]]


class RequestError(ValueError):
    """A request that cannot run as given; the message says why."""


class MissingIndexError(RequestError):
    """The FITS file has no row index yet."""


class Stopped(Exception):
    """A run stopped before the end because its progress callback asked it
    to (e.g. a GUI save's Stop)."""


def _silent(level: str, text: str) -> None:
    pass


@dataclass
class CheckedRequest:
    """A request's plots, checked without the file."""

    plot_names: list[str]
    plots_by_name: dict[str, list[PlotSpec]]  # streamed plots per name (table plots have none)
    locate_box: tuple | None
    figure_size: tuple[float, float]  # inches, of each streamed plot's figure
    page_figure_size: tuple[float, float] = (16.0, 11.0)  # inches, of a page of several plots (T26)
    page_grid: tuple[int, int] | None = None  # --page-grid as given; None: the default (`page_layout`)


def check_request(request: PlotRequest) -> CheckedRequest:
    """Everything about `request` that needs no file: plot names, ranges,
    percentiles, the locate box, scales, units (all but the visibility
    quantities', which depend on the file's BUNIT), dpi and figure size.
    Raises RequestError."""
    try:
        resolve_dpi_arg(request.dpi)
        figure_size = resolve_figure_size_arg(request.figure_size or DEFAULT_FIGURE_SIZE)
        page_figure_size = resolve_figure_size_arg(request.figure_size or DEFAULT_PAGE_FIGURE_SIZE)
        font_file(request.panel_font)  # installed in the venv (FontNotInstalled says how to install it)
        samplers = {"--every-nth": request.every_nth, "--every-nth-integration": request.every_nth_integration,
                    "--random-subset-n": request.random_subset_n}
        given = [name for name, value in samplers.items() if value is not None]
        if len(given) > 1:
            raise ValueError(f"give one of --every-nth, --every-nth-integration, --random-subset-n; got {', '.join(given)}")
        if not 50.0 <= request.density_top <= 100.0:
            raise ValueError(f"--density-top is a percentile from 50 to 100, got {request.density_top:g}")
        if request.baselines_with and not request.antennas:
            raise ValueError("--baselines-with needs --antennas: the baselines between one of --antennas and one of "
                             "--baselines-with")
        if request.baselines_with and request.correlation_type == "auto":
            raise ValueError("--baselines-with narrows cross-correlations; with --correlation-type auto, --antennas "
                             "alone chooses whose autocorrelations")
        page_grid = resolve_page_grid_arg(request.page_grid)
        for option, value in (("--page-grid", request.page_grid), ("--x-range-from", request.x_range_from),
                              ("--y-range-from", request.y_range_from)):
            if value is not None and not request.one_plot_per:
                raise ValueError(f"{option} lays out the plots of --one-plot-per; it needs --one-plot-per")
        if request.one_plot_per == "stokes" and request.colorize_by == "stokes":
            raise ValueError("with --one-plot-per stokes each plot holds one Stokes product, so coloring by Stokes "
                             "gives every plot one color; color by source, or leave --colorize-by out")
        if request.one_plot_per and request.locate:
            raise ValueError("--locate lists the samples of one plot; --one-plot-per makes many")
        if request.one_plot_per and not request.output_dir:
            raise ValueError("--one-plot-per saves its pages with --output-dir; the window's pages are not built yet")
        for name in given:
            if samplers[name] < 1:
                raise ValueError(f"{name} needs 1 or more, got {samplers[name]}")
        plot_names = parse_plot_names(request.plots)
        if request.colorize_by:
            validate_colorize_by(request.colorize_by)
        x_range = resolve_plain_range_arg(request.x_range)
        y_range = resolve_plain_range_arg(request.y_range)
        range_percentiles = resolve_percentiles_arg(request.range_percentiles)
        locate_box = resolve_locate_box_arg(request.locate)
        if locate_box is not None and not request.locate_csv:
            raise ValueError("--locate needs --locate-csv FILE")
        if request.scale_linear_width <= 0:
            raise ValueError("--scale-linear-width must be positive")
        for axis, scale, fixed in (("x", request.x_scale, x_range), ("y", request.y_scale, y_range)):
            if scale == "log" and fixed is not None and fixed[0] <= 0:
                raise ValueError(f"--{axis}-scale log shows positive values only; --{axis}-range starts at {fixed[0]:g}")
        style = PlotSpec(
            y="", x="", colorize_by=request.colorize_by, show_flagged=request.show_flagged,
            combine_flags=request.combine_flags, mirror=request.mirror,
            x_range=x_range, y_range=y_range, point_size=request.point_size, color=request.color, style=request.style,
            density_scale=request.density_scale, density_top=request.density_top,
            x_scale=request.x_scale, y_scale=request.y_scale, x_range_mode=request.x_range_mode,
            y_range_mode=request.y_range_mode, range_percentiles=range_percentiles,
            scale_linear_width=request.scale_linear_width, aspect=request.aspect,
            x_unit=request.x_unit, y_unit=request.y_unit,
        )
    except ValueError as err:
        raise RequestError(str(err)) from err
    plots_by_name = {}
    for name in plot_names:
        if name in TABLE_PLOTS:
            continue
        try:
            plots = expand_plot_name(name, style)
            for p in plots:
                for axis in ("x", "y"):
                    quantity = p.x if axis == "x" else p.y
                    if QUANTITIES[quantity].categorical and not p.axis_scale(axis).is_linear:
                        raise ValueError(f"--{axis}-scale applies to numeric axes; {quantity!r} is a category")
                    if request.time_format == "iso" and quantity == "time" and p.unit(axis).base == "lst":
                        raise ValueError("--time-format iso writes calendar dates; LST, a sidereal time, has none: "
                                         "choose dd/hh:mm:ss, dd:hh:mm:ss or hh:mm:ss")
                if p.aspect == "equal" and not (p.axis_scale("x").is_linear and p.axis_scale("y").is_linear):
                    raise ValueError("--aspect equal needs linear axes (--x-scale and --y-scale linear)")
        except ValueError as err:
            raise RequestError(f"plot {name!r}: {err}") from err
        plots_by_name[name] = plots
    if locate_box is not None:
        n_streamed = sum(len(p) for p in plots_by_name.values())
        if n_streamed != 1:
            raise RequestError(f"--locate needs exactly one streamed plot; --plots gives {n_streamed}")
    return CheckedRequest(plot_names, plots_by_name, locate_box, figure_size, page_figure_size, page_grid)


def with_elevation_limits(plot: PlotSpec, telescope: str | None) -> PlotSpec:
    """`plot` with the telescope's elevation limits (T47; the horizon and the
    zenith for a telescope with none known) on its elevation axis, in that
    axis's unit, drawn as dashed lines where the telescope's are known; the
    axis shows the sky, 0 to 90 degrees, unless a range is given (the user,
    2026-10-01). A plot without elevation unchanged."""
    for axis in ("y", "x"):
        if (plot.y if axis == "y" else plot.x) == "el":
            unit = plot.unit(axis).name
            low, high = (convert(v, "el", "deg", unit) for v in elevation_limits_deg(telescope))
            lines = ((axis, low), (axis, high)) if known_elevation_limits_deg(telescope) else ()
            sky = {} if getattr(plot, f"{axis}_range") is not None else {
                f"{axis}_range": (0.0, convert(90.0, "el", "deg", unit))}
            return replace(plot, limits=(axis, low, high), reference_lines=plot.reference_lines + lines, **sky)
    return plot


def _header_keyword(fits_path, keyword: str) -> str | None:
    with fits.open(fits_path) as hdul:
        value = str(hdul[0].header.get(keyword, "")).strip()
    return value or None


@dataclass
class OpenedFile:
    """A FITS file's row index, headers and tables, read once."""

    fits_path: str
    index: object  # RowIndex
    antennas: list  # the AN table's antennas, its structural DUD entries left out
    array_location: object  # EarthLocation
    source_table: dict
    telescope: str | None
    bunit: str | None
    time_reference: object  # TimeReference
    dud_antennas: list = field(default_factory=list)  # the structural DUD entries left out

    @property
    def reference_date_jd(self) -> float | None:
        date = self.time_reference.reference_date
        return float(Time(date, scale="utc").jd) if date else None

    @cached_property
    def timestamp_check(self):
        """The timestamps checked against the file's u, v, w (about 2 s; once
        per file)."""
        try:
            declared = self.time_reference.recorded_minus_utc_s
        except ValueError:
            declared = float("nan")
        return check_timestamps(self.index, self.antennas, self.source_table, declared)


def open_file(fits_path) -> OpenedFile:
    """Read what every request on this file needs, the AN table's structural
    DUD entries left out of its antennas (GMRT's C07 and S05, in the GSB
    file's table). Raises MissingIndexError when the row index has not been
    built."""
    idx_path = default_row_index_path(fits_path)
    if not idx_path.exists():
        raise MissingIndexError(f"no row index at {idx_path} -- build it first (the pipeline's build_index stage)")
    telescope = _header_keyword(fits_path, "TELESCOP")
    antennas = without_structural_duds(read_antenna_table(fits_path), telescope)
    return OpenedFile(
        fits_path=str(fits_path), index=load_row_index(idx_path), antennas=antennas.active_antennas,
        array_location=read_array_earth_location(fits_path), source_table=read_source_table(fits_path),
        telescope=telescope, bunit=_header_keyword(fits_path, "BUNIT"),
        time_reference=read_time_reference(fits_path), dud_antennas=antennas.dud_antennas,
    )


@dataclass
class PreparedRun:
    """A request ready to stream: its selection, context, source and figures."""

    request: PlotRequest
    file: OpenedFile
    plot_names: list[str]
    plots_by_name: dict[str, list[PlotSpec]]
    locate_box: tuple | None
    ctx: QuantityContext
    selection: object  # RowSelection
    source: XYSource
    xy_figures: dict[PlotSpec, XYFigure]
    cache: RangeCache | None
    cached: dict
    labels: tuple[str, str]
    figure_size: tuple[float, float] = (8.0, 7.0)  # inches, of each streamed plot's figure
    iterations: list[Iteration] = field(default_factory=list)  # with --one-plot-per (T26)
    page_figure_size: tuple[float, float] = (16.0, 11.0)  # inches, of a page of several plots
    page_grid: tuple[int, int] | None = None  # --page-grid as given
    record_id: str | None = None  # the run's provenance record (`set_record`)
    _figures: list | None = field(default=None, repr=False)

    @property
    def xy_plots(self) -> list[PlotSpec]:
        return [p for plots in self.plots_by_name.values() for p in plots]

    def set_record(self, run_id: str | None) -> None:
        """Name the provenance record of this run on every plot's panel (and
        on the pages' panels, made later)."""
        self.record_id = run_id
        for figure in self.xy_figures.values():
            figure.set_record(run_id)

    def figures(self) -> list[tuple[str, object]]:
        """Every figure, in the order the plots were named (table plots drawn
        from the file's tables, once)."""
        if self._figures is None:
            f, theme = self.file, self.request.plot_theme
            figures = []
            for name in self.plot_names:
                if name == "antenna-layout":
                    figures.append((name, antenna_layout(f.antennas, f.array_location, telescope=f.telescope,
                                                         source_path=f.fits_path, theme=theme,
                                                         left_out=[a.name for a in f.dud_antennas])))
                elif name == "source-listing":
                    figures.append((name, source_listing(f.source_table, theme=theme)))
                else:
                    figures += [(p.name, self.xy_figures[p]) for p in self.plots_by_name[name]]
            self._figures = figures
        return self._figures

    def iteration_plot(self, plot: PlotSpec, iteration: Iteration, fixed: dict[str, tuple] | None = None) -> PlotSpec:
        """`plot` of `iteration`, named for its files, with the axis ranges in
        `fixed` ({"x": range, "y": range}: a range from every plot) set."""
        ranges = {f"{axis}_range": value for axis, value in (fixed or {}).items()}
        return replace(plot, iteration=iteration, name=f"{plot.name}_{iteration.file_label}", **ranges)

    def page_figure(self, page: PageOfPlots) -> XYFigure:
        """The figure of `page`: one iteration's plot (a 1 x 1 page), or a
        grid of them; its title and panel describe the page's own part of
        the selection (the union of its iterations')."""
        index = self.file.index
        parts = [iteration_source(self.source, plot.iteration) for plot in page.plots]
        rows = np.unique(np.concatenate([part.row_indices for part in parts]))
        stokes = [label for label in self.ctx.stokes_labels if any(label in part.ctx.stokes_labels for part in parts)]
        ctx = replace(self.ctx, stokes_labels=tuple(stokes))
        present = set(np.unique(np.asarray(index.source_id)[rows]).tolist())
        sources = {sid: name for sid, name in self.selection.sources.items() if sid in present}
        facts = panel_facts(self.request, index, rows, (self.source.axis_selection or {}).get("FREQ"), stokes, sources,
                            ctx, stride_pairs=self.selection.stride_pairs, part=page.panel_text)
        by_source = page.plots[0].iteration.by == "source"  # a source's plot or page: named by its sources
        common = dict(telescope=self.file.telescope, source_path=self.request.fits_path, facts=facts,
                      panel_font=self.request.panel_font, theme=self.request.plot_theme,
                      time_format=self.request.time_format)
        if page.layout is None:
            figure = XYFigure(page.plots[0], parts[0].ctx, list(sources.values()), figsize=self.figure_size, **common)
        else:
            figure = XYFigure(page.plot, ctx, None if by_source else list(sources.values()),
                              figsize=self.page_figure_size, page=page.layout, **common)
        figure.set_record(self.record_id)
        return figure


def prepare(request: PlotRequest, opened: OpenedFile | None = None, report: Report = _silent,
            checked: CheckedRequest | None = None) -> PreparedRun:
    """Everything before the first pass: see the module docstring. Raises
    RequestError (MissingIndexError when `opened` is not given and the file
    has no index)."""
    checked = checked or check_request(request)
    opened = opened or open_file(request.fits_path)
    index = opened.index
    time_zone = request.time_zone or observatory_time_zone(opened.telescope)
    try:
        recorded_minus_utc_s = opened.time_reference.recorded_minus_utc_s
    except ValueError as err:
        raise RequestError(str(err)) from err

    # The units that depend on the file: BUNIT's, and local time's zone. Every plot then
    # names its units explicitly.
    header_ctx = QuantityContext(
        time_reference_jd=float(index.jd.min()), bunit=opened.bunit, time_zone=time_zone,
        reference_date_jd=opened.reference_date_jd, time_system=opened.time_reference.time_system,
        recorded_minus_utc_s=recorded_minus_utc_s,
    )
    plots_by_name = {}
    for name, plots in checked.plots_by_name.items():
        try:
            for p in plots:
                p.check_units(header_ctx)
                if any(p.unit(axis, header_ctx).base == "local" for axis in ("x", "y")):
                    _check_time_zone(time_zone, opened.telescope, header_ctx.time_reference_jd)
        except ValueError as err:
            raise RequestError(f"plot {name!r}: {err}") from err
        plots_by_name[name] = [with_elevation_limits(p.with_units(header_ctx), opened.telescope) for p in plots]
    xy_plots = [p for plots in plots_by_name.values() for p in plots]
    if _reads_utc(xy_plots, header_ctx, request):
        report("info", f"time system: {opened.time_reference.describe()}")
        check = opened.timestamp_check
        report("info" if check.agrees else "warning", check.summary())

    selected = select(request, opened, recorded_minus_utc_s)
    selection, axis_selection, stokes_labels = selected.selection, selected.axis_selection, selected.stokes_labels
    if xy_plots and not selection.n_rows:
        raise RequestError(empty_selection_message(request, index, selected.kwargs))
    report("info", f"selected {selection.n_rows:,} rows; sources present: {selection.sources}")
    if selection.stride_pairs is not None and selection.stride_pairs[0] < selection.stride_pairs[1]:
        kept, available = selection.stride_pairs
        what = "baselines" if request.correlation_type == "cross" else "antenna pairs"
        per_integration = int(np.median(np.diff(index.integration_boundaries)))
        report("warning", f"--every-nth {request.every_nth} keeps {kept:,} of the selection's {available:,} {what}: "
                          f"rows are in baseline order within each integration ({per_integration:,} rows here), so "
                          f"a row stride keeps only some {what}; --every-nth-integration {request.every_nth} keeps "
                          f"every one")

    time_reference_jd = float(index.jd[selection.row_indices].min()) if selection.n_rows else float(index.jd.min())
    ctx = context_from_source_table(
        time_reference_jd, opened.source_table, opened.array_location, opened.bunit, stokes_labels,
        antenna_names={a.station_number: a.name for a in opened.antennas}, time_zone=time_zone,
        reference_date_jd=opened.reference_date_jd, time_system=opened.time_reference.time_system,
        recorded_minus_utc_s=recorded_minus_utc_s,
    )
    source = XYSource(request.fits_path, index, selection.row_indices, axis_selection, ctx,
                      stream_chunk_bytes(), threads=max(1, request.threads))
    sources_present = list(selection.sources.values())
    channel_indices = (axis_selection or {}).get("FREQ")
    facts = panel_facts(request, index, selection.row_indices, channel_indices, stokes_labels, selection.sources, ctx,
                        stride_pairs=selection.stride_pairs)
    xy_figures = {p: XYFigure(p, ctx, sources_present, opened.telescope, request.fits_path,
                              figsize=checked.figure_size, facts=facts, panel_font=request.panel_font,
                              theme=request.plot_theme, time_format=request.time_format)
                  for p in xy_plots}
    cache = RangeCache(request.cache_dir) if request.cache_dir else None
    if cache is not None:
        for stale in cache.remove_stale_partials():
            report("info", f"removed {stale} (left by an earlier run that stopped while writing)")
    cached = cached_ranges(source, xy_plots, cache)
    iterations = list_iterations(request.one_plot_per, index, selection.row_indices, stokes_labels, ctx) \
        if request.one_plot_per and xy_plots else []
    if iterations:  # the save reports its passes, batch by batch
        for line in describe_passes(source, xy_plots, with_passes=False):
            report("info", line)
        rows, cols = page_layout(len(iterations), checked.page_grid, DEFAULT_PAGE_GRID)
        n_pages = math.ceil(len(iterations) / (rows * cols))
        shown = ", ".join(it.label for it in iterations[:6]) + (", …" if len(iterations) > 6 else "")
        report("info", f"{len(iterations):,} plot{'s' if len(iterations) != 1 else ''}, one per "
                       f"{request.one_plot_per} ({shown}), on {n_pages:,} page{'s' if n_pages != 1 else ''} of "
                       f"{rows} x {cols}")
    elif xy_plots and (checked.locate_box is None or request.output_dir):
        for line in describe_passes(source, xy_plots, cached):
            report("info", line)
    elif checked.locate_box is not None:
        reads = (f"reads visibility data, {source.n_rows * source.row_bytes / 1e9:.1f} GB"
                 if xy_plots[0].needs_data else "row metadata only, no visibility data read")
        report("info", f"one pass: locate the samples of {xy_plots[0].title} in the box ({reads})")
    _report_ut1(index, selection, xy_plots, xy_figures, ctx, selected.geometry_filters, report)
    _report_local_time(index, selection, xy_plots, xy_figures, ctx, report)
    n_passes = 2 if [k for k in range_pass_axes(xy_plots) if k not in cached] else 1
    labels = (f"pass 1 of {n_passes}: finding data ranges", f"pass {n_passes} of {n_passes}: drawing")
    return PreparedRun(request, opened, checked.plot_names, plots_by_name, checked.locate_box, ctx, selection,
                       source, xy_figures, cache, cached, labels, figure_size=checked.figure_size,
                       iterations=iterations, page_figure_size=checked.page_figure_size, page_grid=checked.page_grid)


@dataclass
class Selected:
    """The rows and axis indices a request selects."""

    selection: object  # RowSelection
    axis_selection: dict | None  # FREQ / STOKES indices; None: every channel and Stokes
    stokes_labels: list[str]  # of the selected Stokes
    geometry_filters: bool  # hour angle, Az/El or parallactic angle filters given
    kwargs: dict  # the `select_rows` arguments


def select(request: PlotRequest, opened: OpenedFile, recorded_minus_utc_s: float) -> Selected:
    """The request's row selection (`select_rows`) and channel/Stokes
    selection. Raises RequestError for a filter that does not parse or
    match."""
    index = opened.index
    geometry_filters = any(v is not None for v in (request.ha_range, request.az_range, request.el_range,
                                                    request.pa_range))
    try:
        select_kwargs = dict(
            sources=[s.strip() for s in request.sources.split(",")] if request.sources else None,
            correlation_type=request.correlation_type,
            antennas=resolve_antennas_arg(request.antennas, opened.antennas),
            baselines_with=resolve_antennas_arg(request.baselines_with, opened.antennas),
            exclude_antennas=resolve_antennas_arg(request.exclude_antennas, opened.antennas),
            jd_range=resolve_time_range_arg(request.time_range, float(index.jd.min()), recorded_minus_utc_s),
            uvdist_range_m=resolve_uvdist_range_arg(request.uvdist_range),
            u_range_klambda=resolve_klambda_range_arg(request.u_range_klambda),
            v_range_klambda=resolve_klambda_range_arg(request.v_range_klambda),
            w_range_klambda=resolve_klambda_range_arg(request.w_range_klambda),
            uvdist_range_klambda=resolve_klambda_range_arg(request.uvdist_range_klambda),
            ha_range_hours=resolve_ha_range_arg(request.ha_range),
            az_range_deg=resolve_deg_range_arg(request.az_range),
            el_range_deg=resolve_deg_range_arg(request.el_range),
            parallactic_angle_range_deg=resolve_deg_range_arg(request.pa_range),
            every_nth=request.every_nth,
            every_nth_integration=request.every_nth_integration,
            random_subset_n=request.random_subset_n,
            random_seed=request.random_seed,
        )
        if geometry_filters:
            select_kwargs.update(source_table=opened.source_table, array_location=opened.array_location,
                                 recorded_minus_utc_s=recorded_minus_utc_s)
        # The kλ row filters evaluate each row at the edges of a frequency span:
        # that span must be the band of the selected channels.
        channel_indices = resolve_channels_arg(request.channels, index.chan_freqs_hz)
        if channel_indices is not None:
            selected_freqs_hz = np.asarray(index.chan_freqs_hz)[channel_indices]
            select_kwargs["freq_range_hz"] = (float(selected_freqs_hz.min()), float(selected_freqs_hz.max()))
        stokes_indices = resolve_stokes_axis_selection(request.stokes, index.stokes_labels)
        selection = select_rows(index, **select_kwargs)
    except ValueError as err:
        raise RequestError(str(err)) from err
    axis_selection = {}
    if channel_indices is not None:
        axis_selection["FREQ"] = np.array(channel_indices)
    if stokes_indices is not None:
        axis_selection["STOKES"] = np.array(stokes_indices)
    stokes_labels = list(index.stokes_labels or [])
    if stokes_indices is not None:
        stokes_labels = [stokes_labels[i] for i in stokes_indices]
    return Selected(selection, axis_selection or None, stokes_labels, geometry_filters, select_kwargs)


# The row filters in the command line's order: each option and the `select_rows` arguments it gives.
_ROW_FILTERS = (
    ("sources", ("sources",)), ("correlation_type", ("correlation_type",)),
    ("antennas", ("antennas", "baselines_with")), ("exclude_antennas", ("exclude_antennas",)),
    ("time_range", ("jd_range",)), ("uvdist_range", ("uvdist_range_m",)), ("u_range_klambda", ("u_range_klambda",)),
    ("v_range_klambda", ("v_range_klambda",)), ("w_range_klambda", ("w_range_klambda",)),
    ("uvdist_range_klambda", ("uvdist_range_klambda",)), ("ha_range", ("ha_range_hours",)),
    ("az_range", ("az_range_deg",)), ("el_range", ("el_range_deg",)), ("pa_range", ("parallactic_angle_range_deg",)),
)
# What the filters are evaluated with, and never select rows by themselves.
_FILTER_CONTEXT = ("freq_range_hz", "source_table", "array_location", "recorded_minus_utc_s")


def empty_selection_message(request: PlotRequest, index, kwargs: dict) -> str:
    """Why `kwargs` (a request's `select_rows` arguments) select no rows: the
    request's row filters applied one at a time, in the command line's
    order, from every row of the file; the first after which none are left
    is named, with the number of rows it had (a correlation type the file
    has no rows of, as that)."""
    applied = {"correlation_type": "both", **{k: kwargs[k] for k in _FILTER_CONTEXT if k in kwargs}}
    n_before = index.gcount
    for dest, keys in _ROW_FILTERS:
        if all(kwargs.get(k) is None for k in keys):
            continue
        applied.update({k: kwargs[k] for k in keys})
        n = select_rows(index, **applied).n_rows
        if n:
            n_before = n
            continue
        option = f"--{dest.replace('_', '-')} {getattr(request, dest)}"
        if dest == "antennas" and request.baselines_with:
            option += f" --baselines-with {request.baselines_with}"
        if dest == "correlation_type" and not select_rows(index, correlation_type=request.correlation_type).n_rows:
            kind = "autocorrelation" if request.correlation_type == "auto" else "cross-correlation"
            return f"no rows selected: this file has no {kind} rows ({option})"
        return f"no rows selected: {option} leaves none of the {n_before:,} rows selected before it"
    return "no rows selected"


@dataclass(frozen=True)
class SelectionCount:
    rows: int
    samples_per_row: int  # visibility samples per row under the channel/Stokes selection
    gigabytes: float  # visibility data read when a plot needs it (whole rows)

    @property
    def samples(self) -> int:
        return self.rows * self.samples_per_row


def count_selection(request: PlotRequest, opened: OpenedFile) -> SelectionCount:
    """How much a request selects, without reading visibility data (the
    same `select` a plot runs). Raises RequestError."""
    try:
        recorded_minus_utc_s = opened.time_reference.recorded_minus_utc_s
    except ValueError as err:
        raise RequestError(str(err)) from err
    selected = select(request, opened, recorded_minus_utc_s)
    if not selected.selection.n_rows and check_request(request).plots_by_name:  # streamed plots need rows
        raise RequestError(empty_selection_message(request, opened.index, selected.kwargs))
    source = XYSource(request.fits_path, opened.index, selected.selection.row_indices, selected.axis_selection,
                      ctx=None, chunk_bytes=0)
    return SelectionCount(source.n_rows, source.samples_per_row, source.n_rows * source.row_bytes / 1e9)


def _uses_unit_base(plot, ctx, *bases: str) -> bool:
    return any(plot.unit(axis, ctx).base in bases for axis in ("x", "y"))


def _reads_utc(plots, ctx, request: PlotRequest) -> bool:
    """Whether this run reads timestamps as UTC: a geometry quantity, a UTC,
    local or LST axis, a geometry filter, or an absolute --time-range."""
    for p in plots:
        if GEOMETRY_QUANTITIES.intersection(p.quantities) or _uses_unit_base(p, ctx, "utc", "local", "lst"):
            return True
    if any(v is not None for v in (request.ha_range, request.az_range, request.el_range, request.pa_range)):
        return True
    spec = (request.time_range or "").strip().lower()
    return "/" in spec or spec.endswith("jd")


def _check_time_zone(time_zone, telescope, jd) -> None:
    if not time_zone:
        known = ", ".join(f"{k}: {v}" for k, v in OBSERVATORY_TIME_ZONES.items())
        raise ValueError(f"local time needs the observatory's time zone; TELESCOP {telescope!r} is not among "
                         f"the known ones ({known}): give --time-zone, an IANA name such as Asia/Kolkata")
    local_time_zone(time_zone, jd)  # raises for a zone the system does not know


def _report_local_time(index, selection, xy_plots, xy_figures, ctx, report: Report) -> None:
    """Local time is drawn at the zone's UTC offset at the first selected
    integration; if the offset differs at the last (a daylight-saving
    change), say so, before any pass and on the plot."""
    local_plots = [p for p in xy_plots if _uses_unit_base(p, ctx, "local")]
    if not local_plots:
        return
    jd = index.jd[selection.row_indices] if selection.n_rows else index.jd
    start, abbreviation = local_time_zone(ctx.time_zone, float(utc_jd(ctx, jd.min())))
    end, _ = local_time_zone(ctx.time_zone, float(utc_jd(ctx, jd.max())))
    report("info", f"local time: {ctx.time_zone} ({abbreviation}, {utc_offset_text(start)})")
    if end != start:
        text = (f"{ctx.time_zone} changes from {utc_offset_text(start)} to {utc_offset_text(end)} during the "
                f"selection; local time is drawn at {utc_offset_text(start)} throughout")
        report("warning", text)
        for p in local_plots:
            xy_figures[p].set_note(text)


def _report_ut1(index, selection, xy_plots, xy_figures, ctx, geometry_filters: bool, report: Report) -> None:
    """Where UT1 - UTC comes from for this selection's dates, before any
    pass; a fallback to UT1 = UTC is also noted on every geometry plot
    (hour angle, azimuth, elevation, parallactic angle, LST)."""
    geometry_plots = [p for p in xy_plots
                      if GEOMETRY_QUANTITIES.intersection(p.quantities) or _uses_unit_base(p, ctx, "lst")]
    if not geometry_plots and not geometry_filters:
        return
    jd = index.jd[selection.row_indices] if selection.n_rows else index.jd
    DEFAULT_UT1.ut1_minus_utc_s(utc_jd(ctx, [float(jd.min()), float(jd.max())]))
    report("info", f"UT1 - UTC (for hour angle, azimuth, elevation, parallactic angle, LST): "
                   f"{', '.join(sorted(DEFAULT_UT1.sources_used))}")
    if DEFAULT_UT1.fallback_used:
        report("warning", fallback_message())
        for p in geometry_plots:
            xy_figures[p].set_note("UT1 = UTC assumed: hour angle and LST may be off by up to 0.9 s of time")


def _no_progress(progress: PassProgress) -> Callable[[int], bool]:
    return lambda rows_done: True


def run_locate(run: PreparedRun, report: Report = _silent, progress: ProgressFactory = _no_progress,
               record: str | None = None) -> Path | None:
    """Write every sample of the run's one streamed plot inside its locate
    box to the request's --locate-csv, whose header names the command and
    `record` (the run's provenance record, when given). Returns the path
    written, or None if the pass stopped early."""
    plot = run.xy_plots[0]
    box = run.locate_box
    locate = LocateReducer(plot, box[0], box[1], limit=0)
    writer = LocateCsvWriter(run.request.locate_csv, locate, run.source.ctx, fits_path=run.request.fits_path,
                             command=run.request.command_line(), record=record)
    locate.sink = writer
    source = source_in_views(run.source, {plot: box}, plot.needs_data)  # the rows that can reach the box
    on_chunk = progress(pass_progress(source, "locating samples", plot.needs_data))
    completed = source.stream([locate], read_data=plot.needs_data, on_chunk=on_chunk)
    written = writer.close(locate, completed)
    n = len(locate.by_baseline)
    report("info", f"located {locate.n_found:,} samples on {n:,} baseline{'s' if n != 1 else ''}; wrote {written}")
    return written


def highres_dpi(dpi: int) -> int:
    """The high-resolution PDF's dpi for PNGs at `dpi`: the smallest whole
    multiple of `dpi` that is at least HIGHRES_MIN_DPI (600 for 150), so the
    `dpi` grid is an exact reduction of it."""
    return dpi * max(1, math.ceil(HIGHRES_MIN_DPI / dpi))


def save_outputs(run: PreparedRun, report: Report = _silent, progress: ProgressFactory = _no_progress) -> list[Path]:
    """The range pass (when needed) and one plotting pass at `highres_dpi`,
    then one PNG per plot at the request's --dpi (an exact reduction) and the
    low- and high-resolution PDFs, in the request's --output-dir. Returns the
    paths written. Raises Stopped, with nothing written, when a pass's
    `progress` callback stops it. With --one-plot-per, `_save_pages`."""
    if run.iterations:
        return _save_pages(run, report, progress)
    request, source, xy_plots, xy_figures = run.request, run.source, run.xy_plots, run.xy_figures
    extents = {}
    if xy_plots:
        on_chunk = progress(pass_progress(source, run.labels[0], range_pass_reads_data(xy_plots, run.cached)))
        extents = resolve_extents(source, xy_plots, on_chunk=on_chunk, cache=run.cache)
        if extents is None:
            raise Stopped("stopped while finding the data ranges; nothing saved")
    lowres_dpi, highres = request.dpi, highres_dpi(request.dpi)
    factor = highres // lowres_dpi
    shapes = {}
    for p in xy_plots:
        extents[p] = xy_figures[p].set_view(*extents[p])  # the limits after the aspect applies
        h, w = xy_figures[p].grid_shape(lowres_dpi)
        # density counts at the PNG's pixels (a count per pixel means the pixel seen; T21), points at highres
        shapes[p] = (h, w) if p.style == "density" else (h * factor, w * factor)
    grids = {}
    drawn = source
    if xy_plots:
        read_data = any(p.needs_data for p in xy_plots)
        drawn = source_in_views(source, {p: extents[p] for p in xy_plots}, read_data)  # rows that can reach a view
        on_chunk = progress(pass_progress(drawn, run.labels[1], read_data))
        grids, completed = plot_grids(drawn, xy_plots, extents, shapes, on_chunk=on_chunk)
        if not completed:
            raise Stopped("stopped while drawing; nothing saved")

    def _mpl_figure(item):
        return item.fig if isinstance(item, XYFigure) else item

    from matplotlib.backends.backend_pdf import PdfPages

    output_dir = Path(request.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in xy_figures.values():
        item.set_status(grid_summary(grids[item.plot], drawn.n_rows, source.n_rows))
    written = []
    lowres_path = output_dir / f"{request.output_prefix}_lowres.pdf"
    with PdfPages(lowres_path) as pdf:
        for name, item in run.figures():
            if isinstance(item, XYFigure):
                item.show(grids[item.plot], display_dpi=lowres_dpi,
                          downsample=1 if item.plot.style == "density" else factor)
                if item.limits_warning:
                    report("warning", f"{name}: {item.limits_warning}")
            png_path = output_dir / f"{request.output_prefix}_{name}.png"
            _mpl_figure(item).savefig(png_path, dpi=lowres_dpi)
            pdf.savefig(_mpl_figure(item), dpi=lowres_dpi)
            report("info", f"saved {png_path}")
            written.append(png_path)
    report("info", f"saved {lowres_path}")
    written.append(lowres_path)
    if not request.no_highres_pdf:
        highres_path = output_dir / f"{request.output_prefix}_highres.pdf"
        with PdfPages(highres_path) as pdf:
            for _, item in run.figures():
                if isinstance(item, XYFigure):
                    item.show(grids[item.plot], display_dpi=highres)
                pdf.savefig(_mpl_figure(item), dpi=highres)
        report("info", f"saved {highres_path}")
        written.append(highres_path)
    return written


def _category_count(run: PreparedRun, plot: PlotSpec) -> int:
    """How many categories `plot` colors (1 without --colorize-by)."""
    if plot.colorize_by == "stokes":
        return max(1, len(run.ctx.stokes_labels))
    if plot.colorize_by == "source":
        return max(1, len(run.selection.sources))
    return 1


def grid_bytes(plot: PlotSpec, shape: tuple[int, int], n_categories: int = 1) -> int:
    """Memory of `plot`'s `GridReducer` at `shape`: an int16 per pixel (three
    grids with elevation limits: the points, those below and those above),
    or for a density plot a uint32 count per pixel per category (and one for
    the flagged samples shown)."""
    pixels = shape[0] * shape[1]
    if plot.style == "density":
        return pixels * 4 * (max(1, n_categories) + (1 if plot.show_flagged else 0))
    return pixels * 2 * (3 if plot.limits else 1)


def _size_text(n_bytes: float) -> str:
    """e.g. "42 MB", "8.0 GB"."""
    return f"{n_bytes / 1e9:.1f} GB" if n_bytes >= 1e9 else f"{n_bytes / 1e6:.0f} MB"


def page_batches(pages: list, page_bytes: int, budget_bytes: int) -> list[list]:
    """`pages` in batches whose pixel grids (`page_bytes` per page) fit
    `budget_bytes`, at least one page each, in order."""
    per_batch = max(1, budget_bytes // max(1, page_bytes))
    return [pages[i:i + per_batch] for i in range(0, len(pages), per_batch)]


@dataclass
class PageOfPlots:
    """One saved page of iterations' plots (T26): `plot`, the streamed plot
    its iterations share; their `plots`; the grid's `layout` (None: one
    iteration's plot, a 1 x 1 page); the page's `name` in filenames; and
    what it shows, for its panel's Selection line (`panel_text`)."""

    plot: PlotSpec
    plots: list[PlotSpec]
    layout: PageLayout | None
    name: str
    panel_text: str


_PLURALS = {"baseline": "baselines", "antenna": "antennas", "source": "sources", "stokes": "Stokes"}


def plan_pages(run: PreparedRun, grid: tuple[int, int], fixed: dict[PlotSpec, dict],
               shared: dict[PlotSpec, tuple[bool, bool]]) -> list[list[PageOfPlots]]:
    """The pages of a run with --one-plot-per, `grid` (rows, columns) of
    iterations to a page, in groups of one page per streamed plot (the
    plots of the same iterations, drawn and saved together); each
    iteration's plot with the ranges in `fixed` set (`iteration_plot`), and
    a grid's axes `shared` (x, y) per streamed plot."""
    iterations = run.iterations
    per_page = grid[0] * grid[1]
    n_pages = math.ceil(len(iterations) / per_page)
    digits = max(2, len(str(n_pages)))
    plural = _PLURALS[iterations[0].by]
    groups = []
    for k in range(n_pages):
        chunk = iterations[k * per_page:(k + 1) * per_page]
        if per_page == 1:
            title_part, panel_text = None, chunk[0].text
        else:
            first = k * per_page + 1
            title_part = (f"{plural} {first:,}-{first + len(chunk) - 1:,} of {len(iterations):,}" if n_pages > 1
                          else f"{len(iterations):,} {plural}")
            panel_text = f"{title_part}: {chunk[0].label}" + (f" to {chunk[-1].label}" if len(chunk) > 1 else "")
        group = []
        for plot in run.xy_plots:
            plots = [run.iteration_plot(plot, iteration, fixed[plot]) for iteration in chunk]
            if per_page == 1:
                group.append(PageOfPlots(plot, plots, None, plots[0].name, panel_text))
            else:
                layout = PageLayout(tuple(plots), grid, shared[plot], title_part)
                group.append(PageOfPlots(plot, plots, layout, f"{plot.name}_page{k + 1:0{digits}d}", panel_text))
        groups.append(group)
    return groups


def _rows_of(source: XYSource, iterations) -> np.ndarray:
    """The rows of `source` the `iterations` hold, each once."""
    return np.unique(np.concatenate([iteration_source(source, it).row_indices for it in iterations]))


def draw_pages(run: PreparedRun, report: Report = _silent, progress: ProgressFactory = _no_progress,
               lowres_dpi: int | None = None) -> Iterator[tuple[PageOfPlots, XYFigure, dict[PlotSpec, GridReducer]]]:
    """Every page of a run with --one-plot-per drawn for saving (T26), in
    order: (page, figure, {plot: grid}), the figure's status set, its views
    the plots' ranges, the grids at the saved resolution (`save_outputs`).

    The iterations go --page-grid to a page (`page_layout`: 5 x 6, or the
    smallest grid within it that holds fewer). Each plot's range, without
    --x-range/--y-range, comes from its own data or from every plot's
    (--x-range-from, --y-range-from; default every plot's on a grid, its own
    on 1 x 1 pages), the latter from one pass over the whole selection. The
    pages are drawn in batches whose pixel grids fit the memory budget
    (`page_batches`), each a range pass for the plots' own ranges (when
    any) and a drawing pass, over the rows its iterations hold. Raises
    Stopped when a pass is stopped."""
    request, source, iterations = run.request, run.source, run.iterations
    lowres_dpi = lowres_dpi or request.dpi
    factor = highres_dpi(lowres_dpi) // lowres_dpi
    grid = page_layout(len(iterations), run.page_grid, DEFAULT_PAGE_GRID)

    def range_from(axis: str) -> str:
        return getattr(request, f"{axis}_range_from") or ("all" if grid[0] * grid[1] > 1 else "each")

    def given(plot: PlotSpec, axis: str):
        return plot.x_range if axis == "x" else plot.y_range

    fixed = {p: {} for p in run.xy_plots}
    from_all = [(p, axis) for p in run.xy_plots for axis in ("x", "y")
                if range_from(axis) == "all" and given(p, axis) is None]
    if from_all:
        read_data = any(QUANTITIES[p.x if axis == "x" else p.y].needs_data for p, axis in from_all)
        on_chunk = progress(pass_progress(source, "finding the range of every plot", read_data))
        ranged = resolve_axis_ranges(source, from_all, on_chunk=on_chunk, cache=run.cache)
        if ranged is None:
            raise Stopped("stopped while finding the range of every plot; nothing saved")
        for (p, axis), extent in ranged.items():
            fixed[p][axis] = extent
    shared = {p: tuple(range_from(axis) == "all" or given(p, axis) is not None for axis in ("x", "y"))
              for p in run.xy_plots}
    groups = plan_pages(run, grid, fixed, shared)

    def save_shapes(page: PageOfPlots, figure: XYFigure) -> dict[PlotSpec, tuple[int, int]]:
        # density counts at the PNG's pixels (a count per pixel means the pixel seen; T21), points at highres
        shapes = ({page.plots[0]: figure.grid_shape(lowres_dpi)} if page.layout is None
                  else figure.grid_shapes(lowres_dpi))
        return {plot: (h, w) if plot.style == "density" else (h * factor, w * factor)
                for plot, (h, w) in shapes.items()}

    group_bytes = 0
    for page in groups[0]:
        figure = run.page_figure(page)
        group_bytes += sum(grid_bytes(plot, shape, _category_count(run, plot))
                           for plot, shape in save_shapes(page, figure).items())
        figure.fig.clear()
    budget = int(host_total_memory_bytes() * DEFAULT_RAM_FRACTION_TO_USE)
    batches = page_batches(groups, group_bytes, budget)
    each = f" for each of {len(run.xy_plots)} plots" if len(run.xy_plots) > 1 else ""
    report("info", f"{len(groups):,} page{'s' if len(groups) != 1 else ''} of {grid[0]} x {grid[1]}{each}, drawn "
                   f"in {len(batches)} batch{'es' if len(batches) != 1 else ''} of up to {len(batches[0]):,} "
                   f"({_size_text(len(batches[0]) * group_bytes)} of pixel grids each, within {_size_text(budget)}, "
                   f"{DEFAULT_RAM_FRACTION_TO_USE:.0%} of this host's memory)")

    done = 0
    for batch in batches:
        label = f"pages {done + 1:,}-{done + len(batch):,} of {len(groups):,}" if len(batches) > 1 else "pages"
        done += len(batch)
        pages = [page for group in batch for page in group]
        figures = [run.page_figure(page) for page in pages]
        plots = [plot for page in pages for plot in page.plots]
        part = source.subset(_rows_of(source, dict.fromkeys(plot.iteration for plot in plots)))
        on_chunk = progress(pass_progress(part, f"{label}: finding data ranges", range_pass_reads_data(plots)))
        extents = resolve_extents(part, plots, on_chunk=on_chunk, cache=run.cache)
        if extents is None:
            raise Stopped(f"stopped while finding the data ranges of {label}")
        shapes = {}
        for page, figure in zip(pages, figures):  # the limits after the aspect applies
            if page.layout is None:
                extents[page.plots[0]] = figure.set_view(*extents[page.plots[0]])
            else:
                extents.update(figure.set_views({plot: extents[plot] for plot in page.plots}))
            shapes.update(save_shapes(page, figure))
        read_data = any(plot.needs_data for plot in plots)
        drawn = source_in_views(part, {plot: extents[plot] for plot in plots}, read_data)
        on_chunk = progress(pass_progress(drawn, f"{label}: drawing", read_data))
        grids, completed = plot_grids(drawn, plots, extents, shapes, on_chunk=on_chunk)
        if not completed:
            raise Stopped(f"stopped while drawing {label}")
        for page, figure in zip(pages, figures):
            page_grids = {plot: grids.pop(plot) for plot in page.plots}
            its = [plot.iteration for plot in page.plots]
            figure.set_status(grids_summary(list(page_grids.values()), len(_rows_of(drawn, its)),
                                            len(_rows_of(part, its))))
            yield page, figure, page_grids


def _save_pages(run: PreparedRun, report: Report, progress: ProgressFactory) -> list[Path]:
    """`save_outputs` with --one-plot-per (T26): a PNG per page of each
    streamed plot, and the PDFs with the table plots, then every page in
    turn (`draw_pages`), each page's figure cleared once saved. Raises
    Stopped when a pass is stopped; the pages saved before it are kept."""
    from contextlib import ExitStack

    from matplotlib.backends.backend_pdf import PdfPages

    request = run.request
    lowres_dpi, highres = request.dpi, highres_dpi(request.dpi)
    factor = highres // lowres_dpi
    output_dir = Path(request.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lowres_path = output_dir / f"{request.output_prefix}_lowres.pdf"
    highres_path = None if request.no_highres_pdf else output_dir / f"{request.output_prefix}_highres.pdf"
    written = []
    with ExitStack() as stack:
        lowres_pdf = stack.enter_context(PdfPages(lowres_path))
        highres_pdf = stack.enter_context(PdfPages(highres_path)) if highres_path else None
        for name, item in run.figures():
            if isinstance(item, XYFigure):
                continue  # drawn page by page below
            png_path = output_dir / f"{request.output_prefix}_{name}.png"
            item.savefig(png_path, dpi=lowres_dpi)
            lowres_pdf.savefig(item, dpi=lowres_dpi)
            if highres_pdf is not None:
                highres_pdf.savefig(item, dpi=highres)
            report("info", f"saved {png_path}")
            written.append(png_path)
        for page, figure, grids in draw_pages(run, report, progress):
            figure.show_page(grids, display_dpi=lowres_dpi, downsample=1 if page.plot.style == "density" else factor)
            if figure.limits_warning:
                report("warning", f"{page.name}: {figure.limits_warning}")
            png_path = output_dir / f"{request.output_prefix}_{page.name}.png"
            figure.fig.savefig(png_path, dpi=lowres_dpi)
            lowres_pdf.savefig(figure.fig, dpi=lowres_dpi)
            if highres_pdf is not None:
                figure.show_page(grids, display_dpi=highres)
                highres_pdf.savefig(figure.fig, dpi=highres)
            figure.fig.clear()  # its images released before the next page's
            report("info", f"saved {png_path}")
            written.append(png_path)
    report("info", f"saved {lowres_path}")
    written.append(lowres_path)
    if highres_path:
        report("info", f"saved {highres_path}")
        written.append(highres_path)
    return written
