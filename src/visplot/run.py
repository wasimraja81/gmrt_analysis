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
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Callable

import numpy as np
from astropy.io import fits
from astropy.time import Time

from data_io.antenna_table import read_antenna_table, read_array_earth_location, read_time_reference
from data_io.astrometry import DEFAULT_UT1, fallback_message
from data_io.row_index import default_row_index_path, load_row_index
from data_io.row_selection import select_rows
from data_io.source_table import read_source_table
from data_io.timestamp_check import check_timestamps
from instruments.observatory_time_zones import OBSERVATORY_TIME_ZONES, observatory_time_zone
from instruments.structural_duds import without_structural_duds
from visplot.antenna_layout import antenna_layout
from visplot.fonts import font_file
from visplot.locate_csv import LocateCsvWriter
from visplot.plot_panel import panel_facts
from visplot.plot_spec import PlotSpec, expand_plot_name
from visplot.quantities import QUANTITIES, QuantityContext, context_from_source_table, local_time_zone, utc_jd, \
    utc_offset_text
from visplot.range_cache import RangeCache
from visplot.request import PlotRequest
from visplot.request_args import (
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
    resolve_percentiles_arg,
    resolve_plain_range_arg,
    resolve_stokes_axis_selection,
    resolve_time_range_arg,
    resolve_uvdist_range_arg,
    validate_colorize_by,
)
from visplot.source_listing import source_listing
from visplot.stream import LocateReducer
from visplot.xy_figure import XYFigure, grid_summary
from visplot.xy_session import (
    PassProgress,
    XYSource,
    cached_ranges,
    describe_passes,
    pass_progress,
    plot_grids,
    range_pass_axes,
    range_pass_reads_data,
    resolve_extents,
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


def check_request(request: PlotRequest) -> CheckedRequest:
    """Everything about `request` that needs no file: plot names, ranges,
    percentiles, the locate box, scales, units (all but the visibility
    quantities', which depend on the file's BUNIT), dpi and figure size.
    Raises RequestError."""
    try:
        resolve_dpi_arg(request.dpi)
        figure_size = resolve_figure_size_arg(request.figure_size)
        font_file(request.panel_font)  # installed in the venv (FontNotInstalled says how to install it)
        samplers = {"--every-nth": request.every_nth, "--every-nth-integration": request.every_nth_integration,
                    "--random-subset-n": request.random_subset_n}
        given = [name for name, value in samplers.items() if value is not None]
        if len(given) > 1:
            raise ValueError(f"give one of --every-nth, --every-nth-integration, --random-subset-n; got {', '.join(given)}")
        if request.baselines_with and not request.antennas:
            raise ValueError("--baselines-with needs --antennas: the baselines between one of --antennas and one of "
                             "--baselines-with")
        if request.baselines_with and request.correlation_type == "auto":
            raise ValueError("--baselines-with narrows cross-correlations; with --correlation-type auto, --antennas "
                             "alone chooses whose autocorrelations")
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
            x_range=x_range, y_range=y_range, point_size=request.point_size, color=request.color,
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
                if p.aspect == "equal" and not (p.axis_scale("x").is_linear and p.axis_scale("y").is_linear):
                    raise ValueError("--aspect equal needs linear axes (--x-scale and --y-scale linear)")
        except ValueError as err:
            raise RequestError(f"plot {name!r}: {err}") from err
        plots_by_name[name] = plots
    if locate_box is not None:
        n_streamed = sum(len(p) for p in plots_by_name.values())
        if n_streamed != 1:
            raise RequestError(f"--locate needs exactly one streamed plot; --plots gives {n_streamed}")
    return CheckedRequest(plot_names, plots_by_name, locate_box, figure_size)


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
    _figures: list | None = field(default=None, repr=False)

    @property
    def xy_plots(self) -> list[PlotSpec]:
        return [p for plots in self.plots_by_name.values() for p in plots]

    def set_record(self, run_id: str | None) -> None:
        """Name the provenance record of this run on every plot's panel."""
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
        plots_by_name[name] = [p.with_units(header_ctx) for p in plots]
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
                              theme=request.plot_theme)
                  for p in xy_plots}
    cache = RangeCache(request.cache_dir) if request.cache_dir else None
    if cache is not None:
        for stale in cache.remove_stale_partials():
            report("info", f"removed {stale} (left by an earlier run that stopped while writing)")
    cached = cached_ranges(source, xy_plots, cache)
    if xy_plots and (checked.locate_box is None or request.output_dir):
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
                       source, xy_figures, cache, cached, labels)


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
    on_chunk = progress(pass_progress(run.source, "locating samples", plot.needs_data))
    completed = run.source.stream([locate], read_data=plot.needs_data, on_chunk=on_chunk)
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
    `progress` callback stops it."""
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
        shapes[p] = (h * factor, w * factor)
    grids = {}
    if xy_plots:
        on_chunk = progress(pass_progress(source, run.labels[1], any(p.needs_data for p in xy_plots)))
        grids, completed = plot_grids(source, xy_plots, extents, shapes, on_chunk=on_chunk)
        if not completed:
            raise Stopped("stopped while drawing; nothing saved")

    def _mpl_figure(item):
        return item.fig if isinstance(item, XYFigure) else item

    from matplotlib.backends.backend_pdf import PdfPages

    output_dir = Path(request.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in xy_figures.values():
        item.set_status(grid_summary(grids[item.plot], source.n_rows))
    written = []
    lowres_path = output_dir / f"{request.output_prefix}_lowres.pdf"
    with PdfPages(lowres_path) as pdf:
        for name, item in run.figures():
            if isinstance(item, XYFigure):
                item.show(grids[item.plot], display_dpi=lowres_dpi, downsample=factor)
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
