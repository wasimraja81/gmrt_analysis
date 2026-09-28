"""Entry point for the visPlot exploration tool -- invoked by
`bin/visplot.sh`.

Usage: gmrt/bin/python3 src/cli/visplot.py FITS_PATH --plots PLOT[,PLOT...] [options]

Selects rows with `select_rows` (sources, correlation type, antennas, time,
uv distance, kilo-wavelength and observing-geometry ranges), then draws:
- table plots (antenna layout, source listing) from the file's tables;
- every other plot -- a generic "Y-vs-X" pair or a geometry preset -- by
  streaming the selection chunk by chunk into a fixed pixel grid per plot
  (`visplot.xy_session`), so memory does not grow with the selection.
Saves PNGs and PDFs with --output-dir; otherwise opens windows that fill as
the data streams in, re-stream on zoom, and locate and export (`visplot.qt_inspector`).
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

# The backend must be chosen before any submodule imports pyplot: Agg
# (headless, file-only) when only saving is requested, since no display
# may be available at all in that case; otherwise Qt, for the inspection window.
_HEADLESS_OPTIONS = ("--output-dir", "--clear-cache", "--locate")
matplotlib.use("Agg" if any(a.split("=")[0] in _HEADLESS_OPTIONS for a in sys.argv[1:]) else "QtAgg")

import numpy as np
from astropy.io import fits
from matplotlib.backends.backend_pdf import PdfPages

from cli.visplot_args import (  # noqa: E402
    CATEGORY_NAMES,
    TABLE_PLOTS,
    parse_plot_names,
    resolve_antennas_arg,
    resolve_channels_arg,
    resolve_deg_range_arg,
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
from data_io.antenna_table import read_antenna_table, read_array_earth_location  # noqa: E402
from data_io.astrometry import DEFAULT_UT1, AstrometryWarning, fallback_message  # noqa: E402
from data_io.row_index import default_row_index_path, load_row_index  # noqa: E402
from data_io.row_selection import select_rows  # noqa: E402
from data_io.source_table import read_source_table  # noqa: E402
from visplot.antenna_layout import antenna_layout  # noqa: E402
from visplot.locate_csv import LocateCsvWriter  # noqa: E402
from visplot.axis_scale import SCALE_NAMES  # noqa: E402
from visplot.plot_spec import PlotSpec, expand_plot_name  # noqa: E402
from visplot.quantities import QUANTITIES, context_from_source_table  # noqa: E402
from visplot.range_cache import RangeCache, clear_cache  # noqa: E402
from visplot.source_listing import source_listing  # noqa: E402
from visplot.stream import LocateReducer  # noqa: E402
from visplot.xy_figure import XYFigure, grid_summary  # noqa: E402
from visplot.xy_session import (  # noqa: E402
    DEFAULT_STREAM_THREADS,
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

_EPILOG = """\
plot names (--plots, comma-separated):
  table plots:
    antenna-layout             antenna positions, local East/North (m)
    source-listing             the file's source table
  observing geometry per selected row, colored by source:
    ha-range                   hour angle vs time
    az-el-range                elevation vs time and azimuth vs time
    parallactic-angle-range    parallactic angle vs time
  generic plots are written Y-vs-X, where Y and X are any two of:
    real, imag, amp            visibility; unit from the file's BUNIT keyword
    phase_deg                  visibility phase (deg)
    time_h                     time since the first selected integration (h)
    u_sec, v_sec, w_sec        u, v, w (s)
    uvdist_m                   sqrt(u^2 + v^2) (m)
    u_klambda, v_klambda       u, v, w, sqrt(u^2 + v^2) at each channel's
    w_klambda, uvdist_klambda  frequency (kilo-wavelengths)
    freq_mhz                   channel frequency (MHz)
    ha_h, az_deg, el_deg       hour angle (h), azimuth and elevation (deg)
    pa_deg                     parallactic angle (deg)
    stokes, source             categories (also for --colorize-by)
  e.g. amp-vs-uvdist_klambda, phase_deg-vs-time_h, v_klambda-vs-u_klambda.
  Every generic plot names two quantities; a single name such as
  uvdist_klambda is rejected.

axis ranges and scales: without --x-range/--y-range, an axis spans the data's
minimum to maximum, outliers included; --x-range-mode/--y-range-mode percentile
narrows it to --range-percentiles. Samples outside an axis range are left out
and counted on the plot. --x-scale/--y-scale choose linear, log (positive values
only), symlog or asinh; samples are binned evenly in the scale, so each pixel
spans the same width on screen.

how plots are drawn: the selection is read in chunks and each chunk's
samples are marked on a fixed pixel grid per plot, then dropped, so memory
does not depend on how much is selected. The first pass finds each axis's
range (reading visibility data only for amp, real, imag or phase axes);
--x-range/--y-range skip it for that axis.

window (no --output-dir): one window, a tab per plot. Each plot fills in as
chunks are read, with progress in the status bar; after a zoom or pan it
re-reads the selection and redraws the new region at the window's own
resolution. On a plot's toolbar, Locate lists the samples in a dragged box
(baseline, time, channel, Stokes, values; saved as CSV) and Export re-reads
the view at a chosen dpi and saves it (PNG, PDF, SVG, EPS, TIFF, JPEG).

examples:
  # antenna layout and source list only (reads no visibilities)
  bin/visplot.sh OBS.FITS --plots antenna-layout,source-listing

  # observing geometry of two calibrators
  bin/visplot.sh OBS.FITS --plots ha-range,az-el-range,parallactic-angle-range \\
      --sources 3C286,3C48

  # amplitude vs frequency, RR only, first 30 minutes of the file, 400-450 MHz
  bin/visplot.sh OBS.FITS --plots amp-vs-freq_mhz --sources 3C286 \\
      --stokes RR --time-range 0:0.5 --channels 400:450MHz

  # UV coverage in kilo-wavelengths, with the conjugate points, saved to disk
  bin/visplot.sh OBS.FITS --plots v_klambda-vs-u_klambda --mirror \\
      --sources 3C286 --output-dir plots/

range syntax: 'lo:hi' for one range; several joined by commas ('1:5,10,12:14').
A unit suffix on any term applies to all terms ('100:100.5MHz,103.5:104MHz').

saved output (--output-dir): one PNG per plot (150 dpi), plus two combined
PDFs, one page per plot, with the samples as an image and axes, labels and
titles as vector:
  PREFIX_lowres.pdf   samples at 150 dpi
  PREFIX_highres.pdf  samples at 600 dpi, for zooming in
                      (skip it with --no-highres-pdf)
"""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bin/visplot.sh",
        description="Explore a UVFITS observation: antenna layout, source listing,\n"
        "observing geometry, and any quantity-vs-quantity visibility plot.\n"
        "Shows plots interactively unless --output-dir is given.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "fits_path",
        help="the UVFITS file (opened read-only); its row index (FITS_PATH.idx.npz) must "
        "already exist -- build it with the pipeline's build_index stage",
    )
    parser.add_argument(
        "--plots", required=True,
        help="comma-separated plot names; see 'plot names' below",
    )

    sel = parser.add_argument_group("row selection (all given filters are combined)")
    sel.add_argument("--sources", help="comma-separated source names or ids, e.g. 3C286,3C48 (default: all)")
    sel.add_argument(
        "--correlation-type", choices=["cross", "auto", "both"], default="cross",
        help="cross-correlations, autocorrelations, or both (default: cross)",
    )
    sel.add_argument(
        "--antennas",
        help="keep baselines involving these antennas: ids, full names, or GMRT code "
        "prefixes, e.g. '1:5,10,W01:25,C00' (default: all)",
    )
    sel.add_argument("--exclude-antennas", help="drop baselines involving these antennas (same syntax as --antennas)")
    sel.add_argument(
        "--time-range",
        help="'lo:hi' in hours from the first integration in the file ('h' suffix optional), "
        "or absolute JD with a 'jd' suffix, or 'start/end' as ISO-8601 UTC timestamps",
    )
    sel.add_argument("--uvdist-range", help="'lo:hi' baseline length, metres by default, or with a unit suffix, e.g. 'km'")
    klambda_help = (
        "'lo:hi' in kilo-wavelengths; a row is kept if its value at any frequency between the "
        "lowest and highest --channels frequency (whole band if --channels is not given) falls in range"
    )
    sel.add_argument("--u-range-klambda", help=klambda_help)
    sel.add_argument("--v-range-klambda", help=klambda_help)
    sel.add_argument("--w-range-klambda", help=klambda_help)
    sel.add_argument("--uvdist-range-klambda", help=klambda_help)
    sel.add_argument("--ha-range", help="'lo:hi' hour angle, hours by default, or with a 'deg' suffix")
    sel.add_argument("--az-range", help="'lo:hi' azimuth, degrees by default, or with a 'rad' suffix")
    sel.add_argument("--el-range", help="'lo:hi' elevation, degrees by default, or with a 'rad' suffix")
    sel.add_argument("--pa-range", help="'lo:hi' parallactic angle, degrees by default, or with a 'rad' suffix")
    sel.add_argument(
        "--every-nth", type=int, metavar="N",
        help="keep every Nth row of those matching all other filters (not with --random-subset-n)",
    )
    sel.add_argument(
        "--random-subset-n", type=int, metavar="N",
        help="keep N randomly chosen rows of those matching all other filters; requires --random-seed",
    )
    sel.add_argument("--random-seed", type=int, help="seed for --random-subset-n, so the subset is reproducible")

    axis = parser.add_argument_group("channel/Stokes selection (generic Y-vs-X plots)")
    axis.add_argument(
        "--channels",
        help="channel indices ('10:20') or a frequency band with a unit ('400:450MHz') (default: all)",
    )
    axis.add_argument("--stokes", help="comma-separated Stokes/correlation labels, e.g. RR,LL (default: all)")

    style = parser.add_argument_group("style (generic Y-vs-X plots)")
    style.add_argument(
        "--point-size", type=float, metavar="S",
        help="marker area in points^2 (default: chosen from the number of samples -- 20 up to "
        "1e3, 4 up to 1e5, 1 up to 1e6, 0.25 beyond). Markers are squares above 1e5 "
        "samples, circles otherwise",
    )
    style.add_argument("--color", default="tab:blue", help="marker color, any matplotlib color (default: tab:blue)")
    style.add_argument(
        "--colorize-by", choices=sorted(CATEGORY_NAMES),
        help="color samples by a category (overrides --color)",
    )
    style.add_argument(
        "--show-flagged", action="store_true",
        help="also draw flagged samples (weight <= 0), in light red on top (default: flagged samples are left out)",
    )
    style.add_argument("--mirror", action="store_true", help="also plot (-x, -y), e.g. for UV coverage")
    style.add_argument("--x-range", help="'lo:hi' x-axis range in the quantity's units (default: from the data)")
    style.add_argument("--y-range", help="'lo:hi' y-axis range in the quantity's units (default: from the data)")
    for axis in ("x", "y"):
        style.add_argument(
            f"--{axis}-range-mode", choices=["minmax", "percentile"], default="minmax",
            help=f"when --{axis}-range is not given: the data's minimum to maximum, outliers included "
            "(default), or the --range-percentiles range",
        )
    style.add_argument(
        "--range-percentiles", default="0.1:99.9", metavar="LO:HI",
        help="percentiles for the percentile range mode (default: 0.1:99.9)",
    )
    for axis in ("x", "y"):
        style.add_argument(
            f"--{axis}-scale", choices=list(SCALE_NAMES), default="linear",
            help=f"{axis}-axis scale (default: linear); log shows positive values only",
        )
    style.add_argument(
        "--scale-linear-width", type=float, default=1.0, metavar="W",
        help="for symlog and asinh scales: the width around zero that stays linear (default: 1)",
    )

    out = parser.add_argument_group("output")
    out.add_argument(
        "--output-dir",
        help="save plots here (see 'saved output' below); without it, plots open in a window",
    )
    out.add_argument("--output-prefix", default="visplot", help="filename prefix for saved output (default: visplot)")
    out.add_argument(
        "--no-highres-pdf", action="store_true",
        help="skip PREFIX_highres.pdf (write only the PNGs and PREFIX_lowres.pdf)",
    )

    locate = parser.add_argument_group("locate (without a window)")
    locate.add_argument(
        "--locate", metavar="XLO:XHI,YLO:YHI",
        help="list every sample of the plot inside this box (in the plot's axis units) into --locate-csv; "
        "needs exactly one streamed plot; runs without a window",
    )
    locate.add_argument("--locate-csv", metavar="FILE", help="CSV file for --locate")

    cache = parser.add_argument_group("cache (files only where you name a directory)")
    cache.add_argument(
        "--cache-dir", metavar="DIR",
        help="save each axis range found by the range pass in DIR, and reuse it when the same "
        "selection is plotted again (skipping that pass). Files are named visplot-cache_*",
    )
    cache.add_argument(
        "--clear-cache", metavar="DIR",
        help="remove every visplot-cache_* file in DIR, and exit (FITS_PATH and --plots are not needed)",
    )

    perf = parser.add_argument_group("performance")
    perf.add_argument(
        "--threads", type=int, default=DEFAULT_STREAM_THREADS, metavar="N",
        help=f"worker threads computing each chunk (default: {DEFAULT_STREAM_THREADS}); "
        "reading the next chunk overlaps with computing in any case",
    )

    return parser


GEOMETRY_QUANTITIES = {"ha_h", "az_deg", "el_deg", "pa_deg"}

LOWRES_DPI = 150
HIGHRES_DPI = 600  # a whole multiple of LOWRES_DPI: the 150 dpi grid is an exact 4x4 reduction


def _read_header_keyword(fits_path: str, keyword: str) -> str | None:
    with fits.open(fits_path) as hdul:
        value = str(hdul[0].header.get(keyword, "")).strip()
    return value or None


def _terminal_progress(progress):
    """An on_chunk callback printing `progress` (a PassProgress) at every 10%
    of the selection."""
    next_mark = [0.1]

    def on_chunk(rows_done: int) -> bool:
        total = progress.total_rows
        if total and rows_done / total >= next_mark[0]:
            print(progress.text(rows_done), flush=True)
            while rows_done / total >= next_mark[0]:
                next_mark[0] += 0.1
        return True

    return on_chunk


def main(argv: list[str]) -> int:
    parser = build_arg_parser()
    if any(a == "--clear-cache" or a.startswith("--clear-cache=") for a in argv[1:]):
        pre = argparse.ArgumentParser(add_help=False)
        pre.add_argument("--clear-cache", required=True)
        known, _ = pre.parse_known_args(argv[1:])
        removed = clear_cache(known.clear_cache)
        for path in removed:
            print(f"removed {path}")
        print(f"removed {len(removed)} visplot cache file(s) from {known.clear_cache}")
        return 0
    args = parser.parse_args(argv[1:])
    fits_path = args.fits_path
    warnings.simplefilter("ignore", AstrometryWarning)  # reported once, by _report_ut1

    # Checked before the file is opened, so a typo fails immediately rather
    # than after a (possibly long) row selection or data read.
    try:
        plot_names = parse_plot_names(args.plots)
        if args.colorize_by:
            validate_colorize_by(args.colorize_by)
        x_range = resolve_plain_range_arg(args.x_range)
        y_range = resolve_plain_range_arg(args.y_range)
        range_percentiles = resolve_percentiles_arg(args.range_percentiles)
        locate_box = resolve_locate_box_arg(args.locate)
        if locate_box is not None and not args.locate_csv:
            raise ValueError("--locate needs --locate-csv FILE")
        if args.scale_linear_width <= 0:
            raise ValueError("--scale-linear-width must be positive")
        for axis, scale, fixed in (("x", args.x_scale, x_range), ("y", args.y_scale, y_range)):
            if scale == "log" and fixed is not None and fixed[0] <= 0:
                raise ValueError(f"--{axis}-scale log shows positive values only; --{axis}-range starts at {fixed[0]:g}")
    except ValueError as err:
        parser.error(str(err))

    idx_path = default_row_index_path(fits_path)
    if not idx_path.exists():
        print(f"no row index at {idx_path} -- build it first (the pipeline's build_index stage)", file=sys.stderr)
        return 1
    index = load_row_index(idx_path)

    antennas = read_antenna_table(fits_path)
    array_location = read_array_earth_location(fits_path)
    source_table = read_source_table(fits_path)
    telescope = _read_header_keyword(fits_path, "TELESCOP")
    bunit = _read_header_keyword(fits_path, "BUNIT")

    try:
        select_kwargs = dict(
            sources=[s.strip() for s in args.sources.split(",")] if args.sources else None,
            correlation_type=args.correlation_type,
            antennas=resolve_antennas_arg(args.antennas, antennas),
            exclude_antennas=resolve_antennas_arg(args.exclude_antennas, antennas),
            jd_range=resolve_time_range_arg(args.time_range, float(index.jd.min())),
            uvdist_range_m=resolve_uvdist_range_arg(args.uvdist_range),
            u_range_klambda=resolve_klambda_range_arg(args.u_range_klambda),
            v_range_klambda=resolve_klambda_range_arg(args.v_range_klambda),
            w_range_klambda=resolve_klambda_range_arg(args.w_range_klambda),
            uvdist_range_klambda=resolve_klambda_range_arg(args.uvdist_range_klambda),
            ha_range_hours=resolve_ha_range_arg(args.ha_range),
            az_range_deg=resolve_deg_range_arg(args.az_range),
            el_range_deg=resolve_deg_range_arg(args.el_range),
            parallactic_angle_range_deg=resolve_deg_range_arg(args.pa_range),
            every_nth=args.every_nth,
            random_subset_n=args.random_subset_n,
            random_seed=args.random_seed,
        )
        if any(v is not None for v in (args.ha_range, args.az_range, args.el_range, args.pa_range)):
            select_kwargs["source_table"] = source_table
            select_kwargs["array_location"] = array_location

        # The kλ row filters evaluate each row at the edges of a frequency span:
        # that span must be the band of the selected channels.
        channel_indices = resolve_channels_arg(args.channels, index.chan_freqs_hz)
        if channel_indices is not None:
            selected_freqs_hz = np.asarray(index.chan_freqs_hz)[channel_indices]
            select_kwargs["freq_range_hz"] = (float(selected_freqs_hz.min()), float(selected_freqs_hz.max()))
        stokes_indices = resolve_stokes_axis_selection(args.stokes, index.stokes_labels)
    except ValueError as err:
        parser.error(str(err))

    selection = select_rows(index, **select_kwargs)
    print(f"selected {selection.n_rows:,} rows; sources present: {selection.sources}")

    style = PlotSpec(
        y="", x="", colorize_by=args.colorize_by, show_flagged=args.show_flagged, mirror=args.mirror,
        x_range=x_range, y_range=y_range, point_size=args.point_size, color=args.color,
        x_scale=args.x_scale, y_scale=args.y_scale, x_range_mode=args.x_range_mode, y_range_mode=args.y_range_mode,
        range_percentiles=range_percentiles, scale_linear_width=args.scale_linear_width,
    )
    plots_by_name = {name: expand_plot_name(name, style) for name in plot_names if name not in TABLE_PLOTS}
    xy_plots = [p for plots in plots_by_name.values() for p in plots]
    for p in xy_plots:
        for axis in ("x", "y"):
            name = p.x if axis == "x" else p.y
            if QUANTITIES[name].categorical and not p.axis_scale(axis).is_linear:
                parser.error(f"--{axis}-scale applies to numeric axes; {name!r} is a category")

    axis_selection = {}
    if channel_indices is not None:
        axis_selection["FREQ"] = np.array(channel_indices)
    if stokes_indices is not None:
        axis_selection["STOKES"] = np.array(stokes_indices)
    stokes_labels = index.stokes_labels or []
    if stokes_indices is not None:
        stokes_labels = [stokes_labels[i] for i in stokes_indices]

    time_reference_jd = float(index.jd[selection.row_indices].min()) if selection.n_rows else float(index.jd.min())
    ctx = context_from_source_table(time_reference_jd, source_table, array_location, bunit, stokes_labels,
                                    antenna_names={a.station_number: a.name for a in antennas})
    source = XYSource(fits_path, index, selection.row_indices, axis_selection or None, ctx, stream_chunk_bytes(),
                      threads=max(1, args.threads))
    sources_present = list(selection.sources.values())
    xy_figures = {p: XYFigure(p, ctx, sources_present, telescope, fits_path) for p in xy_plots}
    cache = RangeCache(args.cache_dir) if args.cache_dir else None
    if cache is not None:
        for stale in cache.remove_stale_partials():
            print(f"removed {stale} (left by an earlier run that stopped while writing)")
    cached = cached_ranges(source, xy_plots, cache)
    if xy_plots and (locate_box is None or args.output_dir):
        for line in describe_passes(source, xy_plots, cached):
            print(line)
    elif locate_box is not None:
        reads = (f"reads visibility data, {source.n_rows * source.row_bytes / 1e9:.1f} GB"
                 if xy_plots[0].needs_data else "row metadata only, no visibility data read")
        print(f"one pass: locate the samples of {xy_plots[0].title} in the box ({reads})")
    _report_ut1(index, selection, xy_plots, xy_figures, geometry_filters=select_kwargs.get("source_table") is not None)
    n_passes = 2 if [k for k in range_pass_axes(xy_plots) if k not in cached] else 1
    labels = (f"pass 1 of {n_passes}: finding data ranges", f"pass {n_passes} of {n_passes}: drawing")

    if locate_box is not None:
        if len(xy_plots) != 1:
            parser.error(f"--locate needs exactly one streamed plot; --plots gives {len(xy_plots)}")
        _locate_to_csv(source, xy_plots[0], locate_box, args.locate_csv, fits_path)
        if not args.output_dir:
            return 0

    # Figures in the order the plots were named.
    figures: list[tuple[str, object]] = []
    for name in plot_names:
        if name == "antenna-layout":
            figures.append((name, antenna_layout(antennas, array_location, telescope=telescope, source_path=fits_path)))
        elif name == "source-listing":
            figures.append((name, source_listing(source_table)))
        else:
            figures += [(p.name, xy_figures[p]) for p in plots_by_name[name]]

    if args.output_dir:
        extents = {}
        if xy_plots:
            progress = pass_progress(source, labels[0], range_pass_reads_data(xy_plots, cached))
            extents = resolve_extents(source, xy_plots, on_chunk=_terminal_progress(progress), cache=cache)
        _save_outputs(args, source, xy_plots, xy_figures, extents, figures, labels[1])
    else:
        from visplot.qt_inspector import run_inspector

        run_inspector(source, figures, labels, cache=cache, cached=cached)
    if cache is not None:
        n_files, n_bytes = cache.usage()
        size = f"{n_bytes / 1e6:.1f} MB" if n_bytes >= 1e6 else f"{n_bytes / 1e3:.0f} KB"
        print(f"cache: {len(cache.written)} file(s) written this run; {n_files} in {cache.directory} "
              f"({size}); remove them with: {parser.prog} --clear-cache {cache.directory}")
    return 0


def _report_ut1(index, selection, xy_plots, xy_figures, geometry_filters: bool) -> None:
    """Where UT1 - UTC comes from for this selection's dates, printed before
    any pass; a fallback to UT1 = UTC is also noted on every geometry plot."""
    geometry_plots = [p for p in xy_plots if GEOMETRY_QUANTITIES.intersection(p.quantities)]
    if not geometry_plots and not geometry_filters:
        return
    jd = index.jd[selection.row_indices] if selection.n_rows else index.jd
    DEFAULT_UT1.ut1_minus_utc_s([float(jd.min()), float(jd.max())])
    print(f"UT1 - UTC (for hour angle, azimuth, elevation, parallactic angle): {', '.join(sorted(DEFAULT_UT1.sources_used))}")
    if DEFAULT_UT1.fallback_used:
        sys.stdout.flush()
        print(f"WARNING: {fallback_message()}", file=sys.stderr)
        for p in geometry_plots:
            xy_figures[p].set_note("UT1 = UTC assumed: hour angle may be off by up to 0.9 s of time")


def _locate_to_csv(source, plot, box, csv_path, fits_path) -> None:
    locate = LocateReducer(plot, box[0], box[1], limit=0)
    writer = LocateCsvWriter(csv_path, locate, source.ctx, fits_path=fits_path)
    locate.sink = writer
    progress = pass_progress(source, "locating samples", plot.needs_data)
    completed = source.stream([locate], read_data=plot.needs_data, on_chunk=_terminal_progress(progress))
    written = writer.close(locate, completed)
    n = len(locate.by_baseline)
    print(f"located {locate.n_found:,} samples on {n:,} baseline{'s' if n != 1 else ''}; wrote {written}")


def _save_outputs(args, source, xy_plots, xy_figures, extents, figures, draw_label) -> None:
    # One plotting pass at the high-res grid; the low-res grid is its exact 4x4 reduction.
    factor = HIGHRES_DPI // LOWRES_DPI
    shapes = {}
    for p in xy_plots:
        h, w = xy_figures[p].grid_shape(LOWRES_DPI)
        shapes[p] = (h * factor, w * factor)
    grids = {}
    if xy_plots:
        progress = pass_progress(source, draw_label, any(p.needs_data for p in xy_plots))
        grids, _ = plot_grids(source, xy_plots, extents, shapes, on_chunk=_terminal_progress(progress))

    def _mpl_figure(item):
        return item.fig if isinstance(item, XYFigure) else item

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in xy_figures.values():
        grid = grids[item.plot]
        item.set_status(grid_summary(grid, source.n_rows))

    lowres_path = output_dir / f"{args.output_prefix}_lowres.pdf"
    with PdfPages(lowres_path) as pdf:
        for name, item in figures:
            if isinstance(item, XYFigure):
                item.show(grids[item.plot], display_dpi=LOWRES_DPI, downsample=factor)
            png_path = output_dir / f"{args.output_prefix}_{name}.png"
            _mpl_figure(item).savefig(png_path, dpi=LOWRES_DPI)
            pdf.savefig(_mpl_figure(item), dpi=LOWRES_DPI)
            print(f"saved {png_path}")
    print(f"saved {lowres_path}")

    if not args.no_highres_pdf:
        highres_path = output_dir / f"{args.output_prefix}_highres.pdf"
        with PdfPages(highres_path) as pdf:
            for _, item in figures:
                if isinstance(item, XYFigure):
                    item.show(grids[item.plot], display_dpi=HIGHRES_DPI)
                pdf.savefig(_mpl_figure(item), dpi=HIGHRES_DPI)
        print(f"saved {highres_path}")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
