"""Entry point for the visPlot exploration tool -- invoked by
`bin/visplot.sh`.

Usage: gmrt/bin/python3 src/cli/visplot.py FITS_PATH --plots PLOT[,PLOT...] [options]

Assembles a selection via `select_rows` (reusing every filter already
built: sources, correlation type, antennas by id/name/GMRT prefix, a time
range, uvdist/u/v/w in metres or exact kilo-wavelengths, HA/Az/El/
parallactic-angle ranges) and, for the fixed plot types, calls the matching
`visplot` function directly; for a generic "Y-vs-X" plot, reads one shared
`VisibilityBlock` (channel/Stokes axis selection applied once, reused
across every generic plot in this invocation) and pairs two
`derived_quantities` via `scatter_xy`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

# The backend must be chosen before any submodule imports pyplot: Agg
# (headless, file-only) when only saving is requested, since no display
# may be available at all in that case; otherwise this host's own default
# interactive backend.
if "--output-dir" in sys.argv:
    matplotlib.use("Agg")

import numpy as np
from astropy.io import fits
from matplotlib.backends.backend_pdf import PdfPages

from cli.visplot_args import (  # noqa: E402
    is_generic_quantity_plot,
    parse_plot_names,
    parse_quantity_pair,
    resolve_antennas_arg,
    resolve_channels_arg,
    resolve_deg_range_arg,
    resolve_ha_range_arg,
    resolve_klambda_range_arg,
    resolve_stokes_axis_selection,
    resolve_time_range_arg,
    resolve_uvdist_range_arg,
    validate_quantity_name,
)
from data_io.antenna_table import read_antenna_table, read_array_earth_location  # noqa: E402
from data_io.astrometry import altaz_deg, hour_angle_hours, parallactic_angle_deg  # noqa: E402
from data_io.row_index import default_row_index_path, load_row_index  # noqa: E402
from data_io.row_selection import select_rows  # noqa: E402
from data_io.source_table import read_source_table  # noqa: E402
from data_io.uvfits_group_params import DEFAULT_RAM_FRACTION_TO_USE, host_total_memory_bytes  # noqa: E402
from data_io.visibility_data import VisibilityReadTooLarge, read_visibility_data  # noqa: E402
from visplot.antenna_layout import antenna_layout  # noqa: E402
from visplot.derived_quantities import compute_quantity, quantity_display_name, quantity_label  # noqa: E402
from visplot.geometry_range import az_el_range, hour_angle_range, parallactic_angle_range  # noqa: E402
from visplot.scatter_xy import scatter_xy  # noqa: E402
from visplot.source_listing import source_listing  # noqa: E402

GEOMETRY_PLOTS = {"ha-range", "az-el-range", "parallactic-angle-range"}


_EPILOG = """\
plot names (--plots, comma-separated):
  named plots:
    antenna-layout             antenna positions, local East/North (m)
    source-listing             the file's source table
    ha-range                   hour angle vs time, per source
    az-el-range                azimuth and elevation vs time, per source
    parallactic-angle-range    parallactic angle vs time, per source
  generic plots are written Y-vs-X, where Y and X are any two of:
    real, imag, amp            visibility; unit from the file's BUNIT keyword
    phase_deg                  visibility phase (deg)
    time_h                     time since the first selected integration (h)
    u_sec, v_sec, w_sec        u, v, w (s)
    uvdist_m                   sqrt(u^2 + v^2) (m)
    u_klambda, v_klambda       u, v, w, sqrt(u^2 + v^2) at each channel's
    w_klambda, uvdist_klambda  frequency (kilo-wavelengths)
    freq_mhz                   channel frequency (MHz)
    stokes                     Stokes/correlation label (for --colorize-by)
  e.g. amp-vs-uvdist_klambda, phase_deg-vs-time_h, v_klambda-vs-u_klambda.
  Every generic plot names two quantities; a single name such as
  uvdist_klambda is rejected.

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
      --sources 3C286 --channels 0:10 --output-dir plots/

range syntax: 'lo:hi' for one range; several joined by commas ('1:5,10,12:14').
A unit suffix on any term applies to all terms ('100:100.5MHz,103.5:104MHz').

saved output (--output-dir): one PNG per plot (150 dpi), plus two combined
PDFs, one page per plot. In both PDFs the data points are drawn into an image
(as AIPS UVPLT's array method does), while axes, labels and titles stay vector;
this keeps a page quick to open however many points it has:
  PREFIX_lowres.pdf   points at 150 dpi
  PREFIX_highres.pdf  points at 600 dpi, for zooming in
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
        help="marker area in points^2 (default: chosen from the number of points -- 20 up to "
        "1e3 points, 4 up to 1e5, 1 up to 1e6, 0.25 beyond). Markers are squares above 1e5 "
        "points, circles otherwise",
    )
    style.add_argument("--linewidths", type=float, default=0.0, help="marker edge width in points (default: 0)")
    style.add_argument("--color", default="tab:blue", help="marker color, any matplotlib color (default: tab:blue)")
    style.add_argument(
        "--colorize-by",
        help="color points by a category quantity, e.g. stokes (overrides --color)",
    )
    style.add_argument(
        "--show-flagged", action="store_true",
        help="also draw flagged points (weight <= 0), as light-red crosses (default: flagged points are hidden)",
    )
    style.add_argument("--mirror", action="store_true", help="also plot (-x, -y), e.g. for UV coverage")

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

    return parser


# Interim, until generic plots stream in chunks (plan doc T22): the reader's
# size check counts its output arrays (24 bytes per sample), but a generic plot
# peaked at ~110 bytes per sample (measured 2026-09-27, 3C286 and 3C468.1, up to
# 122 million samples). Scaling the read budget by this ratio keeps the whole
# plot within the host-RAM fraction.
_READ_TO_PLOT_BYTES = 24 / 110

LOWRES_DPI = 150
HIGHRES_DPI = 600


def _save_combined_pdf(path: Path, figures, dpi: int) -> None:
    """All figures into one PDF, one page each. Each figure's data points
    (its scatter collections) are drawn into an image at `dpi`; axes,
    labels, and titles stay vector."""
    with PdfPages(path) as pdf:
        for _, fig in figures:
            for ax in fig.axes:
                for collection in ax.collections:
                    collection.set_rasterized(True)
            pdf.savefig(fig, dpi=dpi)


def _read_header_keyword(fits_path: str, keyword: str) -> str | None:
    with fits.open(fits_path) as hdul:
        value = str(hdul[0].header.get(keyword, "")).strip()
    return value or None


def main(argv: list[str]) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv[1:])
    fits_path = args.fits_path

    # Checked before the file is opened, so a typo fails immediately rather
    # than after a (possibly long) row selection or data read.
    try:
        plot_names = parse_plot_names(args.plots)
        if args.colorize_by:
            validate_quantity_name(args.colorize_by, context="in --colorize-by")
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

    needs_geometry = bool(GEOMETRY_PLOTS.intersection(plot_names)) or any(
        v is not None for v in (args.ha_range, args.az_range, args.el_range, args.pa_range)
    )

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
    if needs_geometry:
        select_kwargs["source_table"] = source_table
        select_kwargs["array_location"] = array_location

    # The kλ row filters evaluate each row at the edges of a frequency span:
    # that span must be the band of the selected channels.
    channel_indices = resolve_channels_arg(args.channels, index.chan_freqs_hz)
    if channel_indices is not None:
        selected_freqs_hz = np.asarray(index.chan_freqs_hz)[channel_indices]
        select_kwargs["freq_range_hz"] = (float(selected_freqs_hz.min()), float(selected_freqs_hz.max()))

    selection = select_rows(index, **select_kwargs)
    print(f"selected {selection.n_rows} rows; sources present: {selection.sources}")

    # A generic "Y-vs-X" plot needs the actual visibility data; every such
    # plot in this invocation shares one read (and one channel/Stokes axis
    # selection), rather than re-reading per plot.
    block = None
    if any(is_generic_quantity_plot(name) for name in plot_names):
        axis_selection = {}
        if channel_indices is not None:
            axis_selection["FREQ"] = np.array(channel_indices)
        stokes_indices = resolve_stokes_axis_selection(args.stokes, index.stokes_labels)
        if stokes_indices is not None:
            axis_selection["STOKES"] = np.array(stokes_indices)
        try:
            block = read_visibility_data(
                fits_path, index, selection.row_indices, axis_selection=axis_selection or None,
                max_bytes=int(host_total_memory_bytes() * DEFAULT_RAM_FRACTION_TO_USE * _READ_TO_PLOT_BYTES),
            )
        except VisibilityReadTooLarge as err:
            sys.stdout.flush()
            print(
                f"{parser.prog}: plotting this selection needs ~{err.estimated_bytes / _READ_TO_PLOT_BYTES / 1e9:.1f} GB "
                f"({err.n_rows:,} rows x {int(np.prod(err.selected_shape)):,} samples per row), over the "
                f"{err.max_bytes / _READ_TO_PLOT_BYTES / 1e9:.1f} GB limit ({err.ram_fraction:.0%} of this host's RAM).\n"
                "Narrow it with --channels, --stokes, --time-range, --every-nth or --random-subset-n.",
                file=sys.stderr,
            )
            return 1

    figures: list[tuple[str, object]] = []
    for name in plot_names:
        if name == "antenna-layout":
            fig = antenna_layout(antennas, array_location, telescope=telescope, source_path=fits_path)
        elif name == "source-listing":
            fig = source_listing(source_table)
        elif name in GEOMETRY_PLOTS:
            jd = index.jd[selection.row_indices]
            source_ids = index.source_id[selection.row_indices]
            ra_deg = np.array([source_table[int(sid)].ra_apparent_deg for sid in source_ids])
            dec_deg = np.array([source_table[int(sid)].dec_apparent_deg for sid in source_ids])
            labels = np.array([source_table[int(sid)].name for sid in source_ids])
            if name == "ha-range":
                fig = hour_angle_range(
                    jd, hour_angle_hours(jd, ra_deg, array_location), labels,
                    telescope=telescope, source_path=fits_path,
                )
            elif name == "az-el-range":
                az_deg, el_deg = altaz_deg(jd, ra_deg, dec_deg, array_location)
                fig = az_el_range(jd, az_deg, el_deg, labels, telescope=telescope, source_path=fits_path)
            else:
                fig = parallactic_angle_range(
                    jd, parallactic_angle_deg(jd, ra_deg, dec_deg, array_location), labels,
                    telescope=telescope, source_path=fits_path,
                )
        else:  # a generic Y-vs-X plot (parse_plot_names allows nothing else)
            y_name, x_name = parse_quantity_pair(name)
            y = compute_quantity(y_name, block)
            x = compute_quantity(x_name, block)
            colorize_by = compute_quantity(args.colorize_by, block) if args.colorize_by else None
            fig = scatter_xy(
                x, y, weight=block.weight, colorize_by=colorize_by, show_flagged=args.show_flagged,
                mirror=args.mirror, point_size=args.point_size, linewidths=args.linewidths, color=args.color,
                xlabel=quantity_label(x_name, bunit=bunit), ylabel=quantity_label(y_name, bunit=bunit),
                title=f"{quantity_display_name(y_name)} vs {quantity_display_name(x_name)}",
            )
        figures.append((name, fig))

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, fig in figures:
            png_path = output_dir / f"{args.output_prefix}_{name}.png"
            fig.savefig(png_path, dpi=LOWRES_DPI)
            print(f"saved {png_path}")

        lowres_path = output_dir / f"{args.output_prefix}_lowres.pdf"
        _save_combined_pdf(lowres_path, figures, dpi=LOWRES_DPI)
        print(f"saved {lowres_path}")

        if not args.no_highres_pdf:
            highres_path = output_dir / f"{args.output_prefix}_highres.pdf"
            _save_combined_pdf(highres_path, figures, dpi=HIGHRES_DPI)
            print(f"saved {highres_path}")
    else:
        import matplotlib.pyplot as plt

        plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
