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
)
from data_io.antenna_table import read_antenna_table, read_array_earth_location  # noqa: E402
from data_io.astrometry import altaz_deg, hour_angle_hours, parallactic_angle_deg  # noqa: E402
from data_io.row_index import default_row_index_path, load_row_index  # noqa: E402
from data_io.row_selection import select_rows  # noqa: E402
from data_io.source_table import read_source_table  # noqa: E402
from data_io.visibility_data import read_visibility_data  # noqa: E402
from visplot.antenna_layout import antenna_layout  # noqa: E402
from visplot.derived_quantities import compute_quantity  # noqa: E402
from visplot.geometry_range import az_el_range, hour_angle_range, parallactic_angle_range  # noqa: E402
from visplot.scatter_xy import scatter_xy  # noqa: E402
from visplot.source_listing import source_listing  # noqa: E402

GEOMETRY_PLOTS = {"ha-range", "az-el-range", "parallactic-angle-range"}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Explore a UVFITS observation: antenna layout, source listing, "
        "observing geometry, and any quantity-vs-quantity visibility plot.",
    )
    parser.add_argument("fits_path", help="the raw UVFITS file (read-only; its .idx.npz row index must already exist)")
    parser.add_argument(
        "--plots", required=True,
        help="comma-separated plot names: antenna-layout, source-listing, ha-range, "
        "az-el-range, parallactic-angle-range, or a generic 'Y-vs-X' pair using "
        "derived-quantity names (real, imag, amp, phase_deg, time_h, u_sec, v_sec, "
        "w_sec, uvdist_m, u_klambda, v_klambda, w_klambda, uvdist_klambda, freq_mhz, "
        "stokes), e.g. amp-vs-freq_mhz, u_klambda-vs-v_klambda",
    )

    sel = parser.add_argument_group("selection (mirrors select_rows)")
    sel.add_argument("--sources", help="comma-separated source names or ids")
    sel.add_argument("--correlation-type", choices=["cross", "auto", "both"], default="cross")
    sel.add_argument("--antennas", help="e.g. '1:5,10,W01:25,C00' (id, full name, or GMRT code prefix)")
    sel.add_argument("--exclude-antennas", help="same syntax as --antennas")
    sel.add_argument("--time-range", help="'lo:hi' (relative hours, or 'jd'-suffixed absolute) or 'start/end' (ISO-8601)")
    sel.add_argument("--uvdist-range", help="'lo:hi', metres by default, or with an explicit 'km' suffix")
    sel.add_argument("--u-range-klambda", help="'lo:hi' in kilo-wavelengths")
    sel.add_argument("--v-range-klambda", help="'lo:hi' in kilo-wavelengths")
    sel.add_argument("--w-range-klambda", help="'lo:hi' in kilo-wavelengths")
    sel.add_argument("--uvdist-range-klambda", help="'lo:hi' in kilo-wavelengths")
    sel.add_argument("--ha-range", help="'lo:hi', hours by default, or with an explicit 'deg' suffix")
    sel.add_argument("--az-range", help="'lo:hi', degrees by default, or with an explicit 'rad' suffix")
    sel.add_argument("--el-range", help="'lo:hi', degrees by default, or with an explicit 'rad' suffix")
    sel.add_argument("--pa-range", help="'lo:hi', degrees by default, or with an explicit 'rad' suffix")
    sel.add_argument("--every-nth", type=int)
    sel.add_argument("--random-subset-n", type=int)
    sel.add_argument("--random-seed", type=int)

    axis = parser.add_argument_group("axis-level selection")
    axis.add_argument("--channels", help="index range ('10:20') or frequency band with a unit ('300:310MHz')")
    axis.add_argument("--stokes", help="comma-separated Stokes/correlation labels, e.g. RR,LL")

    style = parser.add_argument_group("style (generic quantity-vs-quantity plots)")
    style.add_argument("--point-size", type=float, default=4.0)
    style.add_argument("--linewidths", type=float, default=0.0)
    style.add_argument("--color", default="tab:blue")
    style.add_argument("--colorize-by", help="a derived-quantity name to color points by, e.g. stokes")
    style.add_argument("--show-flagged", action="store_true")
    style.add_argument("--mirror", action="store_true", help="also plot (-x, -y), e.g. for UV coverage")

    out = parser.add_argument_group("output")
    out.add_argument("--output-dir", help="save a PNG per plot plus a combined PDF here, instead of showing interactively")
    out.add_argument("--output-prefix", default="visplot", help="filename prefix for saved output")

    return parser


def _read_telescope(fits_path: str) -> str | None:
    with fits.open(fits_path) as hdul:
        telescope = str(hdul[0].header.get("TELESCOP", "")).strip()
    return telescope or None


def main(argv: list[str]) -> int:
    args = build_arg_parser().parse_args(argv[1:])
    fits_path = args.fits_path

    idx_path = default_row_index_path(fits_path)
    if not idx_path.exists():
        print(f"no row index at {idx_path} -- build it first (the pipeline's build_index stage)", file=sys.stderr)
        return 1
    index = load_row_index(idx_path)

    antennas = read_antenna_table(fits_path)
    array_location = read_array_earth_location(fits_path)
    source_table = read_source_table(fits_path)
    telescope = _read_telescope(fits_path)

    plot_names = parse_plot_names(args.plots)
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

    selection = select_rows(index, **select_kwargs)
    print(f"selected {selection.n_rows} rows; sources present: {selection.sources}")

    # A generic "Y-vs-X" plot needs the actual visibility data; every such
    # plot in this invocation shares one read (and one channel/Stokes axis
    # selection), rather than re-reading per plot.
    block = None
    if any(is_generic_quantity_plot(name) for name in plot_names):
        axis_selection = {}
        channel_indices = resolve_channels_arg(args.channels, index.chan_freqs_hz)
        if channel_indices is not None:
            axis_selection["FREQ"] = np.array(channel_indices)
        stokes_indices = resolve_stokes_axis_selection(args.stokes, index.stokes_labels)
        if stokes_indices is not None:
            axis_selection["STOKES"] = np.array(stokes_indices)
        block = read_visibility_data(fits_path, index, selection.row_indices, axis_selection=axis_selection or None)

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
        elif is_generic_quantity_plot(name):
            y_name, x_name = parse_quantity_pair(name)
            y = compute_quantity(y_name, block)
            x = compute_quantity(x_name, block)
            colorize_by = compute_quantity(args.colorize_by, block) if args.colorize_by else None
            fig = scatter_xy(
                x, y, weight=block.weight, colorize_by=colorize_by, show_flagged=args.show_flagged,
                mirror=args.mirror, point_size=args.point_size, linewidths=args.linewidths, color=args.color,
                xlabel=x_name, ylabel=y_name, title=f"{y_name} vs {x_name}",
            )
        else:
            print(f"unrecognized plot name: {name!r}", file=sys.stderr)
            return 1
        figures.append((name, fig))

    if args.output_dir:
        from matplotlib.backends.backend_pdf import PdfPages

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = output_dir / f"{args.output_prefix}.pdf"
        with PdfPages(pdf_path) as pdf:
            for name, fig in figures:
                png_path = output_dir / f"{args.output_prefix}_{name}.png"
                fig.savefig(png_path, dpi=150)
                pdf.savefig(fig)
                print(f"saved {png_path}")
        print(f"saved {pdf_path}")
    else:
        import matplotlib.pyplot as plt

        plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
