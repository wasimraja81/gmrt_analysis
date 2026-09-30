"""A plot request: everything one visplot run takes, defined once, by the
command-line parser (`build_arg_parser`).

`PlotRequest` holds a value for each of the parser's options, keyed by the
option's own name (its argparse `dest`), with the parser's defaults: the
command line fills it from its arguments, the GUI from its form, and both run
it through `visplot.run`. It has no list of options of its own, so an option
added to the parser is at once part of every request, and `to_argv` writes a
request as a complete command line (every option, defaults included, paths
absolute) that `from_argv` reads back as the same request: the command that
reproduces a GUI plot is the request itself.
"""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

from visplot.axis_scale import SCALE_NAMES
from visplot.fonts import DEFAULT_PANEL_FONT, PANEL_FONTS
from visplot.plot_theme import DEFAULT_PLOT_THEME, MIN_CONTRAST, PLOT_THEMES
from visplot.request_args import CATEGORY_NAMES, DEFAULT_DPI, DEFAULT_FIGURE_SIZE, DPI_LIMITS, quantity_help
from visplot.xy_session import DEFAULT_STREAM_THREADS

PROG = "bin/visplot.sh"
# Where each run's provenance record goes (relative: from the directory visplot starts in).
DEFAULT_PROVENANCE_DIR = "visplot_runs"

_EPILOG = """\
plot names (--plots, comma-separated):
  table plots:
    antenna-layout             antenna positions, local East/North (m)
    source-listing             the file's source table
  observing geometry per selected row, colored by source:
    ha-range                   hour angle vs time
    az-el-range                elevation vs time and azimuth vs time
    parallactic-angle-range    parallactic angle vs time
  generic plots are written Y-vs-X, where Y and X are any two of these
  quantities, each shown in its default unit or the one --x-unit/--y-unit
  names:
{quantities}
  e.g. amp-vs-uvdist, phase-vs-time, v-vs-u. Every generic plot names two
  quantities; a single name such as uvdist is rejected.

units: --x-unit and --y-unit apply to every plot's x or y axis, presets
included (their fixed ranges converted). Clock time (recorded, UTC, local,
LST) is labelled dd:hh:mm:ss, days counted from the file's reference date
(RDATE in the antenna table, else DATE-OBS), as AIPS does; recorded time is
the file's DATE parameters as they are, in the time system the file declares
(TIMSYS). Local time is in the observatory's time zone, from the file's
TELESCOP (GMRT: Asia/Kolkata, IST) or --time-zone, at the UTC offset of the
first selected integration.

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
the view at a chosen dpi and saves it (PNG, PDF, SVG, EPS, TIFF, JPEG). Each
save's provenance record has the command that repeats it: an export's fixes
the view's ranges, --dpi and --figure-size, and with --output-dir saves the
same image.

examples:
  # antenna layout and source list only (reads no visibilities)
  bin/visplot.sh OBS.FITS --plots antenna-layout,source-listing

  # observing geometry of two calibrators
  bin/visplot.sh OBS.FITS --plots ha-range,az-el-range,parallactic-angle-range \\
      --sources 3C286,3C48

  # amplitude vs frequency, RR only, first 30 minutes of the file, 400-450 MHz
  bin/visplot.sh OBS.FITS --plots amp-vs-freq --sources 3C286 \\
      --stokes RR --time-range 0:0.5 --channels 400:450MHz

  # UV coverage in kilo-wavelengths, with the conjugate points, saved to disk
  bin/visplot.sh OBS.FITS --plots v-vs-u --mirror \\
      --sources 3C286 --output-dir plots/

  # phase against local observatory time (IST for GMRT)
  bin/visplot.sh OBS.FITS --plots phase-vs-time --x-unit local --sources 3C286

range syntax: 'lo:hi' for one range; several joined by commas ('1:5,10,12:14').
A unit suffix on any term applies to all terms ('100:100.5MHz,103.5:104MHz').

saved output (--output-dir): one PNG per plot (at --dpi, default 150), plus
two combined PDFs, one page per plot, with the samples as an image and axes,
labels and titles as vector:
  PREFIX_lowres.pdf   samples at --dpi
  PREFIX_highres.pdf  samples at the smallest whole multiple of --dpi that is
                      at least 600 dpi (600 for 150), for zooming in; the
                      PNGs are exact reductions of it
                      (skip it with --no-highres-pdf)
Each streamed plot's figure is --figure-size inches (default 8,7); the
antenna layout and source listing keep their own sizes. Under each plot a
panel states what it shows: the color key (Stokes or sources, as
--colorize-by chooses; one color without it), the Stokes, channels,
baselines, time span and row filters of the selection, what was drawn
(flagged samples as light-coral crosses), how flags combine, and the run's
provenance record.

flags: each visibility (time, baseline, channel, Stokes) has its own flag.
A plot whose quantities do not vary with Stokes (e.g. u-v) draws one point
per visibility's time, baseline and channel, flagged if any of its selected
Stokes is (select one Stokes for that product's own flags).
"""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Explore a UVFITS observation: antenna layout, source listing,\n"
        "observing geometry, and any quantity-vs-quantity visibility plot.\n"
        "Shows plots interactively unless --output-dir is given.",
        epilog=_EPILOG.replace("{quantities}", quantity_help()),
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
        help="keep every baseline with one of these antennas (one antenna: its baselines to all others), and "
        "their autocorrelations where --correlation-type keeps them; as AIPS's ANTENNAS and CASA's antenna='A,B': "
        "ids, full names, or GMRT code prefixes, e.g. '1:5,10,W01:25,C00' (default: all)",
    )
    sel.add_argument(
        "--baselines-with",
        help="with --antennas, keep only the cross-correlation baselines between one of --antennas and one of "
        "these (the same list in both: the baselines among them), as AIPS's BASELINE and CASA's antenna='A&B' "
        "(same syntax as --antennas); refused with --correlation-type auto, where --antennas alone chooses whose "
        "autocorrelations",
    )
    sel.add_argument(
        "--exclude-antennas", help="drop every baseline with at least one of these antennas (same syntax as --antennas)",
    )
    sel.add_argument(
        "--time-range",
        help="'lo:hi' in hours from the first integration in the file ('h' suffix optional), "
        "or absolute JD (UTC) with a 'jd' suffix, or 'start/end' as ISO-8601 UTC timestamps; absolute times "
        "become the file's recorded time by its TIMSYS/IATUTC (see 'time' under 'plot names')",
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
        help="keep every Nth row of those matching all other filters. Rows are in baseline order within "
        "each integration, so an N sharing a factor with the rows per integration skips whole baselines "
        "(the run warns and says how many it kept); --every-nth-integration keeps them all",
    )
    sel.add_argument(
        "--every-nth-integration", type=int, metavar="N",
        help="keep every Nth integration of those matching all other filters, with all its rows: every "
        "baseline stays",
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
        help="also draw flagged samples (weight <= 0), as light-coral crosses on top (default: flagged samples are "
        "left out)",
    )
    style.add_argument("--mirror", action="store_true", help="also plot (-x, -y), e.g. for UV coverage")
    style.add_argument(
        "--panel-font", choices=sorted(PANEL_FONTS), default=DEFAULT_PANEL_FONT,
        help="font of the panel under each plot, loaded from its file: tex-gyre-heros (default; Helvetica's "
        "metric clone, installed into the venv by bin/build_venv.sh) or dejavu-sans (matplotlib's own)",
    )
    style.add_argument(
        "--plot-theme", choices=list(PLOT_THEMES), default=DEFAULT_PLOT_THEME,
        help="colors of the plots, shown and saved: light (default; matplotlib's own) or dark (the GUI's dark "
        f"window colors; a marker color with less than {MIN_CONTRAST:g}:1 contrast against its background is "
        "mirrored in lightness)",
    )
    for axis in ("x", "y"):
        style.add_argument(
            f"--{axis}-unit", metavar="UNIT",
            help=f"unit of every plot's {axis} axis, from its quantity's list under 'plot names' "
            "(default: the quantity's default)",
        )
    style.add_argument(
        "--time-zone", metavar="ZONE",
        help="time zone for local time (--x-unit/--y-unit local), an IANA name such as Asia/Kolkata "
        "(default: the observatory's, from the file's TELESCOP)",
    )
    style.add_argument("--x-range", help="'lo:hi' x-axis range in the axis's unit (default: from the data)")
    style.add_argument("--y-range", help="'lo:hi' y-axis range in the axis's unit (default: from the data)")
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
        "--aspect", choices=["auto", "equal", "free"], default="auto",
        help="equal: one unit the same length on both axes; auto (default): equal when x and y are the same "
        "kind of quantity (u, v, w; real, imag) on linear axes, free otherwise",
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
    out.add_argument(
        "--dpi", type=int, default=DEFAULT_DPI, metavar="N",
        help=f"resolution of the saved PNGs and PREFIX_lowres.pdf, {DPI_LIMITS[0]} to {DPI_LIMITS[1]} "
        f"(default: {DEFAULT_DPI}); PREFIX_highres.pdf is at the smallest whole multiple of N that is at "
        "least 600",
    )
    out.add_argument(
        "--figure-size", default=DEFAULT_FIGURE_SIZE, metavar="W,H",
        help=f"each streamed plot's saved figure, width and height in inches (default: {DEFAULT_FIGURE_SIZE}); "
        "in the window, plots take the window's size",
    )
    out.add_argument(
        "--provenance-dir", default=DEFAULT_PROVENANCE_DIR, metavar="DIR",
        help="where each run's provenance record goes: the full command, host, git state, package "
        f"versions, input files and outputs (default: ./{DEFAULT_PROVENANCE_DIR}, in the directory "
        "visplot starts in)",
    )

    locate = parser.add_argument_group("locate (without a window)")
    locate.add_argument(
        "--locate", metavar="XLO:XHI,YLO:YHI",
        help="list every sample of the plot inside this box (in the axes' units) into --locate-csv; "
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


# Parser actions that are not part of a plot request: help, and --clear-cache (its own command).
_NOT_REQUEST_OPTIONS = {"help", "clear_cache"}
# Options naming files or directories: held as absolute paths, so the command runs anywhere.
PATH_OPTIONS = {"fits_path", "output_dir", "locate_csv", "cache_dir", "provenance_dir"}


def request_actions(parser: argparse.ArgumentParser | None = None) -> list[argparse.Action]:
    """The parser's actions that make up a request, in the parser's order
    (the FITS path first)."""
    parser = parser or build_arg_parser()
    return [a for a in parser._actions if a.dest not in _NOT_REQUEST_OPTIONS]


def request_option_names() -> list[str]:
    """Every option of a request, by its `dest` name, in the parser's order."""
    return [a.dest for a in request_actions()]


class PlotRequest:
    """One visplot run's options, keyed by the parser's `dest` names (see
    the module docstring). Immutable; `replace` returns a changed copy.
    Attribute access reads an option, e.g. `request.x_unit`."""

    __slots__ = ("_values",)

    def __init__(self, fits_path: str, plots: str, **options):
        actions = request_actions()
        defaults = {a.dest: a.default for a in actions}
        unknown = sorted(set(options) - set(defaults))
        if unknown:
            raise TypeError(f"not visplot options: {unknown}")
        values = {**defaults, **options, "fits_path": fits_path, "plots": plots}
        for action in actions:  # an option with a default always has a value (argparse never gives None)
            if action.default is not None and values[action.dest] is None:
                raise ValueError(f"{action.option_strings[-1]} needs a value (default: {action.default})")
        for name in PATH_OPTIONS:
            if values.get(name):
                values[name] = str(Path(values[name]).expanduser().resolve())
        object.__setattr__(self, "_values", values)

    @classmethod
    def from_namespace(cls, namespace: argparse.Namespace) -> PlotRequest:
        values = {k: v for k, v in vars(namespace).items() if k not in _NOT_REQUEST_OPTIONS}
        return cls(**values)

    @classmethod
    def from_argv(cls, argv: list[str]) -> PlotRequest:
        """A request from command-line arguments (without the program name)."""
        return cls.from_namespace(build_arg_parser().parse_args(argv))

    def to_argv(self) -> list[str]:
        """The request as command-line arguments: the FITS path, then every
        option in the parser's order, defaults included (a flag only when
        set; an option with no value, i.e. not given, left out)."""
        argv = [self._values["fits_path"]]
        for action in request_actions():
            if not action.option_strings:
                continue
            value = self._values[action.dest]
            option = action.option_strings[-1]
            if isinstance(action, argparse._StoreTrueAction):
                if value:
                    argv.append(option)
            elif value is not None:
                text = repr(value) if isinstance(value, float) else str(value)
                # a value starting with '-' (e.g. an hour-angle range '-2:2') would read as an option
                argv += [f"{option}={text}"] if text.startswith("-") else [option, text]
        return argv

    def command_line(self) -> str:
        """The shell command that runs this request."""
        return shlex.join([PROG, *self.to_argv()])

    def replace(self, **changes) -> PlotRequest:
        values = dict(self._values)
        unknown = sorted(set(changes) - set(values))
        if unknown:
            raise TypeError(f"not visplot options: {unknown}")
        values.update(changes)
        return PlotRequest(**values)

    def as_dict(self) -> dict:
        return dict(self._values)

    def __getattr__(self, name: str):
        try:
            return self._values[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        raise AttributeError("PlotRequest is immutable; use replace()")

    def __eq__(self, other) -> bool:
        return isinstance(other, PlotRequest) and self._values == other._values

    def __hash__(self) -> int:
        return hash(tuple(sorted((k, repr(v)) for k, v in self._values.items())))

    def __repr__(self) -> str:
        return f"PlotRequest({self.command_line()})"
