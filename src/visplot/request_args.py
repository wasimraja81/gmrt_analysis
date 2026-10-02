"""The grammar of a plot request's option values -- translating the strings
the command line and the GUI's fields take into what `select_rows`,
`iter_visibility_chunks` and the plot specs take, via the resolvers in
`visplot/` and `instruments/gmrt/`. The command line and the GUI both check
and resolve their values here (`visplot.run`), so they accept exactly the
same text.
"""

from __future__ import annotations

import math
import textwrap

import astropy.units as u

from data_io.antenna_table import Antenna
from instruments.gmrt.antenna_selection import resolve_antenna_selection
from visplot.channel_selection import resolve_channel_selection
from visplot.plot_spec import PRESETS
from visplot.quantities import ALIASES, QUANTITIES, RETIRED_NAMES
from visplot.range_spec import parse_single_quantity_range, parse_single_range
from visplot.time_range import resolve_time_range_jd

TABLE_PLOTS = {"antenna-layout", "source-listing"}  # drawn from the file's tables
NAMED_PLOTS = TABLE_PLOTS | set(PRESETS)
QUANTITY_NAMES = set(QUANTITIES) | set(ALIASES)  # earlier names that carry a unit still work
CATEGORY_NAMES = {name for name, q in QUANTITIES.items() if q.categorical}
# Saved files: --dpi (the plot window's Export offers the same range) and --figure-size.
DEFAULT_DPI = 150
DPI_LIMITS = (50, 2400)
DEFAULT_FIGURE_SIZE = "8,7"  # 8 x 6 in for the plot, and an inch for the panel under it
DEFAULT_PAGE_FIGURE_SIZE = "16,11"  # a page of several plots (--one-plot-per): landscape (T26)
MAX_FIGURE_INCHES = 100.0
# --page-grid: the plots on a page, rows by columns (the user, 2026-10-02: 5 x 6 by default), and the most of either.
DEFAULT_PAGE_GRID = (5, 6)
MAX_PAGE_GRID = 12


def resolve_dpi_arg(dpi: int) -> int:
    """--dpi N, within DPI_LIMITS."""
    lo, hi = DPI_LIMITS
    if not lo <= dpi <= hi:
        raise ValueError(f"--dpi needs {lo} to {hi}, got {dpi}")
    return dpi


def resolve_figure_size_arg(spec: str) -> tuple[float, float]:
    """--figure-size 'W,H': width and height in inches, each above 0 and at
    most MAX_FIGURE_INCHES."""
    parts = spec.split(",")
    try:
        width, height = (float(p) for p in parts) if len(parts) == 2 else (None, None)
    except ValueError:
        width = height = None
    if width is None or not all(math.isfinite(v) and 0 < v <= MAX_FIGURE_INCHES for v in (width, height)):
        raise ValueError(f"--figure-size takes 'W,H' in inches, each above 0 and at most {MAX_FIGURE_INCHES:g}, "
                         f"got {spec!r}")
    return width, height


def resolve_page_grid_arg(spec: str | None) -> tuple[int, int] | None:
    """--page-grid 'ROWS,COLS': whole numbers from 1 to MAX_PAGE_GRID; None
    when not given (the default grid, `visplot.iterations.page_layout`)."""
    if spec is None:
        return None
    parts = spec.split(",")
    try:
        rows, cols = (int(p) for p in parts) if len(parts) == 2 else (0, 0)
    except ValueError:
        rows = cols = 0
    if not (1 <= rows <= MAX_PAGE_GRID and 1 <= cols <= MAX_PAGE_GRID):
        raise ValueError(f"--page-grid takes 'ROWS,COLS', whole numbers from 1 to {MAX_PAGE_GRID}, got {spec!r}")
    return rows, cols


def quantity_help(indent: int = 4, width: int = 79) -> str:
    """The quantities and their units, for --help, from the registry:
    quantities sharing a description on one entry."""
    groups: list[tuple[list[str], str]] = []
    for name, q in QUANTITIES.items():
        if groups and groups[-1][1] == q.description:
            groups[-1][0].append(name)
        else:
            groups.append(([name], q.description))
    column = indent + 17
    lines = []
    for names, description in groups:
        head = " " * indent + ", ".join(names)
        text = textwrap.wrap(description, width - column)
        if len(head) >= column - 1:
            lines.append(head)
        else:
            lines.append(head.ljust(column) + text.pop(0))
        lines += [" " * column + t for t in text]
    aliases = ", ".join(f"{name} ({q} in {unit})" for name, (q, unit) in ALIASES.items())
    lines += textwrap.wrap(f"earlier names carrying a unit still work: {aliases}", width,
                           initial_indent=" " * indent, subsequent_indent=" " * indent)
    return "\n".join(lines)


def parse_plot_names(spec: str) -> list[str]:
    """`--plots` into an ordered list of plot names, e.g.
    "antenna-layout,amp-vs-time" -- each one checked by
    `validate_plot_name`, so a bad name fails before any data is read."""
    names = [p.strip() for p in spec.split(",") if p.strip()]
    if not names:
        raise ValueError("--plots requires at least one plot name")
    for name in names:
        validate_plot_name(name)
    return names


def validate_plot_name(name: str) -> None:
    """Raises `ValueError` unless `name` is a named plot or a 'Y-vs-X' pair
    of known quantity names. A bare quantity name gets a message suggesting
    pairs."""
    if name in NAMED_PLOTS:
        return
    if "-vs-" in name:
        for quantity in parse_quantity_pair(name):
            validate_quantity_name(quantity, context=f"in plot {name!r}")
        return
    if name in QUANTITY_NAMES:
        raise ValueError(
            f"{name!r} is a quantity; a plot needs two, written 'Y-vs-X', "
            f"e.g. 'amp-vs-{name}' or '{name}-vs-time'"
        )
    raise ValueError(
        f"unrecognized plot name {name!r}: expected one of {sorted(NAMED_PLOTS)} "
        f"or a 'Y-vs-X' pair of quantities from {sorted(QUANTITIES)}"
    )


def validate_quantity_name(name: str, context: str = "") -> None:
    if name in RETIRED_NAMES:
        raise ValueError(f"{name!r} is no longer available: use {RETIRED_NAMES[name]}")
    if name not in QUANTITY_NAMES:
        where = f" {context}" if context else ""
        raise ValueError(f"unknown quantity {name!r}{where}: expected one of {sorted(QUANTITIES)} "
                         f"(or an earlier name: {sorted(ALIASES)})")


def parse_quantity_pair(name: str) -> tuple[str, str]:
    """A generic "Y-vs-X" plot name into (y_quantity, x_quantity) -- names
    from the quantity registry, joined by the separator "-vs-"
    (distinct from the underscores within a quantity name like "freq_mhz",
    an earlier name for freq in MHz)."""
    if "-vs-" not in name:
        raise ValueError(f"{name!r} is not a recognized plot name (not one of {sorted(NAMED_PLOTS)}, no '-vs-')")
    y_name, x_name = name.split("-vs-", 1)
    if not y_name or not x_name:
        raise ValueError(f"{name!r} is not a valid 'Y-vs-X' plot name")
    return y_name, x_name


def resolve_antennas_arg(spec: str | None, antennas: list[Antenna]) -> list[int] | None:
    return resolve_antenna_selection(spec, antennas) if spec else None


def resolve_channels_arg(spec: str | None, chan_freqs_hz) -> list[int] | None:
    return resolve_channel_selection(spec, chan_freqs_hz) if spec else None


def resolve_time_range_arg(spec: str | None, reference_jd: float,
                           recorded_minus_utc_s: float = 0.0) -> tuple[float, float] | None:
    return resolve_time_range_jd(spec, reference_jd, recorded_minus_utc_s) if spec else None


def resolve_uvdist_range_arg(spec: str | None) -> tuple[float, float] | None:
    """Metres by default (matching `select_rows`'s own `uvdist_range_m`),
    or an explicit unit ("km")."""
    return parse_single_quantity_range(spec, u.m) if spec else None


def resolve_klambda_range_arg(spec: str | None) -> tuple[float, float] | None:
    """A plain "lo:hi" in kilo-wavelengths -- no unit suffix is meaningful
    here, since kilo-wavelengths is the unit (see `select_rows`'s own
    u/v/w/uvdist_range_klambda docstring for why it isn't a scalar-unit
    conversion in the first place)."""
    return parse_single_range(spec) if spec else None


def resolve_ha_range_arg(spec: str | None) -> tuple[float, float] | None:
    """Hours by default (matching `select_rows`'s `ha_range_hours`), or an
    explicit "deg" suffix."""
    return parse_single_quantity_range(spec, u.hourangle) if spec else None


def resolve_deg_range_arg(spec: str | None) -> tuple[float, float] | None:
    """Degrees by default (Az/El/parallactic angle), or an explicit "rad"
    suffix."""
    return parse_single_quantity_range(spec, u.deg) if spec else None


def resolve_stokes_axis_selection(spec: str | None, stokes_labels: list[str] | None):
    """Which STOKES-axis indices `spec` (a comma-separated list of labels,
    e.g. "RR,LL") selects, matched case-insensitively against this file's
    own `stokes_labels` -- `None` if `spec` is `None`. Raises if `spec` is
    given but matches no label at all, or if this file has no STOKES axis."""
    if spec is None:
        return None
    if not stokes_labels:
        raise ValueError("--stokes was given, but this file has no STOKES axis")
    wanted = {s.strip().upper() for s in spec.split(",") if s.strip()}
    indices = [i for i, label in enumerate(stokes_labels) if label.upper() in wanted]
    if not indices:
        raise ValueError(f"--stokes {spec!r} matched none of this file's Stokes labels {stokes_labels}")
    return indices


def validate_colorize_by(name: str) -> None:
    """--colorize-by takes a category quantity (one color per category)."""
    if name not in CATEGORY_NAMES:
        raise ValueError(f"--colorize-by takes a category, one of {sorted(CATEGORY_NAMES)}; got {name!r}")


def resolve_plain_range_arg(spec: str | None) -> tuple[float, float] | None:
    """--x-range/--y-range: 'lo:hi' in the axis quantity's own units."""
    return parse_single_range(spec) if spec else None


def resolve_percentiles_arg(spec: str) -> tuple[float, float]:
    """--range-percentiles 'LO:HI', with 0 <= LO < HI <= 100."""
    lo, hi = parse_single_range(spec)
    if not 0.0 <= lo < hi <= 100.0:
        raise ValueError(f"--range-percentiles needs 0 <= LO < HI <= 100, got {spec!r}")
    return lo, hi


def resolve_locate_box_arg(spec: str | None) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """--locate 'XLO:XHI,YLO:YHI': a box in the plot's own axis units."""
    if not spec:
        return None
    parts = spec.split(",")
    if len(parts) != 2:
        raise ValueError(f"--locate takes 'XLO:XHI,YLO:YHI', got {spec!r}")
    return parse_single_range(parts[0]), parse_single_range(parts[1])
