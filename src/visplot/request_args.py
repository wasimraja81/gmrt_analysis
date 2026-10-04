"""The grammar of a plot request's option values -- translating the strings
the command line and the GUI's fields take into what `select_rows`,
`iter_visibility_chunks` and the plot specs take, via the resolvers in
`visplot/` and `instruments/gmrt/`. The command line and the GUI both check
and resolve their values here (`visplot.run`), so they accept exactly the
same text.
"""

from __future__ import annotations

import math
import re
import textwrap
from dataclasses import dataclass

from data_io.antenna_table import Antenna
from data_io.stokes_products import ProductPlan, plan_products
from instruments.gmrt.antenna_selection import resolve_antenna_selection
from visplot.channel_selection import resolve_channel_selection
from visplot.plot_spec import PRESETS
from visplot.quantities import QUANTITIES, convert, domain_in, domain_text, resolve_unit, units_of
from visplot.range_spec import NUMBER, parse_single_range
from visplot.time_range import resolve_time_range_jd

TABLE_PLOTS = {"antenna-layout", "source-listing"}  # drawn from the file's tables
NAMED_PLOTS = TABLE_PLOTS | set(PRESETS)
QUANTITY_NAMES = set(QUANTITIES)
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
# --correlation-type not given: a plot's cross-correlations (autocorrelation amplitudes sit far above them), a
# listing's every row (what the file holds); resolved by what runs, as --figure-size is (the user, 2026-10-03).
DEFAULT_PLOT_CORRELATION_TYPE = "cross"
DEFAULT_LISTING_CORRELATION_TYPE = "both"


def resolve_correlation_type(correlation_type: str | None, listing: bool = False) -> str:
    """--correlation-type as given, or a plot's or a listing's default."""
    if correlation_type is not None:
        return correlation_type
    return DEFAULT_LISTING_CORRELATION_TYPE if listing else DEFAULT_PLOT_CORRELATION_TYPE


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
    return "\n".join(lines)


def parse_plot_names(spec: str) -> list[str]:
    """`--plots` into an ordered list of plot names, e.g.
    "antenna-layout,amp-vs-time" -- each one checked by
    `validate_plot_name`, so a bad name fails before any data is read. A
    name may stack two streamed plots with "+" (T26: "amp-vs-time+phase-vs-time",
    the second under the first, sharing x)."""
    names = [p.strip() for p in spec.split(",") if p.strip()]
    if not names:
        raise ValueError("--plots requires at least one plot name")
    for name in names:
        parts = name.split("+")
        if len(parts) > 2:
            raise ValueError(f"plot {name!r}: a stack holds two plots, joined by one '+'")
        for part in parts:
            if len(parts) > 1 and part in TABLE_PLOTS:
                raise ValueError(f"plot {name!r}: {part} is drawn from the file's tables; only streamed plots stack")
            validate_plot_name(part)
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
    if name not in QUANTITY_NAMES:
        where = f" {context}" if context else ""
        raise ValueError(f"unknown quantity {name!r}{where}: expected one of {sorted(QUANTITIES)}")


def parse_quantity_pair(name: str) -> tuple[str, str]:
    """A generic "Y-vs-X" plot name into (y_quantity, x_quantity) -- names
    from the quantity registry, joined by the separator "-vs-"."""
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


_UNIT_RANGE = re.compile(rf"^\s*({NUMBER})\s*:\s*({NUMBER})\s*(\S*)\s*$")
_IN_WAVELENGTHS = ("in wavelengths a row is kept when its value at some frequency between the lowest and highest "
                   "--channels frequency (the whole band without --channels) is in the range")


@dataclass(frozen=True)
class UnitRange:
    """A selection range of quantity `quantity`, 'LO:HI UNIT', as the help
    names it (`name`). `arguments`: each unit `select_rows` takes, and its
    argument taking it; a unit given goes to the one of its base (km to m,
    Mlambda to klambda). `wraps`: an example of LO above HI wrapping through
    the quantity's boundary, or None: it does not wrap. `note`: added to the
    help."""

    quantity: str
    name: str
    arguments: dict[str, str]
    wraps: str | None = None
    note: str | None = None


UNIT_RANGES = {
    "--uvdist-range": UnitRange("uvdist", "uv distance", {"m": "uvdist_range_m", "klambda": "uvdist_range_klambda"},
                                note=_IN_WAVELENGTHS),
    **{f"--{c}-range": UnitRange(c, f"baseline coordinate {c}", {"m": f"{c}_range_m", "klambda": f"{c}_range_klambda"},
                                 note=_IN_WAVELENGTHS) for c in ("uu", "vv", "ww")},
    "--ha-range": UnitRange("ha", "hour angle", {"h": "ha_range_hours"}, "10:-10 h, through lower culmination"),
    "--az-range": UnitRange("az", "azimuth", {"deg": "az_range_deg"}, "350:10 deg, through north"),
    "--el-range": UnitRange("el", "elevation", {"deg": "el_range_deg"}),
    "--pa-range": UnitRange("pa", "parallactic angle", {"deg": "parallactic_angle_range_deg"},
                            "170:-170 deg, through 180 deg"),
}


def unit_range_help(option: str) -> str:
    """The help of a selection range option, from `UNIT_RANGES`."""
    unit_range = UNIT_RANGES[option]
    units = ", ".join(unit.name for unit in units_of(unit_range.quantity))
    wraps = f"; LO above HI wraps ({unit_range.wraps})" if unit_range.wraps else ""
    note = f"; {unit_range.note}" if unit_range.note else ""
    return f"'LO:HI UNIT' {unit_range.name}, both ends included, the unit required: {units}{wraps}{note}"


def split_range_unit(spec: str) -> tuple[str, str | None]:
    """'LO:HI UNIT' as written, split into its 'LO:HI' and its unit (None:
    none given); text not of that form whole, with None."""
    match = _UNIT_RANGE.match(spec)
    if not match:
        return spec.strip(), None
    return spec[:match.start(3)].strip(), match.group(3) or None


def resolve_range_arg(option: str, spec: str | None) -> dict[str, tuple[float, float]]:
    """A selection range (`UNIT_RANGES`), 'LO:HI UNIT': both ends included;
    the unit required, one of the quantity's (as --where takes them); values
    outside its physical domain refused; for a cyclic quantity LO above HI
    wraps through its boundary. Returned as the `select_rows` argument it
    gives, in that argument's unit, e.g. {"uvdist_range_klambda": (2.0, 5.0)};
    {} when `spec` is None."""
    if spec is None:
        return {}
    unit_range = UNIT_RANGES[option]
    quantity = unit_range.quantity
    units = ", ".join(unit.name for unit in units_of(quantity))
    match = _UNIT_RANGE.match(spec)
    if not match:
        raise ValueError(f"{option} {spec!r}: write 'LO:HI UNIT', the unit one of: {units}")
    lo, hi, unit = float(match.group(1)), float(match.group(2)), match.group(3)
    if not unit:
        raise ValueError(f"{option} {spec!r}: give the unit, one of: {units}")
    try:
        given = resolve_unit(quantity, unit)
    except ValueError:
        raise ValueError(f"{option} {spec!r}: unit {unit!r} does not apply; one of: {units}") from None
    lowest, highest = domain_in(quantity, given.name) or (None, None)
    for value in (lo, hi):
        if (lowest is not None and value < lowest) or (highest is not None and value > highest):
            raise ValueError(f"{option} {spec!r}: {quantity} lies in {domain_text(lowest, highest, given.name)}; "
                             f"{value:g} {given.name} is outside it")
    if lo > hi and not unit_range.wraps:
        raise ValueError(f"{option} {spec!r}: the lower end {lo:g} is above the upper end {hi:g}")
    to_unit = next(to for to in unit_range.arguments if resolve_unit(quantity, to).base == given.base)
    return {unit_range.arguments[to_unit]: (convert(lo, quantity, given.name, to_unit),
                                            convert(hi, quantity, given.name, to_unit))}


def resolve_stokes_products(spec: str | None, stokes_labels: list[str] | None) -> ProductPlan | None:
    """--stokes: the file's own products (e.g. "RR,LL", any case) and I, Q, U,
    V derived from them as the IAU defines them (`data_io.stokes_products`) --
    `None` if `spec` is `None`. Raises if this file has no STOKES axis, and for
    any product it neither holds nor can derive, naming what is missing."""
    if spec is None:
        return None
    if not stokes_labels:
        raise ValueError("--stokes was given, but this file has no STOKES axis")
    try:
        return plan_products(spec.split(","), stokes_labels)
    except ValueError as err:
        raise ValueError(f"--stokes {spec!r}: {err}") from err


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
