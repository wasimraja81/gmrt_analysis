"""Pure argument-resolution logic for `bin/visplot.sh` -- translating CLI
strings into what `select_rows`, `iter_visibility_chunks` and the plot specs
take, via the resolvers in `visplot/` and `instruments/gmrt/`. Kept separate
from the argparse/orchestration in `cli/visplot.py` so this logic is
directly testable without invoking a subprocess.
"""

from __future__ import annotations

import astropy.units as u

from data_io.antenna_table import Antenna
from instruments.gmrt.antenna_selection import resolve_antenna_selection
from visplot.channel_selection import resolve_channel_selection
from visplot.plot_spec import PRESETS
from visplot.quantities import QUANTITIES
from visplot.range_spec import parse_single_quantity_range, parse_single_range
from visplot.time_range import resolve_time_range_jd

TABLE_PLOTS = {"antenna-layout", "source-listing"}  # drawn from the file's tables
NAMED_PLOTS = TABLE_PLOTS | set(PRESETS)
QUANTITY_NAMES = set(QUANTITIES)
CATEGORY_NAMES = {name for name, q in QUANTITIES.items() if q.categorical}


def parse_plot_names(spec: str) -> list[str]:
    """`--plots` into an ordered list of plot names, e.g.
    "antenna-layout,amp-vs-time_h" -- each one checked by
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
            f"e.g. 'amp-vs-{name}' or '{name}-vs-time_h'"
        )
    raise ValueError(
        f"unrecognized plot name {name!r}: expected one of {sorted(NAMED_PLOTS)} "
        f"or a 'Y-vs-X' pair of quantities from {sorted(QUANTITY_NAMES)}"
    )


def validate_quantity_name(name: str, context: str = "") -> None:
    if name not in QUANTITY_NAMES:
        where = f" {context}" if context else ""
        raise ValueError(f"unknown quantity {name!r}{where}: expected one of {sorted(QUANTITY_NAMES)}")


def parse_quantity_pair(name: str) -> tuple[str, str]:
    """A generic "Y-vs-X" plot name into (y_quantity, x_quantity) -- names
    from the quantity registry, joined by the separator "-vs-"
    (distinct from the underscores within a quantity name like "freq_mhz")."""
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


def resolve_time_range_arg(spec: str | None, reference_jd: float) -> tuple[float, float] | None:
    return resolve_time_range_jd(spec, reference_jd) if spec else None


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
