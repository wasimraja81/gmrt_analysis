"""One compact "lo:hi" range syntax, shared by every range-taking CLI flag
(antennas, channels, time, uvdist, w, HA, Az, El, parallactic angle, ...) --
not a different format per flag.

`"1:5,10,12:14"` is a comma-separated union of terms, each either a single
value ("10") or an inclusive range ("1:5"). `parse_range_spec` is the one
parser; `expand_int_ranges` and `parse_single_range` adapt it for the two
shapes a caller actually needs: a discrete set of ids (antennas, channel
indices), or one continuous (lo, hi) bound (everything `select_rows`
already takes as a range tuple).

For a quantity with a genuine physical unit (frequency, distance, angle --
not kilo-wavelengths, which isn't one: see `data_io.row_selection`'s
u/v/w_range_klambda, which handle it exactly via each row's own channel
frequencies, not a unit conversion), `parse_quantity_range`/
`parse_single_quantity_range` accept an optional unit suffix on the whole
spec ("300:310MHz", "-6:6deg") and convert via `astropy.units` rather than
a hand-maintained conversion table -- `astropy.units` already knows every
frequency/distance/angle unit correctly, including the 15deg-per-hour
`hourangle` conversion HA needs, so there is no reason to reimplement it.
"""

from __future__ import annotations

import re

import astropy.units as u

_TRAILING_UNIT_RE = re.compile(r"[A-Za-zµ]+$")


def parse_range_spec(spec: str) -> list[tuple[float, float]]:
    """A comma-separated union of "lo:hi" ranges and/or bare values, as a
    list of (lo, hi) pairs -- a bare value "10" becomes (10, 10).

    `lo` may exceed `hi` within one term (e.g. "22:2"): kept as given, not
    rejected or reordered, since some callers (`select_rows`'s HA/Az/
    parallactic-angle filters) treat that as wrapping through the
    quantity's own cyclic boundary rather than an invalid range."""
    ranges = []
    for term in spec.split(","):
        term = term.strip()
        if not term:
            continue
        if ":" in term:
            lo_str, hi_str = term.split(":", 1)
            ranges.append((float(lo_str), float(hi_str)))
        else:
            value = float(term)
            ranges.append((value, value))
    if not ranges:
        raise ValueError(f"empty range spec: {spec!r}")
    return ranges


def expand_int_ranges(spec: str) -> list[int]:
    """A range spec expanded into a sorted, deduplicated list of integers --
    for a discrete id selection (antennas, channel indices), where "1:5"
    means every integer from 1 to 5 inclusive, not just its two endpoints.
    A reversed term ("5:1") is treated the same as "1:5" -- there's no
    "wrapping" concept for a discrete id list, unlike a cyclic continuous
    range."""
    values: set[int] = set()
    for lo, hi in parse_range_spec(spec):
        lo_i, hi_i = sorted((int(lo), int(hi)))
        values.update(range(lo_i, hi_i + 1))
    return sorted(values)


def parse_single_range(spec: str) -> tuple[float, float]:
    """A range spec that must be exactly one "lo:hi" bound -- for a
    continuous range filter (time, uvdist, HA, Az, El, parallactic angle,
    w, ...) that takes one (lo, hi) pair, not a set. Raises `ValueError` if
    `spec` names more than one term (a comma), rather than silently using
    only the first."""
    ranges = parse_range_spec(spec)
    if len(ranges) != 1:
        raise ValueError(f"expected a single lo:hi range, got {spec!r} ({len(ranges)} terms)")
    return ranges[0]


def _split_trailing_unit(spec: str) -> tuple[str, str | None]:
    """`spec` with any trailing unit letters removed, and those letters
    separately -- `None` if there weren't any. The unit, if present, is
    expected once at the very end of the whole spec, not per comma-term
    (a spec is in one unit throughout, never mixed)."""
    spec = spec.strip()
    match = _TRAILING_UNIT_RE.search(spec)
    if match is None:
        return spec, None
    return spec[: match.start()], match.group()


def parse_quantity_range(spec: str, native_unit: u.Unit) -> list[tuple[float, float]]:
    """Like `parse_range_spec`, but `spec` may end in a unit (e.g.
    "300:310MHz", "-6:6deg") -- converted to `native_unit` via
    `astropy.units`, exactly (case-sensitive, as real units are: "mHz" and
    "MHz" are not the same thing, so this never folds case to be lenient).
    A bare spec with no unit suffix is assumed to already be in
    `native_unit`."""
    numeric_part, unit_str = _split_trailing_unit(spec)
    unit = u.Unit(unit_str) if unit_str else native_unit
    return [
        ((lo * unit).to_value(native_unit), (hi * unit).to_value(native_unit))
        for lo, hi in parse_range_spec(numeric_part)
    ]


def parse_single_quantity_range(spec: str, native_unit: u.Unit) -> tuple[float, float]:
    """`parse_quantity_range`, requiring exactly one lo:hi term (see
    `parse_single_range`)."""
    ranges = parse_quantity_range(spec, native_unit)
    if len(ranges) != 1:
        raise ValueError(f"expected a single lo:hi range, got {spec!r} ({len(ranges)} terms)")
    return ranges[0]
