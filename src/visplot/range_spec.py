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


def _parse_one_term(term: str) -> tuple[float, float]:
    if ":" in term:
        lo_str, hi_str = term.split(":", 1)
        return float(lo_str), float(hi_str)
    value = float(term)
    return value, value


def parse_range_spec(spec: str) -> list[tuple[float, float]]:
    """A comma-separated union of "lo:hi" ranges and/or bare values, as a
    list of (lo, hi) pairs -- a bare value "10" becomes (10, 10).

    `lo` may exceed `hi` within one term (e.g. "22:2"): kept as given, not
    rejected or reordered, since some callers (`select_rows`'s HA/Az/
    parallactic-angle filters) treat that as wrapping through the
    quantity's own cyclic boundary rather than an invalid range."""
    terms = [t.strip() for t in spec.split(",") if t.strip()]
    if not terms:
        raise ValueError(f"empty range spec: {spec!r}")
    return [_parse_one_term(term) for term in terms]


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


def split_trailing_unit(term: str) -> tuple[str, str | None]:
    """One term (not a whole comma-separated spec) with any trailing unit
    letters removed, and those letters separately -- `None` if there
    weren't any."""
    term = term.strip()
    match = _TRAILING_UNIT_RE.search(term)
    if match is None:
        return term, None
    return term[: match.start()], match.group()


def parse_quantity_range(spec: str, native_unit: u.Unit) -> list[tuple[float, float]]:
    """Like `parse_range_spec`, but any one term (not necessarily every
    term) may end in a unit (e.g. "300:310MHz", "-6:6deg,10deg") --
    converted to `native_unit` via `astropy.units`, exactly (case-sensitive,
    as real units are: "mHz" and "MHz" are not the same thing, so this
    never folds case to be lenient). A unit found on one term applies to
    every term in the spec; giving two different units in one spec raises,
    rather than silently picking one. A spec with no unit anywhere is
    assumed to already be in `native_unit`."""
    raw_terms = [t.strip() for t in spec.split(",") if t.strip()]
    if not raw_terms:
        raise ValueError(f"empty range spec: {spec!r}")

    detected_unit_str = None
    numeric_terms = []
    for term in raw_terms:
        numeric_part, unit_str = split_trailing_unit(term)
        if unit_str is not None:
            if detected_unit_str is not None and unit_str != detected_unit_str:
                raise ValueError(f"mixed units {detected_unit_str!r} and {unit_str!r} in one range spec: {spec!r}")
            detected_unit_str = unit_str
        numeric_terms.append(numeric_part)

    unit = u.Unit(detected_unit_str) if detected_unit_str else native_unit
    return [
        ((lo * unit).to_value(native_unit), (hi * unit).to_value(native_unit))
        for lo, hi in (_parse_one_term(term) for term in numeric_terms)
    ]


def parse_single_quantity_range(spec: str, native_unit: u.Unit) -> tuple[float, float]:
    """`parse_quantity_range`, requiring exactly one lo:hi term (see
    `parse_single_range`)."""
    ranges = parse_quantity_range(spec, native_unit)
    if len(ranges) != 1:
        raise ValueError(f"expected a single lo:hi range, got {spec!r} ({len(ranges)} terms)")
    return ranges[0]
