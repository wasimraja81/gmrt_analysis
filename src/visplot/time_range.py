"""Resolving a time range spec into absolute (lo, hi) Julian Dates.

Not a `range_spec.parse_quantity_range` case: relative hours and absolute
Julian Date aren't scalar multiples of each other (converting relative
hours to JD means *adding* to a reference JD, not multiplying), and an
ISO-8601 UTC timestamp has its own colons ("10:00:00"), which would
collide with this codebase's colon-delimited "lo:hi" grammar used
everywhere else. So this has its own two-form syntax instead:

- "lo:hi" (the usual colon grammar): each side a plain number, in
  relative hours from a caller-supplied reference JD by default (e.g.
  "0:2" = the first two hours), or an explicit unit -- "h"/"hr"/"hour"
  (still relative hours) or "jd" (absolute Julian Date in UTC, e.g.
  "2460123.5:2460123.6jd").
- "start/end" (a slash, not a colon -- ISO-8601's own interval notation):
  each side a full ISO-8601 UTC timestamp ("2021-07-25T10:00:00") or bare
  date ("2021-07-25"), parsed via `astropy.time.Time`.

The result is in the file's recorded time: absolute (UTC) bounds have the
file's recorded time - UTC added; relative hours count from the reference,
itself a recorded time.
"""

from __future__ import annotations

from astropy.time import Time

from visplot.range_spec import split_trailing_unit

_RELATIVE_HOUR_UNITS = {"h", "hr", "hour", "hours"}


def resolve_time_range_jd(spec: str, reference_jd: float, recorded_minus_utc_s: float = 0.0) -> tuple[float, float]:
    """`spec` as an absolute (lo_jd, hi_jd) pair -- see the module docstring
    for the two accepted forms. `reference_jd` (typically a `RowIndex`'s own
    `jd.min()`, the observation's actual start) is what a relative-hours
    bound in the "lo:hi" form is measured from; unused for the "start/end"
    absolute-timestamp form.

    As with `range_spec.parse_quantity_range`, a unit found on either term
    of the "lo:hi" form applies to both -- "0:2h" and "0h:2h" mean the same
    thing -- and conflicting units on the two terms raise, rather than
    silently preferring one."""
    spec = spec.strip()
    to_recorded = recorded_minus_utc_s / 86400.0
    if "/" in spec:
        start_str, end_str = (s.strip() for s in spec.split("/", 1))
        return Time(start_str, scale="utc").jd + to_recorded, Time(end_str, scale="utc").jd + to_recorded

    raw_terms = [t.strip() for t in spec.split(":")]
    if len(raw_terms) != 2:
        raise ValueError(
            f"expected a single lo:hi time range, or an ISO 'start/end', got {spec!r}"
        )

    split_terms = [split_trailing_unit(t) for t in raw_terms]
    detected_kind = None  # "h" or "jd"
    for _, unit_str in split_terms:
        if unit_str is None:
            continue
        normalized = unit_str.lower()
        if normalized in _RELATIVE_HOUR_UNITS:
            kind = "h"
        elif normalized == "jd":
            kind = "jd"
        else:
            raise ValueError(f"unrecognized time unit {unit_str!r} in {spec!r}; use 'h' (relative hours) or 'jd'")
        if detected_kind is not None and detected_kind != kind:
            raise ValueError(f"mixed time units in one range spec: {spec!r}")
        detected_kind = kind
    kind = detected_kind or "h"

    values = []
    for original_term, (numeric_part, _) in zip(raw_terms, split_terms):
        try:
            value = float(numeric_part)
        except ValueError:
            raise ValueError(
                f"{original_term!r} is not a valid time bound; use a number (optionally with an "
                f"'h'/'jd' suffix) for lo:hi, or a full ISO-8601 timestamp with lo/hi"
            ) from None
        values.append(reference_jd + value / 24.0 if kind == "h" else value + to_recorded)
    return values[0], values[1]
