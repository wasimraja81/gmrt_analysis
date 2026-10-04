"""Instants of time, as `--where` conditions and `--time-range` take them.

    instant    := d/hh:mm:ss                      1/17:42:09: days from the file's day 0
                | YYYY-MM-DDThh:mm:ss             2021-07-25T17:42:09, an ISO date
    scale      := recorded | UTC
    time range := instant ".." instant scale      1/17:00:00 .. 1/18:00:00 UTC

Days count from the file's day 0 (AN RDATE, else DATE-OBS, else the date of its
first integration), as the plots' time axes count them. The scale says how the
clock reading is meant: `recorded`, as the rows record time (their TIMSYS), or
`UTC`; it is always written. Hours lie below 24, minutes and seconds below 60.
An instant becomes a recorded JD, the rows' own time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from astropy.time import Time

TIME_SCALES = ("recorded", "UTC")
INSTANT_FORMS = "d/hh:mm:ss (days from the file's day 0) or an ISO date, e.g. 2021-07-25T17:42:09"
TIME_RANGE_FORM = f"'LO .. HI SCALE', each end {INSTANT_FORMS}, the scale one of: {', '.join(TIME_SCALES)}"

_CLOCK = r"(\d{1,2}):(\d{2}):(\d{2}(?:\.\d+)?)"
_DAY_CLOCK = re.compile(rf"^(\d+)/{_CLOCK}$")
_ISO = re.compile(rf"^(\d{{4}}-\d{{2}}-\d{{2}})T{_CLOCK}$")


def check_instant(text: str) -> None:
    """Raises ValueError unless `text` is an instant (module docstring)."""
    match = _DAY_CLOCK.match(text) or _ISO.match(text)
    if not match:
        raise ValueError(f"{text!r} is not an instant: write {INSTANT_FORMS}")
    hh, mm, ss = int(match.group(2)), int(match.group(3)), float(match.group(4))
    if hh >= 24 or mm >= 60 or ss >= 60:
        raise ValueError(f"{text!r}: the clock reads hh:mm:ss, hours below 24, minutes and seconds below 60")
    if match.re is _ISO:
        try:
            Time(text, format="isot", scale="tai")
        except ValueError:
            raise ValueError(f"{text!r} is not a calendar date") from None


def instant_jd(text: str, scale: str, day0_jd: float, recorded_minus_utc_s: float) -> float:
    """Instant `text` (checked: `check_instant`) in `scale`, as a recorded JD.
    `day0_jd`: 0h of the file's day 0; `recorded_minus_utc_s`: the file's
    recorded time - UTC (`TimeReference.recorded_minus_utc_s`)."""
    match = _DAY_CLOCK.match(text)
    if match:
        day, hh, mm, ss = int(match.group(1)), int(match.group(2)), int(match.group(3)), float(match.group(4))
        jd = day0_jd + day + (hh * 3600 + mm * 60 + ss) / 86400.0
    else:
        jd = Time(text, format="isot", scale="tai").jd  # the calendar instant's JD number, in its own scale
    return jd + recorded_minus_utc_s / 86400.0 if scale == "UTC" else jd


@dataclass(frozen=True)
class TimeRange:
    """A time range as written: its two instants and their scale."""

    lo: str
    hi: str
    scale: str


def split_time_range_scale(spec: str) -> tuple[str, str | None]:
    """'LO .. HI SCALE' split into 'LO .. HI' and its scale; text ending in
    no scale whole, with None."""
    text, _, last = spec.strip().rpartition(" ")
    return (text.strip(), last) if text and last in TIME_SCALES else (spec.strip(), None)


def parse_time_range(spec: str) -> TimeRange:
    """`--time-range` 'LO .. HI SCALE' (module docstring), both ends
    included, checked without a file; ends of one form in order (an ISO date
    against a day count waits for the file's day 0: `resolve_time_range_jd`).
    Raises ValueError saying how to write it."""
    ends = spec.split("..")
    if len(ends) != 2:
        raise ValueError(f"--time-range {spec!r}: write {TIME_RANGE_FORM}")
    lo, rest = ends[0].split(), ends[1].split()
    if len(lo) == 2 and lo[1] in TIME_SCALES:
        raise ValueError(f"--time-range {spec!r}: the scale is written once, after HI")
    if len(lo) != 1 or len(rest) not in (1, 2):
        raise ValueError(f"--time-range {spec!r}: write {TIME_RANGE_FORM}")
    if len(rest) == 1:
        raise ValueError(f"--time-range {spec!r}: give the scale, one of: {', '.join(TIME_SCALES)}")
    time_range = TimeRange(lo[0], rest[0], rest[1])
    if time_range.scale not in TIME_SCALES:
        raise ValueError(f"--time-range {spec!r}: scale {time_range.scale!r} does not apply; one of: "
                         f"{', '.join(TIME_SCALES)}")
    for instant in (time_range.lo, time_range.hi):
        try:
            check_instant(instant)
        except ValueError as err:
            raise ValueError(f"--time-range {spec!r}: {err}") from None
    if bool(_DAY_CLOCK.match(time_range.lo)) == bool(_DAY_CLOCK.match(time_range.hi)):  # one form: no file needed
        _check_order(spec, time_range, *(instant_jd(t, time_range.scale, 0.0, 0.0)
                                         for t in (time_range.lo, time_range.hi)))
    return time_range


def resolve_time_range_jd(spec: str, day0_jd: float, recorded_minus_utc_s: float) -> tuple[float, float]:
    """`--time-range` `spec` (`parse_time_range`) as recorded JDs (lo, hi),
    the rows' own time."""
    time_range = parse_time_range(spec)
    lo, hi = (instant_jd(t, time_range.scale, day0_jd, recorded_minus_utc_s) for t in (time_range.lo, time_range.hi))
    _check_order(spec, time_range, lo, hi)
    return lo, hi


def _check_order(spec: str, time_range: TimeRange, lo_jd: float, hi_jd: float) -> None:
    if lo_jd > hi_jd:
        raise ValueError(f"--time-range {spec!r}: the lower end {time_range.lo} is after the upper end "
                         f"{time_range.hi}")
