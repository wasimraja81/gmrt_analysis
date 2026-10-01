"""Tick placement and labels for an axis in hours of clock time (recorded,
UTC, local observatory time, LST).

`ClockLocator` puts ticks at whole clock steps (1 s ... 30 min, 1 h ... 12 h,
days); `ClockFormatter` labels them in a --time-format (T48), the day counted
from the axis's day 0 (hours 24 to 48 are day 1), with fractions of a second
when the ticks need them:

- dd/hh:mm:ss (default): AIPS's day/time form, e.g. 1/16:44:00 -- the "/"
  marks the day apart from the time (the user's choice, 2026-10-01);
- dd:hh:mm:ss: e.g. 01:16:44:00;
- hh:mm:ss: the time of day, e.g. 16:44:00; the first tick, and the first
  tick of each later day, carry their day (1/16:44:00, ..., 2/00:00:00);
- iso: the calendar date and time, e.g. 2021-07-25 16:44:00, from day 0's
  date (none for LST, whose days are sidereal).
"""

from __future__ import annotations

import datetime
import math

import numpy as np
from matplotlib.ticker import Formatter, Locator, MaxNLocator

CLOCK_STEPS_S = (1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800,
                 3600, 7200, 10800, 21600, 43200, 86400)
SECONDS_PER_DAY = 86400
TIME_FORMATS = ("dd/hh:mm:ss", "dd:hh:mm:ss", "hh:mm:ss", "iso")
DEFAULT_TIME_FORMAT = "dd/hh:mm:ss"


class ClockLocator(Locator):
    """At most `max_ticks` ticks, at the smallest whole clock step allowing that."""

    def __init__(self, max_ticks: int = 7):
        self.max_ticks = max_ticks

    def __call__(self):
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

    def tick_values(self, vmin, vmax):
        lo, hi = sorted((float(vmin), float(vmax)))
        span_s = (hi - lo) * 3600.0
        if span_s < 2.0:  # below whole seconds: decimal steps of a second
            ticks_s = MaxNLocator(self.max_ticks).tick_values(lo * 3600.0, hi * 3600.0)
            return self.raise_if_exceeds(np.asarray(ticks_s) / 3600.0)
        intervals = self.max_ticks - 1
        step = next((s for s in CLOCK_STEPS_S if span_s / s <= intervals), None)
        if step is None:  # many days: whole days
            step = SECONDS_PER_DAY * math.ceil(span_s / SECONDS_PER_DAY / intervals)
        first = math.ceil(lo * 3600.0 / step) * step
        ticks_s = np.arange(first, hi * 3600.0 + step * 1e-9, step, dtype=np.float64)
        return self.raise_if_exceeds(ticks_s / 3600.0)


def clock_precision(hours) -> int:
    """Digits of seconds the values need: 0 for whole seconds, up to 3 for
    fractions."""
    seconds = np.asarray(hours, dtype=np.float64) * 3600.0
    for digits in range(3):
        if seconds.size == 0 or np.allclose(seconds, np.round(seconds, digits), rtol=0, atol=1e-6):
            return digits
    return 3


def _day_and_time(hours: float, precision: int) -> tuple[int, str]:
    """(day, "hh:mm:ss[.fff]") of `hours` since 0h of day 0."""
    scale = 10 ** precision
    q = round(hours * 3600.0 * scale)
    day, q = divmod(q, SECONDS_PER_DAY * scale)
    h, q = divmod(q, 3600 * scale)
    m, q = divmod(q, 60 * scale)
    s, frac = divmod(q, scale)
    return day, f"{h:02d}:{m:02d}:{s:02d}" + (f".{frac:0{precision}d}" if precision else "")


def clock_text(hours: float, precision: int, time_format: str = DEFAULT_TIME_FORMAT,
               day0: datetime.date | None = None, with_day: bool = True) -> str:
    """`hours` (since 0h of day 0) in `time_format`, with `precision` digits
    of seconds after the point. "iso" needs day 0's date (`day0`);
    "hh:mm:ss" writes the day as dd/hh:mm:ss when `with_day`."""
    day, text = _day_and_time(hours, precision)
    if time_format == "dd/hh:mm:ss" or (time_format == "hh:mm:ss" and with_day):
        return f"{day}/{text}"
    if time_format == "hh:mm:ss":
        return text
    if time_format == "dd:hh:mm:ss":
        return f"{day:02d}:{text}" if day >= 0 else f"-{-day:02d}:{text}"
    if time_format == "iso":
        if day0 is None:
            raise ValueError("the iso time format needs day 0's calendar date")
        return f"{(day0 + datetime.timedelta(days=day)).isoformat()} {text}"
    raise ValueError(f"no time format {time_format!r}; choose from {', '.join(TIME_FORMATS)}")


class ClockFormatter(Formatter):
    """Tick labels in `time_format` at the precision the ticks need, all
    alike (with hh:mm:ss, the first tick and the first of each later day
    carry their day); the cursor readout to a hundredth of a second, with
    its day."""

    def __init__(self, time_format: str = DEFAULT_TIME_FORMAT, day0: datetime.date | None = None):
        self.time_format = time_format
        self.day0 = day0
        self._precision = 0
        self._with_day: set[int] = set()

    def set_locs(self, locs):
        super().set_locs(locs)
        self._precision = clock_precision(locs)
        days = [math.floor(round(float(x) * 3600.0, 6) / SECONDS_PER_DAY) for x in locs]
        self._with_day = {i for i, day in enumerate(days) if i == 0 or day != days[i - 1]}

    def __call__(self, x, pos=None):
        return clock_text(x, self._precision, self.time_format, self.day0,
                          with_day=pos is None or pos in self._with_day)

    def format_data_short(self, value):
        return clock_text(value, 2, self.time_format, self.day0)
