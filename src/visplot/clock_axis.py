"""Tick placement and labels for an axis in hours of clock time (recorded,
UTC, local observatory time, LST).

`ClockLocator` puts ticks at whole clock steps (1 s ... 30 min, 1 h ... 12 h,
days); `ClockFormatter` labels them dd:hh:mm:ss, the day counted from the
axis's day 0 (hours 24 to 48 are day 01), with fractions of a second when
the ticks need them.
"""

from __future__ import annotations

import math

import numpy as np
from matplotlib.ticker import Formatter, Locator, MaxNLocator

CLOCK_STEPS_S = (1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800,
                 3600, 7200, 10800, 21600, 43200, 86400)
SECONDS_PER_DAY = 86400


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


def clock_text(hours: float, precision: int) -> str:
    """`hours` (since 0h of day 0) as dd:hh:mm:ss, with `precision` digits
    of seconds after the point."""
    scale = 10 ** precision
    q = round(hours * 3600.0 * scale)
    day, q = divmod(q, SECONDS_PER_DAY * scale)
    h, q = divmod(q, 3600 * scale)
    m, q = divmod(q, 60 * scale)
    s, frac = divmod(q, scale)
    text = f"{h:02d}:{m:02d}:{s:02d}" + (f".{frac:0{precision}d}" if precision else "")
    return f"{day:02d}:{text}" if day >= 0 else f"-{-day:02d}:{text}"


class ClockFormatter(Formatter):
    """Tick labels at the precision the ticks need, all alike; the cursor
    readout to a hundredth of a second."""

    def __init__(self):
        self._precision = 0

    def set_locs(self, locs):
        super().set_locs(locs)
        self._precision = clock_precision(locs)

    def __call__(self, x, pos=None):
        return clock_text(x, self._precision)

    def format_data_short(self, value):
        return clock_text(value, 2)
