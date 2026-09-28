"""Percentiles of a stream of values, in fixed memory.

Values are counted in bins of equal width in log10 of their magnitude, one
set for each sign plus one for zero, so the same bins serve amplitudes of
1e-6 and of 1e6 alike. With `BINS_PER_DECADE` = 1000 a percentile is found
to within 0.23% of its value; the bins take ~0.5 MB whatever the number of
values.
"""

from __future__ import annotations

import numpy as np

DECADE_MIN = -15  # magnitudes below 1e-15 count as zero
DECADE_MAX = 15  # magnitudes above 1e15 count in the top bin
BINS_PER_DECADE = 1000
_N = (DECADE_MAX - DECADE_MIN) * BINS_PER_DECADE  # bins per sign


class ValueHistogram:
    """Counts of values in log-magnitude bins, ordered by value: most
    negative first, then zero, then positive."""

    def __init__(self):
        self.counts = np.zeros(2 * _N + 1, dtype=np.int64)

    def bin_indices(self, values: np.ndarray) -> np.ndarray:
        magnitude = np.abs(values)
        with np.errstate(divide="ignore"):
            k = np.floor((np.log10(magnitude) - DECADE_MIN) * BINS_PER_DECADE)
        k = np.clip(np.nan_to_num(k, nan=-1, neginf=-1), -1, _N - 1).astype(np.int64)
        index = np.where(values > 0, _N + 1 + k, _N - 1 - k)
        index[k < 0] = _N  # zero, or too small to tell from zero
        return index

    def add(self, values: np.ndarray) -> None:
        if values.size:
            self.counts += np.bincount(self.bin_indices(values), minlength=self.counts.size)

    def merge(self, other: "ValueHistogram") -> None:
        self.counts += other.counts

    def percentile(self, q: float, upper: bool) -> float:
        """The value at percentile `q` (0-100) in the inverted-CDF sense (the
        smallest value with at least q% of all values at or below it), to bin
        precision: the bin's lower edge (`upper=False`) or upper edge
        (`upper=True`), so a range from `percentile(lo, False)` to
        `percentile(hi, True)` contains at least the requested fraction."""
        total = self.counts.sum()
        if total == 0:
            raise ValueError("no values counted")
        target = q / 100.0 * total
        index = int(np.searchsorted(np.cumsum(self.counts), max(target, 1), side="left"))
        return _bin_edge(index, upper)


def _bin_edge(index: int, upper: bool) -> float:
    if index == _N:
        return 0.0
    if index > _N:
        k = index - _N - 1
        return 10.0 ** (DECADE_MIN + (k + (1 if upper else 0)) / BINS_PER_DECADE)
    k = _N - 1 - index
    return -(10.0 ** (DECADE_MIN + (k + (0 if upper else 1)) / BINS_PER_DECADE))
