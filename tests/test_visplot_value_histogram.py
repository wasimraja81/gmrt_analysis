import numpy as np
import pytest

from visplot.value_histogram import ValueHistogram


def _hist(values):
    h = ValueHistogram()
    h.add(np.asarray(values, dtype=float))
    return h


def test_percentiles_match_numpy_to_bin_precision():
    rng = np.random.default_rng(0)
    values = np.concatenate([rng.lognormal(3, 2, 100_000), -rng.lognormal(1, 1, 20_000), np.zeros(500)])
    h = _hist(values)
    for q in (0.1, 1, 25, 50, 75, 99, 99.9):
        exact = np.percentile(values, q, method="inverted_cdf")  # an order statistic, as the histogram gives
        lower, upper = h.percentile(q, upper=False), h.percentile(q, upper=True)
        assert lower <= exact <= upper or abs(exact) < 1e-15
        assert abs(upper - lower) <= abs(exact) * 0.0024 + 1e-15  # one bin: 10**(1/1000) - 1 = 0.23%


def test_a_range_from_lower_and_upper_edges_contains_the_requested_fraction():
    rng = np.random.default_rng(1)
    values = rng.normal(0, 50, 200_000)
    h = _hist(values)
    lo, hi = h.percentile(0.5, upper=False), h.percentile(99.5, upper=True)
    assert np.mean((values >= lo) & (values <= hi)) >= 0.99


def test_order_runs_from_most_negative_through_zero_to_most_positive():
    h = _hist([-100.0, -1.0, 0.0, 1.0, 100.0])
    assert h.percentile(1, upper=False) < -99 and h.percentile(99, upper=True) > 99
    assert h.percentile(50, upper=False) == 0.0


def test_merge_adds_counts():
    a, b = _hist([1.0, 2.0]), _hist([3.0])
    a.merge(b)
    assert a.counts.sum() == 3


def test_empty_histogram_has_no_percentile():
    with pytest.raises(ValueError, match="no values"):
        ValueHistogram().percentile(50, upper=False)
