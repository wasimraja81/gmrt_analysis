"""Unit tests for ripple_characterisation.mad_sigma (RC-03)."""

import numpy as np

from ripple_characterisation import mad_sigma


def test_recovers_known_sigma_on_gaussian_noise():
    rng = np.random.default_rng(0)
    sigma_true = 0.02
    values = rng.normal(0.0, sigma_true, size=1000)

    sigma_hat = mad_sigma(values)

    assert abs(sigma_hat - sigma_true) / sigma_true < 0.15


def test_recovers_known_sigma_at_smaller_n_with_wider_tolerance():
    rng = np.random.default_rng(1)
    sigma_true = 0.02
    values = rng.normal(0.0, sigma_true, size=100)

    sigma_hat = mad_sigma(values)

    assert abs(sigma_hat - sigma_true) / sigma_true < 0.35


def test_robust_to_single_extreme_outlier():
    rng = np.random.default_rng(2)
    sigma_true = 0.02
    values = rng.normal(0.0, sigma_true, size=1000)

    sigma_clean = mad_sigma(values)
    std_clean = float(np.std(values))

    contaminated = values.copy()
    contaminated[0] = 100 * sigma_true

    sigma_contaminated = mad_sigma(contaminated)
    std_contaminated = float(np.std(contaminated))

    # MAD-based estimate barely moves...
    assert abs(sigma_contaminated - sigma_clean) / sigma_clean < 0.05
    # ...while a naive std shifts noticeably from a single outlier in a
    # 1000-sample array, documenting *why* MAD is used here instead.
    assert std_contaminated / std_clean > 3.0


def test_all_nan_returns_nan():
    values = np.full(10, np.nan)
    assert np.isnan(mad_sigma(values))


def test_empty_array_returns_nan():
    assert np.isnan(mad_sigma(np.array([])))


def test_constant_array_returns_exactly_zero():
    values = np.full(20, 5.0)
    assert mad_sigma(values) == 0.0


def test_ignores_nan_values_mixed_with_finite():
    rng = np.random.default_rng(3)
    sigma_true = 0.02
    values = rng.normal(0.0, sigma_true, size=500)
    values_with_nan = values.copy()
    values_with_nan[::10] = np.nan

    sigma_hat = mad_sigma(values_with_nan)

    assert abs(sigma_hat - sigma_true) / sigma_true < 0.2
