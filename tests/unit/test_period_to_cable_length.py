"""Unit tests for ripple_characterisation.period_to_cable_length_m (RC-07)."""

import math

import numpy as np

from ripple_characterisation import period_to_cable_length_m

_C = 299792458.0


def test_exact_closed_form_value_at_10mhz():
    length_m = period_to_cable_length_m(10.0, velocity_factor=1.0)
    expected = (1.0 * _C) / (2.0 * 10.0e6)
    assert math.isclose(length_m, expected, rel_tol=1e-9)
    assert math.isclose(length_m, 14.9896229, rel_tol=1e-6)


def test_linear_scaling_with_velocity_factor():
    length_full = period_to_cable_length_m(10.0, velocity_factor=1.0)
    length_scaled = period_to_cable_length_m(10.0, velocity_factor=0.66)
    assert math.isclose(length_scaled, length_full * 0.66, rel_tol=1e-9)


def test_non_positive_period_returns_nan():
    assert np.isnan(period_to_cable_length_m(0.0))
    assert np.isnan(period_to_cable_length_m(-5.0))


def test_nan_period_returns_nan():
    assert np.isnan(period_to_cable_length_m(float('nan')))


def test_non_positive_velocity_factor_returns_nan():
    assert np.isnan(period_to_cable_length_m(10.0, velocity_factor=0.0))
    assert np.isnan(period_to_cable_length_m(10.0, velocity_factor=-1.0))


def test_nan_velocity_factor_returns_nan():
    assert np.isnan(period_to_cable_length_m(10.0, velocity_factor=float('nan')))


def test_never_raises_on_degenerate_inputs():
    # The engineer report (RC-14) must not crash on a degenerate fit result.
    for period in [0.0, -1.0, float('nan'), float('inf')]:
        for vf in [0.0, -1.0, float('nan'), float('inf')]:
            result = period_to_cable_length_m(period, velocity_factor=vf)
            assert isinstance(result, float)
