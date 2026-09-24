"""Unit tests for ripple_characterisation.fit_harmonic_ripple (RC-06)."""

import numpy as np
import pytest

from ripple_characterisation import fit_harmonic_ripple


def _freqs_hz(n=256, start_mhz=300.0, span_mhz=32.0):
    return np.linspace(start_mhz, start_mhz + span_mhz, n) * 1e6


def _sinusoid_residual(freqs_hz, period_mhz, amplitude, phase_rad=0.0, noise_sigma=0.0, seed=0):
    freqs_mhz = freqs_hz / 1e6
    r = amplitude * np.sin(2.0 * np.pi * freqs_mhz / period_mhz + phase_rad)
    if noise_sigma > 0:
        rng = np.random.default_rng(seed)
        r = r + rng.normal(0.0, noise_sigma, size=freqs_hz.shape)
    return r


def test_single_sinusoid_high_snr_recovered_within_2_percent():
    freqs_hz = _freqs_hz()
    residual = _sinusoid_residual(freqs_hz, period_mhz=8.0, amplitude=0.05, noise_sigma=1e-4, seed=1)

    result = fit_harmonic_ripple(freqs_hz, residual, period_candidates=[8.0], n_harmonics=1)

    assert len(result['components']) == 1
    comp = result['components'][0]
    assert abs(comp['period_mhz'] - 8.0) / 8.0 < 0.02
    assert abs(comp['amplitude'] - 0.05) / 0.05 < 0.02
    assert result['rms_after'] < result['rms_before']


def test_fundamental_plus_exact_second_harmonic_classified(monkeypatch=None):
    freqs_hz = _freqs_hz()
    freqs_mhz = freqs_hz / 1e6
    fundamental_period = 8.0
    residual = (
        0.05 * np.sin(2.0 * np.pi * freqs_mhz / fundamental_period)
        + 0.02 * np.sin(2.0 * np.pi * freqs_mhz / (fundamental_period / 2.0))
    )

    result = fit_harmonic_ripple(
        freqs_hz, residual,
        period_candidates=[fundamental_period, fundamental_period / 2.0],
        n_harmonics=1,
    )

    by_period = {round(c['period_mhz'], 1): c for c in result['components']}
    primary = [c for c in result['components'] if c['classification'] == 'primary']
    harmonics = [c for c in result['components'] if c['classification'] == 'harmonic']

    assert len(primary) == 1
    assert abs(primary[0]['period_mhz'] - fundamental_period) / fundamental_period < 0.05
    assert len(harmonics) == 1
    assert round(harmonics[0]['ratio_to_primary']) == 2


def test_two_unrelated_periods_classified_independent():
    freqs_hz = _freqs_hz()
    residual = (
        _sinusoid_residual(freqs_hz, period_mhz=8.0, amplitude=0.05)
        + _sinusoid_residual(freqs_hz, period_mhz=5.3, amplitude=0.03)
    )

    result = fit_harmonic_ripple(freqs_hz, residual, period_candidates=[8.0, 5.3], n_harmonics=1)

    classifications = {c['classification'] for c in result['components']}
    assert classifications == {'primary', 'independent'}


def test_low_snr_does_not_crash_and_rms_never_increases():
    freqs_hz = _freqs_hz()
    residual = _sinusoid_residual(freqs_hz, period_mhz=8.0, amplitude=0.01, noise_sigma=0.01, seed=3)

    result = fit_harmonic_ripple(freqs_hz, residual, period_candidates=[8.0], n_harmonics=1)

    assert result['rms_after'] <= result['rms_before'] + 1e-12


def test_raises_on_insufficient_points():
    freqs_hz = _freqs_hz(n=10)
    residual = np.zeros(10)
    with pytest.raises(ValueError):
        fit_harmonic_ripple(freqs_hz, residual, period_candidates=[8.0], n_harmonics=1)


def test_contract_differs_from_period_search_on_insufficient_data():
    """fit_harmonic_ripple raises on too-little-data; find_ripple_period_candidates
    (RC-05) instead returns empty lists. This test documents that asymmetry
    directly (rather than relying on a comment alone) so a future change to
    either function's contract is caught here.
    """
    from ripple_characterisation import find_ripple_period_candidates

    freqs_hz = _freqs_hz(n=10)
    residual = np.zeros(10)

    period_search_result = find_ripple_period_candidates(freqs_hz, residual, 1.0, 20.0)
    assert period_search_result['periods_mhz'] == []  # does not raise

    with pytest.raises(ValueError):
        fit_harmonic_ripple(freqs_hz, residual, period_candidates=[8.0], n_harmonics=1)  # raises


def test_no_candidates_returns_zero_components_without_raising():
    freqs_hz = _freqs_hz()
    residual = np.zeros(freqs_hz.shape)

    result = fit_harmonic_ripple(freqs_hz, residual, period_candidates=[], n_harmonics=1)

    assert result['components'] == []
    assert result['rms_after'] == result['rms_before']


def test_refinement_window_never_produces_non_positive_period():
    # period_refine_frac > 1 makes the naive lower bound period*(1-frac)
    # negative; the positivity clamp inside fit_harmonic_ripple must catch
    # this and never hand a non-positive trial period to the design matrix.
    freqs_hz = _freqs_hz(n=256, span_mhz=8.0)
    tiny_period = 0.05
    residual = _sinusoid_residual(freqs_hz, period_mhz=tiny_period, amplitude=0.05)

    result = fit_harmonic_ripple(
        freqs_hz, residual, period_candidates=[tiny_period], n_harmonics=1, period_refine_frac=1.5,
    )

    assert len(result['components']) == 1
    assert result['components'][0]['period_mhz'] > 0
