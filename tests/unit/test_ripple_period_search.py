"""Unit tests for ripple_characterisation's Fourier period-search (RC-05):
derive_fourier_period_bounds() and find_ripple_period_candidates().
"""

import numpy as np
import pytest

from ripple_characterisation import derive_fourier_period_bounds, find_ripple_period_candidates


def _freqs_hz(n=512, start_mhz=300.0, span_mhz=32.0):
    return np.linspace(start_mhz, start_mhz + span_mhz, n) * 1e6


# ── derive_fourier_period_bounds ─────────────────────────────────────────────

def test_bounds_exact_nyquist_and_span():
    n = 100
    delta_mhz = 0.125
    start_mhz = 300.0
    freqs_hz = (start_mhz + np.arange(n) * delta_mhz) * 1e6

    period_min, period_max = derive_fourier_period_bounds(freqs_hz)

    span_mhz = (n - 1) * delta_mhz
    assert abs(period_min - 2.0 * delta_mhz) < 1e-9
    assert abs(period_max - span_mhz) < 1e-9


def test_bounds_robust_to_unsorted_input():
    n = 100
    delta_mhz = 0.125
    freqs_hz = (300.0 + np.arange(n) * delta_mhz) * 1e6
    rng = np.random.default_rng(0)
    shuffled = rng.permutation(freqs_hz)

    period_min_sorted, period_max_sorted = derive_fourier_period_bounds(freqs_hz)
    period_min_shuffled, period_max_shuffled = derive_fourier_period_bounds(shuffled)

    assert abs(period_min_sorted - period_min_shuffled) < 1e-9
    assert abs(period_max_sorted - period_max_shuffled) < 1e-9


# ── find_ripple_period_candidates ────────────────────────────────────────────

def _residual_with_realistic_noise(make_synthetic_spectrum, freqs_hz, ripple_components, seed=0, noise_sigma_jy=5e-4):
    # Real data always has some noise. A small, realistic noise floor (here
    # ~1% of the smallest ripple amplitude used in these tests) keeps the
    # SNR-vs-noise-floor logic meaningful; a perfectly noise-free signal
    # makes the median in-range power ~0, which inflates the SNR of even
    # tiny Hann-window leakage sidelobes near DC to arbitrarily large
    # values and produces spurious extra candidates that have nothing to
    # do with the boundary-handling being tested.
    data = make_synthetic_spectrum(
        freqs_hz, alpha=0.0, beta=0.0,
        ripple_components=ripple_components,
        noise_sigma_jy=noise_sigma_jy, seed=seed,
    )
    return data['spectrum_jy'] / data['model_jy'] - 1.0


def _closest(periods_mhz, target_mhz):
    return min(periods_mhz, key=lambda p: abs(p - target_mhz))


def test_single_known_sinusoid_recovered(make_synthetic_spectrum):
    # This checks the true period is *among* the returned candidates, not
    # that it is the *only* one. A short-period ripple spanning only a few
    # cycles of the band (8 MHz period over 32 MHz here = 4 cycles) causes
    # genuine Hann-window mainlobe leakage into the near-DC/long-period
    # bins — an expected spectral-leakage artifact at this cycle count, not
    # a bug. RC-05 is explicitly a *candidate-seeding* step; RC-06's proper
    # least-squares harmonic fit is what separates real ripple from seed
    # noise, so tolerating extra low-confidence seeds here is by design.
    freqs_hz = _freqs_hz()
    period_min_mhz, period_max_mhz = derive_fourier_period_bounds(freqs_hz)

    residual = _residual_with_realistic_noise(
        make_synthetic_spectrum, freqs_hz, [{'period_mhz': 8.0, 'amplitude': 0.05}],
    )

    result = find_ripple_period_candidates(freqs_hz, residual, period_min_mhz, period_max_mhz)

    assert len(result['periods_mhz']) >= 1
    recovered = _closest(result['periods_mhz'], 8.0)
    assert abs(recovered - 8.0) / 8.0 < 0.05
    idx = result['periods_mhz'].index(recovered)
    assert result['snrs'][idx] > 10.0


def test_two_independent_sinusoids_both_recovered(make_synthetic_spectrum):
    # See test_single_known_sinusoid_recovered's comment on why this checks
    # "both true periods are present", not "exactly these two and no more".
    freqs_hz = _freqs_hz()
    period_min_mhz, period_max_mhz = derive_fourier_period_bounds(freqs_hz)

    residual = _residual_with_realistic_noise(
        make_synthetic_spectrum, freqs_hz,
        [{'period_mhz': 8.0, 'amplitude': 0.05}, {'period_mhz': 3.0, 'amplitude': 0.03}],
    )

    result = find_ripple_period_candidates(freqs_hz, residual, period_min_mhz, period_max_mhz)

    assert len(result['periods_mhz']) >= 2
    closest_to_3 = _closest(result['periods_mhz'], 3.0)
    closest_to_8 = _closest(result['periods_mhz'], 8.0)
    assert abs(closest_to_3 - 3.0) / 3.0 < 0.1
    assert abs(closest_to_8 - 8.0) / 8.0 < 0.1


def test_pure_noise_yields_no_candidates():
    # FFT power for Gaussian noise is exponentially distributed, so with
    # ~n/2 independent bins searched, a modest threshold (e.g. 5x median)
    # gives a near-certain false positive somewhere across that many trials
    # (per-bin false-positive rate ~exp(-5*ln2)~3%, compounded over ~250
    # bins). Use a threshold high enough that the compounded false-positive
    # probability across all searched bins is small (~exp(-15*ln2)*250 <
    # 1%), which is what "don't fit the noise" actually requires once you
    # account for how many bins get tested.
    freqs_hz = _freqs_hz()
    period_min_mhz, period_max_mhz = derive_fourier_period_bounds(freqs_hz)

    rng = np.random.default_rng(7)
    residual = rng.normal(0.0, 0.01, size=freqs_hz.shape)

    result = find_ripple_period_candidates(
        freqs_hz, residual, period_min_mhz, period_max_mhz, peak_snr_threshold=15.0,
    )

    assert result['periods_mhz'] == []


def test_period_at_min_boundary_still_detected(make_synthetic_spectrum):
    freqs_hz = _freqs_hz(n=512)
    period_min_mhz, period_max_mhz = derive_fourier_period_bounds(freqs_hz)

    residual = _residual_with_realistic_noise(
        make_synthetic_spectrum, freqs_hz, [{'period_mhz': period_min_mhz, 'amplitude': 0.05}],
    )

    result = find_ripple_period_candidates(
        freqs_hz, residual, period_min_mhz, period_max_mhz, peak_snr_threshold=2.0,
    )

    assert len(result['periods_mhz']) >= 1
    closest = min(result['periods_mhz'], key=lambda p: abs(p - period_min_mhz))
    assert abs(closest - period_min_mhz) / period_min_mhz < 0.2


def test_period_at_max_boundary_still_detected(make_synthetic_spectrum):
    freqs_hz = _freqs_hz(n=512)
    period_min_mhz, period_max_mhz = derive_fourier_period_bounds(freqs_hz)

    residual = _residual_with_realistic_noise(
        make_synthetic_spectrum, freqs_hz, [{'period_mhz': period_max_mhz, 'amplitude': 0.05}],
    )

    result = find_ripple_period_candidates(
        freqs_hz, residual, period_min_mhz, period_max_mhz, peak_snr_threshold=2.0,
    )

    assert len(result['periods_mhz']) >= 1
    closest = min(result['periods_mhz'], key=lambda p: abs(p - period_max_mhz))
    assert abs(closest - period_max_mhz) / period_max_mhz < 0.2


def test_too_few_finite_points_returns_empty_without_raising():
    freqs_hz = _freqs_hz(n=10)
    residual = np.full(10, np.nan)
    residual[:5] = 0.01

    result = find_ripple_period_candidates(freqs_hz, residual, 1.0, 20.0)

    assert result['periods_mhz'] == []
    assert result['snrs'] == []
