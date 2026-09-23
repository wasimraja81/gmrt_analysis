"""Unit tests for ripple_characterisation.fit_power_law_spectrum (RC-04)."""

import numpy as np
import pytest

from ripple_characterisation import fit_power_law_spectrum


def _freqs_hz(n=128, start_mhz=300.0, span_mhz=32.0):
    return np.linspace(start_mhz, start_mhz + span_mhz, n) * 1e6


def test_clean_power_law_no_curvature_no_noise_exact_recovery(make_synthetic_spectrum):
    freqs_hz = _freqs_hz()
    data = make_synthetic_spectrum(freqs_hz, alpha=-0.7, beta=0.0)

    result = fit_power_law_spectrum(freqs_hz, data['spectrum_jy'], allow_curvature=True)

    assert abs(result['alpha_nu0'] - (-0.7)) < 1e-6
    assert abs(result['beta']) < 1e-6


def test_curved_noisy_data_recovers_coeffs_within_tolerance(make_synthetic_spectrum):
    # Curvature (beta) is only well-constrained by data spanning a wide
    # fractional bandwidth (x=log10(nu/nu0) needs to vary enough that x^2
    # carries real signal). GMRT's actual ~16 MHz/300 MHz (~5%) band is far
    # too narrow for this — deliberately use an octave-plus span here so
    # this test is actually testing curvature recovery, not noise floor.
    freqs_hz = np.linspace(100.0, 800.0, 200) * 1e6
    true_alpha, true_beta = -0.5, 0.1
    data = make_synthetic_spectrum(
        freqs_hz, alpha=true_alpha, beta=true_beta, noise_sigma_jy=0.0, seed=0,
    )
    # Add noise directly in log-space (matches the model's own noise
    # convention: Gaussian scatter around the log-log fit).
    rng = np.random.default_rng(0)
    noisy_spectrum = data['spectrum_jy'] * 10.0 ** rng.normal(0.0, 0.01, size=freqs_hz.shape)

    result = fit_power_law_spectrum(freqs_hz, noisy_spectrum, allow_curvature=True)

    assert abs(result['alpha_nu0'] - true_alpha) < 0.05
    assert abs(result['beta'] - true_beta) < 0.05


def test_curvature_off_has_higher_rss_on_curved_input(make_synthetic_spectrum):
    freqs_hz = _freqs_hz()
    data = make_synthetic_spectrum(freqs_hz, alpha=-0.5, beta=0.3)

    curved_fit = fit_power_law_spectrum(freqs_hz, data['spectrum_jy'], allow_curvature=True)
    linear_fit = fit_power_law_spectrum(freqs_hz, data['spectrum_jy'], allow_curvature=False)

    assert curved_fit['rss_log'] < linear_fit['rss_log']
    assert linear_fit['beta'] == 0.0


def test_raises_on_too_few_valid_points():
    freqs_hz = _freqs_hz(n=5)
    spectrum_jy = np.ones(5)
    with pytest.raises(ValueError):
        fit_power_law_spectrum(freqs_hz, spectrum_jy)


def test_masks_out_non_positive_flux_values(make_synthetic_spectrum):
    freqs_hz = _freqs_hz()
    data = make_synthetic_spectrum(freqs_hz, alpha=-0.7, beta=0.0)
    spectrum = data['spectrum_jy'].copy()
    spectrum[0] = -5.0
    spectrum[1] = 0.0

    result = fit_power_law_spectrum(freqs_hz, spectrum, allow_curvature=True)

    assert result['mask_n'] == freqs_hz.size - 2
    assert abs(result['alpha_nu0'] - (-0.7)) < 1e-6


def test_raises_on_all_nan_spectrum():
    freqs_hz = _freqs_hz()
    spectrum_jy = np.full(freqs_hz.shape, np.nan)
    with pytest.raises(ValueError):
        fit_power_law_spectrum(freqs_hz, spectrum_jy)


def test_model_jy_has_full_length_including_masked_points(make_synthetic_spectrum):
    freqs_hz = _freqs_hz()
    data = make_synthetic_spectrum(freqs_hz, alpha=-0.7, beta=0.0)
    spectrum = data['spectrum_jy'].copy()
    spectrum[0] = -5.0

    result = fit_power_law_spectrum(freqs_hz, spectrum, allow_curvature=True)

    assert result['model_jy'].shape == freqs_hz.shape
    assert np.all(np.isfinite(result['model_jy']))
