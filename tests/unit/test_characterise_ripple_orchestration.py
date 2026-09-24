"""Unit tests for characterise_ripple.characterise_ripple (RC-10).

Uses synthetic vis-shaped fixtures with a known injected power law + known
injected ripple, broadcast identically across all rows so the weighted
vector-average recovers the injected spectrum exactly (isolating this
orchestration test from get_vector_avg_spectrum's own weighting logic,
which RC-08 already tests separately).
"""

import numpy as np
import pytest

from characterise_ripple import characterise_ripple


def _make_vis_with_known_spectrum(make_synthetic_vis_corrected, freqs_hz, target_spectrum_jy, pols=('RR',), nrows=5):
    npol = len(pols)
    vis_complex = np.broadcast_to(
        target_spectrum_jy[None, :, None], (nrows, freqs_hz.size, npol)
    ).copy().astype(np.complex128)

    vis = make_synthetic_vis_corrected(nrows=nrows, nchan=freqs_hz.size, pols=pols, vis_complex=vis_complex)
    vis['freqs_hz'] = freqs_hz
    return vis


def test_end_to_end_recovers_known_injected_ripple(make_synthetic_vis_corrected, make_synthetic_spectrum):
    freqs_hz = np.linspace(300.0, 332.0, 256) * 1e6
    data = make_synthetic_spectrum(
        freqs_hz, alpha=-0.7, beta=0.0,
        ripple_components=[{'period_mhz': 8.0, 'amplitude': 0.05}],
        noise_sigma_jy=1e-4, seed=0,
    )

    vis = _make_vis_with_known_spectrum(make_synthetic_vis_corrected, freqs_hz, data['spectrum_jy'])

    result = characterise_ripple(vis, solution=None, source='UNREGISTEREDSRC', physical_model_mode='fit')

    assert result['known_model_used'] is False
    assert result['known_model_local_alpha_crosscheck'] is None  # no registry entry for this source

    components = result['per_pol']['RR']['components']
    assert len(components) >= 1
    closest = min(components, key=lambda c: abs(c['period_mhz'] - 8.0))
    assert abs(closest['period_mhz'] - 8.0) / 8.0 < 0.05
    assert abs(closest['amplitude'] - 0.05) / 0.05 < 0.1
    assert closest['cable_length_m'] > 0
    assert isinstance(closest['significant'], bool)

    # A single non-iterative power-law fit performed directly on
    # ripple-contaminated data is expected to be somewhat biased by the
    # ripple itself (this is exactly why real pipelines iterate). The
    # precise, ripple-free numerical accuracy of the fit itself is already
    # covered by RC-04's dedicated tests; here we only sanity-check the
    # orchestration recovered something in the right ballpark.
    power_law_fit = result['per_pol']['RR']['power_law_fit']
    assert abs(power_law_fit['alpha_nu0'] - (-0.7)) < 0.3


def test_known_model_mode_populates_crosscheck_for_registered_source(make_synthetic_vis_corrected, make_synthetic_spectrum):
    freqs_hz = np.linspace(300.0, 332.0, 256) * 1e6
    data = make_synthetic_spectrum(freqs_hz, alpha=-0.7, beta=0.0)
    vis = _make_vis_with_known_spectrum(make_synthetic_vis_corrected, freqs_hz, data['spectrum_jy'])

    result = characterise_ripple(vis, solution=None, source='3C48', physical_model_mode='known')

    assert result['known_model_used'] is True
    assert result['known_model_local_alpha_crosscheck'] is not None
    assert 'RR' in result['known_model_local_alpha_crosscheck']
    crosscheck = result['known_model_local_alpha_crosscheck']['RR']
    assert 'alpha_known_nu0' in crosscheck
    assert 'alpha_local_fit_nu0' in crosscheck
    assert 'delta' in crosscheck


def test_crosscheck_always_computed_for_registered_source_even_in_fit_mode(
    make_synthetic_vis_corrected, make_synthetic_spectrum,
):
    # This is the core "never silently trust a possibly-wrong known model"
    # design requirement: the crosscheck must be populated regardless of
    # physical_model_mode whenever the source has a registered flux model.
    freqs_hz = np.linspace(300.0, 332.0, 256) * 1e6
    data = make_synthetic_spectrum(freqs_hz, alpha=-0.7, beta=0.0)
    vis = _make_vis_with_known_spectrum(make_synthetic_vis_corrected, freqs_hz, data['spectrum_jy'])

    result = characterise_ripple(vis, solution=None, source='3C48', physical_model_mode='fit')

    assert result['known_model_used'] is False
    assert result['known_model_local_alpha_crosscheck'] is not None


def test_known_mode_raises_for_unregistered_source(make_synthetic_vis_corrected, make_synthetic_spectrum):
    freqs_hz = np.linspace(300.0, 332.0, 256) * 1e6
    data = make_synthetic_spectrum(freqs_hz, alpha=-0.7, beta=0.0)
    vis = _make_vis_with_known_spectrum(make_synthetic_vis_corrected, freqs_hz, data['spectrum_jy'])

    with pytest.raises(ValueError):
        characterise_ripple(vis, solution=None, source='NOT_A_REAL_SOURCE', physical_model_mode='known')


def test_invalid_physical_model_mode_raises(make_synthetic_vis_corrected, make_synthetic_spectrum):
    freqs_hz = np.linspace(300.0, 332.0, 256) * 1e6
    data = make_synthetic_spectrum(freqs_hz, alpha=-0.7, beta=0.0)
    vis = _make_vis_with_known_spectrum(make_synthetic_vis_corrected, freqs_hz, data['spectrum_jy'])

    with pytest.raises(ValueError):
        characterise_ripple(vis, solution=None, source='UNREGISTEREDSRC', physical_model_mode='bogus')


def test_manual_period_bounds_without_both_values_raises(make_synthetic_vis_corrected, make_synthetic_spectrum):
    freqs_hz = np.linspace(300.0, 332.0, 256) * 1e6
    data = make_synthetic_spectrum(freqs_hz, alpha=-0.7, beta=0.0)
    vis = _make_vis_with_known_spectrum(make_synthetic_vis_corrected, freqs_hz, data['spectrum_jy'])

    with pytest.raises(ValueError):
        characterise_ripple(
            vis, solution=None, source='UNREGISTEREDSRC',
            period_bounds_mode='manual', period_min_mhz=5.0,  # period_max_mhz missing
        )


def test_manual_period_bounds_used_when_provided(make_synthetic_vis_corrected, make_synthetic_spectrum):
    freqs_hz = np.linspace(300.0, 332.0, 256) * 1e6
    data = make_synthetic_spectrum(freqs_hz, alpha=-0.7, beta=0.0)
    vis = _make_vis_with_known_spectrum(make_synthetic_vis_corrected, freqs_hz, data['spectrum_jy'])

    result = characterise_ripple(
        vis, solution=None, source='UNREGISTEREDSRC',
        period_bounds_mode='manual', period_min_mhz=5.0, period_max_mhz=20.0,
    )

    assert result['period_bounds_used_mhz'] == {'min': 5.0, 'max': 20.0}
