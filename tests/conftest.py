"""Shared synthetic-data fixtures for the ripple-characterisation test suite.

No real UVFITS/CASA data is used anywhere in this directory tree — every
fixture here builds small, hand-computable, NumPy-only structures with known
ground truth, so unit tests can assert exact or tightly-bounded numerical
outcomes.
"""

import numpy as np
import pytest


@pytest.fixture
def make_synthetic_spectrum():
    """Factory fixture: build a synthetic real-valued flux spectrum.

    Returns a callable ``make(freqs_hz, alpha, **kwargs) -> dict`` producing:

        S(nu) = S0 * (nu/nu0)^(alpha + beta*log10(nu/nu0)) * (1 + ripple(nu)) + noise

    Parameters (of the returned callable)
    --------------------------------------
    freqs_hz : array-like
        Frequency grid in Hz.
    alpha : float
        Spectral index at the reference frequency ``nu0_hz``.
    beta : float, optional
        Curvature term (default 0.0).
    S0_jy : float, optional
        Flux at ``nu0_hz`` before ripple/noise (default 1.0).
    nu0_hz : float, optional
        Reference frequency in Hz. Defaults to the geometric mean of
        ``freqs_hz``.
    ripple_components : list of dict, optional
        Each dict: ``{'period_mhz': ..., 'amplitude': ..., 'phase_rad': ...}``
        (amplitude is fractional, i.e. of the residual
        ``r(nu) = S(nu)/S_PL(nu) - 1``).
    noise_sigma_jy : float, optional
        Gaussian noise sigma added to the flux (default 0.0, no noise).
    seed : int, optional
        RNG seed for the noise draw (default 0).

    Returns
    -------
    dict with keys: ``freqs_hz``, ``spectrum_jy`` (noisy), ``model_jy``
    (noise-free, ripple-free power law), ``ripple_true`` (exact fractional
    ripple added, no noise), ``nu0_hz``.
    """

    def _make(
        freqs_hz,
        alpha,
        *,
        beta=0.0,
        S0_jy=1.0,
        nu0_hz=None,
        ripple_components=None,
        noise_sigma_jy=0.0,
        seed=0,
    ):
        freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
        if nu0_hz is None:
            nu0_hz = float(np.sqrt(freqs_hz.min() * freqs_hz.max()))

        x = np.log10(freqs_hz / nu0_hz)
        log_model = np.log10(S0_jy) + alpha * x + beta * x**2
        model_jy = 10.0**log_model

        freqs_mhz = freqs_hz / 1e6
        ripple_true = np.zeros_like(freqs_hz)
        if ripple_components:
            for comp in ripple_components:
                period_mhz = comp['period_mhz']
                amplitude = comp['amplitude']
                phase_rad = comp.get('phase_rad', 0.0)
                ripple_true = ripple_true + amplitude * np.sin(
                    2.0 * np.pi * freqs_mhz / period_mhz + phase_rad
                )

        spectrum_jy = model_jy * (1.0 + ripple_true)

        if noise_sigma_jy > 0:
            rng = np.random.default_rng(seed)
            spectrum_jy = spectrum_jy + rng.normal(0.0, noise_sigma_jy, size=freqs_hz.shape)

        return {
            'freqs_hz': freqs_hz,
            'spectrum_jy': spectrum_jy,
            'model_jy': model_jy,
            'ripple_true': ripple_true,
            'nu0_hz': nu0_hz,
        }

    return _make


@pytest.fixture
def make_synthetic_vis_corrected():
    """Factory fixture: build a small vis_corrected-shaped dict.

    Matches the shape ``apply_bandpass_solution()`` produces (and that
    ``get_vector_avg_spectrum()`` consumes): a dict with ``stokes_labels``,
    ``vis_complex_corrected``, ``weight``, ``flagged_corrected``, each of
    shape ``(nrows, nchan, npol)`` except ``stokes_labels`` (length ``npol``).

    Returns a callable ``make(nrows=4, nchan=3, pols=('RR','LL'), seed=42,
    vis_complex=None, weight=None, flagged=None) -> dict``. When
    ``vis_complex``/``weight``/``flagged`` are omitted, small random/default
    arrays are generated (real+imag ~ N(0,1), weight=1, nothing flagged) so
    tests can either supply exact known values or use a quick random fixture
    for shape/plumbing checks.
    """

    def _make(nrows=4, nchan=3, pols=('RR', 'LL'), seed=42,
               vis_complex=None, weight=None, flagged=None):
        npol = len(pols)
        rng = np.random.default_rng(seed)

        if vis_complex is None:
            vis_complex = (
                rng.normal(0, 1, size=(nrows, nchan, npol))
                + 1j * rng.normal(0, 1, size=(nrows, nchan, npol))
            )
        if weight is None:
            weight = np.ones((nrows, nchan, npol), dtype=np.float64)
        if flagged is None:
            flagged = np.zeros((nrows, nchan, npol), dtype=bool)

        return {
            'stokes_labels': list(pols),
            'vis_complex_corrected': np.asarray(vis_complex, dtype=np.complex128),
            'weight': np.asarray(weight, dtype=np.float64),
            'flagged_corrected': np.asarray(flagged, dtype=bool),
        }

    return _make
