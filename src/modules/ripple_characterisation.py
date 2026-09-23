"""Pure-numerics core for standing-wave ripple characterisation.

Every function in this module takes and returns NumPy arrays and plain
dicts only — no ``vis`` dicts, no matplotlib, no file I/O. This is what the
pytest suite exercises directly with synthetic ground truth, with zero
UVFITS/CASA dependency. Orchestration (loading real data, applying a
bandpass solution, writing plots/JSON/reports) lives in
``src/characterise_ripple.py``.

See ``ripple_characterisation_tickets.md`` for the ticket-by-ticket design
of each function, and ``ripple_convergence_todos.md`` (Step 1 / Step 1b) for
the underlying physical model.
"""

from __future__ import annotations

import numpy as np

_MIN_POWER_LAW_POINTS = 10


def mad_sigma(values: np.ndarray) -> float:
    """Robust noise-floor estimate via the median absolute deviation.

    Implements the noise floor convention from ``ripple_convergence_todos.md``
    Step 1:

        sigma_hat = 1.4826 * MAD(values)

    where MAD is the median absolute deviation from the median. The 1.4826
    factor converts MAD to a Gaussian-equivalent standard deviation.

    Parameters
    ----------
    values : np.ndarray
        1D array, may contain NaN (NaNs are ignored).

    Returns
    -------
    float
        ``sigma_hat``, or ``nan`` if ``values`` is empty or entirely NaN.
    """
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float('nan')
    median = np.median(finite)
    mad = np.median(np.abs(finite - median))
    return float(1.4826 * mad)


def fit_power_law_spectrum(
    freqs_hz: np.ndarray,
    spectrum_jy: np.ndarray,
    allow_curvature: bool = True,
) -> dict:
    """Fit a power-law-with-curvature continuum to a real-valued spectrum.

    Implements the Step 1b physical model, parametrised around a reference
    frequency ``nu0_hz`` (the geometric mean of the valid input frequencies)
    for numerical conditioning:

        log10(S(nu)) = a0 + a1*x + a2*x^2,   x = log10(nu / nu0)

    so that ``a1`` is exactly the local spectral index at ``nu0`` (since
    ``x=0`` there) and ``a2`` is the curvature term. When
    ``allow_curvature=False``, ``a2`` is forced to 0 (a plain power law).

    Parameters
    ----------
    freqs_hz : np.ndarray
        Frequency grid in Hz.
    spectrum_jy : np.ndarray
        Real-valued flux spectrum in Jy, same shape as ``freqs_hz``. Values
        that are non-finite or non-positive are excluded from the fit (a
        log-log fit cannot use them).
    allow_curvature : bool
        If True (default), fit the quadratic (curved) model; if False, fit
        a plain power law (``a2 = 0``).

    Returns
    -------
    dict with keys:
        ``coeffs``: ``(a0, a1, a2)`` in the ``nu0``-centred log10 parametrisation.
        ``nu0_hz``: the reference frequency used.
        ``alpha_nu0``: local spectral index at ``nu0`` (equals ``a1``).
        ``beta``: curvature term (equals ``a2``; ``0.0`` if ``allow_curvature=False``).
        ``model_jy``: the fitted model evaluated at every input ``freqs_hz``
            (including points excluded from the fit itself).
        ``rss_log``: residual sum of squares in log10-space, over the valid
            (fitted) points only.
        ``mask_n``: number of valid points used in the fit.

    Raises
    ------
    ValueError
        If fewer than 10 valid (finite, positive) points are available.
    """
    freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
    spectrum_jy = np.asarray(spectrum_jy, dtype=np.float64)

    valid = np.isfinite(freqs_hz) & np.isfinite(spectrum_jy) & (freqs_hz > 0) & (spectrum_jy > 0)
    mask_n = int(np.sum(valid))
    if mask_n < _MIN_POWER_LAW_POINTS:
        raise ValueError(
            f'fit_power_law_spectrum: need at least {_MIN_POWER_LAW_POINTS} valid '
            f'(finite, positive) points, got {mask_n}.'
        )

    nu0_hz = float(np.sqrt(freqs_hz[valid].min() * freqs_hz[valid].max()))
    x_all = np.log10(freqs_hz / nu0_hz)
    y_all = np.log10(np.where(spectrum_jy > 0, spectrum_jy, np.nan))

    x_fit = x_all[valid]
    y_fit = y_all[valid]

    degree = 2 if allow_curvature else 1
    poly_coeffs = np.polyfit(x_fit, y_fit, degree)
    if allow_curvature:
        a2, a1, a0 = poly_coeffs
    else:
        a1, a0 = poly_coeffs
        a2 = 0.0

    model_log_jy = a0 + a1 * x_all + a2 * x_all**2
    model_jy = 10.0**model_log_jy

    y_pred_fit = a0 + a1 * x_fit + a2 * x_fit**2
    rss_log = float(np.sum((y_fit - y_pred_fit) ** 2))

    return {
        'coeffs': (float(a0), float(a1), float(a2)),
        'nu0_hz': nu0_hz,
        'alpha_nu0': float(a1),
        'beta': float(a2),
        'model_jy': model_jy,
        'rss_log': rss_log,
        'mask_n': mask_n,
    }
