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


_MIN_PERIOD_SEARCH_POINTS = 16


def derive_fourier_period_bounds(freqs_hz: np.ndarray) -> tuple:
    """Nyquist-style ripple-period search bounds for a frequency grid.

    ``period_min_mhz = 2 * channel_spacing_mhz`` — the shortest ripple
    period the channel spacing can resolve (Nyquist).
    ``period_max_mhz = total_bandwidth_mhz`` — the longest ripple period
    that fits within the observed band at all.

    Parameters
    ----------
    freqs_hz : np.ndarray
        Frequency grid in Hz (need not be pre-sorted).

    Returns
    -------
    (period_min_mhz, period_max_mhz) : tuple of float
    """
    freqs_mhz = np.sort(np.asarray(freqs_hz, dtype=np.float64)) / 1e6
    delta_mhz = float(np.median(np.diff(freqs_mhz)))
    span_mhz = float(freqs_mhz[-1] - freqs_mhz[0])
    return 2.0 * delta_mhz, span_mhz


def find_ripple_period_candidates(
    freqs_hz: np.ndarray,
    residual: np.ndarray,
    period_min_mhz: float,
    period_max_mhz: float,
    peak_snr_threshold: float = 10.0,
) -> dict:
    """FFT-seeded search for candidate ripple periods in a residual spectrum.

    Strategy: linear-detrend -> Hann window -> FFT -> local-maxima
    peak-picking within ``[period_min_mhz, period_max_mhz]`` -> parabolic
    sub-bin frequency refinement -> SNR vs. the median in-range power ->
    near-duplicate period merge (keeping the highest-SNR one).

    Any non-finite samples in ``residual`` are linearly interpolated (in
    frequency) before the FFT, so the transform always sees a complete,
    uniformly-sampled series; interpolation is smooth and does not
    manufacture short-period power.

    Parameters
    ----------
    freqs_hz : np.ndarray
        Frequency grid in Hz.
    residual : np.ndarray
        Fractional ripple residual, same shape as ``freqs_hz``. May contain
        NaN.
    period_min_mhz, period_max_mhz : float
        Search range in MHz (typically from :func:`derive_fourier_period_bounds`).
    peak_snr_threshold : float
        Minimum power-SNR (relative to the median in-range power) for a
        local maximum to be reported as a candidate. Default 10.0 — see
        Notes below for why a naive "3-sigma-style" threshold is not
        appropriate here.

    Returns
    -------
    dict with keys:
        ``periods_mhz``: list of candidate periods in MHz, sorted by
            descending SNR.
        ``snrs``: list of corresponding SNR values.
        ``noise_floor_power``: median in-range FFT power, used as the SNR
            denominator (``nan`` if the search could not run at all).

    Notes
    -----
    **Why the default threshold is 10, not something like 3:** FFT power of
    Gaussian noise is exponentially distributed, so for a single bin,
    ``P(power > k * median) ~= exp(-k * ln(2))``. This function searches
    every bin in the requested period range at once — for a typical GMRT
    channel count (~128 channels, ~65 rfft bins), a threshold of 3 gives a
    per-bin false-positive rate of ~12.5%, and across ~65 independent bins
    the chance of *at least one* spurious "candidate" from pure noise alone
    is close to certain (confirmed directly in this module's test suite: a
    threshold of 3-5 reliably produces multiple false candidates from pure
    Gaussian noise; a Bonferroni-style back-of-envelope for a false-positive
    budget of ~5% across that many bins requires roughly ``ln(n_bins/0.05) /
    ln(2)``, i.e. threshold ~10-12 for 60-250 bins). 10.0 is a deliberately
    conservative default reflecting this; callers with an unusually small
    channel count (fewer bins searched) may reasonably lower it.

    Returns empty lists (does not raise) when fewer than 16 finite residual
    points are available, or when the search range contains no valid FFT
    bins — deliberately looser than :func:`fit_harmonic_ripple`'s contract,
    which raises on insufficient data.
    """
    freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64)

    order = np.argsort(freqs_hz)
    freqs_mhz = freqs_hz[order] / 1e6
    residual_sorted = residual[order]

    finite = np.isfinite(residual_sorted)
    if int(np.sum(finite)) < _MIN_PERIOD_SEARCH_POINTS:
        return {'periods_mhz': [], 'snrs': [], 'noise_floor_power': float('nan')}

    if not np.all(finite):
        residual_filled = np.interp(freqs_mhz, freqs_mhz[finite], residual_sorted[finite])
    else:
        residual_filled = residual_sorted

    n = residual_filled.size
    delta_mhz = float(np.median(np.diff(freqs_mhz)))

    trend = np.polyval(np.polyfit(freqs_mhz, residual_filled, 1), freqs_mhz)
    detrended = residual_filled - trend
    windowed = detrended * np.hanning(n)

    spectrum = np.fft.rfft(windowed)
    power = np.abs(spectrum) ** 2
    freq_cpmhz = np.fft.rfftfreq(n, d=delta_mhz)  # cycles per MHz
    bin_width_cpmhz = float(freq_cpmhz[1] - freq_cpmhz[0]) if freq_cpmhz.size > 1 else 0.0

    # Expand the nominal [1/period_max, 1/period_min] frequency range by half
    # a bin width on each side: period_max_mhz = span_mhz and the FFT's bin
    # spacing (1/(n*delta_mhz)) differ from each other at the sub-percent
    # level (span uses (n-1) samples, bin spacing uses n), which would
    # otherwise push a period sitting exactly at either boundary just
    # outside the strict range and silently drop it.
    freq_hi = (1.0 / period_min_mhz if period_min_mhz > 0 else np.inf) + 0.5 * bin_width_cpmhz
    freq_lo = max(0.0, (1.0 / period_max_mhz if period_max_mhz > 0 else 0.0) - 0.5 * bin_width_cpmhz)
    in_range = (freq_cpmhz >= freq_lo) & (freq_cpmhz <= freq_hi) & (freq_cpmhz > 0)

    if not np.any(in_range):
        return {'periods_mhz': [], 'snrs': [], 'noise_floor_power': float('nan')}

    noise_floor_power = float(np.median(power[in_range]))
    if noise_floor_power <= 0:
        noise_floor_power = float(np.mean(power[in_range])) or 1e-300

    idxs_in_range = np.where(in_range)[0]
    last_bin = power.size - 1

    periods_mhz = []
    snrs = []
    for i in idxs_in_range:
        if i == 0:
            continue  # DC is never a legitimate finite-period candidate.

        if i == last_bin:
            # Edge bin (e.g. the Nyquist bin, which is exactly where a
            # ripple at period_min_mhz lands): only one neighbour exists, so
            # use a one-sided local-max test and no parabolic refinement.
            if not (power[i] > power[i - 1]):
                continue
            f_refined = float(freq_cpmhz[i])
        else:
            if not (power[i] > power[i - 1] and power[i] > power[i + 1]):
                continue
            y0, y1, y2 = power[i - 1], power[i], power[i + 1]
            denom = y0 - 2.0 * y1 + y2
            delta_bins = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
            delta_bins = float(np.clip(delta_bins, -0.5, 0.5))
            f_refined = float(freq_cpmhz[i] + delta_bins * bin_width_cpmhz)

        snr = power[i] / noise_floor_power
        if snr < peak_snr_threshold:
            continue
        if f_refined <= 0:
            continue

        period = 1.0 / f_refined
        # Clip (rather than reject) periods that the boundary-expanded search
        # nudges just past the nominal range, so a true boundary signal is
        # reported at the boundary instead of being dropped a second time.
        period = float(np.clip(period, period_min_mhz, period_max_mhz))

        periods_mhz.append(period)
        snrs.append(float(snr))

    if periods_mhz:
        rank = np.argsort(snrs)[::-1]
        periods_ranked = [periods_mhz[i] for i in rank]
        snrs_ranked = [snrs[i] for i in rank]
        merged_periods: list = []
        merged_snrs: list = []
        for p, s in zip(periods_ranked, snrs_ranked):
            if any(abs(p - mp) / mp < 0.05 for mp in merged_periods):
                continue
            merged_periods.append(p)
            merged_snrs.append(s)
        periods_mhz, snrs = merged_periods, merged_snrs

    return {
        'periods_mhz': periods_mhz,
        'snrs': snrs,
        'noise_floor_power': noise_floor_power,
    }
