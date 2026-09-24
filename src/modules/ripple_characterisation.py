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


def _min_harmonic_fit_points(n_harmonics: int) -> int:
    return max(20, 2 * n_harmonics + 4)


def _harmonic_design_columns(freqs_mhz: np.ndarray, period_mhz: float, n_harmonics: int) -> np.ndarray:
    """Build the sin/cos design columns for one period's harmonic series.

    Returns an array of shape ``(n_points, 2*n_harmonics)``: columns
    ``[sin(2*pi*1*nu/P), cos(2*pi*1*nu/P), sin(2*pi*2*nu/P), cos(2*pi*2*nu/P), ...]``.
    """
    cols = []
    for k in range(1, n_harmonics + 1):
        phase = 2.0 * np.pi * k * freqs_mhz / period_mhz
        cols.append(np.sin(phase))
        cols.append(np.cos(phase))
    return np.column_stack(cols)


def _joint_design_matrix(freqs_mhz: np.ndarray, periods_mhz: list, n_harmonics: int) -> np.ndarray:
    """Concatenate each candidate period's harmonic columns into one matrix."""
    blocks = [_harmonic_design_columns(freqs_mhz, p, n_harmonics) for p in periods_mhz]
    return np.hstack(blocks) if blocks else np.zeros((freqs_mhz.size, 0))


def _fit_for_periods(freqs_mhz: np.ndarray, y: np.ndarray, periods_mhz: list, n_harmonics: int):
    """Solve the joint OLS fit for a fixed set of periods.

    Returns ``(coeffs, rss)`` where ``coeffs`` has shape
    ``(n_candidates, n_harmonics, 2)`` (last axis is ``[A_k, B_k]``), and
    ``rss`` is the residual sum of squares of the fit.
    """
    n_candidates = len(periods_mhz)
    if n_candidates == 0:
        return np.zeros((0, n_harmonics, 2)), float(np.sum(y**2))

    design = _joint_design_matrix(freqs_mhz, periods_mhz, n_harmonics)
    coeffs_flat, residuals, _, _ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coeffs_flat
    rss = float(np.sum((y - fitted) ** 2))
    coeffs = coeffs_flat.reshape(n_candidates, n_harmonics, 2)
    return coeffs, rss


def fit_harmonic_ripple(
    freqs_hz: np.ndarray,
    residual: np.ndarray,
    period_candidates: list,
    n_harmonics: int = 1,
    period_refine_frac: float = 0.15,
) -> dict:
    """Fit a joint multi-component harmonic ripple model.

    Implements the harmonic Fourier ripple model from
    ``ripple_convergence_todos.md``:

        r(nu) = sum_k [A_k*sin(2*pi*k*nu/P) + B_k*cos(2*pi*k*nu/P)]

    Each entry in ``period_candidates`` (typically seeded by
    :func:`find_ripple_period_candidates`) gets its own harmonic series
    (``k=1..n_harmonics`` of its own fundamental frequency); all candidates'
    harmonic bases are combined into one joint ordinary-least-squares design
    matrix and solved together, so that one candidate's fit cannot silently
    absorb power that actually belongs to another. Each candidate's own
    period is then refined by local coordinate descent: holding every other
    period fixed, a small grid of trial periods within
    ``period_i * (1 +/- period_refine_frac)`` is tried, re-solving the joint
    fit each time, keeping whichever trial period minimises the overall
    residual sum of squares. A few such passes over all candidates are run.

    Parameters
    ----------
    freqs_hz : np.ndarray
        Frequency grid in Hz.
    residual : np.ndarray
        Fractional ripple residual, same shape as ``freqs_hz``. May contain
        NaN (excluded from the fit).
    period_candidates : list of float
        Seed periods in MHz, one per component (e.g. from
        :func:`find_ripple_period_candidates`).
    n_harmonics : int
        Number of harmonics (``k=1..n_harmonics``) fit per candidate.
    period_refine_frac : float
        Fractional half-width of each candidate's local period-refinement
        search window (default 0.15, i.e. +/-15%).

    Returns
    -------
    dict with keys:
        ``components``: list of dicts, one per input candidate period, each
            with ``period_mhz`` (refined), ``amplitude`` (of the
            fundamental, k=1), ``phase_rad`` (of the fundamental),
            ``snr`` (amplitude / overall residual MAD-sigma after fit),
            ``classification`` (``'primary'``, ``'harmonic'``, or
            ``'independent'``), ``ratio_to_primary``.
        ``rms_before``: RMS of the input residual over finite points.
        ``rms_after``: RMS of the residual after subtracting the full joint
            fitted model.
        ``model``: the full joint fitted model, evaluated at every input
            ``freqs_hz`` (NaN where the model cannot be evaluated, which
            does not occur here since the model is defined everywhere).

    Raises
    ------
    ValueError
        If fewer than ``max(20, 2*n_harmonics + 4)`` finite residual points
        are available. Unlike :func:`find_ripple_period_candidates` (which
        returns empty results on too little data), this function raises —
        a caller with too few points to reliably fit should not silently
        receive a fit result.
    """
    freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64)
    freqs_mhz = freqs_hz / 1e6

    finite = np.isfinite(freqs_hz) & np.isfinite(residual)
    n_finite = int(np.sum(finite))
    min_points = _min_harmonic_fit_points(n_harmonics)
    if n_finite < min_points:
        raise ValueError(
            f'fit_harmonic_ripple: need at least {min_points} finite points '
            f'for n_harmonics={n_harmonics}, got {n_finite}.'
        )

    x_fit = freqs_mhz[finite]
    y_fit = residual[finite]
    rms_before = float(np.sqrt(np.mean(y_fit**2)))

    periods = [float(p) for p in period_candidates]

    if periods:
        for _ in range(2):  # a couple of coordinate-descent passes
            for i, p_i in enumerate(periods):
                lo = max(p_i * (1.0 - period_refine_frac), 1e-6)
                hi = p_i * (1.0 + period_refine_frac)
                trial_periods = np.linspace(lo, hi, 21)
                best_rss = np.inf
                best_p = p_i
                for trial in trial_periods:
                    candidate_periods = list(periods)
                    candidate_periods[i] = float(trial)
                    _, rss = _fit_for_periods(x_fit, y_fit, candidate_periods, n_harmonics)
                    if rss < best_rss:
                        best_rss = rss
                        best_p = float(trial)
                periods[i] = best_p

    coeffs, _ = _fit_for_periods(x_fit, y_fit, periods, n_harmonics)

    model_full = np.zeros_like(freqs_mhz)
    for i, p_i in enumerate(periods):
        cols = _harmonic_design_columns(freqs_mhz, p_i, n_harmonics)
        model_full += cols @ coeffs[i].reshape(-1)

    resid_after = residual - model_full
    rms_after = float(np.sqrt(np.mean(resid_after[finite] ** 2))) if periods else rms_before
    sigma_hat = mad_sigma(resid_after[finite]) if periods else mad_sigma(y_fit)

    components = []
    if periods:
        amplitudes = [float(np.hypot(coeffs[i, 0, 0], coeffs[i, 0, 1])) for i in range(len(periods))]
        primary_idx = int(np.argmax(amplitudes))
        primary_period = periods[primary_idx]

        for i, p_i in enumerate(periods):
            a1, b1 = coeffs[i, 0, 0], coeffs[i, 0, 1]
            amplitude = float(np.hypot(a1, b1))
            phase_rad = float(np.arctan2(b1, a1))
            snr = amplitude / sigma_hat if (sigma_hat and np.isfinite(sigma_hat) and sigma_hat > 0) else float('inf')

            if i == primary_idx:
                classification = 'primary'
                ratio_to_primary = 1.0
            else:
                ratio_to_primary = primary_period / p_i
                nearest_order = round(ratio_to_primary)
                is_harmonic = (
                    nearest_order >= 2
                    and abs(ratio_to_primary - nearest_order) / nearest_order < 0.1
                )
                classification = 'harmonic' if is_harmonic else 'independent'

            components.append({
                'period_mhz': float(p_i),
                'amplitude': amplitude,
                'phase_rad': phase_rad,
                'snr': float(snr),
                'classification': classification,
                'ratio_to_primary': float(ratio_to_primary),
            })

    return {
        'components': components,
        'rms_before': rms_before,
        'rms_after': rms_after,
        'model': model_full,
    }


_SPEED_OF_LIGHT_M_PER_S = 299792458.0


def period_to_cable_length_m(period_mhz: float, velocity_factor: float = 1.0) -> float:
    """Convert a fitted ripple period to an implied round-trip cable length.

    Standing-wave ripple from a cable reflection has a spectral period
    related to the reflection's round-trip travel time:

        L = (velocity_factor * c) / (2 * f_ripple)

    where ``f_ripple = 1 / period`` and the factor of 2 accounts for the
    round trip (signal travels to the reflection point and back).

    Parameters
    ----------
    period_mhz : float
        Ripple period in MHz. Non-positive or non-finite values return
        ``nan`` rather than raising, since this is typically called on a
        fitted value that may be degenerate.
    velocity_factor : float
        Propagation velocity as a fraction of ``c`` (e.g. ~0.66 for typical
        foam coax). Non-positive or non-finite values return ``nan``.

    Returns
    -------
    float
        Implied cable length in metres, or ``nan`` for invalid inputs.
    """
    if not np.isfinite(period_mhz) or period_mhz <= 0:
        return float('nan')
    if not np.isfinite(velocity_factor) or velocity_factor <= 0:
        return float('nan')

    # A ripple period P (MHz) in the frequency domain corresponds to a
    # round-trip travel-time delay tau = 1/(P * 1e6) seconds between the
    # direct and reflected paths (a two-path interference completes one
    # full 2*pi phase cycle every Delta_nu = 1/tau). L = v*tau/2 (round trip).
    round_trip_delay_s = 1.0 / (period_mhz * 1e6)
    length_m = (velocity_factor * _SPEED_OF_LIGHT_M_PER_S * round_trip_delay_s) / 2.0
    return float(length_m)
