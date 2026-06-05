#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullFormatter, FormatStrFormatter, FuncFormatter, LogLocator
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_MODULES = REPO_ROOT / 'src' / 'modules'
if str(SRC_MODULES) not in sys.path:
    sys.path.insert(0, str(SRC_MODULES))

import ugmrt_query as q


def _canonical_source_name(source_name: str) -> str:
    return ''.join(ch for ch in str(source_name).upper() if ch.isalnum())


def _source_slug(source_name: str) -> str:
    slug = ''.join(ch.lower() if ch.isalnum() else '-' for ch in str(source_name))
    while '--' in slug:
        slug = slug.replace('--', '-')
    slug = slug.strip('-')
    return slug or 'source'


def lookup_known_flux_model(source_name: str):
    registry = getattr(q, '_FLUX_MODEL_REGISTRY', {})
    canonical_registry = {}
    for name, fn in registry.items():
        if not callable(fn):
            continue
        key = _canonical_source_name(name)
        if key not in canonical_registry:
            canonical_registry[key] = (name, fn)
    key = _canonical_source_name(source_name)
    if key not in canonical_registry:
        available = sorted(registry.keys())
        raise ValueError(
            f'No registered known flux model for source={source_name!r}. '
            f'Available sources: {available}'
        )
    return canonical_registry[key]


def period_to_cable_length_m(period_mhz: float, velocity_factor: float = 1.0) -> float:
    period_hz = float(period_mhz) * 1e6
    if not np.isfinite(period_hz) or period_hz <= 0:
        return float('nan')
    vf = float(velocity_factor)
    if not np.isfinite(vf) or vf <= 0:
        return float('nan')
    c = 299792458.0
    return (vf * c) / (2.0 * period_hz)


def vector_avg_real(vis: dict, pol_label: str) -> np.ndarray:
    labels = list(vis['stokes_labels'])
    if pol_label not in labels:
        raise ValueError(f'Polarization {pol_label!r} not in stokes_labels={labels}')

    p = labels.index(pol_label)
    z = np.asarray(vis['vis_complex'], dtype=np.complex128)[:, :, p]
    flagged = np.asarray(vis['flagged'], dtype=bool)[:, :, p]
    w = np.asarray(vis['weight'], dtype=np.float64)[:, :, p]

    good = (~flagged) & np.isfinite(z.real) & np.isfinite(z.imag) & np.isfinite(w) & (w > 0)
    num = np.nansum(np.where(good, w * z, 0.0), axis=0)
    den = np.nansum(np.where(good, w, 0.0), axis=0)

    out = np.full(z.shape[1], np.nan, dtype=np.float64)
    ok = den > 0
    out[ok] = np.real(num[ok] / den[ok])
    return out


def fit_models(freqs_hz: np.ndarray, spectrum_jy: np.ndarray) -> dict:
    mask = np.isfinite(spectrum_jy) & (spectrum_jy > 0)
    x = np.log(freqs_hz[mask])
    y = np.log(spectrum_jy[mask])

    if x.size < 10:
        raise ValueError('Too few valid points for log-log fitting.')

    Xq = np.column_stack([np.ones_like(x), x, x**2])
    a0, a1, a2 = np.linalg.lstsq(Xq, y, rcond=None)[0]
    rss_q = float(np.sum((y - (a0 + a1*x + a2*x**2))**2))

    Xl = np.column_stack([np.ones_like(x), x])
    b0, b1 = np.linalg.lstsq(Xl, y, rcond=None)[0]
    rss_l = float(np.sum((y - (b0 + b1*x))**2))

    x_all = np.log(freqs_hz)
    S_q = np.exp(a0 + a1*x_all + a2*x_all**2)
    S_l = np.exp(b0 + b1*x_all)

    nu0 = float(np.sqrt(freqs_hz.min() * freqs_hz.max()))
    ln_nu0 = np.log(nu0)
    alpha_q = float(a1 + 2.0 * a2 * ln_nu0)
    beta_q = float(a2)
    alpha_l = float(b1)

    return {
        'mask_n': int(mask.sum()),
        'nu0_hz': nu0,
        'quad': {
            'a0': float(a0),
            'a1': float(a1),
            'a2': float(a2),
            'alpha_nu0': alpha_q,
            'beta': beta_q,
            'rss_log': rss_q,
        },
        'linear': {
            'b0': float(b0),
            'b1': float(b1),
            'alpha_nu0': alpha_l,
            'beta': 0.0,
            'rss_log': rss_l,
        },
        'S_quad': S_q,
        'S_linear': S_l,
    }


def local_spectral_index_at_nu0(model_fn, nu0_hz: float, frac_step: float = 1e-3) -> float:
    nu0 = float(nu0_hz)
    eps = float(frac_step)
    if (not np.isfinite(nu0)) or nu0 <= 0:
        return float('nan')
    if (not np.isfinite(eps)) or eps <= 0 or eps >= 1:
        return float('nan')

    nu_lo = nu0 * (1.0 - eps)
    nu_hi = nu0 * (1.0 + eps)
    if nu_lo <= 0:
        return float('nan')

    s_pair = np.asarray(model_fn(np.array([nu_lo, nu_hi], dtype=np.float64)), dtype=np.float64)
    if s_pair.size != 2:
        return float('nan')
    s_lo = float(s_pair[0])
    s_hi = float(s_pair[1])
    if (not np.isfinite(s_lo)) or (not np.isfinite(s_hi)) or s_lo <= 0 or s_hi <= 0:
        return float('nan')

    return float((np.log(s_hi) - np.log(s_lo)) / (np.log(nu_hi) - np.log(nu_lo)))


def _build_harmonic_design(f_mhz: np.ndarray, period_mhz: float, n_harmonics: int) -> np.ndarray:
    cols = [np.ones_like(f_mhz)]
    for k in range(1, int(n_harmonics) + 1):
        arg = 2.0 * np.pi * k * f_mhz / period_mhz
        cols.append(np.sin(arg))
        cols.append(np.cos(arg))
    return np.column_stack(cols)


def _build_multi_harmonic_design(f_mhz: np.ndarray, periods_mhz: list[float], n_harmonics: int | list[int]) -> np.ndarray:
    if isinstance(n_harmonics, int):
        n_harm_list = [int(n_harmonics)] * len(periods_mhz)
    else:
        n_harm_list = [int(v) for v in n_harmonics]
        if len(n_harm_list) != len(periods_mhz):
            raise ValueError('n_harmonics list length must match number of periods')

    cols = [np.ones_like(f_mhz)]
    for period_mhz, n_h in zip(periods_mhz, n_harm_list):
        for k in range(1, int(n_h) + 1):
            arg = 2.0 * np.pi * k * f_mhz / float(period_mhz)
            cols.append(np.sin(arg))
            cols.append(np.cos(arg))
    return np.column_stack(cols)


def _fit_for_periods(f_mhz: np.ndarray, y: np.ndarray, periods_mhz: list[float], n_harmonics: int | list[int]) -> dict:
    if isinstance(n_harmonics, int):
        n_harm_list = [int(n_harmonics)] * len(periods_mhz)
    else:
        n_harm_list = [int(v) for v in n_harmonics]
    A = _build_multi_harmonic_design(f_mhz, [float(p) for p in periods_mhz], n_harm_list)
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    yhat = A @ beta
    rss = float(np.sum((y - yhat) ** 2))
    return {
        'periods_mhz': [float(p) for p in periods_mhz],
        'n_harmonics_per_component': n_harm_list,
        'coeff': beta,
        'rss': rss,
    }


def _fit_for_period(f_mhz: np.ndarray, y: np.ndarray, period_mhz: float, n_harmonics: int) -> dict:
    out = _fit_for_periods(f_mhz, y, [float(period_mhz)], int(n_harmonics))
    out['period_mhz'] = float(period_mhz)
    return out


def classify_period_family(periods_mhz: list[float], rel_tol: float = 0.06, max_order: int = 16) -> tuple[float, list[dict[str, float | int | bool | str]]]:
    if not periods_mhz:
        return float('nan'), []

    primary_period = float(np.nanmin(np.asarray(periods_mhz, dtype=np.float64)))
    out: list[dict[str, float | int | bool | str]] = []
    for p in periods_mhz:
        period = float(p)
        ratio = float(period / primary_period) if primary_period > 0 else float('nan')
        order = int(np.round(ratio)) if np.isfinite(ratio) else 1
        order = max(1, min(int(max_order), order))
        rel_err = abs(ratio - order) / max(order, 1)
        is_primary = abs(period - primary_period) / max(primary_period, 1e-9) < 1e-6
        is_harmonic = bool((not is_primary) and np.isfinite(ratio) and order >= 2 and rel_err <= float(rel_tol))
        classification = 'primary' if is_primary else ('harmonic' if is_harmonic else 'independent')
        out.append({
            'ratio_to_primary': float(ratio),
            'nearest_integer_order': int(order),
            'ratio_rel_error': float(rel_err),
            'is_likely_harmonic': bool(is_harmonic),
            'classification': classification,
        })
    return primary_period, out


def harmonics_from_cycles(n_cycles: float, n_harmonics_max: int, factor: float = 1.0) -> int:
    if not np.isfinite(n_cycles):
        return 1
    raw = int(np.floor(max(0.0, float(factor)) * max(0.0, float(n_cycles))))
    return max(1, min(int(n_harmonics_max), raw))


def build_harmonic_family_groups(components: list[dict]) -> dict:
    primary = [c for c in components if c.get('classification') == 'primary']
    harmonics = [c for c in components if c.get('classification') == 'harmonic']
    independent = [c for c in components if c.get('classification') == 'independent']
    return {
        'primary': primary,
        'harmonics': harmonics,
        'independent': independent,
        'n_primary': len(primary),
        'n_harmonics': len(harmonics),
        'n_independent': len(independent),
    }


def derive_fourier_period_bounds(freqs_hz: np.ndarray) -> tuple[float, float]:
    nu_mhz = np.asarray(freqs_hz, dtype=np.float64) / 1e6
    nu_mhz = nu_mhz[np.isfinite(nu_mhz)]
    if nu_mhz.size < 4:
        raise ValueError('Too few finite frequency points to derive Fourier bounds.')
    nu_mhz = np.sort(nu_mhz)
    dnu = float(np.median(np.diff(nu_mhz)))
    span = float(nu_mhz[-1] - nu_mhz[0])
    if (not np.isfinite(dnu)) or dnu <= 0 or (not np.isfinite(span)) or span <= 0:
        raise ValueError('Invalid frequency sampling for Fourier bounds.')
    period_min = 2.0 * dnu
    period_max = span
    if period_min >= period_max:
        period_min = max(1e-6, 0.5 * period_max)
    return float(period_min), float(period_max)


def estimate_period_candidates_from_spectrum(
    freqs_hz: np.ndarray,
    residual: np.ndarray,
    period_min_mhz: float,
    period_max_mhz: float,
    peak_snr_threshold: float = 3.0,
) -> tuple[list[float], list[float]]:
    """
    Return (periods_mhz, snrs) for all significant spectral peaks in the
    power spectrum of *residual* that fall within [period_min_mhz, period_max_mhz].

    Significance is defined as: peak power > peak_snr_threshold * noise_floor,
    where noise_floor is estimated as the median power of all in-range bins
    (robust to individual peaks).

    Peaks are local maxima (higher than their immediate neighbours).  Close
    duplicates (within 4 % in period) are merged, keeping the strongest.
    """
    mask = np.isfinite(freqs_hz) & np.isfinite(residual)
    if int(mask.sum()) < 16:
        return [], []

    f_mhz = np.asarray(freqs_hz[mask], dtype=np.float64) / 1e6
    y = np.asarray(residual[mask], dtype=np.float64)

    order = np.argsort(f_mhz)
    f_mhz = f_mhz[order]
    y = y[order]

    span = float(f_mhz[-1] - f_mhz[0])
    if not np.isfinite(span) or span <= 0:
        return [], []

    n_uniform = max(512, int(8 * f_mhz.size))
    f_uniform = np.linspace(float(f_mhz[0]), float(f_mhz[-1]), n_uniform, dtype=np.float64)
    y_uniform = np.interp(f_uniform, f_mhz, y)

    # Detrend: remove low-order polynomial to suppress spectral leakage from
    # smooth source-model residuals bleeding into low-frequency FFT bins.
    xc = f_uniform - float(np.mean(f_uniform))
    poly_deg = 2 if xc.size >= 12 else 1
    trend = np.polyval(np.polyfit(xc, y_uniform, deg=poly_deg), xc)
    y_centered = y_uniform - trend
    y_centered -= float(np.mean(y_centered))

    window = np.hanning(n_uniform)
    yw = y_centered * window

    dnu = float(np.mean(np.diff(f_uniform)))
    if not np.isfinite(dnu) or dnu <= 0:
        return [], []

    spec = np.fft.rfft(yw)
    freqs_cpmhz = np.fft.rfftfreq(n_uniform, d=dnu)
    power = np.abs(spec) ** 2

    valid = freqs_cpmhz > 0
    if not np.any(valid):
        return [], []

    period_arr = np.full_like(freqs_cpmhz, np.nan, dtype=np.float64)
    period_arr[valid] = 1.0 / freqs_cpmhz[valid]

    in_range = valid & (period_arr >= float(period_min_mhz)) & (period_arr <= float(period_max_mhz))
    if not np.any(in_range):
        return [], []

    # Noise floor: median power of all in-range bins (robust to peaks).
    noise_floor = float(np.median(power[in_range]))
    if not np.isfinite(noise_floor) or noise_floor <= 0:
        return [], []

    threshold = float(peak_snr_threshold) * noise_floor

    # Find local maxima (strict: higher than both neighbours) above threshold.
    # Refine peak location with quadratic (parabolic) interpolation for sub-bin
    # frequency estimation.
    idx_range = np.where(in_range)[0]
    peaks: list[tuple[float, float]] = []   # (period_mhz, snr)
    for j, i in enumerate(idx_range):
        p_val = float(power[i])
        if p_val < threshold:
            continue
        # Boundary bins count as local maxima if they're above threshold.
        left  = float(power[i - 1]) if i > 0 else -np.inf
        right = float(power[i + 1]) if i + 1 < len(power) else -np.inf
        if p_val >= left and p_val >= right:
            refined_freq = float(freqs_cpmhz[i])
            refined_power = p_val

            if (i > 0) and (i + 1 < len(power)):
                p_l = float(power[i - 1])
                p_0 = float(power[i])
                p_r = float(power[i + 1])
                denom = (p_l - 2.0 * p_0 + p_r)
                if np.isfinite(denom) and abs(denom) > 0:
                    delta = 0.5 * (p_l - p_r) / denom
                    if np.isfinite(delta):
                        delta = float(np.clip(delta, -0.5, 0.5))
                        refined_freq = float(freqs_cpmhz[i] + delta * (freqs_cpmhz[1] - freqs_cpmhz[0]))
                        refined_power = float(p_0 - 0.25 * (p_l - p_r) * delta)

            if np.isfinite(refined_freq) and refined_freq > 0:
                refined_period = float(1.0 / refined_freq)
            else:
                refined_period = float(period_arr[i])

            if (refined_period >= float(period_min_mhz)) and (refined_period <= float(period_max_mhz)):
                snr_val = float(max(refined_power, p_val) / noise_floor)
                peaks.append((refined_period, snr_val))

    if not peaks:
        return [], []

    # Sort by SNR descending, then merge near-duplicates (within 4 %).
    peaks.sort(key=lambda t: -t[1])
    selected_periods: list[float] = []
    selected_snrs: list[float] = []
    for period_cand, snr_cand in peaks:
        if not np.isfinite(period_cand):
            continue
        if all(abs(period_cand - ep) / max(ep, 1e-9) > 0.04 for ep in selected_periods):
            selected_periods.append(period_cand)
            selected_snrs.append(snr_cand)

    return selected_periods, selected_snrs


def fit_harmonic_residual(
    freqs_hz: np.ndarray,
    residual: np.ndarray,
    n_harmonics: int = 3,
    period_min_mhz: float = 5.0,
    period_max_mhz: float = 20.0,
    n_period_grid: int = 1200,
    period_search_mode: str = 'auto',
    peak_snr_threshold: float = 3.0,
    max_peaks: int = 0,
    period_refine_frac: float = 0.15,
    min_cycles_in_band: float = 0.0,
    n_components: int = 1,
    harmonic_ratio_tol: float = 0.06,
    harmonic_per_cycle_factor: float = 1.0,
    prefer_harmonic_components: bool = True,
) -> dict:
    mask = np.isfinite(residual)
    if int(mask.sum()) < max(20, 2 * n_harmonics + 4):
        raise ValueError('Too few finite residual points for harmonic fit.')

    f_mhz = np.asarray(freqs_hz[mask], dtype=np.float64) / 1e6
    y = np.asarray(residual[mask], dtype=np.float64)
    band_span = float(np.nanmax(f_mhz) - np.nanmin(f_mhz)) if f_mhz.size else float('nan')

    period_grid = np.linspace(float(period_min_mhz), float(period_max_mhz), int(n_period_grid), dtype=np.float64)
    candidates: list[float] = []
    candidate_snrs: list[float] = []
    search_notes: dict[str, object] = {
        'mode': period_search_mode,
        'peak_snr_threshold': float(peak_snr_threshold),
        'max_peaks': int(max_peaks),
    }
    search_notes['period_min_used_mhz'] = float(period_min_mhz)
    search_notes['period_max_used_mhz'] = float(period_max_mhz)
    if period_search_mode == 'auto':
        candidates, candidate_snrs = estimate_period_candidates_from_spectrum(
            freqs_hz=freqs_hz,
            residual=residual,
            period_min_mhz=period_min_mhz,
            period_max_mhz=period_max_mhz,
            peak_snr_threshold=peak_snr_threshold,
        )
        search_notes['spectral_candidates_mhz'] = [float(v) for v in candidates]
        search_notes['spectral_candidate_snrs'] = [float(v) for v in candidate_snrs]
        search_notes['n_significant_peaks'] = len(candidates)

    if period_search_mode == 'grid' or len(candidates) == 0:
        search_periods = period_grid
        search_notes['fallback'] = 'full_grid'
        use_multi_component = False
        selected_periods = []
        selected_snrs = []
        selected_nharm = []
    else:
        # Default recipe: use all significant FFT peaks by S/N, optionally capped
        # by max_peaks (if >0). No harmonic-family pre-filtering by default.
        use_multi_component = True
        selected_periods = []
        selected_snrs = []
        selected_nharm = []
        candidate_rows = []
        for cand_p, cand_snr in zip(candidates, candidate_snrs):
            n_cyc = float(band_span / cand_p) if np.isfinite(band_span) and cand_p > 0 else float('nan')
            candidate_rows.append((float(cand_p), float(cand_snr), float(n_cyc)))

        ordered_rows = sorted(candidate_rows, key=lambda r: -r[1])
        search_notes['component_selection_policy'] = 'snr_only'

        max_components = int(max_peaks) if int(max_peaks) > 0 else len(ordered_rows)
        max_components = max(1, min(max_components, len(ordered_rows))) if ordered_rows else 0
        for cand_p, cand_snr, n_cyc in ordered_rows[:max_components]:
            nh = 1
            selected_periods.append(float(cand_p))
            selected_snrs.append(float(cand_snr))
            selected_nharm.append(int(nh))

        if len(selected_periods) == 0:
            # No peaks pass min_cycles_in_band — fall back to full grid.
            search_periods = period_grid
            search_notes['fallback'] = 'full_grid_no_significant_peaks'
            use_multi_component = False
        else:
            search_notes['selected_component_seeds_mhz'] = [float(v) for v in selected_periods]
            search_notes['selected_component_seed_snrs'] = [float(v) for v in selected_snrs]
            search_notes['selected_component_nharm'] = [int(v) for v in selected_nharm]
            search_notes['n_components_requested'] = int(max_components)
            search_notes['n_components_used'] = int(len(selected_periods))
            if use_multi_component:
                search_periods = np.array([], dtype=np.float64)
                search_notes['fallback'] = 'multi_peak_joint_fit'
            else:
                dominant_period = float(selected_periods[0])
                search_notes['dominant_peak_snr'] = float(selected_snrs[0])
                dominant_nharm = int(selected_nharm[0]) if selected_nharm else int(n_harmonics)
                search_notes['dominant_nharm'] = int(dominant_nharm)
                # Dense grid within ± period_refine_frac around dominant peak.
                half = max(1e-6, float(period_refine_frac) * dominant_period)
                lo_c = max(float(period_min_mhz), dominant_period - half)
                hi_c = min(float(period_max_mhz), dominant_period + half)
                search_periods = np.linspace(lo_c, hi_c, int(n_period_grid), dtype=np.float64)
                search_notes['fallback'] = 'dominant_peak_window'
                search_notes['dominant_period_seed_mhz'] = float(dominant_period)

    best = None
    fit_records = []
    if use_multi_component and len(selected_periods) > 0:
        # Joint fit for multiple independent ripple components, then coordinate-descent
        # refinement of each component period.
        current_periods = [float(v) for v in selected_periods]
        current_nharm = [int(v) for v in selected_nharm] if selected_nharm else [int(n_harmonics)] * len(current_periods)
        best = _fit_for_periods(f_mhz, y, current_periods, current_nharm)
        refine_pts = max(40, min(200, int(n_period_grid // max(1, len(current_periods) * 3))))
        for _ in range(2):
            improved_any = False
            for comp_idx in range(len(current_periods)):
                p0 = float(current_periods[comp_idx])
                half = max(1e-6, float(period_refine_frac) * p0)
                lo_c = max(float(period_min_mhz), p0 - half)
                hi_c = min(float(period_max_mhz), p0 + half)
                grid = np.linspace(lo_c, hi_c, refine_pts, dtype=np.float64)
                local_best = None
                local_period = p0
                for cand in grid:
                    n_cycles = float(band_span / cand) if np.isfinite(band_span) and cand > 0 else float('nan')
                    if np.isfinite(min_cycles_in_band) and min_cycles_in_band > 0 and np.isfinite(n_cycles):
                        if n_cycles < float(min_cycles_in_band):
                            continue
                    trial = list(current_periods)
                    trial[comp_idx] = float(cand)
                    fit = _fit_for_periods(f_mhz, y, trial, current_nharm)
                    if local_best is None or fit['rss'] < local_best['rss']:
                        local_best = fit
                        local_period = float(cand)
                if local_best is not None and local_best['rss'] < best['rss']:
                    best = local_best
                    current_periods[comp_idx] = local_period
                    improved_any = True
            if not improved_any:
                break
    else:
        nh_single = int(selected_nharm[0]) if selected_nharm else int(n_harmonics)
        for period in search_periods:
            n_cycles = float(band_span / period) if np.isfinite(band_span) and period > 0 else float('nan')
            if np.isfinite(min_cycles_in_band) and min_cycles_in_band > 0 and np.isfinite(n_cycles):
                if n_cycles < float(min_cycles_in_band):
                    continue
            fit = _fit_for_period(f_mhz, y, float(period), int(nh_single))
            fit_records.append({'period_mhz': float(period), 'rss': float(fit['rss']), 'n_cycles': float(n_cycles)})
            if best is None or fit['rss'] < best['rss']:
                best = fit

    if best is None:
        raise RuntimeError('Harmonic residual fit failed. No candidate periods satisfied min_cycles_in_band constraint.')

    if use_multi_component and len(best.get('periods_mhz', [])) > 0:
        periods_used = [float(v) for v in best['periods_mhz']]
        lo = float(np.nanmin(periods_used))
        hi = float(np.nanmax(periods_used))
        refine_n = 0
    else:
        center = float(best['period_mhz'])
        half = max(1e-6, float(period_refine_frac) * center)
        lo = max(float(period_min_mhz), center - half)
        hi = min(float(period_max_mhz), center + half)
        refine_n = max(120, min(800, int(n_period_grid // 2)))
        refine_grid = np.linspace(lo, hi, refine_n, dtype=np.float64)
        for period in refine_grid:
            n_cycles = float(band_span / period) if np.isfinite(band_span) and period > 0 else float('nan')
            if np.isfinite(min_cycles_in_band) and min_cycles_in_band > 0 and np.isfinite(n_cycles):
                if n_cycles < float(min_cycles_in_band):
                    continue
            fit = _fit_for_period(f_mhz, y, float(period), int(nh_single))
            fit_records.append({'period_mhz': float(period), 'rss': float(fit['rss']), 'n_cycles': float(n_cycles)})
            if fit['rss'] < best['rss']:
                best = fit

    selected_via = 'min_rss'

    if 'periods_mhz' in best:
        periods_used = [float(v) for v in best['periods_mhz']]
    else:
        periods_used = [float(best['period_mhz'])]

    f_all_mhz = np.asarray(freqs_hz, dtype=np.float64) / 1e6
    nharm_per_comp = [int(v) for v in best.get('n_harmonics_per_component', [int(n_harmonics)] * len(periods_used))]
    A_all = _build_multi_harmonic_design(f_all_mhz, periods_used, nharm_per_comp)
    model_all = A_all @ best['coeff']

    valid_all = np.isfinite(residual)
    resid_after = np.full_like(residual, np.nan, dtype=np.float64)
    resid_after[valid_all] = residual[valid_all] - model_all[valid_all]

    rms_before = float(np.sqrt(np.nanmean(np.square(residual[valid_all])))) if np.any(valid_all) else float('nan')
    rms_after = float(np.sqrt(np.nanmean(np.square(resid_after[valid_all])))) if np.any(valid_all) else float('nan')

    dc = float(best['coeff'][0])
    comp_snr_map = {float(p): float(s) for p, s in zip(candidates, candidate_snrs)}
    components = []
    coeff_offset = 1
    harmonic_bins: dict[int, list[dict[str, float | int]]] = {}
    for idx, p_comp in enumerate(periods_used):
        n_hc = int(nharm_per_comp[idx]) if idx < len(nharm_per_comp) else int(n_harmonics)
        base = int(coeff_offset)
        sin1 = float(best['coeff'][base]) if base < best['coeff'].size else float('nan')
        cos1 = float(best['coeff'][base + 1]) if (base + 1) < best['coeff'].size else float('nan')
        amp1_comp = float(np.hypot(sin1, cos1)) if np.isfinite(sin1) and np.isfinite(cos1) else float('nan')
        phase1_rad = float(np.arctan2(cos1, sin1)) if np.isfinite(sin1) and np.isfinite(cos1) else float('nan')
        phase1_deg = float(np.degrees(phase1_rad)) if np.isfinite(phase1_rad) else float('nan')
        n_cycles_comp = float(band_span / p_comp) if np.isfinite(band_span) and p_comp > 0 else float('nan')
        freq_comp = float(1.0 / p_comp) if np.isfinite(p_comp) and p_comp > 0 else float('nan')
        snr_comp = float('nan')
        for pk, sv in comp_snr_map.items():
            if abs(pk - p_comp) / max(pk, 1e-9) <= max(0.04, float(period_refine_frac) + 0.01):
                snr_comp = float(sv)
                break
        components.append({
            'period_mhz': float(p_comp),
            'frequency_cpmhz': freq_comp,
            'n_harmonics_used': int(n_hc),
            'amp1': amp1_comp,
            'phase1_deg': phase1_deg,
            'n_cycles_in_band': n_cycles_comp,
            'spectral_peak_snr': snr_comp,
        })

        for k in range(1, int(n_hc) + 1):
            s_idx = base + 2 * (k - 1)
            c_idx = s_idx + 1
            sin_k = float(best['coeff'][s_idx]) if s_idx < best['coeff'].size else float('nan')
            cos_k = float(best['coeff'][c_idx]) if c_idx < best['coeff'].size else float('nan')
            if not (np.isfinite(sin_k) and np.isfinite(cos_k)):
                continue
            amp_k = float(np.hypot(sin_k, cos_k))
            pow_k = float(amp_k ** 2)
            phase_k = float(np.degrees(np.arctan2(cos_k, sin_k)))
            harmonic_bins.setdefault(int(k), []).append({
                'power': pow_k,
                'phase_deg': phase_k,
                'period_mhz': float(p_comp),
                'component_index': int(idx + 1),
            })
        coeff_offset += 2 * int(n_hc)

    primary_period, family_info = classify_period_family(
        [float(c['period_mhz']) for c in components],
        rel_tol=float(harmonic_ratio_tol),
        max_order=max(8, int(n_harmonics) * 3),
    )
    for comp, fam in zip(components, family_info):
        comp.update(fam)

    harmonic_family = build_harmonic_family_groups(components)

    if components:
        primary_idx = int(np.argmin([abs(float(c['period_mhz']) - primary_period) for c in components]))
        primary_amp1 = float(components[primary_idx]['amp1'])
        primary_cycles = float(components[primary_idx]['n_cycles_in_band'])
    else:
        primary_amp1 = float('nan')
        primary_cycles = float('nan')

    return {
        'period_mhz': primary_period,
        'periods_mhz': [float(c['period_mhz']) for c in components],
        'components': components,
        'harmonic_family': harmonic_family,
        'n_components': int(len(components)),
        'primary_period_mhz': primary_period,
        'harmonic_ratio_tol': float(harmonic_ratio_tol),
        'n_harmonics': int(n_harmonics),
        'n_harmonics_per_component': [int(v) for v in nharm_per_comp],
        'dc': dc,
        'amp1': primary_amp1,
        'rss': float(best['rss']),
        'rms_before': rms_before,
        'rms_after': rms_after,
        'n_cycles_in_band': primary_cycles,
        'search': {
            **search_notes,
            'search_points': int(len(search_periods)),
            'refine_lo_mhz': float(lo),
            'refine_hi_mhz': float(hi),
            'refine_points': int(refine_n),
            'min_cycles_in_band': float(min_cycles_in_band),
            'harmonic_per_cycle_factor': float(harmonic_per_cycle_factor),
            'prefer_harmonic_components': bool(prefer_harmonic_components),
            'multi_component_mode': bool(use_multi_component),
            'selected_via': selected_via,
        },
        'model': model_all,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Source-agnostic ripple characterization from RR/LL residuals using fit-derived or known source model.'
    )
    parser.add_argument('--fits', required=True, help='Input UVFITS path')
    parser.add_argument('--index-cache', default=None, help='Row-index cache path (.npz)')
    parser.add_argument('--source', required=True, help='Source name in UVFITS')
    parser.add_argument('--chan-range', nargs=2, type=int, default=(0, 127), metavar=('START', 'END'))
    parser.add_argument('--elevation-min', type=float, default=25.0)
    parser.add_argument('--elevation-max', type=float, default=None)
    parser.add_argument('--n-harmonics', type=int, default=3, help='Legacy max harmonic count (kept for backward compatibility)')
    parser.add_argument('--period-min-mhz', type=float, default=None, help='Minimum period (MHz) for harmonic residual search (manual mode)')
    parser.add_argument('--period-max-mhz', type=float, default=None, help='Maximum period (MHz) for harmonic residual search (manual mode)')
    parser.add_argument('--period-bounds-mode', choices=['fourier', 'manual'], default='fourier', help='Set period bounds from Fourier sampling theory or manual min/max values')
    parser.add_argument('--period-grid-size', type=int, default=1200, help='Grid size for period search')
    parser.add_argument('--period-search-mode', choices=['auto', 'grid'], default='auto', help='Period search strategy: auto uses FFT peak seeding + refinement; grid uses full dense grid only.')
    parser.add_argument('--peak-snr-threshold', type=float, default=3.0, help='Minimum peak-power SNR above noise floor to be considered a candidate period (auto mode only)')
    parser.add_argument('--max-peaks', type=int, default=0, help='Maximum number of significant FFT peaks used in fitting; 0 means use all significant peaks')
    parser.add_argument('--period-refine-frac', type=float, default=0.15, help='Fractional half-width around best seed period for local refinement')
    parser.add_argument('--min-cycles-in-band', type=float, default=0.0, help='Legacy cycle filter (default disabled)')
    parser.add_argument('--n-ripple-components', type=int, default=0, help='Legacy component-count control (ignored by default FFT-peak mode)')
    parser.add_argument('--harmonic-ratio-tol', type=float, default=0.06, help='Relative tolerance for labeling a component as a harmonic of the primary period')
    parser.add_argument('--harmonic-per-cycle-factor', type=float, default=1.0, help='Practical cap factor for harmonics from cycles-in-band: Nh <= floor(factor * cycles)')
    parser.add_argument('--prefer-harmonic-components', action='store_true', default=True, help='Prefer selecting harmonic-family peaks before independent peaks when adding components')
    parser.add_argument('--no-prefer-harmonic-components', action='store_false', dest='prefer_harmonic_components', help='Disable harmonic-first component selection and use pure SNR order')
    parser.add_argument('--velocity-factor', type=float, default=1.0, help='Cable velocity factor for period→length conversion (default: 1.0)')
    parser.add_argument(
        '--physical-model-mode',
        choices=['fit', 'known'],
        default='fit',
        help='Physical spectrum mode used for residual construction: fit (log-log fit) or known (registered source model).',
    )
    parser.add_argument(
        '--source-model-mode',
        choices=['free-beta', 'beta0', 'both'],
        default='beta0',
        help='When --physical-model-mode=fit, choose physical fit branch: free-beta, beta0, or both.',
    )
    parser.add_argument('--outdir', default='experimental/out', help='Output directory')
    parser.add_argument('--outfile-prefix', default=None, help='Output file prefix; default: ripple_characterisation_src-<source>')
    parser.add_argument(
        '--annotation-min-separation-px',
        type=float,
        default=14.0,
        help='Minimum pixel separation enforced between row-3 annotation text boxes.',
    )
    args = parser.parse_args()

    fits_path = Path(args.fits).expanduser().resolve()
    cache_path = Path(args.index_cache).expanduser().resolve() if args.index_cache else None
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    outfile_prefix = args.outfile_prefix if args.outfile_prefix else f'ripple_characterisation_src-{_source_slug(args.source)}'

    index = q.get_or_build_row_index(
        fits_path,
        cache_path=cache_path,
        validation_mode='fast+sha',
        write_cache=bool(cache_path),
    )

    vis = q.load_vis_for_source(
        index,
        source=args.source,
        chan_range=tuple(args.chan_range) if args.chan_range else None,
        stokes=['RR', 'LL'],
        max_rows=0,
        elevation_min_deg=args.elevation_min,
        elevation_max_deg=args.elevation_max,
    )

    freqs_hz = np.asarray(vis['freqs_hz'], dtype=np.float64)
    freqs_mhz = freqs_hz / 1e6

    if args.period_bounds_mode == 'fourier':
        period_min_mhz, period_max_mhz = derive_fourier_period_bounds(freqs_hz)
    else:
        period_min_mhz = float(args.period_min_mhz) if args.period_min_mhz is not None else 0.2
        period_max_mhz = float(args.period_max_mhz) if args.period_max_mhz is not None else 20.0
    if (not np.isfinite(period_min_mhz)) or (not np.isfinite(period_max_mhz)) or period_min_mhz <= 0 or period_max_mhz <= period_min_mhz:
        raise ValueError(f'Invalid period bounds: min={period_min_mhz}, max={period_max_mhz}')

    rr = vector_avg_real(vis, 'RR')
    ll = vector_avg_real(vis, 'LL')

    fit_rr = fit_models(freqs_hz, rr)
    fit_ll = fit_models(freqs_hz, ll)

    source_model_mode = args.source_model_mode
    physical_model_mode = args.physical_model_mode
    velocity_factor = float(args.velocity_factor)
    known_model_used = None
    known_phys = None
    known_model_for_crosscheck = None
    known_model_crosscheck_name = None

    try:
        known_model_crosscheck_name, known_model_for_crosscheck = lookup_known_flux_model(args.source)
    except ValueError:
        known_model_for_crosscheck = None
        known_model_crosscheck_name = None

    model_variants = []
    rr_residuals: dict[str, np.ndarray] = {}
    ll_residuals: dict[str, np.ndarray] = {}

    if physical_model_mode == 'fit':
        rr_phys_free = np.asarray(fit_rr['S_quad'], dtype=np.float64)
        ll_phys_free = np.asarray(fit_ll['S_quad'], dtype=np.float64)
        rr_phys_beta0 = np.asarray(fit_rr['S_linear'], dtype=np.float64)
        ll_phys_beta0 = np.asarray(fit_ll['S_linear'], dtype=np.float64)

        if source_model_mode in ('free-beta', 'both'):
            rr_ripple_free = np.full_like(rr, np.nan, dtype=np.float64)
            ll_ripple_free = np.full_like(ll, np.nan, dtype=np.float64)
            rr_ok_free = np.isfinite(rr) & np.isfinite(rr_phys_free) & (rr_phys_free > 0)
            ll_ok_free = np.isfinite(ll) & np.isfinite(ll_phys_free) & (ll_phys_free > 0)
            rr_ripple_free[rr_ok_free] = rr[rr_ok_free] / rr_phys_free[rr_ok_free] - 1.0
            ll_ripple_free[ll_ok_free] = ll[ll_ok_free] / ll_phys_free[ll_ok_free] - 1.0
            rr_residuals['free_beta'] = rr_ripple_free
            ll_residuals['free_beta'] = ll_ripple_free
            model_variants.append({
                'key': 'free_beta',
                'label': 'free-β',
                'residual_color': 'tab:blue',
                'residual_alpha': 0.50,
                'residual_ms': 4,
                'fit_color': 'tab:red',
                'fit_ls': '-',
                'fit_lw': 2.0,
                'scatter_color': 'tab:red',
                'scatter_marker': 'o',
                'profile_ls': '-',
                'profile_color': 'tab:red',
            })

        if source_model_mode in ('beta0', 'both'):
            rr_ripple_beta0 = np.full_like(rr, np.nan, dtype=np.float64)
            ll_ripple_beta0 = np.full_like(ll, np.nan, dtype=np.float64)
            rr_ok_beta0 = np.isfinite(rr) & np.isfinite(rr_phys_beta0) & (rr_phys_beta0 > 0)
            ll_ok_beta0 = np.isfinite(ll) & np.isfinite(ll_phys_beta0) & (ll_phys_beta0 > 0)
            rr_ripple_beta0[rr_ok_beta0] = rr[rr_ok_beta0] / rr_phys_beta0[rr_ok_beta0] - 1.0
            ll_ripple_beta0[ll_ok_beta0] = ll[ll_ok_beta0] / ll_phys_beta0[ll_ok_beta0] - 1.0
            rr_residuals['beta0'] = rr_ripple_beta0
            ll_residuals['beta0'] = ll_ripple_beta0
            model_variants.append({
                'key': 'beta0',
                'label': 'β=0',
                'residual_color': 'tab:green',
                'residual_alpha': 0.40,
                'residual_ms': 3,
                'fit_color': 'tab:orange',
                'fit_ls': '--',
                'fit_lw': 1.8,
                'scatter_color': 'tab:orange',
                'scatter_marker': 's',
                'profile_ls': '--',
                'profile_color': 'tab:orange',
            })
    else:
        known_model_name, known_model_fn = lookup_known_flux_model(args.source)
        known_model_used = str(known_model_name)
        known_phys = np.asarray(known_model_fn(freqs_hz), dtype=np.float64)
        rr_ripple_known = np.full_like(rr, np.nan, dtype=np.float64)
        ll_ripple_known = np.full_like(ll, np.nan, dtype=np.float64)
        rr_ok_known = np.isfinite(rr) & np.isfinite(known_phys) & (known_phys > 0)
        ll_ok_known = np.isfinite(ll) & np.isfinite(known_phys) & (known_phys > 0)
        rr_ripple_known[rr_ok_known] = rr[rr_ok_known] / known_phys[rr_ok_known] - 1.0
        ll_ripple_known[ll_ok_known] = ll[ll_ok_known] / known_phys[ll_ok_known] - 1.0
        rr_residuals['known_model'] = rr_ripple_known
        ll_residuals['known_model'] = ll_ripple_known
        model_variants.append({
            'key': 'known_model',
            'label': f'known model ({known_model_used})',
            'residual_color': 'tab:green',
            'residual_alpha': 0.45,
            'residual_ms': 3,
            'fit_color': 'tab:orange',
            'fit_ls': '--',
            'fit_lw': 1.9,
            'scatter_color': 'tab:orange',
            'scatter_marker': 'D',
            'profile_ls': '--',
            'profile_color': 'tab:orange',
        })

    known_model_local_alpha_crosscheck = None
    known_alpha_rr_nu0 = None
    known_alpha_ll_nu0 = None
    if callable(known_model_for_crosscheck):
        known_alpha_rr_nu0 = local_spectral_index_at_nu0(known_model_for_crosscheck, float(fit_rr['nu0_hz']))
        known_alpha_ll_nu0 = local_spectral_index_at_nu0(known_model_for_crosscheck, float(fit_ll['nu0_hz']))
        known_model_local_alpha_crosscheck = {
            'available': True,
            'known_model': str(known_model_crosscheck_name),
            'convention': 'alpha = d ln S / d ln nu (S ∝ nu^alpha)',
            'rr': {
                'nu0_hz': float(fit_rr['nu0_hz']),
                'alpha_known_local_nu0': float(known_alpha_rr_nu0),
                'alpha_fit_quad_nu0': float(fit_rr['quad']['alpha_nu0']),
                'alpha_fit_linear_nu0': float(fit_rr['linear']['alpha_nu0']),
                'delta_quad_minus_known': float(fit_rr['quad']['alpha_nu0'] - known_alpha_rr_nu0) if np.isfinite(known_alpha_rr_nu0) else None,
                'delta_linear_minus_known': float(fit_rr['linear']['alpha_nu0'] - known_alpha_rr_nu0) if np.isfinite(known_alpha_rr_nu0) else None,
            },
            'll': {
                'nu0_hz': float(fit_ll['nu0_hz']),
                'alpha_known_local_nu0': float(known_alpha_ll_nu0),
                'alpha_fit_quad_nu0': float(fit_ll['quad']['alpha_nu0']),
                'alpha_fit_linear_nu0': float(fit_ll['linear']['alpha_nu0']),
                'delta_quad_minus_known': float(fit_ll['quad']['alpha_nu0'] - known_alpha_ll_nu0) if np.isfinite(known_alpha_ll_nu0) else None,
                'delta_linear_minus_known': float(fit_ll['linear']['alpha_nu0'] - known_alpha_ll_nu0) if np.isfinite(known_alpha_ll_nu0) else None,
            },
        }

    if len(model_variants) == 0:
        raise ValueError('No physical-model variant selected. Check --physical-model-mode and --source-model-mode.')

    ripple_fit_rr = {}
    ripple_fit_ll = {}
    for variant in model_variants:
        key = str(variant['key'])
        ripple_fit_rr[key] = fit_harmonic_residual(
            freqs_hz, rr_residuals[key],
            n_harmonics=args.n_harmonics,
            period_min_mhz=period_min_mhz,
            period_max_mhz=period_max_mhz,
            n_period_grid=args.period_grid_size,
            period_search_mode=args.period_search_mode,
            peak_snr_threshold=args.peak_snr_threshold,
            max_peaks=args.max_peaks,
            period_refine_frac=args.period_refine_frac,
            min_cycles_in_band=args.min_cycles_in_band,
            n_components=args.n_ripple_components,
            harmonic_ratio_tol=args.harmonic_ratio_tol,
            harmonic_per_cycle_factor=args.harmonic_per_cycle_factor,
            prefer_harmonic_components=args.prefer_harmonic_components,
        )
        ripple_fit_ll[key] = fit_harmonic_residual(
            freqs_hz, ll_residuals[key],
            n_harmonics=args.n_harmonics,
            period_min_mhz=period_min_mhz,
            period_max_mhz=period_max_mhz,
            n_period_grid=args.period_grid_size,
            period_search_mode=args.period_search_mode,
            peak_snr_threshold=args.peak_snr_threshold,
            max_peaks=args.max_peaks,
            period_refine_frac=args.period_refine_frac,
            min_cycles_in_band=args.min_cycles_in_band,
            n_components=args.n_ripple_components,
            harmonic_ratio_tol=args.harmonic_ratio_tol,
            harmonic_per_cycle_factor=args.harmonic_per_cycle_factor,
            prefer_harmonic_components=args.prefer_harmonic_components,
        )

    plot_path = outdir / f'{outfile_prefix}.png'

    fig, axes = plt.subplots(3, 2, figsize=(14.5, 13.2), sharex=False)

    x_ticks = [315, 320, 325, 330]
    top_panel_data = [
        (axes[0, 0], rr, fit_rr, 'RR baseline-averaged spectrum (log-log)', 'tab:blue', 'RR'),
        (axes[0, 1], ll, fit_ll, 'LL baseline-averaged spectrum (log-log)', 'tab:purple', 'LL'),
    ]

    for ax, spec, fit, title, color, pol in top_panel_data:
        mask = np.isfinite(spec) & (spec > 0)

        ax.loglog(freqs_mhz[mask], spec[mask], '.', ms=4, alpha=0.55, color=color, label='Data')
        if physical_model_mode == 'fit':
            if source_model_mode in ('free-beta', 'both'):
                ax.loglog(freqs_mhz, fit['S_quad'], '-', lw=2.2, color='tab:orange', label='Curvature fit (β free)')
            if source_model_mode in ('beta0', 'both'):
                ax.loglog(freqs_mhz, fit['S_linear'], '--', lw=2.1, color='tab:green', label='Forced β = 0')
        else:
            ax.loglog(freqs_mhz, known_phys, '-', lw=2.2, color='k', label=f'Known flux model ({known_model_used})')

        lines = [f"Physical model mode = {physical_model_mode}"]
        if physical_model_mode == 'fit':
            if source_model_mode in ('free-beta', 'both'):
                lines.extend([
                    f"α(ν₀), free = {fit['quad']['alpha_nu0']:.5f}",
                    f"β, free = {fit['quad']['beta']:.5f}",
                    f"RSS(log), free = {fit['quad']['rss_log']:.3e}",
                ])
            if source_model_mode in ('beta0', 'both'):
                lines.extend([
                    f"α(ν₀) = {fit['linear']['alpha_nu0']:.5f}",
                    f"RSS(log) = {fit['linear']['rss_log']:.3e}",
                ])
        else:
            lines.append(f"Known model = {known_model_used}")
        known_alpha_here = known_alpha_rr_nu0 if pol == 'RR' else known_alpha_ll_nu0
        if known_model_crosscheck_name is not None and known_alpha_here is not None and np.isfinite(known_alpha_here):
            lines.append(f"Known α(ν₀), {known_model_crosscheck_name} = {known_alpha_here:.5f}")
        txt = '\n'.join(lines)

        ax.text(
            0.03, 0.04, txt,
            transform=ax.transAxes,
            fontsize=10,
            linespacing=1.35,
            bbox=dict(boxstyle='round,pad=0.45', facecolor='white', alpha=0.95, edgecolor='gray'),
        )

        ax.set_title(title)
        ax.grid(True, which='major', alpha=0.24)
        ax.grid(True, which='minor', alpha=0.10)
        ax.set_xlabel('Frequency (MHz, log scale)')
        ax.set_xticks(x_ticks)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=tuple(float(v) for v in range(1, 10)), numticks=14))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}' if y > 0 else ''))
        ax.yaxis.set_minor_formatter(NullFormatter())

    axes[0, 0].set_ylabel('Flux Density (Jy)')
    axes[0, 0].legend(loc='upper right', fontsize=9)

    # Second row: ripple characterisation residuals r = S/S_phys - 1.
    ripple_panel_data = [
        (
            axes[1, 0],
            'RR ripple residual: $S_{RR}/S_{phys}-1$',
            'tab:blue',
            'RR',
        ),
        (
            axes[1, 1],
            'LL ripple residual: $S_{LL}/S_{phys}-1$',
            'tab:purple',
            'LL',
        ),
    ]

    for ax, title, color, pol in ripple_panel_data:
        for variant in model_variants:
            key = str(variant['key'])
            r_spec = rr_residuals[key] if pol == 'RR' else ll_residuals[key]
            rfit = ripple_fit_rr.get(key) if pol == 'RR' else ripple_fit_ll.get(key)
            if rfit is None:
                continue
            mask_v = np.isfinite(r_spec)
            ax.plot(
                freqs_mhz[mask_v],
                r_spec[mask_v],
                '.',
                ms=float(variant['residual_ms']),
                alpha=float(variant['residual_alpha']),
                color=str(variant['residual_color']) if len(model_variants) > 1 else color,
                label=f"Residual (S_phys: {variant['label']})",
            )
            ax.plot(
                freqs_mhz,
                rfit['model'],
                str(variant['fit_ls']),
                lw=float(variant['fit_lw']),
                color=str(variant['fit_color']),
                label=f"Multi-ripple fit ({variant['label']})",
            )

        ax.axhline(0.0, color='k', lw=0.9, alpha=0.5)
        ax.set_xscale('linear')
        ax.set_yscale('linear')

        ax.set_title(title)
        ax.grid(True, which='major', alpha=0.24)
        ax.grid(True, which='minor', alpha=0.10)
        ax.set_xlabel('Frequency (MHz, linear scale)')
        ax.set_ylabel('Residual Ripple (fractional)')
        ax.set_xticks(x_ticks)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())

    # Legend removed for reduced clutter; panel titles and styling are sufficient.

    # Third row: fitted peak power profiles vs frequency, plus compact text summary (top peaks + phase).
    def _place_nonoverlap_annotations(ax, items, fontsize=8.2, min_separation_px: float = 14.0):
        if not items:
            return
        sep = max(0.0, float(min_separation_px))
        sep_x = sep
        sep_y = 0.7 * sep

        angles_deg = [0.0, 15.0, -15.0, 30.0, -30.0, 45.0, -45.0, 60.0, -60.0,
                      75.0, -75.0, 90.0, -90.0, 105.0, -105.0, 120.0, -120.0,
                      135.0, -135.0, 150.0, -150.0, 165.0, -165.0, 180.0]
        radii = [
            max(22.0, 0.9 * sep),
            max(34.0, 1.5 * sep),
            max(48.0, 2.1 * sep),
            max(66.0, 2.9 * sep),
            max(88.0, 3.9 * sep),
            max(114.0, 5.0 * sep),
            max(146.0, 6.4 * sep),
            max(184.0, 8.0 * sep),
        ]

        ax.figure.canvas.draw()
        axes_bbox = ax.get_window_extent()
        trans = ax.transData

        def _bbox_intersects(a, b, margin_x=0.0, margin_y=0.0):
            ax0, ay0, ax1, ay1 = a
            bx0, by0, bx1, by1 = b
            return not (
                (ax1 + margin_x < bx0) or
                (ax0 - margin_x > bx1) or
                (ay1 + margin_y < by0) or
                (ay0 - margin_y > by1)
            )

        point_pixels: list[np.ndarray] = []

        def _candidate_cost(cand, placed):
            x0, y0, x1, y1 = cand['bbox']
            cx = 0.5 * (x0 + x1)
            cy = 0.5 * (y0 + y1)
            outside = (x0 < axes_bbox.x0) or (x1 > axes_bbox.x1) or (y0 < axes_bbox.y0) or (y1 > axes_bbox.y1)
            overlap_count = 0
            near_count = 0
            overlap_area = 0.0
            direction_penalty = 0.0
            direction_hits = 0
            min_edge_clearance = float('inf')
            for other in placed:
                ox0, oy0, ox1, oy1 = other['bbox']
                ocx = 0.5 * (ox0 + ox1)
                ocy = 0.5 * (oy0 + oy1)
                x_gap = max(0.0, max(ox0 - x1, x0 - ox1))
                y_gap = max(0.0, max(oy0 - y1, y0 - oy1))
                edge_clearance = float(np.hypot(x_gap, y_gap))
                if edge_clearance < min_edge_clearance:
                    min_edge_clearance = edge_clearance
                if _bbox_intersects((x0, y0, x1, y1), (ox0, oy0, ox1, oy1)):
                    overlap_count += 1
                    overlap_area += max(0.0, min(x1, ox1) - max(x0, ox0)) * max(0.0, min(y1, oy1) - max(y0, oy0))
                if _bbox_intersects((x0, y0, x1, y1), (ox0, oy0, ox1, oy1), margin_x=sep_x, margin_y=sep_y):
                    near_count += 1
                vx = ocx - cx
                vy = ocy - cy
                dist = float(np.hypot(vx, vy))
                if dist > 1e-6:
                    cand_ang = float(np.arctan2(cand['dy'], cand['dx']))
                    other_ang = float(np.arctan2(vy, vx))
                    ang_diff = abs((cand_ang - other_ang + np.pi) % (2.0 * np.pi) - np.pi)
                    # Penalize candidates that point into a crowded sector.
                    if ang_diff <= np.deg2rad(28.0):
                        direction_hits += 1
                        direction_penalty += max(0.0, 1.0 - (ang_diff / np.deg2rad(28.0))) * (140.0 / (1.0 + dist / 80.0))
            point_cover_count = 0
            point_margin = 2.0
            for point_xy in point_pixels:
                px = float(point_xy[0])
                py = float(point_xy[1])
                if (x0 - point_margin) <= px <= (x1 + point_margin) and (y0 - point_margin) <= py <= (y1 + point_margin):
                    point_cover_count += 1

            valid = (not outside) and (overlap_count == 0) and (near_count == 0) and (point_cover_count == 0)
            score = (
                (2.0e6 if outside else 0.0)
                + (5.0e5 * overlap_count)
                + (1.5e5 * near_count)
                + (9.0e5 * point_cover_count)
                + direction_penalty
                + (2.0e4 * direction_hits)
                + (2.5e3 / max(min_edge_clearance, 1.0))
                + (20.0 * overlap_area)
                + (0.12 * cand['radius'])
                + (0.02 * (abs(cand['dx']) + abs(cand['dy'])))
            )
            return score, valid, outside, overlap_count, near_count, point_cover_count

        prepared = []
        for fx, ay, note, color in items:
            note_lines = str(note).splitlines() or [str(note)]
            widest_line = max(note_lines, key=len)
            approx_w = max(124.0, (7.6 * len(widest_line)) + 18.0)
            approx_h = max(30.0, (fontsize * 1.70 * len(note_lines)) + 10.0)
            base_px = trans.transform((float(fx), float(ay)))
            candidates = []
            for radius in radii:
                for ang_deg in angles_deg:
                    ang = np.deg2rad(float(ang_deg))
                    dx = float(radius) * float(np.cos(ang))
                    dy = float(radius) * float(np.sin(ang))
                    if abs(dx) < 2.0 and abs(dy) < 2.0:
                        continue
                    cx = base_px[0] + dx
                    cy = base_px[1] + dy
                    x0 = cx - 0.5 * approx_w
                    x1 = cx + 0.5 * approx_w
                    y0 = cy - 0.5 * approx_h
                    y1 = cy + 0.5 * approx_h
                    candidates.append({
                        'dx': dx,
                        'dy': dy,
                        'radius': float(radius),
                        'bbox': (x0, y0, x1, y1),
                    })
            prepared.append({
                'fx': float(fx),
                'ay': float(ay),
                'note': note,
                'color': color,
                'base_px': base_px,
                'w': approx_w,
                'h': approx_h,
                'candidates': candidates,
                'pref_dy_sign': 0,
            })

        n_items = len(prepared)
        if n_items == 0:
            return

        # Local-shape prior for initial direction:
        # - local maxima (neighbors lower) prefer +dy (upward)
        # - local minima (neighbors higher) prefer -dy (downward)
        # This is a soft bias; collision constraints still dominate.
        by_x = sorted(range(n_items), key=lambda i: prepared[i]['fx'])
        for pos, idx in enumerate(by_x):
            y0 = float(prepared[idx]['ay'])
            prev_y = float(prepared[by_x[pos - 1]]['ay']) if pos > 0 else None
            next_y = float(prepared[by_x[pos + 1]]['ay']) if pos < (n_items - 1) else None
            pref = 0
            if (prev_y is not None) and (next_y is not None):
                if (y0 > prev_y) and (y0 > next_y):
                    pref = 1
                elif (y0 < prev_y) and (y0 < next_y):
                    pref = -1
            prepared[idx]['pref_dy_sign'] = pref

        point_pixels = [np.asarray(item['base_px'], dtype=np.float64) for item in prepared]

        base_positions = np.array([item['base_px'] for item in prepared], dtype=np.float64)
        if n_items > 1:
            dist_mat = np.sqrt(np.sum((base_positions[:, None, :] - base_positions[None, :, :]) ** 2, axis=2))
            np.fill_diagonal(dist_mat, np.inf)
            hardness = np.min(dist_mat, axis=1)
        else:
            hardness = np.array([0.0], dtype=np.float64)

        order_templates = [
            list(np.argsort(hardness)),
            list(np.argsort(-hardness)),
            list(np.argsort(base_positions[:, 0])),
            list(np.argsort(-base_positions[:, 0])),
            list(np.argsort(base_positions[:, 1])),
            list(np.argsort(-base_positions[:, 1])),
        ]

        def _choose_best_for_index(idx, placed):
            item = prepared[idx]
            pref_dy_sign = int(item.get('pref_dy_sign', 0))
            best_any = None
            best_valid = None
            for cand in item['candidates']:
                score, valid, outside, overlap_count, near_count, point_cover_count = _candidate_cost(cand, placed)
                if pref_dy_sign > 0 and float(cand['dy']) < 0.0:
                    score += 3.5e4
                elif pref_dy_sign < 0 and float(cand['dy']) > 0.0:
                    score += 3.5e4
                choice = dict(cand)
                choice.update({
                    'score': score,
                    'valid': valid,
                    'outside': outside,
                    'overlap_count': overlap_count,
                    'near_count': near_count,
                    'point_cover_count': point_cover_count,
                })
                if (best_any is None) or (choice['score'] < best_any['score']):
                    best_any = choice
                if valid and ((best_valid is None) or (choice['score'] < best_valid['score'])):
                    best_valid = choice
            if best_valid is not None:
                return best_valid

            # If nothing is truly valid, still prefer the candidate that most
            # improves edge clearance, even if that means increasing radius or
            # changing angle. This gives the solver a direct route to resolve a
            # conflicting pair via a different vector rather than only nudging.
            return best_any

        def _layout_cost(layout):
            placed = [c for c in layout if c is not None]
            cost = 0.0
            for cand in placed:
                cost += 0.08 * cand['radius']
                if cand.get('outside', False):
                    cost += 2.0e6
                x0, y0, x1, y1 = cand['bbox']
                for point_xy in point_pixels:
                    px = float(point_xy[0])
                    py = float(point_xy[1])
                    if (x0 - 2.0) <= px <= (x1 + 2.0) and (y0 - 2.0) <= py <= (y1 + 2.0):
                        cost += 1.2e6
            for i in range(len(placed)):
                for j in range(i + 1, len(placed)):
                    c1 = placed[i]
                    c2 = placed[j]
                    x0, y0, x1, y1 = c1['bbox']
                    ox0, oy0, ox1, oy1 = c2['bbox']
                    if _bbox_intersects((x0, y0, x1, y1), (ox0, oy0, ox1, oy1)):
                        cost += 7.5e5
                        cost += 10.0 * max(0.0, min(x1, ox1) - max(x0, ox0)) * max(0.0, min(y1, oy1) - max(y0, oy0))
                    if _bbox_intersects((x0, y0, x1, y1), (ox0, oy0, ox1, oy1), margin_x=sep_x, margin_y=sep_y):
                        cost += 1.25e5
            return cost

        def _solve_one_pass(order):
            layout = [None] * n_items
            placed = []
            for idx in order:
                cand = _choose_best_for_index(idx, placed)
                layout[idx] = cand
                placed.append(cand)
            return layout

        def _refine_layout(layout, max_passes=4):
            current = list(layout)
            for _ in range(max_passes):
                changed = False
                conflict_order = []
                for idx, cand in enumerate(current):
                    if cand is None:
                        conflict_order.append((1e9, idx))
                        continue
                    _, valid, outside, overlap_count, near_count, point_cover_count = _candidate_cost(cand, [c for j, c in enumerate(current) if j != idx and c is not None])
                    conflict_score = (
                        (5000.0 if outside else 0.0)
                        + (2600.0 * overlap_count)
                        + (900.0 * near_count)
                        + (3200.0 * point_cover_count)
                    )
                    conflict_order.append((conflict_score, idx))
                for _, idx in sorted(conflict_order, reverse=True):
                    others = [c for j, c in enumerate(current) if j != idx and c is not None]
                    best = _choose_best_for_index(idx, others)
                    current_item = current[idx]
                    if current_item is None:
                        current[idx] = best
                        changed = True
                    else:
                        assert best is not None
                        if best['score'] + 1e-6 < current_item['score']:
                            current[idx] = best
                            changed = True
                if not changed:
                    break
            return current

        best_layout = None
        best_cost = None
        for order in order_templates:
            layout = _solve_one_pass(order)
            layout = _refine_layout(layout, max_passes=4)
            cost = _layout_cost(layout)
            if (best_cost is None) or (cost < best_cost):
                best_cost = cost
                best_layout = layout

        if best_layout is None:
            return

        for idx, cand in enumerate(best_layout):
            item = prepared[idx]
            use_arrow = (abs(cand['dx']) >= 20.0) or (abs(cand['dy']) >= 20.0)
            ax.annotate(
                item['note'],
                (item['fx'], item['ay']),
                textcoords='offset pixels',
                xytext=(cand['dx'], cand['dy']),
                fontsize=fontsize,
                color=item['color'],
                ha='center',
                va='center',
                arrowprops=(
                    dict(
                        arrowstyle='-',
                        color=item['color'],
                        lw=0.6,
                        alpha=0.55,
                        shrinkA=0,
                        shrinkB=2,
                    ) if use_arrow else None
                ),
            )

    harmonic_panel_data = [
        (axes[2, 0], 'RR', 'RR fitted-peak power profile'),
        (axes[2, 1], 'LL', 'LL fitted-peak power profile'),
    ]

    for ax, pol, title in harmonic_panel_data:
        x_all = []
        y_all = []
        annotation_items = []

        for variant in model_variants:
            key = str(variant['key'])
            rfit = ripple_fit_rr.get(key) if pol == 'RR' else ripple_fit_ll.get(key)
            if rfit is None:
                continue

            comps = rfit.get('components', [])
            if comps:
                x = [float(c.get('frequency_cpmhz', float('nan'))) for c in comps]
                y = [float(c.get('amp1', float('nan'))) for c in comps]
                x_all.extend(x)
                y_all.extend(y)
                ax.scatter(
                    x,
                    y,
                    marker=str(variant['scatter_marker']),
                    color=str(variant['scatter_color']),
                    s=34,
                    alpha=0.9,
                    label=f"Peak amplitude vs ripple frequency ({variant['label']})",
                )
                for fx, ay, comp in zip(x, y, comps):
                    period_mhz = float(comp.get('period_mhz', float('nan')))
                    p_m = period_to_cable_length_m(float(comp.get('period_mhz', float('nan'))), velocity_factor=velocity_factor)
                    ph = float(comp.get('phase1_deg', float('nan')))
                    note = f"Δν={period_mhz:.3f} MHz\nL={p_m:.1f} m, φ={ph:.1f}°"
                    annotation_items.append((fx, ay, note, 'darkblue'))
                if len(x) > 1:
                    ord_idx = np.argsort(np.asarray(x, dtype=np.float64))
                    xs = np.asarray(x, dtype=np.float64)[ord_idx]
                    ys = np.asarray(y, dtype=np.float64)[ord_idx]
                    ax.plot(xs, ys, str(variant['profile_ls']), color=str(variant['profile_color']), alpha=0.35, lw=1.0)

        ax.set_title(title)
        ax.set_xlabel('Ripple Frequency (Cycles / MHz)')
        ax.set_ylabel('Amplitude (fractional; S/S_phys - 1)')
        ax.set_yscale('log')

        x_arr = np.asarray(x_all, dtype=np.float64)
        y_arr = np.asarray(y_all, dtype=np.float64)
        valid = np.isfinite(x_arr) & np.isfinite(y_arr) & (y_arr > 0)
        if np.any(valid):
            xv = x_arr[valid]
            yv = y_arr[valid]
            x_min = float(np.min(xv))
            x_max = float(np.max(xv))
            x_span = x_max - x_min
            x_pad = 0.08 * x_span if x_span > 0 else max(0.05 * max(x_max, 1.0), 0.05)
            ax.set_xlim(max(0.0, x_min - x_pad), x_max + 2.0 * x_pad)

            y_min = float(np.min(yv))
            y_max = float(np.max(yv))
            y_low = max(y_min / 1.8, 1e-12)
            y_high = y_max * 3.0
            ax.set_ylim(y_low, y_high)

        _place_nonoverlap_annotations(
            ax,
            annotation_items,
            fontsize=8.2,
            min_separation_px=float(args.annotation_min_separation_px),
        )
        if annotation_items:
            ax.text(
                0.02,
                0.04,
                'Note: arrows indicate displaced labels',
                transform=ax.transAxes,
                fontsize=8.0,
                color='dimgray',
                alpha=0.9,
            )

        ax.grid(True, which='major', alpha=0.24)
        ax.grid(True, which='minor', alpha=0.10)
        # Legend removed to avoid clutter; title and axis labels identify the quantity.

    fig.suptitle(f'Ripple characterisation ({args.source})', fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(plot_path, dpi=170)
    plt.close(fig)

    def _ft_search_view(search: dict | None) -> dict:
        if not isinstance(search, dict):
            return {}
        keep_keys = [
            'mode',
            'peak_snr_threshold',
            'max_peaks',
            'period_min_used_mhz',
            'period_max_used_mhz',
            'spectral_candidates_mhz',
            'spectral_candidate_snrs',
            'n_significant_peaks',
            'selected_component_seeds_mhz',
            'selected_component_seed_snrs',
            'component_selection_policy',
            'n_components_requested',
            'n_components_used',
            'multi_component_mode',
            'selected_via',
        ]
        return {k: search[k] for k in keep_keys if k in search}

    def _ripple_summary_view(rfit: dict | None) -> dict | None:
        if rfit is None:
            return None
        components = list(rfit.get('components', []))
        return {
            'n_components': int(rfit.get('n_components', len(components))),
            'periods_mhz': [float(c.get('period_mhz', float('nan'))) for c in components],
            'components': components,
            'rms_before': rfit.get('rms_before'),
            'rms_after': rfit.get('rms_after'),
            'rss': rfit.get('rss'),
            'search': _ft_search_view(rfit.get('search')),
            'model_uses_displayed_components': True,
        }

    summary = {
        'input_uvfits': str(fits_path),
        'input_uvfits_filename': fits_path.name,
        'index_cache': str(cache_path) if cache_path else None,
        'source': args.source,
        'velocity_factor': velocity_factor,
        'physical_model_mode': physical_model_mode,
        'source_model_mode': source_model_mode if physical_model_mode == 'fit' else None,
        'known_model_used': known_model_used,
        'known_model_local_alpha_crosscheck': known_model_local_alpha_crosscheck,
        'model_variants': [str(v['key']) for v in model_variants],
        'period_search_mode': args.period_search_mode,
        'period_bounds_mode': args.period_bounds_mode,
        'period_bounds_used_mhz': {
            'min': float(period_min_mhz),
            'max': float(period_max_mhz),
        },
        'peak_snr_threshold': float(args.peak_snr_threshold),
        'max_peaks': int(args.max_peaks),
        'period_refine_frac': float(args.period_refine_frac),
        'fit_domain': 'log-log: ln S vs ln nu',
        'plot': str(plot_path),
        'rr': {
            'mask_n': fit_rr['mask_n'],
            'nu0_hz': fit_rr['nu0_hz'],
            'quad': fit_rr['quad'],
            'linear': fit_rr['linear'],
            'rss_improvement_fraction': float((fit_rr['linear']['rss_log'] - fit_rr['quad']['rss_log']) / fit_rr['linear']['rss_log']) if fit_rr['linear']['rss_log'] != 0 else None,
        },
        'll': {
            'mask_n': fit_ll['mask_n'],
            'nu0_hz': fit_ll['nu0_hz'],
            'quad': fit_ll['quad'],
            'linear': fit_ll['linear'],
            'rss_improvement_fraction': float((fit_ll['linear']['rss_log'] - fit_ll['quad']['rss_log']) / fit_ll['linear']['rss_log']) if fit_ll['linear']['rss_log'] != 0 else None,
        },
        'ripple_rr': {
            **{str(v['key']): _ripple_summary_view(ripple_fit_rr.get(str(v['key']))) for v in model_variants},
        },
        'ripple_ll': {
            **{str(v['key']): _ripple_summary_view(ripple_fit_ll.get(str(v['key']))) for v in model_variants},
        },
    }

    summary_path = outdir / f'{outfile_prefix}_summary.json'
    with summary_path.open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)

    print(f'WROTE plot: {plot_path}')
    print(f'WROTE summary: {summary_path}')
    print('DONE')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
