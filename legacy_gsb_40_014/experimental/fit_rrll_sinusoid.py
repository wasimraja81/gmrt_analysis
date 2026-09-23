#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import cal_apply as ca
import plotVis as pv
import ugmrt_query as q
from workflow_common import apply_overrides_to_globals, load_config_into_globals


CAL_FITS = None
INDEX_CACHE = None
SOURCE = None
CHAN_RANGE = None
SOLVE_TIMERANGE = None
SOLVE_ELEVATION_MIN_DEG = None
SOLVE_ELEVATION_MAX_DEG = None
SOLVE_UVRANGE_M = None
SOLVE_UVRANGE_KLAMBDA = None
EXCLUDE_FOR_PLOTS = []


def robust_sigma_mad(values: np.ndarray) -> float:
    finite_values = np.asarray(values, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return float('nan')
    median_value = float(np.nanmedian(finite_values))
    mad_value = float(np.nanmedian(np.abs(finite_values - median_value)))
    return 1.4826 * mad_value


def _canonical_source_name(source_name: str) -> str:
    return ''.join(ch for ch in str(source_name).upper() if ch.isalnum())


def evaluate_source_flux_model_jy(freq_hz: np.ndarray, source_name: str) -> np.ndarray:
    registry = getattr(q, '_FLUX_MODEL_REGISTRY', {})
    canonical_registry = {
        _canonical_source_name(name): func
        for name, func in registry.items()
        if callable(func)
    }
    source_key = _canonical_source_name(source_name)
    model_func = canonical_registry.get(source_key)
    if model_func is None:
        raise ValueError(
            f'No registered Perley-Butler flux model for source={source_name!r}. '
            f'Available: {sorted(registry.keys())}'
        )
    return np.asarray(model_func(np.asarray(freq_hz, dtype=np.float64)), dtype=np.float64)


def _nan_running_median(values: np.ndarray, half_window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.size)
    out = np.full(n, np.nan, dtype=np.float64)
    hw = int(max(0, half_window))
    for idx in range(n):
        lo = max(0, idx - hw)
        hi = min(n, idx + hw + 1)
        window = arr[lo:hi]
        finite = window[np.isfinite(window)]
        if finite.size > 0:
            out[idx] = float(np.nanmedian(finite))
    return out


def detect_channel_outliers(signal_jy: np.ndarray, clip_sigma: float, half_window: int) -> np.ndarray:
    arr = np.asarray(signal_jy, dtype=np.float64)
    if arr.size == 0 or clip_sigma <= 0:
        return np.zeros(arr.shape, dtype=bool)
    trend = _nan_running_median(arr, half_window=half_window)
    residual = arr - trend
    sigma = robust_sigma_mad(residual)
    if not np.isfinite(sigma) or sigma <= 0:
        return np.zeros(arr.shape, dtype=bool)
    bad = np.isfinite(residual) & (np.abs(residual) > float(clip_sigma) * sigma)
    return bad


def dilate_channel_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    r = int(max(0, radius))
    if r == 0 or m.size == 0:
        return m
    out = m.copy()
    idx = np.flatnonzero(m)
    n = m.size
    for i in idx:
        lo = max(0, int(i) - r)
        hi = min(n, int(i) + r + 1)
        out[lo:hi] = True
    return out


def detect_residual_outliers(residual_jy: np.ndarray, clip_sigma: float) -> np.ndarray:
    resid = np.asarray(residual_jy, dtype=np.float64)
    if resid.size == 0 or clip_sigma <= 0:
        return np.zeros(resid.shape, dtype=bool)
    sigma = robust_sigma_mad(resid)
    if not np.isfinite(sigma) or sigma <= 0:
        return np.zeros(resid.shape, dtype=bool)
    return np.isfinite(resid) & (np.abs(resid) > float(clip_sigma) * sigma)


def evaluate_multi_reflection_fit(freq_mhz: np.ndarray, fit_result: dict) -> np.ndarray:
    freq = np.asarray(freq_mhz, dtype=np.float64)
    model = fit_result['c0'] + fit_result['c1'] * freq
    model = model + fit_result['sin_coeff'] * np.sin(fit_result['omega'] * freq)
    model = model + fit_result['cos_coeff'] * np.cos(fit_result['omega'] * freq)
    for component in fit_result.get('multi_reflection_components', []):
        model = model + component['sin_coeff'] * np.sin(component['omega'] * freq)
        model = model + component['cos_coeff'] * np.cos(component['omega'] * freq)
    return model


def evaluate_transfer_fit_model(freq_mhz: np.ndarray, transfer_fit: dict, transfer_mode: str, template_ripple_jy: np.ndarray | None) -> np.ndarray:
    freq = np.asarray(freq_mhz, dtype=np.float64)
    model = transfer_fit['c0'] + transfer_fit['c1'] * freq
    if transfer_mode == 'fixed_template':
        if template_ripple_jy is None:
            raise ValueError('template_ripple_jy required for fixed_template model evaluation')
        model = model + transfer_fit['template_scale'] * np.asarray(template_ripple_jy, dtype=np.float64)
    elif transfer_mode == 'free_phase_template':
        for component in transfer_fit.get('template_components_fitted', []):
            model = model + component['sin_coeff'] * np.sin(component['omega'] * freq)
            model = model + component['cos_coeff'] * np.cos(component['omega'] * freq)
    else:
        model = model + transfer_fit['sin_coeff'] * np.sin(transfer_fit['omega'] * freq)
        model = model + transfer_fit['cos_coeff'] * np.cos(transfer_fit['omega'] * freq)
    for component in transfer_fit.get('multi_reflection_components', []):
        model = model + component['sin_coeff'] * np.sin(component['omega'] * freq)
        model = model + component['cos_coeff'] * np.cos(component['omega'] * freq)
    return model


def baseline_vector_average(vis: dict, corrected_cache: dict | None, product: str) -> dict:
    spectral_data, spectral_flags = pv._get_product_data(product, vis, corrected_cache, solution=None)
    finite = np.isfinite(spectral_data.real) & np.isfinite(spectral_data.imag)
    good = (~spectral_flags) & finite
    weights = np.asarray(vis['weight'])[:, :, 0]

    weighted_numerator = np.nansum(np.where(good, weights * spectral_data, 0.0), axis=0)
    weighted_denominator = np.nansum(np.where(good, weights, 0.0), axis=0)

    vector_average = np.full(spectral_data.shape[1], np.nan + 1j * np.nan, dtype=np.complex128)
    scalar_average_amplitude = np.full(spectral_data.shape[1], np.nan, dtype=np.float64)
    valid_channel = weighted_denominator > 0
    vector_average[valid_channel] = weighted_numerator[valid_channel] / weighted_denominator[valid_channel]

    scalar_numerator = np.nansum(np.where(good, weights * np.abs(spectral_data), 0.0), axis=0)
    scalar_average_amplitude[valid_channel] = scalar_numerator[valid_channel] / weighted_denominator[valid_channel]

    return {
        'vec': vector_average,
        'scalar_amp': scalar_average_amplitude,
        'valid': valid_channel,
    }


def fit_sinusoid_plus_linear(freq_mhz: np.ndarray, amplitude_jy: np.ndarray, min_cycles: float, max_cycles: float, n_grid: int) -> dict:
    finite = np.isfinite(freq_mhz) & np.isfinite(amplitude_jy)
    x = np.asarray(freq_mhz[finite], dtype=np.float64)
    y = np.asarray(amplitude_jy[finite], dtype=np.float64)
    if x.size < 8:
        raise ValueError('Too few finite channels for sinusoid fit.')

    x_span = float(np.nanmax(x) - np.nanmin(x))
    if x_span <= 0:
        raise ValueError('Frequency span is zero; cannot fit sinusoid.')

    candidate_cycles = np.linspace(float(min_cycles), float(max_cycles), int(n_grid), dtype=np.float64)
    best = None

    for cycles in candidate_cycles:
        omega = 2.0 * np.pi * cycles / x_span
        design = np.column_stack([
            np.ones_like(x),
            x,
            np.sin(omega * x),
            np.cos(omega * x),
        ])
        coeffs, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        model = design @ coeffs
        residual = y - model
        residual_sigma = robust_sigma_mad(residual)
        if best is None or residual_sigma < best['resid_sigma']:
            best = {
                'cycles': float(cycles),
                'omega': float(omega),
                'coeffs': coeffs,
                'resid_sigma': float(residual_sigma),
            }

    if best is None:
        raise RuntimeError('Sinusoid fit failed to converge.')

    c0, c1, s_coeff, c_coeff = best['coeffs']
    sinusoid_amplitude = float(np.hypot(s_coeff, c_coeff))
    sinusoid_phase_rad = float(np.arctan2(c_coeff, s_coeff))
    period_mhz = float(x_span / best['cycles'])

    full_model = c0 + c1 * freq_mhz + s_coeff * np.sin(best['omega'] * freq_mhz) + c_coeff * np.cos(best['omega'] * freq_mhz)
    full_residual = amplitude_jy - full_model

    return {
        'c0': float(c0),
        'c1': float(c1),
        'omega': float(best['omega']),
        'sin_coeff': float(s_coeff),
        'cos_coeff': float(c_coeff),
        'sinusoid_amplitude_jy': sinusoid_amplitude,
        'sinusoid_phase_rad': sinusoid_phase_rad,
        'cycles_across_band': best['cycles'],
        'period_mhz': period_mhz,
        'model': full_model,
        'residual': full_residual,
        'resid_sigma_jy': best['resid_sigma'],
        'input_sigma_jy': robust_sigma_mad(y),
    }


def fit_sinusoid_plus_linear_multicomponent(
    freq_mhz: np.ndarray,
    amplitude_jy: np.ndarray,
    min_cycles: float,
    max_cycles: float,
    n_grid: int,
    n_multi_reflection_components: int,
    min_improvement_fraction: float,
) -> dict:
    primary = fit_sinusoid_plus_linear(
        freq_mhz=freq_mhz,
        amplitude_jy=amplitude_jy,
        min_cycles=min_cycles,
        max_cycles=max_cycles,
        n_grid=n_grid,
    )

    finite = np.isfinite(freq_mhz) & np.isfinite(amplitude_jy)
    x = np.asarray(freq_mhz[finite], dtype=np.float64)
    y = np.asarray(amplitude_jy[finite], dtype=np.float64)
    x_span = float(np.nanmax(x) - np.nanmin(x))
    if x_span <= 0:
        primary['multi_reflection_components'] = []
        primary['n_multi_reflection_components_used'] = 0
        primary['ripple_component_jy'] = evaluate_sinusoid_component(freq_mhz, primary)
        return primary

    omegas = [float(primary['omega'])]
    design_cols = [
        np.ones_like(x),
        x,
        np.sin(primary['omega'] * x),
        np.cos(primary['omega'] * x),
    ]
    design = np.column_stack(design_cols)
    coeffs, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    model = design @ coeffs
    resid = y - model
    sigma = robust_sigma_mad(resid)

    candidate_cycles = np.linspace(float(min_cycles), float(max_cycles), int(n_grid), dtype=np.float64)
    multi_reflection_components: list[dict] = []

    for _ in range(int(max(0, n_multi_reflection_components))):
        best = None
        for cycles in candidate_cycles:
            omega = 2.0 * np.pi * float(cycles) / x_span
            if any(abs(omega - existing) / max(abs(existing), 1e-12) < 0.03 for existing in omegas):
                continue
            trial_design = np.column_stack([design, np.sin(omega * x), np.cos(omega * x)])
            trial_coeffs, _, _, _ = np.linalg.lstsq(trial_design, y, rcond=None)
            trial_model = trial_design @ trial_coeffs
            trial_resid = y - trial_model
            trial_sigma = robust_sigma_mad(trial_resid)
            if best is None or trial_sigma < best['sigma']:
                best = {
                    'omega': float(omega),
                    'cycles': float(cycles),
                    'sigma': float(trial_sigma),
                    'design': trial_design,
                    'coeffs': trial_coeffs,
                    'model': trial_model,
                }

        if best is None:
            break

        frac_improve = (sigma - best['sigma']) / sigma if sigma > 0 else 0.0
        if frac_improve < float(min_improvement_fraction):
            break

        design = best['design']
        coeffs = best['coeffs']
        model = best['model']
        resid = y - model
        sigma = best['sigma']
        omegas.append(best['omega'])
        # Store omega/period now; coefficients updated from final joint coeffs below
        multi_reflection_components.append({
            'omega': best['omega'],
            'cycles_across_band': best['cycles'],
            'period_mhz': float(x_span / best['cycles']) if best['cycles'] > 0 else np.nan,
            'sin_coeff': 0.0,  # placeholder
            'cos_coeff': 0.0,
        })

    # Re-extract all MR coefficients from the FINAL joint-fit coeffs.
    # Base columns: [const, linear, sin_primary, cos_primary] => MR starts at offset 4.
    for mr_idx, comp in enumerate(multi_reflection_components):
        ci = 4 + 2 * mr_idx
        comp['sin_coeff'] = float(coeffs[ci])
        comp['cos_coeff'] = float(coeffs[ci + 1])

    # Also update primary sin/cos from the final joint-fit (they shift too)
    primary['sin_coeff'] = float(coeffs[2])
    primary['cos_coeff'] = float(coeffs[3])

    full_model = np.full_like(np.asarray(freq_mhz, dtype=np.float64), np.nan, dtype=np.float64)
    full_resid = np.full_like(np.asarray(freq_mhz, dtype=np.float64), np.nan, dtype=np.float64)
    full_model[finite] = model
    full_resid[finite] = y - model

    ripple_only = np.full_like(np.asarray(freq_mhz, dtype=np.float64), np.nan, dtype=np.float64)
    ripple_only[finite] = model - (coeffs[0] + coeffs[1] * x)

    primary['c0'] = float(coeffs[0])
    primary['c1'] = float(coeffs[1])
    primary['sin_coeff'] = float(coeffs[2])
    primary['cos_coeff'] = float(coeffs[3])
    primary['sinusoid_amplitude_jy'] = float(np.hypot(coeffs[2], coeffs[3]))
    primary['sinusoid_phase_rad'] = float(np.arctan2(coeffs[3], coeffs[2]))
    primary['model'] = full_model
    primary['residual'] = full_resid
    primary['resid_sigma_jy'] = robust_sigma_mad(full_resid)
    primary['input_sigma_jy'] = robust_sigma_mad(y)
    primary['multi_reflection_components'] = multi_reflection_components
    primary['n_multi_reflection_components_used'] = len(multi_reflection_components)
    primary['ripple_component_jy'] = ripple_only
    return primary


def evaluate_sinusoid_component(freq_mhz: np.ndarray, fit_result: dict) -> np.ndarray:
    return (
        fit_result['sin_coeff'] * np.sin(fit_result['omega'] * freq_mhz)
        + fit_result['cos_coeff'] * np.cos(fit_result['omega'] * freq_mhz)
    )


def fit_linear_plus_template(freq_mhz: np.ndarray, amplitude_jy: np.ndarray, template_jy: np.ndarray) -> dict:
    finite = np.isfinite(freq_mhz) & np.isfinite(amplitude_jy) & np.isfinite(template_jy)
    x = np.asarray(freq_mhz[finite], dtype=np.float64)
    y = np.asarray(amplitude_jy[finite], dtype=np.float64)
    t = np.asarray(template_jy[finite], dtype=np.float64)
    if x.size < 8:
        raise ValueError('Too few finite channels for template-transfer fit.')

    design = np.column_stack([np.ones_like(x), x, t])
    coeffs, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    c0, c1, tscale = coeffs

    model = c0 + c1 * freq_mhz + tscale * template_jy
    ripple_component = tscale * template_jy
    residual = amplitude_jy - model

    return {
        'c0': float(c0),
        'c1': float(c1),
        'template_scale': float(tscale),
        'ripple_component_jy': ripple_component,
        'model': model,
        'residual': residual,
        'resid_sigma_jy': robust_sigma_mad(residual),
        'input_sigma_jy': robust_sigma_mad(y),
    }


def fit_linear_plus_fixed_period(freq_mhz: np.ndarray, amplitude_jy: np.ndarray, omega: float) -> dict:
    finite = np.isfinite(freq_mhz) & np.isfinite(amplitude_jy)
    x = np.asarray(freq_mhz[finite], dtype=np.float64)
    y = np.asarray(amplitude_jy[finite], dtype=np.float64)
    if x.size < 8:
        raise ValueError('Too few finite channels for fixed-period transfer fit.')

    design = np.column_stack([
        np.ones_like(x),
        x,
        np.sin(omega * x),
        np.cos(omega * x),
    ])
    coeffs, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    c0, c1, s_coeff, c_coeff = coeffs

    ripple_component = s_coeff * np.sin(omega * freq_mhz) + c_coeff * np.cos(omega * freq_mhz)
    model = c0 + c1 * freq_mhz + ripple_component
    residual = amplitude_jy - model

    return {
        'c0': float(c0),
        'c1': float(c1),
        'omega': float(omega),
        'sin_coeff': float(s_coeff),
        'cos_coeff': float(c_coeff),
        'ripple_component_jy': ripple_component,
        'model': model,
        'residual': residual,
        'resid_sigma_jy': robust_sigma_mad(residual),
        'input_sigma_jy': robust_sigma_mad(y),
    }


def _prepare_finite_arrays(*arrays: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    finite = np.ones_like(np.asarray(arrays[0], dtype=np.float64), dtype=bool)
    clean_arrays = []
    for arr in arrays:
        arr_np = np.asarray(arr, dtype=np.float64)
        finite &= np.isfinite(arr_np)
        clean_arrays.append(arr_np)
    return finite, [arr[finite] for arr in clean_arrays]


def _solve_linear_model(design: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coeffs, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    model = design @ coeffs
    return coeffs, model


def fit_transfer_with_multi_reflection_ripples(
    freq_mhz: np.ndarray,
    amplitude_jy: np.ndarray,
    transfer_mode: str,
    template_ripple_jy: np.ndarray | None,
    template_omega: float | None,
    n_multi_reflection_components: int,
    multi_reflection_min_cycles: float,
    multi_reflection_max_cycles: float,
    multi_reflection_grid_size: int,
    min_improvement_fraction: float,
    template_components: list[dict] | None = None,
) -> dict:
    """Fit a transfer ripple model to a target spectrum.

    transfer_mode options:
      'fixed_template'   : fit a single scale factor for the whole composite template waveform
      'fixed_period'     : fit independent amplitude+phase at the template primary period only
      'free_phase_template': fit independent amplitude+phase for EACH template frequency component
                             (primary + all multi-reflection). Correct when reflection path length
                             differs between template and target sources (phase offset).
    """
    finite, (x, y) = _prepare_finite_arrays(freq_mhz, amplitude_jy)
    if x.size < 8:
        raise ValueError('Too few finite channels for transfer fit.')

    x_span = float(np.nanmax(x) - np.nanmin(x))
    if x_span <= 0:
        raise ValueError('Frequency span is zero; cannot fit transfer ripple.')

    base_columns = [np.ones_like(x), x]
    base_labels = ['const', 'linear']
    # For free_phase_template: the omegas of each individually-fitted template component
    free_phase_omegas: list[float] = []

    if transfer_mode == 'fixed_template':
        if template_ripple_jy is None:
            raise ValueError('template_ripple_jy is required for fixed_template mode.')
        t_full = np.asarray(template_ripple_jy, dtype=np.float64)
        t = t_full[finite]
        base_columns.append(t)
        base_labels.append('template')
    elif transfer_mode == 'free_phase_template':
        if not template_components:
            raise ValueError('template_components list is required for free_phase_template mode.')
        for comp in template_components:
            omega_c = float(comp['omega'])
            base_columns.append(np.sin(omega_c * x))
            base_columns.append(np.cos(omega_c * x))
            base_labels.extend([f'tmpl_sin_{omega_c:.5f}', f'tmpl_cos_{omega_c:.5f}'])
            free_phase_omegas.append(omega_c)
    elif transfer_mode == 'fixed_period':
        if template_omega is None:
            raise ValueError('template_omega is required for fixed_period mode.')
        base_columns.append(np.sin(template_omega * x))
        base_columns.append(np.cos(template_omega * x))
        base_labels.extend(['template_sin', 'template_cos'])
    else:
        raise ValueError(f'Unsupported transfer_mode={transfer_mode}')

    design = np.column_stack(base_columns)
    coeffs, model = _solve_linear_model(design, y)
    resid = y - model
    sigma = robust_sigma_mad(resid)

    used_omegas: list[float] = list(free_phase_omegas)
    if transfer_mode == 'fixed_period' and template_omega is not None:
        used_omegas.append(float(template_omega))

    multi_reflection_components: list[dict] = []
    candidate_cycles = np.linspace(float(multi_reflection_min_cycles), float(multi_reflection_max_cycles), int(multi_reflection_grid_size), dtype=np.float64)

    for _ in range(int(max(0, n_multi_reflection_components))):
        best = None
        for cycles in candidate_cycles:
            omega = 2.0 * np.pi * float(cycles) / x_span
            if any(abs(omega - u) / max(abs(u), 1e-12) < 0.03 for u in used_omegas):
                continue

            trial_design = np.column_stack([design, np.sin(omega * x), np.cos(omega * x)])
            trial_coeffs, trial_model = _solve_linear_model(trial_design, y)
            trial_resid = y - trial_model
            trial_sigma = robust_sigma_mad(trial_resid)
            if best is None or trial_sigma < best['sigma']:
                best = {
                    'omega': float(omega),
                    'cycles': float(cycles),
                    'sigma': float(trial_sigma),
                    'design': trial_design,
                    'coeffs': trial_coeffs,
                    'model': trial_model,
                }

        if best is None:
            break

        fractional_improvement = (sigma - best['sigma']) / sigma if sigma > 0 else 0.0
        if fractional_improvement < float(min_improvement_fraction):
            break

        design = best['design']
        coeffs = best['coeffs']
        model = best['model']
        resid = y - model
        sigma = best['sigma']
        used_omegas.append(best['omega'])
        base_labels.extend(['multi_reflection_sin', 'multi_reflection_cos'])
        # Store omega/period now; coefficients will be updated after the loop
        multi_reflection_components.append({
            'omega': best['omega'],
            'cycles_across_band': best['cycles'],
            'period_mhz': float(x_span / best['cycles']) if best['cycles'] > 0 else np.nan,
            'sin_coeff': 0.0,  # placeholder — updated below from final joint coeffs
            'cos_coeff': 0.0,
        })

    # Re-extract all MR coefficients from the FINAL joint-fit coeffs.
    # The design matrix columns after the base columns are: [mr_sin_0, mr_cos_0, mr_sin_1, mr_cos_1, ...]
    # For free_phase_template: base = 2 + 2*n_tmpl_comps; fixed_template: base = 3; fixed_period: base = 4
    if transfer_mode == 'free_phase_template':
        n_tmpl_comps = len(template_components) if template_components else 0
        mr_coeff_offset = 2 + 2 * n_tmpl_comps
    elif transfer_mode == 'fixed_template':
        mr_coeff_offset = 3
    else:  # fixed_period
        mr_coeff_offset = 4
    for mr_idx, comp in enumerate(multi_reflection_components):
        ci = mr_coeff_offset + 2 * mr_idx
        comp['sin_coeff'] = float(coeffs[ci])
        comp['cos_coeff'] = float(coeffs[ci + 1])

    full_model = np.full_like(np.asarray(freq_mhz, dtype=np.float64), np.nan, dtype=np.float64)
    full_residual = np.full_like(np.asarray(freq_mhz, dtype=np.float64), np.nan, dtype=np.float64)
    full_model[finite] = model
    full_residual[finite] = y - model

    result = {
        'model': full_model,
        'residual': full_residual,
        'resid_sigma_jy': robust_sigma_mad(full_residual),
        'input_sigma_jy': robust_sigma_mad(y),
        'multi_reflection_components': multi_reflection_components,
        'n_multi_reflection_components_used': len(multi_reflection_components),
    }

    result['c0'] = float(coeffs[0])
    result['c1'] = float(coeffs[1])

    if transfer_mode == 'fixed_template':
        result['template_scale'] = float(coeffs[2])
        template_component_full = float(coeffs[2]) * np.asarray(template_ripple_jy, dtype=np.float64)
        result['template_component_jy'] = template_component_full
    elif transfer_mode == 'free_phase_template':
        # Decode per-component coefficients (2 per template component, starting at index 2)
        fitted_template_components: list[dict] = []
        for idx, comp in enumerate(template_components or []):
            s_c = float(coeffs[2 + 2 * idx])
            c_c = float(coeffs[2 + 2 * idx + 1])
            fitted_template_components.append({
                'omega': float(comp['omega']),
                'period_mhz': float(comp.get('period_mhz', (2.0 * np.pi / float(comp['omega'])) if float(comp['omega']) != 0 else np.nan)),
                'sin_coeff': s_c,
                'cos_coeff': c_c,
                'amplitude_jy': float(np.hypot(s_c, c_c)),
            })
        result['template_components_fitted'] = fitted_template_components
    else:
        result['omega'] = float(template_omega)
        result['sin_coeff'] = float(coeffs[2])
        result['cos_coeff'] = float(coeffs[3])
        period = float((2.0 * np.pi) / template_omega) if template_omega not in (None, 0.0) else np.nan
        result['period_mhz'] = period

    return result


def summarize_product(product: str, fit_result: dict) -> str:
    input_sigma_mjy = 1e3 * fit_result['input_sigma_jy']
    residual_sigma_mjy = 1e3 * fit_result['resid_sigma_jy']
    sinusoid_amp_mjy = 1e3 * fit_result['sinusoid_amplitude_jy']
    improvement = fit_result['input_sigma_jy'] / fit_result['resid_sigma_jy'] if fit_result['resid_sigma_jy'] > 0 else np.nan
    return (
        f'{product}: input_robust_sigma={input_sigma_mjy:.3f} mJy, '
        f'sinusoid_amp={sinusoid_amp_mjy:.3f} mJy, '
        f'period={fit_result["period_mhz"]:.3f} MHz, '
        f'residual_robust_sigma={residual_sigma_mjy:.3f} mJy, '
        f'improvement={improvement:.2f}x'
    )


def main() -> int:
    parser = argparse.ArgumentParser(description='Fit RR/LL baseline-average spectral sinusoid and estimate residual robust noise.')
    cfg = parser.add_argument_group('Config and Inputs')
    cfg.add_argument('--config', default=None)
    cfg.add_argument('--fits', default=None)
    cfg.add_argument('--index-cache', type=str, default=None)
    cfg.add_argument('--index-validation', type=str, default='fast')
    cfg.add_argument('--set', action='append', default=[], help='Override config/global key: --set "KEY=expr"')

    sel = parser.add_argument_group('Selection')
    sel.add_argument('--source', type=str, default=None)
    sel.add_argument('--time-range', nargs=2, default=None, metavar=('START', 'END'))
    sel.add_argument('--chan-range', nargs=2, type=int, metavar=('START', 'END'), default=None)
    sel.add_argument('--antennas', type=str, default=None)
    sel.add_argument('--baselines', type=str, default=None)
    sel.add_argument('--elevation-min', type=float, default=None)
    sel.add_argument('--elevation-max', type=float, default=None)
    sel.add_argument('--uvrange-m', nargs=2, type=float, default=None, metavar=('MIN_M', 'MAX_M'))
    sel.add_argument('--uvrange-klambda', nargs=2, type=float, default=None, metavar=('MIN_KL', 'MAX_KL'))

    cal = parser.add_argument_group('Flagging and Gain Application')
    cal.add_argument('--bpcal', nargs='+', default=None)
    cal.add_argument('--flag', default=None)
    cal.add_argument('--time-interp-scheme', choices=ca.TIME_INTERP_SCHEMES, default='nearest')
    cal.add_argument('--time-extrapolation', choices=['hold', 'nearest', 'none'], default='hold')
    cal.add_argument('--exclude-for-plots', action='append', default=[])

    fit_group = parser.add_argument_group('Sinusoid Fit')
    fit_group.add_argument('--products', type=str, default='RR,LL', help='Products to fit (default RR,LL)')
    fit_group.add_argument('--fit-min-cycles', type=float, default=0.5)
    fit_group.add_argument('--fit-max-cycles', type=float, default=25.0)
    fit_group.add_argument('--fit-grid-size', type=int, default=3000)
    fit_group.add_argument('--fit-multi-reflection-components', type=int, default=0)
    fit_group.add_argument('--fit-multi-reflection-min-improvement-fraction', type=float, default=0.001)
    fit_group.add_argument('--channel-outlier-clip-sigma', type=float, default=0.0, help='Sigma threshold for auto-masking narrow channel outliers before fitting (0 disables)')
    fit_group.add_argument('--channel-outlier-half-window', type=int, default=2, help='Half-window size (channels) for running-median trend used in outlier masking')
    fit_group.add_argument('--channel-outlier-dilate', type=int, default=1, help='Dilate channel outlier mask by this many neighboring channels on each side')

    transfer = parser.add_argument_group('Template Transfer (optional)')
    transfer.add_argument('--template-source', type=str, default=None, help='Template source to fit ripple from (e.g. 3C48)')
    transfer.add_argument('--target-source', type=str, default=None, help='Target source to correct with template ripple (e.g. 3C468.1)')
    transfer.add_argument('--transfer-mode', choices=['fixed_template', 'free_phase_template', 'fixed_period'], default='free_phase_template')
    transfer.add_argument('--template-fit-multi-reflection-components', type=int, default=2)
    transfer.add_argument('--template-fit-multi-reflection-min-improvement-fraction', type=float, default=0.001)
    transfer.add_argument(
        '--derive-additional-ripples-on-target',
        type=int,
        default=2,
        help='TRANSFER MODE ONLY: number of additional target-only sinusoid components to derive (beyond transferred primary/template components)',
    )
    transfer.add_argument('--target-extra-multi-reflection-min-cycles', type=float, default=12.0)
    transfer.add_argument('--target-extra-multi-reflection-max-cycles', type=float, default=80.0)
    transfer.add_argument('--target-extra-multi-reflection-grid-size', type=int, default=1800)
    transfer.add_argument('--target-extra-multi-reflection-min-improvement-fraction', type=float, default=0.002)
    transfer.add_argument('--transfer-residual-clip-sigma', type=float, default=0.0, help='Sigma threshold for iterative residual-based channel masking on target (0 disables)')
    transfer.add_argument('--transfer-residual-clip-max-iters', type=int, default=2, help='Maximum residual-clip refit iterations on target')

    out = parser.add_argument_group('Output')
    out.add_argument('--outdir', default='./diagnostics_out')
    out.add_argument('--outfile-prefix', default=None)

    args = parser.parse_args()

    if args.config:
        load_config_into_globals(args.config, globals())
    apply_overrides_to_globals(args.set, globals())

    fits_path = args.fits if args.fits is not None else globals().get('CAL_FITS', None)
    if fits_path is None:
        raise SystemExit('ERROR: no input vis provided. Pass --fits or provide CAL_FITS via --config.')

    index_cache = args.index_cache if args.index_cache is not None else globals().get('INDEX_CACHE', None)
    source_name_cli = args.source if args.source is not None else globals().get('SOURCE', None)
    cfg_chan_range = globals().get('CHAN_RANGE', None)
    chan_range = tuple(args.chan_range) if args.chan_range else (tuple(cfg_chan_range) if cfg_chan_range is not None else None)
    timerange = tuple(args.time_range) if args.time_range else globals().get('SOLVE_TIMERANGE', None)
    elevation_min = args.elevation_min if args.elevation_min is not None else globals().get('SOLVE_ELEVATION_MIN_DEG', None)
    elevation_max = args.elevation_max if args.elevation_max is not None else globals().get('SOLVE_ELEVATION_MAX_DEG', None)
    uvrange_m = tuple(args.uvrange_m) if args.uvrange_m else globals().get('SOLVE_UVRANGE_M', None)
    uvrange_klambda = tuple(args.uvrange_klambda) if args.uvrange_klambda else globals().get('SOLVE_UVRANGE_KLAMBDA', None)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg_exclude = globals().get('EXCLUDE_FOR_PLOTS', [])
    if cfg_exclude is None:
        cfg_exclude_list = []
    elif isinstance(cfg_exclude, (list, tuple)):
        cfg_exclude_list = list(cfg_exclude)
    else:
        cfg_exclude_list = [cfg_exclude]
    exclude_for_plots = cfg_exclude_list + list(args.exclude_for_plots or [])

    index = q.get_or_build_row_index(
        fits_path,
        cache_path=index_cache,
        force_rebuild=False,
        validation_mode=args.index_validation,
        write_cache=True,
        override_dud_names=None,
    )
    ant_name_map, name_to_id = pv._build_ant_maps(index)

    source_name = source_name_cli
    if not source_name:
        if index.get('id_to_name'):
            source_name = sorted(index['id_to_name'].values())[0]
        else:
            raise RuntimeError('No source found in index and no --source provided.')

    transfer_mode_active = bool(args.template_source) or bool(args.target_source)
    if transfer_mode_active and not (args.template_source and args.target_source):
        raise SystemExit('ERROR: --template-source and --target-source must be provided together.')

    analysis_source_name = source_name if not transfer_mode_active else args.target_source
    template_source_name = args.template_source if transfer_mode_active else None
    target_source_name = args.target_source if transfer_mode_active else source_name

    ant_list = []
    for token in pv._parse_csv_list(args.antennas):
        ant_id = pv._resolve_ant_selector(token, ant_name_map, name_to_id)
        ant_list.append(ant_id if ant_id is not None else token)

    baseline_pairs = pv._parse_baselines(args.baselines, ant_name_map, name_to_id) if args.baselines else None

    solutions = []
    if args.bpcal:
        for bandpass_path in args.bpcal:
            solution = pv._load_solution(bandpass_path, analysis_source_name, index)
            solution['_path'] = str(bandpass_path)
            solutions.append(solution)

    requested_products = [product.upper() for product in pv._parse_csv_list(args.products)]
    if not requested_products:
        requested_products = ['RR', 'LL']

    def load_source_products(source_to_load: str, source_solutions: list[dict] | None) -> tuple[np.ndarray, dict]:
        vis = q.load_vis_for_source(
            index,
            source=source_to_load,
            stokes=('RR', 'LL'),
            ant_list=ant_list if ant_list else None,
            chan_range=chan_range,
            timerange=timerange,
            uvrange_m=uvrange_m,
            uvrange_klambda=uvrange_klambda,
            elevation_min_deg=elevation_min,
            elevation_max_deg=elevation_max,
            flag_all_corrs_if_any_rawvis_flagged=True,
        )

        if args.flag:
            vis_with_flags, _ = q.apply_flag_tables_to_vis(vis, antenna_name_map=ant_name_map, flag_table_paths=[args.flag])
            vis = vis_with_flags

        if baseline_pairs:
            vis = pv._slice_rows(vis, pv._row_filter_by_baselines(vis, baseline_pairs))

        if source_solutions and exclude_for_plots:
            vis = q._filter_vis_excluded_antennas(vis, source_solutions[0], exclude_antennas=exclude_for_plots)

        corrected_cache = pv._apply_solutions_by_time(
            vis,
            source_solutions,
            time_interp_scheme=args.time_interp_scheme,
            time_extrapolation=args.time_extrapolation,
        ) if source_solutions else None

        freqs_hz = np.asarray(vis['freqs_hz'], dtype=np.float64)
        local_freq_mhz = freqs_hz / 1e6

        source_products = {}
        for product in requested_products:
            source_products[product] = baseline_vector_average(vis, corrected_cache, product)
        return local_freq_mhz, source_products

    transfer_results = {}
    fit_results = {}
    template_fit_results = {}
    channel_mask_by_product: dict[str, np.ndarray] = {}

    if transfer_mode_active:
        template_solutions = solutions[:1] if solutions else []
        target_solutions = solutions

        template_freq_mhz, template_products = load_source_products(template_source_name, template_solutions)
        target_freq_mhz, target_products = load_source_products(target_source_name, target_solutions)

        if template_freq_mhz.shape != target_freq_mhz.shape or not np.allclose(template_freq_mhz, target_freq_mhz, atol=1e-9, rtol=0.0):
            raise RuntimeError('Template and target channel grids differ; use identical channel selection for transfer.')

        freq_mhz = target_freq_mhz
        for product in requested_products:
            template_amp = template_products[product]['scalar_amp']
            template_flux_model_jy = evaluate_source_flux_model_jy(freq_mhz * 1e6, template_source_name)
            template_pb_subtracted_amp = template_amp - template_flux_model_jy
            target_amp = target_products[product]['scalar_amp']

            if args.channel_outlier_clip_sigma > 0:
                template_bad = detect_channel_outliers(
                    template_pb_subtracted_amp,
                    clip_sigma=args.channel_outlier_clip_sigma,
                    half_window=args.channel_outlier_half_window,
                )
                target_bad = detect_channel_outliers(
                    target_amp,
                    clip_sigma=args.channel_outlier_clip_sigma,
                    half_window=args.channel_outlier_half_window,
                )
                channel_mask = dilate_channel_mask(
                    template_bad | target_bad,
                    radius=args.channel_outlier_dilate,
                )
            else:
                channel_mask = np.zeros_like(freq_mhz, dtype=bool)

            template_pb_subtracted_for_fit = np.where(channel_mask, np.nan, template_pb_subtracted_amp)
            target_amp_for_fit = np.where(channel_mask, np.nan, target_amp)
            channel_mask_by_product[product] = channel_mask

            template_fit = fit_sinusoid_plus_linear_multicomponent(
                freq_mhz=freq_mhz,
                amplitude_jy=template_pb_subtracted_for_fit,
                min_cycles=args.fit_min_cycles,
                max_cycles=args.fit_max_cycles,
                n_grid=args.fit_grid_size,
                n_multi_reflection_components=args.template_fit_multi_reflection_components,
                min_improvement_fraction=args.template_fit_multi_reflection_min_improvement_fraction,
            )
            template_fit_results[product] = {
                'avg': template_products[product],
                'pb_model_jy': template_flux_model_jy,
                'pb_subtracted_amp_jy': template_pb_subtracted_amp,
                'pb_subtracted_fit_input_jy': template_pb_subtracted_for_fit,
                'fit': template_fit,
            }

            template_fit['model'] = evaluate_multi_reflection_fit(freq_mhz, template_fit)
            template_fit['residual'] = template_pb_subtracted_amp - template_fit['model']
            template_fit['ripple_component_jy'] = template_fit['model'] - (template_fit['c0'] + template_fit['c1'] * freq_mhz)

            template_ripple = template_fit.get('ripple_component_jy', evaluate_sinusoid_component(freq_mhz, template_fit))

            # Build a flat list of all template frequency components for free_phase_template mode
            template_primary_component = {
                'omega': template_fit['omega'],
                'period_mhz': template_fit['period_mhz'],
                'sin_coeff': template_fit['sin_coeff'],
                'cos_coeff': template_fit['cos_coeff'],
            }
            template_all_components = [template_primary_component] + list(template_fit.get('multi_reflection_components', []))

            transfer_fit = fit_transfer_with_multi_reflection_ripples(
                freq_mhz=freq_mhz,
                amplitude_jy=target_amp_for_fit,
                transfer_mode=args.transfer_mode,
                template_ripple_jy=template_ripple,
                template_omega=template_fit['omega'],
                template_components=template_all_components,
                n_multi_reflection_components=args.derive_additional_ripples_on_target,
                multi_reflection_min_cycles=args.target_extra_multi_reflection_min_cycles,
                multi_reflection_max_cycles=args.target_extra_multi_reflection_max_cycles,
                multi_reflection_grid_size=args.target_extra_multi_reflection_grid_size,
                min_improvement_fraction=args.target_extra_multi_reflection_min_improvement_fraction,
            )

            residual_clip_iterations = 0
            if args.transfer_residual_clip_sigma > 0:
                max_iters = int(max(0, args.transfer_residual_clip_max_iters))
                for _ in range(max_iters):
                    target_residual_unmasked = target_amp - transfer_fit['model']
                    residual_bad = detect_residual_outliers(
                        target_residual_unmasked,
                        clip_sigma=args.transfer_residual_clip_sigma,
                    )
                    residual_bad = dilate_channel_mask(residual_bad, radius=args.channel_outlier_dilate)
                    new_channel_mask = channel_mask | residual_bad
                    if int(np.count_nonzero(new_channel_mask)) <= int(np.count_nonzero(channel_mask)):
                        break
                    channel_mask = new_channel_mask
                    channel_mask_by_product[product] = channel_mask
                    target_amp_for_fit = np.where(channel_mask, np.nan, target_amp)
                    transfer_fit = fit_transfer_with_multi_reflection_ripples(
                        freq_mhz=freq_mhz,
                        amplitude_jy=target_amp_for_fit,
                        transfer_mode=args.transfer_mode,
                        template_ripple_jy=template_ripple,
                        template_omega=template_fit['omega'],
                        template_components=template_all_components,
                        n_multi_reflection_components=args.derive_additional_ripples_on_target,
                        multi_reflection_min_cycles=args.target_extra_multi_reflection_min_cycles,
                        multi_reflection_max_cycles=args.target_extra_multi_reflection_max_cycles,
                        multi_reflection_grid_size=args.target_extra_multi_reflection_grid_size,
                        min_improvement_fraction=args.target_extra_multi_reflection_min_improvement_fraction,
                    )
                    residual_clip_iterations += 1

            transfer_fit['model'] = evaluate_transfer_fit_model(
                freq_mhz,
                transfer_fit,
                transfer_mode=args.transfer_mode,
                template_ripple_jy=template_ripple,
            )
            transfer_fit['residual'] = target_amp - transfer_fit['model']

            transfer_results[product] = {
                'target_avg': target_products[product],
                'template_ripple_jy': template_ripple,
                'target_fit_input_jy': target_amp_for_fit,
                'residual_clip_iterations': residual_clip_iterations,
                'transfer_fit': transfer_fit,
            }
    else:
        freq_mhz, source_products = load_source_products(source_name, solutions)
        for product in requested_products:
            vector_summary = source_products[product]
            if args.channel_outlier_clip_sigma > 0:
                channel_mask = dilate_channel_mask(
                    detect_channel_outliers(
                        vector_summary['scalar_amp'],
                        clip_sigma=args.channel_outlier_clip_sigma,
                        half_window=args.channel_outlier_half_window,
                    ),
                    radius=args.channel_outlier_dilate,
                )
            else:
                channel_mask = np.zeros_like(freq_mhz, dtype=bool)
            amp_for_fit = np.where(channel_mask, np.nan, vector_summary['scalar_amp'])
            channel_mask_by_product[product] = channel_mask
            fit_results[product] = {
                'avg': vector_summary,
                'fit_input_jy': amp_for_fit,
                'fit': fit_sinusoid_plus_linear_multicomponent(
                    freq_mhz=freq_mhz,
                    amplitude_jy=amp_for_fit,
                    min_cycles=args.fit_min_cycles,
                    max_cycles=args.fit_max_cycles,
                    n_grid=args.fit_grid_size,
                    n_multi_reflection_components=args.fit_multi_reflection_components,
                    min_improvement_fraction=args.fit_multi_reflection_min_improvement_fraction,
                ),
            }
            fit_results[product]['fit']['model'] = evaluate_multi_reflection_fit(freq_mhz, fit_results[product]['fit'])
            fit_results[product]['fit']['residual'] = vector_summary['scalar_amp'] - fit_results[product]['fit']['model']
            fit_results[product]['fit']['ripple_component_jy'] = fit_results[product]['fit']['model'] - (
                fit_results[product]['fit']['c0'] + fit_results[product]['fit']['c1'] * freq_mhz
            )

    if transfer_mode_active:
        prefix = args.outfile_prefix if args.outfile_prefix else (
            f'ripple_transfer_{template_source_name.strip().lower().replace(" ", "_")}'
            f'_to_{target_source_name.strip().lower().replace(" ", "_")}'
        )
    else:
        prefix = args.outfile_prefix if args.outfile_prefix else f'sinusoid_fit_{source_name.strip().lower().replace(" ", "_")}'
    png_path = outdir / f'{prefix}.png'
    txt_path = outdir / f'{prefix}.txt'
    npz_path = outdir / f'{prefix}.npz'

    nrows = max(1, len(requested_products))
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(12, 3.8 * nrows), dpi=160, squeeze=False)

    if transfer_mode_active:
        template_table_names = [Path(str(s.get('_path', '<unknown>'))).name for s in (solutions[:1] if solutions else [])]
        target_table_names = [Path(str(s.get('_path', '<unknown>'))).name for s in (solutions if solutions else [])]
        text_lines = [
            f'template_source={template_source_name}',
            f'target_source={target_source_name}',
            f'transfer_mode={args.transfer_mode}',
            f'template_bpcal_tables={template_table_names}',
            f'target_bpcal_tables={target_table_names}',
            f'fits={fits_path}',
            f'products={requested_products}',
            f'channel_outlier_clip_sigma={args.channel_outlier_clip_sigma}',
            f'channel_outlier_half_window={args.channel_outlier_half_window}',
            f'channel_outlier_dilate={args.channel_outlier_dilate}',
            f'transfer_residual_clip_sigma={args.transfer_residual_clip_sigma}',
            f'transfer_residual_clip_max_iters={args.transfer_residual_clip_max_iters}',
            f'derive_additional_ripples_on_target={args.derive_additional_ripples_on_target}',
            '',
        ]
    else:
        text_lines = [
            f'source={source_name}',
            f'fits={fits_path}',
            f'products={requested_products}',
            f'channel_outlier_clip_sigma={args.channel_outlier_clip_sigma}',
            f'channel_outlier_half_window={args.channel_outlier_half_window}',
            f'channel_outlier_dilate={args.channel_outlier_dilate}',
            f'derive_additional_ripples_on_target_ignored={args.derive_additional_ripples_on_target}',
            '',
        ]

    save_payload = {'freq_mhz': freq_mhz}
    for row_index, product in enumerate(requested_products):
        channel_mask = np.asarray(channel_mask_by_product.get(product, np.zeros_like(freq_mhz, dtype=bool)), dtype=bool)
        masked_channel_indices = np.flatnonzero(channel_mask)
        masked_channel_count = int(masked_channel_indices.size)
        if transfer_mode_active:
            template_fit = template_fit_results[product]['fit']
            target_avg = transfer_results[product]['target_avg']
            transfer_fit = transfer_results[product]['transfer_fit']
            template_ripple = transfer_results[product]['template_ripple_jy']

            spectrum = target_avg['scalar_amp']
            model = transfer_fit['model']
            residual = transfer_fit['residual']
        else:
            avg = fit_results[product]['avg']
            fit = fit_results[product]['fit']
            spectrum = avg['scalar_amp']
            model = fit['model']
            residual = fit['residual']

        ax0 = axes[row_index, 0]
        ax1 = axes[row_index, 1]

        ax0.plot(freq_mhz, 1e3 * spectrum, lw=1.2, color='tab:blue', label=f'{product} baseline-avg |V|')
        if transfer_mode_active:
            ax0.plot(freq_mhz, 1e3 * model, lw=1.1, color='tab:orange', label='transfer correction model')
            ax0.plot(freq_mhz, 1e3 * template_ripple, lw=1.0, color='tab:purple', alpha=0.8, label='template ripple (3C48)')
        else:
            ax0.plot(freq_mhz, 1e3 * model, lw=1.1, color='tab:orange', label='sinusoid+linear fit')
        if masked_channel_count > 0:
            ax0.scatter(
                freq_mhz[channel_mask],
                1e3 * spectrum[channel_mask],
                s=22,
                facecolors='none',
                edgecolors='magenta',
                linewidths=1.2,
                zorder=5,
                label='masked channels',
            )
        ax0.set_xlabel('Frequency (MHz)')
        ax0.set_ylabel('Amplitude (mJy)')
        if transfer_mode_active:
            ax0.set_title(f'{product}: target spectrum with transferred ripple model')
        else:
            ax0.set_title(f'{product}: spectrum and fitted sinusoid')
        ax0.grid(True, alpha=0.25)
        ax0.legend(loc='best', fontsize=8)

        residual_line = np.where(channel_mask, np.nan, residual)
        ax1.plot(freq_mhz, 1e3 * residual_line, lw=1.0, color='tab:green')
        if masked_channel_count > 0:
            ax1.scatter(
                freq_mhz[channel_mask],
                1e3 * residual[channel_mask],
                s=24,
                color='magenta',
                edgecolors='white',
                linewidths=0.4,
                zorder=6,
                label='masked residual channels',
            )
        ax1.axhline(0.0, color='k', lw=0.8, alpha=0.6)
        ax1.set_xlabel('Frequency (MHz)')
        ax1.set_ylabel('Residual (mJy)')
        if transfer_mode_active:
            ax1.set_title(f'{product}: residual (unmasked robust σ={1e3 * transfer_fit["resid_sigma_jy"]:.3f} mJy)')
        else:
            ax1.set_title(f'{product}: residual (unmasked robust σ={1e3 * fit["resid_sigma_jy"]:.3f} mJy)')
        ax1.grid(True, alpha=0.25)
        if masked_channel_count > 0:
            ax1.legend(loc='best', fontsize=8)
            ax1.text(
                0.02,
                0.98,
                f'magenta=masked\nexcluded from fit/σ\ncount={masked_channel_count}',
                transform=ax1.transAxes,
                va='top',
                ha='left',
                fontsize=8,
                bbox=dict(facecolor='white', alpha=0.75, edgecolor='0.8'),
            )

        if transfer_mode_active:
            input_sigma_mjy = 1e3 * transfer_fit['input_sigma_jy']
            residual_sigma_mjy = 1e3 * transfer_fit['resid_sigma_jy']
            improvement = transfer_fit['input_sigma_jy'] / transfer_fit['resid_sigma_jy'] if transfer_fit['resid_sigma_jy'] > 0 else np.nan

            # Build unified list of all fitted sinusoidal components with their amplitudes on the target,
            # sorted by descending amplitude (highest power first).
            all_fitted_components: list[tuple[float, float]] = []  # (period_mhz, amplitude_jy)
            if args.transfer_mode == 'free_phase_template':
                for comp in transfer_fit.get('template_components_fitted', []):
                    all_fitted_components.append((float(comp['period_mhz']), float(comp['amplitude_jy'])))
            elif args.transfer_mode == 'fixed_template':
                tscale = abs(float(transfer_fit['template_scale']))
                # Primary template component amplitude as seen on target
                tmpl_primary_amp = tscale * float(np.hypot(template_fit['sin_coeff'], template_fit['cos_coeff']))
                all_fitted_components.append((float(template_fit['period_mhz']), tmpl_primary_amp))
                for comp in template_fit.get('multi_reflection_components', []):
                    all_fitted_components.append((float(comp['period_mhz']), tscale * float(np.hypot(comp['sin_coeff'], comp['cos_coeff']))))
            else:  # fixed_period
                all_fitted_components.append((float(transfer_fit['period_mhz']), float(np.hypot(transfer_fit['sin_coeff'], transfer_fit['cos_coeff']))))
            # Append any additional target-specific multi-reflection components
            for comp in transfer_fit.get('multi_reflection_components', []):
                all_fitted_components.append((float(comp['period_mhz']), float(np.hypot(comp['sin_coeff'], comp['cos_coeff']))))
            all_fitted_components.sort(key=lambda t: t[1], reverse=True)
            periods_text = ','.join(f'{p:.3f}({1e3*a:.1f}mJy)' for p, a in all_fitted_components)

            summary_line = (
                f'{product}: transfer_mode={args.transfer_mode}, '
                f'input_robust_sigma={input_sigma_mjy:.3f} mJy, '
                f'template_flux_model=Perley-Butler-2017, '
                f'periods_mhz(amp)=[{periods_text}], '
                f'masked_channels={masked_channel_count}, '
                f'residual_clip_iters={transfer_results[product]["residual_clip_iterations"]}, '
                f'residual_robust_sigma={residual_sigma_mjy:.3f} mJy, '
                f'improvement={improvement:.2f}x'
            )
        else:
            summary_line = summarize_product(product, fit)
            summary_line = f'{summary_line}, masked_channels={masked_channel_count}'
        text_lines.append(summary_line)

        save_payload[f'{product}_scalar_amp_jy'] = spectrum
        if transfer_mode_active:
            save_payload[f'{product}_transfer_model_jy'] = model
            save_payload[f'{product}_transfer_residual_jy'] = residual
            save_payload[f'{product}_template_ripple_jy'] = template_ripple
            save_payload[f'{product}_channel_mask'] = channel_mask.astype(np.uint8)
            save_payload[f'{product}_masked_channel_indices'] = masked_channel_indices.astype(np.int32)
            save_payload[f'{product}_template_period_mhz'] = np.array([template_fit['period_mhz']])
            template_multi_reflection_periods = [component['period_mhz'] for component in template_fit.get('multi_reflection_components', [])]
            save_payload[f'{product}_template_multi_reflection_periods_mhz'] = np.asarray(template_multi_reflection_periods, dtype=np.float64)
            save_payload[f'{product}_template_model_jy'] = np.asarray(template_fit.get('model'), dtype=np.float64)
            save_payload[f'{product}_template_scalar_amp_jy'] = np.asarray(template_fit_results[product]['avg']['scalar_amp'], dtype=np.float64)
            save_payload[f'{product}_template_flux_model_jy'] = np.asarray(template_fit_results[product]['pb_model_jy'], dtype=np.float64)
            save_payload[f'{product}_template_pb_subtracted_amp_jy'] = np.asarray(template_fit_results[product]['pb_subtracted_amp_jy'], dtype=np.float64)
            save_payload[f'{product}_template_pb_subtracted_fit_input_jy'] = np.asarray(template_fit_results[product]['pb_subtracted_fit_input_jy'], dtype=np.float64)
            save_payload[f'{product}_target_fit_input_jy'] = np.asarray(transfer_results[product]['target_fit_input_jy'], dtype=np.float64)
            save_payload[f'{product}_transfer_residual_clip_iterations'] = np.array([transfer_results[product]['residual_clip_iterations']], dtype=np.int32)
            save_payload[f'{product}_target_extra_n_multi_reflection_components'] = np.array([transfer_fit.get('n_multi_reflection_components_used', 0)])
            save_payload[f'{product}_transfer_resid_sigma_jy'] = np.array([transfer_fit['resid_sigma_jy']])
            save_payload[f'{product}_transfer_input_sigma_jy'] = np.array([transfer_fit['input_sigma_jy']])
            multi_reflection_periods = [component['period_mhz'] for component in transfer_fit.get('multi_reflection_components', [])]
            save_payload[f'{product}_target_extra_multi_reflection_periods_mhz'] = np.asarray(multi_reflection_periods, dtype=np.float64)
            if args.transfer_mode == 'fixed_template':
                save_payload[f'{product}_template_scale'] = np.array([transfer_fit['template_scale']])
            elif args.transfer_mode == 'free_phase_template':
                # Save per-frequency fitted components: periods and amplitudes on target
                fitted_comps = transfer_fit.get('template_components_fitted', [])
                save_payload[f'{product}_free_phase_periods_mhz'] = np.asarray([c['period_mhz'] for c in fitted_comps], dtype=np.float64)
                save_payload[f'{product}_free_phase_amplitudes_jy'] = np.asarray([c['amplitude_jy'] for c in fitted_comps], dtype=np.float64)
            else:
                save_payload[f'{product}_transfer_sin_coeff'] = np.array([transfer_fit['sin_coeff']])
                save_payload[f'{product}_transfer_cos_coeff'] = np.array([transfer_fit['cos_coeff']])
        else:
            save_payload[f'{product}_fit_model_jy'] = model
            save_payload[f'{product}_fit_residual_jy'] = residual
            save_payload[f'{product}_channel_mask'] = channel_mask.astype(np.uint8)
            save_payload[f'{product}_masked_channel_indices'] = masked_channel_indices.astype(np.int32)
            save_payload[f'{product}_fit_input_jy'] = np.asarray(fit_results[product]['fit_input_jy'], dtype=np.float64)
            multi_reflection_periods = [component['period_mhz'] for component in fit.get('multi_reflection_components', [])]
            save_payload[f'{product}_fit_multi_reflection_periods_mhz'] = np.asarray(multi_reflection_periods, dtype=np.float64)

    if transfer_mode_active:
        fig.suptitle(
            'RR/LL ripple transfer\n'
            f'template={template_source_name} -> target={target_source_name} (mode={args.transfer_mode})',
            fontsize=11,
        )
    else:
        fig.suptitle(f'RR/LL spectral sinusoid fit\nsource={source_name}', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(png_path, bbox_inches='tight')
    plt.close(fig)

    txt_path.write_text('\n'.join(text_lines) + '\n', encoding='utf-8')
    np.savez_compressed(npz_path, **save_payload)

    print('\n'.join(text_lines))
    print(f'Wrote: {png_path}')
    print(f'Wrote: {txt_path}')
    print(f'Wrote: {npz_path}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())