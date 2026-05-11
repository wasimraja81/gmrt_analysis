#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import plotVis as pv
import ugmrt_query as q


def robust_sigma_mad(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float('nan')
    med = float(np.nanmedian(arr))
    mad = float(np.nanmedian(np.abs(arr - med)))
    return 1.4826 * mad


def _nan_running_median(values: np.ndarray, half_window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    out = np.full(n, np.nan, dtype=np.float64)
    hw = int(max(0, half_window))
    for idx in range(n):
        lo = max(0, idx - hw)
        hi = min(n, idx + hw + 1)
        window = arr[lo:hi]
        finite = window[np.isfinite(window)]
        if finite.size:
            out[idx] = float(np.nanmedian(finite))
    return out


def detect_channel_outliers(signal_jy: np.ndarray, clip_sigma: float, half_window: int) -> np.ndarray:
    if clip_sigma <= 0:
        return np.zeros_like(signal_jy, dtype=bool)
    trend = _nan_running_median(signal_jy, half_window=half_window)
    resid = np.asarray(signal_jy, dtype=np.float64) - trend
    sigma = robust_sigma_mad(resid)
    if not np.isfinite(sigma) or sigma <= 0:
        return np.zeros_like(signal_jy, dtype=bool)
    return np.isfinite(resid) & (np.abs(resid) > float(clip_sigma) * sigma)


def dilate_channel_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    r = int(max(0, radius))
    if r == 0 or m.size == 0:
        return m
    out = m.copy()
    idx = np.flatnonzero(m)
    for i in idx:
        lo = max(0, int(i) - r)
        hi = min(m.size, int(i) + r + 1)
        out[lo:hi] = True
    return out


def baseline_vector_average(vis: dict, corrected_cache: dict | None, product: str) -> np.ndarray:
    spectral_data, spectral_flags = pv._get_product_data(product, vis, corrected_cache, solution=None)
    finite = np.isfinite(spectral_data.real) & np.isfinite(spectral_data.imag)
    good = (~spectral_flags) & finite
    weights = np.asarray(vis['weight'])[:, :, 0]

    weighted_num = np.nansum(np.where(good, weights * spectral_data, 0.0), axis=0)
    weighted_den = np.nansum(np.where(good, weights, 0.0), axis=0)

    scalar_amp = np.full(spectral_data.shape[1], np.nan, dtype=np.float64)
    valid = weighted_den > 0
    scalar_num = np.nansum(np.where(good, weights * np.abs(spectral_data), 0.0), axis=0)
    scalar_amp[valid] = scalar_num[valid] / weighted_den[valid]
    return scalar_amp


def evaluate_source_flux_model_jy(freq_hz: np.ndarray, source_name: str) -> np.ndarray:
    registry = getattr(q, '_FLUX_MODEL_REGISTRY', {})
    model_fn = registry.get(str(source_name).upper())
    if model_fn is None:
        raise ValueError(f'No flux model registered for source={source_name!r}')
    return np.asarray(model_fn(np.asarray(freq_hz, dtype=np.float64)), dtype=np.float64)


def ripple_model_from_components(freq_mhz: np.ndarray, components: list[dict]) -> np.ndarray:
    model = np.zeros_like(freq_mhz, dtype=np.float64)
    for comp in components:
        amp = float(comp.get('amplitude_jy', 0.0))
        period = float(comp.get('period_mhz', np.nan))
        phase = float(comp.get('phase_rad', 0.0))
        if not np.isfinite(period) or period <= 0:
            continue
        model += amp * np.sin(2.0 * np.pi * freq_mhz / period + phase)
    return model


def fit_sinusoid_fft_seeded(f_mhz: np.ndarray, y_jy: np.ndarray, period_range_mhz=(0.5, 20.0)):
    from scipy.optimize import curve_fit

    f = np.asarray(f_mhz, dtype=np.float64)
    y = np.asarray(y_jy, dtype=np.float64)
    finite = np.isfinite(f) & np.isfinite(y)
    f = f[finite]
    y = y[finite]
    if f.size < 20:
        return None

    n_grid = 512
    f_grid = np.linspace(f[0], f[-1], n_grid)
    y_grid = np.interp(f_grid, f, y)
    y_grid -= np.nanmean(y_grid)
    df = (f_grid[-1] - f_grid[0]) / (n_grid - 1)
    fft_amp = np.abs(np.fft.rfft(y_grid))
    fft_freq = np.fft.rfftfreq(n_grid, d=df)
    with np.errstate(divide='ignore'):
        fft_period = np.where(fft_freq > 0, 1.0 / fft_freq, np.inf)
    in_range = (fft_period >= period_range_mhz[0]) & (fft_period <= period_range_mhz[1])
    if in_range.any():
        p0_period = float(fft_period[in_range][np.argmax(fft_amp[in_range])])
    else:
        p0_period = float(np.sqrt(period_range_mhz[0] * period_range_mhz[1]))

    def model(fx, a0, A, P, phi):
        return a0 + A * np.sin(2.0 * np.pi * fx / P + phi)

    try:
        popt, _ = curve_fit(
            model,
            f,
            y,
            p0=[float(np.nanmedian(y)), float(np.nanstd(y)), p0_period, 0.0],
            bounds=([-np.inf, -np.inf, period_range_mhz[0], -np.pi], [np.inf, np.inf, period_range_mhz[1], np.pi]),
            maxfev=20000,
        )
        return popt
    except Exception:
        return None


def load_source_products(
    index: dict,
    source: str,
    products: list[str],
    chan_range,
    timerange,
    elevation_min,
    elevation_max,
    uvrange_m,
    uvrange_klambda,
    flag_paths: list[str] | None,
    apply_flags: bool,
    solutions: list[dict] | None,
    apply_correction: bool,
):
    vis = q.load_vis_for_source(
        index,
        source=source,
        stokes=products,
        chan_range=chan_range,
        timerange=timerange,
        uvrange_m=uvrange_m,
        uvrange_klambda=uvrange_klambda,
        elevation_min_deg=elevation_min,
        elevation_max_deg=elevation_max,
        flag_all_corrs_if_any_rawvis_flagged=True,
    )

    if apply_flags and flag_paths:
        ant_name_map, _ = pv._build_ant_maps(index)
        vis, _ = q.apply_flag_tables_to_vis(vis, antenna_name_map=ant_name_map, flag_table_paths=list(flag_paths))

    vis_for_avg = vis
    corrected_cache = None
    if apply_correction and solutions:
        # Apply exactly in CLI-provided order (no time-selector logic).
        for sol in solutions:
            corr = q.apply_bandpass_solution(vis_for_avg, sol)
            vis_for_avg = dict(vis_for_avg)
            vis_for_avg['vis_complex'] = np.asarray(corr['vis_complex_corrected'], dtype=np.complex128)
            vis_for_avg['flagged'] = np.asarray(corr.get('flagged_corrected', corr.get('flagged')), dtype=bool)

    freq_hz = np.asarray(vis_for_avg['freqs_hz'], dtype=np.float64)
    data = {p: baseline_vector_average(vis_for_avg, corrected_cache, p) for p in products}
    return freq_hz, data


def main() -> int:
    parser = argparse.ArgumentParser(description='Simple template-first ripple calibration workflow.')
    parser.add_argument('--fits', required=True)
    parser.add_argument('--index-cache', required=True)
    parser.add_argument('--chan-range', nargs=2, type=int, default=None)
    parser.add_argument('--docal-template', choices=['on', 'off'], default='on', help='Apply gain calibration to template source.')
    parser.add_argument('--docal-target', choices=['on', 'off'], default='on', help='Apply gain calibration to target source.')
    parser.add_argument('--doflag-template', choices=['on', 'off'], default='on', help='Apply flag tables to template source.')
    parser.add_argument('--doflag-target', choices=['on', 'off'], default='on', help='Apply flag tables to target source.')
    parser.add_argument('--template-gain-tables', nargs='+', default=None, help='Gain tables used for template source (primary).')
    parser.add_argument('--target-gain-tables', nargs='+', default=None, help='Gain tables used for target source.')
    parser.add_argument('--template-flag-tables', nargs='+', default=None, help='Flag tables used for template source.')
    parser.add_argument('--target-flag-tables', nargs='+', default=None, help='Flag tables used for target source.')
    parser.add_argument('--template-source', required=True, help='Primary/template source (mandatory).')
    parser.add_argument('--target-source', default=None, help='Optional target source for transfer application.')
    parser.add_argument('--products', default='RR,LL')
    parser.add_argument('--derive-additional-ripples-on-target', type=int, default=0)
    parser.add_argument('--input-outlier-clip-sigma', type=float, default=0.0)
    parser.add_argument('--input-outlier-half-window', type=int, default=2)
    parser.add_argument('--input-outlier-dilate', type=int, default=1)
    parser.add_argument('--time-range', nargs=2, default=None)
    parser.add_argument('--elevation-min', type=float, default=None)
    parser.add_argument('--elevation-max', type=float, default=None)
    parser.add_argument('--uvrange-m', nargs=2, type=float, default=None)
    parser.add_argument('--uvrange-klambda', nargs=2, type=float, default=None)
    parser.add_argument('--outdir', default='tmp')
    parser.add_argument('--outfile-prefix', default='ripple_calibration')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    index = q.get_or_build_row_index(args.fits, cache_path=args.index_cache, force_rebuild=False, validation_mode='fast', write_cache=True, override_dud_names=None)

    products = [p.strip().upper() for p in args.products.split(',') if p.strip()]
    if not products:
        products = ['RR', 'LL']

    template_gain_paths = list(args.template_gain_tables or [])
    target_gain_paths = list(args.target_gain_tables or [])
    template_flag_paths = list(args.template_flag_tables or [])
    target_flag_paths = list(args.target_flag_tables or [])

    if args.docal_template == 'on' and not template_gain_paths:
        raise SystemExit('ERROR: --docal-template=on requires --template-gain-tables.')
    if args.target_source and args.docal_target == 'on' and not target_gain_paths:
        raise SystemExit('ERROR: --docal-target=on with --target-source requires --target-gain-tables.')

    if args.docal_template == 'on':
        template_solutions_for_template = []
        for path in template_gain_paths:
            template_solutions_for_template.append(pv._load_solution(path, args.template_source, index))
    else:
        template_solutions_for_template = None

    template_diag_png = outdir / f'{args.outfile_prefix}_template_diagnostics.png'
    template_diag = q.run_bandpass_diagnostics(
        index,
        template_solutions_for_template,
        source=args.template_source,
        chan_range=tuple(args.chan_range) if args.chan_range else None,
        stokes=tuple(products),
        apply_flag_tables=(args.doflag_template == 'on' and len(template_flag_paths) > 0),
        flag_table_path=(template_flag_paths if (args.doflag_template == 'on') else None),
        apply_correction=(args.docal_template == 'on'),
        title=f'Template diagnostics ({args.template_source})',
        save_path=template_diag_png,
        timerange=tuple(args.time_range) if args.time_range else None,
        uvrange_m=tuple(args.uvrange_m) if args.uvrange_m else None,
        uvrange_klambda=tuple(args.uvrange_klambda) if args.uvrange_klambda else None,
        elevation_min_deg=args.elevation_min,
        elevation_max_deg=args.elevation_max,
        include_metric_rows=False,
    )

    template_params = {}
    for p in products:
        comps = template_diag.get('pol_results', {}).get(p, {}).get('ripple_fit_components', [])
        template_params[p] = comps

    if not args.target_source:
        txt = outdir / f'{args.outfile_prefix}.txt'
        with open(txt, 'w', encoding='utf-8') as f:
            f.write(f'template_source={args.template_source}\n')
            f.write(f'docal_template={args.docal_template}\n')
            f.write(f'template_gain_tables={template_gain_paths}\n')
            f.write(f'template_gain_tables_used={template_gain_paths if args.docal_template == "on" else []}\n')
            f.write(f'doflag_template={args.doflag_template}\n')
            f.write(f'template_flag_tables={template_flag_paths}\n')
            f.write(f'template_diagnostics_plot={template_diag_png}\n')
            for p in products:
                comps = template_params.get(p, [])
                if not comps:
                    f.write(f'{p}: ripple_components=none\n')
                    continue
                comp_txt = ', '.join([f"A={c['amplitude_jy']:.4g}Jy P={c['period_mhz']:.3f}MHz phi={c['phase_rad']:.3f}rad" for c in comps])
                f.write(f'{p}: ripple_components=[{comp_txt}]\n')
        print(f'Wrote: {txt}')
        print(f'Wrote: {template_diag_png}')
        return 0

    all_solutions_for_target = None
    if args.docal_target == 'on':
        all_solutions_for_target = []
        for path in target_gain_paths:
            sol = pv._load_solution(path, args.target_source, index)
            all_solutions_for_target.append(sol)

    freq_hz, target_products = load_source_products(
        index=index,
        source=args.target_source,
        products=products,
        chan_range=tuple(args.chan_range) if args.chan_range else None,
        timerange=tuple(args.time_range) if args.time_range else None,
        elevation_min=args.elevation_min,
        elevation_max=args.elevation_max,
        uvrange_m=tuple(args.uvrange_m) if args.uvrange_m else None,
        uvrange_klambda=tuple(args.uvrange_klambda) if args.uvrange_klambda else None,
        flag_paths=target_flag_paths,
        apply_flags=(args.doflag_target == 'on'),
        solutions=all_solutions_for_target,
        apply_correction=(args.docal_target == 'on'),
    )
    freq_mhz = freq_hz / 1e6
    target_has_flux_model = True
    try:
        target_model = evaluate_source_flux_model_jy(freq_hz, args.target_source)
    except ValueError:
        target_model = np.zeros_like(freq_hz, dtype=np.float64)
        target_has_flux_model = False

    fig, axes = plt.subplots(len(products), 2, figsize=(13, 4 * len(products)), squeeze=False, dpi=150)
    lines = [
        f'template_source={args.template_source}',
        f'target_source={args.target_source}',
        f'docal_template={args.docal_template}',
        f'docal_target={args.docal_target}',
        f'doflag_template={args.doflag_template}',
        f'doflag_target={args.doflag_target}',
        f'target_has_flux_model={target_has_flux_model}',
        f'template_gain_tables={template_gain_paths}',
        f'template_gain_tables_used={template_gain_paths if args.docal_template == "on" else []}',
        f'target_gain_tables={target_gain_paths}',
        f'template_flag_tables={template_flag_paths}',
        f'target_flag_tables={target_flag_paths}',
        f'template_diagnostics_plot={template_diag_png}',
        f'derive_additional_ripples_on_target={args.derive_additional_ripples_on_target}',
        f'input_outlier_clip_sigma={args.input_outlier_clip_sigma}',
        '',
    ]

    for i, p in enumerate(products):
        spec = np.asarray(target_products[p], dtype=np.float64)
        resid = spec - target_model if target_has_flux_model else spec.copy()

        mask = detect_channel_outliers(resid, args.input_outlier_clip_sigma, args.input_outlier_half_window)
        mask = dilate_channel_mask(mask, args.input_outlier_dilate)

        template_components = template_params.get(p, [])
        template_ripple = ripple_model_from_components(freq_mhz, template_components)

        valid = np.isfinite(resid) & np.isfinite(template_ripple) & (~mask)
        if np.count_nonzero(valid) < 10:
            scale = 0.0
        else:
            t = template_ripple[valid]
            y = resid[valid]
            denom = float(np.dot(t, t))
            scale = float(np.dot(t, y) / denom) if denom > 0 else 0.0

        transferred = scale * template_ripple
        corrected = resid - transferred

        extra_components = []
        work = corrected.copy()
        for j in range(int(max(0, args.derive_additional_ripples_on_target))):
            popt = fit_sinusoid_fft_seeded(freq_mhz[~mask], work[~mask], period_range_mhz=(0.5, 20.0))
            if popt is None:
                break
            a0, A, P, phi = [float(x) for x in popt]
            comp = a0 + A * np.sin(2.0 * np.pi * freq_mhz / P + phi)
            work = work - comp
            extra_components.append({'order': j + 1, 'offset_jy': a0, 'amplitude_jy': A, 'period_mhz': P, 'phase_rad': phi})

        sigma_in = robust_sigma_mad(np.where(mask, np.nan, resid))
        sigma_out = robust_sigma_mad(np.where(mask, np.nan, work))
        improvement = sigma_in / sigma_out if sigma_out > 0 else np.nan

        ax0 = axes[i, 0]
        ax1 = axes[i, 1]
        ax0.plot(freq_mhz, 1e3 * spec, lw=1.1, color='tab:blue', label='target spectrum')
        if target_has_flux_model:
            ax0.plot(freq_mhz, 1e3 * (target_model + transferred), lw=1.1, color='tab:orange', label='target model + transferred ripple')
        else:
            ax0.plot(freq_mhz, 1e3 * transferred, lw=1.1, color='tab:orange', label='transferred ripple (no target flux model)')
        if np.count_nonzero(mask):
            ax0.scatter(freq_mhz[mask], 1e3 * spec[mask], s=18, facecolors='none', edgecolors='magenta', linewidths=1.0, label='masked')
        ax0.set_title(f'{p}: target spectrum and transferred template ripple')
        ax0.set_xlabel('Frequency (MHz)')
        ax0.set_ylabel('Amplitude (mJy)')
        ax0.grid(True, alpha=0.25)
        ax0.legend(fontsize=8)

        residual_plot = np.where(mask, np.nan, work)
        ax1.plot(freq_mhz, 1e3 * residual_plot, lw=1.0, color='tab:green', label='final residual')
        if np.count_nonzero(mask):
            ax1.scatter(freq_mhz[mask], 1e3 * work[mask], s=18, color='magenta', edgecolors='white', linewidths=0.4, label='masked')
        ax1.axhline(0.0, color='k', lw=0.8, alpha=0.6)
        ax1.set_title(f'{p}: residual (unmasked robust σ={1e3*sigma_out:.3f} mJy, improvement={improvement:.2f}x)')
        ax1.set_xlabel('Frequency (MHz)')
        ax1.set_ylabel('Residual (mJy)')
        ax1.grid(True, alpha=0.25)
        if np.count_nonzero(mask):
            ax1.legend(fontsize=8)

        tpl_txt = '; '.join([f"A={c['amplitude_jy']:.3g},P={c['period_mhz']:.3f},phi={c['phase_rad']:.3f}" for c in template_components]) or 'none'
        extra_txt = '; '.join([f"A={c['amplitude_jy']:.3g},P={c['period_mhz']:.3f},phi={c['phase_rad']:.3f}" for c in extra_components]) or 'none'
        lines.append(
            f'{p}: template_scale={scale:.6g}, masked_channels={int(np.count_nonzero(mask))}, '
            f'input_sigma={1e3*sigma_in:.3f} mJy, residual_sigma={1e3*sigma_out:.3f} mJy, improvement={improvement:.2f}x\n'
            f'  template_components=[{tpl_txt}]\n'
            f'  additional_target_components=[{extra_txt}]'
        )

    fig.suptitle(f'Template-first ripple calibration | {args.template_source} -> {args.target_source}', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    png = outdir / f'{args.outfile_prefix}.png'
    txt = outdir / f'{args.outfile_prefix}.txt'
    fig.savefig(png, dpi=170, bbox_inches='tight')
    plt.close(fig)

    with open(txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    print(f'Wrote: {png}')
    print(f'Wrote: {txt}')
    print(f'Wrote: {template_diag_png}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
