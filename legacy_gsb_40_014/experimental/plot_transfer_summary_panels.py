#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def robust_sigma_mad(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float('nan')
    med = float(np.nanmedian(v))
    mad = float(np.nanmedian(np.abs(v - med)))
    return 1.4826 * mad


def fit_fixed_period_component(freq_mhz: np.ndarray, signal_jy: np.ndarray, period_mhz: float) -> dict:
    omega = 2.0 * np.pi / float(period_mhz)
    finite = np.isfinite(freq_mhz) & np.isfinite(signal_jy)
    x = np.asarray(freq_mhz[finite], dtype=np.float64)
    y = np.asarray(signal_jy[finite], dtype=np.float64)

    design = np.column_stack([np.sin(omega * x), np.cos(omega * x)])
    coeffs, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    s_coeff, c_coeff = coeffs

    model = np.full_like(freq_mhz, np.nan, dtype=np.float64)
    model[finite] = s_coeff * np.sin(omega * x) + c_coeff * np.cos(omega * x)

    amp = float(np.hypot(s_coeff, c_coeff))
    phase_rad = float(np.arctan2(c_coeff, s_coeff))
    freq_cyc_per_mhz = float(1.0 / period_mhz)
    return {
        'omega': float(omega),
        'sin_coeff': float(s_coeff),
        'cos_coeff': float(c_coeff),
        'model': model,
        'amp_jy': amp,
        'phase_rad': phase_rad,
        'period_mhz': float(period_mhz),
        'freq_cyc_per_mhz': freq_cyc_per_mhz,
    }


def load_product(npz_path: Path, product: str) -> dict:
    with np.load(npz_path) as data:
        freq_mhz = np.asarray(data['freq_mhz'], dtype=np.float64)
        scalar = np.asarray(data[f'{product}_scalar_amp_jy'], dtype=np.float64)
        model = np.asarray(data[f'{product}_transfer_model_jy'], dtype=np.float64)
        residual = np.asarray(data[f'{product}_transfer_residual_jy'], dtype=np.float64)
        template = np.asarray(data[f'{product}_template_ripple_jy'], dtype=np.float64)
        period = float(np.asarray(data[f'{product}_template_period_mhz']).ravel()[0])

        template_scalar_key = f'{product}_template_scalar_amp_jy'
        template_model_key = f'{product}_template_model_jy'
        template_flux_model_key = f'{product}_template_flux_model_jy'
        template_pb_subtracted_key = f'{product}_template_pb_subtracted_amp_jy'
        template_scalar = np.asarray(data[template_scalar_key], dtype=np.float64) if template_scalar_key in data.files else None
        template_model = np.asarray(data[template_model_key], dtype=np.float64) if template_model_key in data.files else None
        template_flux_model = np.asarray(data[template_flux_model_key], dtype=np.float64) if template_flux_model_key in data.files else None
        template_pb_subtracted = np.asarray(data[template_pb_subtracted_key], dtype=np.float64) if template_pb_subtracted_key in data.files else None

        template_multi_reflection_key = f'{product}_template_multi_reflection_periods_mhz'
        template_multi_reflection_periods = np.asarray(data[template_multi_reflection_key], dtype=np.float64) if template_multi_reflection_key in data.files else np.array([], dtype=np.float64)

        multi_reflection_periods_key = f'{product}_target_extra_multi_reflection_periods_mhz'
        multi_reflection_periods = np.asarray(data[multi_reflection_periods_key], dtype=np.float64) if multi_reflection_periods_key in data.files else np.array([], dtype=np.float64)

        n_multi_reflection_key = f'{product}_target_extra_n_multi_reflection_components'
        n_multi_reflection = int(np.asarray(data[n_multi_reflection_key]).ravel()[0]) if n_multi_reflection_key in data.files else int(multi_reflection_periods.size)

        channel_mask_key = f'{product}_channel_mask'
        channel_mask = np.asarray(data[channel_mask_key], dtype=bool) if channel_mask_key in data.files else np.zeros(freq_mhz.shape, dtype=bool)

        resid_sigma_key = f'{product}_transfer_resid_sigma_jy'
        stored_resid_sigma = float(np.asarray(data[resid_sigma_key]).ravel()[0]) if resid_sigma_key in data.files else None
        input_sigma_key = f'{product}_transfer_input_sigma_jy'
        stored_input_sigma = float(np.asarray(data[input_sigma_key]).ravel()[0]) if input_sigma_key in data.files else None

        # Unified sorted-by-amplitude periods list (free_phase_template stores these directly)
        fp_periods_key = f'{product}_free_phase_periods_mhz'
        fp_amps_key = f'{product}_free_phase_amplitudes_jy'
        if fp_periods_key in data.files and fp_amps_key in data.files:
            fp_periods = np.asarray(data[fp_periods_key], dtype=np.float64)
            fp_amps = np.asarray(data[fp_amps_key], dtype=np.float64)
            # Merge with additional target multi-reflection components
            all_periods = list(fp_periods)
            all_amps_jy = list(fp_amps)
            for p, a in zip(multi_reflection_periods, [0.0] * multi_reflection_periods.size):
                all_periods.append(float(p))
                all_amps_jy.append(float(a))
            # Sort by amp descending
            sorted_pa = sorted(zip(all_amps_jy, all_periods), reverse=True)
            unified_periods_text = ','.join(f'{p:.3f}({1e3*a:.1f}mJy)' for a, p in sorted_pa)
        else:
            # fixed_template or fixed_period: build from template + multi-reflection
            unified_periods_text = None

    tmpl_fit = fit_fixed_period_component(freq_mhz, template, period)
    return {
        'freq_mhz': freq_mhz,
        'scalar_jy': scalar,
        'model_jy': model,
        'residual_jy': residual,
        'template_jy': template,
        'template_scalar_jy': template_scalar,
        'template_model_jy': template_model,
        'template_flux_model_jy': template_flux_model,
        'template_pb_subtracted_jy': template_pb_subtracted,
        'template_multi_reflection_periods': template_multi_reflection_periods,
        'template_fit': tmpl_fit,
        'multi_reflection_periods': multi_reflection_periods,
        'n_multi_reflection': n_multi_reflection,
        'channel_mask': channel_mask,
        'stored_resid_sigma': stored_resid_sigma,
        'stored_input_sigma': stored_input_sigma,
        'unified_periods_text': unified_periods_text,
    }


def make_summary(rr: dict, ll: dict, out_png: Path, title: str):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(17, 9), dpi=160)

    for row_idx, (label, d) in enumerate((('RR', rr), ('LL', ll))):
        freq = d['freq_mhz']
        template_fit = d['template_fit']
        channel_mask = np.asarray(d.get('channel_mask', np.zeros(freq.shape, dtype=bool)), dtype=bool)
        masked_count = int(np.count_nonzero(channel_mask))

        ax_t = axes[row_idx, 0]
        ax_s = axes[row_idx, 1]
        ax_r = axes[row_idx, 2]

        if d['template_scalar_jy'] is not None:
            ax_t.plot(freq, 1e3 * d['template_scalar_jy'], lw=1.2, color='tab:purple', label='3C48 spectrum |V|')
            if d['template_flux_model_jy'] is not None:
                ax_t.plot(freq, 1e3 * d['template_flux_model_jy'], lw=1.1, color='tab:red', alpha=0.9, label='3C48 PB2017 model')
            if d['template_model_jy'] is not None:
                ax_t.plot(freq, 1e3 * d['template_model_jy'], lw=1.1, color='tab:orange', label='3C48 multi-reflection ripple fit')
            ax_t.plot(freq, 1e3 * d['template_jy'], lw=0.9, ls='--', color='k', alpha=0.7, label='3C48 ripple component')
            ax_t.set_title(f'{label}: 3C48 spectrum, PB2017 model, and ripple fit')
        else:
            ax_t.plot(freq, 1e3 * d['template_jy'], lw=1.0, color='tab:purple', label='3C48 ripple (derived)')
            ax_t.plot(freq, 1e3 * template_fit['model'], lw=1.0, ls='--', color='k', alpha=0.8, label='3C48 sine fit')
            ax_t.set_title(f'{label}: 3C48 template fit')
        ax_t.set_xlabel('Frequency (MHz)')
        ax_t.set_ylabel('Amplitude (mJy)')
        ax_t.grid(True, alpha=0.25)
        ax_t.legend(loc='best', fontsize=8)
        template_multi_reflections_txt = ','.join(f'{p:.3f}' for p in d['template_multi_reflection_periods']) if d['template_multi_reflection_periods'].size else 'none'
        ax_t.text(
            0.02,
            0.98,
            (
                f'period={template_fit["period_mhz"]:.3f} MHz\n'
                f'freq={template_fit["freq_cyc_per_mhz"]:.4f} cyc/MHz\n'
                f'amp={1e3 * template_fit["amp_jy"]:.2f} mJy\n'
                f'phase={template_fit["phase_rad"]:.3f} rad\n'
                f'multi-reflection periods={template_multi_reflections_txt}'
            ),
            transform=ax_t.transAxes,
            va='top',
            ha='left',
            fontsize=8,
            bbox=dict(facecolor='white', alpha=0.75, edgecolor='0.8'),
        )

        ax_s.plot(freq, 1e3 * d['scalar_jy'], lw=1.2, color='tab:blue', label=f'{label} target |V|')
        ax_s.plot(freq, 1e3 * d['model_jy'], lw=1.1, color='tab:orange', label='transfer + multi-reflection model')
        ax_s.plot(freq, 1e3 * d['template_jy'], lw=1.0, color='tab:purple', alpha=0.8, label='3C48 template ripple')
        if masked_count > 0:
            ax_s.scatter(
                freq[channel_mask],
                1e3 * d['scalar_jy'][channel_mask],
                s=22,
                facecolors='none',
                edgecolors='magenta',
                linewidths=1.2,
                zorder=6,
                label='masked channels',
            )
        ax_s.set_title(f'{label}: target spectrum and full model')
        ax_s.set_xlabel('Frequency (MHz)')
        ax_s.set_ylabel('Amplitude (mJy)')
        ax_s.grid(True, alpha=0.25)
        ax_s.legend(loc='best', fontsize=8)

        # Prefer the authoritative sigma values stored by the fit script;
        # fall back to recomputing from the stored residual if not available.
        if d.get('stored_resid_sigma') is not None and d.get('stored_input_sigma') is not None:
            resid_sigma = float(d['stored_resid_sigma'])
            input_sigma = float(d['stored_input_sigma'])
        else:
            resid_for_sigma = np.where(channel_mask, np.nan, d['residual_jy'])
            input_for_sigma = np.where(channel_mask, np.nan, d['scalar_jy'])
            resid_sigma = robust_sigma_mad(resid_for_sigma)
            input_sigma = robust_sigma_mad(input_for_sigma)
        improvement = input_sigma / resid_sigma if resid_sigma > 0 else np.nan
        if d.get('unified_periods_text') is not None:
            periods_ann = f'periods (MHz)[amp]: {d["unified_periods_text"]}'
        else:
            multi_reflections_txt = ','.join(f'{p:.3f}' for p in d['multi_reflection_periods']) if d['multi_reflection_periods'].size else 'none'
            periods_ann = f'n_multi_reflection={d["n_multi_reflection"]}\nmulti-reflection periods (MHz): {multi_reflections_txt}'
        ax_s.text(
            0.02,
            0.98,
            f'{periods_ann}\nmasked_channels={masked_count}',
            transform=ax_s.transAxes,
            va='top',
            ha='left',
            fontsize=7,
            bbox=dict(facecolor='white', alpha=0.75, edgecolor='0.8'),
        )

        residual_line = np.where(channel_mask, np.nan, d['residual_jy'])
        ax_r.plot(freq, 1e3 * residual_line, lw=1.0, color='tab:green')
        if masked_count > 0:
            ax_r.scatter(
                freq[channel_mask],
                1e3 * d['residual_jy'][channel_mask],
                s=24,
                color='magenta',
                edgecolors='white',
                linewidths=0.4,
                zorder=6,
                label='masked residual channels',
            )
        ax_r.axhline(0.0, color='k', lw=0.8, alpha=0.6)
        ax_r.set_title(
            f'{label}: residual | unmasked robust σ={1e3 * resid_sigma:.2f} mJy | improvement={improvement:.2f}x'
        )
        ax_r.set_xlabel('Frequency (MHz)')
        ax_r.set_ylabel('Residual (mJy)')
        ax_r.grid(True, alpha=0.25)
        if masked_count > 0:
            ax_r.legend(loc='best', fontsize=8)
            ax_r.text(
                0.02,
                0.98,
                f'magenta=masked\nexcluded from fit/σ\ncount={masked_count}',
                transform=ax_r.transAxes,
                va='top',
                ha='left',
                fontsize=8,
                bbox=dict(facecolor='white', alpha=0.75, edgecolor='0.8'),
            )

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches='tight')
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description='Combine RR/LL transfer results into one multi-panel summary plot.')
    parser.add_argument('--rr-npz', required=True, help='RR transfer output npz (product RR)')
    parser.add_argument('--ll-npz', required=True, help='LL transfer output npz (product LL)')
    parser.add_argument('--out', required=True, help='Output PNG path')
    parser.add_argument('--title', default='3C48 -> 3C468.1 Ripple Transfer Summary')
    args = parser.parse_args()

    rr = load_product(Path(args.rr_npz), 'RR')
    ll = load_product(Path(args.ll_npz), 'LL')
    make_summary(rr, ll, Path(args.out), args.title)
    print(f'Wrote: {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
