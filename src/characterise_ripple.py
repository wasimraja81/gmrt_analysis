#!/usr/bin/env python3
"""
Characterise the standing-wave ripple in a calibrated source's baseline-averaged
spectrum: fit a power-law continuum, detect and fit the residual harmonic ripple,
and convert period(s) to physical cable length(s).

This module's ``characterise_ripple()`` is the orchestration seam between the
pure-numerics core in ``modules.ripple_characterisation`` and real vis-dict
I/O in ``modules.ugmrt_query``. See ``ripple_characterisation_tickets.md``
(RC-10) for the design, and ``ripple_convergence_todos.md`` (Step 1 / Step 1b)
for the underlying physical model.

Scope note: this characterises the ripple only — it does not build a gain
correction, does not modify any bandpass solution, and does not write a
corrected UVFITS (that is explicitly out of scope for this branch; see
"Explicitly out of scope" in ripple_characterisation_tickets.md).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

try:
    from modules import ugmrt_query as q
    from modules import ripple_characterisation as rc
except ImportError:
    print(
        "ERROR: Could not import modules. Make sure you're running from the "
        "src/ directory (or that src/ and src/modules/ are on PYTHONPATH).",
        file=sys.stderr,
    )
    sys.exit(1)


def characterise_ripple(
    vis: dict,
    solution: Optional[dict],
    source: str,
    *,
    physical_model_mode: str = 'fit',
    n_harmonics: int = 1,
    peak_snr_threshold: float = 10.0,
    period_bounds_mode: str = 'fourier',
    period_min_mhz: Optional[float] = None,
    period_max_mhz: Optional[float] = None,
    velocity_factor: float = 1.0,
    tau_ripple: float = 2.0,
) -> dict:
    """Characterise the ripple in a (optionally bandpass-corrected) spectrum.

    Parameters
    ----------
    vis : dict
        A vis dict from ``ugmrt_query.load_vis_for_source()``.
    solution : dict, optional
        A bandpass solution from ``ugmrt_query.load_bandpass_solution()``.
        If given, ``ugmrt_query.apply_bandpass_solution()`` is applied first
        (read-only; the original ``vis`` is not modified). If ``None``, the
        raw (uncorrected) visibilities are characterised directly.
    source : str
        Source name, used to look up a registered flux model
        (``ugmrt_query._FLUX_MODEL_REGISTRY``) for ``physical_model_mode='known'``
        and for the always-computed known-model-vs-local-fit crosscheck.
    physical_model_mode : {'fit', 'known'}
        ``'fit'`` (default): fit a local power law to the baseline-averaged
        spectrum via :func:`ripple_characterisation.fit_power_law_spectrum`
        and use it as the continuum against which the ripple residual is
        formed. ``'known'``: use the registered flux model directly as the
        continuum instead. Default is ``'fit'`` — deliberately not
        ``'known'``, because a registered model can disagree substantially
        with what is measured locally (e.g. 3C468.1's Riseley et al. 2017
        model vs. a local power-law fit differ in spectral index by
        Delta-alpha ~ 0.49 over the GMRT GSB band), which risks conflating
        flux-scale error with genuine ripple. The crosscheck below is
        computed regardless of this setting so that discrepancy is always
        visible.
    n_harmonics : int
        Harmonics per ripple component, passed to
        :func:`ripple_characterisation.fit_harmonic_ripple`.
    peak_snr_threshold : float
        SNR threshold for candidate seeding, passed to
        :func:`ripple_characterisation.find_ripple_period_candidates`.
    period_bounds_mode : {'fourier', 'manual'}
        ``'fourier'`` (default): derive the period search range from the
        frequency grid via
        :func:`ripple_characterisation.derive_fourier_period_bounds`.
        ``'manual'``: use the caller-supplied ``period_min_mhz``/
        ``period_max_mhz`` directly (both required in that case).
    period_min_mhz, period_max_mhz : float, optional
        Required when ``period_bounds_mode='manual'``; ignored otherwise.
    velocity_factor : float
        Cable propagation velocity as a fraction of c, passed to
        :func:`ripple_characterisation.period_to_cable_length_m`.
    tau_ripple : float
        Significance threshold (in units of the post-fit noise floor) above
        which a ripple component is flagged ``significant`` in the output.
        Default 2.0, matching the Step 1 convergence-criteria convention in
        ``ripple_convergence_todos.md``.

    Returns
    -------
    dict with keys:
        ``source``, ``physical_model_mode``, ``known_model_used`` (bool),
        ``known_model_local_alpha_crosscheck`` (dict or ``None`` if the
        source has no registered flux model),
        ``period_bounds_mode``, ``period_bounds_used_mhz`` (``{min, max}``),
        ``freqs_hz`` (the frequency grid, shared across polarisations),
        ``per_pol``: dict keyed by polarisation label, each value a dict
            with ``power_law_fit`` (the local fit, always computed),
            ``noise_floor_sigma``, ``rms_before``, ``rms_after``,
            ``components`` (list of dicts, each with ``period_mhz``,
            ``amplitude``, ``phase_rad``, ``snr``, ``classification``,
            ``ratio_to_primary``, ``cable_length_m``, ``significant``),
            and the per-channel arrays ``real_spectrum_jy`` (measured
            spectrum), ``model_jy`` (the continuum actually used — local fit
            or known model, per ``physical_model_mode``), ``residual``
            (``real_spectrum_jy/model_jy - 1``), and ``harmonic_model`` (the
            fitted multi-component ripple model). These arrays feed the
            diagnostics plot (RC-11); the JSON summary (RC-12) stays a
            compact per-component/scalar record and does not embed them.

    Raises
    ------
    ValueError
        If ``physical_model_mode`` is not ``'fit'``/``'known'``, if
        ``period_bounds_mode='manual'`` without both bounds supplied, or if
        ``physical_model_mode='known'`` but ``source`` has no registered
        flux model.
    """
    if physical_model_mode not in ('fit', 'known'):
        raise ValueError(f"physical_model_mode must be 'fit' or 'known', got {physical_model_mode!r}.")
    if period_bounds_mode not in ('fourier', 'manual'):
        raise ValueError(f"period_bounds_mode must be 'fourier' or 'manual', got {period_bounds_mode!r}.")
    if period_bounds_mode == 'manual' and (period_min_mhz is None or period_max_mhz is None):
        raise ValueError("period_bounds_mode='manual' requires both period_min_mhz and period_max_mhz.")

    vis_use = q.apply_bandpass_solution(vis, solution) if solution is not None else vis

    freqs_hz = np.asarray(vis_use['freqs_hz'], dtype=np.float64)
    stokes_labels = list(vis_use['stokes_labels'])

    known_model_fn = q._FLUX_MODEL_REGISTRY.get(str(source).upper())
    known_model_used = physical_model_mode == 'known'
    if known_model_used and known_model_fn is None:
        raise ValueError(
            f"physical_model_mode='known' requested but no flux model is registered "
            f"for source {source!r}. Registered sources: {sorted(q._FLUX_MODEL_REGISTRY)}."
        )

    if period_bounds_mode == 'fourier':
        period_min_mhz, period_max_mhz = rc.derive_fourier_period_bounds(freqs_hz)

    known_model_local_alpha_crosscheck = None

    per_pol = {}
    for pol in stokes_labels:
        real_spec = q.get_vector_avg_spectrum(vis_use, pol)

        # Always compute a local power-law fit, regardless of mode, since it
        # both feeds 'fit' mode directly and is needed for the crosscheck
        # against a registered model in 'known' mode (or just as a sanity
        # reference when no registered model exists at all).
        local_fit = rc.fit_power_law_spectrum(freqs_hz, real_spec, allow_curvature=True)

        if known_model_used:
            model_jy = known_model_fn(freqs_hz)
        else:
            model_jy = local_fit['model_jy']

        if known_model_fn is not None:
            nu0 = local_fit['nu0_hz']
            eps = 1e-3
            s_lo = float(known_model_fn(np.array([nu0 * (1.0 - eps)]))[0])
            s_hi = float(known_model_fn(np.array([nu0 * (1.0 + eps)]))[0])
            alpha_known_nu0 = (
                (np.log10(s_hi) - np.log10(s_lo))
                / (np.log10(1.0 + eps) - np.log10(1.0 - eps))
            )
            known_model_local_alpha_crosscheck = known_model_local_alpha_crosscheck or {}
            known_model_local_alpha_crosscheck[pol] = {
                'alpha_known_nu0': float(alpha_known_nu0),
                'alpha_local_fit_nu0': float(local_fit['alpha_nu0']),
                'delta': float(local_fit['alpha_nu0'] - alpha_known_nu0),
            }

        residual = real_spec / model_jy - 1.0

        candidates = rc.find_ripple_period_candidates(
            freqs_hz, residual, period_min_mhz, period_max_mhz,
            peak_snr_threshold=peak_snr_threshold,
        )
        harmonic_fit = rc.fit_harmonic_ripple(
            freqs_hz, residual, candidates['periods_mhz'], n_harmonics=n_harmonics,
        )

        finite = np.isfinite(residual)
        noise_floor_sigma = rc.mad_sigma(residual[finite])

        components = []
        for comp in harmonic_fit['components']:
            cable_length_m = rc.period_to_cable_length_m(comp['period_mhz'], velocity_factor)
            significant = bool(comp['snr'] >= tau_ripple) if np.isfinite(comp['snr']) else False
            components.append({
                **comp,
                'cable_length_m': cable_length_m,
                'significant': significant,
            })

        per_pol[pol] = {
            'power_law_fit': local_fit,
            'noise_floor_sigma': noise_floor_sigma,
            'rms_before': harmonic_fit['rms_before'],
            'rms_after': harmonic_fit['rms_after'],
            'components': components,
            # Per-channel arrays needed by the diagnostics plot (RC-11) and
            # useful for ad hoc inspection; not embedded in the JSON summary
            # (RC-12), which stays a compact per-component/scalar record.
            'real_spectrum_jy': real_spec,
            'model_jy': model_jy,
            'residual': residual,
            'harmonic_model': harmonic_fit['model'],
        }

    return {
        'source': source,
        'physical_model_mode': physical_model_mode,
        'known_model_used': known_model_used,
        'known_model_local_alpha_crosscheck': known_model_local_alpha_crosscheck,
        'period_bounds_mode': period_bounds_mode,
        'period_bounds_used_mhz': {'min': float(period_min_mhz), 'max': float(period_max_mhz)},
        'freqs_hz': freqs_hz,
        'per_pol': per_pol,
    }


def plot_ripple_characterisation(result: dict, save_path, title: str = '') -> 'plt.Figure':
    """Render the science-user diagnostics plot for a characterisation run.

    Three rows, one column per polarisation present in ``result['per_pol']``
    (typically RR/LL):

    - Row 1: log-log spectrum with the fitted continuum overlay, annotated
      with ``(alpha_nu0, beta, rss_log)``.
    - Row 2: linear fractional ripple residual with the fitted harmonic
      model overlay, annotated per-component (period, amplitude, cable
      length, classification).
    - Row 3: post-fit residual with the noise floor drawn as a shaded
      +/-2*sigma_hat band, making the significance test
      (``amplitude / sigma_hat >= tau_ripple``) visible directly on the plot.

    Always renders something informative even when a polarisation has zero
    fitted components (an explicit "no significant ripple found" note,
    rather than an empty panel).

    Parameters
    ----------
    result : dict
        Output of :func:`characterise_ripple`.
    save_path : str or Path
        Output PNG path. Parent directories are created if missing.
    title : str
        Optional custom figure title; defaults to a summary built from
        ``result['source']``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    pols = sorted(result['per_pol'].keys())
    n_cols = max(1, len(pols))
    freqs_hz = np.asarray(result['freqs_hz'], dtype=np.float64)
    freqs_mhz = freqs_hz / 1e6

    fig, axes = plt.subplots(3, n_cols, figsize=(7 * n_cols, 11), squeeze=False)

    for col, pol in enumerate(pols):
        pol_result = result['per_pol'][pol]
        ax_spec, ax_ripple, ax_clean = axes[0][col], axes[1][col], axes[2][col]

        real_spec = pol_result['real_spectrum_jy']
        model_jy = pol_result['model_jy']
        residual = pol_result['residual']
        harmonic_model = pol_result['harmonic_model']
        sigma_hat = pol_result['noise_floor_sigma']
        components = pol_result['components']
        fit = pol_result['power_law_fit']

        # Row 1: log-log spectrum + continuum overlay.
        positive = real_spec > 0
        ax_spec.plot(freqs_mhz[positive], real_spec[positive], lw=1.0, label=f'{pol} data')
        ax_spec.plot(freqs_mhz, model_jy, lw=1.5, ls='--', color='k', label='continuum')
        ax_spec.set_xscale('log')
        ax_spec.set_yscale('log')
        ax_spec.set_xlabel('Frequency (MHz)')
        ax_spec.set_ylabel('Flux density (Jy)')
        ax_spec.set_title(f'{pol}: spectrum + continuum fit')
        ax_spec.grid(True, alpha=0.3, which='both')
        ax_spec.legend(fontsize=8, loc='best')
        ax_spec.text(
            0.02, 0.02,
            f"alpha_nu0={fit['alpha_nu0']:.3f}\nbeta={fit['beta']:.3f}\nrss_log={fit['rss_log']:.4g}",
            transform=ax_spec.transAxes, fontsize=8, va='bottom', ha='left',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8),
        )

        # Row 2: fractional ripple residual + harmonic model overlay.
        ax_ripple.plot(freqs_mhz, residual, lw=1.0, label='residual')
        ax_ripple.plot(freqs_mhz, harmonic_model, lw=1.4, ls='--', color='crimson', label='harmonic fit')
        ax_ripple.axhline(0.0, color='k', lw=0.8, ls=':')
        ax_ripple.set_xlabel('Frequency (MHz)')
        ax_ripple.set_ylabel('Fractional residual')
        ax_ripple.set_title(f'{pol}: ripple residual + fit')
        ax_ripple.grid(True, alpha=0.3)
        ax_ripple.legend(fontsize=8, loc='best')

        if components:
            comp_lines = [
                f"P={c['period_mhz']:.2f}MHz  A={c['amplitude']:.4f}  "
                f"L={c['cable_length_m']:.2f}m  {c['classification']}"
                + ('*' if c['significant'] else '')
                for c in components
            ]
        else:
            comp_lines = ['No ripple components found.']
        ax_ripple.text(
            0.02, 0.98, '\n'.join(comp_lines),
            transform=ax_ripple.transAxes, fontsize=7, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.85),
        )

        # Row 3: post-fit residual with noise floor band.
        cleaned = residual - harmonic_model
        ax_clean.plot(freqs_mhz, cleaned, lw=1.0, color='darkgreen', label='post-fit residual')
        if np.isfinite(sigma_hat):
            ax_clean.axhspan(-2 * sigma_hat, 2 * sigma_hat, color='gray', alpha=0.2, label='+/-2*sigma_hat')
        ax_clean.axhline(0.0, color='k', lw=0.8, ls=':')
        ax_clean.set_xlabel('Frequency (MHz)')
        ax_clean.set_ylabel('Fractional residual')
        ax_clean.set_title(f'{pol}: post-fit residual vs noise floor')
        ax_clean.grid(True, alpha=0.3)
        ax_clean.legend(fontsize=8, loc='best')

    default_title = f"Ripple characterisation: {result['source']} ({result['physical_model_mode']} mode)"
    fig.suptitle(title or default_title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches='tight')
    return fig
