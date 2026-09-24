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
        ``per_pol``: dict keyed by polarisation label, each value a dict
            with ``power_law_fit`` (the local fit, always computed),
            ``noise_floor_sigma``, ``rms_before``, ``rms_after``,
            ``components`` (list of dicts, each with ``period_mhz``,
            ``amplitude``, ``phase_rad``, ``snr``, ``classification``,
            ``ratio_to_primary``, ``cable_length_m``, ``significant``).

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
        }

    return {
        'source': source,
        'physical_model_mode': physical_model_mode,
        'known_model_used': known_model_used,
        'known_model_local_alpha_crosscheck': known_model_local_alpha_crosscheck,
        'period_bounds_mode': period_bounds_mode,
        'period_bounds_used_mhz': {'min': float(period_min_mhz), 'max': float(period_max_mhz)},
        'per_pol': per_pol,
    }
