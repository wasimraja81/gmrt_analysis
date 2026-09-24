"""Unit tests for ripple_characterisation.render_engineer_report_md (RC-14).

Pure string-content assertions against hand-built summary dicts (matching
the schema produced by build_summary_dict) — decoupled from
characterise_ripple() itself so each scenario (significant-in-one-pol-only,
large crosscheck delta, missing bandpass solution) can be constructed
directly and exactly.
"""

import re

from modules.ripple_characterisation import render_engineer_report_md


def _make_component(period_mhz, amplitude, cable_length_m, snr, classification='primary', significant=True):
    return {
        'period_mhz': period_mhz,
        'amplitude': amplitude,
        'phase_rad': 0.0,
        'snr': snr,
        'classification': classification,
        'ratio_to_primary': 1.0,
        'cable_length_m': cable_length_m,
        'significant': significant,
    }


def _make_summary(per_pol_components, *, crosscheck=None, bandpass_solution_path=None,
                   run_ts='', workflow_run_id='', git_commit=''):
    per_pol = {}
    for pol, components in per_pol_components.items():
        per_pol[pol] = {
            'power_law_fit': {
                'coeffs': [0.0, -0.7, 0.0], 'nu0_hz': 3.16e8,
                'alpha_nu0': -0.7, 'beta': 0.0, 'rss_log': 0.01, 'mask_n': 256,
            },
            'noise_floor_sigma': 0.01,
            'rms_before': 0.05,
            'rms_after': 0.01,
            'components': components,
        }
    return {
        'schema': 'gmrt-ripple-characterisation-v1',
        'generated_at': '2026-09-24T00:00:00+00:00',
        'run_ts': run_ts,
        'workflow_run_id': workflow_run_id,
        'git_commit': git_commit,
        'source': 'TESTSRC',
        'physical_model_mode': 'fit',
        'known_model_used': False,
        'bandpass_solution_path': bandpass_solution_path,
        'period_bounds_mode': 'fourier',
        'period_bounds_used_mhz': {'min': 1.0, 'max': 50.0},
        'velocity_factor': 0.66,
        'tau_ripple': 2.0,
        'known_model_local_alpha_crosscheck': crosscheck,
        'per_pol': per_pol,
    }


def test_significant_in_one_pol_only_names_both_pols():
    summary = _make_summary({
        'RR': [_make_component(8.0, 0.05, 18.71, 20.0)],
        'LL': [_make_component(30.0, 0.001, 5.0, 1.0, classification='independent', significant=False)],
    })

    report = render_engineer_report_md(summary)

    assert 'RR' in report and 'LL' in report
    assert '18.71' in report  # RR's cable length
    assert 'no significant ripple was detected' in report
    # The non-significant LL component must not be reported as a "Bottom line" detection.
    bottom_line_section = report.split('## Bottom line')[1].split('## What this means')[0]
    assert 'LL' in bottom_line_section
    assert 'no significant ripple was detected' in bottom_line_section


def test_large_known_model_crosscheck_delta_triggers_caveat_with_numeric_value():
    crosscheck = {
        'RR': {'alpha_known_nu0': -0.9, 'alpha_local_fit_nu0': -0.41, 'delta': 0.49},
    }
    summary = _make_summary(
        {'RR': [_make_component(8.0, 0.05, 18.71, 20.0)]},
        crosscheck=crosscheck,
    )

    report = render_engineer_report_md(summary)

    assert 'Caveats' in report
    assert '0.49' in report
    assert 'RR' in report.split('## Caveats')[1].split('## What would change')[0]


def test_small_known_model_crosscheck_delta_does_not_trigger_caveat():
    crosscheck = {
        'RR': {'alpha_known_nu0': -0.72, 'alpha_local_fit_nu0': -0.70, 'delta': 0.02},
    }
    summary = _make_summary(
        {'RR': [_make_component(8.0, 0.05, 18.71, 20.0)]},
        crosscheck=crosscheck,
    )

    report = render_engineer_report_md(summary)

    caveats_section = report.split('## Caveats')[1].split('## What would change')[0]
    assert 'Delta-alpha' not in caveats_section


def test_missing_bandpass_solution_renders_as_none_raw_data():
    summary = _make_summary({'RR': [], 'LL': []}, bandpass_solution_path=None)

    report = render_engineer_report_md(summary)

    assert 'none — raw data' in report
    assert 'None\n' not in report


def test_no_unresolved_template_placeholders():
    summary = _make_summary({
        'RR': [_make_component(8.0, 0.05, 18.71, 20.0)],
        'LL': [],
    })

    report = render_engineer_report_md(summary)

    assert not re.search(r'\{[a-zA-Z_]+\}', report)
    assert '{' not in report and '}' not in report
