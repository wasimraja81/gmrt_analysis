"""Unit tests for the RC-12 JSON/CSV summary schema.

Covers ``ripple_characterisation.build_summary_dict`` (pure dict
construction + NaN/Inf sanitisation) and ``characterise_ripple.write_summary_json``
/ ``write_components_csv`` (the file-I/O writers).
"""

import csv
import json
import math

import numpy as np
import pytest

from characterise_ripple import (
    characterise_ripple,
    write_components_csv,
    write_summary_json,
)
from modules.ripple_characterisation import build_summary_dict

_SCHEMA_TOP_LEVEL_KEYS = {
    'schema', 'generated_at', 'run_ts', 'workflow_run_id', 'git_commit',
    'source', 'physical_model_mode', 'known_model_used',
    'bandpass_solution_path', 'period_bounds_mode', 'period_bounds_used_mhz',
    'known_model_local_alpha_crosscheck', 'per_pol',
}
_PER_POL_KEYS = {'power_law_fit', 'noise_floor_sigma', 'rms_before', 'rms_after', 'components'}
_POWER_LAW_FIT_KEYS = {'coeffs', 'nu0_hz', 'alpha_nu0', 'beta', 'rss_log', 'mask_n'}
_COMPONENT_KEYS = {
    'period_mhz', 'amplitude', 'phase_rad', 'snr', 'classification',
    'ratio_to_primary', 'cable_length_m', 'significant',
}


def _make_result(make_synthetic_vis_corrected, make_synthetic_spectrum, ripple_components=None):
    freqs_hz = np.linspace(300.0, 332.0, 256) * 1e6
    data = make_synthetic_spectrum(
        freqs_hz, alpha=-0.7, beta=0.0,
        ripple_components=ripple_components, noise_sigma_jy=1e-4, seed=0,
    )
    vis_complex = np.broadcast_to(
        data['spectrum_jy'][None, :, None], (5, freqs_hz.size, 1)
    ).copy().astype(np.complex128)
    vis = make_synthetic_vis_corrected(nrows=5, nchan=freqs_hz.size, pols=('RR',), vis_complex=vis_complex)
    vis['freqs_hz'] = freqs_hz
    return characterise_ripple(vis, solution=None, source='UNREGISTEREDSRC')


def test_build_summary_dict_matches_schema_keys(make_synthetic_vis_corrected, make_synthetic_spectrum):
    result = _make_result(
        make_synthetic_vis_corrected, make_synthetic_spectrum,
        ripple_components=[{'period_mhz': 8.0, 'amplitude': 0.05}],
    )
    summary = build_summary_dict(result, run_ts='20260101_000000', workflow_run_id='abc123', git_commit='deadbee')

    assert set(summary.keys()) == _SCHEMA_TOP_LEVEL_KEYS
    assert summary['schema'] == 'gmrt-ripple-characterisation-v1'
    assert summary['run_ts'] == '20260101_000000'
    assert summary['workflow_run_id'] == 'abc123'
    assert summary['git_commit'] == 'deadbee'
    assert summary['bandpass_solution_path'] is None
    assert isinstance(summary['generated_at'], str) and summary['generated_at']

    pol_summary = summary['per_pol']['RR']
    assert set(pol_summary.keys()) == _PER_POL_KEYS
    assert set(pol_summary['power_law_fit'].keys()) == _POWER_LAW_FIT_KEYS
    assert len(pol_summary['components']) >= 1
    assert set(pol_summary['components'][0].keys()) == _COMPONENT_KEYS


def test_build_summary_dict_defaults_provenance_to_empty_string(make_synthetic_vis_corrected, make_synthetic_spectrum):
    result = _make_result(make_synthetic_vis_corrected, make_synthetic_spectrum)
    summary = build_summary_dict(result)

    assert summary['run_ts'] == ''
    assert summary['workflow_run_id'] == ''
    assert summary['git_commit'] == ''
    assert summary['bandpass_solution_path'] is None


def test_build_summary_dict_sanitises_nan_and_inf(make_synthetic_vis_corrected, make_synthetic_spectrum):
    # Zero injected ripple + near-zero noise floor can drive component SNR to
    # inf (see fit_harmonic_ripple's own inf-on-zero-sigma_hat behaviour);
    # confirm build_summary_dict never lets a raw NaN/Inf survive into the
    # returned dict (regression test for the exact bug found in the
    # prototype's existing JSON output).
    result = _make_result(make_synthetic_vis_corrected, make_synthetic_spectrum, ripple_components=None)
    summary = build_summary_dict(result)

    def _assert_no_nan_or_inf(value):
        if isinstance(value, float):
            assert math.isfinite(value), value
        elif isinstance(value, dict):
            for v in value.values():
                _assert_no_nan_or_inf(v)
        elif isinstance(value, list):
            for v in value:
                _assert_no_nan_or_inf(v)

    _assert_no_nan_or_inf(summary)
    # json.dumps with allow_nan=False raises on any surviving NaN/Inf token.
    json.dumps(summary, allow_nan=False)


def test_write_summary_json_round_trips(tmp_path, make_synthetic_vis_corrected, make_synthetic_spectrum):
    result = _make_result(
        make_synthetic_vis_corrected, make_synthetic_spectrum,
        ripple_components=[{'period_mhz': 8.0, 'amplitude': 0.05}],
    )
    summary = build_summary_dict(result)
    out_path = write_summary_json(summary, tmp_path / 'nested' / 'summary.json')

    assert out_path.exists()
    with open(out_path) as f:
        loaded = json.load(f)
    assert loaded['schema'] == 'gmrt-ripple-characterisation-v1'
    assert loaded['source'] == 'UNREGISTEREDSRC'


def test_write_components_csv_round_trips_with_one_row_per_component(
    tmp_path, make_synthetic_vis_corrected, make_synthetic_spectrum,
):
    result = _make_result(
        make_synthetic_vis_corrected, make_synthetic_spectrum,
        ripple_components=[{'period_mhz': 8.0, 'amplitude': 0.05}],
    )
    summary = build_summary_dict(result)
    out_path = write_components_csv(summary, tmp_path / 'components.csv')

    assert out_path.exists()
    with open(out_path, newline='') as f:
        rows = list(csv.DictReader(f))

    n_components = len(summary['per_pol']['RR']['components'])
    assert len(rows) == n_components
    assert all(row['pol'] == 'RR' for row in rows)
    assert all(row['source'] == 'UNREGISTEREDSRC' for row in rows)
    assert float(rows[0]['period_mhz']) > 0


def test_write_components_csv_writes_one_blank_row_for_zero_component_pol(
    tmp_path, make_synthetic_vis_corrected, make_synthetic_spectrum,
):
    result = _make_result(make_synthetic_vis_corrected, make_synthetic_spectrum, ripple_components=None)
    summary = build_summary_dict(result)
    out_path = write_components_csv(summary, tmp_path / 'components.csv')

    with open(out_path, newline='') as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert rows[0]['pol'] == 'RR'
    assert rows[0]['period_mhz'] == ''
