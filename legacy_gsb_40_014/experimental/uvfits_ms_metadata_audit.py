#!/usr/bin/env python3
"""Audit UVFITS -> MeasurementSet conversion fidelity.

This script deliberately separates three things:

1. The raw header values present in the original UVFITS.
2. The minimal compatibility patch needed so `pyuvdata` can ingest that UVFITS.
3. The actual scientific equivalence checks between the UVFITS visibilities and
   the MeasurementSet produced for CASA.

The MS side is read with `casatools`, so the audit stays usable inside the
user's CASA-enabled venv even if `casacore` is unavailable.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from pyuvdata import UVData


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Audit metadata equivalence between original UVFITS and converted MS')
    p.add_argument('--uvfits', required=True, help='Original UVFITS path')
    p.add_argument('--ms', required=True, help='Converted MeasurementSet path')
    p.add_argument('--outdir', default='./work/uvfits_ms_audit', help='Output directory')
    p.add_argument('--tag', default='audit', help='Output filename tag')
    return p.parse_args()


def _safe_float(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def _angle_diff_rad(a: float, b: float) -> float:
    return float((a - b + np.pi) % (2.0 * np.pi) - np.pi)


def _pol_label_from_uvfits(num: int) -> str:
    mapping = {
        -8: 'yx',
        -7: 'xy',
        -6: 'yy',
        -5: 'xx',
        -4: 'lr',
        -3: 'rl',
        -2: 'll',
        -1: 'rr',
        1: 'i',
        2: 'q',
        3: 'u',
        4: 'v',
    }
    return mapping.get(int(num), str(int(num)))


def _pol_label_from_ms(num: int) -> str:
    mapping = {
        5: 'rr',
        6: 'rl',
        7: 'lr',
        8: 'll',
        9: 'xx',
        10: 'xy',
        11: 'yx',
        12: 'yy',
    }
    return mapping.get(int(num), str(int(num)))


def _raw_uvfits_header_info(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    with fits.open(path) as hdul:
        primary = hdul[0].header
        out['primary_RADESYS'] = primary.get('RADESYS')
        out['primary_EQUINOX'] = primary.get('EQUINOX')
        out['primary_DATEOBS'] = primary.get('DATE-OBS')
        out['primary_TIMESYS'] = primary.get('TIMESYS')
        out['primary_CTYPE2'] = primary.get('CTYPE2')
        out['primary_CTYPE3'] = primary.get('CTYPE3')
        out['primary_CTYPE4'] = primary.get('CTYPE4')
        out['primary_CTYPE5'] = primary.get('CTYPE5')
        out['primary_CTYPE6'] = primary.get('CTYPE6')
        if 'AIPS AN' in hdul:
            an = hdul['AIPS AN'].header
            out['an_ARRAYX'] = an.get('ARRAYX')
            out['an_ARRAYY'] = an.get('ARRAYY')
            out['an_ARRAYZ'] = an.get('ARRAYZ')
            out['an_FRAME'] = an.get('FRAME')
            out['an_XYZHAND'] = an.get('XYZHAND')
            out['an_RDATE'] = an.get('RDATE')
            out['an_FREQ'] = an.get('FREQ')
            out['an_NO_IF'] = an.get('NO_IF')
        if 'AIPS SU' in hdul:
            su = hdul['AIPS SU'].header
            out['su_EQUINOX'] = su.get('EQUINOX')
    return out


def _read_uvdata_uvfits(path: Path) -> UVData:
    uv = UVData()
    uv.read_uvfits(str(path), run_check=False)
    return uv


def _read_uvdata_uvfits_with_minimal_fix(path: Path) -> tuple[UVData, dict[str, Any]]:
    try:
        return _read_uvdata_uvfits(path), {'used_minimal_fix': False, 'patches': []}
    except Exception as exc:
        tempdir = Path(tempfile.mkdtemp(prefix='uvfits_audit_'))
        patched = tempdir / path.name
        shutil.copy2(path, patched)
        patches: list[dict[str, Any]] = []

        with fits.open(patched, mode='update') as hdul:
            primary = hdul[0].header
            if primary.get('RADESYS') == 'FK5':
                primary['RADESYS'] = 'fk5'
                patches.append({'hdu': 'PRIMARY', 'key': 'RADESYS', 'old': 'FK5', 'new': 'fk5'})
            if primary.get('TIMESYS') is None:
                primary['TIMESYS'] = 'UTC'
                patches.append({'hdu': 'PRIMARY', 'key': 'TIMESYS', 'old': None, 'new': 'UTC'})
            if 'AIPS AN' in hdul:
                an = hdul['AIPS AN'].header
                if an.get('FRAME') is None:
                    an['FRAME'] = 'ITRF'
                    patches.append({'hdu': 'AIPS AN', 'key': 'FRAME', 'old': None, 'new': 'ITRF'})
                if an.get('XYZHAND') is None:
                    an['XYZHAND'] = 'RIGHT'
                    patches.append({'hdu': 'AIPS AN', 'key': 'XYZHAND', 'old': None, 'new': 'RIGHT'})
            hdul.flush(output_verify='ignore')

        uv = _read_uvdata_uvfits(patched)
        return uv, {
            'used_minimal_fix': True,
            'read_error': str(exc),
            'patches': patches,
            'temporary_copy': str(patched),
        }


def _getcol_dict(tb: Any, columns: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    existing = set(tb.colnames())
    for col in columns:
        if col in existing:
            out[col] = tb.getcol(col)
    return out


def _read_ms_with_casatools(path: Path) -> dict[str, Any]:
    from casatools import table  # type: ignore

    ms: dict[str, Any] = {}
    tb = table()

    tb.open(str(path))
    main = _getcol_dict(
        tb,
        [
            'TIME', 'ANTENNA1', 'ANTENNA2', 'UVW', 'DATA', 'FLAG', 'WEIGHT',
            'SIGMA', 'INTERVAL', 'EXPOSURE', 'DATA_DESC_ID', 'FIELD_ID',
        ],
    )
    tb.close()
    ms['main'] = {
        'time_jd': np.asarray(main['TIME'], dtype=np.float64) / 86400.0 + 2400000.5,
        'ant1': np.asarray(main['ANTENNA1'], dtype=np.int32) + 1,
        'ant2': np.asarray(main['ANTENNA2'], dtype=np.int32) + 1,
        'uvw': np.asarray(main['UVW'], dtype=np.float64).T,
        'data': np.asarray(main['DATA']).transpose(2, 1, 0),
        'flag': np.asarray(main['FLAG'], dtype=bool).transpose(2, 1, 0),
        'weight': np.asarray(main['WEIGHT'], dtype=np.float64).T,
        'sigma': np.asarray(main['SIGMA'], dtype=np.float64).T,
        'interval': np.asarray(main['INTERVAL'], dtype=np.float64),
        'exposure': np.asarray(main['EXPOSURE'], dtype=np.float64),
        'ddid': np.asarray(main['DATA_DESC_ID'], dtype=np.int32),
        'field_id': np.asarray(main['FIELD_ID'], dtype=np.int32),
    }

    tb.open(str(path / 'ANTENNA'))
    ant = _getcol_dict(tb, ['NAME', 'STATION', 'POSITION', 'DISH_DIAMETER'])
    tb.close()
    ms['antenna'] = {
        'name': np.asarray(ant['NAME']),
        'station': np.asarray(ant['STATION']),
        'position': np.asarray(ant['POSITION'], dtype=np.float64).T,
        'dish_diameter': np.asarray(ant['DISH_DIAMETER'], dtype=np.float64),
    }

    tb.open(str(path / 'SPECTRAL_WINDOW'))
    spw = _getcol_dict(tb, ['CHAN_FREQ', 'CHAN_WIDTH', 'REF_FREQUENCY'])
    tb.close()
    ms['spw'] = {
        'chan_freq': np.asarray(spw['CHAN_FREQ'], dtype=np.float64).ravel(),
        'chan_width': np.asarray(spw['CHAN_WIDTH'], dtype=np.float64).ravel(),
        'ref_frequency': np.asarray(spw['REF_FREQUENCY'], dtype=np.float64).ravel(),
    }

    tb.open(str(path / 'POLARIZATION'))
    pol = _getcol_dict(tb, ['CORR_TYPE'])
    tb.close()
    ms['polarization'] = {
        'corr_type': np.asarray(pol['CORR_TYPE'], dtype=np.int32).ravel(),
    }

    tb.open(str(path / 'FIELD'))
    field = _getcol_dict(tb, ['NAME', 'PHASE_DIR', 'REFERENCE_DIR', 'DELAY_DIR', 'SOURCE_ID', 'TIME'])
    tb.close()
    ms['field'] = {
        'name': np.asarray(field['NAME']),
        'phase_dir': np.asarray(field['PHASE_DIR'], dtype=np.float64),
        'reference_dir': np.asarray(field['REFERENCE_DIR'], dtype=np.float64),
        'delay_dir': np.asarray(field['DELAY_DIR'], dtype=np.float64),
        'source_id': np.asarray(field['SOURCE_ID'], dtype=np.int32),
        'time': np.asarray(field['TIME'], dtype=np.float64),
    }

    source_path = path / 'SOURCE'
    if source_path.exists():
        tb.open(str(source_path))
        src = _getcol_dict(tb, ['NAME', 'DIRECTION', 'SOURCE_ID', 'TIME'])
        tb.close()
        ms['source'] = {
            'name': np.asarray(src['NAME']),
            'direction': np.asarray(src['DIRECTION'], dtype=np.float64),
            'source_id': np.asarray(src['SOURCE_ID'], dtype=np.int32),
            'time': np.asarray(src['TIME'], dtype=np.float64),
        }
    else:
        ms['source'] = None

    tb.open(str(path / 'OBSERVATION'))
    obs = _getcol_dict(tb, ['TIME_RANGE', 'OBSERVER', 'PROJECT', 'RELEASE_DATE', 'TELESCOPE_NAME'])
    tb.close()
    ms['observation'] = {
        'time_range': np.asarray(obs['TIME_RANGE'], dtype=np.float64),
        'observer': np.asarray(obs['OBSERVER']),
        'project': np.asarray(obs['PROJECT']),
        'release_date': np.asarray(obs['RELEASE_DATE'], dtype=np.float64),
        'telescope_name': np.asarray(obs['TELESCOPE_NAME']),
    }
    return ms


def _read_uvfits_an_table(path: Path) -> dict[str, Any]:
    with fits.open(path) as hdul:
        an_hdu = hdul['AIPS AN']
        hdr = an_hdu.header
        data = an_hdu.data
        center = np.array([hdr['ARRAYX'], hdr['ARRAYY'], hdr['ARRAYZ']], dtype=np.float64)
        stabxyz = np.asarray(data['STABXYZ'], dtype=np.float64)
        return {
            'array_center_xyz_m': center,
            'stabxyz_m': stabxyz,
            'absolute_xyz_m': stabxyz + center,
            'nosta': np.asarray(data['NOSTA'], dtype=np.int32),
            'anname': np.asarray(data['ANNAME']),
        }


def _phasecenter_summary(uv: UVData) -> dict[str, Any]:
    cat = getattr(uv, 'phase_center_catalog', {}) or {}
    entries: dict[str, Any] = {}
    for key, val in cat.items():
        entries[str(key)] = {
            'cat_name': val.get('cat_name'),
            'cat_type': val.get('cat_type'),
            'cat_lon': _safe_float(val.get('cat_lon')),
            'cat_lat': _safe_float(val.get('cat_lat')),
            'cat_frame': val.get('cat_frame'),
            'cat_epoch': _safe_float(val.get('cat_epoch')),
        }
    return {'count': len(entries), 'entries': entries}


def _ms_primary_field(ms: dict[str, Any]) -> dict[str, Any] | None:
    names = ms['field']['name']
    for idx in range(len(names) - 1, -1, -1):
        if str(names[idx]).strip():
            return {
                'index': idx,
                'name': str(names[idx]),
                'source_id': int(ms['field']['source_id'][idx]),
                'phase_dir_rad': ms['field']['phase_dir'][:, 0, idx].astype(float).tolist(),
                'reference_dir_rad': ms['field']['reference_dir'][:, 0, idx].astype(float).tolist(),
                'delay_dir_rad': ms['field']['delay_dir'][:, 0, idx].astype(float).tolist(),
                'time_sec': float(ms['field']['time'][idx]),
            }
    return None


def _ms_source_summary(ms: dict[str, Any]) -> dict[str, Any] | None:
    src = ms.get('source')
    if not src or src['name'].size == 0:
        return None
    return {
        'name': str(src['name'][0]),
        'source_id': int(src['source_id'][0]),
        'direction_rad': src['direction'][:, 0].astype(float).tolist(),
        'time_sec': float(src['time'][0]),
    }


def _compare_antenna_positions(uv: UVData, ms: dict[str, Any], uvfits_an: dict[str, Any]) -> dict[str, Any]:
    uv_pos = np.asarray(uv.telescope.antenna_positions, dtype=np.float64)
    ms_pos = np.asarray(ms['antenna']['position'], dtype=np.float64)
    delta = np.median(ms_pos - uv_pos, axis=0)
    residual = ms_pos - (uv_pos + delta)
    uvfits_abs = np.asarray(uvfits_an['absolute_xyz_m'], dtype=np.float64)
    ms_numeric_names = np.array([int(str(x)) for x in ms['antenna']['name']], dtype=np.int32)
    uvfits_nosta = np.asarray(uvfits_an['nosta'], dtype=np.int32)
    order_ok = np.array_equal(ms_numeric_names, uvfits_nosta)
    if order_ok:
        abs_residual = ms_pos - uvfits_abs
        abs_max = float(np.max(np.abs(abs_residual)))
        abs_rms = float(np.sqrt(np.mean(abs_residual ** 2)))
    else:
        abs_max = None
        abs_rms = None
    return {
        'uvfits_an_table_matches_ms_order': bool(order_ok),
        'raw_absolute_max_abs_m': abs_max,
        'raw_absolute_rms_m': abs_rms,
        'best_translation_m': delta.tolist(),
        'pyuvdata_relative_residual_max_abs_m': float(np.max(np.abs(residual))),
        'pyuvdata_relative_residual_rms_m': float(np.sqrt(np.mean(residual ** 2))),
    }


def _compare_phase_centers(uv: UVData, ms: dict[str, Any]) -> dict[str, Any]:
    uv_entry = list((getattr(uv, 'phase_center_catalog', {}) or {}).values())[0]
    uv_lon = float(uv_entry['cat_lon'])
    uv_lat = float(uv_entry['cat_lat'])
    ms_field = _ms_primary_field(ms)
    ms_source = _ms_source_summary(ms)
    rep: dict[str, Any] = {
        'uvfits': {
            'name': uv_entry.get('cat_name'),
            'lon_rad': uv_lon,
            'lat_rad': uv_lat,
            'frame': uv_entry.get('cat_frame'),
            'epoch': _safe_float(uv_entry.get('cat_epoch')),
        },
        'ms_field': ms_field,
        'ms_source': ms_source,
    }
    if ms_field is not None:
        fld_lon = float(ms_field['phase_dir_rad'][0])
        fld_lat = float(ms_field['phase_dir_rad'][1])
        rep['field_vs_uvfits'] = {
            'lon_diff_rad_wrapped': _angle_diff_rad(fld_lon, uv_lon),
            'lat_diff_rad': fld_lat - uv_lat,
        }
    if ms_source is not None:
        src_lon = float(ms_source['direction_rad'][0])
        src_lat = float(ms_source['direction_rad'][1])
        rep['source_vs_uvfits'] = {
            'lon_diff_rad_wrapped': _angle_diff_rad(src_lon, uv_lon),
            'lat_diff_rad': src_lat - uv_lat,
        }
    return rep


def _compare_uvfits_to_ms(uv: UVData, ms: dict[str, Any], uvfits_an: dict[str, Any]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    main = ms['main']

    report['uvfits_summary'] = {
        'Nblts': int(uv.Nblts),
        'Nfreqs': int(uv.Nfreqs),
        'Npols': int(uv.Npols),
        'Nants_telescope': int(uv.telescope.Nants),
        'telescope_name': uv.telescope.name,
    }
    report['ms_summary'] = {
        'Nblts': int(main['time_jd'].size),
        'Nfreqs': int(main['data'].shape[1]),
        'Npols': int(main['data'].shape[2]),
        'Nants_telescope': int(ms['antenna']['name'].size),
        'telescope_name': str(ms['observation']['telescope_name'][0]),
    }

    report['telescope_name_equal'] = bool(str(ms['observation']['telescope_name'][0]) == str(uv.telescope.name))
    report['antenna_station_names_equal'] = bool(np.array_equal(ms['antenna']['station'], np.asarray(uv.telescope.antenna_names)))
    ms_numeric_names = np.array([int(str(x)) for x in ms['antenna']['name']], dtype=np.int32)
    report['antenna_numbers_equal'] = bool(np.array_equal(ms_numeric_names, np.asarray(uv.telescope.antenna_numbers, dtype=np.int32)))
    report['antenna_position_fit'] = _compare_antenna_positions(uv, ms, uvfits_an)

    report['freq_array_equal'] = bool(np.array_equal(ms['spw']['chan_freq'], np.asarray(uv.freq_array, dtype=np.float64)))
    report['freq_max_abs_diff_hz'] = float(np.max(np.abs(ms['spw']['chan_freq'] - np.asarray(uv.freq_array, dtype=np.float64))))
    report['channel_width_equal'] = bool(np.array_equal(ms['spw']['chan_width'], np.asarray(uv.channel_width, dtype=np.float64)))
    report['channel_width_max_abs_diff_hz'] = float(np.max(np.abs(ms['spw']['chan_width'] - np.asarray(uv.channel_width, dtype=np.float64))))

    uv_pol_labels = [_pol_label_from_uvfits(x) for x in np.asarray(uv.polarization_array, dtype=np.int32)]
    ms_pol_labels = [_pol_label_from_ms(x) for x in ms['polarization']['corr_type']]
    report['polarization'] = {
        'uvfits_codes': np.asarray(uv.polarization_array, dtype=np.int32).tolist(),
        'uvfits_labels': uv_pol_labels,
        'ms_codes': ms['polarization']['corr_type'].tolist(),
        'ms_labels': ms_pol_labels,
        'labels_equal': bool(uv_pol_labels == ms_pol_labels),
    }

    report['time_alignment'] = {
        'max_abs_diff_sec': float(np.max(np.abs(main['time_jd'] - np.asarray(uv.time_array, dtype=np.float64))) * 86400.0),
        'exact_match': bool(np.array_equal(main['time_jd'], np.asarray(uv.time_array, dtype=np.float64))),
    }
    report['antenna_row_alignment'] = {
        'ant1_equal': bool(np.array_equal(main['ant1'], np.asarray(uv.ant_1_array, dtype=np.int32))),
        'ant2_equal': bool(np.array_equal(main['ant2'], np.asarray(uv.ant_2_array, dtype=np.int32))),
    }

    uv_uvw = np.asarray(uv.uvw_array, dtype=np.float64)
    ms_uvw = main['uvw']
    report['uvw_comparison'] = {
        'direct_max_abs_diff_m': float(np.max(np.abs(ms_uvw - uv_uvw))),
        'direct_rms_diff_m': float(np.sqrt(np.mean((ms_uvw - uv_uvw) ** 2))),
        'negated_max_abs_diff_m': float(np.max(np.abs(ms_uvw + uv_uvw))),
        'negated_rms_diff_m': float(np.sqrt(np.mean((ms_uvw + uv_uvw) ** 2))),
    }

    uv_data = np.asarray(uv.data_array)
    ms_data = main['data']
    report['visibility_comparison'] = {
        'direct_max_abs_diff': float(np.nanmax(np.abs(ms_data - uv_data))),
        'direct_median_abs_diff': float(np.nanmedian(np.abs(ms_data - uv_data))),
        'conjugated_max_abs_diff': float(np.nanmax(np.abs(ms_data - np.conj(uv_data)))),
        'conjugated_median_abs_diff': float(np.nanmedian(np.abs(ms_data - np.conj(uv_data)))),
    }

    report['flag_comparison'] = {
        'mismatch_count': int(np.count_nonzero(main['flag'] ^ np.asarray(uv.flag_array, dtype=bool))),
    }
    report['integration_time_comparison'] = {
        'interval_max_abs_diff_sec': float(np.max(np.abs(main['interval'] - np.asarray(uv.integration_time, dtype=np.float64)))),
        'exposure_max_abs_diff_sec': float(np.max(np.abs(main['exposure'] - np.asarray(uv.integration_time, dtype=np.float64)))),
    }
    report['phase_center'] = _compare_phase_centers(uv, ms)
    report['uvfits_phase_center_catalog'] = _phasecenter_summary(uv)
    return report


def _assessment(report: dict[str, Any], read_info: dict[str, Any]) -> list[str]:
    msgs: list[str] = []
    if read_info.get('used_minimal_fix'):
        msgs.append('original UVFITS was not directly readable by pyuvdata; a minimal compatibility patch was required')
    if report['freq_max_abs_diff_hz'] == 0.0:
        msgs.append('frequency grid matches exactly')
    if report['channel_width_max_abs_diff_hz'] == 0.0:
        msgs.append('channel widths match exactly')
    if report['time_alignment']['max_abs_diff_sec'] == 0.0:
        msgs.append('time stamps match exactly row-by-row')
    if report['antenna_row_alignment']['ant1_equal'] and report['antenna_row_alignment']['ant2_equal']:
        msgs.append('baseline row ordering matches exactly')

    uvw = report['uvw_comparison']
    if uvw['negated_max_abs_diff_m'] == 0.0:
        msgs.append('UVW coordinates match exactly after a global sign flip')
    elif uvw['direct_max_abs_diff_m'] == 0.0:
        msgs.append('UVW coordinates match exactly without sign change')

    vis = report['visibility_comparison']
    if vis['conjugated_max_abs_diff'] == 0.0:
        msgs.append('complex visibilities match exactly after complex conjugation')
    elif vis['direct_max_abs_diff'] == 0.0:
        msgs.append('complex visibilities match exactly without conjugation')

    if report['flag_comparison']['mismatch_count'] == 0:
        msgs.append('all flags match exactly')

    pf = report['phase_center'].get('field_vs_uvfits')
    ps = report['phase_center'].get('source_vs_uvfits')
    if pf and abs(pf['lon_diff_rad_wrapped']) < 1e-12 and abs(pf['lat_diff_rad']) < 1e-12:
        msgs.append('MS FIELD phase centre matches UVFITS source position modulo RA wrapping')
    if ps and abs(ps['lon_diff_rad_wrapped']) < 1e-12 and abs(ps['lat_diff_rad']) < 1e-12:
        msgs.append('MS SOURCE direction matches UVFITS source position modulo RA wrapping')

    antfit = report['antenna_position_fit']
    raw_abs = antfit.get('raw_absolute_max_abs_m')
    if raw_abs == 0.0:
        msgs.append('raw UVFITS AN-table antenna coordinates match MS antenna positions exactly')
    elif raw_abs is not None:
        msgs.append(f'raw UVFITS AN-table antenna coordinates differ from MS by max {raw_abs:.6g} m')

    msgs.append(
        'pyuvdata relative antenna positions are not directly comparable to MS absolute positions; '
        f"best-fit residual max = {antfit['pyuvdata_relative_residual_max_abs_m']:.6g} m"
    )
    return msgs


def main() -> int:
    args = _parse_args()
    uvfits_path = Path(args.uvfits).expanduser().resolve()
    ms_path = Path(args.ms).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    raw_header = _raw_uvfits_header_info(uvfits_path)
    uv, read_info = _read_uvdata_uvfits_with_minimal_fix(uvfits_path)
    ms = _read_ms_with_casatools(ms_path)
    uvfits_an = _read_uvfits_an_table(uvfits_path)
    comparison = _compare_uvfits_to_ms(uv, ms, uvfits_an)
    assessment = _assessment(comparison, read_info)

    payload = {
        'uvfits': str(uvfits_path),
        'ms': str(ms_path),
        'raw_uvfits_header': raw_header,
        'uvfits_reading': read_info,
        'comparison': comparison,
        'assessment': assessment,
    }

    json_path = outdir / f'{args.tag}_uvfits_ms_audit.json'
    txt_path = outdir / f'{args.tag}_uvfits_ms_audit.txt'
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        f'UVFITS: {uvfits_path}',
        f'MS: {ms_path}',
        '',
        'Raw UVFITS header values:',
    ]
    for k, v in raw_header.items():
        lines.append(f'  - {k}: {v}')

    lines += ['', 'UVFITS reading compatibility:']
    lines.append(f"  - used_minimal_fix: {read_info.get('used_minimal_fix')}")
    if read_info.get('read_error'):
        lines.append(f"  - original_read_error: {read_info['read_error']}")
    for patch in read_info.get('patches', []):
        lines.append(f"  - patched {patch['hdu']} {patch['key']}: {patch['old']} -> {patch['new']}")

    lines += ['', 'Assessment:']
    lines += [f'  - {msg}' for msg in assessment]
    txt_path.write_text('\n'.join(lines) + '\n')

    print(f'[audit] wrote {json_path}')
    print(f'[audit] wrote {txt_path}')
    for msg in assessment:
        print(f'[audit] {msg}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
