#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any, Iterable, cast

import numpy as np
from astropy.io import fits

from modules import cal_apply as ca
from modules import ugmrt_query as q
from modules.workflow_common import (
    apply_overrides_to_globals,
    bootstrap_config_from_cli,
    derive_index_cache,
)

log = logging.getLogger(__name__)

# Config-backed defaults
CAL_FITS: Path | None = None
INDEX_CACHE: Path | None = None
WORK_DIR: Path = Path('.')
SOURCE: str | None = None
CHAN_RANGE: tuple | None = None
STOKES: tuple | None = None
SOLVE_TIMERANGE: tuple | None = None
SOLVE_ELEVATION_MIN_DEG: float | None = None
SOLVE_ELEVATION_MAX_DEG: float | None = None
SOLVE_UVRANGE_M: tuple | None = None
SOLVE_UVRANGE_KLAMBDA: tuple | None = None
FLAG_TABLE_PATHS: list | None = None
INDEX_VALIDATION_MODE: str = 'fast'
LOG_LEVEL: str = 'INFO'


def _apply_overrides(overrides: list) -> None:
    apply_overrides_to_globals(overrides, globals(), log=log)


def _derive_index_cache(fits_path: Path, explicit_index_cache: Path | None) -> Path:
    if explicit_index_cache is not None:
        return Path(explicit_index_cache)
    return derive_index_cache(INDEX_CACHE, fits_path, WORK_DIR)


def _sanitize_tag(text: str) -> str:
    s = str(text).strip().lower()
    s = re.sub(r'[^a-z0-9._-]+', '_', s)
    s = re.sub(r'_+', '_', s).strip('_')
    return s or 'source'


def _parse_args(pre_args: argparse.Namespace) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Split a single source to calibrated/flagged UVFITS output')

    cfg = parser.add_argument_group('Config and Inputs')
    cfg.add_argument('--config', default=pre_args.config, metavar='FILE', help='Path to .cfg file (default: preprocess_ugmrt.cfg)')
    cfg.add_argument('--set', action='append', dest='set_overrides', metavar='KEY=expr', default=[],
                     help='Override any config key after loading: --set "KEY=expr"')

    inp = parser.add_argument_group('Selection')
    inp.add_argument('--fits', default=None, help='Input UVFITS path')
    inp.add_argument('--index-cache', default=None)
    inp.add_argument('--source', default=None, help='Source name or numeric source id')
    inp.add_argument('--chan-range', nargs=2, type=int, default=None, metavar=('START', 'END'))
    inp.add_argument('--stokes', nargs='*', default=None, help='Optional subset of stokes/correlations to keep')
    inp.add_argument('--time-range', nargs=2, default=None, metavar=('START', 'END'))
    inp.add_argument('--elevation-min', type=float, default=None)
    inp.add_argument('--elevation-max', type=float, default=None)
    inp.add_argument('--uvrange-m', nargs=2, type=float, default=None, metavar=('MIN_M', 'MAX_M'))
    inp.add_argument('--uvrange-klambda', nargs=2, type=float, default=None, metavar=('MIN_KL', 'MAX_KL'))

    cal = parser.add_argument_group('Flagging and Gain Application')
    cal.add_argument('--tables', nargs='*', default=None, help='Zero or more gain calibration tables to apply in order')
    cal.add_argument('--flag-tables', nargs='*', default=None, help='Zero or more flag-table paths to apply')
    cal.add_argument(
        '--time-interp-scheme',
        choices=ca.TIME_INTERP_SCHEMES,
        default='nearest',
        help='Time interpolation scheme for gain-table application only (used when --tables is given).',
    )
    cal.add_argument(
        '--time-extrapolation',
        choices=['hold', 'nearest', 'none'],
        default='hold',
        help='Time extrapolation mode for gain-table application only (used when --tables is given).',
    )

    out = parser.add_argument_group('Output')
    out.add_argument('--output-format', choices=['uvfits', 'ms'], default='uvfits', help='Currently uvfits is implemented; ms is reserved for future use')
    out.add_argument('--out', default=None, help='Output file path. Default derives from input/source and mode.')
    out.add_argument('--overwrite', action='store_true')
    out.add_argument(
        '--drop-zero-weight-rows',
        action='store_true',
        default=False,
        help=(
            'Remove rows whose weight is zero across ALL channels and polarisations '
            'before writing.  This is safe for AIPS and all UVFITS consumers: '
            'GCOUNT is updated to match, timestamps are preserved on surviving rows, '
            'and non-contiguous cadence is fully legal in the UVFITS standard.'
        ),
    )

    misc = parser.add_argument_group('Logging')
    misc.add_argument(
        '--log-level',
        default=None,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        metavar='LEVEL',
        help='Logging verbosity (default from config LOG_LEVEL or INFO).',
    )
    return parser.parse_args()


def _resolve_source_ids(index: dict, source: str | int) -> list[int]:
    if isinstance(source, int):
        return [int(source)]
    s = str(source).strip()
    if s.isdigit():
        return [int(s)]
    matched = [sid for sid, name in index['id_to_name'].items() if str(name).strip().lower() == s.lower()]
    if not matched:
        raise ValueError(f'Source {source!r} not found. Available: {sorted(index["id_to_name"].values())}')
    return [int(x) for x in matched]


def _select_source_rows(
    index: dict,
    *,
    source: str | int,
    timerange=None,
    uvrange_m=None,
    uvrange_klambda=None,
    elevation_min_deg=None,
    elevation_max_deg=None,
) -> np.ndarray:
    src_ids = _resolve_source_ids(index, source)
    row_mask = np.isin(np.asarray(index['source_id'], dtype=np.int32), np.asarray(src_ids, dtype=np.int32))
    if not np.any(row_mask):
        raise ValueError(f'No rows for source {source!r}.')

    rows = np.where(row_mask)[0]
    jd = np.asarray(index['jd'][rows], dtype=np.float64)
    uu = np.asarray(index['uu_sec'][rows], dtype=np.float64)
    vv = np.asarray(index['vv_sec'][rows], dtype=np.float64)
    keep = np.ones(rows.size, dtype=bool)

    if timerange is not None:
        from astropy.time import Time as _T
        t0, t1 = timerange
        if isinstance(t0, str):
            t0 = _T(t0, format='iso', scale='utc').jd
        if isinstance(t1, str):
            t1 = _T(t1, format='iso', scale='utc').jd
        t0f = float(np.asarray(t0, dtype=np.float64).reshape(-1)[0])
        t1f = float(np.asarray(t1, dtype=np.float64).reshape(-1)[0])
        keep &= (jd >= t0f) & (jd <= t1f)

    if uvrange_m is not None:
        uv_m = np.sqrt(uu**2 + vv**2) * 2.998e8
        keep &= (uv_m >= float(uvrange_m[0])) & (uv_m <= float(uvrange_m[1]))

    if uvrange_klambda is not None:
        freqs = np.asarray(index['chan_freqs_hz'], dtype=np.float64)
        ref_f = float(0.5 * (freqs[0] + freqs[-1]))
        uv_kl = np.sqrt(uu**2 + vv**2) * ref_f / 1e3
        keep &= (uv_kl >= float(uvrange_klambda[0])) & (uv_kl <= float(uvrange_klambda[1]))

    if elevation_min_deg is not None or elevation_max_deg is not None:
        from astropy.coordinates import AltAz, EarthLocation, SkyCoord
        from astropy.time import Time as _T
        import astropy.units as _u

        fits_path = index['path']
        geo = q._get_array_enu(fits_path)
        loc = EarthLocation.from_geocentric(*geo['ecef_centre'], unit=_u.m)
        with fits.open(str(fits_path), memmap=True, lazy_load_hdus=True) as hdul:
            su = cast(Any, q._get_hdu(hdul, 'AIPS SU'))
            if su is None:
                raise ValueError('No AIPS SU extension in FITS file.')
            names = set(su.columns.names)
            id_col = 'ID. NO.' if 'ID. NO.' in names else ('ID_NO.' if 'ID_NO.' in names else None)
            src_row = None
            for row in su.data:
                sid = int(row[id_col]) if id_col else None
                if sid in src_ids:
                    src_row = row
                    break
            if src_row is None:
                raise ValueError(f'Could not locate source row for {source!r} in AIPS SU table.')
            ra_deg = float(src_row['RAEPO'])
            dec_deg = float(src_row['DECEPO'])

        coord = SkyCoord(ra=ra_deg * _u.deg, dec=dec_deg * _u.deg, frame='icrs')
        times = _T(jd, format='jd', scale='utc')
        altaz = cast(Any, coord.transform_to(AltAz(obstime=times, location=loc)))
        el = np.asarray(altaz.alt.deg, dtype=np.float64)
        if elevation_min_deg is not None:
            keep &= el >= float(elevation_min_deg)
        if elevation_max_deg is not None:
            keep &= el <= float(elevation_max_deg)

    out = rows[keep]
    if out.size == 0:
        raise ValueError('Selection removed all rows; no data remains for output.')
    return np.asarray(out, dtype=np.int64)


def _read_selected_raw(index: dict, row_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pcount = int(index['pcount'])
    raw = np.memmap(index['path'], dtype='>f4', mode='r', offset=index['data_offset'], shape=(index['gcount'], index['group_size']))
    params = np.array(raw[row_indices, :pcount], dtype=np.float32)
    data_flat = np.array(raw[row_indices, pcount:], dtype=np.float32)
    del raw
    data = data_flat.reshape(
        row_indices.size,
        int(index['naxis5']),
        int(index['naxis4']),
        int(index['naxis3']),
        int(index['naxis2']),
    )
    return params, data


def _build_vis_from_rows(
    index: dict,
    row_indices: np.ndarray,
    data: np.ndarray,
    *,
    source_name: str,
    chan_range=None,
    stokes=None,
    flag_all_corrs_if_any_rawvis_flagged: bool = True,
) -> tuple[dict, np.ndarray, np.ndarray]:
    nchan_tot = int(index['naxis4'])
    all_stokes = list(index['stokes_labels'])

    if chan_range is not None:
        c0, c1 = chan_range
        chan_idx = np.arange(max(0, int(c0)), min(nchan_tot, int(c1) + 1), dtype=np.int32)
    else:
        chan_idx = np.arange(nchan_tot, dtype=np.int32)

    if stokes is not None:
        stokes_idx = []
        for s in stokes:
            if isinstance(s, int):
                stokes_idx.append(int(s))
            elif s in all_stokes:
                stokes_idx.append(all_stokes.index(s))
            else:
                raise ValueError(f'Stokes {s!r} not present. Available: {all_stokes}')
    else:
        stokes_idx = list(range(len(all_stokes)))

    stokes_sel = [all_stokes[i] for i in stokes_idx]
    freqs_sel = np.asarray(index['chan_freqs_hz'], dtype=np.float64)[chan_idx]

    data_sel = data[:, :, chan_idx][:, :, :, stokes_idx, :]
    if data_sel.shape[1] != 1:
        raise ValueError(f'visSplit currently expects NAXIS5/IF axis = 1, found {data_sel.shape[1]}.')
    data_sel = data_sel[:, 0, :, :, :]

    re_ = np.asarray(data_sel[..., 0], dtype=np.float32)
    im_ = np.asarray(data_sel[..., 1], dtype=np.float32)
    wt_ = np.asarray(data_sel[..., 2], dtype=np.float32)
    amp = np.sqrt(re_**2 + im_**2)
    phase = np.degrees(np.arctan2(im_, re_)).astype(np.float32)
    flagged = wt_ <= 0

    if flag_all_corrs_if_any_rawvis_flagged and flagged.ndim == 3 and flagged.shape[2] > 1:
        shared_flagged = np.any(flagged, axis=2, keepdims=True)
        flagged = np.broadcast_to(shared_flagged, flagged.shape).copy()
        wt_[flagged] = 0.0

    exact_zero = (re_ == 0.0) & (im_ == 0.0) & ~flagged
    if np.any(exact_zero):
        flagged = flagged | exact_zero
        wt_[flagged] = 0.0

    amp[flagged] = np.nan
    phase[flagged] = np.nan
    vis_complex = (re_.astype(np.float64) + 1j * im_.astype(np.float64)).astype(np.complex64)
    vis_complex[flagged] = np.nan + 0j

    uu = np.asarray(index['uu_sec'][row_indices], dtype=np.float32)
    vv = np.asarray(index['vv_sec'][row_indices], dtype=np.float32)
    jd = np.asarray(index['jd'][row_indices], dtype=np.float64)
    ref_freq = float(0.5 * (freqs_sel[0] + freqs_sel[-1]))
    uv_sec = np.sqrt(uu**2 + vv**2)
    uvdist_klambda = (uv_sec * ref_freq) / 1e3
    uvdist_per_chan = uv_sec[:, None] * freqs_sel[None, :] / 1e3

    vis = {
        '_row_indices_local': np.arange(row_indices.size, dtype=np.int64),
        'jd': jd,
        'ant1': np.asarray(index['ant1'][row_indices], dtype=int),
        'ant2': np.asarray(index['ant2'][row_indices], dtype=int),
        'source_name': str(source_name),
        'uu_sec': uu,
        'vv_sec': vv,
        'uvdist_klambda': uvdist_klambda,
        'uvdist_per_chan': uvdist_per_chan,
        'freqs_hz': freqs_sel,
        'vis_complex': vis_complex,
        'amp': amp,
        'phase_deg': phase,
        'weight': wt_,
        'flagged': flagged,
        'stokes_labels': stokes_sel,
        'chan_indices': chan_idx,
        'nrows': int(row_indices.size),
    }
    return vis, chan_idx, np.asarray(stokes_idx, dtype=np.int32)


def _load_solutions(paths: Iterable[str | Path]) -> list[dict]:
    out: list[dict] = []
    for p in paths:
        rp = Path(p).expanduser().resolve()
        if not rp.exists():
            raise FileNotFoundError(f'Calibration table not found: {rp}')
        sol = q.load_bandpass_solution(rp)
        sol['_path'] = str(rp)
        out.append(sol)
    return out


def _apply_solutions(vis: dict, solutions: list[dict], *, scheme: str, extrapolation: str) -> dict:
    if not solutions:
        return {
            'vis_complex_corrected': np.asarray(vis['vis_complex'], dtype=np.complex128),
            'flagged_corrected': np.asarray(vis.get('flagged', np.zeros_like(vis['weight'], dtype=bool)), dtype=bool),
        }

    groups = ca._group_tables_in_order(solutions)
    corrected = np.asarray(vis['vis_complex'], dtype=np.complex128).copy()
    corrected_flagged = np.asarray(vis.get('flagged', np.zeros_like(vis['weight'], dtype=bool)), dtype=bool).copy()
    vis_labels = [str(x) for x in vis['stokes_labels']]

    for group in groups:
        ca._check_timed_tables(group, strict=False)
        corrected, corrected_flagged = ca._apply_group_inplace(
            corrected,
            corrected_flagged,
            jd=np.asarray(vis['jd'], dtype=np.float64),
            ant1=np.asarray(vis['ant1'], dtype=np.int32),
            ant2=np.asarray(vis['ant2'], dtype=np.int32),
            vis_chan_indices=np.asarray(vis['chan_indices'], dtype=np.int32),
            vis_labels=vis_labels,
            group=group,
            scheme=scheme,
            extrapolation=extrapolation,
        )

    return {
        'vis_complex_corrected': corrected,
        'flagged_corrected': corrected_flagged,
    }


def _derive_output_path(
    fits_path: Path,
    *,
    source_name: str,
    output_format: str,
    tables: list[dict],
    flag_paths: list[str | Path],
    explicit_out: str | None,
) -> Path:
    if explicit_out:
        return Path(explicit_out).expanduser().resolve()
    stem = fits_path.name
    if stem.upper().endswith('.FITS'):
        stem = stem[:-5]
    tag = _sanitize_tag(source_name)
    parts = [stem, tag]
    if flag_paths:
        parts.append('flag')
    if tables:
        parts.append('cal')
    else:
        parts.append('split')
    ext = '.FITS' if output_format == 'uvfits' else '.ms'
    return fits_path.with_name('.'.join(parts) + ext)


def _filter_su_hdu(hdu, source_ids: list[int], n_if: int):
    data = hdu.data
    if data is None:
        return hdu.copy()
    cols = set(data.dtype.names)
    id_col = 'ID. NO.' if 'ID. NO.' in cols else ('ID_NO.' if 'ID_NO.' in cols else None)
    if id_col is None:
        return hdu.copy()
    mask = np.isin(np.asarray(data[id_col], dtype=np.int32), np.asarray(source_ids, dtype=np.int32))
    su_data = data[mask]

    # Columns with one element per IF in AIPS SU tables.
    if_cols = {'IFLUX', 'QFLUX', 'UFLUX', 'VFLUX', 'FREQOFF', 'BANDWIDTH', 'LSRVEL', 'RESTFREQ'}
    out_cols = []
    for col in hdu.columns:
        name = col.name
        arr = np.asarray(su_data[name])
        fmt = str(col.format)

        if name in if_cols and arr.ndim == 2 and arr.shape[1] != int(n_if):
            trimmed = arr[:, : int(n_if)]
            m = re.match(r'^\s*(\d+)([A-Za-z])\s*$', fmt)
            if m:
                fmt = f'{int(n_if)}{m.group(2)}'
            out_cols.append(fits.Column(name=name, format=fmt, unit=col.unit, array=trimmed))
        else:
            out_cols.append(fits.Column(name=name, format=fmt, unit=col.unit, array=arr))

    out_hdu = fits.BinTableHDU.from_columns(out_cols, header=hdu.header.copy(), name=hdu.name)
    out_hdu.header['NO_IF'] = int(n_if)
    if 'RADECSYS' not in out_hdu.header or not str(out_hdu.header.get('RADECSYS', '')).strip():
        out_hdu.header['RADECSYS'] = 'FK5'
    try:
        if 'EPOCH' in (su_data.dtype.names or []):
            out_hdu.header['EQUINOX'] = float(np.asarray(su_data['EPOCH'])[0])
        elif 'EQUINOX' not in out_hdu.header:
            out_hdu.header['EQUINOX'] = 2000.0
    except Exception:
        out_hdu.header['EQUINOX'] = 2000.0
    return out_hdu


def _extract_source_radec_deg(ext_hdus: list[Any]) -> tuple[float | None, float | None]:
    for hdu in ext_hdus:
        if getattr(hdu, 'name', '') != 'AIPS SU':
            continue
        data = getattr(hdu, 'data', None)
        if data is None or len(data) == 0:
            continue
        names = set(data.dtype.names or [])
        ra_col = 'RAAPP' if 'RAAPP' in names else ('RAEPO' if 'RAEPO' in names else None)
        dec_col = 'DECAPP' if 'DECAPP' in names else ('DECEPO' if 'DECEPO' in names else None)
        if ra_col is None or dec_col is None:
            continue
        try:
            ra_deg = float(np.asarray(data[ra_col])[0])
            dec_deg = float(np.asarray(data[dec_col])[0])
            return ra_deg, dec_deg
        except Exception:
            continue
    return None, None


def _write_uvfits(
    *,
    fits_path: Path,
    index: dict,
    row_indices: np.ndarray,
    chan_idx: np.ndarray,
    params: np.ndarray,
    data_out: np.ndarray,
    source_name: str,
    source_ids: list[int],
    out_path: Path,
    overwrite: bool,
    notes: list[str],
) -> Path:
    with fits.open(str(fits_path), memmap=True, lazy_load_hdus=True) as hdul:
        primary_in = cast(Any, hdul[0])
        orig_header = primary_in.header.copy()
        ext_hdus = []
        for hdu in hdul[1:]:
            hdu = cast(Any, hdu)
            if hdu.name == 'AIPS SU':
                ext_hdus.append(_filter_su_hdu(hdu, source_ids, int(data_out.shape[-4])))
            else:
                ext_hdus.append(hdu.copy())

    parnames = [str(orig_header.get(f'PTYPE{i+1}', f'PAR{i+1}')).strip() for i in range(int(index['pcount']))]
    pardata = [np.asarray(params[:, i], dtype=np.float32) for i in range(int(index['pcount']))]
    # CASA importuvfits requires explicit RA/DEC axes in random-groups UVFITS.
    # Add two singleton axes before IF so written FITS has NAXIS6=1 and NAXIS7=1.
    data_write = data_out.astype(np.float32)[:, np.newaxis, np.newaxis, ...]
    group_data = fits.GroupData(data_write, parnames=parnames, pardata=pardata, bitpix=-32)
    primary = fits.GroupsHDU(data=group_data, header=orig_header)
    ra_deg, dec_deg = _extract_source_radec_deg(ext_hdus)
    primary.header['OBJECT'] = str(source_name)
    primary.header['GCOUNT'] = int(row_indices.size)
    primary.header['NAXIS2'] = int(data_write.shape[-1])
    primary.header['NAXIS3'] = int(data_write.shape[-2])
    primary.header['NAXIS4'] = int(data_write.shape[-3])
    primary.header['NAXIS5'] = int(data_write.shape[-4])
    primary.header['NAXIS6'] = int(data_write.shape[-5])
    primary.header['NAXIS7'] = int(data_write.shape[-6])
    primary.header['CTYPE6'] = 'RA'
    primary.header['CRPIX6'] = 1.0
    primary.header['CDELT6'] = 1.0
    primary.header['CROTA6'] = 0.0
    primary.header['CUNIT6'] = 'deg'
    primary.header['CTYPE7'] = 'DEC'
    primary.header['CRPIX7'] = 1.0
    primary.header['CDELT7'] = 1.0
    primary.header['CROTA7'] = 0.0
    primary.header['CUNIT7'] = 'deg'
    try:
        primary.header['EQUINOX'] = float(orig_header.get('EQUINOX', 2000.0))
    except Exception:
        primary.header['EQUINOX'] = 2000.0
    primary.header['RADESYS'] = str(orig_header.get('RADESYS', 'FK5'))
    # VELDEF is required by CASA importuvfits to resolve the FREQ axis frame.
    # Propagate from original if present; default to RADIO TOPO (topocentric,
    # standard for correlator/dump output) if absent.
    primary.header['VELDEF'] = str(orig_header.get('VELDEF', 'RADIO TOPO'))
    if ra_deg is not None and dec_deg is not None:
        primary.header['CRVAL6'] = ra_deg
        primary.header['CRVAL7'] = dec_deg
        primary.header['OBSRA'] = ra_deg
        primary.header['OBSDEC'] = dec_deg
    else:
        try:
            primary.header['CRVAL6'] = float(orig_header.get('CRVAL6', 0.0))
        except Exception:
            primary.header['CRVAL6'] = 0.0
        try:
            primary.header['CRVAL7'] = float(orig_header.get('CRVAL7', 0.0))
        except Exception:
            primary.header['CRVAL7'] = 0.0
    if chan_idx.size > 0:
        try:
            crval4 = float(orig_header.get('CRVAL4', 0.0))
            cdelt4 = float(orig_header.get('CDELT4', 1.0))
            primary.header['CRVAL4'] = crval4 + float(int(chan_idx[0])) * cdelt4
        except Exception:
            pass
    for note in notes:
        primary.header.add_history(note)

    hdul_out = fits.HDUList([primary] + ext_hdus)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hdul_out.writeto(out_path, overwrite=overwrite)
    return out_path


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    default_cfg = str(script_dir.parent / 'preprocess_ugmrt.cfg')
    pre_args = bootstrap_config_from_cli(default_cfg, globals(), log=log)
    args = _parse_args(pre_args)
    _apply_overrides(args.set_overrides)

    effective_level = args.log_level or globals().get('LOG_LEVEL', 'INFO')
    logging.basicConfig(
        level=getattr(logging, str(effective_level).upper(), logging.INFO),
        format='%(asctime)s  %(levelname)-8s  %(message)s',
    )

    fits_path = Path(args.fits) if args.fits is not None else (Path(CAL_FITS) if CAL_FITS is not None else None)
    if fits_path is None:
        sys.exit('ERROR: no input vis provided. Pass --fits or set CAL_FITS in config.')
    if not fits_path.exists():
        sys.exit(f'ERROR: input vis not found: {fits_path}')

    source = args.source if args.source is not None else globals().get('SOURCE', None)
    if source is None:
        sys.exit('ERROR: no source provided. Pass --source or set SOURCE in config.')

    if args.output_format == 'ms':
        sys.exit('ERROR: output-format=ms is not implemented yet. Use --output-format uvfits for now.')

    index_cache = _derive_index_cache(fits_path=fits_path, explicit_index_cache=(Path(args.index_cache) if args.index_cache else None))
    index = q.get_or_build_row_index(
        fits_path,
        cache_path=index_cache,
        force_rebuild=False,
        validation_mode=INDEX_VALIDATION_MODE,
        write_cache=True,
    )

    chan_range = tuple(args.chan_range) if args.chan_range is not None else tuple(CHAN_RANGE) if CHAN_RANGE is not None else None
    stokes = list(args.stokes) if args.stokes else list(STOKES) if STOKES else None
    timerange = tuple(args.time_range) if args.time_range is not None else globals().get('SOLVE_TIMERANGE', None)
    elevation_min = float(args.elevation_min) if args.elevation_min is not None else globals().get('SOLVE_ELEVATION_MIN_DEG', None)
    elevation_max = float(args.elevation_max) if args.elevation_max is not None else globals().get('SOLVE_ELEVATION_MAX_DEG', None)
    uvrange_m = tuple(args.uvrange_m) if args.uvrange_m is not None else globals().get('SOLVE_UVRANGE_M', None)
    uvrange_kl = tuple(args.uvrange_klambda) if args.uvrange_klambda is not None else globals().get('SOLVE_UVRANGE_KLAMBDA', None)

    row_indices = _select_source_rows(
        index,
        source=source,
        timerange=timerange,
        uvrange_m=uvrange_m,
        uvrange_klambda=uvrange_kl,
        elevation_min_deg=elevation_min,
        elevation_max_deg=elevation_max,
    )
    src_ids = _resolve_source_ids(index, source)
    source_name = index['id_to_name'].get(src_ids[0], str(source))

    params, data = _read_selected_raw(index, row_indices)
    vis, chan_idx, stokes_idx = _build_vis_from_rows(
        index,
        row_indices,
        data,
        source_name=source_name,
        chan_range=chan_range,
        stokes=stokes,
        flag_all_corrs_if_any_rawvis_flagged=True,
    )

    antenna_name_map = {
        int(a['antenna_no']): str(a['name'])
        for a in index.get('antennas', [])
        if a.get('antenna_no') is not None
    }

    flag_paths = list(args.flag_tables) if args.flag_tables is not None else list(globals().get('FLAG_TABLE_PATHS', []) or [])
    if flag_paths:
        vis, flag_stats = q.apply_flag_tables_to_vis(
            vis,
            antenna_name_map=antenna_name_map,
            flag_table_paths=flag_paths,
            strict=False,
            context='visSplit',
        )
        kept_local = np.asarray(vis['_row_indices_local'], dtype=np.int64)
        row_indices = row_indices[kept_local]
        params = params[kept_local]
        data = data[kept_local]
        log.info('[visSplit] flag tables applied: dropped_rows=%d kept_rows=%d zeroed_cells=%d tables=%d',
                 int(flag_stats.get('dropped_rows', 0)),
                 int(flag_stats.get('kept_rows', 0)),
                 int(flag_stats.get('zeroed_cells_by_timerange_flags', 0)),
                 int(flag_stats.get('flag_table_count', 0)))
    else:
        flag_stats = {'flag_table_count': 0, 'dropped_rows': 0, 'kept_rows': int(vis['nrows'])}

    table_paths = list(args.tables or [])
    solutions = _load_solutions(table_paths) if table_paths else []
    corr = _apply_solutions(
        vis,
        solutions,
        scheme=str(args.time_interp_scheme),
        extrapolation=str(args.time_extrapolation),
    )

    if solutions:
        out_complex = np.asarray(corr['vis_complex_corrected'], dtype=np.complex128)
        out_flagged = np.asarray(corr['flagged_corrected'], dtype=bool)
    else:
        out_complex = np.asarray(vis['vis_complex'], dtype=np.complex128)
        out_flagged = np.asarray(vis.get('flagged', np.zeros_like(vis['weight'], dtype=bool)), dtype=bool)

    out_weight = np.asarray(vis['weight'], dtype=np.float32).copy()
    out_weight[out_flagged] = 0.0
    out_re = np.where(out_flagged, 0.0, np.real(out_complex)).astype(np.float32)
    out_im = np.where(out_flagged, 0.0, np.imag(out_complex)).astype(np.float32)

    data_out = np.array(data[:, :, chan_idx][:, :, :, stokes_idx, :], dtype=np.float32, copy=True)
    data_out[:, 0, :, :, 0] = out_re
    data_out[:, 0, :, :, 1] = out_im
    if data_out.shape[-1] > 2:
        data_out[:, 0, :, :, 2] = out_weight

    if args.drop_zero_weight_rows:
        # A row is fully flagged when every (chan, pol) weight is zero.
        # out_weight shape: (nrows, nchans, npols) after the vis arrays;
        # use data_out weight plane if present (index 2 of last axis), else out_weight.
        if data_out.shape[-1] > 2:
            weight_plane = data_out[:, 0, :, :, 2]  # (nrows, nchans, npols)
        else:
            weight_plane = out_weight
        zero_rows_mask = np.all(weight_plane == 0.0, axis=tuple(range(1, weight_plane.ndim)))
        keep_mask = ~zero_rows_mask
        n_dropped = int(zero_rows_mask.sum())
        if n_dropped > 0:
            data_out    = data_out[keep_mask]
            params      = params[keep_mask]
            row_indices = row_indices[keep_mask]
            log.info('[visSplit] --drop-zero-weight-rows: dropped %d all-zero-weight rows, %d rows remaining',
                     n_dropped, int(keep_mask.sum()))
        else:
            log.info('[visSplit] --drop-zero-weight-rows: no all-zero-weight rows found')

    out_path = _derive_output_path(
        fits_path,
        source_name=source_name,
        output_format=args.output_format,
        tables=solutions,
        flag_paths=flag_paths,
        explicit_out=args.out,
    )

    notes = [
        f'visSplit source={source_name}',
        f'rows={int(row_indices.size)} chan_range={tuple(chan_idx[[0, -1]].tolist()) if chan_idx.size else None}',
        f'flag_tables={len(flag_paths)} cal_tables={len(solutions)}',
    ]
    if solutions:
        notes.append(f'time_interp_scheme={args.time_interp_scheme} time_extrapolation={args.time_extrapolation}')

    _write_uvfits(
        fits_path=fits_path,
        index=index,
        row_indices=row_indices,
        chan_idx=chan_idx,
        params=params,
        data_out=data_out,
        source_name=source_name,
        source_ids=src_ids,
        out_path=out_path,
        overwrite=bool(args.overwrite),
        notes=notes,
    )

    print(f'[visSplit] source={source_name} rows_out={row_indices.size} flag_tables={len(flag_paths)} cal_tables={len(solutions)}')
    print(f'[visSplit] wrote {out_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
