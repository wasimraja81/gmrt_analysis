"""
uGMRT UVFITS query and plotting utilities.

Lightweight memory-mapped I/O functions for uGMRT UVFITS files.
Designed for low-RAM environments (Raspberry Pi 4, 8 GB).
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from astropy.io import fits
from astropy.time import Time

# ---------------------------------------------------------------------------
# File discovery & core I/O utilities
# ---------------------------------------------------------------------------

def find_fits_files(base_dir: Union[str, Path]) -> List[Path]:
    base_dir = Path(base_dir)
    return sorted(base_dir.rglob('*.FITS'))

def open_uvfits(path: Union[str, Path]):
    path = Path(path)
    return fits.open(path, memmap=True, lazy_load_hdus=True)

def _get_hdu(hdul, extname: str):
    for hdu in hdul:
        if hdu.name == extname:
            return hdu
    return None

def _decode_bytes(value):
    if isinstance(value, (bytes, bytearray)):
        return value.decode('ascii', errors='ignore').strip()
    return str(value).strip()

# ---------------------------------------------------------------------------
# Header metadata inspector
# ---------------------------------------------------------------------------

def get_key_header_properties(path: Union[str, Path]) -> Dict[str, Union[str, int, float]]:
    with open_uvfits(path) as hdul:
        primary = hdul[0]
        h = primary.header

        an = _get_hdu(hdul, 'AIPS AN')
        fq = _get_hdu(hdul, 'AIPS FQ')
        su = _get_hdu(hdul, 'AIPS SU')

        out = {
            'file': str(path),
            'object': h.get('OBJECT'),
            'project': h.get('PROJECT'),
            'telescope': h.get('TELESCOP'),
            'instrument': h.get('INSTRUME'),
            'observer': h.get('OBSERVER'),
            'date_obs': h.get('DATE-OBS'),
            'backend_hint': None,

            # PRIMARY (GroupsHDU) data axes
            'naxis3_stokes': int(h.get('NAXIS3', -1)),
            'naxis4_channels': int(h.get('NAXIS4', -1)),
            'gcount_nvis_groups': int(h.get('GCOUNT', -1)),
            'pcount_group_params': int(h.get('PCOUNT', -1)),

            # Frequency axis from PRIMARY header
            'ref_freq_hz': float(h.get('CRVAL4', np.nan)),
            'chan_width_hz': float(h.get('CDELT4', np.nan)),
            'estimated_bw_hz': abs(float(h.get('CDELT4', np.nan)) * int(h.get('NAXIS4', 0))),

            # Table row counts from extension HDUs
            'n_antennas_aips_an': int(len(an.data)) if an is not None and an.data is not None else 0,
            'n_freqset_aips_fq': int(len(fq.data)) if fq is not None and fq.data is not None else 0,
            'n_sources_aips_su': int(len(su.data)) if su is not None and su.data is not None else 0,
        }

        hist = h.get('HISTORY', [])
        if isinstance(hist, str):
            hist = [hist]
        for rec in hist:
            recs = str(rec)
            if 'GMRT CORR Version' in recs:
                out['backend_hint'] = recs.strip()
                break

        return out

# ---------------------------------------------------------------------------
# Antenna information query
# ---------------------------------------------------------------------------

def list_antennas(path: Union[str, Path]) -> List[Dict[str, Union[str, int, float]]]:
    with open_uvfits(path) as hdul:
        an = _get_hdu(hdul, 'AIPS AN')
        if an is None:
            return []

        rows = []
        for r in an.data:
            xyz = r['STABXYZ'] if 'STABXYZ' in an.columns.names else [np.nan, np.nan, np.nan]
            rows.append({
                'antenna_no': int(r['NOSTA']) if 'NOSTA' in an.columns.names else None,
                'name': _decode_bytes(r['ANNAME']) if 'ANNAME' in an.columns.names else None,
                'x_m': float(xyz[0]),
                'y_m': float(xyz[1]),
                'z_m': float(xyz[2]),
            })
        return rows

# ---------------------------------------------------------------------------
# Frequency and bandwidth query
# ---------------------------------------------------------------------------

def list_frequency_properties(path: Union[str, Path]) -> Dict[str, Union[float, int, List[float]]]:
    with open_uvfits(path) as hdul:
        h = hdul[0].header
        nchan = int(h.get('NAXIS4', 0))
        crval4 = float(h.get('CRVAL4', np.nan))
        cdelt4 = float(h.get('CDELT4', np.nan))
        crpix4 = float(h.get('CRPIX4', 1.0))

        chan_idx = np.arange(1, nchan + 1, dtype=np.float64)
        chan_freq = crval4 + (chan_idx - crpix4) * cdelt4

        fq = _get_hdu(hdul, 'AIPS FQ')
        fq_info = {}
        if fq is not None and len(fq.data) > 0:
            row0 = fq.data[0]
            names = set(fq.columns.names)
            if 'TOTAL BANDWIDTH' in names:
                fq_info['total_bandwidth_hz'] = float(np.atleast_1d(row0['TOTAL BANDWIDTH'])[0])
            if 'CH WIDTH' in names:
                fq_info['ch_width_hz_from_fq'] = float(np.atleast_1d(row0['CH WIDTH'])[0])
            if 'IF FREQ' in names:
                fq_info['if_freq_hz'] = np.atleast_1d(row0['IF FREQ']).astype(float).tolist()

        out = {
            'nchan': nchan,
            'ref_freq_hz': crval4,
            'chan_width_hz': cdelt4,
            'bandwidth_hz_estimated': abs(cdelt4) * nchan,
            'freq_min_hz': float(np.nanmin(chan_freq)) if nchan > 0 else np.nan,
            'freq_max_hz': float(np.nanmax(chan_freq)) if nchan > 0 else np.nan,
            'first_8_channel_freq_hz': chan_freq[:8].astype(float).tolist(),
            'last_8_channel_freq_hz': chan_freq[-8:].astype(float).tolist(),
        }
        out.update(fq_info)
        return out

# ---------------------------------------------------------------------------
# Source information and filtering
# ---------------------------------------------------------------------------

def list_sources(path: Union[str, Path]) -> List[Dict[str, Union[int, str, float]]]:
    with open_uvfits(path) as hdul:
        su = _get_hdu(hdul, 'AIPS SU')
        if su is None:
            return []

        names = set(su.columns.names)
        id_col = 'ID. NO.' if 'ID. NO.' in names else ('ID_NO.' if 'ID_NO.' in names else None)
        name_col = 'SOURCE' if 'SOURCE' in names else None

        rows = []
        for r in su.data:
            item = {
                'source_id': int(r[id_col]) if id_col else None,
                'source_name': _decode_bytes(r[name_col]) if name_col else None,
            }
            if 'RAEPO' in names:
                item['ra_deg'] = float(r['RAEPO'])
            if 'DECEPO' in names:
                item['dec_deg'] = float(r['DECEPO'])
            rows.append(item)

        rows.sort(key=lambda x: (x['source_id'] is None, x['source_id']))
        return rows

def filter_sources(path: Union[str, Path], include: Optional[List[str]] = None, regex: Optional[str] = None) -> List[Dict[str, Union[int, str, float]]]:
    import re

    src = list_sources(path)
    if include:
        include_l = {s.lower() for s in include}
        src = [s for s in src if (s['source_name'] or '').lower() in include_l]
    if regex:
        pattern = re.compile(regex)
        src = [s for s in src if pattern.search(s['source_name'] or '')]
    return src

# ---------------------------------------------------------------------------
# Row index builder and source analysis (index-based, fast I/O)
# ---------------------------------------------------------------------------

_STOKES_MAP = {
    1: 'I', 2: 'Q', 3: 'U', 4: 'V',
    -1: 'RR', -2: 'LL', -3: 'RL', -4: 'LR',
    -5: 'XX', -6: 'YY', -7: 'XY', -8: 'YX',
}

def _decode_baseline_array(bl_arr):
    """Decode UVFITS baseline encoding to (ant1, ant2) arrays."""
    bl = np.rint(np.asarray(bl_arr, dtype=np.float64)).astype(np.int64)
    ext = bl > 65536
    ant1 = np.where(ext, (bl - 65536) // 2048, bl // 256)
    ant2 = np.where(ext, (bl - 65536) % 2048, bl % 256)
    return ant1.astype(np.int16), ant2.astype(np.int16)


def build_row_index(
    path: Union[str, Path],
    override_dud_names: Optional[List[str]] = None,
) -> dict:
    """Build a fast row index by reading group parameters in one sequential pass.

    Parameters
    ----------
    override_dud_names : list[str], optional
        Explicit list of antenna *names* (as they appear in the AIPS AN table,
        e.g. ``['C07', 'S05']``) that should always be treated as dud antennas,
        **bypassing** automatic STABXYZ-based detection entirely.  Use this when
        you know from domain knowledge or prior observation which antennas are
        non-functional.  If ``None`` (the default) the automatic detector runs.
    """
    import time as _time
    t0 = _time.monotonic()
    path = Path(path)

    with fits.open(path, memmap=True, lazy_load_hdus=True) as hdul:
        h = hdul[0].header
        fi = hdul.fileinfo(0)
        data_offset = fi['datLoc']

        gcount = int(h['GCOUNT'])
        pcount = int(h['PCOUNT'])
        naxis2 = int(h.get('NAXIS2', 3))
        naxis3 = int(h.get('NAXIS3', 1))
        naxis4 = int(h.get('NAXIS4', 1))
        naxis5 = int(h.get('NAXIS5', 1))

        parnames = [h.get(f'PTYPE{i+1}', '').strip().upper()
                     for i in range(pcount)]

        crval4 = float(h.get('CRVAL4', 0.0))
        cdelt4 = float(h.get('CDELT4', 1.0))
        crpix4 = float(h.get('CRPIX4', 1.0))
        chan_freqs = crval4 + (np.arange(1, naxis4 + 1, dtype=np.float64)
                               - crpix4) * cdelt4

        crval3 = int(round(float(h.get('CRVAL3', -5))))
        cdelt3 = int(round(float(h.get('CDELT3', -1))))
        stokes_labels = [
            _STOKES_MAP.get(crval3 + i * cdelt3, f'S{crval3 + i * cdelt3}')
            for i in range(naxis3)
        ]

        su = _get_hdu(hdul, 'AIPS SU')
        id_to_name = {}
        if su is not None:
            cols = set(su.columns.names)
            id_col = ('ID. NO.' if 'ID. NO.' in cols
                      else ('ID_NO.' if 'ID_NO.' in cols else None))
            if id_col and 'SOURCE' in cols:
                for r in su.data:
                    id_to_name[int(r[id_col])] = _decode_bytes(r['SOURCE'])

        antennas = list_antennas(path)
        freq = list_frequency_properties(path)

    # ── Dud-antenna resolution ─────────────────────────────────────────────────
    # Caller must supply override_dud_names (set DUD_ANTENNA_NAMES in the cfg /
    # notebook configuration cell).  Prefix matching is used: 'C07' matches
    # the AN-table entry 'C07:31', 'S05' matches 'S05:32', etc.
    # If override_dud_names is None or empty, no antennas are excluded.
    _geo_local = _get_array_enu(path)
    _dud_nos, _dud_names = [], []
    if override_dud_names:
        _name_to_no = {nm: no for no, nm in zip(_geo_local['nos'], _geo_local['names'])}
        for _wanted in override_dud_names:
            _hit = next(
                (no for nm, no in _name_to_no.items()
                 if nm == _wanted or nm.startswith(_wanted + ':') or nm.startswith(_wanted)),
                None,
            )
            if _hit is not None:
                _full_name = _geo_local['names'][_geo_local['nos'].index(_hit)]
                _dud_nos.append(_hit)
                _dud_names.append(_full_name)
            else:
                print(f'  WARNING: override_dud_names entry "{_wanted}" not found in AN table; skipped.')
        _dud_nos  = sorted(_dud_nos)
        _dud_names = [_geo_local['names'][_geo_local['nos'].index(n)] for n in _dud_nos]
    if _dud_nos:
        print(f'  Dud antennas (excluded): {list(zip(_dud_names, _dud_nos))}')
    _dud_set = set(_dud_nos)
    active_antennas = [a for a in antennas if a['antenna_no'] not in _dud_set]

    data_per_group = naxis5 * naxis4 * naxis3 * naxis2
    group_size = pcount + data_per_group

    raw = np.memmap(path, dtype='>f4', mode='r',
                    offset=data_offset,
                    shape=(gcount, group_size))
    params = np.array(raw[:, :pcount], dtype=np.float64)
    del raw

    norm = [n.replace('_', '').replace('-', '') for n in parnames]
    src_col   = next(i for i, n in enumerate(norm) if n == 'SOURCE')
    bl_col    = next(i for i, n in enumerate(norm) if n == 'BASELINE')
    date_cols = [i for i, n in enumerate(norm) if n == 'DATE']

    source_id = params[:, src_col].astype(np.int32)

    arr1 = params[:, date_cols[0]]
    if len(date_cols) > 1:
        arr2 = params[:, date_cols[1]]
        if np.allclose(arr1, arr2, rtol=0.0, atol=1e-9):
            arr2 = np.zeros_like(arr1)
    else:
        arr2 = np.zeros_like(arr1)
    jd = arr1 + arr2

    ant1, ant2 = _decode_baseline_array(params[:, bl_col])

    uu_col = next(i for i, n in enumerate(norm) if n.startswith('UU'))
    vv_col = next(i for i, n in enumerate(norm) if n.startswith('VV'))
    uu_sec = params[:, uu_col].astype(np.float32)
    vv_sec = params[:, vv_col].astype(np.float32)
    del params

    source_ranges = {}
    for sid in np.unique(source_id):
        indices = np.where(source_id == sid)[0]
        breaks = np.where(np.diff(indices) > 1)[0] + 1
        runs = np.split(indices, breaks)
        source_ranges[int(sid)] = [(int(r[0]), int(r[-1]) + 1) for r in runs]

    elapsed = _time.monotonic() - t0

    idx = {
        'source_id': source_id, 'jd': jd,
        'ant1': ant1, 'ant2': ant2,
        'uu_sec': uu_sec, 'vv_sec': vv_sec,
        'source_ranges': source_ranges,
        'id_to_name': id_to_name,
        'antennas': antennas, 'freq': freq,
        'dud_antenna_nos': _dud_nos,
        'dud_antenna_names': _dud_names,
        'active_antennas': active_antennas,
        'chan_freqs_hz': chan_freqs,
        'stokes_labels': stokes_labels,
        'path': str(path),
        'gcount': gcount, 'pcount': pcount,
        'naxis2': naxis2, 'naxis3': naxis3,
        'naxis4': naxis4, 'naxis5': naxis5,
        'data_offset': data_offset,
        'group_size': group_size,
        'data_per_group': data_per_group,
        'parnames': parnames,
        'build_time_sec': elapsed,
    }

    mem_mb = (source_id.nbytes + jd.nbytes + ant1.nbytes + ant2.nbytes
              + uu_sec.nbytes + vv_sec.nbytes) / 1e6
    n_active = len(active_antennas)
    n_dud    = len(_dud_nos)
    print(f'Index built: {gcount:,} rows, {len(id_to_name)} sources, '
          f'{n_active} active antennas'
          f'{f" ({n_dud} dud: {_dud_names})" if n_dud else ""},'
          f' {mem_mb:.1f} MB RAM, {elapsed:.1f}s')
    for sid, name in sorted(id_to_name.items()):
        ranges = source_ranges.get(sid, [])
        total_rows = sum(e - s for s, e in ranges)
        print(f'  {name:>12s} (id={sid}): {total_rows:>8,} rows in {len(ranges)} block(s)')
    return idx


def compute_file_sha256(path: Union[str, Path], chunk_size: int = 8 * 1024 * 1024) -> str:
    """Compute SHA256 of a file for cache validity checks."""
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def get_file_identity(path: Union[str, Path]) -> dict:
    """Return cheap file identity metadata for cache validation."""
    path = Path(path)
    stat = path.stat()
    return {
        'size_bytes': int(stat.st_size),
        'mtime_ns': int(stat.st_mtime_ns),
    }


def default_row_index_cache_path(path: Union[str, Path], cache_dir: Optional[Union[str, Path]] = None) -> Path:
    """Return a deterministic cache path for a row-index sidecar."""
    path = Path(path)
    if cache_dir is None:
        cache_dir = path.parent
    cache_dir = Path(cache_dir)
    return cache_dir / f'{path.name}.row_index_cache.npz'


def save_row_index_cache(index: dict, cache_path: Union[str, Path], source_sha256: Optional[str] = None) -> Path:
    """Persist a built row index to a sidecar NPZ cache file."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        'path': str(index['path']),
        'source_sha256': source_sha256,
        'source_identity': index.get('source_identity'),
        'source_ranges': index['source_ranges'],
        'id_to_name': {str(k): v for k, v in index['id_to_name'].items()},
        'antennas': index['antennas'],
        'dud_antenna_nos': index.get('dud_antenna_nos', []),
        'dud_antenna_names': index.get('dud_antenna_names', []),
        'freq': index['freq'],
        'stokes_labels': list(index['stokes_labels']),
        'gcount': int(index['gcount']),
        'pcount': int(index['pcount']),
        'naxis2': int(index['naxis2']),
        'naxis3': int(index['naxis3']),
        'naxis4': int(index['naxis4']),
        'naxis5': int(index['naxis5']),
        'data_offset': int(index['data_offset']),
        'group_size': int(index['group_size']),
        'data_per_group': int(index['data_per_group']),
        'parnames': list(index['parnames']),
        'build_time_sec': float(index.get('build_time_sec', 0.0)),
    }

    np.savez_compressed(
        cache_path,
        source_id=np.asarray(index['source_id'], dtype=np.int32),
        jd=np.asarray(index['jd'], dtype=np.float64),
        ant1=np.asarray(index['ant1'], dtype=np.int16),
        ant2=np.asarray(index['ant2'], dtype=np.int16),
        uu_sec=np.asarray(index['uu_sec'], dtype=np.float32),
        vv_sec=np.asarray(index['vv_sec'], dtype=np.float32),
        chan_freqs_hz=np.asarray(index['chan_freqs_hz'], dtype=np.float64),
        metadata_json=json.dumps(metadata),
    )
    return cache_path


def load_row_index_cache(
    cache_path: Union[str, Path],
    source_path: Optional[Union[str, Path]] = None,
    validation_mode: str = 'fast',
) -> dict:
    """Load a persisted row-index cache and optionally verify source identity.

    validation_mode:
    - 'fast': compare cached size + mtime_ns against current file
    - 'fast+sha': fast check first, then SHA256 confirmation on mismatch
    - 'sha256': compare cached SHA256 against current file
    - 'none': trust cache without source-file validation
    """
    cache_path = Path(cache_path)
    with np.load(cache_path, allow_pickle=False) as npz:
        metadata = json.loads(str(npz['metadata_json']))

        cached_sha = metadata.get('source_sha256')
        cached_identity = metadata.get('source_identity') or None
        if source_path is not None and validation_mode != 'none':
            if validation_mode == 'fast':
                current_identity = get_file_identity(source_path)
                if cached_identity is None:
                    raise ValueError(
                        'Row-index cache has no stored file identity for fast validation; rebuild the cache.'
                    )
                if current_identity != cached_identity:
                    raise ValueError(
                        'Row-index cache fast validation mismatch (size/mtime). '
                        'Source visdata appears to have changed; rebuild the cache.'
                    )
            elif validation_mode == 'fast+sha':
                current_identity = get_file_identity(source_path)
                if cached_identity is None:
                    raise ValueError(
                        'Row-index cache has no stored file identity for fast+sha validation; rebuild the cache.'
                    )
                if current_identity != cached_identity:
                    if not cached_sha:
                        raise ValueError(
                            'Row-index cache fast validation mismatch and no cached SHA256 is available; '
                            'rebuild the cache.'
                        )
                    current_sha = compute_file_sha256(source_path)
                    if current_sha != cached_sha:
                        raise ValueError(
                            'Row-index cache fast validation mismatch and SHA256 mismatch. '
                            'Source visdata appears to have changed; rebuild the cache.'
                        )
            elif validation_mode == 'sha256':
                if not cached_sha:
                    raise ValueError(
                        'Row-index cache has no stored SHA256 for strict validation; rebuild the cache.'
                    )
                current_sha = compute_file_sha256(source_path)
                if current_sha != cached_sha:
                    raise ValueError(
                        'Row-index cache SHA256 mismatch. Source visdata appears to have changed; '
                        'rebuild the cache.'
                    )
            else:
                raise ValueError("validation_mode must be one of: 'fast', 'fast+sha', 'sha256', 'none'.")

        source_ranges = {
            int(k): [(int(s), int(e)) for s, e in v]
            for k, v in metadata['source_ranges'].items()
        }
        id_to_name = {int(k): str(v) for k, v in metadata['id_to_name'].items()}

        return {
            'source_id': np.asarray(npz['source_id'], dtype=np.int32),
            'jd': np.asarray(npz['jd'], dtype=np.float64),
            'ant1': np.asarray(npz['ant1'], dtype=np.int16),
            'ant2': np.asarray(npz['ant2'], dtype=np.int16),
            'uu_sec': np.asarray(npz['uu_sec'], dtype=np.float32),
            'vv_sec': np.asarray(npz['vv_sec'], dtype=np.float32),
            'source_ranges': source_ranges,
            'id_to_name': id_to_name,
            'antennas': metadata['antennas'],
            'dud_antenna_nos': metadata.get('dud_antenna_nos', []),
            'dud_antenna_names': metadata.get('dud_antenna_names', []),
            'active_antennas': [a for a in metadata['antennas']
                                if a['antenna_no'] not in set(metadata.get('dud_antenna_nos', []))],
            'freq': metadata['freq'],
            'chan_freqs_hz': np.asarray(npz['chan_freqs_hz'], dtype=np.float64),
            'stokes_labels': list(metadata['stokes_labels']),
            'path': str(metadata['path']),
            'gcount': int(metadata['gcount']),
            'pcount': int(metadata['pcount']),
            'naxis2': int(metadata['naxis2']),
            'naxis3': int(metadata['naxis3']),
            'naxis4': int(metadata['naxis4']),
            'naxis5': int(metadata['naxis5']),
            'data_offset': int(metadata['data_offset']),
            'group_size': int(metadata['group_size']),
            'data_per_group': int(metadata['data_per_group']),
            'parnames': list(metadata['parnames']),
            'build_time_sec': float(metadata.get('build_time_sec', 0.0)),
            'source_sha256': cached_sha,
            'source_identity': cached_identity,
            'index_cache_path': str(cache_path),
        }


def get_or_build_row_index(
    path: Union[str, Path],
    cache_path: Optional[Union[str, Path]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    force_rebuild: bool = False,
    validation_mode: str = 'fast',
    write_cache: bool = True,
    override_dud_names: Optional[List[str]] = None,
) -> dict:
    """Load a persistent row index if valid, otherwise build and cache one.

    override_dud_names : list[str], optional
        Passed to :func:`build_row_index`.  Explicit antenna names to treat as
        duds, bypassing automatic STABXYZ-based detection.  Example:
        ``override_dud_names=['C07', 'S05']``.
    """
    path = Path(path)
    cache_path = Path(cache_path) if cache_path is not None else default_row_index_cache_path(path, cache_dir=cache_dir)

    if cache_path.exists() and not force_rebuild:
        try:
            index = load_row_index_cache(cache_path, source_path=path, validation_mode=validation_mode)
            print(f'Loaded row index cache: {cache_path}')
            return index
        except ValueError as exc:
            print(f'[row-index-cache] warning: {exc}')
            print('[row-index-cache] Rebuilding cache from source file.')

    if validation_mode not in ('fast', 'fast+sha', 'sha256', 'none'):
        raise ValueError("validation_mode must be one of: 'fast', 'fast+sha', 'sha256', 'none'.")

    source_identity = get_file_identity(path)
    # Always store SHA256 in cache so fast+sha fallback works in future runs.
    source_sha256 = compute_file_sha256(path)
    index = build_row_index(path, override_dud_names=override_dud_names)
    index['source_sha256'] = source_sha256
    index['source_identity'] = source_identity
    index['index_cache_path'] = str(cache_path)
    if write_cache:
        save_row_index_cache(index, cache_path, source_sha256=source_sha256)
        print(f'Saved row index cache: {cache_path}')
    return index


def get_source_observation_properties(
    index: dict,
    source: Union[str, int],
    scan_gap_seconds: float = 5.0,
    time_tolerance_seconds: float = 1e-3,
    include_integrations: bool = False,
) -> dict:
    """Source-level observation metadata computed entirely from the cached row index."""
    id_to_name = index['id_to_name']

    if isinstance(source, int):
        candidate_ids = np.array([source], dtype=np.int32)
    else:
        wanted = str(source).strip().lower()
        matched = [sid for sid, sname in id_to_name.items()
                   if sname.lower() == wanted]
        if not matched:
            raise ValueError(
                f'Source "{source}" not found. '
                f'Available: {sorted(id_to_name.values())}')
        candidate_ids = np.array(matched, dtype=np.int32)

    row_mask = np.isin(index['source_id'], candidate_ids)
    if not np.any(row_mask):
        raise ValueError(f'No rows for source "{source}".')

    jd_src = index['jd'][row_mask]
    a1_src = index['ant1'][row_mask]
    a2_src = index['ant2'][row_mask]
    order  = np.argsort(jd_src)
    jd_src = jd_src[order]; a1_src = a1_src[order]; a2_src = a2_src[order]

    tol_days = time_tolerance_seconds / 86400.0
    splits = np.where(np.diff(jd_src) > tol_days)[0] + 1
    int_groups = np.split(np.arange(jd_src.size), splits)

    n_int = len(int_groups)
    int_times = np.empty(n_int, dtype=np.float64)
    bl_counts = np.empty(n_int, dtype=np.int32)
    auto_cnts = np.empty(n_int, dtype=np.int32)
    integrations = [] if include_integrations else None

    for ii, grp in enumerate(int_groups):
        t_jd = float(jd_src[grp[0]])
        int_times[ii] = t_jd
        pairs = np.sort(np.stack([a1_src[grp], a2_src[grp]], axis=1), axis=1)
        uniq = np.unique(pairs, axis=0)
        n_bl = int(uniq.shape[0])
        n_au = int(np.sum(uniq[:, 0] == uniq[:, 1]))
        bl_counts[ii] = n_bl
        auto_cnts[ii] = n_au
        if include_integrations:
            integrations.append({
                'integration_index': ii + 1,
                'time_jd': t_jd,
                'time_utc': Time(t_jd, format='jd', scale='utc').isot,
                'n_vis_rows': int(grp.size),
                'n_baselines': n_bl,
                'n_auto_baselines': n_au,
                'n_cross_baselines': n_bl - n_au,
            })

    diffs = np.diff(int_times) * 86400.0
    pos = diffs[diffs > 0]
    inferred_sec = float(np.median(pos)) if pos.size else 0.0

    eff_gap = float(scan_gap_seconds)
    if inferred_sec > 0 and eff_gap <= inferred_sec:
        eff_gap = 1.5 * inferred_sec
    scan_splits = np.where(np.diff(int_times) > eff_gap / 86400.0)[0] + 1
    scan_groups = np.split(np.arange(n_int), scan_splits)

    sid0 = int(candidate_ids[0])
    sname0 = id_to_name.get(sid0, str(source))

    scans = []
    for si, sg in enumerate(scan_groups, 1):
        t0 = float(int_times[sg[0]]); t1 = float(int_times[sg[-1]])
        sbl = bl_counts[sg]; sau = auto_cnts[sg]
        scans.append({
            'source_id': sid0, 'source_name': sname0, 'scan_index': si,
            'start_jd': t0, 'end_jd': t1,
            'start_utc': Time(t0, format='jd', scale='utc').isot,
            'end_utc':   Time(t1, format='jd', scale='utc').isot,
            'n_integrations': int(len(sg)),
            'n_vis_rows': int(sum(int_groups[i].size for i in sg)),
            'duration_sec': float((t1 - t0) * 86400.0),
            'baseline_count_min': int(sbl.min()),
            'baseline_count_median': float(np.median(sbl)),
            'baseline_count_max': int(sbl.max()),
            'autos_present': bool(sau.max() > 0),
        })

    # Use active (non-dud) antenna count for all statistics
    n_ant = len(index.get('active_antennas', index['antennas']))
    cross_th = n_ant * (n_ant - 1) // 2
    freq = index['freq']

    # On-source time: number of actual integrations × integration duration.
    # This differs from total_duration_sec (= last - first timestamp) when the
    # source is observed in multiple scans separated by slews or other targets.
    on_source_time_sec = float(n_int * inferred_sec)

    summary = {
        'file': index['path'],
        'source_id': sid0 if len(candidate_ids) == 1 else candidate_ids.tolist(),
        'source_name': sname0 if len(candidate_ids) == 1 else str(source),
        'n_vis_rows': int(jd_src.size),
        'n_integrations': n_int,
        'n_scans': len(scans),
        'requested_scan_gap_seconds': float(scan_gap_seconds),
        'effective_scan_gap_seconds': eff_gap,
        'integration_time_sec': inferred_sec,
        'begin_time_utc': Time(float(int_times[0]), format='jd', scale='utc').isot,
        'end_time_utc':   Time(float(int_times[-1]), format='jd', scale='utc').isot,
        # total_duration_sec = timespan from first to last integration (includes gaps between scans).
        # on_source_time_sec = n_integrations * integration_time — the actual on-sky accumulation time.
        'total_duration_sec': float((int_times[-1] - int_times[0]) * 86400.0),
        'on_source_time_sec': on_source_time_sec,
        'n_channels': index['naxis4'],
        'n_correlations': index['naxis3'],
        'centre_freq_hz': 0.5 * (freq['freq_min_hz'] + freq['freq_max_hz']),
        'bFreq_hz': freq['freq_min_hz'],
        'eFreq_hz': freq['freq_max_hz'],
        'antenna_count': n_ant,
        'dud_antenna_nos': index.get('dud_antenna_nos', []),
        'dud_antenna_names': index.get('dud_antenna_names', []),
        'cross_baselines_theoretical': cross_th,
        'baselines_with_autos_theoretical': cross_th + n_ant,
        'autos_present_anywhere': bool(auto_cnts.max() > 0),
        'baseline_count_per_integration_min': int(bl_counts.min()),
        'baseline_count_per_integration_median': float(np.median(bl_counts)),
        'baseline_count_per_integration_max': int(bl_counts.max()),
        'auto_baseline_count_per_integration_min': int(auto_cnts.min()),
        'auto_baseline_count_per_integration_median': float(np.median(auto_cnts)),
        'auto_baseline_count_per_integration_max': int(auto_cnts.max()),
        'scans': scans,
    }
    if include_integrations:
        summary['integrations'] = integrations
    return summary


def list_scan_times_by_source(
    index: dict, source: Union[str, int], merge_gap_seconds: float = 5.0,
):
    return get_source_observation_properties(
        index, source=source, scan_gap_seconds=merge_gap_seconds,
    )['scans']

# ---------------------------------------------------------------------------
# Antenna geometry — ENU positions and inter-dish separations
# ---------------------------------------------------------------------------

def _get_array_enu(fits_path: Union[str, Path]) -> dict:
    """Read AIPS AN table and return per-antenna ENU coordinates.

    Returns
    -------
    dict with keys:
        'nos'       : list[int]   antenna numbers (NOSTA)
        'names'     : list[str]   antenna names   (ANNAME)
        'enu'       : ndarray (N, 3)  East/North/Up in metres relative to array centre
        'ecef_centre': ndarray (3,)   array centre ECEF XYZ in metres
        'lat_deg'   : float  geodetic latitude of array centre
        'lon_deg'   : float  geodetic longitude of array centre
    """
    from astropy.coordinates import EarthLocation
    import astropy.units as _u
    with fits.open(str(fits_path), memmap=True, lazy_load_hdus=True) as hdul:
        an  = _get_hdu(hdul, 'AIPS AN')
        if an is None:
            raise ValueError('No AIPS AN extension found in FITS file.')
        ah  = an.header
        arr_xyz = np.array([float(ah['ARRAYX']), float(ah['ARRAYY']), float(ah['ARRAYZ'])])
        nos, names, xyz = [], [], []
        for row in an.data:
            nos.append(int(row['NOSTA']))
            names.append(_decode_bytes(row['ANNAME']))
            xyz.append(np.array(row['STABXYZ'], dtype=np.float64))
    xyz = np.array(xyz)   # (N, 3) ECEF offsets from array centre
    loc = EarthLocation.from_geocentric(*arr_xyz, unit=_u.m)
    lat = np.radians(loc.lat.deg)
    lon = np.radians(loc.lon.deg)
    R = np.array([
        [-np.sin(lon),               np.cos(lon),              0.0          ],
        [-np.sin(lat)*np.cos(lon),  -np.sin(lat)*np.sin(lon),  np.cos(lat) ],
        [ np.cos(lat)*np.cos(lon),   np.cos(lat)*np.sin(lon),  np.sin(lat) ],
    ])
    enu = xyz @ R.T  # (N, 3): East, North, Up
    return {
        'nos': nos, 'names': names, 'enu': enu,
        # stabxyz: raw ECEF offset of each antenna from the array-centre (ARRAYX/Y/Z).
        # An all-zero row means no position was ever recorded in the AN table (dud antenna).
        'stabxyz': xyz,
        'ecef_centre': arr_xyz,
        'lat_deg': loc.lat.deg, 'lon_deg': loc.lon.deg,
    }


def compute_source_azel(
    fits_path_or_index,
    source: Union[str, int],
    timerange: Optional[Tuple] = None,
    time_step_s: float = 60.0,
) -> dict:
    """Compute Azimuth / Elevation vs time for a source in the FITS file.

    Parameters
    ----------
    fits_path_or_index : str | Path | dict
        FITS path, or a row index dict (``index['path']`` is used for the FITS file).
    source : str | int
        Source name (as in AIPS SU) or integer source ID.
    timerange : (start, end) optional
        Restrict to this JD range or ISO string range, e.g.:
        ``(2459421.0, 2459421.5)``  or
        ``('2021-07-25 19:00:00', '2021-07-26 02:00:00')``
    time_step_s : float
        Sampling interval in seconds (default 60 s).

    Returns
    -------
    dict with keys:
        'source_name', 'ra_deg', 'dec_deg',
        'jd'         : ndarray  Julian dates,
        'utc'        : list[str]  ISO UTC strings,
        'az_deg'     : ndarray  Azimuth (N→E convention),
        'el_deg'     : ndarray  Elevation in degrees,
        'array_lat_deg', 'array_lon_deg'
    """
    from astropy.coordinates import SkyCoord, EarthLocation, AltAz
    from astropy.time import Time
    import astropy.units as _u

    if isinstance(fits_path_or_index, dict):
        fits_path = fits_path_or_index['path']
        index = fits_path_or_index
    else:
        fits_path = fits_path_or_index
        index = None

    # ── Source RA/Dec from AIPS SU ─────────────────────────────────────────────
    with fits.open(str(fits_path), memmap=True, lazy_load_hdus=True) as hdul:
        su = _get_hdu(hdul, 'AIPS SU')
        if su is None:
            raise ValueError('No AIPS SU extension in FITS file.')
        su_data = su.data
        id_col = next(
            (c for c in su_data.dtype.names if 'ID' in c.upper() and 'NO' in c.upper()),
            None)
        src_row = None
        for row in su_data:
            sid  = int(row[id_col]) if id_col else None
            name = _decode_bytes(row['SOURCE'])
            if isinstance(source, str) and name.strip().lower() == source.strip().lower():
                src_row = row; break
            if isinstance(source, int) and sid == source:
                src_row = row; break
        if src_row is None:
            available = [_decode_bytes(r['SOURCE']).strip() for r in su_data]
            raise ValueError(f'Source "{source}" not found. Available: {available}')
        src_name = _decode_bytes(src_row['SOURCE']).strip()
        ra_deg   = float(src_row['RAEPO'])
        dec_deg  = float(src_row['DECEPO'])

    # ── Array location from AIPS AN ────────────────────────────────────────────
    geo = _get_array_enu(fits_path)
    loc = EarthLocation.from_geocentric(*geo['ecef_centre'], unit=_u.m)

    # ── JD timestamps: all rows for this source ────────────────────────────────
    if index is not None:
        id_to_name = index['id_to_name']
        if isinstance(source, str):
            src_ids = [k for k, v in id_to_name.items()
                       if v.strip().lower() == source.strip().lower()]
        else:
            src_ids = [int(source)]
        jd_all = index['jd']
        src_id_arr = index['source_id']
        mask_src = np.isin(src_id_arr, src_ids)
        jd_src = jd_all[mask_src]
    else:
        # Lightweight read without full index
        with fits.open(str(fits_path), memmap=True) as hdul:
            h = hdul[0].header
            pz5 = float(h.get('PZERO5', 0)); ps5 = float(h.get('PSCAL5', 1))
            pz6 = float(h.get('PZERO6', 0)); ps6 = float(h.get('PSCAL6', 1))
            pars = hdul[0].data.par
            step = max(1, h['GCOUNT'] // 20_000)
            idx = np.arange(0, h['GCOUNT'], step)
            p5 = ps5 * np.array([pars(4)[i] for i in idx], dtype=np.float64) + pz5
            p6 = ps6 * np.array([pars(5)[i] for i in idx], dtype=np.float64) + pz6
            jd_src = p5 + p6

    jd_src = jd_src[np.isfinite(jd_src)]
    if jd_src.size == 0:
        raise ValueError(f'No timestamps found for source "{source}".')

    # Apply optional JD / ISO timerange
    if timerange is not None:
        t0, t1 = timerange
        if isinstance(t0, str):
            t0 = Time(t0, format='iso', scale='utc').jd
        if isinstance(t1, str):
            t1 = Time(t1, format='iso', scale='utc').jd
        jd_src = jd_src[(jd_src >= t0) & (jd_src <= t1)]

    # Downsample to requested time_step_s
    jd_step = time_step_s / 86400.0
    jd_min, jd_max = jd_src.min(), jd_src.max()
    n_steps = max(2, int((jd_max - jd_min) / jd_step) + 1)
    jd_grid = np.linspace(jd_min, jd_max, n_steps)

    # ── AltAz transform ────────────────────────────────────────────────────────
    coord  = SkyCoord(ra=ra_deg * _u.deg, dec=dec_deg * _u.deg, frame='icrs')
    t_grid = Time(jd_grid, format='jd', scale='utc')
    frame  = AltAz(obstime=t_grid, location=loc)
    altaz  = coord.transform_to(frame)

    return {
        'source_name':    src_name,
        'ra_deg':         ra_deg,
        'dec_deg':        dec_deg,
        'jd':             jd_grid,
        'utc':            [t.iso for t in t_grid],
        'az_deg':         altaz.az.deg,
        'el_deg':         altaz.alt.deg,
        'array_lat_deg':  geo['lat_deg'],
        'array_lon_deg':  geo['lon_deg'],
    }


# ---------------------------------------------------------------------------
# Source data-query application
# ---------------------------------------------------------------------------
#
# Computes every property relevant to deciding observation quality and setting
# data-selection parameters (elevation cuts, UV ranges, time windows).
#
# Public API:
#   query_source(fits_path_or_index, source, ...)  → dict
#   print_source_query(result)                      → None   (pretty-print)
#   plot_source_query(result, ...)                  → Figure (4-panel)
# ---------------------------------------------------------------------------

def _gmst_deg(jd: np.ndarray) -> np.ndarray:
    """Greenwich Mean Sidereal Time in degrees from Julian Date(s), via astropy."""
    from astropy.time import Time as _T
    t = _T(np.asarray(jd, dtype=np.float64), format='jd', scale='utc')
    return np.asarray(t.sidereal_time('mean', 'greenwich').deg, dtype=np.float64)


def _lmst_deg(jd: np.ndarray, lon_deg: float) -> np.ndarray:
    """Local Mean Sidereal Time in degrees."""
    return (_gmst_deg(jd) + lon_deg) % 360.0


def _hour_angle_deg(jd: np.ndarray, ra_deg: float, lon_deg: float) -> np.ndarray:
    """Hour angle in degrees in range (-180, 180]."""
    ha = (_lmst_deg(jd, lon_deg) - ra_deg) % 360.0
    ha[ha > 180.0] -= 360.0
    return ha


def _parallactic_angle_deg(
    ha_deg: np.ndarray,
    dec_deg: float,
    lat_deg: float,
) -> np.ndarray:
    """Parallactic angle in degrees.

    Uses the standard formula:
      PA = atan2( -cos(φ)·sin(H), sin(φ)·cos(δ) - cos(φ)·sin(δ)·cos(H) )
    where φ = geodetic latitude, δ = source declination, H = hour angle.
    """
    h   = np.radians(ha_deg)
    phi = np.radians(lat_deg)
    dec = np.radians(dec_deg)
    num = -np.cos(phi) * np.sin(h)
    den = np.sin(phi) * np.cos(dec) - np.cos(phi) * np.sin(dec) * np.cos(h)
    return np.degrees(np.arctan2(num, den))


def _uv_stats_for_source(
    index: dict,
    source: Union[str, int],
) -> dict:
    """Compute UV-coverage statistics from the row index.

    Returns statistics in seconds (light-travel), metres, and kλ at
    the band centre frequency.
    """
    id_to_name = index['id_to_name']
    if isinstance(source, int):
        src_ids = [source]
    else:
        wanted = str(source).strip().lower()
        src_ids = [k for k, v in id_to_name.items() if v.strip().lower() == wanted]
        if not src_ids:
            raise ValueError(f'Source "{source}" not found. Available: {sorted(id_to_name.values())}')

    mask = np.isin(index['source_id'], src_ids)
    uu   = index['uu_sec'][mask].astype(np.float64)
    vv   = index['vv_sec'][mask].astype(np.float64)

    uv_sec  = np.hypot(uu, vv)
    c_mps   = 299_792_458.0
    freq    = index['freq']
    nu_hz   = 0.5 * (freq['freq_min_hz'] + freq['freq_max_hz'])

    uv_m      = uv_sec * c_mps
    uv_klam   = uv_sec * nu_hz * 1e-3   # kλ at band centre

    # Remove exact zeros (auto-correlations if present)
    cross = uv_sec > 1e-10
    uv_m_x    = uv_m[cross]
    uv_klam_x = uv_klam[cross]
    uu_x      = uu[cross] * c_mps
    vv_x      = vv[cross] * c_mps

    if uv_m_x.size == 0:
        return {'warning': 'No cross-baseline UV data found for this source.'}

    # UV plane filling: 2-D histogram density
    n_bins = 50
    uv_max_m = float(np.percentile(uv_m_x, 99.0))
    hist2d, _, _ = np.histogram2d(
        uu_x, vv_x,
        bins=n_bins,
        range=[[-uv_max_m, uv_max_m], [-uv_max_m, uv_max_m]],
    )
    n_filled = int(np.sum(hist2d > 0))
    n_total  = n_bins * n_bins
    uv_fill_pct = 100.0 * n_filled / n_total

    # Angular resolution at band centre: θ ~ λ/(2·Bmax) in arcsec
    lam_m = c_mps / nu_hz
    theta_min_arcsec = float(206265.0 * lam_m / (2.0 * np.max(uv_m_x)))
    # Largest angular scale: LAS ~ λ / (2·Bmin)
    theta_las_arcmin = float(206265.0 * lam_m / (2.0 * np.min(uv_m_x))) / 60.0

    return {
        'n_vis_rows_cross':         int(uv_m_x.size),
        'centre_freq_mhz':          round(nu_hz / 1e6, 3),
        'wavelength_m':             round(lam_m, 4),
        'bmin_m':                   round(float(np.min(uv_m_x)),   1),
        'bmedian_m':                round(float(np.median(uv_m_x)), 1),
        'bmax_m':                   round(float(np.max(uv_m_x)),   1),
        'bmin_klambda':             round(float(np.min(uv_klam_x)),    3),
        'bmedian_klambda':          round(float(np.median(uv_klam_x)), 3),
        'bmax_klambda':             round(float(np.max(uv_klam_x)),    3),
        'p10_blen_m':               round(float(np.percentile(uv_m_x, 10)), 1),
        'p90_blen_m':               round(float(np.percentile(uv_m_x, 90)), 1),
        'angular_resolution_arcsec': round(theta_min_arcsec, 3),
        'largest_angular_scale_arcmin': round(theta_las_arcmin, 2),
        'uv_filling_pct_50bins':    round(uv_fill_pct, 1),
        'uu_m_raw':   uu_x,   # kept for plotting — not serialisable
        'vv_m_raw':   vv_x,   # kept for plotting
    }



def _physical_baseline_stats(geo: dict, dud_nos=None) -> dict:
    """Shortest, median, and longest physical dish separations from ENU coords.

    Parameters
    ----------
    geo : dict
        Output of :func:`_get_array_enu`.
    dud_nos : list[int], optional
        Antenna numbers to exclude from baseline statistics.
        Defaults to ``None`` (no exclusion).
    """
    _dud_set = set(dud_nos or [])
    # Filter out dud antennas before computing separations
    _keep    = [i for i, no in enumerate(geo['nos']) if no not in _dud_set]
    enu   = geo['enu'][_keep]
    nos   = [geo['nos'][i]   for i in _keep]
    names = [geo['names'][i] for i in _keep]
    n     = len(nos)

    lengths = []
    pairs   = []
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(enu[i] - enu[j]))
            lengths.append(d)
            pairs.append((names[i], names[j], d))

    lengths = np.array(lengths)
    pairs   = sorted(pairs, key=lambda x: x[2])

    return {
        'n_antennas':              n,
        'n_physical_baselines':    len(lengths),
        'shortest_m':              round(float(lengths.min()), 1),
        'shortest_pair':           f'{pairs[0][0]}-{pairs[0][1]}',
        'longest_m':               round(float(lengths.max()), 1),
        'longest_pair':            f'{pairs[-1][0]}-{pairs[-1][1]}',
        'median_m':                round(float(np.median(lengths)), 1),
        'p10_m':                   round(float(np.percentile(lengths, 10)), 1),
        'p90_m':                   round(float(np.percentile(lengths, 90)), 1),
        'five_shortest_pairs':     [(f'{p[0]}-{p[1]}', round(p[2], 1)) for p in pairs[:5]],
        'five_longest_pairs':      [(f'{p[0]}-{p[1]}', round(p[2], 1)) for p in pairs[-5:]],
        'enu':                     enu,          # kept for plotting
        'names':                   names,
        'nos':                     nos,
    }


def query_source(
    fits_path_or_index,
    source: Union[str, int],
    azel_time_step_s: float = 60.0,
    elevation_thresholds_deg: Optional[List[float]] = None,
    scan_gap_seconds: float = 5.0,
    include_uv_raw: bool = True,
    override_dud_names: Optional[List[str]] = None,
) -> dict:
    """Comprehensive per-source data-quality and observation-planning query.

    Extracts **every** property that is relevant for deciding observation
    quality, choosing data-selection parameters, and planning calibration
    strategy — all from the FITS file alone, with no external input required.

    Parameters
    ----------
    fits_path_or_index : str | Path | dict
        Either the path to the UVFITS file or an already-built row index dict
        (from :func:`build_row_index` or :func:`get_or_build_row_index`).
    source : str | int
        Source name (case-insensitive) or integer AIPS source ID.
    azel_time_step_s : float
        Time resolution for Az/El, HA, PA tracks (seconds).  Default 60 s.
    elevation_thresholds_deg : list[float], optional
        Elevation cut values to compute ``time_above_*_deg`` statistics.
        Defaults to [15, 20, 25, 30, 35, 40, 45, 50, 60].
    scan_gap_seconds : float
        Gap in seconds used to identify scan breaks.  Default 5 s.
    include_uv_raw : bool
        Whether to include raw numpy arrays (``uu_m_raw``, ``vv_m_raw``,
        ``enu``) in the returned dict.  Set to ``False`` for JSON export.

    Returns
    -------
    dict
        A nested dict with the following top-level keys:

        ``source``
            Name, RA, Dec, Galactic l/b.
        ``file``
            FITS path, file size.
        ``frequency``
            Centre frequency, bandwidth, channel width, number of channels.
        ``timing``
            Total duration, number of integrations, integration time,
            per-scan start/end UTC strings and durations.
        ``pointing``
            Az/El time track, elevation statistics, time above each threshold.
        ``hour_angle``
            HA track, HA at start/end, transit UTC (if within observation).
        ``parallactic_angle``
            PA track, PA range, maximum PA rate — important for polarisation
            calibration planning.
        ``uv_coverage``
            B_min/B_med/B_max in metres and kλ, angular resolution and
            largest angular scale at band centre, UV-plane fill fraction.
        ``physical_baselines``
            Physical dish separations from the AIPS AN table ENU positions,
            shortest/longest pairs (orthogonal to UV which depends on
            frequency and hour angle).
        ``sensitivity``
            Geometric factors (N_pol, N_baselines, Δt, Δν) that go into
            the noise formula σ = SEFD / sqrt(N_pol · N_bl · Δν · Δt).
        ``data_selection_advice``
            Ready-to-use ``--set`` strings and Python-dict values for
            SOLVE_ELEVATION_MIN_DEG, SOLVE_UVRANGE_KLAMBDA, SOLVE_TIMERANGE.

    Examples
    --------
    >>> import ugmrt_query as uq
    >>> FITS = '/data/40_014_25jul2021_gsb.FITS'
    >>> idx  = uq.get_or_build_row_index(FITS)
    >>> result = uq.query_source(idx, '3C48')
    >>> uq.print_source_query(result)
    >>> fig = uq.plot_source_query(result)
    >>> fig.savefig('3c48_query.png', dpi=150)
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as _u

    if elevation_thresholds_deg is None:
        elevation_thresholds_deg = [15, 20, 25, 30, 35, 40, 45, 50, 60]

    # ── Resolve fits path and index ────────────────────────────────────────────
    if isinstance(fits_path_or_index, dict):
        fits_path = fits_path_or_index['path']
        index     = fits_path_or_index
    else:
        fits_path = str(fits_path_or_index)
        # Build once here so override_dud_names is honoured and the index is
        # reused for all subsequent internal calls (obs, UV stats, etc.).
        index     = build_row_index(Path(fits_path),
                                    override_dud_names=override_dud_names)

    fits_path = str(fits_path)

    # ── 1. Frequency ───────────────────────────────────────────────────────────
    freq_props = list_frequency_properties(fits_path)
    nu_centre_hz = 0.5 * (freq_props['freq_min_hz'] + freq_props['freq_max_hz'])

    # ── 2. Antenna geometry ────────────────────────────────────────────────────
    geo = _get_array_enu(fits_path)
    lat_deg = geo['lat_deg']
    lon_deg = geo['lon_deg']

    # Dud-antenna list resolution (priority order):
    #   1. Explicit override_dud_names passed by caller (highest authority)
    #   2. Cached values from a pre-built index (already resolved at build time)
    #   3. Empty — no auto-detection; caller must provide the list explicitly.
    if override_dud_names is not None:
        _name_to_no_q = {nm: no for no, nm in zip(geo['nos'], geo['names'])}
        _query_dud_nos = sorted(
            no for _w in override_dud_names
            for nm, no in _name_to_no_q.items()
            if nm == _w or nm.startswith(_w + ':') or nm.startswith(_w)
        )
    elif index and 'dud_antenna_nos' in index:
        _query_dud_nos = index['dud_antenna_nos']
    else:
        _query_dud_nos = []

    # ── 3. Az/El track (uniform grid — spans full timerange incl. scan gaps) ───
    azel = compute_source_azel(
        index,
        source=source,
        time_step_s=azel_time_step_s,
    )
    jd_grid  = azel['jd']
    el_deg   = azel['el_deg']
    az_deg   = azel['az_deg']
    ra_deg   = azel['ra_deg']
    dec_deg  = azel['dec_deg']
    src_name = azel['source_name']

    # Grid step (seconds) — used for timespan-based statistics
    dt_s = (jd_grid[-1] - jd_grid[0]) * 86400.0 / max(1, len(jd_grid) - 1)

    # Elevation statistics from uniform grid (timespan — includes scan gaps)
    el_time_above = {}
    for thr in elevation_thresholds_deg:
        n_above = int(np.sum(el_deg >= thr))
        el_time_above[f'time_above_{int(thr)}_deg_min'] = round(n_above * dt_s / 60.0, 1)

    # ── 4. Hour angle (from grid) ──────────────────────────────────────────────
    ha_deg  = _hour_angle_deg(jd_grid.copy(), ra_deg, lon_deg)
    ha_mins = ha_deg * 4.0   # 1 deg HA = 4 minutes

    # Find transit (HA closest to 0) on the grid
    i_transit = int(np.argmin(np.abs(ha_deg)))
    transit_utc = azel['utc'][i_transit]
    ha_at_transit = float(ha_deg[i_transit])

    # ── 5. Parallactic angle (from grid) ──────────────────────────────────────
    pa_deg  = _parallactic_angle_deg(ha_deg, dec_deg, lat_deg)
    # PA rate from grid: max instantaneous rate in deg/min
    if len(pa_deg) > 1:
        dpa = np.diff(pa_deg)
        dpa[dpa >  180] -= 360
        dpa[dpa < -180] += 360
        pa_rate_deg_per_min = float(np.max(np.abs(dpa)) / (dt_s / 60.0))
    else:
        pa_rate_deg_per_min = 0.0

    # ── 6. Galactic coordinates ────────────────────────────────────────────────
    sky = SkyCoord(ra=ra_deg * _u.deg, dec=dec_deg * _u.deg, frame='icrs')
    gal = sky.galactic
    gal_l = float(gal.l.deg)
    gal_b = float(gal.b.deg)

    # ── 7. Scan/timing properties ─────────────────────────────────────────────
    obs = get_source_observation_properties(
        index,
        source=source,
        scan_gap_seconds=scan_gap_seconds,
        include_integrations=False,
    )
    t_int_s = obs['integration_time_sec']

    # ── 7b. On-source pointing statistics (actual integration timestamps) ──────
    # The azel grid above spans from first to last integration timestamp at a
    # uniform step, which includes inter-scan gaps.  Here we extract the actual
    # on-source integration timestamps and re-evaluate el/HA/PA at each one to
    # produce statistics that reflect only the time the telescope was on source.
    from astropy.coordinates import EarthLocation as _ELq, AltAz as _AAq
    from astropy.time import Time as _TTq

    _id_to_nm_q   = index['id_to_name']
    _s_lower_q    = str(source).strip().lower()
    _obs_ids_q    = np.array(
        [k for k, v in _id_to_nm_q.items() if v.strip().lower() == _s_lower_q],
        dtype=np.int32)
    if _obs_ids_q.size == 0:
        _obs_ids_q = np.array(
            [k for k, v in _id_to_nm_q.items() if _s_lower_q in v.strip().lower()],
            dtype=np.int32)

    _jd_rows_q = np.sort(index['jd'][np.isin(index['source_id'], _obs_ids_q)])
    # Deduplicate baseline-sharing rows into one timestamp per integration
    # (use 5 s gap; GMRT integrations are 2–16 s so this is safe)
    if _jd_rows_q.size > 1:
        _keep_q = np.concatenate([[True], np.diff(_jd_rows_q) > 5.0 / 86400.0])
        _jd_int = _jd_rows_q[_keep_q]
    elif _jd_rows_q.size == 1:
        _jd_int = _jd_rows_q.copy()
    else:
        _jd_int = np.array([], dtype=np.float64)

    if _jd_int.size > 0:
        _loc_q    = _ELq.from_geocentric(*geo['ecef_centre'], unit=_u.m)
        _coord_q  = SkyCoord(ra=ra_deg * _u.deg, dec=dec_deg * _u.deg, frame='icrs')
        _t_int_q  = _TTq(_jd_int, format='jd', scale='utc')
        _altaz_q  = _coord_q.transform_to(_AAq(obstime=_t_int_q, location=_loc_q))
        _el_int   = np.array(_altaz_q.alt.deg)
        _ha_int   = _hour_angle_deg(_jd_int.copy(), ra_deg, lon_deg)
        _pa_int   = _parallactic_angle_deg(_ha_int, dec_deg, lat_deg)
        _utc_int  = [t.iso for t in _t_int_q]
    else:
        _el_int  = np.array([])
        _ha_int  = np.array([])
        _pa_int  = np.array([])
        _utc_int = []

    # On-source elevation stats
    el_on_source_time_above = {}
    el_on_source_windows    = {}
    for thr in elevation_thresholds_deg:
        _mask_el = _el_int >= thr if _el_int.size > 0 else np.array([], dtype=bool)
        on_src_t = round(float(np.sum(_mask_el)) * t_int_s / 60.0, 1)
        el_on_source_time_above[f'on_source_time_above_{int(thr)}_deg_min'] = on_src_t
        # UTC windows: contiguous on-source blocks above threshold
        _wins = []
        if _mask_el.size > 0 and np.any(_mask_el):
            _in_win = False
            _tw0 = None
            for _ii, _ok in enumerate(_mask_el):
                if _ok and not _in_win:
                    _tw0 = _utc_int[_ii]; _in_win = True
                elif not _ok and _in_win:
                    _wins.append((_tw0, _utc_int[_ii - 1])); _in_win = False
            if _in_win:
                _wins.append((_tw0, _utc_int[-1]))
        el_on_source_windows[f'on_source_windows_above_{int(thr)}_deg'] = _wins

    # On-source HA range
    if _ha_int.size > 0:
        ha_on_src_min  = float(_ha_int.min())
        ha_on_src_max  = float(_ha_int.max())
        # Cumulative in-scan HA swept: sum |dHA| only across consecutive integration
        # pairs that belong to the same scan (gap < 3 × t_int_s).  This avoids
        # counting the large HA jump across inter-scan gaps, so the result is
        # always ≤ total on-source time × 15°/hr and never equals the full
        # first-to-last timespan when there are gaps.
        if _ha_int.size > 1:
            _jd_gap_s = np.diff(_jd_int) * 86400.0
            _in_scan  = _jd_gap_s < 3.0 * t_int_s
            _dha = np.diff(_ha_int)
            _dha[_dha >  180] -= 360
            _dha[_dha < -180] += 360
            ha_on_src_range_deg = float(np.sum(np.abs(_dha[_in_scan])))
        else:
            ha_on_src_range_deg = 0.0
        # Integration closest to transit (HA = 0)
        _i_tr_int = int(np.argmin(np.abs(_ha_int)))
        ha_transit_closest_deg = float(_ha_int[_i_tr_int])
        transit_closest_utc    = _utc_int[_i_tr_int]
        # Sampled if nearest integration is within 1.5 × integration time of HA=0
        _ha_tol_deg = (t_int_s * 1.5) * (15.0 / 3600.0)  # 15 deg/hr × t_int_s
        transit_sampled = abs(ha_transit_closest_deg) <= _ha_tol_deg
    else:
        ha_on_src_min  = ha_on_src_max  = ha_on_src_range_deg = 0.0
        ha_transit_closest_deg = float('nan')
        transit_closest_utc    = 'N/A'
        transit_sampled        = False

    # On-source PA stats
    if _pa_int.size > 1:
        _dpa_int = np.diff(_pa_int)
        _dpa_int[_dpa_int >  180] -= 360
        _dpa_int[_dpa_int < -180] += 360
        _rates_int            = np.abs(_dpa_int) / (t_int_s / 60.0)
        _i_max_pa             = int(np.argmax(_rates_int))
        pa_rate_on_src        = float(_rates_int[_i_max_pa])
        pa_rate_on_src_utc    = _utc_int[_i_max_pa]
    else:
        pa_rate_on_src     = 0.0
        pa_rate_on_src_utc = _utc_int[0] if _utc_int else 'N/A'

    # ── 8. UV statistics ───────────────────────────────────────────────────────
    uv_stats = _uv_stats_for_source(index, source)

    # ── 9. Physical baselines from ENU ────────────────────────────────────────
    bl_stats = _physical_baseline_stats(geo, dud_nos=_query_dud_nos)

    # ── 10. File metadata ──────────────────────────────────────────────────────
    import os
    file_size_gb = os.path.getsize(fits_path) / 1e9

    # ── 11. Sensitivity geometric factors ─────────────────────────────────────
    # Use active (non-dud) antenna count for noise formula
    n_ant  = (len(index['active_antennas'])
              if index and 'active_antennas' in index
              else len(geo['nos']) - len(_query_dud_nos))
    n_cross_bl = n_ant * (n_ant - 1) // 2
    n_pol  = obs['n_correlations']
    # t_int_s already set in step 7
    delta_nu_hz = freq_props['bandwidth_hz_estimated']
    noise_formula = (
        f'σ = SEFD / sqrt({n_pol} · {n_cross_bl} · '
        f'{delta_nu_hz/1e6:.1f}MHz · t_s)'
    )

    # ── 12. Data-selection advice ─────────────────────────────────────────────
    # Recommend elevation cut based on time-above statistics
    recommended_el = None
    for thr in sorted(elevation_thresholds_deg):
        key = f'time_above_{int(thr)}_deg_min'
        if el_time_above.get(key, 0) >= 20.0:   # at least 20 min above threshold
            recommended_el = thr
    # Recommended UV range derived from p2–p98 of the actual UV distribution
    # Get percentiles from raw arrays if available
    uu_raw = uv_stats.get('uu_m_raw')
    vv_raw = uv_stats.get('vv_m_raw')
    c_mps  = 299_792_458.0
    if uu_raw is not None and len(uu_raw) > 0:
        uv_rad_m = np.hypot(uu_raw, vv_raw)
        uv_rad_klam_all = uv_rad_m / (c_mps / nu_centre_hz) * 1e-3
        uv_adv_min_klam = round(float(np.percentile(uv_rad_klam_all, 2)), 2)
        uv_adv_max_klam = round(float(np.percentile(uv_rad_klam_all, 98)), 2)
    else:
        uv_adv_min_klam = uv_stats.get('bmin_klambda', 0.0)
        uv_adv_max_klam = uv_stats.get('bmax_klambda', 999.9)

    # Best time window: UTC range where elevation >= recommended_el (or 30°)
    el_cut_advice = recommended_el if recommended_el is not None else 30.0
    above_mask = el_deg >= el_cut_advice
    best_windows = []
    in_win = False
    t0_win = None
    for ii, ok in enumerate(above_mask):
        if ok and not in_win:
            t0_win = azel['utc'][ii]; in_win = True
        elif not ok and in_win:
            best_windows.append((t0_win, azel['utc'][ii - 1])); in_win = False
    if in_win and t0_win is not None:
        best_windows.append((t0_win, azel['utc'][-1]))

    # ── 13. Strip raw arrays if not wanted ────────────────────────────────────
    uv_export = {k: v for k, v in uv_stats.items()
                 if k not in ('uu_m_raw', 'vv_m_raw')}
    bl_export  = {k: v for k, v in bl_stats.items()
                  if k not in ('enu', 'names', 'nos')}

    result = {
        'source': {
            'name':         src_name,
            'ra_deg':       round(ra_deg, 6),
            'dec_deg':      round(dec_deg, 6),
            'gal_l_deg':    round(gal_l, 4),
            'gal_b_deg':    round(gal_b, 4),
        },
        'file': {
            'path':         fits_path,
            'size_gb':      round(file_size_gb, 3),
        },
        'frequency': {
            'centre_mhz':   round(nu_centre_hz / 1e6, 3),
            'min_mhz':      round(freq_props['freq_min_hz'] / 1e6, 3),
            'max_mhz':      round(freq_props['freq_max_hz'] / 1e6, 3),
            'bandwidth_mhz': round(freq_props['bandwidth_hz_estimated'] / 1e6, 3),
            'n_channels':   freq_props['nchan'],
            'chan_width_khz': round(abs(freq_props['chan_width_hz']) / 1e3, 4),
            'wavelength_m': round(c_mps / nu_centre_hz, 4),
        },
        'timing': {
            'begin_utc':              obs['begin_time_utc'],
            'end_utc':                obs['end_time_utc'],
            # timespan: clock time from first to last integration (includes gaps between scans)
            'total_duration_min':     round(obs['total_duration_sec'] / 60.0, 1),
            'total_duration_hr':      round(obs['total_duration_sec'] / 3600.0, 3),
            # on_source_time: n_integrations × integration_time — actual data accumulation
            'on_source_time_min':     round(obs['on_source_time_sec'] / 60.0, 1),
            'on_source_time_hr':      round(obs['on_source_time_sec'] / 3600.0, 3),
            'n_integrations':         obs['n_integrations'],
            'integration_time_s':     round(t_int_s, 2),
            'n_scans':                obs['n_scans'],
            'n_vis_rows':             obs['n_vis_rows'],
            'scans':                  obs['scans'],
        },
        'pointing': {
            'el_min_deg':          round(float(el_deg.min()), 2),
            'el_max_deg':          round(float(el_deg.max()), 2),
            'el_mean_deg':         round(float(el_deg.mean()), 2),
            'el_at_start_deg':     round(float(el_deg[0]), 2),
            'el_at_end_deg':       round(float(el_deg[-1]), 2),
            'az_grid_deg':         az_deg.tolist(),
            'el_grid_deg':         el_deg.tolist(),
            'utc_grid':            azel['utc'],
            # timespan-based thresholds (grid, includes scan gaps):
            **el_time_above,
            # on-source thresholds (actual integrations × t_int_s):
            **el_on_source_time_above,
            **el_on_source_windows,
        },
        'hour_angle': {
            'ha_at_start_deg':     round(float(ha_deg[0]), 3),
            'ha_at_start_min':     round(float(ha_mins[0]), 2),
            'ha_at_end_deg':       round(float(ha_deg[-1]), 3),
            'ha_at_end_min':       round(float(ha_mins[-1]), 2),
            'ha_range_deg':        round(float(ha_deg.max() - ha_deg.min()), 3),
            'ha_range_min':        round(float(ha_deg.max() - ha_deg.min()) * 4.0, 2),
            'ha_on_source_min_deg':   round(ha_on_src_min, 3),
            'ha_on_source_max_deg':   round(ha_on_src_max, 3),
            'ha_on_source_range_deg': round(ha_on_src_range_deg, 3),
            'ha_on_source_range_min': round(ha_on_src_range_deg * 4.0, 2),
            'transit_utc':         transit_utc,
            'ha_at_transit_deg':   round(ha_at_transit, 4),
            'transit_sampled':     transit_sampled,
            'transit_closest_integration_ha_deg': round(ha_transit_closest_deg, 4)
                                                   if not _jd_int.size == 0 else None,
            'transit_closest_integration_utc':    transit_closest_utc,
            'ha_grid_deg':         ha_deg.tolist(),
        },
        'parallactic_angle': {
            'pa_at_start_deg':     round(float(pa_deg[0]), 2),
            'pa_at_end_deg':       round(float(pa_deg[-1]), 2),
            'pa_min_deg':          round(float(pa_deg.min()), 2),
            'pa_max_deg':          round(float(pa_deg.max()), 2),
            'pa_range_deg':        round(float(pa_deg.max() - pa_deg.min()), 2),
            # Rate from uniform grid (includes unsampled times near transit):
            'pa_max_rate_deg_per_min':          round(pa_rate_deg_per_min, 4),
            # Rate from actual on-source integrations only:
            'pa_max_rate_on_source_deg_per_min': round(pa_rate_on_src, 4),
            'pa_max_rate_on_source_utc':         pa_rate_on_src_utc,
            'note': ('PA range relevant for polarisation leakage; '
                     'max rate at transit if source passes near zenith.'),
            'pa_grid_deg': pa_deg.tolist(),
        },
        'uv_coverage': uv_export,
        'physical_baselines': bl_export,
        'sensitivity': {
            'n_antennas':          n_ant,
            'n_cross_baselines':   n_cross_bl,
            'n_polarisations':     n_pol,
            'integration_time_s':  round(t_int_s, 2),
            'bandwidth_mhz':       round(delta_nu_hz / 1e6, 3),
            'noise_formula_hint':  noise_formula,
            'note': ('Multiply by on-source time in seconds to get image noise. '
                     'Provide SEFD (Jy) for your band to get σ in Jy/beam.'),
        },
        'data_selection_advice': {
            'recommended_elevation_min_deg':   el_cut_advice,
            'recommended_uvrange_klambda':     (uv_adv_min_klam, uv_adv_max_klam),
            'best_time_windows_above_el_cut':  best_windows,
            'set_strings': {
                'SOLVE_ELEVATION_MIN_DEG': f'--set "SOLVE_ELEVATION_MIN_DEG={el_cut_advice}"',
                'SOLVE_UVRANGE_KLAMBDA':   f'--set "SOLVE_UVRANGE_KLAMBDA=({uv_adv_min_klam}, {uv_adv_max_klam})"',
                'SOLVE_TIMERANGE':         (
                    f'--set "SOLVE_TIMERANGE=(\'{best_windows[0][0]}\', \'{best_windows[0][1]}\')"'
                    if best_windows else 'No window above elevation cut found.'
                ),
            },
        },
        # Keep raw arrays attached for plot_source_query (not JSON-serialisable)
        '_raw': {
            'uu_m':   uu_raw if include_uv_raw else None,
            'vv_m':   vv_raw if include_uv_raw else None,
            'enu':    geo['enu'] if include_uv_raw else None,
            'ant_names': geo['names'],
            'ant_nos':   geo['nos'],
            'jd_grid':   jd_grid,
        },
    }
    return result


def print_source_query(result: dict) -> None:
    """Pretty-print the output of :func:`query_source` to stdout.

    Parameters
    ----------
    result : dict
        The dict returned by :func:`query_source`.

    Examples
    --------
    >>> result = uq.query_source(idx, '3C48')
    >>> uq.print_source_query(result)
    """
    W = 70
    SEP = '─' * W

    def hdr(title):
        print(f'\n{SEP}')
        print(f'  {title}')
        print(SEP)

    def row(label, value, unit=''):
        u = f'  {unit}' if unit else ''
        print(f'  {label:<42s} {value}{u}')

    src  = result['source']
    freq = result['frequency']
    tim  = result['timing']
    pt   = result['pointing']
    ha   = result['hour_angle']
    pa   = result['parallactic_angle']
    uv   = result['uv_coverage']
    bl   = result['physical_baselines']
    sens = result['sensitivity']
    adv  = result['data_selection_advice']
    fil  = result['file']

    print(f'\n{"━"*W}')
    print(f'  SOURCE QUERY REPORT  ·  {src["name"]}')
    print(f'{"━"*W}')

    hdr('SOURCE & FILE')
    row('Source',           src['name'])
    row('RA (J2000)',        f'{src["ra_deg"]:.4f}°  ({_deg_to_hms(src["ra_deg"])})')
    row('Dec (J2000)',       f'{src["dec_deg"]:.4f}°  ({_deg_to_dms(src["dec_deg"])})')
    row('Galactic (l, b)',   f'({src["gal_l_deg"]:.2f}°, {src["gal_b_deg"]:.2f}°)')
    row('File',              fil['path'])
    row('File size',         f'{fil["size_gb"]:.2f}', 'GB')

    hdr('FREQUENCY')
    row('Centre frequency',  f'{freq["centre_mhz"]:.3f}', 'MHz')
    row('Band',              f'{freq["min_mhz"]:.3f} – {freq["max_mhz"]:.3f}', 'MHz')
    row('Bandwidth',         f'{freq["bandwidth_mhz"]:.3f}', 'MHz')
    row('Channels',          f'{freq["n_channels"]}  ×  {freq["chan_width_khz"]:.1f} kHz/chan')
    row('Wavelength (λ)',    f'{freq["wavelength_m"]:.4f}', 'm')

    hdr('TIMING')
    row('Observation start',     tim['begin_utc'])
    row('Observation end',       tim['end_utc'])
    row('Time span (first–last)',
        f'{tim["total_duration_hr"]:.3f} hr  ({tim["total_duration_min"]:.1f} min)',
        '← includes gaps between scans')
    row('On-source time (Σ ints)',
        f'{tim["on_source_time_hr"]:.3f} hr  ({tim["on_source_time_min"]:.1f} min)',
        '← actual data accumulation')
    if tim.get('on_source_time_min', 0) < tim.get('total_duration_min', 0) * 0.8:
        print(f'  *** NOTE: on-source time is much less than timespan — '
              f'source was observed in {tim["n_scans"]} separate scan(s) ***')
    row('Integration time',      f'{tim["integration_time_s"]:.1f}', 's')
    row('Integrations / scans',  f'{tim["n_integrations"]}  /  {tim["n_scans"]}')
    row('Total visibility rows', f'{tim["n_vis_rows"]:,}')
    if tim['scans']:
        print(f'\n  {"Scan":>4}  {"Start UTC":>22}  {"End UTC":>22}  {"Dur(s)":>7}  {"N_int":>6}')
        print(f'  {"-"*4}  {"-"*22}  {"-"*22}  {"-"*7}  {"-"*6}')
        for s in tim['scans']:
            print(f'  {s["scan_index"]:>4}  {s["start_utc"]:>22}  {s["end_utc"]:>22}  '
                  f'{s["duration_sec"]:>7.0f}  {s["n_integrations"]:>6}')

    hdr('POINTING  (Az / Elevation)')
    row('El at obs start',   f'{pt["el_at_start_deg"]:.1f}°')
    row('El at obs end',     f'{pt["el_at_end_deg"]:.1f}°')
    row('El min / mean / max', f'{pt["el_min_deg"]:.1f}° / {pt["el_mean_deg"]:.1f}° / {pt["el_max_deg"]:.1f}°')
    print()
    thr_keys = sorted([k for k in pt if k.startswith('time_above_')])
    print(f'  {"El threshold":>12}  {"On-source (min)":>15}  {"Timespan (min)":>14}  On-source UTC window (first → last integration)')
    print(f'  {"-"*12}  {"-"*15}  {"-"*14}  {"-"*48}')
    for key in thr_keys:
        thr_str = key.replace('time_above_', '').replace('_deg_min', '')
        lbl = thr_str + '°'
        on_src_key  = f'on_source_time_above_{thr_str}_deg_min'
        win_key     = f'on_source_windows_above_{thr_str}_deg'
        on_src_t    = pt.get(on_src_key, float('nan'))
        wins        = pt.get(win_key, [])
        if wins:
            utc0 = wins[0][0][11:19]    # HH:MM:SS of first window start
            utc1 = wins[-1][1][11:19]   # HH:MM:SS of last window end
            n_gaps = len(wins) - 1
            win_str = f'{utc0} → {utc1}'
            if n_gaps > 0:
                win_str += f'  ({n_gaps} gap{"s" if n_gaps > 1 else ""})'  
        else:
            win_str = '—'
        print(f'  {lbl:>12}  {on_src_t:>15.1f}  {pt[key]:>14.1f}  {win_str}')

    hdr('HOUR ANGLE')
    row('HA at start',       f'{ha["ha_at_start_deg"]:+.2f}°  ({ha["ha_at_start_min"]:+.1f} min)')
    row('HA at end',         f'{ha["ha_at_end_deg"]:+.2f}°  ({ha["ha_at_end_min"]:+.1f} min)')
    row('HA range (timespan)',
        f'{ha["ha_range_deg"]:.2f}°  ({ha["ha_range_min"]:.1f} min)',
        '← includes scan gaps')
    row('HA sweep (on-source, Σ scans)',
        f'{ha["ha_on_source_range_deg"]:.2f}°  ({ha["ha_on_source_range_min"]:.1f} min)',
        '← cumulative in-scan rotation only')
    row('Transit UTC',       ha['transit_utc'])
    row('HA at transit',     f'{ha["ha_at_transit_deg"]:+.3f}°')
    _sampled = ha.get('transit_sampled', False)
    _cl_ha   = ha.get('transit_closest_integration_ha_deg')
    _cl_utc  = ha.get('transit_closest_integration_utc', 'N/A')
    _sam_str = 'YES' if _sampled else 'NO'
    if _cl_ha is not None and not (_cl_ha != _cl_ha):   # not NaN
        _sam_str += f'  (nearest integration: HA={_cl_ha:+.3f}°  at {_cl_utc[11:19]})'
    row('Transit sampled?',  _sam_str)

    hdr('PARALLACTIC ANGLE  (polarisation planning)')
    row('PA at start / end', f'{pa["pa_at_start_deg"]:+.1f}° / {pa["pa_at_end_deg"]:+.1f}°')
    row('PA range',          f'{pa["pa_range_deg"]:.1f}°')
    row('Max PA rate (timespan)',
        f'{pa["pa_max_rate_deg_per_min"]:.3f}  deg/min',
        '← grid incl. unsampled intervals')
    _on_src_pa_rate = pa.get('pa_max_rate_on_source_deg_per_min', float('nan'))
    _on_src_pa_utc  = pa.get('pa_max_rate_on_source_utc', 'N/A')
    row('Max PA rate (on-source)',
        f'{_on_src_pa_rate:.3f}  deg/min  @ {_on_src_pa_utc[11:19] if len(str(_on_src_pa_utc)) > 11 else _on_src_pa_utc}',
        '← actual integrations only')
    if not ha.get('transit_sampled', True):
        print(f'  *** NOTE: transit was NOT sampled — true max PA rate (near zenith) may be'
              f' higher than measured. ***')
    print(f'\n  Note: {pa["note"]}')

    hdr('UV COVERAGE')
    row('B_min',  f'{uv.get("bmin_m", "—"):.1f} m  =  {uv.get("bmin_klambda", "—"):.2f} kλ')
    row('B_median', f'{uv.get("bmedian_m", "—"):.1f} m  =  {uv.get("bmedian_klambda", "—"):.2f} kλ')
    row('B_max',  f'{uv.get("bmax_m", "—"):.1f} m  =  {uv.get("bmax_klambda", "—"):.2f} kλ')
    row('Angular resolution (≈ λ/2B_max)', f'{uv.get("angular_resolution_arcsec", "—"):.2f}", ')
    row('Largest angular scale',  f'{uv.get("largest_angular_scale_arcmin", "—"):.2f}′')
    row('UV-plane fill fraction', f'{uv.get("uv_filling_pct_50bins", "—"):.1f}%  (50×50 grid)')

    hdr('PHYSICAL BASELINES  (dish separations, freq-independent)')
    row('Antennas',          f'{bl.get("n_antennas", "—")}')
    row('Physical baselines',f'{bl.get("n_physical_baselines", "—")}')
    row('Shortest pair',     f'{bl.get("shortest_pair","—")}  =  {bl.get("shortest_m","—")} m')
    row('Median separation', f'{bl.get("median_m","—")} m')
    row('Longest pair',      f'{bl.get("longest_pair","—")}  =  {bl.get("longest_m","—")} m')
    print(f'\n  5 shortest pairs (risk of standing-wave / shadowing):')
    for pair, d in bl.get('five_shortest_pairs', []):
        print(f'    {pair:<20s}  {d:.0f} m')

    hdr('SENSITIVITY  (geometric factors)')
    row('Antennas',           f'{sens["n_antennas"]}')
    row('Cross-baselines',    f'{sens["n_cross_baselines"]}')
    row('Polarisations',      f'{sens["n_polarisations"]}')
    row('Δt (integration)',   f'{sens["integration_time_s"]:.1f}', 's')
    row('Δν (bandwidth)',     f'{sens["bandwidth_mhz"]:.1f}', 'MHz')
    print(f'\n  Noise formula:  {sens["noise_formula_hint"]}')
    print(f'  ({sens["note"]})')

    hdr('DATA SELECTION ADVICE')
    row('Recommended elevation cut', f'{adv["recommended_elevation_min_deg"]}°')
    row('Recommended UV range',     f'{adv["recommended_uvrange_klambda"][0]} → {adv["recommended_uvrange_klambda"][1]} kλ')
    print()
    print('  Best time windows (elevation ≥ cut):')
    for w0, w1 in adv['best_time_windows_above_el_cut']:
        print(f'    {w0}  →  {w1}')
    print()
    print('  Shell --set strings:')
    for key, val in adv['set_strings'].items():
        print(f'    {val}')

    print(f'\n{"━"*W}\n')


def _deg_to_hms(deg: float) -> str:
    """Format decimal degrees as HH:MM:SS.s (RA convention)."""
    total_s = (deg % 360.0) * 3600.0 / 15.0
    h  = int(total_s // 3600)
    m  = int((total_s % 3600) // 60)
    s  = total_s % 60
    return f'{h:02d}h{m:02d}m{s:05.2f}s'


def _deg_to_dms(deg: float) -> str:
    """Format decimal degrees as ±DD:MM:SS (Dec convention)."""
    sign = '+' if deg >= 0 else '-'
    ad   = abs(deg)
    d    = int(ad)
    m    = int((ad - d) * 60)
    s    = ((ad - d) * 60 - m) * 60
    return f'{sign}{d:02d}d{m:02d}m{s:04.1f}s'


def plot_source_query(
    result: dict,
    figsize: Tuple[float, float] = (18, 14),
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 120,
) -> 'plt.Figure':
    """Four-panel diagnostic plot from :func:`query_source` output.

    Panels
    ------
    Top-left  : Elevation and Azimuth vs UT time
    Top-right : UV coverage (u–v plane, cross-baselines only)
    Bot-left  : Hour angle and Parallactic angle vs UT time
    Bot-right : Histogram of baseline lengths (metres and kλ)

    Parameters
    ----------
    result : dict
        Return value of :func:`query_source`.
    figsize : (float, float)
        Figure width × height in inches.
    save_path : str | Path, optional
        If given, save the figure to this path.
    dpi : int
        Resolution for saving.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> fig = uq.plot_source_query(result)
    >>> fig.savefig('3c48_query.png', dpi=150, bbox_inches='tight')
    """
    import matplotlib
    # Do NOT call matplotlib.use() here — the caller (notebook or script) sets
    # the backend via %matplotlib inline / %matplotlib widget / savefig().
    # Forcing Agg here would break interactive display for all subsequent cells.
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import AutoMinorLocator

    src  = result['source']
    freq = result['frequency']
    pt   = result['pointing']
    ha   = result['hour_angle']
    pa   = result['parallactic_angle']
    uv   = result['uv_coverage']
    adv  = result['data_selection_advice']
    raw  = result.get('_raw', {})

    utc_strs = pt['utc_grid']
    from astropy.time import Time as _T
    t_dt = [_T(s, format='iso', scale='utc').to_datetime() for s in utc_strs]

    el   = np.array(pt['el_grid_deg'])
    az   = np.array(pt['az_grid_deg'])
    ha_v = np.array(ha['ha_grid_deg'])
    pa_v = np.array(pa['pa_grid_deg'])

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.patch.set_facecolor('#1a1a2e')
    for ax in axes.flat:
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='#cfd8dc', labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor('#37474f')
        ax.xaxis.label.set_color('#cfd8dc')
        ax.yaxis.label.set_color('#cfd8dc')
        ax.title.set_color('#eceff1')

    # ── Panel A: Elevation + Azimuth ──────────────────────────────────────────
    ax_el = axes[0, 0]
    ax_az = ax_el.twinx()
    ax_az.set_facecolor('#16213e')
    ax_az.tick_params(colors='#90a4ae', labelsize=8)
    ax_az.spines['right'].set_edgecolor('#455a64')
    ax_az.yaxis.label.set_color('#90a4ae')

    ax_el.plot(t_dt, el, color='#4fc3f7', lw=2, label='Elevation')
    ax_az.plot(t_dt, az, color='#ffb74d', lw=1.2, ls='--', alpha=0.7, label='Azimuth')

    el_cut = adv['recommended_elevation_min_deg']
    ax_el.axhline(el_cut, color='#ef5350', ls=':', lw=1.4,
                  label=f'El cut ({el_cut}°)')
    ax_el.fill_between(t_dt, el_cut, el,
                       where=(el >= el_cut), alpha=0.15, color='#4fc3f7')

    ax_el.set_xlabel('UTC')
    ax_el.set_ylabel('Elevation (°)')
    ax_az.set_ylabel('Azimuth (°)')
    ax_el.set_title(f'{src["name"]}  ·  Elevation & Azimuth track  '
                    f'·  {freq["centre_mhz"]:.0f} MHz')
    ax_el.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax_el.xaxis.set_minor_locator(AutoMinorLocator())
    ax_el.yaxis.set_minor_locator(AutoMinorLocator())
    lines_a  = ax_el.get_lines() + ax_az.get_lines()
    labels_a = [l.get_label() for l in lines_a]
    ax_el.legend(lines_a, labels_a, fontsize=8, facecolor='#1a1a2e', labelcolor='#eceff1')
    fig.autofmt_xdate(rotation=30, ha='right')

    # ── Panel B: UV coverage ──────────────────────────────────────────────────
    ax_uv = axes[0, 1]
    uu_m = raw.get('uu_m')
    vv_m = raw.get('vv_m')
    if uu_m is not None and len(uu_m) > 0:
        step = max(1, len(uu_m) // 15_000)
        uu_s = uu_m[::step] / 1e3   # km for plotting
        vv_s = vv_m[::step] / 1e3
        ax_uv.scatter( uu_s,  vv_s, s=0.5, c='#4dd0e1', alpha=0.4, rasterized=True)
        ax_uv.scatter(-uu_s, -vv_s, s=0.5, c='#80cbc4', alpha=0.4, rasterized=True)
    ax_uv.set_xlabel('u  (km)')
    ax_uv.set_ylabel('v  (km)')
    ax_uv.set_title(f'UV coverage  ·  {uv.get("angular_resolution_arcsec","—"):.2f}″ resolution  '
                    f'/  {uv.get("largest_angular_scale_arcmin","—"):.1f}′ LAS')
    ax_uv.set_aspect('equal')
    ax_uv.axhline(0, color='#37474f', lw=0.5)
    ax_uv.axvline(0, color='#37474f', lw=0.5)
    ax_uv.text(0.02, 0.97,
               f'B_max = {uv.get("bmax_m","—"):.0f} m  /  {uv.get("bmax_klambda","—"):.1f} kλ\n'
               f'B_min = {uv.get("bmin_m","—"):.0f} m  /  {uv.get("bmin_klambda","—"):.1f} kλ\n'
               f'Fill = {uv.get("uv_filling_pct_50bins","—"):.1f}%',
               transform=ax_uv.transAxes, va='top', fontsize=8,
               color='#b0bec5', family='monospace',
               bbox=dict(facecolor='#0d0d1a', alpha=0.7, edgecolor='none', pad=4))

    # ── Panel C: Hour angle + Parallactic angle ───────────────────────────────
    ax_ha = axes[1, 0]
    ax_pa = ax_ha.twinx()
    ax_pa.set_facecolor('#16213e')
    ax_pa.tick_params(colors='#a5d6a7', labelsize=8)
    ax_pa.spines['right'].set_edgecolor('#455a64')
    ax_pa.yaxis.label.set_color('#a5d6a7')

    ax_ha.plot(t_dt, ha_v * 4.0, color='#ce93d8', lw=2, label='Hour angle')
    ax_pa.plot(t_dt, pa_v,        color='#a5d6a7', lw=1.5, ls='--', label='Parallactic angle')
    ax_ha.axhline(0, color='#546e7a', lw=0.8, ls=':')

    ax_ha.set_xlabel('UTC')
    ax_ha.set_ylabel('Hour angle (min)')
    ax_pa.set_ylabel('Parallactic angle (°)')
    ax_ha.set_title(f'Hour angle  &  Parallactic angle')
    ax_ha.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax_ha.xaxis.set_minor_locator(AutoMinorLocator())
    lines_c  = ax_ha.get_lines() + ax_pa.get_lines()
    labels_c = [l.get_label() for l in lines_c]
    ax_ha.legend(lines_c, labels_c, fontsize=8, facecolor='#1a1a2e', labelcolor='#eceff1')
    fig.autofmt_xdate(rotation=30, ha='right')

    # ── Panel D: Baseline length histogram ────────────────────────────────────
    ax_bl = axes[1, 1]
    if uu_m is not None and len(uu_m) > 0:
        blen_m = np.hypot(uu_m, vv_m)
        ax_bl.hist(blen_m / 1e3, bins=60, color='#4fc3f7', alpha=0.75,
                   edgecolor='#1a1a2e', linewidth=0.3)
        ax_bl2 = ax_bl.twiny()
        ax_bl2.set_facecolor('#16213e')
        ax_bl2.tick_params(colors='#ffcc80', labelsize=8)
        lam_m = freq['wavelength_m']
        lo_km = ax_bl.get_xlim()[0]
        hi_km = ax_bl.get_xlim()[1]
        ax_bl2.set_xlim(lo_km * 1e3 / lam_m * 1e-3,
                        hi_km * 1e3 / lam_m * 1e-3)
        ax_bl2.set_xlabel('Baseline length (kλ)', color='#ffcc80')
    ax_bl.set_xlabel('Baseline length (km)')
    ax_bl.set_ylabel('Count')
    ax_bl.set_title(f'Baseline length distribution')
    ax_bl.xaxis.set_minor_locator(AutoMinorLocator())
    ax_bl.yaxis.set_minor_locator(AutoMinorLocator())

    fig.suptitle(
        f'Source Query: {src["name"]}  ·  '
        f'RA {src["ra_deg"]:.3f}°  Dec {src["dec_deg"]:.3f}°  '
        f'(l={src["gal_l_deg"]:.1f}°, b={src["gal_b_deg"]:.1f}°)',
        fontsize=12, color='#eceff1', y=1.01,
    )
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(str(save_path), dpi=dpi, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f'Saved: {save_path}')

    return fig


def plot_antenna_positions(
    fits_path_or_index,
    flagged_antenna_ids: Optional[List[int]] = None,
    flagged_antenna_names: Optional[List[str]] = None,
    highlight_pairs_closer_than_m: Optional[float] = None,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize=(10, 10),
) -> 'plt.Figure':
    """Plot GMRT antenna positions (ENU) from the AIPS AN table.

    Parameters
    ----------
    fits_path_or_index : str | Path | dict
        FITS path, or row index dict.
    flagged_antenna_ids : list[int], optional
        Antenna numbers (NOSTA) to circle in red as flagged.
    flagged_antenna_names : list[str], optional
        Antenna names to circle instead of / in addition to IDs.
    highlight_pairs_closer_than_m : float, optional
        Draw a line between any pair of antennas within this distance (m)
        and annotate with the implied standing-wave period.
    title : str, optional
    save_path : str | Path, optional
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    fits_path = fits_path_or_index['path'] if isinstance(fits_path_or_index, dict) \
                else fits_path_or_index
    geo    = _get_array_enu(fits_path)
    nos    = geo['nos']
    names  = geo['names']
    enu    = geo['enu']   # (N, 3)
    east   = enu[:, 0]
    north  = enu[:, 1]

    # Normalise flagged sets
    _flag_ids   = set(flagged_antenna_ids or [])
    _flag_names = {_norm_antenna_name(n) for n in (flagged_antenna_names or [])}
    def _is_flagged(no, nm):
        return (no in _flag_ids) or (_norm_antenna_name(nm) in _flag_names)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    # Optionally draw inter-dish separation lines
    if highlight_pairs_closer_than_m is not None:
        for i in range(len(nos)):
            for j in range(i + 1, len(nos)):
                d = float(np.hypot(east[i] - east[j], north[i] - north[j]))
                if d < highlight_pairs_closer_than_m and d > 1.0:
                    ax.plot([east[i], east[j]], [north[i], north[j]],
                            color='orange', lw=1.2, alpha=0.6, zorder=1)
                    mid_e = 0.5 * (east[i] + east[j])
                    mid_n = 0.5 * (north[i] + north[j])
                    period = 2.998e8 / (2 * d) / 1e6
                    ax.text(mid_e, mid_n, f'{d:.0f} m\n{period:.2f} MHz',
                            ha='center', va='bottom', fontsize=6,
                            color='darkorange', zorder=5)

    # Plot antennas
    for no, nm, e, n in zip(nos, names, east, north):
        flagged = _is_flagged(no, nm)
        color = 'tomato' if flagged else 'steelblue'
        ax.scatter(e, n, s=60, color=color, zorder=3)
        ax.annotate(f'{nm}\n({no})', (e, n),
                    textcoords='offset points', xytext=(4, 4),
                    fontsize=7, color='black', zorder=4)
        if flagged:
            circ = plt.Circle((e, n), radius=30, fill=False,
                               edgecolor='tomato', lw=2.0, zorder=4)
            ax.add_patch(circ)

    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.25)
    ax.set_title(title or f'Antenna positions — {Path(str(fits_path)).name}')
    if flagged_antenna_ids or flagged_antenna_names:
        from matplotlib.lines import Line2D as _L2D
        ax.legend(handles=[
            _L2D([0],[0], marker='o', color='w', mfc='steelblue',  ms=8, label='Active'),
            _L2D([0],[0], marker='o', color='w', mfc='tomato',     ms=8, label='Flagged'),
        ], fontsize=9)

    if save_path is not None:
        fig.savefig(str(save_path), dpi=120)
    return fig


# ---------------------------------------------------------------------------
# Data selection / slicer — reusable filter spec for all loading functions
# ---------------------------------------------------------------------------

class DataSelection:
    """Reusable filter specification passed to loading and calibration functions.

    All fields are optional.  Any combination can be used simultaneously.

    Parameters
    ----------
    timerange : (start, end) optional
        JD float pair, or ISO string pair ('YYYY-MM-DD HH:MM:SS').
        Rows outside this range are dropped.
    chan_range : (first, last) optional
        0-based inclusive channel indices (same as the existing ``chan_range``
        parameter throughout the codebase).
    uvrange_m : (min_m, max_m) optional
        UV-distance range in metres.  Rows outside are dropped.
    uvrange_klambda : (min_kλ, max_kλ) optional
        UV-distance range in kilolambda (using the band centre frequency).
        If both ``uvrange_m`` and ``uvrange_klambda`` are given, the metres
        limit is applied first and kilolambda second.
    elevation_min_deg : float, optional
        Drop rows observed below this elevation (degrees).
    elevation_max_deg : float, optional
        Drop rows observed above this elevation (degrees).
    ant_list : list[int], optional
        Restrict to these antenna numbers (replaces/overrides the top-level
        ``ant_list`` argument if both are given, with the selection taking
        precedence).

    Examples
    --------
    # Use only data above 30° elevation and within 2–50 kλ:
    sel = DataSelection(elevation_min_deg=30.0,
                        uvrange_klambda=(2.0, 50.0))

    # Use only the first 2 hours of a 3C48 scan:
    sel = DataSelection(timerange=('2021-07-25 19:00:00', '2021-07-25 21:00:00'))

    # Pass to any loading or calibration function:
    vis = load_vis_for_source(index, '3C48', selection=sel)
    sol = derive_point_source_bandpass(index, '3C48', selection=sel)
    run_iterative_bandpass_workflow(..., selection=sel)
    """

    def __init__(
        self,
        timerange: Optional[Tuple] = None,
        chan_range: Optional[Tuple[int, int]] = None,
        uvrange_m: Optional[Tuple[float, float]] = None,
        uvrange_klambda: Optional[Tuple[float, float]] = None,
        elevation_min_deg: Optional[float] = None,
        elevation_max_deg: Optional[float] = None,
        ant_list: Optional[List[int]] = None,
    ):
        self.timerange         = timerange
        self.chan_range        = chan_range
        self.uvrange_m         = uvrange_m
        self.uvrange_klambda   = uvrange_klambda
        self.elevation_min_deg = elevation_min_deg
        self.elevation_max_deg = elevation_max_deg
        self.ant_list          = ant_list

    # normalise JD bounds (accept ISO strings)
    def jd_bounds(self) -> Optional[Tuple[float, float]]:
        if self.timerange is None:
            return None
        from astropy.time import Time as _T
        t0, t1 = self.timerange
        if isinstance(t0, str):
            t0 = _T(t0, format='iso', scale='utc').jd
        if isinstance(t1, str):
            t1 = _T(t1, format='iso', scale='utc').jd
        return (float(t0), float(t1))

    def __repr__(self):
        parts = []
        if self.timerange:         parts.append(f'timerange={self.timerange}')
        if self.chan_range:         parts.append(f'chan_range={self.chan_range}')
        if self.uvrange_m:          parts.append(f'uvrange_m={self.uvrange_m}')
        if self.uvrange_klambda:    parts.append(f'uvrange_klambda={self.uvrange_klambda}')
        if self.elevation_min_deg:  parts.append(f'el_min={self.elevation_min_deg}°')
        if self.elevation_max_deg:  parts.append(f'el_max={self.elevation_max_deg}°')
        if self.ant_list:           parts.append(f'ant_list={self.ant_list}')
        return f'DataSelection({", ".join(parts) or "no filters"})'


# ---------------------------------------------------------------------------
# Visibility loader — contiguous block I/O via the row index
# ---------------------------------------------------------------------------

def load_vis_for_source(
    index,
    source,
    ant_range=None,
    ant_list=None,
    chan_range=None,
    stokes=None,
    max_rows=150_000,
    flag_all_corrs_if_any_rawvis_flagged: bool = False,
    # ── DataSelection convenience params ─────────────────────────────────────
    selection: Optional['DataSelection'] = None,
    timerange: Optional[Tuple] = None,
    uvrange_m: Optional[Tuple[float, float]] = None,
    uvrange_klambda: Optional[Tuple[float, float]] = None,
    elevation_min_deg: Optional[float] = None,
    elevation_max_deg: Optional[float] = None,
):
    """Load visibility data using the pre-built row index.

    Filtering is applied in this order:
    1. Antenna filter (``ant_list`` / ``ant_range``)
    2. Time range     (``timerange`` or ``selection.timerange``)
    3. UV-distance    (``uvrange_m`` / ``uvrange_klambda``)
    4. Elevation      (``elevation_min_deg`` / ``elevation_max_deg``)

    A :class:`DataSelection` object can be passed as ``selection`` to specify
    all filters in one place. Explicit keyword arguments override the
    corresponding field in ``selection``.
    """
    # ── Unpack DataSelection (explicit kwargs win) ────────────────────────────
    if selection is not None:
        if chan_range is None and selection.chan_range is not None:
            chan_range = selection.chan_range
        if ant_list is None and selection.ant_list is not None:
            ant_list = selection.ant_list
        if timerange is None and selection.timerange is not None:
            timerange = selection.timerange
        if uvrange_m is None and selection.uvrange_m is not None:
            uvrange_m = selection.uvrange_m
        if uvrange_klambda is None and selection.uvrange_klambda is not None:
            uvrange_klambda = selection.uvrange_klambda
        if elevation_min_deg is None and selection.elevation_min_deg is not None:
            elevation_min_deg = selection.elevation_min_deg
        if elevation_max_deg is None and selection.elevation_max_deg is not None:
            elevation_max_deg = selection.elevation_max_deg
    # Normalise JD bounds
    _jd_bounds = None
    if timerange is not None:
        from astropy.time import Time as _T
        t0r, t1r = timerange
        if isinstance(t0r, str): t0r = _T(t0r, format='iso', scale='utc').jd
        if isinstance(t1r, str): t1r = _T(t1r, format='iso', scale='utc').jd
        _jd_bounds = (float(t0r), float(t1r))
    import time as _time
    t0 = _time.monotonic()
    id_to_name = index['id_to_name']

    if isinstance(source, str):
        wanted = [sid for sid, nm in id_to_name.items()
                  if nm.lower() == source.strip().lower()]
        if not wanted:
            raise ValueError(
                f'Source "{source}" not found. '
                f'Available: {sorted(id_to_name.values())}')
    else:
        wanted = [int(source)]

    ranges = []
    for sid in wanted:
        ranges.extend(index['source_ranges'].get(sid, []))
    ranges.sort()
    if not ranges:
        raise ValueError(f'No rows for source "{source}".')

    pcount      = index['pcount']
    nchan       = index['naxis4']
    nstokes_tot = index['naxis3']
    nif         = index['naxis5']
    naxis2      = index['naxis2']
    all_stokes  = index['stokes_labels']

    if chan_range is not None:
        c0, c1 = chan_range
        chan_idx = np.arange(max(0, c0), min(nchan, c1 + 1))
    else:
        chan_idx = np.arange(nchan)
    freqs_sel = index['chan_freqs_hz'][chan_idx]

    if stokes is not None:
        stokes_idx = []
        for s in stokes:
            if isinstance(s, int):
                stokes_idx.append(s)
            elif s in all_stokes:
                stokes_idx.append(all_stokes.index(s))
            else:
                raise ValueError(
                    f'Stokes "{s}" not in file. Available: {all_stokes}')
    else:
        stokes_idx = list(range(nstokes_tot))
    stokes_sel = [all_stokes[i] for i in stokes_idx]

    blocks = []
    total_sel = 0
    for rng_start, rng_end in ranges:
        a1 = index['ant1'][rng_start:rng_end]
        a2 = index['ant2'][rng_start:rng_end]
        mask = np.ones(rng_end - rng_start, dtype=bool)
        if ant_list is not None:
            mask &= np.isin(a1, list(ant_list)) & np.isin(a2, list(ant_list))
        elif ant_range is not None:
            lo, hi = ant_range
            mask &= (a1 >= lo) & (a1 <= hi) & (a2 >= lo) & (a2 <= hi)
        n = int(mask.sum())
        if n:
            blocks.append((rng_start, rng_end, mask))
            total_sel += n

    if not total_sel:
        raise ValueError(f'No rows match antenna filters for source "{source}".')

    step = 1
    if max_rows and total_sel > max_rows:
        step = max(1, total_sel // max_rows)
        print(f'[info] Downsampling: {total_sel:,} -> ~{total_sel // step:,} rows (step={step})')

    raw = np.memmap(index['path'], dtype='>f4', mode='r',
                    offset=index['data_offset'],
                    shape=(index['gcount'], index['group_size']))

    out_jd, out_a1, out_a2, out_uu, out_vv, out_data = [], [], [], [], [], []
    bytes_read = 0

    for rng_start, rng_end, mask in blocks:
        block_raw = np.array(raw[rng_start:rng_end, pcount:], dtype=np.float32)
        bytes_read += block_raw.nbytes
        block_raw = block_raw.reshape(rng_end - rng_start, nif, nchan, nstokes_tot, naxis2)
        block_raw = block_raw[:, 0, :, :, :]

        sel_idx = np.where(mask)[0][::step]
        chunk = block_raw[sel_idx][:, chan_idx][:, :, stokes_idx]
        out_data.append(chunk)

        out_jd.append(index['jd'][rng_start:rng_end][sel_idx])
        out_a1.append(index['ant1'][rng_start:rng_end][sel_idx])
        out_a2.append(index['ant2'][rng_start:rng_end][sel_idx])
        out_uu.append(index['uu_sec'][rng_start:rng_end][sel_idx])
        out_vv.append(index['vv_sec'][rng_start:rng_end][sel_idx])

    del raw

    data = np.concatenate(out_data, axis=0)
    jd   = np.concatenate(out_jd)
    a1   = np.concatenate(out_a1)
    a2   = np.concatenate(out_a2)
    uu   = np.concatenate(out_uu)
    vv   = np.concatenate(out_vv)

    # ── Post-load row filtering ───────────────────────────────────────────────
    _row_mask = np.ones(len(jd), dtype=bool)

    # 1. Time range
    if _jd_bounds is not None:
        _row_mask &= (jd >= _jd_bounds[0]) & (jd <= _jd_bounds[1])
        print(f'[load_vis] timerange filter: {_row_mask.sum():,} / {len(jd):,} rows kept')

    # 2. UV-distance (metres)
    if uvrange_m is not None:
        _uv_m = np.sqrt(np.asarray(uu, dtype=np.float64)**2 +
                        np.asarray(vv, dtype=np.float64)**2) * 2.998e8
        _row_mask &= (_uv_m >= uvrange_m[0]) & (_uv_m <= uvrange_m[1])
        print(f'[load_vis] uvrange_m filter: {_row_mask.sum():,} / {len(jd):,} rows kept')

    # 3. UV-distance (kilolambda, using band-centre frequency)
    if uvrange_klambda is not None:
        _ref_f  = float(0.5 * (freqs_sel[0] + freqs_sel[-1]))
        _uv_kl  = np.sqrt(np.asarray(uu, dtype=np.float64)**2 +
                          np.asarray(vv, dtype=np.float64)**2) * _ref_f / 1e3
        _row_mask &= (_uv_kl >= uvrange_klambda[0]) & (_uv_kl <= uvrange_klambda[1])
        print(f'[load_vis] uvrange_klambda filter: {_row_mask.sum():,} / {len(jd):,} rows kept')

    # 4. Elevation filter (uses array ECEF + source coords from FITS header)
    _n_rows_before_elev = int(_row_mask.sum())   # rows surviving filters 1-3
    _n_rows_skipped_elevation = 0
    if elevation_min_deg is not None or elevation_max_deg is not None:
        try:
            from astropy.coordinates import SkyCoord, EarthLocation, AltAz
            from astropy.time import Time as _T
            import astropy.units as _u
            _geo = _get_array_enu(index['path'])
            _loc = EarthLocation.from_geocentric(*_geo['ecef_centre'], unit=_u.m)
            # Look up source RA/Dec from FITS SU table
            with fits.open(index['path'], memmap=True, lazy_load_hdus=True) as _hdul:
                _su = _get_hdu(_hdul, 'AIPS SU')
                _id_to_name = index['id_to_name']
                if isinstance(source, str):
                    _wanted_ids = [k for k, v in _id_to_name.items()
                                   if v.strip().lower() == source.strip().lower()]
                else:
                    _wanted_ids = [int(source)]
                _su_rows = {}
                _id_col = next((c for c in _su.data.dtype.names
                                if 'ID' in c.upper() and 'NO' in c.upper()), None)
                for _row in _su.data:
                    _sid = int(_row[_id_col]) if _id_col else None
                    _su_rows[_sid] = _row
                _ra_deg  = float(_su_rows[_wanted_ids[0]]['RAEPO'])
                _dec_deg = float(_su_rows[_wanted_ids[0]]['DECEPO'])
            _coord   = SkyCoord(ra=_ra_deg * _u.deg, dec=_dec_deg * _u.deg, frame='icrs')
            _times   = _T(jd, format='jd', scale='utc')
            _altaz   = _coord.transform_to(AltAz(obstime=_times, location=_loc))
            _el      = _altaz.alt.deg
            if elevation_min_deg is not None:
                _row_mask &= _el >= elevation_min_deg
            if elevation_max_deg is not None:
                _row_mask &= _el <= elevation_max_deg
            _n_rows_skipped_elevation = _n_rows_before_elev - int(_row_mask.sum())
            print(f'[load_vis] elevation filter (el>={elevation_min_deg}, el<={elevation_max_deg}): '
                  f'{_row_mask.sum():,} / {len(jd):,} rows kept '
                  f'({_n_rows_skipped_elevation:,} rows skipped by elevation cut)')
        except Exception as _exc:
            print(f'[load_vis] WARNING: elevation filter failed ({_exc}); no elevation cut applied.')

    if not _row_mask.all():
        if _row_mask.sum() == 0:
            raise ValueError('DataSelection filters removed all rows — no data remains.')
        data = data[_row_mask]
        jd   = jd[_row_mask]
        a1   = a1[_row_mask]
        a2   = a2[_row_mask]
        uu   = uu[_row_mask]
        vv   = vv[_row_mask]

    re_ = data[..., 0]
    im_ = data[..., 1]
    wt_ = data[..., 2]

    amp   = np.sqrt(re_**2 + im_**2)
    phase = np.degrees(np.arctan2(im_, re_)).astype(np.float32)
    flagged = wt_ <= 0
    if flag_all_corrs_if_any_rawvis_flagged and flagged.ndim == 3 and flagged.shape[2] > 1:
        # At (row, channel), if any correlation is natively flagged (weight <= 0)
        # in the raw FITS data, force all correlations to be flagged.  This keeps
        # RR and LL symmetric: the solve and diagnostics see the same rows for
        # both feed chains.  The raw FITS file is never modified.
        #
        # Sanity-check (TODO #3): verify the result is corr-uniform, i.e. every
        # (row, chan) cell is either flagged in ALL corrs or in NONE.
        shared_flagged = np.any(flagged, axis=2, keepdims=True)
        _n_asymmetric = int(np.count_nonzero(
            shared_flagged.squeeze(axis=2) != flagged.all(axis=2)
        ))
        if _n_asymmetric:
            print(
                f'[load_vis] FLAG_ALL_CORRS symmetrisation corrected '
                f'{_n_asymmetric:,} asymmetric (row,chan) cells '
                f'(≈{100.0 * _n_asymmetric / max(shared_flagged.size, 1):.2f}%)'
            )
        flagged = np.broadcast_to(shared_flagged, flagged.shape).copy()
        wt_[flagged] = 0.0
        # Post-condition: every (row,chan) must be uniformly flagged across corrs.
        assert np.array_equal(flagged.all(axis=2), flagged.any(axis=2)), (
            'FLAG_ALL_CORRS symmetrisation failed: corr-asymmetric flags remain'
        )
    # Flag exact hardware zeros: correlator dropouts produce re_=im_=0 with a
    # valid (positive) weight, which are NOT caught by the wt_<=0 test above.
    # These appear as identically-zero visibilities in amplitude/Stokes-V plots.
    # We flag them here so that all downstream consumers (bandpass solve,
    # clustering detection, Stokes-V plots) see NaN, not 0.
    _exact_zero = (re_ == 0.0) & (im_ == 0.0) & ~flagged
    if _exact_zero.any():
        _n_zero = int(_exact_zero.sum())
        print(f'[load_vis] Flagging {_n_zero:,} exact-zero vis cells '
              f'({100.0 * _n_zero / max(_exact_zero.size, 1):.2f}%) '
              f'[hardware dropout / dead correlator output]')
        flagged = flagged | _exact_zero
    amp[flagged]   = np.nan
    phase[flagged] = np.nan
    # Build vis_complex with NaN at all flagged (row, chan, pol) positions so
    # that apply_bandpass_solution and compute_stokes_vis never see 0+0j for
    # dead cells (0/gain = 0, which would plot as a real point at exactly zero).
    vis_complex = (re_.astype(np.float64) + 1j * im_.astype(np.float64)).astype(np.complex64)
    vis_complex[flagged] = np.nan + 0j

    ref_freq = float(0.5 * (freqs_sel[0] + freqs_sel[-1]))
    uv_sec = np.sqrt(uu**2 + vv**2)              # per-row, in seconds
    uvdist_klambda = (uv_sec * ref_freq) / 1e3    # single ref_freq (back-compat)

    # per-channel UV distance: shape (nrows, nchan)
    # uv_sec[:,None] * freqs_sel[None,:] / 1e3
    uvdist_per_chan = (uv_sec[:, np.newaxis]
                       * freqs_sel[np.newaxis, :]) / 1e3

    elapsed = _time.monotonic() - t0
    print(f'[load_vis] {data.shape[0]:,} rows, {bytes_read / 1e6:.0f} MB read, {elapsed:.1f}s')

    return {
        'jd': jd,
        'ant1': a1.astype(int),
        'ant2': a2.astype(int),
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
        'nrows': int(data.shape[0]),
        'n_rows_skipped_elevation': _n_rows_skipped_elevation,
    }


# ---------------------------------------------------------------------------
# Non-destructive bandpass calibration utilities
# ---------------------------------------------------------------------------

_PB2017_3C48_COEFFS  = np.array([1.3253, -0.7553, -0.1914,  0.0498], dtype=np.float64)
# Perley & Butler 2017 (ApJS 230, 7), Table 2 — 3C286 / J1331+3030, valid 0.05–50 GHz
_PB2017_3C286_COEFFS = np.array([1.2481, -0.4507, -0.1798,  0.0357], dtype=np.float64)


def _norm_antenna_name(name):
    """Normalize antenna labels for tolerant matching.

    Example: "C11:11" and "c11" both normalize to "C11".
    """
    base = str(name).strip().upper().split(':', 1)[0]
    return ''.join(ch for ch in base if ch.isalnum())


def create_flag_table(
    bad_antennas: Optional[List[Union[str, int]]] = None,
    bad_baselines: Optional[List[Union[Tuple[Union[str, int], Union[str, int]], List[Union[str, int]], Dict[str, Union[str, int]]]]] = None,
    notes: str = '',
) -> dict:
    """Create a JSON-serializable flag table for calibration-time exclusions."""
    return {
        'kind': 'ugmrt_flag_table',
        'version': 1,
        'bad_antennas': list(bad_antennas or []),
        'bad_baselines': list(bad_baselines or []),
        'notes': str(notes),
    }


def save_flag_table(flag_table: dict, path: Union[str, Path]) -> Path:
    """Save antenna/baseline exclusions to a standalone JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(flag_table, f, indent=2, sort_keys=True)
    return path


def load_flag_table(path: Union[str, Path]) -> dict:
    """Load a standalone JSON flag table used for calibration-time exclusions."""
    path = Path(path)
    with path.open('r', encoding='utf-8') as f:
        table = json.load(f)

    if not isinstance(table, dict):
        raise ValueError('Flag table JSON must contain an object at top level.')
    if table.get('kind', 'ugmrt_flag_table') != 'ugmrt_flag_table':
        raise ValueError('Unsupported flag table kind. Expected "ugmrt_flag_table".')
    if int(table.get('version', 1)) not in (1, 2):
        raise ValueError('Unsupported flag table version. Expected version 1 or 2.')

    table.setdefault('bad_antennas', [])
    table.setdefault('bad_baselines', [])
    table.setdefault('notes', '')
    return table


def _coerce_flag_tables(
    flag_tables: Optional[Union[dict, List[dict], Tuple[dict, ...]]] = None,
    flag_table_paths: Optional[Union[str, Path, List[Union[str, Path]], Tuple[Union[str, Path], ...]]] = None,
):
    """Return (tables, paths) while accepting singular or plural inputs."""
    tables = []
    paths = []

    if flag_table_paths is not None:
        if isinstance(flag_table_paths, (str, Path)):
            flag_table_paths = [flag_table_paths]
        for p in flag_table_paths:
            table = load_flag_table(p)
            tables.append(table)
            paths.append(str(Path(p)))

    if flag_tables is not None:
        if isinstance(flag_tables, dict):
            flag_tables = [flag_tables]
        for table in flag_tables:
            if not isinstance(table, dict):
                raise ValueError('Each flag table must be a dict.')
            tables.append(table)

    return tables, paths


def _resolve_antenna_selector_ids(
    selectors,
    antenna_name_map: Dict[int, str],
    strict: bool = False,
    context: str = 'flag-table',
):
    """Resolve antenna selectors (IDs or labels) to antenna IDs."""
    ant_ids = sorted(int(a) for a in antenna_name_map.keys())
    ant_id_set = set(ant_ids)

    name_to_id = {
        str(name).strip().upper(): int(ant)
        for ant, name in antenna_name_map.items()
    }
    norm_to_ids = {}
    for ant, name in antenna_name_map.items():
        norm = _norm_antenna_name(name)
        norm_to_ids.setdefault(norm, []).append(int(ant))

    resolved = []
    unresolved = []

    for item in selectors or []:
        if isinstance(item, (int, np.integer)):
            ant_id = int(item)
            if ant_id in ant_id_set:
                resolved.append(ant_id)
            else:
                unresolved.append(item)
            continue

        key = str(item).strip().upper()
        if key in name_to_id:
            resolved.append(name_to_id[key])
            continue

        norm_key = _norm_antenna_name(key)
        norm_matches = norm_to_ids.get(norm_key, [])
        if len(norm_matches) == 1:
            resolved.append(norm_matches[0])
            continue
        if len(norm_matches) > 1:
            msg = f'[{context}] antenna selector "{item}" is ambiguous among IDs {norm_matches}.'
            if strict:
                raise ValueError(msg)
            print(msg + ' Ignored.')
            continue

        try:
            ant_id = int(key)
            if ant_id in ant_id_set:
                resolved.append(ant_id)
            else:
                unresolved.append(item)
        except ValueError:
            unresolved.append(item)

    if unresolved:
        msg = f'[{context}] Unrecognized antenna selectors: {unresolved}'
        if strict:
            raise ValueError(msg)
        print(msg + ' (ignored)')

    return sorted(set(resolved))


def _resolve_baseline_selector_pairs(
    selectors,
    antenna_name_map: Dict[int, str],
    strict: bool = False,
    context: str = 'flag-table',
):
    """Resolve baseline selectors to canonical (min_ant, max_ant) ID pairs."""
    pairs = set()
    unresolved = []

    for item in selectors or []:
        a1 = a2 = None

        if isinstance(item, dict):
            a1 = item.get('ant1')
            a2 = item.get('ant2')
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            a1, a2 = item[0], item[1]
        elif isinstance(item, str) and '-' in item:
            left, right = item.split('-', 1)
            a1, a2 = left.strip(), right.strip()

        if a1 is None or a2 is None:
            unresolved.append(item)
            continue

        ant_ids = _resolve_antenna_selector_ids(
            [a1, a2],
            antenna_name_map=antenna_name_map,
            strict=strict,
            context=context,
        )
        if len(ant_ids) != 2:
            unresolved.append(item)
            continue

        if ant_ids[0] == ant_ids[1]:
            msg = f'[{context}] baseline selector "{item}" maps to identical antennas; ignored.'
            if strict:
                raise ValueError(msg)
            print(msg)
            continue

        pairs.add(tuple(sorted((int(ant_ids[0]), int(ant_ids[1])))))

    if unresolved:
        msg = f'[{context}] Unrecognized baseline selectors: {unresolved}'
        if strict:
            raise ValueError(msg)
        print(msg + ' (ignored)')

    return sorted(pairs)


def apply_flag_table_to_vis(
    vis: dict,
    antenna_name_map: Dict[int, str],
    flag_table: dict,
    strict: bool = False,
    context: str = 'flag-table',
):
    """Filter rows excluded by a flag table and return (filtered_vis, stats)."""
    excluded_ant_ids = _resolve_antenna_selector_ids(
        flag_table.get('bad_antennas', []),
        antenna_name_map=antenna_name_map,
        strict=strict,
        context=context,
    )
    excluded_base_pairs = _resolve_baseline_selector_pairs(
        flag_table.get('bad_baselines', []),
        antenna_name_map=antenna_name_map,
        strict=strict,
        context=context,
    )

    excluded_ant_ids_set = set(excluded_ant_ids)
    excluded_base_pairs_set = set(excluded_base_pairs)

    nrows = int(vis.get('nrows', len(vis.get('ant1', []))))
    row_mask = np.ones(nrows, dtype=bool)

    if excluded_ant_ids_set:
        row_mask &= (~np.isin(vis['ant1'], excluded_ant_ids)) & (~np.isin(vis['ant2'], excluded_ant_ids))

    if excluded_base_pairs_set:
        b1 = np.minimum(np.asarray(vis['ant1'], dtype=np.int32), np.asarray(vis['ant2'], dtype=np.int32))
        b2 = np.maximum(np.asarray(vis['ant1'], dtype=np.int32), np.asarray(vis['ant2'], dtype=np.int32))
        baseline_bad = np.array([(int(a), int(b)) in excluded_base_pairs_set for a, b in zip(b1, b2)], dtype=bool)
        row_mask &= ~baseline_bad

    kept = int(np.count_nonzero(row_mask))
    dropped = int(nrows - kept)
    if kept == 0:
        raise ValueError(f'[{context}] All rows were removed by the flag table filters.')

    vis_use = {}
    for key, value in vis.items():
        if isinstance(value, np.ndarray) and value.shape[:1] == (nrows,):
            vis_use[key] = value[row_mask]
        else:
            vis_use[key] = value
    vis_use['nrows'] = kept

    stats = {
        'dropped_rows': dropped,
        'kept_rows': kept,
        'excluded_antenna_ids': excluded_ant_ids,
        'excluded_baseline_pairs': excluded_base_pairs,
        'flag_table_notes': str(flag_table.get('notes', '')),
        'flag_table_count': 1,
        'flag_table_paths': [],
    }
    return vis_use, stats


def apply_flag_tables_to_vis(
    vis: dict,
    antenna_name_map: Dict[int, str],
    flag_tables: Optional[Union[dict, List[dict], Tuple[dict, ...]]] = None,
    flag_table_paths: Optional[Union[str, Path, List[Union[str, Path]], Tuple[Union[str, Path], ...]]] = None,
    strict: bool = False,
    context: str = 'flag-table',
):
    """Apply one or more flag tables to visibility rows in memory.

    This function merges antenna/baseline exclusions on-the-fly and never
    modifies visibility data on disk.
    """
    tables, paths = _coerce_flag_tables(flag_tables=flag_tables, flag_table_paths=flag_table_paths)
    if not tables:
        stats = {
            'dropped_rows': 0,
            'kept_rows': int(vis.get('nrows', len(vis.get('ant1', [])))),
            'excluded_antenna_ids': [],
            'excluded_baseline_pairs': [],
            'flag_table_notes': '',
            'flag_table_count': 0,
            'flag_table_paths': [],
        }
        return vis, stats

    excluded_ant_ids_set = set()
    excluded_base_pairs_set = set()
    notes = []

    for idx, table in enumerate(tables, start=1):
        table_ctx = f'{context}#{idx}'
        ant_ids = _resolve_antenna_selector_ids(
            table.get('bad_antennas', []),
            antenna_name_map=antenna_name_map,
            strict=strict,
            context=table_ctx,
        )
        base_pairs = _resolve_baseline_selector_pairs(
            table.get('bad_baselines', []),
            antenna_name_map=antenna_name_map,
            strict=strict,
            context=table_ctx,
        )
        excluded_ant_ids_set.update(int(a) for a in ant_ids)
        excluded_base_pairs_set.update(tuple(sorted((int(a), int(b)))) for a, b in base_pairs)
        note = str(table.get('notes', '')).strip()
        if note:
            notes.append(note)

    nrows = int(vis.get('nrows', len(vis.get('ant1', []))))
    row_mask = np.ones(nrows, dtype=bool)

    excluded_ant_ids = sorted(excluded_ant_ids_set)
    if excluded_ant_ids:
        row_mask &= (~np.isin(vis['ant1'], excluded_ant_ids)) & (~np.isin(vis['ant2'], excluded_ant_ids))

    excluded_base_pairs = sorted(excluded_base_pairs_set)
    if excluded_base_pairs:
        b1 = np.minimum(np.asarray(vis['ant1'], dtype=np.int32), np.asarray(vis['ant2'], dtype=np.int32))
        b2 = np.maximum(np.asarray(vis['ant1'], dtype=np.int32), np.asarray(vis['ant2'], dtype=np.int32))
        baseline_bad = np.array([(int(a), int(b)) in excluded_base_pairs_set for a, b in zip(b1, b2)], dtype=bool)
        row_mask &= ~baseline_bad

    kept = int(np.count_nonzero(row_mask))
    dropped = int(nrows - kept)
    if kept == 0:
        raise ValueError(f'[{context}] All rows were removed by merged flag-table filters.')

    vis_use = {}
    for key, value in vis.items():
        if isinstance(value, np.ndarray) and value.shape[:1] == (nrows,):
            vis_use[key] = value[row_mask]
        else:
            vis_use[key] = value
    vis_use['nrows'] = kept

    # ── Step 2: apply time-range/burst/scan flags by cell-level zeroing ──────
    # Wholesale antenna and baseline flags were handled above by row-dropping.
    # All remaining flag sections (bad_antenna_timeranges,
    # bad_baseline_timeranges, bad_scan_timeranges, bad_burst_timeranges)
    # may carry chanrange-selective time windows.  Those are applied here by
    # zeroing vis_complex amplitude and vis weight for the specific (row, chan)
    # cells.  Setting weight=0 ensures the stefCal solver ignores those
    # samples entirely; the rows are kept in the array so that adjacent cells
    # on other channels remain contributing.
    _TR_KEYS = (
        'bad_antenna_timeranges', 'bad_baseline_timeranges',
        'bad_scan_timeranges',    'bad_burst_timeranges',
    )
    _n_cells_zeroed = 0
    if (
        'vis_complex' in vis_use
        and 'jd' in vis_use
        and any(table.get(k) for table in tables for k in _TR_KEYS)
    ):
        # Ensure writable copies (vis arrays may be memory-mapped views).
        _vc  = np.array(vis_use['vis_complex'])
        _wt  = np.array(vis_use['weight'])     if 'weight'    in vis_use else None
        _fl  = np.array(vis_use['flagged'])    if 'flagged'   in vis_use else None
        # amp and phase_deg are pre-computed arrays stored separately; they
        # must be NaN-filled in sync with vis_complex so the plot code — which
        # reads vis['amp'] directly — does not show the original amplitude at
        # time-range-flagged cells.
        _amp = np.array(vis_use['amp'],       dtype=np.float32) \
               if 'amp'       in vis_use else None
        _ph  = np.array(vis_use['phase_deg'], dtype=np.float32) \
               if 'phase_deg' in vis_use else None

        for table in tables:
            if not any(table.get(k) for k in _TR_KEYS):
                continue
            # expand_flag_table_to_mask returns (nrows_kept, nchans) and
            # handles both v1 and v2 schemas including chanrange fields.
            # Wholesale-flagged rows were already dropped above, so any
            # residual antenna/baseline matches in the mask are harmless
            # (those rows no longer exist in vis_use).
            _cell_mask = expand_flag_table_to_mask(
                vis_use, table, antenna_name_map,
                strict=strict, context=context,
            )  # shape (nrows_kept, nchans)
            if not _cell_mask.any():
                continue
            _n_cells_zeroed += int(_cell_mask.sum())
            # Boolean 2-D mask on a 3-D (nrows, nchans, npol) array:
            # arr[mask2d] selects the (npol,) vectors at True positions,
            # so the assignment broadcasts across all polarisations.
            _vc[_cell_mask]  = 0.0
            if _wt is not None:
                _wt[_cell_mask] = 0.0
            if _fl is not None:
                _fl[_cell_mask] = True
            # NaN pre-computed amp / phase_deg so plot code sees missing cells.
            if _amp is not None:
                _amp[_cell_mask] = np.nan
            if _ph is not None:
                _ph[_cell_mask] = np.nan

        vis_use['vis_complex'] = _vc
        if _wt is not None:
            vis_use['weight']    = _wt
        if _fl is not None:
            vis_use['flagged']   = _fl
        if _amp is not None:
            vis_use['amp']       = _amp
        if _ph is not None:
            vis_use['phase_deg'] = _ph
        if _n_cells_zeroed:
            print(
                f'[{context}] time-range/chanrange flags applied: '
                f'{_n_cells_zeroed:,} vis cell(s) zeroed across '
                f'{kept:,} kept row(s)'
            )

    stats = {
        'dropped_rows': dropped,
        'kept_rows': kept,
        'excluded_antenna_ids': excluded_ant_ids,
        'excluded_baseline_pairs': excluded_base_pairs,
        'flag_table_notes': ' | '.join(notes),
        'flag_table_count': len(tables),
        'flag_table_paths': paths,
        'zeroed_cells_by_timerange_flags': _n_cells_zeroed,
    }
    return vis_use, stats


def count_vis_rows_flagged(
    vis: dict,
    flag_table: dict,
    antenna_name_map: Dict[int, str],
    strict: bool = False,
    context: str = 'flag-count',
) -> dict:
    """Count how many visibility rows in *vis* would be flagged by *flag_table*.

    Handles all five flag-table key types:

    * ``bad_antennas``            – whole antenna flagged at all times
    * ``bad_baselines``           – whole baseline flagged at all times
    * ``bad_antenna_timeranges``  – per-antenna selective time-range masks
    * ``bad_baseline_timeranges`` – per-baseline selective time-range masks
    * ``bad_burst_timeranges``    – global time-range masks (all antennas)

    Multiple flag types can overlap; the union is counted (no double-counting).

    Parameters
    ----------
    vis : dict
        Visibility dict with ``ant1``, ``ant2``, and ``jd`` arrays.
    flag_table : dict
        A single ugmrt flag-table dict.
    antenna_name_map : Dict[int, str]
        Mapping antenna_id → name (from ``row_index['antennas']``).
    strict : bool, optional
        Raise on unresolved antenna/baseline selectors (default: warn).
    context : str, optional
        Label prefix used in diagnostic messages.

    Returns
    -------
    dict with keys:
      ``total_rows``   – int   total rows in vis
      ``flagged_rows`` – int   rows masked by the flag table (union of all types)
      ``kept_rows``    – int   total_rows − flagged_rows
      ``pct_flagged``  – float percentage flagged (0–100, two decimal places)
    """
    ant1  = np.asarray(vis['ant1'], dtype=np.int32)
    ant2  = np.asarray(vis['ant2'], dtype=np.int32)
    jd    = np.asarray(vis['jd'],   dtype=np.float64)
    nrows = len(ant1)

    # Reverse map: antenna name → id
    name_to_id: Dict[str, int] = {v: k for k, v in antenna_name_map.items()}

    flag_mask = np.zeros(nrows, dtype=bool)   # True = flagged

    # ── 1.  Whole-antenna flags ───────────────────────────────────────────────
    ant_ids = _resolve_antenna_selector_ids(
        flag_table.get('bad_antennas', []),
        antenna_name_map=antenna_name_map,
        strict=strict,
        context=context,
    )
    if ant_ids:
        flag_mask |= np.isin(ant1, ant_ids) | np.isin(ant2, ant_ids)

    # ── 2.  Whole-baseline flags ──────────────────────────────────────────────
    base_pairs = _resolve_baseline_selector_pairs(
        flag_table.get('bad_baselines', []),
        antenna_name_map=antenna_name_map,
        strict=strict,
        context=context,
    )
    if base_pairs:
        bp_set = {tuple(sorted((int(a), int(b)))) for a, b in base_pairs}
        b1 = np.minimum(ant1, ant2)
        b2 = np.maximum(ant1, ant2)
        flag_mask |= np.array(
            [(int(a), int(b)) in bp_set for a, b in zip(b1, b2)], dtype=bool
        )

    def _iso_to_jd_local(s: str) -> float:
        # Accept both 'T'-separated (isot) and space-separated (iso) strings.
        return float(Time(str(s).replace('T', ' '), format='iso', scale='utc').jd)

    def _time_mask(jd_arr: np.ndarray, ranges: list) -> np.ndarray:
        """Boolean mask: True where jd falls in any (iso_t0, iso_t1) range."""
        m = np.zeros(len(jd_arr), dtype=bool)
        for t0_iso, t1_iso in ranges:
            t0 = _iso_to_jd_local(t0_iso)
            t1 = _iso_to_jd_local(t1_iso)
            m |= (jd_arr >= t0) & (jd_arr <= t1)
        return m

    def _flatten_entries(entries: list) -> list:
        """Normalize v1 or v2 timerange entries to flat [(t0, t1), ...] list."""
        flat = []
        for entry in entries:
            if isinstance(entry, dict):
                for t0, t1 in entry['intervals']:
                    flat.append((t0, t1))
            else:
                flat.append((entry[0], entry[1]))
        return flat

    # ── 3.  Per-antenna time-range flags ─────────────────────────────────────
    for ant_name, entries in flag_table.get('bad_antenna_timeranges', {}).items():
        if not entries:
            continue
        aid = name_to_id.get(str(ant_name))
        if aid is None:
            continue
        rows_for_ant = (ant1 == aid) | (ant2 == aid)
        flag_mask[rows_for_ant] |= _time_mask(jd[rows_for_ant], _flatten_entries(entries))

    # ── 4.  Per-baseline time-range flags ────────────────────────────────────
    for bl_key, entries in flag_table.get('bad_baseline_timeranges', {}).items():
        if not entries:
            continue
        parts = bl_key.split('::', 1)
        if len(parts) != 2:
            continue
        a_id = name_to_id.get(parts[0])
        b_id = name_to_id.get(parts[1])
        if a_id is None or b_id is None:
            continue
        rows_for_bl = (
            ((ant1 == a_id) & (ant2 == b_id)) |
            ((ant1 == b_id) & (ant2 == a_id))
        )
        flag_mask[rows_for_bl] |= _time_mask(jd[rows_for_bl], _flatten_entries(entries))

    # ── 5.  Global burst time-range flags ────────────────────────────────────
    burst_ranges = flag_table.get('bad_burst_timeranges', [])
    if burst_ranges:
        flag_mask |= _time_mask(jd, burst_ranges)

    flagged = int(np.count_nonzero(flag_mask))
    return {
        'total_rows':   nrows,
        'flagged_rows': flagged,
        'kept_rows':    nrows - flagged,
        'pct_flagged':  round(100.0 * flagged / nrows, 2) if nrows > 0 else 0.0,
    }


def expand_flag_table_to_mask(
    vis: dict,
    flag_table: dict,
    antenna_name_map: Dict[int, str],
    strict: bool = False,
    context: str = 'expand-flags',
) -> np.ndarray:
    """Decode a flag table into a 2-D boolean visibility mask.

    Returns ``flag_mask[nRows, nChans]`` where ``True`` means *flagged*.
    This is the canonical v2-schema decode: time-range entries are applied
    only to the channel range specified by their ``chanrange`` field (or
    all channels for ``chanrange=None``).

    Parameters
    ----------
    vis : dict
        Visibility dict with ``ant1``, ``ant2``, ``jd``, and ``chan_indices``
        arrays.  ``chan_indices`` must be the absolute FITS channel indices
        loaded for this vis slice (e.g. ``vis['chan_indices']``).
    flag_table : dict
        A single ugmrt_flag_table, version 1 or 2.
    antenna_name_map : Dict[int, str]
        Mapping antenna_id → name.
    strict : bool
        Raise on unresolved antenna/baseline selectors (default: warn only).
    context : str
        Label prefix for diagnostic messages.

    Returns
    -------
    np.ndarray, shape (nRows, nChans), dtype bool
        ``True`` wherever the visibility cell is flagged.
    """
    ant1 = np.asarray(vis['ant1'], dtype=np.int32)
    ant2 = np.asarray(vis['ant2'], dtype=np.int32)
    jd   = np.asarray(vis['jd'],   dtype=np.float64)
    nrows = len(ant1)

    # Build absolute channel index → local column index mapping
    raw_ci = vis.get('chan_indices')
    if raw_ci is not None:
        chan_indices_abs: List[int] = [int(c) for c in raw_ci]
    else:
        # Fall back: infer from vis_complex shape
        vis_c = vis.get('vis_complex')
        if vis_c is not None:
            nchans = vis_c.shape[1]
        else:
            nchans = 1
        chan_indices_abs = list(range(nchans))

    nchans = len(chan_indices_abs)
    abs_to_local: Dict[int, int] = {c: i for i, c in enumerate(chan_indices_abs)}

    flag_mask = np.zeros((nrows, nchans), dtype=bool)
    # TODO #3 sanity: output is (nRows, nChans) — no pol axis.  pols are handled
    # by broadcasting this mask over vis_complex (nRows, nChans, nPols) at
    # apply time, not stored here.  See FLAG_ALL_CORRS in FLAGGING_DESIGN_NOTES.

    name_to_id: Dict[str, int] = {v: k for k, v in antenna_name_map.items()}

    # ── 1.  Whole-antenna flags (all channels) ────────────────────────────────
    ant_ids = _resolve_antenna_selector_ids(
        flag_table.get('bad_antennas', []),
        antenna_name_map=antenna_name_map,
        strict=strict,
        context=context,
    )
    if ant_ids:
        ant_rows = np.isin(ant1, ant_ids) | np.isin(ant2, ant_ids)
        flag_mask[ant_rows, :] = True

    # ── 2.  Whole-baseline flags (all channels) ───────────────────────────────
    base_pairs = _resolve_baseline_selector_pairs(
        flag_table.get('bad_baselines', []),
        antenna_name_map=antenna_name_map,
        strict=strict,
        context=context,
    )
    if base_pairs:
        bp_set = {tuple(sorted((int(a), int(b)))) for a, b in base_pairs}
        b1 = np.minimum(ant1, ant2)
        b2 = np.maximum(ant1, ant2)
        bl_rows = np.array([(int(a), int(b)) in bp_set for a, b in zip(b1, b2)], dtype=bool)
        flag_mask[bl_rows, :] = True

    def _iso_to_jd_loc(s: str) -> float:
        return float(Time(str(s).replace('T', ' '), format='iso', scale='utc').jd)

    def _time_row_mask(jd_arr: np.ndarray, intervals: list) -> np.ndarray:
        m = np.zeros(len(jd_arr), dtype=bool)
        for t0_iso, t1_iso in intervals:
            t0 = _iso_to_jd_loc(t0_iso)
            t1 = _iso_to_jd_loc(t1_iso)
            m |= (jd_arr >= t0) & (jd_arr <= t1)
        return m

    def _local_chan_slice(chanrange) -> slice:
        """Return all-channel slice if chanrange is None.
        Otherwise map absolute chanrange [c0,c1] to local column indices.
        """
        if chanrange is None:
            return slice(None)
        c0_abs, c1_abs = int(chanrange[0]), int(chanrange[1])
        locs = sorted(abs_to_local[c] for c in range(c0_abs, c1_abs + 1)
                      if c in abs_to_local)
        if not locs:
            return None   # no overlap with loaded channels
        return slice(locs[0], locs[-1] + 1)

    def _iter_entries(entries):
        """Yield (intervals, chanrange) for v1 or v2 entry lists."""
        for entry in entries:
            if isinstance(entry, dict):
                yield entry['intervals'], entry.get('chanrange')
            else:
                yield [list(entry)], None

    # ── 3.  Per-antenna time-range flags ─────────────────────────────────────
    for ant_name, entries in flag_table.get('bad_antenna_timeranges', {}).items():
        if not entries:
            continue
        aid = name_to_id.get(str(ant_name))
        if aid is None:
            continue
        ant_row_idx = np.where((ant1 == aid) | (ant2 == aid))[0]
        if len(ant_row_idx) == 0:
            continue
        for intervals, chanrange in _iter_entries(entries):
            col_slice = _local_chan_slice(chanrange)
            if col_slice is None:
                continue
            time_m = _time_row_mask(jd[ant_row_idx], intervals)
            affected = ant_row_idx[time_m]
            if len(affected):
                flag_mask[np.ix_(affected, np.arange(nchans)[col_slice])] = True

    # ── 4.  Per-baseline time-range flags ────────────────────────────────────
    for bl_key, entries in flag_table.get('bad_baseline_timeranges', {}).items():
        if not entries:
            continue
        parts = bl_key.split('::', 1)
        if len(parts) != 2:
            continue
        a_id = name_to_id.get(parts[0])
        b_id = name_to_id.get(parts[1])
        if a_id is None or b_id is None:
            continue
        bl_row_idx = np.where(
            ((ant1 == a_id) & (ant2 == b_id)) |
            ((ant1 == b_id) & (ant2 == a_id))
        )[0]
        if len(bl_row_idx) == 0:
            continue
        for intervals, chanrange in _iter_entries(entries):
            col_slice = _local_chan_slice(chanrange)
            if col_slice is None:
                continue
            time_m = _time_row_mask(jd[bl_row_idx], intervals)
            affected = bl_row_idx[time_m]
            if len(affected):
                flag_mask[np.ix_(affected, np.arange(nchans)[col_slice])] = True

    # ── 5.  Global burst time-range flags (all channels) ─────────────────────
    burst_ranges = flag_table.get('bad_burst_timeranges', [])
    if burst_ranges:
        burst_flat = [(e[0], e[1]) if not isinstance(e, dict) else
                      (e['intervals'][0][0], e['intervals'][0][1])
                      for e in burst_ranges]
        burst_m = _time_row_mask(jd, burst_flat)
        flag_mask[burst_m, :] = True

    return flag_mask


def materialise_flag_stats(
    flag_mask: np.ndarray,
    vis: dict,
    antenna_name_map: Optional[Dict[int, str]] = None,
) -> dict:
    """Reshape a ``(nRows, nChans)`` bool flag mask and compute fine-grained statistics.

    The mask is mapped into a ``(nInteg, nBL, nChans)`` cube using index
    assignment — no assumption is made about the row-sort order of the file.
    The grid must be *regular*: every ``(integration, baseline)`` pair must
    appear exactly once, i.e. ``nRows == nInteg * nBL``.  DUD antennas satisfy
    this because they contribute zero rows rather than sparse/missing entries.

    Parameters
    ----------
    flag_mask : np.ndarray, shape ``(nRows, nChans)``, dtype bool
        The output of :func:`expand_flag_table_to_mask`.
    vis : dict
        Must contain ``'ant1'``, ``'ant2'``, and ``'jd'`` arrays.
    antenna_name_map : dict, optional
        ``{antenna_id: name}`` mapping.  When supplied, per-antenna statistics
        use names as keys; otherwise antenna IDs are used.

    Returns
    -------
    dict with keys:

    ``nInteg``, ``nBL``, ``nChans``
        Dimension sizes inferred from the data.
    ``is_regular``
        ``True`` when ``nRows == nInteg * nBL`` (reshape is valid).
    ``flags_3d``
        ``(nInteg, nBL, nChans)`` bool array (only present when regular).
    ``pct_total``
        Overall percentage of flagged cells (scalar float).
    ``pct_per_baseline``
        ``(nBL,)`` float — % flagged averaged over integrations and channels.
    ``pct_per_channel``
        ``(nChans,)`` float — % flagged averaged over integrations and baselines.
    ``pct_per_integration``
        ``(nInteg,)`` float — % flagged averaged over baselines and channels.
    ``pct_per_antenna``
        ``{antenna_name: float}`` — % flagged for all baselines involving
        each antenna, averaged over integrations and channels.
    ``bl_labels``
        ``['ant1name:ant2name', …]`` length-``nBL`` list.
    ``jd_unique``
        Sorted array of unique Julian dates (one per integration).
    """
    ant1 = np.asarray(vis['ant1'], dtype=np.int32)
    ant2 = np.asarray(vis['ant2'], dtype=np.int32)
    jd   = np.asarray(vis['jd'],   dtype=np.float64)
    nrows, nchans = flag_mask.shape

    # ── unique integrations and baselines ────────────────────────────────────
    jd_unique, jd_indices = np.unique(jd, return_inverse=True)
    bl_key = np.stack([ant1, ant2], axis=1)              # (nrows, 2)
    bl_unique, bl_indices = np.unique(bl_key, axis=0, return_inverse=True)
    n_integ = int(len(jd_unique))
    n_bl    = int(len(bl_unique))

    # ── regularity check ─────────────────────────────────────────────────────
    is_regular = (nrows == n_integ * n_bl)

    amap = antenna_name_map or {}

    # ── baseline labels ───────────────────────────────────────────────────────
    bl_labels = [
        f'{amap.get(int(bl_unique[b, 0]), str(int(bl_unique[b, 0])))}'
        f':{amap.get(int(bl_unique[b, 1]), str(int(bl_unique[b, 1])))}'
        for b in range(n_bl)
    ]

    result: dict = {
        'nInteg':       n_integ,
        'nBL':          n_bl,
        'nChans':       nchans,
        'is_regular':   is_regular,
        'jd_unique':    jd_unique,
        'bl_labels':    bl_labels,
        'pct_total':    float(flag_mask.mean() * 100.0),
        # per-channel is safe without the 3D cube
        'pct_per_channel': (flag_mask.mean(axis=0) * 100.0),   # (nChans,)
    }

    if not is_regular:
        import warnings
        warnings.warn(
            f'materialise_flag_stats: grid is NOT regular '
            f'(nRows={nrows} ≠ nInteg({n_integ}) × nBL({n_bl})={n_integ*n_bl}). '
            f'3D statistics unavailable; returning per-channel stats only.',
            RuntimeWarning, stacklevel=2,
        )
        return result

    # ── build 3D cube via index assignment (order-independent) ───────────────
    flags_3d = np.zeros((n_integ, n_bl, nchans), dtype=bool)
    flags_3d[jd_indices, bl_indices, :] = flag_mask    # broadcast over chans

    result['flags_3d']             = flags_3d
    result['pct_per_baseline']     = (flags_3d.mean(axis=(0, 2)) * 100.0)   # (nBL,)
    result['pct_per_integration']  = (flags_3d.mean(axis=(1, 2)) * 100.0)   # (nInteg,)

    # ── per-antenna requires a companion lookup ───────────────────────────────
    ant_ids_sorted = sorted(set(ant1.tolist()) | set(ant2.tolist()))
    id_to_idx = {aid: i for i, aid in enumerate(ant_ids_sorted)}
    bl_a1 = np.array([id_to_idx[int(bl_unique[b, 0])] for b in range(n_bl)], dtype=np.int32)
    bl_a2 = np.array([id_to_idx[int(bl_unique[b, 1])] for b in range(n_bl)], dtype=np.int32)
    bl_ant_table = np.stack([bl_a1, bl_a2], axis=1)  # (nBL, 2)

    pct_per_antenna: dict = {}
    for i, aid in enumerate(ant_ids_sorted):
        bl_mask = (bl_ant_table[:, 0] == i) | (bl_ant_table[:, 1] == i)
        if bl_mask.any():
            pct = float(flags_3d[:, bl_mask, :].mean() * 100.0)
        else:
            pct = float('nan')
        name = amap.get(aid, str(aid))
        pct_per_antenna[name] = pct

    result['pct_per_antenna'] = pct_per_antenna
    result['bl_ant_table']    = bl_ant_table
    result['antenna_ids']     = ant_ids_sorted

    return result


def flux_model_3c48_perley_butler_2017(freq_hz):
    """Return the 3C48 Perley-Butler 2017 flux density model in Jy.

    Reference: Perley & Butler 2017, ApJS 230, 7, Table 2.
    Valid frequency range: 0.05 – 50 GHz.
    log10(S/Jy) = a0 + a1*x + a2*x² + a3*x³  where x = log10(ν/GHz)
    Coefficients: a = [1.3253, −0.7553, −0.1914, 0.0498]
    """
    freq_hz = np.asarray(freq_hz, dtype=np.float64)
    if np.any(freq_hz <= 0):
        raise ValueError('All frequencies must be positive.')
    freq_ghz = freq_hz / 1e9
    x = np.log10(freq_ghz)
    c = _PB2017_3C48_COEFFS
    return np.power(10.0, c[0] + c[1]*x + c[2]*x**2 + c[3]*x**3)


def flux_model_3c286_perley_butler_2017(freq_hz):
    """Return the 3C286 Perley-Butler 2017 flux density model in Jy.

    3C286 / J1331+3030 is a compact, steep-spectrum source widely used as a
    primary flux-density calibrator at GMRT and VLA.

    Reference: Perley & Butler 2017, ApJS 230, 7, Table 2.
    Valid frequency range: 0.05 – 50 GHz.
    log10(S/Jy) = a0 + a1*x + a2*x² + a3*x³  where x = log10(ν/GHz)
    Coefficients: a = [1.2481, −0.4507, −0.1798, 0.0357]
    """
    freq_hz = np.asarray(freq_hz, dtype=np.float64)
    if np.any(freq_hz <= 0):
        raise ValueError('All frequencies must be positive.')
    freq_ghz = freq_hz / 1e9
    x = np.log10(freq_ghz)
    c = _PB2017_3C286_COEFFS
    return np.power(10.0, c[0] + c[1]*x + c[2]*x**2 + c[3]*x**3)


# ── Flux model registry ────────────────────────────────────────────────────────
# Maps uppercase source name → flux-model callable (freq_hz → flux Jy array).
# Model-based outlier metrics ('RR', 'LL') can only be used when the calibrator
# source is present here.  'V' (Stokes proxy) is always available regardless.
_FLUX_MODEL_REGISTRY: dict = {
    '3C48':  flux_model_3c48_perley_butler_2017,
    '3C286': flux_model_3c286_perley_butler_2017,
}
# Metrics that require a flux-model entry from _FLUX_MODEL_REGISTRY.
_MODEL_BASED_METRICS: frozenset = frozenset({'RR', 'LL'})
# ──────────────────────────────────────────────────────────────────────────────


def _choose_reference_antenna(ant_numbers, weight_sums):
    ant_numbers = np.asarray(ant_numbers, dtype=np.int32)
    weight_sums = np.asarray(weight_sums, dtype=np.float64)
    if ant_numbers.size == 0:
        raise ValueError('No antennas available for reference selection.')
    return int(ant_numbers[int(np.nanargmax(weight_sums))])


def _build_channel_visibility_matrix(
    ant1,
    ant2,
    values,
    weights,
    antenna_ids,
    ignore_autos=True,
):
    nant = len(antenna_ids)
    ant_to_idx = {int(ant): idx for idx, ant in enumerate(antenna_ids)}

    ant1 = np.asarray(ant1, dtype=np.int32)
    ant2 = np.asarray(ant2, dtype=np.int32)
    values = np.asarray(values, dtype=np.complex128)
    weights = np.asarray(weights, dtype=np.float64)

    valid = np.isfinite(values.real) & np.isfinite(values.imag) & np.isfinite(weights) & (weights > 0)
    if ignore_autos:
        valid &= ant1 != ant2

    if not np.any(valid):
        return None, None, None, 0

    ant1_idx = np.array([ant_to_idx.get(int(ant), -1) for ant in ant1[valid]], dtype=np.int32)
    ant2_idx = np.array([ant_to_idx.get(int(ant), -1) for ant in ant2[valid]], dtype=np.int32)
    keep = (ant1_idx >= 0) & (ant2_idx >= 0)
    if not np.any(keep):
        return None, None, None, 0

    ant1_idx = ant1_idx[keep]
    ant2_idx = ant2_idx[keep]
    values = values[valid][keep]
    weights = weights[valid][keep]

    numer = np.zeros((nant, nant), dtype=np.complex128)
    denom = np.zeros((nant, nant), dtype=np.float64)
    np.add.at(numer, (ant1_idx, ant2_idx), weights * values)
    np.add.at(denom, (ant1_idx, ant2_idx), weights)

    numer = numer + numer.T.conj()
    denom = denom + denom.T
    np.fill_diagonal(denom, 0.0)

    matrix = np.zeros((nant, nant), dtype=np.complex128)
    have_data = denom > 0
    matrix[have_data] = numer[have_data] / denom[have_data]
    baseline_count = int(np.count_nonzero(np.triu(have_data, k=1)))
    connectivity = denom.sum(axis=1)
    return matrix, denom, connectivity, baseline_count


def _stefcal_solve_channel(
    matrix,
    weights,
    ref_ant_idx,
    max_iter=100,
    tol=1e-7,
):
    nant = matrix.shape[0]
    gains = np.ones(nant, dtype=np.complex128)
    valid_ant = weights.sum(axis=1) > 0
    gains[~valid_ant] = np.nan + 1j * np.nan

    for iteration in range(max_iter):
        prev = gains.copy()
        for ant_idx in range(nant):
            row_w = weights[ant_idx]
            row_m = matrix[ant_idx]
            neighbour_mask = row_w > 0
            if not np.any(neighbour_mask):
                gains[ant_idx] = np.nan + 1j * np.nan
                continue

            g_neigh = gains[neighbour_mask]
            finite = np.isfinite(g_neigh.real) & np.isfinite(g_neigh.imag)
            if not np.any(finite):
                gains[ant_idx] = np.nan + 1j * np.nan
                continue

            row_w = row_w[neighbour_mask][finite]
            row_m = row_m[neighbour_mask][finite]
            g_neigh = g_neigh[finite]

            denom = np.sum(row_w * np.abs(g_neigh) ** 2)
            if denom <= 0:
                gains[ant_idx] = np.nan + 1j * np.nan
                continue

            numer = np.sum(row_w * row_m * g_neigh)
            gains[ant_idx] = numer / denom

        if np.isfinite(gains[ref_ant_idx].real) and np.isfinite(gains[ref_ant_idx].imag) and np.abs(gains[ref_ant_idx]) > 0:
            gains /= gains[ref_ant_idx] / np.abs(gains[ref_ant_idx])

        finite_now = np.isfinite(gains.real) & np.isfinite(gains.imag)
        finite_prev = np.isfinite(prev.real) & np.isfinite(prev.imag)
        common = finite_now & finite_prev
        if np.any(common):
            delta = np.nanmax(np.abs(gains[common] - prev[common]))
            scale = np.nanmax(np.abs(prev[common]))
            if delta <= tol * max(scale, 1.0):
                return gains, iteration + 1

    return gains, max_iter


def _compute_channel_residual(matrix, weights, gains):
    finite = np.isfinite(gains.real) & np.isfinite(gains.imag)
    if np.count_nonzero(finite) < 2:
        return np.nan

    model = gains[:, np.newaxis] * gains[np.newaxis, :].conj()
    use = (weights > 0) & np.isfinite(model.real) & np.isfinite(model.imag)
    use = np.triu(use, k=1)
    if not np.any(use):
        return np.nan

    resid = matrix[use] - model[use]
    return float(np.sqrt(np.average(np.abs(resid) ** 2, weights=weights[use])))


def _smooth_complex_bandpass(gains, valid, window):
    if window is None or window <= 1:
        return gains

    window = int(window)
    if window % 2 == 0:
        window += 1

    kernel = np.ones(window, dtype=np.float64)
    smoothed = gains.copy()

    for pol_idx in range(gains.shape[2]):
        for ant_idx in range(gains.shape[1]):
            good = valid[:, ant_idx, pol_idx]
            if np.count_nonzero(good) < 2:
                continue

            g = gains[:, ant_idx, pol_idx]
            re = g.real.copy()
            im = g.imag.copy()
            good_f = good.astype(np.float64)

            # Smooth real and imaginary parts independently using a
            # weighted box-car.  This avoids the branch-cut ambiguity that
            # makes direct phase smoothing unreliable near ±π and across
            # channel gaps where np.unwrap can accumulate error.
            re_num  = np.convolve(np.where(good, re, 0.0), kernel, mode='same')
            im_num  = np.convolve(np.where(good, im, 0.0), kernel, mode='same')
            den     = np.convolve(good_f, kernel, mode='same')
            ok = den > 0

            re_s = re.copy()
            im_s = im.copy()
            re_s[ok] = re_num[ok] / den[ok]
            im_s[ok] = im_num[ok] / den[ok]

            smoothed[:, ant_idx, pol_idx] = re_s + 1j * im_s

    return smoothed


def derive_point_source_bandpass(
    index,
    source='3C48',
    ant_range=None,
    ant_list=None,
    chan_range=None,
    stokes=('RR', 'LL'),
    max_rows=150_000,
    model_flux_jy=None,
    reference_antenna=None,
    smooth_window=5,
    min_baselines=20,
    ignore_autos=True,
    max_iter=100,
    tol=1e-7,
    flag_table: Optional[Union[dict, List[dict], Tuple[dict, ...]]] = None,
    flag_table_path: Optional[Union[str, Path, List[Union[str, Path]], Tuple[Union[str, Path], ...]]] = None,
    strict_flag_table: bool = False,
    flag_all_corrs_if_any_rawvis_flagged: bool = False,
    # ── DataSelection convenience params ─────────────────────────────────────
    selection: Optional['DataSelection'] = None,
    timerange: Optional[Tuple] = None,
    uvrange_m: Optional[Tuple[float, float]] = None,
    uvrange_klambda: Optional[Tuple[float, float]] = None,
    elevation_min_deg: Optional[float] = None,
    elevation_max_deg: Optional[float] = None,
):
    """Derive per-antenna complex bandpass gains without modifying FITS data.

    This function only reads visibilities from disk, derives gains in memory,
    and returns a separate solution table that can be saved independently.

    A :class:`DataSelection` object (or equivalent keyword arguments) can be
    used to restrict the data used for solving: by time, UV range, or
    elevation.  This is useful for example to discard low-elevation data::

        sel = DataSelection(elevation_min_deg=25.0)
        sol = derive_point_source_bandpass(index, '3C48', selection=sel)
    """
    vis = load_vis_for_source(
        index,
        source=source,
        ant_range=ant_range,
        ant_list=ant_list,
        chan_range=chan_range,
        stokes=list(stokes),
        max_rows=max_rows,
        flag_all_corrs_if_any_rawvis_flagged=flag_all_corrs_if_any_rawvis_flagged,
        selection=selection,
        timerange=timerange,
        uvrange_m=uvrange_m,
        uvrange_klambda=uvrange_klambda,
        elevation_min_deg=elevation_min_deg,
        elevation_max_deg=elevation_max_deg,
    )

    # Observed active baselines/antennas: computed BEFORE flag-table row drops
    # so this is the true denominator for all flagging-fraction statistics.
    # (DUD antennas have no rows in FITS, so they naturally drop out here.)
    _obs_ant1 = np.asarray(vis['ant1'], dtype=np.int32)
    _obs_ant2 = np.asarray(vis['ant2'], dtype=np.int32)
    _obs_active_bls  = len(set(zip(_obs_ant1.tolist(), _obs_ant2.tolist())))
    _obs_active_ants = len(set(_obs_ant1.tolist()) | set(_obs_ant2.tolist()))

    antenna_name_map = {
        int(item['antenna_no']): (item.get('name') or f'Ant{int(item["antenna_no"])}')
        for item in index.get('antennas', [])
        if item.get('antenna_no') is not None
    }

    flag_tables, flag_table_paths = _coerce_flag_tables(
        flag_tables=flag_table,
        flag_table_paths=flag_table_path,
    )

    flag_stats = {
        'dropped_rows': 0,
        'kept_rows': int(vis.get('nrows', len(vis.get('ant1', []))),),
        'excluded_antenna_ids': [],
        'excluded_baseline_pairs': [],
        'flag_table_notes': '',
        'flag_table_count': 0,
        'flag_table_paths': [],
    }
    if flag_tables:
        vis, flag_stats = apply_flag_tables_to_vis(
            vis,
            antenna_name_map=antenna_name_map,
            flag_tables=flag_tables,
            flag_table_paths=flag_table_paths,
            strict=bool(strict_flag_table),
            context='bandpass-solve',
        )
        print(
            '[bandpass-solve] merged flag tables applied: '
            f'dropped {flag_stats["dropped_rows"]:,} rows, kept {flag_stats["kept_rows"]:,} rows; '
            f'excluded antennas={flag_stats["excluded_antenna_ids"]}, '
            f'excluded baselines={len(flag_stats["excluded_baseline_pairs"])}; '
            f'tables={flag_stats.get("flag_table_count", 0)}'
        )

    antenna_ids = np.array(sorted(set(vis['ant1']).union(set(vis['ant2']))), dtype=np.int32)
    if antenna_ids.size < 2:
        raise ValueError('Need at least two antennas to derive bandpass solutions.')

    antenna_names = [antenna_name_map.get(int(ant), f'Ant{int(ant)}') for ant in antenna_ids]

    stokes_labels = list(vis['stokes_labels'])
    freqs_hz = np.asarray(vis['freqs_hz'], dtype=np.float64)
    if model_flux_jy is not None:
        flux_jy = np.asarray(model_flux_jy, dtype=np.float64)
    else:
        _model_fn = _FLUX_MODEL_REGISTRY.get(str(source).upper())
        if _model_fn is None:
            raise ValueError(
                f'No flux density model registered for source "{source}". '
                f'Registered sources: {sorted(_FLUX_MODEL_REGISTRY)}. '
                f'Pass model_flux_jy explicitly or register a model via _FLUX_MODEL_REGISTRY.'
            )
        flux_jy = _model_fn(freqs_hz)
    if flux_jy.shape != freqs_hz.shape:
        raise ValueError('model_flux_jy must have one value per selected channel.')

    vis_complex = np.asarray(vis['vis_complex'], dtype=np.complex128)
    weights = np.asarray(vis['weight'], dtype=np.float64)
    nant = antenna_ids.size
    nchan = freqs_hz.size
    npol = len(stokes_labels)

    gains = np.full((nchan, nant, npol), np.nan + 1j * np.nan, dtype=np.complex128)
    valid = np.zeros((nchan, nant, npol), dtype=bool)
    residual_rms = np.full((nchan, npol), np.nan, dtype=np.float64)
    baseline_counts = np.zeros((nchan, npol), dtype=np.int32)
    iterations_used = np.zeros((nchan, npol), dtype=np.int32)
    weight_sums = np.zeros((nant, npol), dtype=np.float64)
    input_flagged_counts = np.count_nonzero(vis['flagged'], axis=0).astype(np.int32)
    used_sample_counts = np.zeros((nchan, npol), dtype=np.int32)
    skipped_channel_mask = np.ones((nchan, npol), dtype=bool)

    if reference_antenna is not None and int(reference_antenna) not in set(int(a) for a in antenna_ids):
        raise ValueError(f'Reference antenna {reference_antenna} is not present in the selected data.')

    ref_indices = []
    for pol_idx in range(npol):
        for chan_idx in range(nchan):
            norm_values = vis_complex[:, chan_idx, pol_idx] / flux_jy[chan_idx]
            matrix, matrix_w, connectivity, nbase = _build_channel_visibility_matrix(
                vis['ant1'],
                vis['ant2'],
                norm_values,
                weights[:, chan_idx, pol_idx],
                antenna_ids,
                ignore_autos=ignore_autos,
            )
            baseline_counts[chan_idx, pol_idx] = nbase
            if matrix is None or nbase < min_baselines:
                continue

            used_sample_counts[chan_idx, pol_idx] = int(np.count_nonzero(matrix_w[np.triu_indices_from(matrix_w, k=1)] > 0))
            skipped_channel_mask[chan_idx, pol_idx] = False

            weight_sums[:, pol_idx] += connectivity
            if reference_antenna is None:
                ref_ant = _choose_reference_antenna(antenna_ids, connectivity)
            else:
                ref_ant = int(reference_antenna)
            ref_idx = int(np.where(antenna_ids == ref_ant)[0][0])

            chan_gains, n_iter = _stefcal_solve_channel(
                matrix,
                matrix_w,
                ref_ant_idx=ref_idx,
                max_iter=max_iter,
                tol=tol,
            )
            gains[chan_idx, :, pol_idx] = chan_gains
            valid[chan_idx, :, pol_idx] = np.isfinite(chan_gains.real) & np.isfinite(chan_gains.imag)
            residual_rms[chan_idx, pol_idx] = _compute_channel_residual(matrix, matrix_w, chan_gains)
            iterations_used[chan_idx, pol_idx] = n_iter
            ref_indices.append(ref_idx)

    if reference_antenna is None:
        chosen_ref = _choose_reference_antenna(antenna_ids, weight_sums.sum(axis=1))
        chosen_ref_idx = int(np.where(antenna_ids == chosen_ref)[0][0])
        _ref_origin = 'auto-selected (most-connected)'
        finite = np.isfinite(gains.real) & np.isfinite(gains.imag)
        for pol_idx in range(npol):
            for chan_idx in range(nchan):
                if not finite[chan_idx, chosen_ref_idx, pol_idx]:
                    continue
                phase_ref = gains[chan_idx, chosen_ref_idx, pol_idx] / np.abs(gains[chan_idx, chosen_ref_idx, pol_idx])
                gains[chan_idx, :, pol_idx] /= phase_ref
    else:
        chosen_ref = int(reference_antenna)
        chosen_ref_idx = int(np.where(antenna_ids == chosen_ref)[0][0])
        _ref_origin = 'user-specified'

    _ref_name = antenna_names[chosen_ref_idx] if chosen_ref_idx < len(antenna_names) else str(chosen_ref)
    print(f'[bandpass-solve] reference antenna: {_ref_name} (antenna_no={chosen_ref})  [{_ref_origin}]')

    gains_raw = gains.copy()  # preserve pre-smoothing solutions
    gains = _smooth_complex_bandpass(gains, valid, smooth_window)

    return {
        'kind': 'point_source_bandpass',
        'source_name': str(source),
        'source_file': str(index['path']),
        'freqs_hz': freqs_hz,
        'chan_indices': np.asarray(vis['chan_indices'], dtype=np.int32),
        'antenna_ids': antenna_ids,
        'antenna_names': antenna_names,
        'stokes_labels': stokes_labels,
        'gains': gains,
        'gains_raw': gains_raw,  # unsmoothed per-channel solutions
        'valid': valid,
        'residual_rms': residual_rms,
        'baseline_counts': baseline_counts,
        'input_flagged_counts': input_flagged_counts,
        'used_sample_counts': used_sample_counts,
        'skipped_channel_mask': skipped_channel_mask,
        'iterations': iterations_used,
        'flux_model_jy': flux_jy,
        'reference_antenna': chosen_ref,
        'smooth_window': int(smooth_window) if smooth_window is not None else None,
        'max_rows': int(max_rows),
        'min_baselines': int(min_baselines),
        'flag_table_applied': bool(flag_stats.get('flag_table_count', 0) > 0),
        'flag_table_path': flag_stats.get('flag_table_paths', [None])[0] if flag_stats.get('flag_table_paths') else None,
        'flag_table_paths': list(flag_stats.get('flag_table_paths', [])),
        'flag_table_count': int(flag_stats.get('flag_table_count', 0)),
        'flag_table_notes': str(flag_stats.get('flag_table_notes', '')),
        'excluded_antenna_ids': np.asarray(flag_stats.get('excluded_antenna_ids', []), dtype=np.int32),
        'excluded_baseline_pairs': np.asarray(flag_stats.get('excluded_baseline_pairs', []), dtype=np.int32).reshape(-1, 2),
        'solve_input_rows': int(flag_stats.get('kept_rows', vis.get('nrows', 0))),
        'solve_dropped_rows_by_flag_table': int(flag_stats.get('dropped_rows', 0)),
        # Observed counts (from raw vis before flag-table row drops):
        'obs_active_baselines': _obs_active_bls,
        'obs_active_antennas':  _obs_active_ants,
        'bad_data_policy': (
            'Ignored non-finite samples, zero-or-negative FITS weights, optional autos, '
            'and channels with insufficient surviving baselines. Optional flag-table exclusions '
            'were applied at solve input. No flags were written to FITS.'
        ),
        'notes': 'Derived from visibilities in memory only. No FITS data were modified.',
    }


def save_bandpass_solution(solution, path: Union[str, Path]):
    """Write bandpass solutions to a separate compressed file on disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # flag_table_notes is a ' | '.join() of every accumulated flag-table notes
    # string; after many iterations it can exceed NumPy's scalar string limit.
    # Truncate to 4 KB — it is human-readable metadata only, not used by code.
    _ft_notes = str(solution.get('flag_table_notes', ''))
    if len(_ft_notes) > 4096:
        _ft_notes = _ft_notes[:4093] + '...'

    metadata = {
        'kind': solution['kind'],
        'source_name': solution['source_name'],
        'source_file': solution['source_file'],
        'reference_antenna': int(solution['reference_antenna']),
        'smooth_window': solution['smooth_window'],
        'max_rows': int(solution['max_rows']),
        'min_baselines': int(solution['min_baselines']),
        'flag_table_applied': bool(solution.get('flag_table_applied', False)),
        'flag_table_path': solution.get('flag_table_path'),
        'flag_table_paths': list(solution.get('flag_table_paths', [])),
        'flag_table_count': int(solution.get('flag_table_count', 0)),
        'flag_table_notes': _ft_notes,
        'solve_input_rows': int(solution.get('solve_input_rows', 0)),
        'solve_dropped_rows_by_flag_table': int(solution.get('solve_dropped_rows_by_flag_table', 0)),
        'bad_data_policy': solution['bad_data_policy'],
        'notes': solution['notes'],
    }

    np.savez_compressed(
        path,
        freqs_hz=np.asarray(solution['freqs_hz'], dtype=np.float64),
        chan_indices=np.asarray(solution['chan_indices'], dtype=np.int32),
        antenna_ids=np.asarray(solution['antenna_ids'], dtype=np.int32),
        antenna_names=np.asarray(solution['antenna_names'], dtype='U32'),
        stokes_labels=np.asarray(solution['stokes_labels'], dtype='U16'),
        gains=np.asarray(solution['gains'], dtype=np.complex128),
        gains_raw=np.asarray(solution.get('gains_raw', solution['gains']), dtype=np.complex128),
        valid=np.asarray(solution['valid'], dtype=bool),
        residual_rms=np.asarray(solution['residual_rms'], dtype=np.float64),
        baseline_counts=np.asarray(solution['baseline_counts'], dtype=np.int32),
        input_flagged_counts=np.asarray(solution['input_flagged_counts'], dtype=np.int32),
        used_sample_counts=np.asarray(solution['used_sample_counts'], dtype=np.int32),
        skipped_channel_mask=np.asarray(solution['skipped_channel_mask'], dtype=bool),
        iterations=np.asarray(solution['iterations'], dtype=np.int32),
        flux_model_jy=np.asarray(solution['flux_model_jy'], dtype=np.float64),
        excluded_antenna_ids=np.asarray(solution.get('excluded_antenna_ids', []), dtype=np.int32),
        excluded_baseline_pairs=np.asarray(solution.get('excluded_baseline_pairs', []), dtype=np.int32).reshape(-1, 2),
        metadata_json=json.dumps(metadata),
    )
    return path


def load_bandpass_solution(path: Union[str, Path]) -> dict:
    """Read a saved bandpass solution file produced by save_bandpass_solution."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as npz:
        metadata = json.loads(str(npz['metadata_json']))
        return {
            'kind': metadata['kind'],
            'source_name': metadata['source_name'],
            'source_file': metadata['source_file'],
            'reference_antenna': int(metadata['reference_antenna']),
            'smooth_window': metadata['smooth_window'],
            'max_rows': int(metadata['max_rows']),
            'min_baselines': int(metadata['min_baselines']),
            'flag_table_applied': bool(metadata.get('flag_table_applied', False)),
            'flag_table_path': metadata.get('flag_table_path'),
            'flag_table_paths': list(metadata.get('flag_table_paths', [])),
            'flag_table_count': int(metadata.get('flag_table_count', 0)),
            'flag_table_notes': metadata.get('flag_table_notes', ''),
            'solve_input_rows': int(metadata.get('solve_input_rows', 0)),
            'solve_dropped_rows_by_flag_table': int(metadata.get('solve_dropped_rows_by_flag_table', 0)),
            'bad_data_policy': metadata['bad_data_policy'],
            'notes': metadata['notes'],
            'freqs_hz': np.asarray(npz['freqs_hz'], dtype=np.float64),
            'chan_indices': np.asarray(npz['chan_indices'], dtype=np.int32),
            'antenna_ids': np.asarray(npz['antenna_ids'], dtype=np.int32),
            'antenna_names': np.asarray(npz['antenna_names']).astype(str).tolist(),
            'stokes_labels': np.asarray(npz['stokes_labels']).astype(str).tolist(),
            'gains': np.asarray(npz['gains'], dtype=np.complex128),
            'gains_raw': (
                np.asarray(npz['gains_raw'], dtype=np.complex128)
                if 'gains_raw' in npz.files
                else None  # older files saved before gains_raw was added
            ),
            'valid': np.asarray(npz['valid'], dtype=bool),
            'residual_rms': np.asarray(npz['residual_rms'], dtype=np.float64),
            'baseline_counts': np.asarray(npz['baseline_counts'], dtype=np.int32),
            'input_flagged_counts': np.asarray(npz['input_flagged_counts'], dtype=np.int32),
            'used_sample_counts': np.asarray(npz['used_sample_counts'], dtype=np.int32),
            'skipped_channel_mask': np.asarray(npz['skipped_channel_mask'], dtype=bool),
            'iterations': np.asarray(npz['iterations'], dtype=np.int32),
            'flux_model_jy': np.asarray(npz['flux_model_jy'], dtype=np.float64),
            'excluded_antenna_ids': (
                np.asarray(npz['excluded_antenna_ids'], dtype=np.int32)
                if 'excluded_antenna_ids' in npz.files
                else np.asarray([], dtype=np.int32)
            ),
            'excluded_baseline_pairs': (
                np.asarray(npz['excluded_baseline_pairs'], dtype=np.int32).reshape(-1, 2)
                if 'excluded_baseline_pairs' in npz.files
                else np.asarray([], dtype=np.int32).reshape(-1, 2)
            ),
        }


def tagged_output_path(path: Union[str, Path], tag: Optional[str] = None) -> Path:
    """Return a path with an iteration tag inserted before the suffix."""
    path = Path(path)
    if not tag:
        return path
    suffix = ''.join(path.suffixes)
    stem = path.name[:-len(suffix)] if suffix else path.name
    return path.with_name(f'{stem}_{tag}{suffix}')


def derive_bandpass_iteration(
    fits_path: Union[str, Path],
    bandpass_out: Optional[Union[str, Path]] = None,
    *,
    source: str = '3C48',
    index: Optional[dict] = None,
    index_cache_path: Optional[Union[str, Path]] = None,
    index_cache_dir: Optional[Union[str, Path]] = None,
    force_rebuild_index: bool = False,
    verify_index_sha256: Optional[bool] = None,
    index_validation_mode: str = 'fast',
    write_index_cache: bool = True,
    iteration_tag: Optional[str] = None,
    dry_run: bool = False,
    flag_table=None,
    flag_table_path=None,
    **solve_kwargs,
) -> dict:
    """High-level non-destructive bandpass derivation workflow."""
    if verify_index_sha256 is not None:
        index_validation_mode = 'sha256' if verify_index_sha256 else 'none'

    if index is None:
        index = get_or_build_row_index(
            fits_path,
            cache_path=index_cache_path,
            cache_dir=index_cache_dir,
            force_rebuild=force_rebuild_index,
            validation_mode=index_validation_mode,
            write_cache=write_index_cache,
        )

    solution = derive_point_source_bandpass(
        index,
        source=source,
        flag_table=flag_table,
        flag_table_path=flag_table_path,
        **solve_kwargs,
    )

    out_path = tagged_output_path(bandpass_out, iteration_tag) if bandpass_out is not None else None
    if out_path is not None and not dry_run:
        save_bandpass_solution(solution, out_path)

    return {
        'index': index,
        'solution': solution,
        'bandpass_out': str(out_path) if out_path is not None else None,
        'iteration_tag': iteration_tag,
        'dry_run': bool(dry_run),
    }


def run_bandpass_diagnostics(
    index: dict,
    solution: Optional[dict],
    *,
    source: str = '3C48',
    chan_range=None,
    stokes=('RR', 'LL'),
    max_rows: int = 60_000,
    exclude_antennas=None,
    apply_flag_tables: bool = False,
    flag_table=None,
    flag_table_path=None,
    strict_flag_table: bool = False,
    flag_all_corrs_if_any_rawvis_flagged: bool = False,
    apply_correction: bool = True,
    skip_edge_channels: Union[int, Tuple[int, int]] = (10, 5),
    top_n: int = 12,
    ranking_metric: Union[str, tuple, list, None] = None,
    title: str = '',
    save_path: Optional[Union[str, Path]] = None,
    # ── DataSelection convenience params ────────────────────────────────────
    selection: Optional['DataSelection'] = None,
    timerange: Optional[Tuple] = None,
    uvrange_m: Optional[Tuple[float, float]] = None,
    uvrange_klambda: Optional[Tuple[float, float]] = None,
    elevation_min_deg: Optional[float] = None,
    elevation_max_deg: Optional[float] = None,
):
    """Run RR/LL residual diagnostics and return structured results.

    solution        - bandpass solution dict, or None when apply_correction=False.
    apply_correction - if True (default), apply bandpass solution before computing
                       residuals.  If False, use raw visibilities with no correction;
                       solution may be None in this case.  Useful for an iter-00
                       baseline view showing the uncorrected data state.
    ranking_metric  - metric(s) used for flagging in this run.  When provided,
                      the corresponding per-metric bar-chart rows are annotated with
                      'used for flagging' so it is clear which signals drove the flags.
                      Accepts the same format as outlier_metric (string or tuple).
    """
    vis = load_vis_for_source(
        index,
        source=source,
        stokes=list(stokes),
        max_rows=max_rows,
        ant_range=None,
        ant_list=None,
        chan_range=chan_range,
        flag_all_corrs_if_any_rawvis_flagged=flag_all_corrs_if_any_rawvis_flagged,
        selection=selection,
        timerange=timerange,
        uvrange_m=uvrange_m,
        uvrange_klambda=uvrange_klambda,
        elevation_min_deg=elevation_min_deg,
        elevation_max_deg=elevation_max_deg,
    )

    diagnostics_flag_stats = {
        'dropped_rows': 0,
        'kept_rows': int(vis.get('nrows', len(vis.get('ant1', [])))),
        'excluded_antenna_ids': [],
        'excluded_baseline_pairs': [],
        'flag_table_notes': '',
        'flag_table_count': 0,
        'flag_table_paths': [],
    }
    vis_for_diag = vis
    if apply_flag_tables:
        antenna_name_map = {
            int(item['antenna_no']): (item.get('name') or f'Ant{int(item["antenna_no"])}')
            for item in index.get('antennas', [])
            if item.get('antenna_no') is not None
        }
        vis_for_diag, diagnostics_flag_stats = apply_flag_tables_to_vis(
            vis,
            antenna_name_map=antenna_name_map,
            flag_tables=flag_table,
            flag_table_paths=flag_table_path,
            strict=bool(strict_flag_table),
            context='bandpass-diagnostics',
        )
        print(
            '[bandpass-diagnostics] merged flag tables applied: '
            f'dropped {diagnostics_flag_stats["dropped_rows"]:,} rows, kept {diagnostics_flag_stats["kept_rows"]:,} rows; '
            f'excluded antennas={diagnostics_flag_stats["excluded_antenna_ids"]}, '
            f'excluded baselines={len(diagnostics_flag_stats["excluded_baseline_pairs"])}; '
            f'tables={diagnostics_flag_stats.get("flag_table_count", 0)}'
        )

    vis_use = vis_for_diag
    if apply_correction and solution is not None:
        vis_use = _filter_vis_excluded_antennas(vis_for_diag, solution, exclude_antennas=exclude_antennas)
        corrected = apply_bandpass_solution(vis_use, solution)
    else:
        # Raw mode: no bandpass correction.  Re-package raw vis as the
        # 'corrected' dict so the rest of the function is unchanged.
        corrected = {
            'freqs_hz':              vis_use['freqs_hz'],
            'vis_complex_corrected': np.array(vis_use['vis_complex'], dtype=np.complex128),
            'weight':                vis_use['weight'],
            'flagged_corrected':     vis_use.get(
                                         'flagged',
                                         np.zeros(vis_use['weight'].shape, dtype=bool)
                                     ),
            'stokes_labels':         vis_use['stokes_labels'],
            'ant1':                  vis_use['ant1'],
            'ant2':                  vis_use['ant2'],
        }

    freqs_hz = np.asarray(corrected['freqs_hz'], dtype=np.float64)
    freqs_mhz = freqs_hz / 1e6

    # Look up the Perley-Butler flux model for this source.
    # Model-based metrics (RR, LL) require the source to be in _FLUX_MODEL_REGISTRY.
    # The Stokes-V proxy metric is always computed regardless.
    _model_fn = _FLUX_MODEL_REGISTRY.get(source.upper()) if source else None
    source_has_flux_model = _model_fn is not None
    if source_has_flux_model:
        model: Optional[np.ndarray] = _model_fn(freqs_hz)
    else:
        model = None
        _supported = ', '.join(sorted(_FLUX_MODEL_REGISTRY))
        print(
            f'[bandpass-diagnostics] WARNING: source "{source}" has no flux model in the '
            f'code registry (registered calibrators: {_supported}).  '
            f'Model-based outlier metrics (RR, LL) will be skipped for this source.  '
            f'Only the Stokes-V proxy metric is available.'
        )

    if isinstance(skip_edge_channels, tuple):
        if len(skip_edge_channels) != 2:
            raise ValueError('skip_edge_channels tuple must have exactly two values: (start, end).')
        skip_start, skip_end = int(skip_edge_channels[0]), int(skip_edge_channels[1])
    else:
        skip_start = skip_end = int(skip_edge_channels)

    chan_mask = np.ones(freqs_hz.size, dtype=bool)
    if skip_start > 0:
        chan_mask[:min(skip_start, chan_mask.size)] = False
    if skip_end > 0:
        chan_mask[max(0, chan_mask.size - skip_end):] = False

    vis_corr = np.asarray(corrected['vis_complex_corrected'], dtype=np.complex128)
    weights = np.asarray(corrected['weight'], dtype=np.float64)
    flagged = np.asarray(corrected.get('flagged_corrected', corrected.get('flagged')), dtype=bool)
    stokes_labels = list(corrected['stokes_labels'])
    ant1 = np.asarray(corrected['ant1'], dtype=np.int32)
    ant2 = np.asarray(corrected['ant2'], dtype=np.int32)

    if solution is not None:
        ant_ids = np.asarray(solution['antenna_ids'], dtype=np.int32)
        ant_names = list(solution.get('antenna_names') or [str(int(a)) for a in ant_ids])
    else:
        # Derive antenna names from the row index when no solution is available.
        _index_ants = [item for item in index.get('antennas', []) if item.get('antenna_no') is not None]
        ant_ids = np.asarray([item['antenna_no'] for item in _index_ants], dtype=np.int32)
        ant_names = [item.get('name', f'Ant{int(item["antenna_no"])}') for item in _index_ants]
    ant_name_map = {int(ant): str(name) for ant, name in zip(ant_ids, ant_names)}

    # Determine how many per-metric rows the figure needs so we can size it
    # correctly before any data is drawn.  One row per computed metric:
    #   RR, LL  — only when a Perley-Butler flux model exists for the source
    #   V       — always, when both RR and LL are loaded
    _pre_metrics: List[str] = []
    if source_has_flux_model:
        _pre_metrics.extend(s for s in ('RR', 'LL') if s in stokes_labels)
    if 'RR' in stokes_labels and 'LL' in stokes_labels:
        _pre_metrics.append('V')
    _n_metric_rows = max(1, len(_pre_metrics))
    _n_rows = 1 + _n_metric_rows   # spectrum row + one row per metric

    from matplotlib.gridspec import GridSpec as _GridSpec, GridSpecFromSubplotSpec as _GSSS
    fig = plt.figure(figsize=(15, 8 + 5 * _n_metric_rows))
    _gs = _GridSpec(_n_rows, 2, figure=fig, hspace=0.38, wspace=0.24)
    ax_spec  = fig.add_subplot(_gs[0, 0])
    # Right panel: split into residuals (top) + ripple-cleaned (bottom).
    _gs_right = _GSSS(2, 1, subplot_spec=_gs[0, 1], hspace=0.80,
                      height_ratios=[3, 2])
    ax_resid = fig.add_subplot(_gs_right[0])
    ax_clean = fig.add_subplot(_gs_right[1])

    if source_has_flux_model:
        model_plot = np.where(chan_mask, model, np.nan)
        ax_spec.plot(freqs_mhz, model_plot, color='k', lw=2.0, label='Perley-Butler 2017')

    pol_results = {}
    _pol_cleaned: dict = {}   # sinusoid-subtracted residual spectra keyed by pol name

    def _fit_sinusoid(
        f_mhz: np.ndarray,
        y: np.ndarray,
        period_range_mhz: tuple = (0.5, 20.0),
    ):
        """Fit a + A*sin(2π*f/P + φ) to masked (f, y). Returns (a, A, P, phi) or None.

        Uses an FFT periodogram to seed the initial period guess so that
        curve_fit reliably converges to the dominant ripple even when its
        period is far from the centre of period_range_mhz.
        """
        from scipy.optimize import curve_fit as _cf
        if f_mhz.size < 20:
            return None
        _valid = np.isfinite(y)
        if _valid.sum() < 20:
            return None

        fv = f_mhz[_valid]
        yv = y[_valid]

        # ── FFT-based initial period estimate ──────────────────────────────────
        # Interpolate onto a uniform grid then FFT to find the dominant frequency
        # within the allowed period range.
        n_grid = 512
        f_grid = np.linspace(fv[0], fv[-1], n_grid)
        y_grid = np.interp(f_grid, fv, yv)
        y_grid -= y_grid.mean()
        df = (f_grid[-1] - f_grid[0]) / (n_grid - 1)           # MHz per sample
        fft_amp = np.abs(np.fft.rfft(y_grid))
        fft_freq = np.fft.rfftfreq(n_grid, d=df)               # cycles / MHz  →  period = 1/freq
        # Avoid DC bin and periods outside the allowed range
        with np.errstate(divide='ignore'):
            fft_period = np.where(fft_freq > 0, 1.0 / fft_freq, np.inf)
        in_range = (fft_period >= period_range_mhz[0]) & (fft_period <= period_range_mhz[1])
        if in_range.any():
            p0_period = float(fft_period[in_range][np.argmax(fft_amp[in_range])])
        else:
            p0_period = float(np.sqrt(period_range_mhz[0] * period_range_mhz[1]))  # geometric mid

        def _model(f, a, A, P, phi):
            return a + A * np.sin(2.0 * np.pi * f / P + phi)

        try:
            popt, _ = _cf(
                _model, fv, yv,
                p0=[float(np.median(yv)), float(np.std(yv)), p0_period, 0.0],
                bounds=([-np.inf, -np.inf, period_range_mhz[0], -np.pi],
                        [ np.inf,  np.inf, period_range_mhz[1],  np.pi]),
                maxfev=20_000,
            )
            return popt
        except Exception:
            return None

    unique_pairs, inv = np.unique(np.column_stack([ant1, ant2]), axis=0, return_inverse=True)

    for pol_idx, pol in enumerate(stokes_labels):
        z = vis_corr[:, :, pol_idx]
        w = weights[:, :, pol_idx]
        good = (
            (~flagged[:, :, pol_idx])
            & np.isfinite(z.real)
            & np.isfinite(z.imag)
            & np.isfinite(w)
            & (w > 0)
            & chan_mask[None, :]
        )

        num = np.nansum(np.where(good, w * z, 0.0), axis=0)
        den = np.nansum(np.where(good, w, 0.0), axis=0)
        vec = np.full(freqs_hz.size, np.nan + 1j * np.nan, dtype=np.complex128)
        ok = den > 0
        vec[ok] = num[ok] / den[ok]
        real_spec = np.real(vec)

        # Always plot the per-pol vector-averaged spectrum so the user can inspect
        # the bandpass shape even when no flux model is registered for the source.
        ax_spec.plot(freqs_mhz, np.where(chan_mask, real_spec, np.nan), lw=1.0, label=f'{pol}')

        # Residual records (and diagnostic scores) require a flux model — skip
        # building pol_results[pol] for RR/LL when no model is available.
        if not source_has_flux_model:
            continue

        resid_spec = real_spec - model

        _resid_line = ax_resid.plot(
            freqs_mhz, np.where(chan_mask, resid_spec, np.nan),
            lw=1.0, label=pol)[0]
        _lc = _resid_line.get_color()
        _fit_annot_lines = []   # collect annotation strings for this pol
        _popt_r = _fit_sinusoid(freqs_mhz[chan_mask], resid_spec[chan_mask])
        if _popt_r is not None:
            _a_r, _A_r, _P_r, _phi_r = _popt_r
            _sinu_r = _a_r + _A_r * np.sin(2.0 * np.pi * freqs_mhz / _P_r + _phi_r)
            ax_resid.plot(
                freqs_mhz, np.where(chan_mask, _sinu_r, np.nan),
                lw=1.4, ls='--', color=_lc, alpha=0.75,
            )
            _fit_annot_lines.append(f'{pol}₁: {_A_r:.2f}Jy, {_P_r:.1f}MHz')
            _cleaned_r = resid_spec - _sinu_r
            # ── Pass 2: if pass 1 found a long-period component (P > 4 MHz),
            # search for a secondary short-period ripple in the 0.5–4 MHz window.
            if _P_r > 4.0:
                _popt_r2 = _fit_sinusoid(
                    freqs_mhz[chan_mask], _cleaned_r[chan_mask],
                    period_range_mhz=(0.5, 4.0),
                )
                if _popt_r2 is not None:
                    _a_r2, _A_r2, _P_r2, _phi_r2 = _popt_r2
                    _sinu_r2 = _a_r2 + _A_r2 * np.sin(2.0 * np.pi * freqs_mhz / _P_r2 + _phi_r2)
                    ax_resid.plot(
                        freqs_mhz, np.where(chan_mask, _sinu_r2, np.nan),
                        lw=1.4, ls=':', color=_lc, alpha=0.60,
                    )
                    _fit_annot_lines.append(f'{pol}₂: {_A_r2:.2f}Jy, {_P_r2:.1f}MHz')
                    _cleaned_r = _cleaned_r - _sinu_r2
            ax_clean.plot(
                freqs_mhz, np.where(chan_mask, _cleaned_r, np.nan),
                lw=1.0, color=_lc, label=pol,
            )
            _pol_cleaned[pol] = _cleaned_r
        else:
            ax_clean.plot(
                freqs_mhz, np.where(chan_mask, resid_spec, np.nan),
                lw=1.0, ls='--', color=_lc, label=f'{pol} (no fit)',
            )
            _pol_cleaned[pol] = resid_spec
        # Store annotation lines for this pol so we can add a text box after all pols
        if not hasattr(ax_resid, '_fit_annot'):
            ax_resid._fit_annot = []
        ax_resid._fit_annot.extend(_fit_annot_lines)

        baseline_records = []
        for base_idx, (a1, a2) in enumerate(unique_pairs):
            row_sel = inv == base_idx
            if not np.any(row_sel):
                continue
            base_good = good[row_sel]
            base_w = w[row_sel]
            base_resid = np.real(z[row_sel]) - model[None, :]
            den_base = np.nansum(np.where(base_good, base_w, 0.0))
            if den_base <= 0:
                continue
            mean_resid = np.nansum(np.where(base_good, base_w * base_resid, 0.0)) / den_base
            mean_abs_resid = np.nansum(np.where(base_good, base_w * np.abs(base_resid), 0.0)) / den_base
            baseline_records.append({
                'pair': (int(a1), int(a2)),
                'label': f'{ant_name_map.get(int(a1), a1)}-{ant_name_map.get(int(a2), a2)}',
                'mean_resid': float(mean_resid),
                'mean_abs_resid': float(mean_abs_resid),
            })

        antenna_records = []
        for ant in np.unique(np.concatenate([ant1, ant2])):
            row_sel = (ant1 == ant) | (ant2 == ant)
            if not np.any(row_sel):
                continue
            ant_good = good[row_sel]
            ant_w = w[row_sel]
            ant_resid = np.real(z[row_sel]) - model[None, :]
            den_ant = np.nansum(np.where(ant_good, ant_w, 0.0))
            if den_ant <= 0:
                continue
            mean_resid = np.nansum(np.where(ant_good, ant_w * ant_resid, 0.0)) / den_ant
            mean_abs_resid = np.nansum(np.where(ant_good, ant_w * np.abs(ant_resid), 0.0)) / den_ant
            antenna_records.append({
                'ant': int(ant),
                'label': ant_name_map.get(int(ant), str(int(ant))),
                'mean_resid': float(mean_resid),
                'mean_abs_resid': float(mean_abs_resid),
            })

        baseline_records.sort(key=lambda item: item['mean_abs_resid'], reverse=True)
        antenna_records.sort(key=lambda item: item['mean_abs_resid'], reverse=True)
        pol_results[pol] = {
            'baseline_records': baseline_records,
            'antenna_records': antenna_records,
            'residual_spectrum_jy': resid_spec,
            'real_spectrum_jy': real_spec,
        }

    # — Stokes-V proxy: |RR − LL| bad-data detector —
    # For an unpolarized calibrator (e.g. 3C48), corrected_RR ≈ corrected_LL ≈ sky
    # model, so |Re(RR) − Re(LL)| ≈ 0.  Any antenna or baseline with a large mean
    # |RR−LL| is most likely affected by RFI or hardware issues.  This metric is
    # sky-model-independent and complementary to the per-pol residual approach.
    # Results are stored in pol_results['V'] and can be used as a detection metric
    # by setting OUTLIER_METRIC='V' (or including 'V' in a tuple of metrics).
    if 'RR' in stokes_labels and 'LL' in stokes_labels:
        rr_idx_v = stokes_labels.index('RR')
        ll_idx_v = stokes_labels.index('LL')
        rr_re_v = np.real(vis_corr[:, :, rr_idx_v])
        ll_re_v = np.real(vis_corr[:, :, ll_idx_v])
        rr_flag_v = flagged[:, :, rr_idx_v]
        ll_flag_v = flagged[:, :, ll_idx_v]
        rr_w_v = weights[:, :, rr_idx_v]
        ll_w_v = weights[:, :, ll_idx_v]
        v_signed = rr_re_v - ll_re_v         # Re(RR − LL) per row × channel — signed
        v_diff = np.abs(v_signed)              # |RR − LL| — used only for per-baseline ranking
        v_good = (
            (~rr_flag_v) & (~ll_flag_v)
            & np.isfinite(v_signed)
            & (rr_w_v > 0) & (ll_w_v > 0)
            & chan_mask[None, :]
        )
        v_w = np.minimum(rr_w_v, ll_w_v)  # conservative: use weaker weight

        # Per-channel coherent baseline-average of Re(RR−LL) — the Stokes-V
        # spectrum at the phase centre.  Signed average so noise cancels;
        # for an unpolarised calibrator this should sit near zero.
        v_num_spec = np.nansum(np.where(v_good, v_w * v_signed, 0.0), axis=0)
        v_den_spec = np.nansum(np.where(v_good, v_w, 0.0), axis=0)
        v_spec = np.full(freqs_hz.size, np.nan, dtype=np.float64)
        v_ok = v_den_spec > 0
        v_spec[v_ok] = v_num_spec[v_ok] / v_den_spec[v_ok]

        v_baseline_records: List[dict] = []
        for base_idx_v, (a1_v, a2_v) in enumerate(unique_pairs):
            row_sel_v = inv == base_idx_v
            if not np.any(row_sel_v):
                continue
            bg = v_good[row_sel_v]; bw = v_w[row_sel_v]; bv = v_diff[row_sel_v]
            den_b = np.nansum(np.where(bg, bw, 0.0))
            if den_b <= 0:
                continue
            mean_v = float(np.nansum(np.where(bg, bw * bv, 0.0)) / den_b)
            v_baseline_records.append({
                'pair': (int(a1_v), int(a2_v)),
                'label': f'{ant_name_map.get(int(a1_v), a1_v)}-{ant_name_map.get(int(a2_v), a2_v)}',
                'mean_resid': mean_v,
                'mean_abs_resid': mean_v,
            })

        v_antenna_records: List[dict] = []
        for ant_v in np.unique(np.concatenate([ant1, ant2])):
            row_sel_v = (ant1 == ant_v) | (ant2 == ant_v)
            if not np.any(row_sel_v):
                continue
            ag = v_good[row_sel_v]; aw = v_w[row_sel_v]; av = v_diff[row_sel_v]
            den_a = np.nansum(np.where(ag, aw, 0.0))
            if den_a <= 0:
                continue
            mean_v = float(np.nansum(np.where(ag, aw * av, 0.0)) / den_a)
            v_antenna_records.append({
                'ant': int(ant_v),
                'label': ant_name_map.get(int(ant_v), str(int(ant_v))),
                'mean_resid': mean_v,
                'mean_abs_resid': mean_v,
            })

        v_baseline_records.sort(key=lambda r: r['mean_abs_resid'], reverse=True)
        v_antenna_records.sort(key=lambda r: r['mean_abs_resid'], reverse=True)
        pol_results['V'] = {
            'baseline_records': v_baseline_records,
            'antenna_records': v_antenna_records,
            'residual_spectrum_jy': v_spec,      # coherent Re⟨RR−LL⟩ spectrum
            'real_spectrum_jy':     v_spec,
            'coherent_v_spectrum_jy': v_spec,    # alias used by C3 convergence
        }

    ax_spec.set_title('Corrected vector-averaged spectrum')
    ax_spec.set_ylabel('Flux Density (Jy)')
    ax_spec.grid(True, alpha=0.25)
    ax_spec.legend(fontsize=9, loc='best', framealpha=0.85)

    ax_resid.axhline(0.0, color='k', lw=0.8, ls='--', alpha=0.5)
    ax_clean.axhline(0.0, color='k', lw=0.8, ls='--', alpha=0.5)

    # ── Stokes-V: fit in residuals panel (light), cleaned trace in cleaned panel ──
    _v_fit_annot_lines = []
    if 'V' in pol_results:
        v_sp = pol_results['V']['coherent_v_spectrum_jy']
        # V data shown on residuals panel — thin dashed, visually subordinate
        ax_resid.plot(
            freqs_mhz, np.where(chan_mask, v_sp, np.nan),
            lw=0.8, ls='--', color='tab:purple', alpha=0.7, label='V',
        )
        _popt_v = _fit_sinusoid(freqs_mhz[chan_mask], v_sp[chan_mask])
        if _popt_v is not None:
            _a_v, _A_v, _P_v, _phi_v = _popt_v
            _sinu_v = _a_v + _A_v * np.sin(2.0 * np.pi * freqs_mhz / _P_v + _phi_v)
            ax_resid.plot(
                freqs_mhz, np.where(chan_mask, _sinu_v, np.nan),
                lw=1.4, ls=':', color='tab:purple', alpha=0.55,
            )
            _v_fit_annot_lines.append(f'V₁: {_A_v:.2f}Jy, {_P_v:.1f}MHz')
            _cleaned_v = v_sp - _sinu_v
            if _P_v > 4.0:
                _popt_v2 = _fit_sinusoid(
                    freqs_mhz[chan_mask], _cleaned_v[chan_mask],
                    period_range_mhz=(0.5, 4.0),
                )
                if _popt_v2 is not None:
                    _a_v2, _A_v2, _P_v2, _phi_v2 = _popt_v2
                    _sinu_v2 = _a_v2 + _A_v2 * np.sin(2.0 * np.pi * freqs_mhz / _P_v2 + _phi_v2)
                    ax_resid.plot(
                        freqs_mhz, np.where(chan_mask, _sinu_v2, np.nan),
                        lw=1.4, ls=(0, (3, 1, 1, 1)), color='tab:purple', alpha=0.45,
                    )
                    _v_fit_annot_lines.append(f'V₂: {_A_v2:.2f}Jy, {_P_v2:.1f}MHz')
                    _cleaned_v = _cleaned_v - _sinu_v2
            ax_clean.plot(
                freqs_mhz, np.where(chan_mask, _cleaned_v, np.nan),
                lw=0.8, ls='--', color='tab:purple', label='V',
            )
            _pol_cleaned['V'] = _cleaned_v
        else:
            ax_clean.plot(
                freqs_mhz, np.where(chan_mask, v_sp, np.nan),
                lw=0.8, ls='--', color='tab:purple', label='V (no fit)',
            )
            _pol_cleaned['V'] = v_sp

    # ── Fit summary text box inside residuals panel ────────────────────────────
    # Format each fit line with explicit labels for clarity, then place in a
    # light-green rounded box in the lower-left corner.  Y-axis limits are
    # expanded downward to guarantee the box never overlaps the data traces.
    _all_annot = getattr(ax_resid, '_fit_annot', []) + _v_fit_annot_lines

    if source_has_flux_model:
        _resid_title = 'Residuals  (data − model)'
        ax_resid.set_ylabel('Data − Model (Jy)')
    else:
        _resid_title = 'Re⟨RR−LL⟩'
        ax_resid.set_ylabel('Re⟨RR−LL⟩ (Jy)')
    ax_resid.set_title(_resid_title, fontsize=9)
    ax_resid.grid(True, alpha=0.25)
    ax_resid.legend(fontsize=8, loc='upper right', framealpha=0.85)

    if _all_annot:
        # Expand the lower y-limit by ~30% of the current data range so the
        # box sits in clear whitespace below the traces.
        _yr_lo, _yr_hi = ax_resid.get_ylim()
        _yr_span = _yr_hi - _yr_lo
        ax_resid.set_ylim(_yr_lo - 0.30 * _yr_span, _yr_hi)
        # Rebuild text with explicit "A=" and "P=" labels now that space is clear
        _box_lines = []
        for _s in _all_annot:
            # _s is e.g. "RR₁: 0.22Jy, 1.7MHz"  →  reformat to "RR₁  A=0.22 Jy  P=1.7 MHz"
            _pol_tag, _vals = (_s.split(': ', 1) + [''])[:2]
            _parts = [p.strip() for p in _vals.split(',')]
            _a_str = _parts[0] if len(_parts) > 0 else ''
            _p_str = _parts[1] if len(_parts) > 1 else ''
            _box_lines.append(f'{_pol_tag:<5s}  A={_a_str:<9s}  P={_p_str}')
        _box_text = '\n'.join(_box_lines)
        ax_resid.text(
            0.015, 0.04, _box_text,
            transform=ax_resid.transAxes,
            fontsize=6.5, va='bottom', ha='left', family='monospace',
            color='#1a3a1a',
            bbox=dict(
                boxstyle='round,pad=0.5',
                facecolor='#d4edda',    # light mint-green
                edgecolor='#7fba7f',    # medium green border
                alpha=0.82,
            ),
        )

    if source_has_flux_model:
        ax_clean.set_title('Cleaned residuals', fontsize=9)
        ax_clean.set_ylabel('Cleaned (Jy)')
    else:
        ax_clean.set_title('Cleaned Re⟨RR−LL⟩', fontsize=9)
        ax_clean.set_ylabel('Cleaned (Jy)')
    ax_clean.grid(True, alpha=0.25)
    ax_clean.legend(fontsize=8, loc='upper right', framealpha=0.85)

    # ── Per-metric bar charts ────────────────────────────────────────────────
    # One row per computed metric (RR, LL, V), each showing the top-N antennas
    # (left) and top-N baselines (right) ranked by that metric's score.
    # Rows for metrics that drove the actual flagging decisions are marked with ★.
    _metric_order = [m for m in ('RR', 'LL', 'V') if m in pol_results]
    _metric_colors  = {'RR': 'tab:blue',   'LL': 'tab:orange', 'V': 'tab:purple'}
    _metric_ylabels = {
        'RR': 'RR mean absolute residual',
        'LL': 'LL mean absolute residual',
        'V':  '|RR − LL| mean absolute (per-baseline outlier score)',
    }
    # Normalise ranking_metric to a frozenset of strings
    if ranking_metric is None:
        _flagging_metrics: set = set()
    elif isinstance(ranking_metric, str):
        _flagging_metrics = {ranking_metric}
    else:
        _flagging_metrics = set(ranking_metric)

    for _row_i, _m in enumerate(_metric_order, start=1):
        _ax_a = fig.add_subplot(_gs[_row_i, 0])
        _ax_b = fig.add_subplot(_gs[_row_i, 1])
        _flag_tag = '  \u2605 used for flagging' if _m in _flagging_metrics else ''
        _col = _metric_colors.get(_m, 'tab:blue')
        _ylabel = _metric_ylabels.get(_m, _m)

        # Antenna bar
        _rec_ant = pol_results[_m].get('antenna_records', [])[:top_n]
        _labels_a = [r['label'] for r in _rec_ant]
        _vals_a   = [r['mean_abs_resid'] for r in _rec_ant]
        _ypos_a   = np.arange(len(_labels_a))
        _ax_a.barh(_ypos_a, _vals_a, color=_col)
        _ax_a.set_yticks(_ypos_a)
        _ax_a.set_yticklabels(_labels_a)
        _ax_a.invert_yaxis()
        _ax_a.set_xlabel('Jy')
        _ax_a.set_title(f'Top antennas — {_ylabel}{_flag_tag}')
        _ax_a.grid(True, axis='x', alpha=0.3)

        # Baseline bar
        _rec_base = pol_results[_m].get('baseline_records', [])[:top_n]
        _labels_b = [r['label'] for r in _rec_base]
        _vals_b   = [r['mean_abs_resid'] for r in _rec_base]
        _ypos_b   = np.arange(len(_labels_b))
        _ax_b.barh(_ypos_b, _vals_b, color=_col)
        _ax_b.set_yticks(_ypos_b)
        _ax_b.set_yticklabels(_labels_b)
        _ax_b.invert_yaxis()
        _ax_b.set_xlabel('Jy')
        _ax_b.set_title(f'Top baselines — {_ylabel}{_flag_tag}')
        _ax_b.grid(True, axis='x', alpha=0.3)

    plotted_freqs = freqs_mhz[chan_mask]
    if plotted_freqs.size:
        plot_freq_min = float(plotted_freqs[0])
        plot_freq_max = float(plotted_freqs[-1])
        for ax in (ax_spec, ax_resid, ax_clean):
            ax.set_xlim(plot_freq_min, plot_freq_max)
    else:
        plot_freq_min = np.nan
        plot_freq_max = np.nan
    ax_resid.set_xlabel('Frequency (MHz)')
    ax_clean.set_xlabel('Frequency (MHz)')

    # Channel numbers as secondary top x-axis on both spectrum panels.
    _diag_chan_idxs = np.asarray(
        solution.get('chan_indices', []) if solution is not None else [],
        dtype=np.int32,
    )
    if _diag_chan_idxs.size >= 2:
        _f0, _f1 = float(freqs_mhz[0]), float(freqs_mhz[-1])
        _c0, _c1 = float(_diag_chan_idxs[0]), float(_diag_chan_idxs[-1])
        def _freq_to_chan(f, _f0=_f0, _f1=_f1, _c0=_c0, _c1=_c1):
            return _c0 + (f - _f0) * (_c1 - _c0) / (_f1 - _f0)
        def _chan_to_freq(c, _f0=_f0, _f1=_f1, _c0=_c0, _c1=_c1):
            return _f0 + (c - _c0) * (_f1 - _f0) / (_c1 - _c0)
        for _ax_d in (ax_spec, ax_resid, ax_clean):
            _sec_d = _ax_d.secondary_xaxis('top', functions=(_freq_to_chan, _chan_to_freq))
            _sec_d.set_xlabel('Channel', fontsize=8)
            _sec_d.tick_params(labelsize=7)

    fig.suptitle(title or 'Bandpass diagnostics', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.show(block=True)

    return {
        'vis': vis,
        'vis_after_flag_tables': vis_for_diag,
        'vis_used': vis_use,
        'corrected': corrected,
        'pol_results': pol_results,
        'freqs_hz': freqs_hz,
        'plot_freq_min_mhz': plot_freq_min,
        'plot_freq_max_mhz': plot_freq_max,
        'chan_mask': chan_mask,
        'figure': fig,
        'save_path': str(save_path) if save_path is not None else None,
        'diagnostics_flag_table_applied': bool(diagnostics_flag_stats.get('flag_table_count', 0) > 0),
        'diagnostics_flag_table_count': int(diagnostics_flag_stats.get('flag_table_count', 0)),
        'diagnostics_flag_table_paths': list(diagnostics_flag_stats.get('flag_table_paths', [])),
        'diagnostics_dropped_rows_by_flag_table': int(diagnostics_flag_stats.get('dropped_rows', 0)),
        'source': source,
        'source_has_flux_model': source_has_flux_model,
    }


def propose_flag_updates_from_diagnostics(
    diagnostics: dict,
    *,
    outlier_metric: Union[str, List[str]] = 'LL',
    mode: str = 'both',
    antenna_flag_threshold_jy: Union[float, Dict[str, float]] = 180.0,
    baseline_flag_threshold_jy: Union[float, Dict[str, float]] = 800.0,
    max_antennas_to_flag: int = 4,
    max_baselines_to_flag: int = 6,
    outlier_metric_merge_strategy: str = 'union',
) -> dict:
    """Convert diagnostic outliers into candidate flag-table updates.

    ``outlier_metric`` may be a single metric string (e.g. ``'LL'``) or a list
    (e.g. ``['RR', 'LL', 'V']``).  Each metric is evaluated independently;
    results are merged via ``outlier_metric_merge_strategy``:

    - ``'union'``        : flag if threshold is exceeded in ANY metric.  Items
                           are ranked by their *maximum* value across metrics.
                           Recommended for GMRT — bad in one corr ~ bad in all.
    - ``'intersection'`` : flag only if threshold is exceeded in ALL metrics.
                           Most conservative choice.
    - ``'max'``          : alias for ``'union'``.

    ``antenna_flag_threshold_jy`` and ``baseline_flag_threshold_jy`` accept either
    a single float (applied to all metrics) or a dict mapping metric name to threshold,
    with an optional ``'default'`` key as fallback for unlisted metrics.  Example::

        antenna_flag_threshold_jy = {'RR': 180.0, 'LL': 180.0, 'V': 30.0}
        baseline_flag_threshold_jy = {'RR': 800.0, 'LL': 800.0, 'V': 150.0}

    Thresholds are applied *per metric* before the merge strategy is evaluated —
    so a V score of 35 Jy can trigger a flag under 'union' even when the RR/LL
    residuals (measured against a higher threshold) are clean.

    Flag-table entries are polarization-agnostic (antenna/baseline names only),
    so any resulting flag is unconditionally applied to **all correlations** on
    the next bandpass solve regardless of which metric triggered it.
    """
    if mode not in ('antennas', 'baselines', 'both'):
        raise ValueError("mode must be 'antennas', 'baselines', or 'both'.")
    if outlier_metric_merge_strategy not in ('union', 'intersection', 'max'):
        raise ValueError("outlier_metric_merge_strategy must be 'union', 'intersection', or 'max'.")

    # Normalise outlier_metric to a list.
    pols: List[str] = [outlier_metric] if isinstance(outlier_metric, str) else list(outlier_metric)
    pol_results_all = diagnostics.get('pol_results', {})
    missing = [p for p in pols if p not in pol_results_all]
    if missing:
        # Give a targeted, actionable message when model-based metrics are requested
        # for a source that has no registered flux model.
        missing_model_based = [p for p in missing if p in _MODEL_BASED_METRICS]
        if missing_model_based:
            _supported = ', '.join(sorted(_FLUX_MODEL_REGISTRY))
            _src = diagnostics.get('source', '<unknown>')
            raise ValueError(
                f'Metric(s) {missing_model_based} require a Perley-Butler flux model, but '
                f'source "{_src}" is not in the code\'s flux model registry '
                f'(registered calibrators: {_supported}).  '
                f'Switch to outlier_metric="V" for the model-free Stokes-V proxy, '
                f'or add a flux model entry for "{_src}" to _FLUX_MODEL_REGISTRY.'
            )
        raise ValueError(f'Diagnostics do not contain metric(s): {missing}.')

    def _resolve_thresholds(param: Union[float, Dict[str, float]]) -> Dict[str, float]:
        """Return a per-metric threshold dict from either a scalar or a dict.

        Extra keys for inactive metrics are silently ignored, so a full dict
        covering all possible metrics (RR, LL, V) can be kept in the notebook
        without needing to be trimmed when switching active metrics.
        """
        if isinstance(param, dict):
            resolved = {}
            for m in pols:
                if m in param:
                    resolved[m] = float(param[m])
                elif 'default' in param:
                    resolved[m] = float(param['default'])
                else:
                    raise ValueError(
                        f'Metric "{m}" has no entry in the threshold dict and no '
                        f'"default" key is present.  Add "{m}" or a "default" key.'
                    )
            return resolved
        return {m: float(param) for m in pols}

    ant_thresholds  = _resolve_thresholds(antenna_flag_threshold_jy)
    base_thresholds = _resolve_thresholds(baseline_flag_threshold_jy)

    def _combine_records(key: str, thresholds: Dict[str, float]) -> List[dict]:
        """Filter by per-metric threshold, merge decisions, return ranked list.

        For each label, only metrics where the score exceeds that metric's own
        threshold count as "triggered".  The merge strategy then decides:
          union        → flag if triggered in ANY metric
          intersection → flag only if triggered in ALL metrics
        The ranking score is max (union) or min (intersection) across triggered metrics.
        """
        per_metric: Dict[str, Dict[str, float]] = {}  # label -> {metric: score}
        for p in pols:
            thresh = thresholds[p]
            for rec in pol_results_all[p][key]:
                if rec['mean_abs_resid'] >= thresh:
                    per_metric.setdefault(rec['label'], {})[p] = rec['mean_abs_resid']

        result = []
        for label, triggered in per_metric.items():
            if outlier_metric_merge_strategy == 'intersection':
                # Must be triggered in every metric.
                if len(triggered) < len(pols):
                    continue
                combined_resid = min(triggered.values())
            else:
                # 'union' / 'max': triggered in at least one metric.
                combined_resid = max(triggered.values())
            result.append({'label': label, 'mean_abs_resid': combined_resid})
        result.sort(key=lambda r: r['mean_abs_resid'], reverse=True)
        return result

    ant_records  = _combine_records('antenna_records',  ant_thresholds)
    base_records = _combine_records('baseline_records', base_thresholds)

    bad_antennas = []
    if mode in ('antennas', 'both'):
        bad_antennas = [rec['label'] for rec in ant_records][:int(max_antennas_to_flag)]

    bad_baselines = []
    if mode in ('baselines', 'both'):
        for rec in base_records:
            left, right = rec['label'].split('-', 1)
            bad_baselines.append([left, right])
            if len(bad_baselines) >= int(max_baselines_to_flag):
                break

    pol_desc = pols[0] if len(pols) == 1 else f"{'+'.join(pols)}({outlier_metric_merge_strategy})"
    proposal = create_flag_table(
        bad_antennas=bad_antennas,
        bad_baselines=bad_baselines,
        notes=(
            f'Proposed from {pol_desc} diagnostics: '
            f'antenna_flag_threshold_jy={antenna_flag_threshold_jy}, '
            f'baseline_flag_threshold_jy={baseline_flag_threshold_jy}, mode={mode}'
        ),
    )
    return {
        'proposal': proposal,
        'candidate_antennas': bad_antennas,
        'candidate_baselines': bad_baselines,
        'mode': mode,
        'outlier_metric': pols[0] if len(pols) == 1 else pols,
        'outlier_metric_merge_strategy': outlier_metric_merge_strategy if len(pols) > 1 else None,
    }


def update_flag_table(
    output_path: Union[str, Path],
    *,
    add_antennas: Optional[List[Union[str, int]]] = None,
    add_baselines: Optional[List[Union[List[Union[str, int]], Tuple[Union[str, int], Union[str, int]], Dict[str, Union[str, int]]]]] = None,
    base_flag_tables=None,
    base_flag_table_paths=None,
    notes: str = '',
    dry_run: bool = True,
) -> dict:
    """Merge new exclusions into a flag table, optionally writing to disk."""
    output_path = Path(output_path)
    tables, _ = _coerce_flag_tables(flag_tables=base_flag_tables, flag_table_paths=base_flag_table_paths)
    if output_path.exists():
        tables.insert(0, load_flag_table(output_path))

    merged_antennas = []
    merged_baselines = []
    seen_antennas = set()
    seen_baselines = set()
    merged_notes = []

    for table in tables + [create_flag_table(bad_antennas=add_antennas, bad_baselines=add_baselines, notes=notes)]:
        for ant in table.get('bad_antennas', []):
            key = str(ant)
            if key in seen_antennas:
                continue
            seen_antennas.add(key)
            merged_antennas.append(ant)
        for baseline in table.get('bad_baselines', []):
            if isinstance(baseline, dict):
                key = f"{baseline.get('ant1')}-{baseline.get('ant2')}"
            elif isinstance(baseline, (list, tuple)) and len(baseline) == 2:
                key = f'{baseline[0]}-{baseline[1]}'
            else:
                key = str(baseline)
            if key in seen_baselines:
                continue
            seen_baselines.add(key)
            merged_baselines.append(baseline)
        note = str(table.get('notes', '')).strip()
        if note:
            merged_notes.append(note)

    merged = create_flag_table(
        bad_antennas=merged_antennas,
        bad_baselines=merged_baselines,
        notes=' | '.join(merged_notes),
    )

    if not dry_run:
        save_flag_table(merged, output_path)

    return {
        'flag_table': merged,
        'output_path': str(output_path),
        'dry_run': bool(dry_run),
        'written': bool(not dry_run),
        'added_antennas': list(add_antennas or []),
        'added_baselines': list(add_baselines or []),
    }


def merge_flag_table_into_file(
    output_path: Union[str, Path],
    new_table: dict,
    *,
    notes: str = '',
    dry_run: bool = True,
) -> dict:
    """Merge *new_table* (ALL six flag sections) into the on-disk flag table.

    Unlike :func:`update_flag_table`, this function merges every section of
    the v2 flag schema:

    * ``bad_antennas``            — merged with deduplication by name
    * ``bad_baselines``           — merged with deduplication by sorted pair
    * ``bad_antenna_timeranges``  — dict of entries appended per antenna key
    * ``bad_baseline_timeranges`` — dict of entries appended per baseline key
    * ``bad_scan_timeranges``     — de-duplicated list of [t0, t1] pairs
    * ``bad_burst_timeranges``    — de-duplicated list of [t0, t1] pairs

    If *output_path* does not yet exist the new table is written directly.
    If *dry_run* is True the merged result is returned but NOT saved to disk.
    """
    output_path = Path(output_path)
    existing: dict = load_flag_table(output_path) if output_path.exists() else {}

    # ── Wholesale antennas: union with dedup by name ──────────────────────
    seen_ants: set = set()
    merged_ants: list = []
    for ant in list(existing.get('bad_antennas', [])) + list(new_table.get('bad_antennas', [])):
        key = str(ant).strip().upper()
        if key not in seen_ants:
            seen_ants.add(key)
            merged_ants.append(ant)

    # ── Wholesale baselines: union with dedup by sorted name pair ─────────
    seen_bls: set = set()
    merged_bls: list = []
    for bl in list(existing.get('bad_baselines', [])) + list(new_table.get('bad_baselines', [])):
        if isinstance(bl, (list, tuple)) and len(bl) == 2:
            key = tuple(sorted([str(bl[0]).strip().upper(), str(bl[1]).strip().upper()]))
        else:
            key = str(bl)
        if key not in seen_bls:
            seen_bls.add(key)
            merged_bls.append(bl)

    # ── Time-range dicts: merge per key; append new entries ───────────────
    def _merge_tr_dict(section_key: str) -> dict:
        old_d: dict = dict(existing.get(section_key, {}))
        new_d: dict = dict(new_table.get(section_key, {}))
        for coord_key, new_entries in new_d.items():
            if coord_key not in old_d:
                old_d[coord_key] = list(new_entries)
            else:
                # Append without structural dedup; duplicate intervals are
                # harmless — they just result in redundant but correct flagging.
                old_d[coord_key] = list(old_d[coord_key]) + list(new_entries)
        return old_d

    # ── Scan / burst timeranges: union by string key ──────────────────────
    def _merge_tr_list(section_key: str) -> list:
        old_l = [tuple(str(t) for t in x) for x in existing.get(section_key, [])]
        new_l = [tuple(str(t) for t in x) for x in new_table.get(section_key, [])]
        seen: set = set(old_l)
        merged: list = list(existing.get(section_key, []))
        for item, key in zip(new_table.get(section_key, []), new_l):
            if key not in seen:
                seen.add(key)
                merged.append(item)
        return merged

    # ── Assemble notes ────────────────────────────────────────────────────
    all_notes = ' | '.join(filter(None, [
        str(existing.get('notes', '')).strip(),
        str(new_table.get('notes', '')).strip(),
        notes,
    ]))

    merged = {
        'kind':    'ugmrt_flag_table',
        'version': 2,
        'source':  new_table.get('source', existing.get('source', '')),
        'notes':   all_notes,
        'bad_antennas':            merged_ants,
        'bad_baselines':           merged_bls,
        'bad_antenna_timeranges':  _merge_tr_dict('bad_antenna_timeranges'),
        'bad_baseline_timeranges': _merge_tr_dict('bad_baseline_timeranges'),
        'bad_scan_timeranges':     _merge_tr_list('bad_scan_timeranges'),
        'bad_burst_timeranges':    _merge_tr_list('bad_burst_timeranges'),
    }

    if not dry_run:
        save_flag_table(merged, output_path)

    return {
        'flag_table':  merged,
        'output_path': str(output_path),
        'dry_run':     bool(dry_run),
        'written':     bool(not dry_run),
    }


def run_iterative_bandpass_workflow(
    fits_path: Union[str, Path],
    *,
    index: Optional[dict] = None,
    index_cache_path: Optional[Union[str, Path]] = None,
    index_cache_dir: Optional[Union[str, Path]] = None,
    force_rebuild_index: bool = False,
    index_validation_mode: str = 'fast',
    write_index_cache: bool = True,
    bandpass_out_base: Optional[Union[str, Path]] = None,
    diag_plot_base: Optional[Union[str, Path]] = None,
    diag_plot_unflagged_base: Optional[Union[str, Path]] = None,
    gain_plot_base: Optional[Union[str, Path]] = None,
    flag_table_session_path: Optional[Union[str, Path]] = None,
    base_flag_table_paths: Optional[List[Union[str, Path]]] = None,
    pending_flag_tables: Optional[List[dict]] = None,
    use_pending_flag_tables: bool = True,
    start_iteration: int = 1,
    n_iterations: int = 1,
    iter_prefix: str = 'iter',
    iter_width: int = 2,
    dry_run_bandpass: bool = True,
    dry_run_flag_write: bool = True,
    source: str = '3C48',
    stokes: Tuple[str, ...] = ('RR', 'LL'),
    chan_range: Optional[Tuple[int, int]] = None,
    max_rows_solve: int = 150_000,
    smooth_window: int = 5,
    min_baselines: int = 20,
    ignore_autos: bool = True,
    max_rows_diag: int = 60_000,
    exclude_for_plots: Optional[List[Union[str, int]]] = None,
    diag_apply_flag_tables: bool = True,
    diag_save_unflagged_comparison: bool = False,
    flag_all_corrs_if_any_rawvis_flagged: bool = False,
    skip_edge_channels: Union[int, Tuple[int, int]] = (0, 0),
    top_n: int = 12,
    outlier_metric: Union[str, List[str]] = 'LL',
    outlier_metric_merge_strategy: str = 'union',
    proposal_mode: str = 'baselines',
    antenna_flag_threshold_jy: Union[float, Dict[str, float]] = 180.0,
    baseline_flag_threshold_jy: Union[float, Dict[str, float]] = 800.0,
    max_antennas_to_flag: int = 4,
    max_baselines_to_flag: int = 6,
    strict_flag_table: bool = False,
    convergence_epsilon: float = 0.0,
    convergence_min_iters: int = 3,
    compare_metrics_for_convergence: Optional[List[str]] = None,
    convergence_combine_strategy: str = 'any',
    run_iter0_diagnostic: bool = True,
    # ── DataSelection params (passed to both solve and diagnostics) ───────────
    selection: Optional['DataSelection'] = None,
    timerange: Optional[Tuple] = None,
    uvrange_m: Optional[Tuple[float, float]] = None,
    uvrange_klambda: Optional[Tuple[float, float]] = None,
    elevation_min_deg: Optional[float] = None,
    elevation_max_deg: Optional[float] = None,
) -> Dict[str, Any]:
    """Run iterative bandpass -> diagnostics -> flag-update workflow.

    Convergence stopping
    --------------------
    C1 (always active): stop when the cumulative flag set is identical to the
        previous iteration.  Another solve would produce the same result.

    C3 / C4 (epsilon-based, enabled when convergence_epsilon > 0):

      convergence_epsilon         - fractional change threshold: stop when
                                    |RMS(i-1) - RMS(i)| / RMS(i-1) < epsilon.
                                    0.0 disables all epsilon criteria.
      convergence_min_iters       - minimum iterations before epsilon criteria
                                    are evaluated (default 3).
      compare_metrics_for_convergence - list of metrics to track.  Supported:
                                    'V'     (C3) coherent Re⟨RR−LL⟩ rms
                                            at the phase centre.  Always
                                            available when RR+LL are present.
                                    'Model' (C4) RR and LL residual rms
                                            (data minus flux model).  Requires
                                            a PB2017 flux model for the source;
                                            silently skipped otherwise.
                                    Default: ['V', 'Model']
      convergence_combine_strategy - how to combine multiple active criteria:
                                    'any'  stop if ANY criterion fires
                                    'all'  stop only if ALL criteria fire
                                           in the same iteration

    run_iter0_diagnostic - if True (default), run a raw-visibility diagnostic
                           (no bandpass correction) before the first iteration,
                           tagged '{iter_prefix}00'.  Provides a baseline view of
                           uncorrected Stokes-V and RR/LL deviations from model.
                           Requires diag_plot_base to be set to produce a plot.

    gain_plot_base - if set, a per-antenna 6×5 amp/phase grid plot is saved
                    after each iteration (e.g. WORK_DIR/'gains.png' produces
                    'gains_iter01.png', 'gains_iter02.png', ...).
    """
    if n_iterations < 1:
        raise ValueError('n_iterations must be >= 1.')

    if index is None:
        index = get_or_build_row_index(
            fits_path,
            cache_path=index_cache_path,
            cache_dir=index_cache_dir,
            force_rebuild=force_rebuild_index,
            validation_mode=index_validation_mode,
            write_cache=write_index_cache,
        )

    session_path = Path(flag_table_session_path) if flag_table_session_path is not None else None
    base_paths = [Path(p) for p in (base_flag_table_paths or [])]

    # ── iter00: raw-visibility diagnostic (no bandpass correction) ──────────
    if run_iter0_diagnostic:
        _iter0_tag  = f'{iter_prefix}00'
        _iter0_plot = tagged_output_path(diag_plot_base, _iter0_tag) if diag_plot_base is not None else None
        print(f'[iter00] raw-visibility diagnostic (no correction){" → " + str(_iter0_plot) if _iter0_plot else ""}')
        run_bandpass_diagnostics(
            index,
            solution=None,
            source=source,
            chan_range=chan_range,
            stokes=stokes,
            max_rows=max_rows_diag,
            exclude_antennas=None,
            apply_flag_tables=diag_apply_flag_tables,
            flag_table_path=list(base_paths) if base_paths else None,
            flag_table=None,
            apply_correction=False,
            flag_all_corrs_if_any_rawvis_flagged=flag_all_corrs_if_any_rawvis_flagged,
            skip_edge_channels=skip_edge_channels,
            top_n=top_n,
            title=f'{source} RAW — no correction | {_iter0_tag}',
            save_path=_iter0_plot,
            selection=selection,
            timerange=timerange,
            uvrange_m=uvrange_m,
            uvrange_klambda=uvrange_klambda,
            elevation_min_deg=elevation_min_deg,
            elevation_max_deg=elevation_max_deg,
        )
    pending_tables = list(pending_flag_tables or [])
    history = []
    last_bandpass_run = None
    last_diag = None
    last_flag_update = None
    _compare_metrics  = list(compare_metrics_for_convergence) if compare_metrics_for_convergence is not None else ['V', 'Model']
    _prev_flag_key    = None   # C1 state
    _prev_v_rms       = None   # C3 state: coherent Stokes-V rms
    _prev_rr_rms      = None   # C4 state: RR residual rms
    _prev_ll_rms      = None   # C4 state: LL residual rms
    _warned_no_model  = False  # warn once if 'Model' requested but no flux model
    _stop_reason      = None   # set on early exit; None means ran to completion
    _canonical_ant_ids   = None  # fixed antenna order for gain plots (set from first solve)
    _canonical_ant_names = None

    for iteration_num in range(start_iteration, start_iteration + n_iterations):
        iter_tag = f'{iter_prefix}{iteration_num:0{int(iter_width)}d}'

        active_disk_flag_paths = list(base_paths)
        if session_path is not None and session_path.exists() and session_path not in active_disk_flag_paths:
            active_disk_flag_paths.append(session_path)

        active_pending = pending_tables if use_pending_flag_tables else []

        bandpass_run = derive_bandpass_iteration(
            fits_path=fits_path,
            index=index,
            bandpass_out=bandpass_out_base,
            source=source,
            stokes=stokes,
            chan_range=chan_range,
            max_rows=max_rows_solve,
            smooth_window=smooth_window,
            min_baselines=min_baselines,
            ignore_autos=ignore_autos,
            flag_table_path=active_disk_flag_paths if active_disk_flag_paths else None,
            flag_table=active_pending if active_pending else None,
            flag_all_corrs_if_any_rawvis_flagged=flag_all_corrs_if_any_rawvis_flagged,
            iteration_tag=iter_tag,
            dry_run=dry_run_bandpass,
            # DataSelection — forwarded through **solve_kwargs → derive_point_source_bandpass
            selection=selection,
            timerange=timerange,
            uvrange_m=uvrange_m,
            uvrange_klambda=uvrange_klambda,
            elevation_min_deg=elevation_min_deg,
            elevation_max_deg=elevation_max_deg,
        )
        bandpass_sol = bandpass_run['solution']

        # Capture canonical antenna list from the very first solve so all
        # subsequent gain-grid plots use the same fixed slot layout.
        if _canonical_ant_ids is None:
            _canonical_ant_ids   = [int(x) for x in bandpass_sol['antenna_ids']]
            _canonical_ant_names = list(
                bandpass_sol.get('antenna_names') or
                [f'Ant{aid}' for aid in _canonical_ant_ids]
            )

        # Per-iteration gain grid plot
        if gain_plot_base is not None:
            _gp = tagged_output_path(gain_plot_base, iter_tag)
            _gain_fig = plot_bandpass_solution_grid(
                bandpass_run['solution'],
                title=f'{source} bandpass gains | {iter_tag}',
                skip_edge_channels=skip_edge_channels,
                save_path=_gp,
                canonical_antenna_ids=_canonical_ant_ids,
                canonical_antenna_names=_canonical_ant_names,
            )
            import matplotlib.pyplot as _plt
            _plt.close(_gain_fig)  # prevent figure accumulation over many iterations

        diag_plot_path = tagged_output_path(diag_plot_base, iter_tag) if diag_plot_base is not None else None
        diag = run_bandpass_diagnostics(
            index,
            bandpass_sol,
            source=source,
            chan_range=chan_range,
            stokes=stokes,
            max_rows=max_rows_diag,
            exclude_antennas=exclude_for_plots or [],
            apply_flag_tables=diag_apply_flag_tables,
            flag_table_path=active_disk_flag_paths if active_disk_flag_paths else None,
            flag_table=active_pending if active_pending else None,
            strict_flag_table=strict_flag_table,
            flag_all_corrs_if_any_rawvis_flagged=flag_all_corrs_if_any_rawvis_flagged,
            skip_edge_channels=skip_edge_channels,
            top_n=top_n,
            ranking_metric=outlier_metric,
            title=f'{source} diagnostics | {iter_tag}',
            save_path=diag_plot_path,
            selection=selection,
            timerange=timerange,
            uvrange_m=uvrange_m,
            uvrange_klambda=uvrange_klambda,
            elevation_min_deg=elevation_min_deg,
            elevation_max_deg=elevation_max_deg,
        )

        unflagged_plot_path = None
        if diag_save_unflagged_comparison and diag_apply_flag_tables:
            unflagged_base = diag_plot_unflagged_base if diag_plot_unflagged_base is not None else diag_plot_base
            if unflagged_base is not None:
                unflagged_plot_path = tagged_output_path(unflagged_base, iter_tag)
                run_bandpass_diagnostics(
                    index,
                    bandpass_sol,
                    source=source,
                    chan_range=chan_range,
                    stokes=stokes,
                    max_rows=max_rows_diag,
                    exclude_antennas=exclude_for_plots or [],
                    apply_flag_tables=False,
                    flag_all_corrs_if_any_rawvis_flagged=flag_all_corrs_if_any_rawvis_flagged,
                    skip_edge_channels=skip_edge_channels,
                    top_n=top_n,
                    title=f'{source} diagnostics (unflagged) | {iter_tag}',
                    save_path=unflagged_plot_path,
                    selection=selection,
                    timerange=timerange,
                    uvrange_m=uvrange_m,
                    uvrange_klambda=uvrange_klambda,
                    elevation_min_deg=elevation_min_deg,
                    elevation_max_deg=elevation_max_deg,
                )

        proposal = propose_flag_updates_from_diagnostics(
            diag,
            outlier_metric=outlier_metric,
            mode=proposal_mode,
            antenna_flag_threshold_jy=antenna_flag_threshold_jy,
            baseline_flag_threshold_jy=baseline_flag_threshold_jy,
            max_antennas_to_flag=max_antennas_to_flag,
            max_baselines_to_flag=max_baselines_to_flag,
            outlier_metric_merge_strategy=outlier_metric_merge_strategy,
        )

        if session_path is not None:
            flag_update = update_flag_table(
                output_path=session_path,
                add_antennas=proposal['proposal']['bad_antennas'],
                add_baselines=proposal['proposal']['bad_baselines'],
                base_flag_tables=active_pending if active_pending else None,
                base_flag_table_paths=active_disk_flag_paths if active_disk_flag_paths else None,
                notes=f'Auto-proposed from diagnostics {iter_tag}',
                dry_run=dry_run_flag_write,
            )
            pending_tables = [flag_update['flag_table']] if dry_run_flag_write else []
        else:
            flag_update = {
                'flag_table': create_flag_table(
                    bad_antennas=proposal['proposal']['bad_antennas'],
                    bad_baselines=proposal['proposal']['bad_baselines'],
                    notes=f'Auto-proposed from diagnostics {iter_tag}',
                ),
                'output_path': None,
                'dry_run': True,
                'written': False,
                'added_antennas': list(proposal['proposal']['bad_antennas']),
                'added_baselines': list(proposal['proposal']['bad_baselines']),
            }
            pending_tables = [flag_update['flag_table']]

        cumulative_table = flag_update.get('flag_table', {})
        history.append({
            'iteration_tag': iter_tag,
            'bandpass_output_path': bandpass_run.get('bandpass_out'),
            'diagnostics_plot_path': str(diag_plot_path) if diag_plot_path is not None else None,
            'diagnostics_unflagged_plot_path': str(unflagged_plot_path) if unflagged_plot_path is not None else None,
            'diagnostics_flag_table_applied': bool(diag.get('diagnostics_flag_table_applied', False)),
            'diagnostics_dropped_rows_by_flag_table': int(diag.get('diagnostics_dropped_rows_by_flag_table', 0)),
            'candidate_antennas': list(proposal.get('candidate_antennas', [])),
            'candidate_baselines': list(proposal.get('candidate_baselines', [])),
            'added_antennas': list(flag_update.get('added_antennas', [])),
            'added_baselines': list(flag_update.get('added_baselines', [])),
            'cumulative_bad_antenna_count': len(cumulative_table.get('bad_antennas', [])),
            'cumulative_bad_baseline_count': len(cumulative_table.get('bad_baselines', [])),
        })

        # ── Convergence checks ───────────────────────────────────────────────
        _n_done = iteration_num - start_iteration + 1

        # C1: flag set unchanged → the next solve would be identical, stop now.
        _cum = flag_update.get('flag_table', {})
        _curr_ants  = frozenset(_cum.get('bad_antennas', []))
        _curr_bases = frozenset(tuple(sorted(b)) for b in _cum.get('bad_baselines', []))
        _curr_flag_key = (_curr_ants, _curr_bases)
        _c1_converged = (_prev_flag_key is not None and _curr_flag_key == _prev_flag_key)
        _prev_flag_key = _curr_flag_key

        # ── C3 / C4: epsilon-based convergence ──────────────────────────────
        # _converged_criteria maps criterion name → human-readable detail string.
        # _active_criteria tracks which metrics had data this iteration.
        _converged_criteria: dict = {}
        _active_criteria:    set  = set()

        if convergence_epsilon > 0 and _n_done >= convergence_min_iters:

            # C3: coherent Stokes-V rms at phase centre
            if 'V' in _compare_metrics:
                _v_spec = diag.get('pol_results', {}).get('V', {}).get('coherent_v_spectrum_jy')
                if _v_spec is not None:
                    _finite_v = _v_spec[np.isfinite(_v_spec)]
                    if _finite_v.size > 0:
                        _curr_v_rms = float(np.sqrt(np.mean(_finite_v ** 2)))
                        _active_criteria.add('C3')
                        if _prev_v_rms is not None and _prev_v_rms > 0:
                            _v_frac = abs(_prev_v_rms - _curr_v_rms) / _prev_v_rms
                            if _v_frac < convergence_epsilon:
                                _converged_criteria['C3'] = (
                                    f'|ΔV_rms|/V_prev={_v_frac * 100:.2f}% '
                                    f'< {convergence_epsilon * 100:.1f}%'
                                )
                                print(f'[convergence] C3: {_converged_criteria["C3"]} after {iter_tag}')
                        _prev_v_rms = _curr_v_rms

            # C4: RR and LL model residual rms (requires flux model for source)
            if 'Model' in _compare_metrics:
                _pol_res = diag.get('pol_results', {})
                _rr_spec = _pol_res.get('RR', {}).get('residual_spectrum_jy')
                _ll_spec = _pol_res.get('LL', {}).get('residual_spectrum_jy')
                if _rr_spec is None and _ll_spec is None:
                    if not _warned_no_model:
                        print(
                            f'[convergence] WARNING: "Model" in compare_metrics but no flux '
                            f'model available for source "{source}" — C4 disabled.'
                        )
                        _warned_no_model = True
                else:
                    _active_criteria.add('C4')
                    _c4_rr_conv = False
                    _c4_ll_conv = False
                    _c4_details: list = []
                    if _rr_spec is not None:
                        _finite_rr = _rr_spec[np.isfinite(_rr_spec)]
                        if _finite_rr.size > 0:
                            _curr_rr_rms = float(np.sqrt(np.mean(_finite_rr ** 2)))
                            if _prev_rr_rms is not None and _prev_rr_rms > 0:
                                _rr_frac = abs(_prev_rr_rms - _curr_rr_rms) / _prev_rr_rms
                                if _rr_frac < convergence_epsilon:
                                    _c4_rr_conv = True
                                    _c4_details.append(f'RR:Δrms={_rr_frac*100:.2f}%')
                                    print(f'[convergence] C4-RR: |ΔRR_rms|/RR_prev={_rr_frac*100:.2f}% < {convergence_epsilon*100:.1f}% after {iter_tag}')
                            _prev_rr_rms = _curr_rr_rms
                    if _ll_spec is not None:
                        _finite_ll = _ll_spec[np.isfinite(_ll_spec)]
                        if _finite_ll.size > 0:
                            _curr_ll_rms = float(np.sqrt(np.mean(_finite_ll ** 2)))
                            if _prev_ll_rms is not None and _prev_ll_rms > 0:
                                _ll_frac = abs(_prev_ll_rms - _curr_ll_rms) / _prev_ll_rms
                                if _ll_frac < convergence_epsilon:
                                    _c4_ll_conv = True
                                    _c4_details.append(f'LL:Δrms={_ll_frac*100:.2f}%')
                                    print(f'[convergence] C4-LL: |ΔLL_rms|/LL_prev={_ll_frac*100:.2f}% < {convergence_epsilon*100:.1f}% after {iter_tag}')
                            _prev_ll_rms = _curr_ll_rms
                    # C4 fires only when BOTH available pols have converged
                    _pols_avail = int(_rr_spec is not None) + int(_ll_spec is not None)
                    _pols_conv  = int(_c4_rr_conv)        + int(_c4_ll_conv)
                    if _pols_avail > 0 and _pols_conv == _pols_avail:
                        _converged_criteria['C4'] = (
                            ', '.join(_c4_details) +
                            f' — both pols < {convergence_epsilon * 100:.1f}%'
                        )

        # Apply combine strategy across active criteria
        if _active_criteria:
            if convergence_combine_strategy == 'any':
                _epsilon_converged = bool(_converged_criteria)
            else:  # 'all'
                _epsilon_converged = (set(_converged_criteria.keys()) == _active_criteria)
        else:
            _epsilon_converged = False

        last_bandpass_run = bandpass_run
        last_diag = diag
        last_flag_update = flag_update

        if _c1_converged:
            _stop_reason = (
                f'C1 — flag set identical to previous iteration ({iter_tag}): '
                f'another solve would produce the same result.'
            )
            break
        if _epsilon_converged:
            _fired = sorted(_converged_criteria.keys())
            _details = '; '.join(f'{k}: {_converged_criteria[k]}' for k in _fired)
            _stop_reason = (
                f'Epsilon convergence (strategy={convergence_combine_strategy}, '
                f'epsilon={convergence_epsilon * 100:.1f}%) after {iter_tag} — '
                f'{_details}.'
            )
            break

    # ── Final stop-reason banner (always printed) ────────────────────────────
    _n_ran = len(history)
    if _stop_reason is None:
        if _n_ran >= n_iterations:
            _stop_reason = (
                f'Completed all {n_iterations} requested iteration(s) '
                f'without early convergence.'
            )
        else:
            _stop_reason = f'Loop ended after {_n_ran} iteration(s) (undetermined reason).'
    return {
        'index': index,
        'history': history,
        'pending_flag_tables': pending_tables,
        'last_bandpass_run': last_bandpass_run,
        'last_diagnostics': last_diag,
        'last_flag_update': last_flag_update,
        'stop_reason': _stop_reason,
    }


def apply_bandpass_solution(vis: dict, solution: dict) -> dict:
    """Apply bandpass gains in memory and return corrected visibilities.

    This function never writes back to FITS files or modifies the original
    visibility data on disk.
    """
    if 'vis_complex' not in vis:
        raise ValueError('vis must contain vis_complex. Re-load using load_vis_for_source().')

    vis_freqs = np.asarray(vis['freqs_hz'], dtype=np.float64)
    sol_freqs = np.asarray(solution['freqs_hz'], dtype=np.float64)

    # Support applying full-band solutions to channel subsets by frequency matching.
    if vis_freqs.shape == sol_freqs.shape and np.allclose(vis_freqs, sol_freqs, rtol=0.0, atol=1e-6):
        sol_chan_idx = np.arange(sol_freqs.size, dtype=np.int32)
    else:
        sol_chan_idx = np.full(vis_freqs.size, -1, dtype=np.int32)
        for i, vf in enumerate(vis_freqs):
            match = np.where(np.isclose(sol_freqs, vf, rtol=0.0, atol=1e-3))[0]
            if match.size == 1:
                sol_chan_idx[i] = int(match[0])
            elif match.size > 1:
                raise ValueError(
                    f'Ambiguous solution channel match for visibility frequency {vf:.6f} Hz.'
                )

        if np.any(sol_chan_idx < 0):
            missing = vis_freqs[sol_chan_idx < 0]
            raise ValueError(
                'Visibility frequencies are not fully covered by the bandpass solution. '
                f'First missing frequency: {missing[0]:.6f} Hz.'
            )

    vis_labels = list(vis['stokes_labels'])
    sol_labels = list(solution['stokes_labels'])
    common_labels = [label for label in vis_labels if label in sol_labels]
    if not common_labels:
        raise ValueError('No common Stokes labels between visibilities and bandpass solution.')

    antenna_ids = np.asarray(solution['antenna_ids'], dtype=np.int32)
    ant_to_idx = {int(ant): idx for idx, ant in enumerate(antenna_ids)}
    ant1_idx = np.array([ant_to_idx.get(int(ant), -1) for ant in vis['ant1']], dtype=np.int32)
    ant2_idx = np.array([ant_to_idx.get(int(ant), -1) for ant in vis['ant2']], dtype=np.int32)
    _missing_bl = (ant1_idx < 0) | (ant2_idx < 0)
    if np.any(_missing_bl):
        _missing_nos = sorted(set(
            [int(a) for a in vis['ant1'][ant1_idx < 0]] +
            [int(a) for a in vis['ant2'][ant2_idx < 0]]
        ))
        import warnings as _warnings
        _warnings.warn(
            f'apply_bandpass_solution: {int(np.sum(_missing_bl))} baseline row(s) reference '
            f'antenna NOSTA {_missing_nos} which are absent from the bandpass solution '
            f'(likely newly flagged in a later iteration). '
            f'Those baselines will be flagged NaN in the corrected output.',
            stacklevel=2,
        )
    # Clamp -1 sentinel to 0 so numpy indexing below is safe; affected rows are
    # flagged after the correction loop regardless of what gain was applied.
    _safe_ant1 = np.where(ant1_idx >= 0, ant1_idx, 0)
    _safe_ant2 = np.where(ant2_idx >= 0, ant2_idx, 0)

    corrected = np.array(vis['vis_complex'], dtype=np.complex128, copy=True)
    corrected_flagged = np.array(vis.get('flagged', np.zeros_like(vis['weight'], dtype=bool)), copy=True)

    sol_gains = np.asarray(solution['gains'], dtype=np.complex128)[sol_chan_idx, :, :]
    sol_valid = np.asarray(solution['valid'], dtype=bool)[sol_chan_idx, :, :]

    for vis_pol_idx, label in enumerate(vis_labels):
        if label not in common_labels:
            continue
        sol_pol_idx = sol_labels.index(label)
        g1 = sol_gains[:, _safe_ant1, sol_pol_idx].T
        g2 = sol_gains[:, _safe_ant2, sol_pol_idx].T
        valid = sol_valid[:, _safe_ant1, sol_pol_idx].T & sol_valid[:, _safe_ant2, sol_pol_idx].T
        denom = g1 * np.conj(g2)
        good = valid & np.isfinite(denom.real) & np.isfinite(denom.imag) & (np.abs(denom) > 0)
        corrected[:, :, vis_pol_idx][good] = corrected[:, :, vis_pol_idx][good] / denom[good]
        corrected[:, :, vis_pol_idx][~good] = np.nan + 1j * np.nan
        corrected_flagged[:, :, vis_pol_idx] |= ~good

    # Flag any baseline rows whose antennas were absent from the solution
    if np.any(_missing_bl):
        corrected[_missing_bl] = np.nan + 1j * np.nan
        corrected_flagged[_missing_bl] = True

    amp = np.abs(corrected)
    phase = np.degrees(np.angle(corrected)).astype(np.float32)
    amp[corrected_flagged] = np.nan
    phase[corrected_flagged] = np.nan
    # Propagate all flagged cells (pre-existing weight<=0 / exact-zero flags AND
    # gain-invalid cells) through to vis_complex_corrected.  Without this,
    # pre-flagged cells that passed through the gain division unmodified would
    # remain as 0+0j (or the raw value) instead of NaN, causing compute_stokes_vis
    # to produce identically-zero amplitudes in Stokes-V plots.
    corrected[corrected_flagged] = np.nan + 1j * np.nan

    out = dict(vis)
    out['vis_complex_corrected'] = corrected
    out['amp_corrected'] = amp
    out['phase_deg_corrected'] = phase
    out['flagged_corrected'] = corrected_flagged
    out['applied_bandpass_reference_antenna'] = int(solution['reference_antenna'])
    out['applied_bandpass_source'] = solution['source_name']
    return out


def compute_stokes_vis(
    vis: dict,
    solution: dict,
    output_stokes: str = 'V',
    signed: bool = False,
) -> dict:
    """Apply bandpass and return a vis-like dict for a derived Stokes parameter.

    Supported Stokes codes and the correlations they require:

    Circular basis (RR, LL, RL, LR)
      'I'  →  (RR + LL) / 2              requires RR, LL
      'V'  →  (RR − LL) / 2              requires RR, LL
      'Q'  →  (RL + LR) / 2              requires RL, LR
      'U'  →  i·(RL − LR) / 2            requires RL, LR

    Linear basis (XX, YY, XY, YX)
      'I'  →  (XX + YY) / 2              requires XX, YY
      'Q'  →  (XX − YY) / 2              requires XX, YY
      'U'  →  (XY + YX) / 2              requires XY, YX
      'V'  →  i·(YX − XY) / 2            requires XY, YX

    A ``ValueError`` is raised with an informative message if the requested
    Stokes parameter cannot be formed from the correlations present in *vis*.

    Parameters
    ----------
    vis : dict
        Raw visibility dict as returned by ``load_vis_for_source``.
    solution : dict
        Bandpass solution dict as returned by ``derive_bandpass_iteration``.
    output_stokes : str
        One of 'I', 'Q', 'U', 'V'.
    signed : bool
        If *True*, the ``amp`` field in the returned dict contains
        Re(out_complex) rather than |out_complex|.  Useful for Stokes-V plots
        where the sign of (RR−LL)/2 carries physical meaning (positive =
        right-hand excess, negative = left-hand excess) and RFI appears as
        large *absolute* deviations from zero.  Detection is always performed
        on |V|, so this only affects plotting.  Default *False*.

    Returns
    -------
    dict
        A vis-like dict with ``amp``, ``phase_deg``, ``vis_complex``, and
        ``stokes_labels = [output_stokes]``, suitable for direct use with
        ``plot_vis_amp_vs_uvdist`` and related plot functions.
    """
    VALID = {'I', 'Q', 'U', 'V'}
    if output_stokes not in VALID:
        raise ValueError(f'output_stokes={output_stokes!r} not one of {sorted(VALID)}')

    corr       = apply_bandpass_solution(vis, solution)
    vc         = corr['vis_complex_corrected']   # (rows, chans, pols)
    labels     = list(vis['stokes_labels'])

    def _get(pol: str):
        """Return corrected-vis slice for *pol*, or None if unavailable."""
        return vc[:, :, labels.index(pol)] if pol in labels else None

    circ_present = {p for p in ('RR', 'LL', 'RL', 'LR') if p in labels}
    lin_present  = {p for p in ('XX', 'YY', 'XY', 'YX') if p in labels}

    if circ_present:
        # ── circular basis ────────────────────────────────────────────────────
        RR, LL = _get('RR'), _get('LL')
        RL, LR = _get('RL'), _get('LR')
        if output_stokes == 'I':
            if RR is None or LL is None:
                raise ValueError(
                    f'Stokes I (circular basis) requires RR+LL; '
                    f'available correlations: {labels}')
            out_cplx = (RR + LL) / 2.0
        elif output_stokes == 'V':
            if RR is None or LL is None:
                raise ValueError(
                    f'Stokes V (circular basis) requires RR+LL; '
                    f'available correlations: {labels}')
            out_cplx = (RR - LL) / 2.0
        elif output_stokes == 'Q':
            if RL is None or LR is None:
                raise ValueError(
                    f'Stokes Q (circular basis) requires RL+LR; '
                    f'available correlations: {labels}  '
                    f'(dual-circ data with only RR+LL cannot produce Q or U)')
            out_cplx = (RL + LR) / 2.0
        else:  # 'U'
            if RL is None or LR is None:
                raise ValueError(
                    f'Stokes U (circular basis) requires RL+LR; '
                    f'available correlations: {labels}  '
                    f'(dual-circ data with only RR+LL cannot produce Q or U)')
            out_cplx = 1j * (RL - LR) / 2.0

    elif lin_present:
        # ── linear basis ──────────────────────────────────────────────────────
        XX, YY = _get('XX'), _get('YY')
        XY, YX = _get('XY'), _get('YX')
        if output_stokes == 'I':
            if XX is None or YY is None:
                raise ValueError(
                    f'Stokes I (linear basis) requires XX+YY; '
                    f'available correlations: {labels}')
            out_cplx = (XX + YY) / 2.0
        elif output_stokes == 'Q':
            if XX is None or YY is None:
                raise ValueError(
                    f'Stokes Q (linear basis) requires XX+YY; '
                    f'available correlations: {labels}')
            out_cplx = (XX - YY) / 2.0
        elif output_stokes == 'U':
            if XY is None or YX is None:
                raise ValueError(
                    f'Stokes U (linear basis) requires XY+YX; '
                    f'available correlations: {labels}  '
                    f'(dual-lin data with only XX+YY cannot produce U or V)')
            out_cplx = (XY + YX) / 2.0
        else:  # 'V'
            if XY is None or YX is None:
                raise ValueError(
                    f'Stokes V (linear basis) requires XY+YX; '
                    f'available correlations: {labels}  '
                    f'(dual-lin data with only XX+YY cannot produce U or V)')
            out_cplx = 1j * (YX - XY) / 2.0

    else:
        raise ValueError(
            f'Cannot determine polarisation basis from correlations {labels}; '
            f'expected circular (RR/LL/RL/LR) or linear (XX/YY/XY/YX).')

    out_amp   = (out_cplx.real if signed else np.abs(out_cplx))[:, :, np.newaxis].astype(np.float32)
    out_phase = np.degrees(np.angle(out_cplx))[:, :, np.newaxis].astype(np.float32)

    return {**vis,
            'amp':           out_amp,
            'phase_deg':     out_phase,
            'vis_complex':   out_cplx[:, :, np.newaxis].astype(np.complex64),
            'stokes_labels': [output_stokes]}


def _filter_vis_excluded_antennas(vis: dict, solution: dict, exclude_antennas=None) -> dict:
    """Return a visibility dict with baselines touching excluded antennas removed."""
    if not exclude_antennas:
        return vis

    ant_ids = np.asarray(solution.get('antenna_ids', []), dtype=np.int32)
    ant_names = solution.get('antenna_names') or [f'Ant{int(a)}' for a in ant_ids]

    def _norm_name(name):
        # ANNAME values can carry a suffix like "C11:11"; match on base tag.
        base = str(name).strip().upper().split(':', 1)[0]
        return ''.join(ch for ch in base if ch.isalnum())

    name_to_id = {str(name).strip().upper(): int(ant) for name, ant in zip(ant_names, ant_ids)}
    norm_to_ids = {}
    for name, ant in zip(ant_names, ant_ids):
        norm = _norm_name(name)
        norm_to_ids.setdefault(norm, []).append(int(ant))

    excluded_ids = set()
    ant_id_set = set(int(a) for a in ant_ids.tolist())

    for item in exclude_antennas:
        if isinstance(item, (int, np.integer)):
            excluded_ids.add(int(item))
            continue
        key = str(item).strip().upper()
        if key in name_to_id:
            excluded_ids.add(name_to_id[key])
            continue

        norm_key = _norm_name(key)
        norm_matches = norm_to_ids.get(norm_key, [])
        if len(norm_matches) == 1:
            excluded_ids.add(norm_matches[0])
            continue
        if len(norm_matches) > 1:
            print(f'[plot] warning: excluded antenna "{item}" is ambiguous among IDs {norm_matches}; ignored')
            continue

        try:
            ant_num = int(key)
            if ant_num in ant_id_set:
                excluded_ids.add(ant_num)
            else:
                print(f'[plot] warning: excluded antenna ID {ant_num} not present; ignored')
        except ValueError:
            print(f'[plot] warning: excluded antenna "{item}" not recognized; ignored')

    if not excluded_ids:
        return vis

    nrows = int(vis.get('nrows', len(vis.get('ant1', []))))
    row_mask = (~np.isin(vis['ant1'], list(excluded_ids))) & (~np.isin(vis['ant2'], list(excluded_ids)))
    kept = int(np.count_nonzero(row_mask))
    dropped = int(row_mask.size - kept)

    if kept == 0:
        raise ValueError('All rows were removed after applying excluded antennas filter.')

    vis_use = {}
    for key, value in vis.items():
        if isinstance(value, np.ndarray) and value.shape[:1] == (nrows,):
            vis_use[key] = value[row_mask]
        else:
            vis_use[key] = value
    vis_use['nrows'] = kept
    print(f'[plot] Excluded antennas {sorted(excluded_ids)}: dropped {dropped:,} rows, kept {kept:,}')
    return vis_use


def plot_bandpass_solution_grid(
    solution: dict,
    rows: int = 6,
    cols: int = 5,
    figsize=(32, 40),
    phase_ylim=(-200.0, 200.0),
    amp_ylim=None,
    skip_edge_channels: Union[int, Tuple[int, int]] = (5, 5),
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    canonical_antenna_ids: Optional[List[int]] = None,
    canonical_antenna_names: Optional[List[str]] = None,
):
    """Plot per-antenna bandpass on a rows x cols page using nested GridSpec.

    Each antenna cell contains two independent stacked subplots:
    - Top:    amplitude (0 to global 99th-percentile max)
    - Bottom: phase     (-200 to 200 degrees)

    Amplitude panel: channel numbers on top x-axis, no bottom tick labels.
    Phase panel:     frequency in MHz on bottom x-axis.

    skip_edge_channels controls exclusion from plotting and auto y-scaling:
    - int k      -> skip first k and last k channels
    - (a, b)     -> skip first a and last b channels

    canonical_antenna_ids / canonical_antenna_names:
        If provided, the grid is built from this fixed ordered list so the
        antenna slot layout is identical across all iterations.  Antennas
        present in the canonical list but absent from the solution are rendered
        as grey "NOT IN SOLUTION" panels.  Antennas present but fully-flagged
        are rendered as grey "FULLY FLAGGED" panels.
    """
    import matplotlib.gridspec as gridspec

    freqs_mhz = np.asarray(solution['freqs_hz'], dtype=np.float64) / 1e6
    chan_indices = np.asarray(solution['chan_indices'], dtype=np.int32)
    antenna_ids = np.asarray(solution['antenna_ids'], dtype=np.int32)
    antenna_names = solution.get('antenna_names') or [f'Ant{int(ant)}' for ant in antenna_ids]
    gains = np.asarray(solution['gains'], dtype=np.complex128)
    valid = np.asarray(solution['valid'], dtype=bool)
    stokes_labels = list(solution['stokes_labels'])

    # Build a lookup from antenna_id -> index within solution arrays
    _sol_id_to_idx = {int(aid): i for i, aid in enumerate(antenna_ids)}

    # Canonical ordered list of antennas to plot (fixed across iterations)
    if canonical_antenna_ids is not None:
        _plot_ids   = [int(x) for x in canonical_antenna_ids]
        _plot_names = list(canonical_antenna_names) if canonical_antenna_names is not None \
                      else [f'Ant{aid}' for aid in _plot_ids]
    else:
        _plot_ids   = [int(x) for x in antenna_ids]
        _plot_names = list(antenna_names)

    # Skip edge channels in plotting and auto-scaling to avoid edge artifacts.
    if isinstance(skip_edge_channels, tuple):
        if len(skip_edge_channels) != 2:
            raise ValueError('skip_edge_channels tuple must have exactly two values: (start, end).')
        skip_start = int(skip_edge_channels[0])
        skip_end = int(skip_edge_channels[1])
    else:
        skip_start = int(skip_edge_channels)
        skip_end = int(skip_edge_channels)

    if skip_start < 0 or skip_end < 0:
        raise ValueError('skip_edge_channels values must be non-negative.')

    plot_mask = np.ones(len(chan_indices), dtype=bool)
    if plot_mask.size > 0 and skip_start > 0:
        plot_mask[:min(skip_start, plot_mask.size)] = False
    if plot_mask.size > 0 and skip_end > 0:
        plot_mask[max(0, plot_mask.size - skip_end):] = False

    # Global amplitude range
    if amp_ylim is None:
        valid_for_scale = valid.copy()
        valid_for_scale[~plot_mask, :, :] = False
        amp_vals = np.abs(gains[valid_for_scale])
        if amp_vals.size == 0:
            amp_vals = np.abs(gains[valid])
        amp_hi = float(np.nanpercentile(amp_vals, 99)) if amp_vals.size else 2.0
        amp_ylim = (0.0, max(1.2, amp_hi * 1.1))

    # Channel tick marks (shared across all panels)
    chan_tick_idx = np.linspace(0, len(chan_indices) - 1, min(5, len(chan_indices)), dtype=int)
    chan_tick_positions = freqs_mhz[chan_tick_idx]
    chan_tick_labels = [str(int(chan_indices[i])) for i in chan_tick_idx]

    colors = ['C0', 'C1', 'C2', 'C3']

    # Outer grid: rows x cols antenna cells
    fig = plt.figure(figsize=figsize)
    outer = gridspec.GridSpec(
        rows, cols,
        figure=fig,
        hspace=0.55,   # vertical space between antenna rows
        wspace=0.35,   # horizontal space between antenna columns
        top=0.94, bottom=0.03, left=0.06, right=0.98,
    )

    legend_handles = []
    legend_labels = []

    for plot_slot, (ant_id, ant_name) in enumerate(zip(_plot_ids, _plot_names)):
        if plot_slot >= rows * cols:
            break

        ant_row = plot_slot // cols
        ant_col = plot_slot % cols
        sol_idx = _sol_id_to_idx.get(ant_id)  # None if antenna absent from solution

        # Inner 2-row gridspec inside this antenna's cell
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1,
            subplot_spec=outer[ant_row, ant_col],
            hspace=0.08,
            height_ratios=[1, 1],
        )
        ax_amp   = fig.add_subplot(inner[0])
        ax_phase = fig.add_subplot(inner[1], sharex=ax_amp)

        # Determine grey-out state
        _absent        = sol_idx is None
        _fully_flagged = (not _absent) and (not np.any(valid[:, sol_idx, :]))
        _grey          = _absent or _fully_flagged

        if not _grey:
            # --- Plot gains ---
            gains_raw = solution.get('gains_raw')
            _sw = solution.get('smooth_window') or 0
            _has_raw = gains_raw is not None and _sw > 1
            for pol_idx, label in enumerate(stokes_labels):
                col = colors[pol_idx % len(colors)]
                good = valid[:, sol_idx, pol_idx]
                amp_sm = np.where(good, np.abs(gains[:, sol_idx, pol_idx]), np.nan)
                pha_sm = np.where(good, np.degrees(np.angle(gains[:, sol_idx, pol_idx])), np.nan)
                amp_sm = np.where(plot_mask, amp_sm, np.nan)
                pha_sm = np.where(plot_mask, pha_sm, np.nan)
                if _has_raw:
                    amp_r = np.where(good, np.abs(np.asarray(gains_raw)[:, sol_idx, pol_idx]), np.nan)
                    pha_r = np.where(good, np.degrees(np.angle(np.asarray(gains_raw)[:, sol_idx, pol_idx])), np.nan)
                    amp_r = np.where(plot_mask, amp_r, np.nan)
                    pha_r = np.where(plot_mask, pha_r, np.nan)
                    # raw: dots only (no connecting line)
                    ax_amp.plot(freqs_mhz, amp_r, color=col, lw=0, marker='.', ms=2.0,
                                alpha=0.35, label=f'{label} raw')
                    ax_phase.plot(freqs_mhz, pha_r, color=col, lw=0, marker='.', ms=2.0, alpha=0.35)
                    # smoothed: solid line only (no markers, so it cleanly overlays)
                    ax_amp.plot(freqs_mhz, amp_sm, color=col, lw=1.2, label=f'{label} (w={_sw})')
                    ax_phase.plot(freqs_mhz, pha_sm, color=col, lw=1.2)
                else:
                    # No smoothing (w<=1) — dots + line show the per-channel values directly
                    ax_amp.plot(freqs_mhz, amp_sm, color=col, lw=0.7, marker='.', ms=2.0, label=label)
                    ax_phase.plot(freqs_mhz, pha_sm, color=col, lw=0.7, marker='.', ms=2.0)

        # Amplitude panel styling
        ax_amp.set_ylim(*amp_ylim)
        ax_amp.set_ylabel('Amp', fontsize=10)
        ax_amp.grid(True, alpha=0.25)
        ax_amp.tick_params(axis='y', labelsize=9)
        ax_amp.tick_params(axis='x', labelbottom=False)
        ax_amp.set_title(f'{ant_name} | Ant {ant_id}', fontsize=11, pad=14)

        # Channel numbers on top of amplitude panel
        top_ax = ax_amp.secondary_xaxis('top')
        top_ax.set_xticks(chan_tick_positions)
        top_ax.set_xticklabels(chan_tick_labels, fontsize=8)
        top_ax.set_xlabel('Channel', fontsize=9)

        # Phase panel styling
        ax_phase.set_ylim(*phase_ylim)
        ax_phase.set_ylabel('Phase\n(deg)', fontsize=10)
        ax_phase.set_xlabel('Freq (MHz)', fontsize=9)
        ax_phase.grid(True, alpha=0.25)
        ax_phase.tick_params(axis='both', labelsize=9)
        ax_phase.yaxis.set_ticks([-180, -90, 0, 90, 180])

        # Grey-out fully-flagged or absent antennas
        if _grey:
            _label = 'NOT IN SOLUTION' if _absent else 'FULLY FLAGGED'
            for _ax in (ax_amp, ax_phase):
                _ax.set_facecolor('#e0e0e0')
                for _s in _ax.spines.values():
                    _s.set_edgecolor('#aaaaaa')
            ax_amp.text(
                0.5, 0.5, _label,
                transform=ax_amp.transAxes,
                ha='center', va='center',
                fontsize=9, color='#888888',
                fontweight='bold', alpha=0.9,
            )

        if not legend_handles:
            legend_handles, legend_labels = ax_amp.get_legend_handles_labels()

    fig.suptitle(
        title or f'Bandpass solutions: {solution["source_name"]} | ref ant {solution["reference_antenna"]}',
        fontsize=18,
        y=0.975,
    )
    if legend_handles:
        fig.legend(
            legend_handles, legend_labels,
            ncol=len(legend_handles),
            loc='upper center',
            bbox_to_anchor=(0.5, 0.965),
            fontsize=11,
        )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show(block=True)
    return fig


def plot_bandpass_corrected_vis_amp_vs_uvdist(
    vis: dict,
    solution: dict,
    title: str = '',
    amp_ylim=None,
    exclude_antennas=None,
    **kwargs,
):
    """Apply a saved bandpass in memory and plot corrected amplitudes/phases vs UV distance.

    exclude_antennas can be a list/tuple of antenna names (e.g. 'C11') and/or
    antenna numbers. Any baseline touching those antennas is dropped before
    correction and plotting.
    """
    vis_use = _filter_vis_excluded_antennas(vis, solution, exclude_antennas=exclude_antennas)
    corrected = apply_bandpass_solution(vis_use, solution)
    plot_payload = dict(corrected)
    plot_payload['amp'] = corrected['amp_corrected']
    plot_payload['phase_deg'] = corrected['phase_deg_corrected']
    return plot_vis_amp_vs_uvdist(plot_payload, title=title, amp_ylim=amp_ylim, **kwargs)


def plot_corrected_vector_avg_spectrum(
    vis: dict,
    solution: dict,
    title: str = '',
    exclude_antennas=None,
    skip_edge_channels: Union[int, Tuple[int, int]] = (10, 5),
    save_path: Optional[Union[str, Path]] = None,
):
    """Plot corrected vector-averaged real spectrum vs the Perley-Butler flux model.

    The flux model is looked up from ``_FLUX_MODEL_REGISTRY`` using
    ``solution['source_name']`` so the correct model is used regardless of
    whether the solution was derived from 3C48, 3C286, or any other registered
    calibrator.

    Top panel: per-pol vector-averaged real(vis) and PB2017 model.
    Bottom panel: residual (data - model) per pol.
    """
    vis_use = _filter_vis_excluded_antennas(vis, solution, exclude_antennas=exclude_antennas)
    corrected = apply_bandpass_solution(vis_use, solution)

    freqs_hz = np.asarray(corrected['freqs_hz'], dtype=np.float64)
    freqs_mhz = freqs_hz / 1e6
    _sol_source = (solution.get('source_name') or '').upper()
    _model_fn = _FLUX_MODEL_REGISTRY.get(_sol_source)
    if _model_fn is None:
        raise ValueError(
            f'plot_corrected_vector_avg_spectrum: no flux model registered for '
            f'source "{_sol_source}". Registered: {sorted(_FLUX_MODEL_REGISTRY)}'
        )
    model = _model_fn(freqs_hz)

    if isinstance(skip_edge_channels, tuple):
        if len(skip_edge_channels) != 2:
            raise ValueError('skip_edge_channels tuple must have exactly two values: (start, end).')
        skip_start, skip_end = int(skip_edge_channels[0]), int(skip_edge_channels[1])
    else:
        skip_start = skip_end = int(skip_edge_channels)
    if skip_start < 0 or skip_end < 0:
        raise ValueError('skip_edge_channels must be non-negative.')

    chan_mask = np.ones(freqs_hz.size, dtype=bool)
    if skip_start > 0:
        chan_mask[:min(skip_start, chan_mask.size)] = False
    if skip_end > 0:
        chan_mask[max(0, chan_mask.size - skip_end):] = False

    vis_corr = np.asarray(corrected['vis_complex_corrected'], dtype=np.complex128)
    weights = np.asarray(corrected['weight'], dtype=np.float64)
    flagged = np.asarray(corrected.get('flagged_corrected', corrected.get('flagged')), dtype=bool)
    stokes_labels = list(corrected['stokes_labels'])

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True,
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.06}
    )

    model_plot = np.where(chan_mask, model, np.nan)
    ax_top.plot(freqs_mhz, model_plot, color='k', lw=2.0, label=f'Perley-Butler 2017 ({_sol_source})')

    for pol_idx, pol in enumerate(stokes_labels):
        z = vis_corr[:, :, pol_idx]
        w = weights[:, :, pol_idx]
        good = (~flagged[:, :, pol_idx]) & np.isfinite(z.real) & np.isfinite(z.imag) & np.isfinite(w) & (w > 0)

        num = np.nansum(np.where(good, w * z, 0.0), axis=0)
        den = np.nansum(np.where(good, w, 0.0), axis=0)
        vec = np.full(freqs_hz.size, np.nan + 1j * np.nan, dtype=np.complex128)
        ok = den > 0
        vec[ok] = num[ok] / den[ok]

        real_spec = np.real(vec)
        real_plot = np.where(chan_mask, real_spec, np.nan)
        resid_plot = np.where(chan_mask, real_spec - model, np.nan)

        ax_top.plot(freqs_mhz, real_plot, lw=1.4, label=f'{pol} vector-avg Re(V)')
        ax_bot.plot(freqs_mhz, resid_plot, lw=1.2, label=f'{pol} residual')

    ax_top.set_ylabel('Flux Density (Jy)')
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(fontsize=9, loc='best')

    ax_bot.axhline(0.0, color='k', lw=1.0, ls='--')
    ax_bot.set_ylabel('Data - Model (Jy)')
    ax_bot.set_xlabel('Frequency (MHz)')
    ax_bot.grid(True, alpha=0.3)
    ax_bot.legend(fontsize=8, loc='best')

    fig.suptitle(title or f'{_sol_source} corrected vector-averaged spectrum vs PB2017 model', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches='tight')
        print(f'[plot] Saved spectrum+residual plot to: {save_path}')
    plt.show(block=True)
    return fig

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

import matplotlib.pyplot as plt


def plot_vis_amp_vs_time(
    vis,
    title='',
    time_unit='min',
    figsize=(14, 3.5),
    show_phase=False,
    alpha=0.15,
):
    stokes = vis['stokes_labels']
    jd = vis['jd']
    amp = vis['amp']
    phase = vis['phase_deg']

    t0 = jd.min()
    scale = {'min': 1440.0, 'sec': 86400.0, 'jd': 1.0}[time_unit]
    t = (jd - t0) * scale
    t0_utc = Time(t0, format='jd', scale='utc').isot
    xlabel = f'Time ({time_unit} from {t0_utc})'

    nchan = amp.shape[1]
    t_broad = np.repeat(t, nchan)

    n_pol = len(stokes)
    rows = n_pol * (2 if show_phase else 1)
    fig, axes = plt.subplots(rows, 1, figsize=(figsize[0], figsize[1] * rows),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]

    for pi, label in enumerate(stokes):
        ax_amp = axes[pi * (2 if show_phase else 1)]
        amp_flat = amp[:, :, pi].ravel()
        ax_amp.scatter(t_broad, amp_flat, s=0.3, alpha=alpha,
                       rasterized=True, color=f'C{pi}')
        ax_amp.set_ylabel(f'{label}\nAmp')
        ax_amp.grid(True, alpha=0.3)
        _n_tot  = amp_flat.size
        _n_bad  = int(np.count_nonzero(~np.isfinite(amp_flat)))
        _n_good = _n_tot - _n_bad
        _n_el   = vis.get('n_rows_skipped_elevation', 0) * amp.shape[1]
        _n_p1   = vis.get('n_rows_dropped_phase1',    0) * amp.shape[1]
        _pct    = 100.0 * _n_bad / _n_tot if _n_tot else 0.0
        _lines  = [f'{_n_good:,} of {_n_tot:,} plotted, {_n_bad:,} flagged ({_pct:.1f}%)']
        if _n_el:
            _lines.append(f'el-skipped: {_n_el:,} (not counted)')
        if _n_p1:
            _lines.append(f'Phase-1 dropped: {_n_p1:,} (not counted)')
        ax_amp.text(
            0.01, 0.98, '\n'.join(_lines),
            transform=ax_amp.transAxes, fontsize=7,
            va='top', ha='left', linespacing=1.4,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7),
        )

        if show_phase:
            ax_ph = axes[pi * 2 + 1]
            ax_ph.scatter(t_broad, phase[:, :, pi].ravel(), s=0.3, alpha=alpha,
                          rasterized=True, color=f'C{pi}', marker='.')
            ax_ph.set_ylabel(f'{label}\nPhase (deg)')
            ax_ph.set_ylim(-185, 185)
            ax_ph.grid(True, alpha=0.3)

    axes[-1].set_xlabel(xlabel)
    fig.suptitle(title or 'Visibility amplitude vs time', fontsize=11)
    fig.tight_layout()
    plt.show(block=True)


def plot_vis_amp_vs_channel(
    vis,
    title='',
    figsize=(14, 3.5),
    show_phase=False,
    alpha=0.15,
):
    stokes = vis['stokes_labels']
    freqs_mhz = vis['freqs_hz'] / 1e6
    amp = vis['amp']
    phase = vis['phase_deg']

    nrows = amp.shape[0]
    f_broad = np.tile(freqs_mhz, nrows)

    n_pol = len(stokes)
    rows = n_pol * (2 if show_phase else 1)
    fig, axes = plt.subplots(rows, 1, figsize=(figsize[0], figsize[1] * rows),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]

    for pi, label in enumerate(stokes):
        ax_amp = axes[pi * (2 if show_phase else 1)]
        amp_flat = amp[:, :, pi].ravel()
        ax_amp.scatter(f_broad, amp_flat, s=0.3, alpha=alpha,
                       rasterized=True, color=f'C{pi}')
        ax_amp.set_ylabel(f'{label}\nAmp')
        ax_amp.grid(True, alpha=0.3)
        _n_tot  = amp_flat.size
        _n_bad  = int(np.count_nonzero(~np.isfinite(amp_flat)))
        _n_good = _n_tot - _n_bad
        _n_el   = vis.get('n_rows_skipped_elevation', 0) * amp.shape[1]
        _n_p1   = vis.get('n_rows_dropped_phase1',    0) * amp.shape[1]
        _pct    = 100.0 * _n_bad / _n_tot if _n_tot else 0.0
        _lines  = [f'{_n_good:,} of {_n_tot:,} plotted, {_n_bad:,} flagged ({_pct:.1f}%)']
        if _n_el:
            _lines.append(f'el-skipped: {_n_el:,} (not counted)')
        if _n_p1:
            _lines.append(f'Phase-1 dropped: {_n_p1:,} (not counted)')
        ax_amp.text(
            0.01, 0.98, '\n'.join(_lines),
            transform=ax_amp.transAxes, fontsize=7,
            va='top', ha='left', linespacing=1.4,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7),
        )

        if show_phase:
            ax_ph = axes[pi * 2 + 1]
            ax_ph.scatter(f_broad, phase[:, :, pi].ravel(), s=0.3, alpha=alpha,
                          rasterized=True, color=f'C{pi}')
            ax_ph.set_ylabel(f'{label}\nPhase (deg)')
            ax_ph.set_ylim(-185, 185)
            ax_ph.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Frequency (MHz)')
    fig.suptitle(title or 'Visibility amplitude vs channel', fontsize=11)
    fig.tight_layout()
    plt.show(block=True)


def plot_vis_amp_vs_uvdist(
    vis,
    title='',
    figsize=(14, 5),
    amp_ylim=None,
    show_phase=False,
    alpha=0.15,
    fit=None,
    fit_gaussian=False,
    nbins=80,
    save_path=None,
    hline_jy=None,
    signed=False,
):
    """Amplitude (and optionally phase) vs UV distance with optional model fit.

    Parameters
    ----------
    fit : None, 'disk', 'gaussian', 'ring', 'annulus', or 'composite'
        'disk'      — uniform-disk Airy pattern  A·|2J₁(πθu)/(πθu)| + C
        'gaussian'  — Gaussian envelope  A·exp(−u²/2σ²) + C
        'ring'      — thin ring (limb-brightened Moon)  A·|J₀(πθu)| + C
        'annulus'   — thick annulus (bright between θ_in and θ_out)
        'composite' — thin ring (thermal limb) + Gaussian (extended) + point source
        None        — no fit  (legacy *fit_gaussian=True* still works)
    """
    from scipy.optimize import curve_fit
    from scipy.special import j0, j1

    # legacy compat
    if fit is None and fit_gaussian:
        fit = 'gaussian'

    stokes = vis['stokes_labels']
    uvd = vis['uvdist_klambda']
    amp = vis['amp']
    phase = vis['phase_deg']

    nchan = amp.shape[1]
    # Use per-channel UV distance if available (frequency-dependent)
    if 'uvdist_per_chan' in vis:
        uvd_broad = vis['uvdist_per_chan'].ravel()   # (nrows*nchan,)
    else:
        uvd_broad = np.repeat(uvd, nchan)

    n_pol = len(stokes)
    rows = n_pol * (2 if show_phase else 1)
    fig, axes = plt.subplots(rows, 1, figsize=(figsize[0], figsize[1] * rows),
                             sharex=True, squeeze=False, dpi=150)
    axes = axes[:, 0]

    # --- model definitions ------------------------------------------------
    def _gauss(u, A, sigma, C):
        return A * np.exp(-u**2 / (2.0 * sigma**2)) + C

    def _disk(u_kl, A, theta_arcmin, C):
        """Uniform-disk visibility: A · |2·J1(x)/x| + C
        where x = π · θ · u_λ,  θ in radians, u_λ = u_kl * 1000."""
        theta_rad = np.radians(theta_arcmin / 60.0)
        x = np.pi * theta_rad * u_kl * 1e3          # u in λ
        # safe division at x=0
        out = np.ones_like(x, dtype=np.float64)
        nz = np.abs(x) > 1e-12
        out[nz] = 2.0 * j1(x[nz]) / x[nz]
        return A * np.abs(out) + C

    def _jinc(theta_rad, u_kl):
        """Helper: 2·J₁(πθu)/(πθu), safe at u=0."""
        x = np.pi * theta_rad * u_kl * 1e3
        out = np.ones_like(x, dtype=np.float64)
        nz = np.abs(x) > 1e-12
        out[nz] = 2.0 * j1(x[nz]) / x[nz]
        return out

    def _ring(u_kl, A, theta_arcmin, C):
        """Thin ring (δ-ring at angular diameter θ): A · |J₀(πθu)| + C"""
        theta_rad = np.radians(theta_arcmin / 60.0)
        x = np.pi * theta_rad * u_kl * 1e3
        return A * np.abs(j0(x)) + C

    def _annulus(u_kl, A, theta_out_arcmin, theta_in_arcmin, C):
        """Thick annulus (uniform brightness between θ_in and θ_out).
        V = A · |[θ_out²·jinc(θ_out) - θ_in²·jinc(θ_in)] / (θ_out² - θ_in²)| + C"""
        theta_out_r = np.radians(theta_out_arcmin / 60.0)
        theta_in_r  = np.radians(theta_in_arcmin / 60.0)
        num = (theta_out_r**2 * _jinc(theta_out_r, u_kl)
               - theta_in_r**2 * _jinc(theta_in_r, u_kl))
        denom = theta_out_r**2 - theta_in_r**2
        denom = max(denom, 1e-30)  # avoid div-by-zero
        return A * np.abs(num / denom) + C

    def _composite(u_kl, A_ring, theta_arcmin, A_gauss, sigma_kl, S_pt):
        """Thin ring (thermal limb) + Gaussian (extended) + point source.
        V = A_ring·|J₀(πθu)| + A_gauss·exp(−u²/2σ²) + S_pt"""
        theta_rad = np.radians(theta_arcmin / 60.0)
        x = np.pi * theta_rad * u_kl * 1e3
        return A_ring * np.abs(j0(x)) + A_gauss * np.exp(-u_kl**2 / (2.0 * sigma_kl**2)) + S_pt

    for pi, label in enumerate(stokes):
        ax_amp = axes[pi * (2 if show_phase else 1)]
        amp_flat = amp[:, :, pi].ravel()
        ax_amp.scatter(uvd_broad, amp_flat, s=0.3, alpha=alpha,
                       rasterized=True, color=f'C{pi}')
        ax_amp.set_ylabel(f'{label}\n{"Re(V)" if signed else "Amp"}')
        if amp_ylim is not None:
            ax_amp.set_ylim(*amp_ylim)
        ax_amp.grid(True, alpha=0.3)
        _n_tot  = amp_flat.size
        _n_bad  = int(np.count_nonzero(~np.isfinite(amp_flat)))
        _n_good = _n_tot - _n_bad
        _n_el   = vis.get('n_rows_skipped_elevation', 0) * amp.shape[1]
        _n_p1   = vis.get('n_rows_dropped_phase1',    0) * amp.shape[1]
        _pct    = 100.0 * _n_bad / _n_tot if _n_tot else 0.0
        _lines  = [f'{_n_good:,} of {_n_tot:,} plotted, {_n_bad:,} flagged ({_pct:.1f}%)']
        if _n_el:
            _lines.append(f'el-skipped: {_n_el:,} (not counted)')
        if _n_p1:
            _lines.append(f'Phase-1 dropped: {_n_p1:,} (not counted)')
        ax_amp.text(
            0.01, 0.98, '\n'.join(_lines),
            transform=ax_amp.transAxes, fontsize=7,
            va='top', ha='left', linespacing=1.4,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7),
        )
        if hline_jy is not None:
            if signed:
                ax_amp.axhline(+hline_jy, color='red', lw=1.5, ls='--', alpha=0.8,
                               label=f'flag threshold = \u00b1{hline_jy} Jy')
                ax_amp.axhline(-hline_jy, color='red', lw=1.5, ls='--', alpha=0.8)
            else:
                ax_amp.axhline(hline_jy, color='red', lw=1.5, ls='--', alpha=0.8,
                               label=f'flag threshold = {hline_jy} Jy')
            ax_amp.legend(fontsize=8, loc='upper right')

        # --- envelope fit -------------------------------------------------
        if fit in ('gaussian', 'disk', 'ring', 'annulus', 'composite'):
            valid = np.isfinite(amp_flat) & np.isfinite(uvd_broad)
            x = uvd_broad[valid]
            y = amp_flat[valid]
            if x.size > 20:
                edges = np.linspace(x.min(), x.max(), nbins + 1)
                centres = 0.5 * (edges[:-1] + edges[1:])
                bin_idx = np.digitize(x, edges) - 1
                bin_idx = np.clip(bin_idx, 0, nbins - 1)
                medians = np.empty(nbins)
                for bi in range(nbins):
                    vals = y[bin_idx == bi]
                    medians[bi] = np.nanmedian(vals) if vals.size else np.nan
                good = np.isfinite(medians)
                if good.sum() > 4:
                    cx = centres[good]
                    cy = medians[good]
                    u_smooth = np.linspace(cx.min(), cx.max(), 300)

                    if fit == 'gaussian':
                        try:
                            p0 = [cy.max() - cy.min(), 0.1, cy.min()]
                            popt, _ = curve_fit(_gauss, cx, cy, p0=p0,
                                                maxfev=5000)
                            A_fit, sigma_fit, C_fit = popt
                            ax_amp.plot(u_smooth, _gauss(u_smooth, *popt),
                                        color='k', lw=2, ls='--',
                                        label=(f'Gauss: '
                                               f'$\\sigma$={abs(sigma_fit):.3f} k$\\lambda$'))
                            fwhm_kl = 2.355 * abs(sigma_fit)
                            # FT pair: sky Gaussian sigma = 1/(2*pi*sigma_uv)
                            sigma_sky_rad = 1.0 / (2.0 * np.pi * abs(sigma_fit) * 1e3)
                            sigma_sky_arcmin = np.degrees(sigma_sky_rad) * 60.0
                            fwhm_sky_arcmin = 2.355 * sigma_sky_arcmin
                            ax_amp.legend(fontsize=8, loc='upper right')
                            print(f'  [{label}] Gaussian: A={A_fit:.4f}, '
                                  f'sigma={abs(sigma_fit):.4f} klambda, '
                                  f'C={C_fit:.4f}, '
                                  f'FWHM(uv)={fwhm_kl:.4f} klambda, '
                                  f'sky sigma={sigma_sky_arcmin:.2f} arcmin, '
                                  f'sky FWHM={fwhm_sky_arcmin:.2f} arcmin')
                        except RuntimeError:
                            print(f'  [{label}] Gaussian fit did not converge')

                    elif fit == 'disk':
                        try:
                            # initial guess: amplitude ~ peak-floor,
                            # theta ~ 31' (Moon), floor ~ min median
                            p0 = [cy.max() - cy.min(), 31.0, cy.min()]
                            popt, pcov = curve_fit(
                                _disk, cx, cy, p0=p0,
                                bounds=([0, 0.1, -np.inf],
                                        [np.inf, 120.0, np.inf]),
                                maxfev=10000)
                            A_fit, theta_fit, C_fit = popt
                            perr = np.sqrt(np.diag(pcov))
                            ax_amp.plot(
                                u_smooth, _disk(u_smooth, *popt),
                                color='k', lw=2, ls='--',
                                label=(f'Disk: '
                                       f'$\\theta$={theta_fit:.1f}\' '
                                       f'$\\pm${perr[1]:.1f}\''))
                            # first null: u_null = 1.22 / θ_rad
                            theta_rad = np.radians(theta_fit / 60.0)
                            null_kl = 1.22 / (theta_rad * 1e3)
                            ax_amp.legend(fontsize=8, loc='upper right')
                            print(f'  [{label}] Disk fit: A={A_fit:.4f}, '
                                  f'theta={theta_fit:.2f} +/- {perr[1]:.2f} arcmin, '
                                  f'C={C_fit:.4f}, '
                                  f'1st null={null_kl:.3f} klambda')
                        except RuntimeError:
                            print(f'  [{label}] Disk fit did not converge')

                    elif fit == 'ring':
                        try:
                            p0 = [cy.max() - cy.min(), 31.0, cy.min()]
                            popt, pcov = curve_fit(
                                _ring, cx, cy, p0=p0,
                                bounds=([0, 0.1, -np.inf],
                                        [np.inf, 120.0, np.inf]),
                                maxfev=10000)
                            A_fit, theta_fit, C_fit = popt
                            perr = np.sqrt(np.diag(pcov))
                            ax_amp.plot(
                                u_smooth, _ring(u_smooth, *popt),
                                color='k', lw=2, ls='--',
                                label=(f'Ring: '
                                       f'$\\theta$={theta_fit:.1f}\' '
                                       f'$\\pm${perr[1]:.1f}\''))
                            # first null of J0: x ≈ 2.4048 → u_null
                            theta_rad = np.radians(theta_fit / 60.0)
                            null_kl = 2.4048 / (np.pi * theta_rad * 1e3)
                            ax_amp.legend(fontsize=8, loc='upper right')
                            print(f'  [{label}] Ring fit: A={A_fit:.4f}, '
                                  f'theta={theta_fit:.2f} +/- {perr[1]:.2f} arcmin, '
                                  f'C={C_fit:.4f}, '
                                  f'1st null={null_kl:.3f} klambda')
                        except RuntimeError:
                            print(f'  [{label}] Ring fit did not converge')

                    elif fit == 'annulus':
                        try:
                            # θ_out ~ 31', θ_in ~ 27' → ~2' wide bright rim
                            peak = cy.max() - cy.min()
                            p0 = [peak, 31.0, 27.0, cy.min()]
                            popt, pcov = curve_fit(
                                _annulus, cx, cy, p0=p0,
                                bounds=([0, 5.0, 0.1, -np.inf],
                                        [np.inf, 120.0, 120.0, np.inf]),
                                maxfev=10000)
                            A_fit, th_out, th_in, C_fit = popt
                            perr = np.sqrt(np.diag(pcov))
                            ax_amp.plot(
                                u_smooth, _annulus(u_smooth, *popt),
                                color='k', lw=2, ls='--',
                                label=(f'Annulus: '
                                       f'$\\theta_{{out}}$={th_out:.1f}\' '
                                       f'$\\theta_{{in}}$={th_in:.1f}\''))
                            width = (th_out - th_in) / 2.0
                            ax_amp.legend(fontsize=8, loc='upper right')
                            print(f'  [{label}] Annulus fit: A={A_fit:.4f}, '
                                  f'theta_out={th_out:.2f} +/- {perr[1]:.2f} arcmin, '
                                  f'theta_in={th_in:.2f} +/- {perr[2]:.2f} arcmin, '
                                  f'rim_width={width:.2f} arcmin, '
                                  f'C={C_fit:.4f}')
                        except RuntimeError:
                            print(f'  [{label}] Annulus fit did not converge')

                    elif fit == 'composite':
                        try:
                            # A_ring, theta, A_gauss, sigma, S_pt
                            peak = cy.max() - cy.min()
                            p0 = [peak * 0.6, 31.0, peak * 0.3, 0.5, cy.min()]
                            popt, pcov = curve_fit(
                                _composite, cx, cy, p0=p0,
                                bounds=([0, 0.1, 0, 0.01, -np.inf],
                                        [np.inf, 120.0, np.inf, 10.0, np.inf]),
                                maxfev=10000)
                            A_r, theta_fit, A_g, sig_fit, S_pt = popt
                            perr = np.sqrt(np.diag(pcov))

                            # --- statistical significance ------------------
                            # Residuals: composite vs constant-only
                            resid_composite = cy - _composite(cx, *popt)
                            resid_const = cy - np.mean(cy)
                            rms_composite = np.sqrt(np.mean(resid_composite**2))
                            rms_const = np.sqrt(np.mean(resid_const**2))
                            # SNR = amplitude / residual RMS (true signal-to-noise)
                            snr_ring = A_r / rms_composite if rms_composite > 0 else 0.0
                            snr_gauss = A_g / rms_composite if rms_composite > 0 else 0.0
                            # Fit constraint: A / sigma_fit (covariance diagonal)
                            constr_ring = A_r / perr[0] if perr[0] > 0 else 0.0
                            constr_gauss = A_g / perr[2] if perr[2] > 0 else 0.0
                            n_pts = len(cx)
                            # BIC = n*ln(RSS/n) + k*ln(n)
                            bic_composite = (n_pts * np.log(rms_composite**2)
                                             + 5 * np.log(n_pts))
                            bic_const = (n_pts * np.log(rms_const**2)
                                         + 1 * np.log(n_pts))
                            delta_bic = bic_const - bic_composite  # >0 favours composite
                            rms_improv = ((rms_const - rms_composite)
                                          / rms_const * 100)
                            # Correlation matrix (degeneracy diagnostic)
                            std_diag = np.sqrt(np.diag(pcov))
                            with np.errstate(divide='ignore', invalid='ignore'):
                                corr_matrix = pcov / np.outer(std_diag, std_diag)
                            corr_ring_gauss = corr_matrix[0, 2]  # A_ring vs A_gauss

                            total_A = A_r + A_g
                            ring_pct = A_r / total_A * 100 if total_A > 0 else 0.0
                            # S_model(u=0) = A_ring + A_gauss + S_pt  (model extrapolation to zero baseline)
                            S_model = A_r + A_g + S_pt
                            pct_ring = A_r / S_model * 100 if S_model > 0 else 0.0
                            pct_gauss = A_g / S_model * 100 if S_model > 0 else 0.0
                            pct_pt = S_pt / S_model * 100 if S_model > 0 else 0.0
                            # σ_kλ → σ_λ → σ_sky (rad) → arcmin
                            sig_lambda = sig_fit * 1e3               # kλ → λ
                            sig_sky_rad = 1.0 / (2.0 * np.pi * sig_lambda)  # radians
                            sig_sky_arcmin = np.degrees(sig_sky_rad) * 60.0  # arcmin
                            fwhm_sky_arcmin = 2.355 * sig_sky_arcmin

                            # --- classification ---------------------------
                            sig_thresh = 3.0
                            ring_sig = snr_ring >= sig_thresh
                            gauss_sig = snr_gauss >= sig_thresh
                            if ring_sig and gauss_sig:
                                morphology = 'RESOLVED: ring + gauss'
                            elif ring_sig:
                                morphology = 'RESOLVED: ring-only (limb-bright)'
                            elif gauss_sig:
                                morphology = 'RESOLVED: gauss-only (extended)'
                            else:
                                morphology = 'POINT SOURCE (unresolved)'

                            # --- plot -------------------------------------
                            ax_amp.plot(
                                u_smooth, _composite(u_smooth, *popt),
                                color='k', lw=2, ls='--',
                                label='Composite fit')
                            ax_amp.plot(
                                u_smooth, A_r * np.abs(j0(np.pi * np.radians(theta_fit / 60.0) * u_smooth * 1e3)) + S_pt,
                                color='blue', lw=1, ls=':',
                                label=f'  ring ({pct_ring:.1f}%)', alpha=0.7)
                            ax_amp.plot(
                                u_smooth, A_g * np.exp(-u_smooth**2 / (2 * sig_fit**2)) + S_pt,
                                color='red', lw=1, ls=':',
                                label=f'  gauss ({pct_gauss:.1f}%)', alpha=0.7)
                            ax_amp.axhline(S_pt, color='green', lw=1, ls=':',
                                label=f'  S_pt ({pct_pt:.1f}%)', alpha=0.7)
                            ax_amp.legend(fontsize=8, loc='upper left')

                            # --- text box with fit parameters -------------
                            sig_tag = ' ***' if ring_sig else ''
                            gau_tag = ' ***' if gauss_sig else ''
                            box_lines = [
                                f'{morphology}',
                                f'',
                                f'A_ring   = {A_r:.4f} \u00b1 {perr[0]:.4f}  '
                                f'SNR={snr_ring:.1f}{sig_tag}',
                                f'\u03b8_ring  = {theta_fit:.2f}\' \u00b1 {perr[1]:.2f}\'',
                                f'A_gauss  = {A_g:.4f} \u00b1 {perr[2]:.4f}  '
                                f'SNR={snr_gauss:.1f}{gau_tag}',
                                f'\u03c3_uv   = {sig_fit:.3f} \u00b1 {perr[3]:.3f} k\u03bb',
                            ]
                            if gauss_sig:
                                box_lines.append(
                                    f'  \u2192 sky FWHM = {fwhm_sky_arcmin:.2f}\'')
                            box_lines += [
                                f'S_pt     = {S_pt:.4f} \u00b1 {perr[4]:.4f}',
                                f'',
                                f'S_model(u=0) = {S_model:.4f}',
                                f'  ring={pct_ring:.1f}%  '
                                f'gauss={pct_gauss:.1f}%  '
                                f'pt={pct_pt:.1f}%',
                                f'rms: {rms_composite:.3f}  '
                                f'({rms_improv:.1f}% better than point src)',
                                f'\u0394BIC      = {delta_bic:.1f}  '
                                f'({"justified" if delta_bic > 10 else "NOT justified" if delta_bic < 2 else "marginal"})',
                                f'corr(r,g)  = {corr_ring_gauss:+.2f}'
                                + ('  \u26a0 degen.' if abs(corr_ring_gauss) > 0.7 else ''),
                            ]
                            box_text = '\n'.join(box_lines)
                            props = dict(boxstyle='round,pad=0.4',
                                         facecolor='wheat', alpha=0.88,
                                         edgecolor='#888888', linewidth=0.8)
                            ax_amp.text(0.98, 0.97, box_text,
                                        transform=ax_amp.transAxes,
                                        fontsize=7.5, verticalalignment='top',
                                        horizontalalignment='right',
                                        fontfamily='monospace', bbox=props)

                            # --- print summary ----------------------------
                            print(f'  [{label}] Composite fit:')
                            print(f'    A_ring   = {A_r:.4f} +/- {perr[0]:.4f}  '
                                  f'SNR={snr_ring:.1f}  A/sig_fit={constr_ring:.1f}  '
                                  f'({pct_ring:.1f}% of S_model)  '
                                  f'{"*** SIGNIFICANT" if ring_sig else "not significant"}')
                            print(f'    theta    = {theta_fit:.2f} +/- {perr[1]:.2f} arcmin'
                                  + ('' if ring_sig else '  (unconstrained)'))
                            print(f'    A_gauss  = {A_g:.4f} +/- {perr[2]:.4f}  '
                                  f'SNR={snr_gauss:.1f}  A/sig_fit={constr_gauss:.1f}  '
                                  f'({pct_gauss:.1f}% of S_model)  '
                                  f'{"*** SIGNIFICANT" if gauss_sig else "not significant"}')
                            print(f'    sigma_uv = {sig_fit:.3f} +/- {perr[3]:.3f} klambda '
                                  f'(={sig_lambda:.1f} lambda)'
                                  + (f'  sky_sigma={sig_sky_arcmin:.2f}\' '
                                     f'sky_FWHM={fwhm_sky_arcmin:.2f}\''
                                     if gauss_sig else '  (unconstrained)'))
                            print(f'    S_pt     = {S_pt:.4f} +/- {perr[4]:.4f}  '
                                  f'({pct_pt:.1f}% of S_model)')
                            print(f'    S_model(u=0) = {S_model:.4f}  '
                                  f'(ring={pct_ring:.1f}% + gauss={pct_gauss:.1f}% + pt={pct_pt:.1f}%)  [model extrapolation]')
                            print(f'    rms: {rms_composite:.5f}  '
                                  f'({rms_improv:.1f}% better than point src model,  '
                                  f'rms_const={rms_const:.5f})')
                            print(f'    corr(A_ring, A_gauss) = {corr_ring_gauss:.3f}'
                                  + ('  *** degenerate' if abs(corr_ring_gauss) > 0.7 else ''))
                            print(f'    delta_BIC={delta_bic:.1f}  '
                                  f'({"composite justified" if delta_bic > 10 else "composite NOT justified" if delta_bic < 2 else "marginal"})')
                            print(f'    ==> {morphology}')
                        except RuntimeError:
                            print(f'  [{label}] Composite fit did not converge')

        if show_phase:
            ax_ph = axes[pi * 2 + 1]
            ax_ph.scatter(uvd_broad, phase[:, :, pi].ravel(), s=0.3, alpha=alpha,
                          rasterized=True, color=f'C{pi}')
            ax_ph.set_ylabel(f'{label}\nPhase (deg)')
            ax_ph.set_ylim(-185, 185)
            ax_ph.grid(True, alpha=0.3)

    axes[-1].set_xlabel(r'UV distance (k$\lambda$)', fontsize=10)
    fig.suptitle(title or 'Visibility vs UV distance', fontsize=12, fontweight='bold')
    fig.tight_layout()
    if save_path is not None:
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  Plot saved → {save_path}')
    plt.show(block=True)
