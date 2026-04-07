"""
uGMRT UVFITS query and plotting utilities.

Lightweight memory-mapped I/O functions for uGMRT UVFITS files.
Designed for low-RAM environments (Raspberry Pi 4, 8 GB).
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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


def build_row_index(path: Union[str, Path]) -> dict:
    """Build a fast row index by reading group parameters in one sequential pass."""
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
    print(f'Index built: {gcount:,} rows, {len(id_to_name)} sources, '
          f'{mem_mb:.1f} MB RAM, {elapsed:.1f}s')
    for sid, name in sorted(id_to_name.items()):
        ranges = source_ranges.get(sid, [])
        total_rows = sum(e - s for s, e in ranges)
        print(f'  {name:>12s} (id={sid}): {total_rows:>8,} rows in {len(ranges)} block(s)')
    return idx


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

    n_ant = len(index['antennas'])
    cross_th = n_ant * (n_ant - 1) // 2
    freq = index['freq']

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
        'total_duration_sec': float((int_times[-1] - int_times[0]) * 86400.0),
        'n_channels': index['naxis4'],
        'n_correlations': index['naxis3'],
        'centre_freq_hz': 0.5 * (freq['freq_min_hz'] + freq['freq_max_hz']),
        'bFreq_hz': freq['freq_min_hz'],
        'eFreq_hz': freq['freq_max_hz'],
        'antenna_count': n_ant,
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
):
    """Load visibility data using the pre-built row index."""
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

    re_ = data[..., 0]
    im_ = data[..., 1]
    wt_ = data[..., 2]

    amp   = np.sqrt(re_**2 + im_**2)
    phase = np.degrees(np.arctan2(im_, re_)).astype(np.float32)
    flagged = wt_ <= 0
    amp[flagged]   = np.nan
    phase[flagged] = np.nan

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
        'vis_complex': re_ + 1j * im_,
        'amp': amp,
        'phase_deg': phase,
        'weight': wt_,
        'flagged': flagged,
        'stokes_labels': stokes_sel,
        'chan_indices': chan_idx,
        'nrows': int(data.shape[0]),
    }


# ---------------------------------------------------------------------------
# Non-destructive bandpass calibration utilities
# ---------------------------------------------------------------------------

_PB2017_3C48_COEFFS = np.array([1.3253, -0.7553, -0.1914, 0.0498], dtype=np.float64)


def flux_model_3c48_perley_butler_2017(freq_hz):
    """Return the 3C48 flux density model in Jy for the given frequencies."""
    freq_hz = np.asarray(freq_hz, dtype=np.float64)
    if np.any(freq_hz <= 0):
        raise ValueError('All frequencies must be positive.')

    freq_ghz = freq_hz / 1e9
    x = np.log10(freq_ghz)
    coeffs = _PB2017_3C48_COEFFS
    log_s = coeffs[0] + coeffs[1] * x + coeffs[2] * x**2 + coeffs[3] * x**3
    return np.power(10.0, log_s)


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

            amp = np.abs(gains[:, ant_idx, pol_idx])
            phase = np.unwrap(np.angle(gains[:, ant_idx, pol_idx]))

            amp_num = np.convolve(np.where(good, amp, 0.0), kernel, mode='same')
            phase_num = np.convolve(np.where(good, phase, 0.0), kernel, mode='same')
            den = np.convolve(good.astype(np.float64), kernel, mode='same')
            ok = den > 0
            amp_s = amp.copy()
            phase_s = phase.copy()
            amp_s[ok] = amp_num[ok] / den[ok]
            phase_s[ok] = phase_num[ok] / den[ok]
            smoothed[:, ant_idx, pol_idx] = amp_s * np.exp(1j * phase_s)

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
):
    """Derive per-antenna complex bandpass gains without modifying FITS data.

    This function only reads visibilities from disk, derives gains in memory,
    and returns a separate solution table that can be saved independently.
    """
    vis = load_vis_for_source(
        index,
        source=source,
        ant_range=ant_range,
        ant_list=ant_list,
        chan_range=chan_range,
        stokes=list(stokes),
        max_rows=max_rows,
    )

    antenna_ids = np.array(sorted(set(vis['ant1']).union(set(vis['ant2']))), dtype=np.int32)
    if antenna_ids.size < 2:
        raise ValueError('Need at least two antennas to derive bandpass solutions.')

    antenna_name_map = {
        int(item['antenna_no']): (item.get('name') or f'Ant{int(item["antenna_no"])}')
        for item in index.get('antennas', [])
        if item.get('antenna_no') is not None
    }
    antenna_names = [antenna_name_map.get(int(ant), f'Ant{int(ant)}') for ant in antenna_ids]

    stokes_labels = list(vis['stokes_labels'])
    freqs_hz = np.asarray(vis['freqs_hz'], dtype=np.float64)
    flux_jy = flux_model_3c48_perley_butler_2017(freqs_hz) if model_flux_jy is None else np.asarray(model_flux_jy, dtype=np.float64)
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
        'bad_data_policy': (
            'Ignored non-finite samples, zero-or-negative FITS weights, optional autos, '
            'and channels with insufficient surviving baselines. No flags were written to FITS.'
        ),
        'notes': 'Derived from visibilities in memory only. No FITS data were modified.',
    }


def save_bandpass_solution(solution, path: Union[str, Path]):
    """Write bandpass solutions to a separate compressed file on disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        'kind': solution['kind'],
        'source_name': solution['source_name'],
        'source_file': solution['source_file'],
        'reference_antenna': int(solution['reference_antenna']),
        'smooth_window': solution['smooth_window'],
        'max_rows': int(solution['max_rows']),
        'min_baselines': int(solution['min_baselines']),
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
        valid=np.asarray(solution['valid'], dtype=bool),
        residual_rms=np.asarray(solution['residual_rms'], dtype=np.float64),
        baseline_counts=np.asarray(solution['baseline_counts'], dtype=np.int32),
        input_flagged_counts=np.asarray(solution['input_flagged_counts'], dtype=np.int32),
        used_sample_counts=np.asarray(solution['used_sample_counts'], dtype=np.int32),
        skipped_channel_mask=np.asarray(solution['skipped_channel_mask'], dtype=bool),
        iterations=np.asarray(solution['iterations'], dtype=np.int32),
        flux_model_jy=np.asarray(solution['flux_model_jy'], dtype=np.float64),
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
            'bad_data_policy': metadata['bad_data_policy'],
            'notes': metadata['notes'],
            'freqs_hz': np.asarray(npz['freqs_hz'], dtype=np.float64),
            'chan_indices': np.asarray(npz['chan_indices'], dtype=np.int32),
            'antenna_ids': np.asarray(npz['antenna_ids'], dtype=np.int32),
            'antenna_names': np.asarray(npz['antenna_names']).astype(str).tolist(),
            'stokes_labels': np.asarray(npz['stokes_labels']).astype(str).tolist(),
            'gains': np.asarray(npz['gains'], dtype=np.complex128),
            'valid': np.asarray(npz['valid'], dtype=bool),
            'residual_rms': np.asarray(npz['residual_rms'], dtype=np.float64),
            'baseline_counts': np.asarray(npz['baseline_counts'], dtype=np.int32),
            'input_flagged_counts': np.asarray(npz['input_flagged_counts'], dtype=np.int32),
            'used_sample_counts': np.asarray(npz['used_sample_counts'], dtype=np.int32),
            'skipped_channel_mask': np.asarray(npz['skipped_channel_mask'], dtype=bool),
            'iterations': np.asarray(npz['iterations'], dtype=np.int32),
            'flux_model_jy': np.asarray(npz['flux_model_jy'], dtype=np.float64),
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
    if np.any(ant1_idx < 0) or np.any(ant2_idx < 0):
        raise ValueError('Some visibility antennas are missing from the bandpass solution.')

    corrected = np.array(vis['vis_complex'], dtype=np.complex128, copy=True)
    corrected_flagged = np.array(vis.get('flagged', np.zeros_like(vis['weight'], dtype=bool)), copy=True)

    sol_gains = np.asarray(solution['gains'], dtype=np.complex128)[sol_chan_idx, :, :]
    sol_valid = np.asarray(solution['valid'], dtype=bool)[sol_chan_idx, :, :]

    for vis_pol_idx, label in enumerate(vis_labels):
        if label not in common_labels:
            continue
        sol_pol_idx = sol_labels.index(label)
        g1 = sol_gains[:, ant1_idx, sol_pol_idx].T
        g2 = sol_gains[:, ant2_idx, sol_pol_idx].T
        valid = sol_valid[:, ant1_idx, sol_pol_idx].T & sol_valid[:, ant2_idx, sol_pol_idx].T
        denom = g1 * np.conj(g2)
        good = valid & np.isfinite(denom.real) & np.isfinite(denom.imag) & (np.abs(denom) > 0)
        corrected[:, :, vis_pol_idx][good] = corrected[:, :, vis_pol_idx][good] / denom[good]
        corrected[:, :, vis_pol_idx][~good] = np.nan + 1j * np.nan
        corrected_flagged[:, :, vis_pol_idx] |= ~good

    amp = np.abs(corrected)
    phase = np.degrees(np.angle(corrected)).astype(np.float32)
    amp[corrected_flagged] = np.nan
    phase[corrected_flagged] = np.nan

    out = dict(vis)
    out['vis_complex_corrected'] = corrected
    out['amp_corrected'] = amp
    out['phase_deg_corrected'] = phase
    out['flagged_corrected'] = corrected_flagged
    out['applied_bandpass_reference_antenna'] = int(solution['reference_antenna'])
    out['applied_bandpass_source'] = solution['source_name']
    return out


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
    figsize=(28, 36),
    phase_ylim=(-200.0, 200.0),
    amp_ylim=None,
    skip_edge_channels: Union[int, Tuple[int, int]] = (5, 5),
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
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
    """
    import matplotlib.gridspec as gridspec

    freqs_mhz = np.asarray(solution['freqs_hz'], dtype=np.float64) / 1e6
    chan_indices = np.asarray(solution['chan_indices'], dtype=np.int32)
    antenna_ids = np.asarray(solution['antenna_ids'], dtype=np.int32)
    antenna_names = solution.get('antenna_names') or [f'Ant{int(ant)}' for ant in antenna_ids]
    gains = np.asarray(solution['gains'], dtype=np.complex128)
    valid = np.asarray(solution['valid'], dtype=bool)
    stokes_labels = list(solution['stokes_labels'])

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

    for ant_idx, ant_id in enumerate(antenna_ids):
        if ant_idx >= rows * cols:
            break

        ant_id = int(ant_id)
        ant_name = antenna_names[ant_idx]
        ant_row = ant_idx // cols
        ant_col = ant_idx % cols

        # Inner 2-row gridspec inside this antenna's cell
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1,
            subplot_spec=outer[ant_row, ant_col],
            hspace=0.08,
            height_ratios=[1, 1],
        )
        ax_amp   = fig.add_subplot(inner[0])
        ax_phase = fig.add_subplot(inner[1], sharex=ax_amp)

        # --- Plot ---
        for pol_idx, label in enumerate(stokes_labels):
            col = colors[pol_idx % len(colors)]
            good = valid[:, ant_idx, pol_idx]
            amp = np.where(good, np.abs(gains[:, ant_idx, pol_idx]), np.nan)
            pha = np.where(good, np.degrees(np.angle(gains[:, ant_idx, pol_idx])), np.nan)
            amp = np.where(plot_mask, amp, np.nan)
            pha = np.where(plot_mask, pha, np.nan)
            ax_amp.plot(freqs_mhz, amp, color=col, lw=1.0, label=label)
            ax_phase.plot(freqs_mhz, pha, color=col, lw=1.0)

        # Amplitude panel
        ax_amp.set_ylim(*amp_ylim)
        ax_amp.set_ylabel('Amp', fontsize=8)
        ax_amp.grid(True, alpha=0.25)
        ax_amp.tick_params(axis='y', labelsize=7)
        ax_amp.tick_params(axis='x', labelbottom=False)  # hide bottom ticks; shared with phase
        ax_amp.set_title(f'{ant_name} | Ant {ant_id}', fontsize=9, pad=14)

        # Channel numbers on top of amplitude panel
        top_ax = ax_amp.secondary_xaxis('top')
        top_ax.set_xticks(chan_tick_positions)
        top_ax.set_xticklabels(chan_tick_labels, fontsize=6)
        top_ax.set_xlabel('Channel', fontsize=7)

        # Phase panel
        ax_phase.set_ylim(*phase_ylim)
        ax_phase.set_ylabel('Phase\n(deg)', fontsize=8)
        ax_phase.set_xlabel('Freq (MHz)', fontsize=7)
        ax_phase.grid(True, alpha=0.25)
        ax_phase.tick_params(axis='both', labelsize=7)
        ax_phase.yaxis.set_ticks([-180, -90, 0, 90, 180])

        if not legend_handles:
            legend_handles, legend_labels = ax_amp.get_legend_handles_labels()

    fig.suptitle(
        title or f'Bandpass solutions: {solution["source_name"]} | ref ant {solution["reference_antenna"]}',
        fontsize=16,
        y=0.975,
    )
    if legend_handles:
        fig.legend(
            legend_handles, legend_labels,
            ncol=len(legend_handles),
            loc='upper center',
            bbox_to_anchor=(0.5, 0.965),
            fontsize=10,
        )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
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
    """Plot corrected vector-averaged real spectrum vs Perley-Butler 3C48 model.

    Top panel: per-pol vector-averaged real(vis) and PB2017 model.
    Bottom panel: residual (data - model) per pol.
    """
    vis_use = _filter_vis_excluded_antennas(vis, solution, exclude_antennas=exclude_antennas)
    corrected = apply_bandpass_solution(vis_use, solution)

    freqs_hz = np.asarray(corrected['freqs_hz'], dtype=np.float64)
    freqs_mhz = freqs_hz / 1e6
    model = flux_model_3c48_perley_butler_2017(freqs_hz)

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
    ax_top.plot(freqs_mhz, model_plot, color='k', lw=2.0, label='Perley-Butler 2017 (3C48)')

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

    fig.suptitle(title or '3C48 corrected vector-averaged spectrum vs PB2017 model', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches='tight')
        print(f'[plot] Saved spectrum+residual plot to: {save_path}')
    plt.show()
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
        ax_amp.scatter(t_broad, amp[:, :, pi].ravel(), s=0.3, alpha=alpha,
                       rasterized=True, color=f'C{pi}')
        ax_amp.set_ylabel(f'{label}\nAmp')
        ax_amp.grid(True, alpha=0.3)

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
    plt.show()


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
        ax_amp.scatter(f_broad, amp[:, :, pi].ravel(), s=0.3, alpha=alpha,
                       rasterized=True, color=f'C{pi}')
        ax_amp.set_ylabel(f'{label}\nAmp')
        ax_amp.grid(True, alpha=0.3)

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
    plt.show()


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
        ax_amp.set_ylabel(f'{label}\nAmp')
        if amp_ylim is not None:
            ax_amp.set_ylim(*amp_ylim)
        ax_amp.grid(True, alpha=0.3)

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
    plt.show()
