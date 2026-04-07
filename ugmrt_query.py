"""
uGMRT UVFITS query and plotting utilities.

Lightweight memory-mapped I/O functions for uGMRT UVFITS files.
Designed for low-RAM environments (Raspberry Pi 4, 8 GB).
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

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
        'amp': amp,
        'phase_deg': phase,
        'weight': wt_,
        'stokes_labels': stokes_sel,
        'chan_indices': chan_idx,
        'nrows': int(data.shape[0]),
    }

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
