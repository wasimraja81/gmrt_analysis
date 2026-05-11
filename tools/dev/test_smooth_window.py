#!/usr/bin/env python3
"""
test_smooth_window.py
=====================
Compare bandpass gain solutions derived with different SMOOTH_WINDOW values
to verify that the parameter is correctly wired through and has the expected
effect (unsmoothed solutions should be noisier channel-to-channel).

Usage
-----
  source gmrt/bin/activate
  python test_smooth_window.py

  # override paths if needed
  python test_smooth_window.py \
      --fits  /data/other.FITS \
      --index /data/other.index.npz \
      --out   /tmp/smooth_compare.png

What is checked
---------------
  1. Raw channel-to-channel amplitude variance for each window size and antenna.
  2. Side-by-side amplitude + phase plot for a small number of antennas.
  3. Summary table: mean and std of the relative difference |g_1 - g_w| / |g_1|
     between the unsmoothed (window=1) solution and each other window, per antenna.

Expected outcome
----------------
  window=1  → highest channel-to-channel variance (per-channel noise is unsuppressed)
  window=5  → moderate smoothing, closely tracks signal
  window=50 → heavy smoothing, misses narrow spectral features / ripple

If window=1 and window=5 produce IDENTICAL solutions the smoothing is not working.
If all windows produce nearly identical solutions the data itself is very smooth and
smoothing is having little effect — this is also important to know.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── Resolve paths from config ──────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

_cfg_ns: dict = {}
_cfg_path = _SCRIPT_DIR / 'preprocess_ugmrt.cfg'
with open(_cfg_path) as _fh:
    exec(compile(_fh.read(), str(_cfg_path), 'exec'), _cfg_ns)

_FITS_DEFAULT      = _cfg_ns.get('CAL_FITS')
_INDEX_DEFAULT     = _cfg_ns.get('WORK_DIR', Path('/tmp')) / (
    Path(str(_cfg_ns.get('CAL_FITS', 'unknown'))).stem + '.index.npz'
)
_FLAG_TABLE_PATHS  = _cfg_ns.get('FLAG_TABLE_PATHS', [])
_CHAN_RANGE        = _cfg_ns.get('CHAN_RANGE', (64, 191))
_SOURCE            = _cfg_ns.get('SOURCE', '3C48')
_STOKES            = _cfg_ns.get('STOKES', ('RR', 'LL'))
_MAX_ROWS          = _cfg_ns.get('MAX_ROWS_SOLVE', 150_000)
_MIN_BASELINES     = _cfg_ns.get('MIN_BASELINES', 20)
_FLAG_ALL          = _cfg_ns.get('FLAG_ALL_CORRS_IF_ANY_RAWVIS_FLAGGED', True)

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Compare gain solutions across SMOOTH_WINDOW values.')
parser.add_argument('--fits',    default=str(_FITS_DEFAULT),  help='FITS path')
parser.add_argument('--index',   default=str(_INDEX_DEFAULT), help='Index cache path (built if missing)')
parser.add_argument('--out',     default=str(_SCRIPT_DIR / 'smooth_window_comparison.png'),
                    help='Output PNG path')
parser.add_argument('--windows', nargs='+', type=int, default=[1, 5, 50],
                    help='SMOOTH_WINDOW values to compare (default: 1 5 50)')
parser.add_argument('--n-ants',  type=int, default=5,
                    help='How many antennas to plot (first N by antenna id)')
args = parser.parse_args()

WINDOWS   = sorted(set(args.windows))
N_ANT_PLOT = args.n_ants
OUT_PATH  = Path(args.out)

print(f'FITS       : {args.fits}')
print(f'Index      : {args.index}')
print(f'CHAN_RANGE  : {_CHAN_RANGE}')
print(f'SOURCE     : {_SOURCE}')
print(f'STOKES     : {_STOKES}')
print(f'Comparing SMOOTH_WINDOW = {WINDOWS}')

import ugmrt_query as q

# ── Build / load index ────────────────────────────────────────────────────────
index = q.get_or_build_row_index(
    args.fits,
    cache_path=args.index,
    force_rebuild=False,
    validation_mode='fast',
    write_cache=True,
)
print(f'Index loaded: {len(index.get("antennas", []))} antennas')

# ── Solve for each window ─────────────────────────────────────────────────────
solutions: dict[int, dict] = {}
for w in WINDOWS:
    print(f'\nSolving with SMOOTH_WINDOW={w} ...')
    sol = q.derive_point_source_bandpass(
        index,
        source=_SOURCE,
        chan_range=_CHAN_RANGE,
        stokes=_STOKES,
        max_rows=_MAX_ROWS,
        smooth_window=w,
        min_baselines=_MIN_BASELINES,
        ignore_autos=True,
        flag_table_path=_FLAG_TABLE_PATHS if _FLAG_TABLE_PATHS else None,
        flag_all_corrs_if_any_rawvis_flagged=_FLAG_ALL,
    )
    solutions[w] = sol
    print(f'  antennas in solution : {len(sol["antenna_ids"])}')
    print(f'  channels             : {len(sol["chan_indices"])}')
    print(f'  smooth_window stored : {sol.get("smooth_window")}')

# ── Common antenna ids across all solutions ────────────────────────────────────
common_ids = set(solutions[WINDOWS[0]]['antenna_ids'])
for w in WINDOWS[1:]:
    common_ids &= set(solutions[w]['antenna_ids'])
common_ids = sorted(common_ids)[:N_ANT_PLOT]
print(f'\nPlotting {len(common_ids)} antennas: {common_ids}')

def get_ant_gains(sol, ant_id):
    """Return (amp, phase_deg, valid) arrays shaped (n_chan, n_stokes)."""
    ids = list(sol['antenna_ids'])
    if ant_id not in ids:
        return None, None, None
    idx = ids.index(ant_id)
    g = np.asarray(sol['gains'])[:, idx, :]       # (n_chan, n_stokes)
    v = np.asarray(sol['valid'])[:, idx, :]       # bool
    amp   = np.where(v, np.abs(g), np.nan)
    phase = np.where(v, np.rad2deg(np.angle(g)), np.nan)
    return amp, phase, v

freqs_mhz = np.asarray(solutions[WINDOWS[0]]['freqs_hz']) / 1e6
n_stokes  = len(_STOKES)

# ── Figure ────────────────────────────────────────────────────────────────────
# Layout: one row per antenna, columns = (amp RR, phase RR, amp LL, phase LL)
n_cols = 2 * n_stokes   # amp+phase per pol
fig, axes = plt.subplots(
    len(common_ids), n_cols,
    figsize=(6 * n_cols, 3.5 * len(common_ids)),
    squeeze=False,
)
fig.suptitle(
    f'SMOOTH_WINDOW comparison — {_SOURCE}  |  chan {_CHAN_RANGE[0]}–{_CHAN_RANGE[1]}\n'
    f'Windows: {WINDOWS}  (dotted=1, solid=5+)',
    fontsize=13,
)

_colors = {w: c for w, c in zip(WINDOWS, plt.cm.tab10.colors)}
_ls     = {w: (':' if w == 1 else '-') for w in WINDOWS}
_lw     = {w: (1.8 if w == 1 else 1.2) for w in WINDOWS}

ant_names = {int(a['antenna_no']): a.get('name', str(a['antenna_no']))
             for a in index.get('antennas', [])}

for row_i, ant_id in enumerate(common_ids):
    ant_label = ant_names.get(ant_id, str(ant_id))
    for pol_i, pol in enumerate(_STOKES):
        ax_amp   = axes[row_i, 2 * pol_i]
        ax_phase = axes[row_i, 2 * pol_i + 1]
        ax_amp.set_ylabel(f'Ant {ant_label}\nAmp (arb)', fontsize=9)
        ax_phase.set_ylabel('Phase (deg)', fontsize=9)
        ax_amp.set_title(f'{pol} amplitude', fontsize=9)
        ax_phase.set_title(f'{pol} phase', fontsize=9)

        for w in WINDOWS:
            amp, phase, valid = get_ant_gains(solutions[w], ant_id)
            if amp is None:
                continue
            ax_amp.plot(freqs_mhz, amp[:, pol_i],
                        color=_colors[w], ls=_ls[w], lw=_lw[w],
                        label=f'w={w}')
            ax_phase.plot(freqs_mhz, phase[:, pol_i],
                          color=_colors[w], ls=_ls[w], lw=_lw[w],
                          label=f'w={w}')

        ax_amp.grid(True, alpha=0.3)
        ax_phase.grid(True, alpha=0.3)
        ax_phase.set_ylim(-200, 200)
        if row_i == len(common_ids) - 1:
            ax_amp.set_xlabel('Freq (MHz)', fontsize=8)
            ax_phase.set_xlabel('Freq (MHz)', fontsize=8)

axes[0, -1].legend(fontsize=8, loc='upper right')

plt.tight_layout(rect=[0, 0, 1, 0.96])
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PATH, dpi=120)
print(f'\nPlot saved: {OUT_PATH}')

# ── Numerical summary ─────────────────────────────────────────────────────────
print('\n─── Channel-to-channel amplitude variability (stdev / mean) ───')
print(f'{"Antenna":<12}', end='')
for pol in _STOKES:
    for w in WINDOWS:
        print(f'  {pol}/w={w:<4}', end='')
print()

ref_w = 1   # unsmoothed reference
for ant_id in common_ids:
    ant_label = ant_names.get(ant_id, str(ant_id))
    print(f'{ant_label:<12}', end='')
    for pol_i, pol in enumerate(_STOKES):
        for w in WINDOWS:
            amp, _, valid = get_ant_gains(solutions[w], ant_id)
            if amp is None:
                print(f'  {"N/A":<8}', end='')
                continue
            a = amp[:, pol_i]
            a = a[np.isfinite(a)]
            cv = float(np.std(a) / np.mean(a)) if a.size > 1 and np.mean(a) > 0 else float('nan')
            print(f'  {cv:.4f}  ', end='')
    print()

print('\n─── Mean |g_w - g_1| / |g_1|  (deviation from unsmoothed) ───')
print(f'{"Antenna":<12}', end='')
for pol in _STOKES:
    for w in [x for x in WINDOWS if x != 1]:
        print(f'  {pol}/w={w:<4}', end='')
print()

for ant_id in common_ids:
    ant_label = ant_names.get(ant_id, str(ant_id))
    amp_ref, _, valid_ref = get_ant_gains(solutions[1], ant_id)
    print(f'{ant_label:<12}', end='')
    for pol_i, pol in enumerate(_STOKES):
        for w in [x for x in WINDOWS if x != 1]:
            amp_w, _, valid_w = get_ant_gains(solutions[w], ant_id)
            if amp_ref is None or amp_w is None:
                print(f'  {"N/A":<8}', end='')
                continue
            mask = np.isfinite(amp_ref[:, pol_i]) & np.isfinite(amp_w[:, pol_i])
            if mask.sum() == 0:
                print(f'  {"N/A":<8}', end='')
                continue
            rel_diff = np.abs(amp_w[mask, pol_i] - amp_ref[mask, pol_i]) / (np.abs(amp_ref[mask, pol_i]) + 1e-30)
            print(f'  {float(np.mean(rel_diff)):.4f}  ', end='')
    print()

print('\nDone.')
