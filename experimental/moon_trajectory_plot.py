#!/usr/bin/env python3
"""Plot Moon trajectory across all integrations on the RA/Dec offset grid.

The telescope tracked a fixed RA/Dec (stored in the MS FIELD table).
The Moon drifts through the field; its per-integration position is computed
from astropy get_body() (GCRS apparent topocentric) and shown as an offset
from the tracked phase centre.

Usage
-----
  python experimental/moon_trajectory_plot.py \
      --ms   /path/to/moon0520_full.ms \
      --out  diagnostics_out/moon_imaging/moon0520_trajectory.png \
      --scan MOON0520
"""
import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from astropy.time import Time
from astropy.coordinates import EarthLocation, get_body


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULT_MS  = ('/Users/raj030/DATA/gmrt_40_014/work/casa_selfcal/'
                'moon0520_dev/moon0520_full.ms')
_DEFAULT_OUT = ('diagnostics_out/moon_imaging/moon0520_trajectory.png')
_IMSIZE       = 2048
_CELL_ARCSEC  = 1.5
_MOON_DIAM_ARCMIN = 30.0   # apparent Moon diameter circle to draw
_N_CIRCLES    = 5           # evenly-spaced integrations to draw circles for

# GMRT C02 ITRF reference coordinates (metres)
_GMRT_XYZ = (1656342.30, 5797947.77, 2073243.16)


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ms',   default=_DEFAULT_MS,
                   help='Path to the full-scan CASA MS (default: %(default)s)')
    p.add_argument('--out',  default=_DEFAULT_OUT,
                   help='Output PNG path (default: %(default)s)')
    p.add_argument('--scan', default='MOON0520',
                   help='Scan name string to locate moon FIELD row (default: %(default)s)')
    p.add_argument('--field-id', type=int, default=None,
                   help='Override FIELD_ID integer (autodetected if omitted)')
    p.add_argument('--imsize',    type=int,   default=_IMSIZE)
    p.add_argument('--cell-arcsec', type=float, default=_CELL_ARCSEC)
    p.add_argument('--dpi',       type=int,   default=150)
    return p.parse_args()


# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    from casatools import table as tb_tool  # type: ignore
    tb = tb_tool()

    # ── 1. Read unique integration timestamps for the target field ────────────
    tb.open(args.ms)
    times_s   = tb.getcol('TIME')
    field_ids = tb.getcol('FIELD_ID')
    tb.close()

    # Determine FIELD_ID: use override, otherwise find by name
    if args.field_id is not None:
        fid = args.field_id
    else:
        tb.open(os.path.join(args.ms, 'FIELD'))
        names = tb.getcol('NAME')
        tb.close()
        matches = [i for i, n in enumerate(names) if args.scan.upper() in n.upper()]
        if not matches:
            raise RuntimeError(f'No FIELD row matches scan name "{args.scan}". '
                               f'Available: {list(names)}')
        fid = matches[0]
        print(f'[trajectory] FIELD_ID={fid}  name="{names[fid]}"')

    sel_times = np.unique(times_s[field_ids == fid])
    jds = sel_times / 86400.0 + 2400000.5
    print(f'[trajectory] {len(jds)} integrations found for field {fid}')

    # ── 2. Stored FIELD phase centre (radians → degrees, wrap RA to [0,360)) ──
    tb.open(os.path.join(args.ms, 'FIELD'))
    pd_col = tb.getcol('PHASE_DIR')   # shape (2, n_poly, n_rows)
    tb.close()
    field_ra  = float(np.degrees(pd_col[0, 0, fid])) % 360.0
    field_dec = float(np.degrees(pd_col[1, 0, fid]))
    print(f'[trajectory] Tracked phase centre: RA={field_ra:.4f}°  Dec={field_dec:.4f}°')

    # ── 3. Moon GCRS apparent position per integration ────────────────────────
    loc = EarthLocation.from_geocentric(*_GMRT_XYZ, unit='m')
    ra_list, dec_list = [], []
    for i, jd in enumerate(jds):
        if i % 20 == 0:
            print(f'[trajectory]   ephemeris {i}/{len(jds)}...', flush=True)
        moon = get_body('moon', Time(jd, format='jd', scale='utc'), location=loc)
        ra_list.append(float(moon.ra.deg))
        dec_list.append(float(moon.dec.deg))

    ra_arr  = np.array(ra_list)
    dec_arr = np.array(dec_list)

    # ── 4. Offset from phase centre (arcmin, RA cos-dec corrected) ────────────
    cos_dec  = np.cos(np.radians(field_dec))
    dra_amin  = (ra_arr  - field_ra)  * cos_dec * 60.0
    ddec_amin = (dec_arr - field_dec) * 60.0

    # ── 5. Plot ───────────────────────────────────────────────────────────────
    hw = args.imsize * args.cell_arcsec / 2.0 / 60.0   # image half-width, arcmin

    fig, ax = plt.subplots(figsize=(8, 8))

    # Image boundary square
    ax.add_patch(mpatches.Rectangle(
        (-hw, -hw), 2 * hw, 2 * hw,
        lw=1.5, edgecolor='black', facecolor='#f5f5f5',
        linestyle='--', zorder=0,
        label=f'Image {args.imsize}×{args.imsize} pix ({args.cell_arcsec}") = ±{hw:.1f}\''
    ))

    # Tracked phase centre
    ax.plot(0, 0, '+', ms=16, mew=2.5, color='black', zorder=6,
            label='Tracked phase centre')

    # Trajectory scatter coloured by integration index
    sc = ax.scatter(dra_amin, ddec_amin,
                    c=np.arange(len(jds)), cmap='plasma', s=10, zorder=4,
                    label=f'Moon ({len(jds)} integrations)')
    cb = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label('Integration index', fontsize=10)

    # Moon circles (N_CIRCLES evenly spaced along trajectory)
    idx_circles = np.linspace(0, len(jds) - 1, _N_CIRCLES, dtype=int)
    moon_r      = _MOON_DIAM_ARCMIN / 2.0
    circle_cols = plt.cm.plasma(np.linspace(0, 1, _N_CIRCLES))
    for k, idx in enumerate(idx_circles):
        ax.add_patch(mpatches.Circle(
            (dra_amin[idx], ddec_amin[idx]), moon_r,
            lw=1.4, edgecolor=circle_cols[k], facecolor='none',
            alpha=0.85, zorder=5
        ))
        ax.annotate(f'#{idx}', (dra_amin[idx], ddec_amin[idx]),
                    fontsize=7.5, color=circle_cols[k],
                    xytext=(4, 4), textcoords='offset points')

    ax.set_xlabel('ΔRA·cos(Dec)  [arcmin]  →W', fontsize=12)
    ax.set_ylabel('ΔDec  [arcmin]', fontsize=12)
    ax.set_title(
        f'Moon trajectory — {args.scan}\n'
        f'Tracked phase centre  RA={field_ra:.3f}°  Dec={field_dec:.3f}°  '
        f'(image ±{hw:.1f}\')',
        fontsize=10
    )
    ax.set_xlim(-hw * 1.25,  hw * 1.25)
    ax.set_ylim(-hw * 1.25,  hw * 1.25)
    ax.set_aspect('equal')
    ax.invert_xaxis()   # East to the left (standard radio convention)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3, linestyle=':')

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=args.dpi)
    plt.close(fig)
    print(f'[trajectory] Saved: {out_path}')


if __name__ == '__main__':
    main()
