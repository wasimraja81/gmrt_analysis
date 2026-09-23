#!/usr/bin/env bash
set -euo pipefail

# ONE TASK: compare derive vs no-shift stacking and print consistency metrics.

PYTHON_CMD="/Users/raj030/github-wasimraja81/gmrt_analysis/gmrt/bin/python"
STACK_SCRIPT="experimental/stack_moon_snapshots.py"
SELFCAL_DIR="/Users/raj030/DATA/gmrt_40_014/work/casa_selfcal/moon0520_stk10_phasecenter"
OUTROOT="/Users/raj030/DATA/gmrt_40_014/work/casa_selfcal/moon0520_stk10_stack_compare"

DERIVE_DIR="${OUTROOT}/derive_phasecorr"
NOSHIFT_DIR="${OUTROOT}/no_shift_direct_coadd"

DERIVE_FITS="${DERIVE_DIR}/moon0520_stack_derive_phasecorr.fits"
NOSHIFT_FITS="${NOSHIFT_DIR}/moon0520_stack_no_shift_direct_coadd.fits"
INPUT_PNG_DIR="${OUTROOT}/input_frames_png"

mkdir -p "${DERIVE_DIR}/provenance" "${NOSHIFT_DIR}/provenance"
mkdir -p "${INPUT_PNG_DIR}"

echo "[0/3] Exporting input FITS previews (PNG)"
"${PYTHON_CMD}" - <<'PY' "${SELFCAL_DIR}" "${INPUT_PNG_DIR}"
from pathlib import Path
import numpy as np
from astropy.io import fits

selfcal_dir = Path(__import__('sys').argv[1])
out_dir = Path(__import__('sys').argv[2])
out_dir.mkdir(parents=True, exist_ok=True)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception as exc:
    print(f'[warn] matplotlib unavailable, skipping input PNG export: {exc}')
    raise SystemExit(0)

paths = sorted(selfcal_dir.rglob('moon0520_stk*_n*_final.fits'))
if not paths:
    print('[warn] no input FITS found for PNG export')
    raise SystemExit(0)

images = []
for p in paths:
    with fits.open(p) as hdul:
        images.append(np.squeeze(hdul[0].data).astype(np.float64))

all_vals = np.concatenate([img[np.isfinite(img)].ravel() for img in images])
vmin = float(np.percentile(all_vals, 2.0))
vmax = float(np.percentile(all_vals, 99.8))

for p, img in zip(paths, images):
    out_png = out_dir / (p.stem + '.png')
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    im = ax.imshow(img, origin='lower', cmap='magma', vmin=vmin, vmax=vmax)
    ax.set_title(p.stem, fontsize=9)
    ax.set_xlabel('x pixel')
    ax.set_ylabel('y pixel')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(out_png, dpi=120, bbox_inches='tight')
    plt.close(fig)

print(f'[ok] wrote {len(paths)} input PNGs to {out_dir}')
print(f'[ok] common display scale: vmin={vmin:.6g}, vmax={vmax:.6g}')
PY

echo "[1/3] Running derive"
"${PYTHON_CMD}" "${STACK_SCRIPT}" \
  --selfcal-dir "${SELFCAL_DIR}" \
  --output "${DERIVE_FITS}" \
  --registration-mode derive \
  --registration-method phase-correlation \
  --upsample 10 \
  --method mean \
  --provenance-dir "${DERIVE_DIR}/provenance"

echo "[2/3] Running no-shift"
"${PYTHON_CMD}" "${STACK_SCRIPT}" \
  --selfcal-dir "${SELFCAL_DIR}" \
  --output "${NOSHIFT_FITS}" \
  --registration-mode none \
  --upsample 10 \
  --method mean \
  --provenance-dir "${NOSHIFT_DIR}/provenance"

DERIVE_LOG="$(ls -1t "${DERIVE_DIR}"/provenance/*.log | head -1)"

echo "[3/3] Comparing outputs"
echo "derive_log=${DERIVE_LOG}"
echo "derive_fits=${DERIVE_FITS}"
echo "noshift_fits=${NOSHIFT_FITS}"

"${PYTHON_CMD}" - <<'PY' "${DERIVE_LOG}" "${DERIVE_FITS}" "${NOSHIFT_FITS}"
import re
import sys
import numpy as np
from astropy.io import fits

derive_log, derive_fits, noshift_fits = sys.argv[1:4]

dr = []
dc = []
row_pat = re.compile(r'^\s*\d+\s+')
with open(derive_log, 'r', encoding='utf-8', errors='ignore') as handle:
    for line in handle:
        if not row_pat.match(line):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            dr.append(abs(float(parts[2])))
            dc.append(abs(float(parts[3])))
        except ValueError:
            continue

if dr:
    mean_dr = float(np.mean(dr))
    mean_dc = float(np.mean(dc))
    max_dr = float(np.max(dr))
    max_dc = float(np.max(dc))
else:
    mean_dr = mean_dc = max_dr = max_dc = float('nan')

with fits.open(derive_fits) as hdul:
    header = hdul[0].header
    a = np.squeeze(hdul[0].data).astype(np.float64)
with fits.open(noshift_fits) as hdul:
    b = np.squeeze(hdul[0].data).astype(np.float64)

pix_arcsec = float(abs(header.get('CDELT2', 0.0004166666666667)) * 3600.0)
pix_arcmin = pix_arcsec / 60.0

diff = a - b
rms_diff = float(np.sqrt(np.mean(diff**2)))
max_abs_diff = float(np.max(np.abs(diff)))
rms_ref = float(np.sqrt(np.mean(a**2)))
frac = rms_diff / (rms_ref + 1e-30)

consistent = (frac < 0.02) and (np.nanmax([max_dr, max_dc]) < 0.5)

print('--- shift summary ---')
print(f'scale: 1 pix = {pix_arcsec:.4f} arcsec = {pix_arcmin:.5f} arcmin')
print(f'mean|dr|={mean_dr:.5f} pix  ({mean_dr*pix_arcsec:.5f} arcsec, {mean_dr*pix_arcmin:.6f} arcmin)')
print(f'mean|dc|={mean_dc:.5f} pix  ({mean_dc*pix_arcsec:.5f} arcsec, {mean_dc*pix_arcmin:.6f} arcmin)')
print(f'max|dr| ={max_dr:.5f} pix  ({max_dr*pix_arcsec:.5f} arcsec, {max_dr*pix_arcmin:.6f} arcmin)')
print(f'max|dc| ={max_dc:.5f} pix  ({max_dc*pix_arcsec:.5f} arcsec, {max_dc*pix_arcmin:.6f} arcmin)')
print('--- image difference ---')
print('units: image intensity units from FITS (same as your map, typically Jy/beam), plus unitless fraction')
print(f'rms_diff={rms_diff:.6e}  max_abs_diff={max_abs_diff:.6e}  frac_rms_diff={frac:.6e}')
print('--- verdict ---')
print('CONSISTENT' if consistent else 'NOT_CONSISTENT')
PY

echo
echo "input_png_dir=${INPUT_PNG_DIR}"
