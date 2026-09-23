#!/usr/bin/env python3
"""Smoke test: tclean(phasecenter='MOON') vs phaseshift-to-CRVAL + tclean.

Strategy:
  A) tclean(phasecenter='MOON')  -> exports FITS -> extract CRVAL1/CRVAL2
  B) phaseshift(vis, phasecenter=CRVAL_string) + tclean(phasecenter='')
  Compare pixel maps of A and B.  Pass if rms(A-B)/mean_rms < 1% and corr > 0.999.
"""
from __future__ import annotations
import shutil
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.time import Time
from casatasks import exportfits, phaseshift, split, tclean
from casatools import table as tb_tool

MS      = Path("casa_out/moon/moon0520.ms")
OUTDIR  = Path("casa_out/smoketest_moon_phaseshift")
CELL    = "4arcsec"
IMSIZE  = [512, 512]


def cleanup(p):
    if p.is_dir():
        shutil.rmtree(p, ignore_errors=True)
    elif p.exists():
        p.unlink()


def cleanup_prefix(prefix):
    for p in prefix.parent.glob(prefix.name + ".*"):
        cleanup(p)


# ── pick the middle integration ──────────────────────────────────────────────
OUTDIR.mkdir(parents=True, exist_ok=True)
tb = tb_tool()
tb.open(str(MS))
times = np.unique(np.round(np.asarray(tb.getcol("TIME"), dtype=float), 3))
tb.close()

int_idx  = len(times) // 2
cadence  = float(np.median(np.diff(times)))
t_mid    = float(times[int_idx])
half     = max(0.1, 0.45 * cadence)
t0 = Time((t_mid - half) / 86400.0, format="mjd", scale="utc")
t1 = Time((t_mid + half) / 86400.0, format="mjd", scale="utc")
timerange = (t0.to_datetime().strftime("%Y/%m/%d/%H:%M:%S.%f")[:-3]
             + "~" + t1.to_datetime().strftime("%Y/%m/%d/%H:%M:%S.%f")[:-3])
print(f"Integration {int_idx+1}/{len(times)}: {timerange}")

int_ms = OUTDIR / "int_native.ms"
cleanup(int_ms)
split(vis=str(MS), outputvis=str(int_ms), datacolumn="data",
      timerange=timerange, keepflags=False)

# ── Step 1: tclean(phasecenter='MOON') ───────────────────────────────────────
img_A = OUTDIR / "img_A_tclean_moon"
cleanup_prefix(img_A)
tclean(vis=str(int_ms), imagename=str(img_A), phasecenter="MOON",
       imsize=IMSIZE, cell=CELL, specmode="mfs", deconvolver="hogbom",
       niter=0, weighting="briggs", robust=0.0, stokes="I",
       gridder="standard", pblimit=-1, pbcor=False, interactive=False)
fits_A = str(img_A) + ".image.fits"
exportfits(imagename=str(img_A) + ".image", fitsimage=fits_A, overwrite=True)

with fits.open(fits_A) as hA:
    hdr_A  = hA[0].header
    data_A = hA[0].data.squeeze().copy()
crval1 = float(hdr_A["CRVAL1"])
crval2 = float(hdr_A["CRVAL2"])

# Convert CRVAL1/CRVAL2 (degrees) to a CASA direction string
ra_deg = crval1 % 360.0
ra_h   = int(ra_deg / 15.0)
ra_m   = int(((ra_deg / 15.0) % 1) * 60.0)
ra_s   = ((ra_deg / 15.0) % 1 * 60.0) % 1 * 60.0
sign   = "+" if crval2 >= 0 else "-"
adec   = abs(crval2)
dec_d  = int(adec)
dec_m  = int((adec % 1) * 60.0)
dec_s  = (adec % 1 * 60.0) % 1 * 60.0
pc_str = "J2000 {:02d}:{:02d}:{:09.6f} {:s}{:02d}.{:02d}.{:07.4f}".format(
    ra_h, ra_m, ra_s, sign, dec_d, dec_m, dec_s)

print("\nStep 1 - tclean(phasecenter=MOON) image center:")
print("  CRVAL1={:.6f} deg  CRVAL2={:.6f} deg".format(crval1, crval2))
print("  Direction string: {}".format(pc_str))

# ── Step 2: phaseshift to that EXACT position + tclean ───────────────────────
shifted_ms = OUTDIR / "int_phaseshift_to_tclean_pos.ms"
cleanup(shifted_ms)
phaseshift(vis=str(int_ms), outputvis=str(shifted_ms), phasecenter=pc_str)

img_B = OUTDIR / "img_B_phaseshift_to_tclean_pos"
cleanup_prefix(img_B)
tclean(vis=str(shifted_ms), imagename=str(img_B), phasecenter="",
       imsize=IMSIZE, cell=CELL, specmode="mfs", deconvolver="hogbom",
       niter=0, weighting="briggs", robust=0.0, stokes="I",
       gridder="standard", pblimit=-1, pbcor=False, interactive=False)
fits_B = str(img_B) + ".image.fits"
exportfits(imagename=str(img_B) + ".image", fitsimage=fits_B, overwrite=True)

with fits.open(fits_B) as hB:
    data_B = hB[0].data.squeeze().copy()

# ── Step 3: pixel comparison ──────────────────────────────────────────────────
residual = data_A - data_B
rms_A    = float(np.nanstd(data_A))
rms_B    = float(np.nanstd(data_B))
rms_diff = float(np.nanstd(residual))
max_diff = float(np.nanmax(np.abs(residual)))
frac     = rms_diff / ((rms_A + rms_B) / 2)
corr     = float(np.corrcoef(data_A.ravel(), data_B.ravel())[0, 1])

print()
print("== Pixel comparison: img_A (tclean MOON) vs img_B (phaseshift+tclean) ==")
print("  Image A rms       : {:.4f} mJy/beam".format(rms_A * 1e3))
print("  Image B rms       : {:.4f} mJy/beam".format(rms_B * 1e3))
print("  (A-B) rms         : {:.4f} mJy/beam  ({:.2f}% of mean rms)".format(
    rms_diff * 1e3, frac * 100))
print("  max |A-B|         : {:.4f} mJy/beam".format(max_diff * 1e3))
print("  pixel correlation : {:.6f}".format(corr))
print()
if frac < 0.01 and corr > 0.999:
    print("PASS  (<1% residual, r>0.999)")
    print("phaseshift to tclean MOON pos replicates tclean(phasecenter=MOON)")
elif frac < 0.05:
    print("CLOSE ({:.1f}% residual, r={:.4f})".format(frac * 100, corr))
else:
    print("FAIL  ({:.1f}% residual, r={:.4f})".format(frac * 100, corr))
print()
print("FITS A: {}".format(fits_A))
print("FITS B: {}".format(fits_B))
