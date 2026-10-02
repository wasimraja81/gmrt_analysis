#!/usr/bin/env python
"""Read-only EXPERIMENT (no header manipulation): run stock CASA importuvfits on
the RAW MULTI-SOURCE gvfits UVFITS with PMRA/PMDEC LEFT INTACT, and report
exactly what happens. Starts from the untouched gvfits output -- NOT any split,
calibrated, flagged, or hand-curated (PMRA-zeroed) product.

Question under test (user hypothesis):
  1. gvfits recorded UVW per standards; read WITH PMRA/PMDEC these should let a
     downstream app produce correct data.
  3. PMRA/PMDEC are essential for CASA to correctly GENERATE UVW for fast movers.

What this probe measures from the fresh stock MS (if import succeeds):
  (a) did importuvfits succeed or crash (full traceback captured)?
  (b) did CASA ingest PMRA -> FIELD::NUM_POLY, PHASE_DIR poly coeff-1,
      SOURCE::PROPER_MOTION?
  (c) for the MOON0520 field, what SKY DIRECTION do the stock-MS UVW encode
      (certified Procrustes decoder) -- the true apparent Moon
      (331.220,-17.385) or the offending ~101,-5?

If import crashes, that itself answers whether the raw file "just works" in a
stock downstream app. The MS is written under the repo tree; kept on success so
it can be inspected further, removed on crash. Nothing else is modified.

Usage:
    gmrt/bin/python tools/dev/probe_clean_importuvfits.py
"""
import os
import sys
import shutil
import traceback
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import moon_uvw_checks as mv                          # certified decoder

RAW = "/Users/raj030/DATA/gmrt_40_014/data/40_014_25jul2021_gsb.FITS"
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUTDIR = os.path.join(REPO, "work", "_stock_import_probe")
MS = os.path.join(OUTDIR, "40_014_stock.ms")

TRUE_MOON = (331.220, -17.385)      # SU RAAPP/DECAPP for moon0520
OFFENDING = (101.3, -5.5)           # recovered offending direction (MV-07/09)
FIELD_NAME = "MOON0520"


def sep_deg(ra1, dec1, ra2, dec2):
    r1, d1, r2, d2 = map(np.radians, (ra1, dec1, ra2, dec2))
    c = np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(r1 - r2)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def show_field_source(ms):
    from casatools import table
    tb = table()
    tb.open(ms + "/FIELD", nomodify=True)
    names = list(tb.getcol("NAME"))
    npoly = np.asarray(tb.getcol("NUM_POLY"))
    pd = tb.getvarcol("PHASE_DIR")                     # var-shaped per field
    try:
        ref = tb.getcolkeyword("PHASE_DIR", "MEASINFO").get("Ref", "?")
    except Exception:
        ref = "?"
    tb.close()
    print(f"    FIELD names          = {[n.strip() for n in names]}")
    print(f"    FIELD::NUM_POLY      = {npoly.tolist()}   ref={ref}")
    for k, nm in enumerate(names):
        arr = np.asarray(pd[f"r{k+1}"])                # (2, ncoeff)
        c0 = np.degrees(arr[:, 0])
        line = f"      [{k}] {nm.strip():12s} coeff0 RA {c0[0] % 360:.4f} Dec {c0[1]:.4f}"
        if arr.shape[1] > 1:
            c1 = arr[:, 1]
            line += f"  coeff1(rate) dRA {c1[0]:.3e} dDec {c1[1]:.3e} rad/s"
        print(line)
    try:
        tb.open(ms + "/SOURCE", nomodify=True)
        snames = [s.strip() for s in tb.getcol("NAME")]
        if "PROPER_MOTION" in tb.colnames():
            pm = np.asarray(tb.getcol("PROPER_MOTION"))   # (2, nrow)
            for i, s in enumerate(snames):
                if abs(pm[0, i]) > 0 or abs(pm[1, i]) > 0 or "MOON" in s.upper():
                    print(f"    SOURCE {s:12s} PROPER_MOTION (rad/s) = "
                          f"[{pm[0, i]:.3e}, {pm[1, i]:.3e}]")
        tb.close()
    except Exception as e:
        print(f"    SOURCE table: {e}")
    return [n.strip() for n in names]


def decode_field(ms, field_id):
    from casatools import table
    tb = table()
    tb.open(ms, nomodify=True)
    sub = tb.query(f"FIELD_ID=={field_id} && ANTENNA1!=ANTENNA2")
    uvw = np.asarray(sub.getcol("UVW"), "f8")          # (3, nrow) metres
    a1 = np.asarray(sub.getcol("ANTENNA1"), int)
    a2 = np.asarray(sub.getcol("ANTENNA2"), int)
    tsec = np.asarray(sub.getcol("TIME"), "f8")
    sub.close()
    tb.close()

    tb.open(ms + "/ANTENNA", nomodify=True)
    pos = np.asarray(tb.getcol("POSITION"), "f8").T
    tb.close()

    from astropy.time import Time
    import astropy.units as u
    rec = uvw.T
    B = pos[a2] - pos[a1]
    jd = tsec / 86400.0 + 2400000.5
    gast = Time(jd, format="jd", scale="utc").sidereal_time(
        "apparent", "greenwich").to(u.rad).value
    lon = np.arctan2(pos[:, 1].mean(), pos[:, 0].mean())   # ITRF site lon
    pra, pdec, pres = mv._pointing_per_integration(rec, B, jd, gast, lon)
    return pra, pdec, pres, len(np.unique(tsec)), np.degrees(lon)


def main():
    print("=" * 78)
    print("STOCK importuvfits on RAW MULTI-SOURCE gvfits UVFITS -- PMRA UNTOUCHED")
    print("=" * 78)
    print(f"    raw file: {RAW}")
    print(f"    -> MS   : {MS}\n")

    if os.path.isdir(OUTDIR):
        shutil.rmtree(OUTDIR)
    os.makedirs(OUTDIR, exist_ok=True)

    from casatasks import importuvfits
    ok = False
    try:
        importuvfits(fitsfile=RAW, vis=MS)
        ok = os.path.isdir(MS)
        print(f"\n(a) importuvfits RESULT: {'SUCCESS' if ok else 'no MS produced'}\n")
    except Exception as e:
        print("\n(a) importuvfits RAISED:")
        print(f"      {type(e).__name__}: {e}")
        print("    ---- traceback ----")
        traceback.print_exc()
        print("    -------------------")
        print("    => Stock CASA could NOT ingest the RAW gvfits file with PMRA "
              "intact.\n")
        shutil.rmtree(OUTDIR, ignore_errors=True)
        return

    print("(b) Rate metadata carried into the MS (PMRA -> FIELD/SOURCE)?")
    names = show_field_source(MS)
    print()

    if FIELD_NAME not in names:
        print(f"    {FIELD_NAME} not found among fields; stopping.")
        return
    fid = names.index(FIELD_NAME)
    print(f"(c) Direction encoded by STOCK-MS {FIELD_NAME} (field {fid}) UVW:")
    pra, pdec, pres, nint, lon_site = decode_field(MS, fid)
    ra_un = np.unwrap(np.radians(pra))
    ra_mean = np.degrees(np.mean(ra_un)) % 360.0
    dec_mean = float(np.mean(pdec))
    s_moon = sep_deg(ra_mean, dec_mean, *TRUE_MOON)
    s_off = sep_deg(ra_mean, dec_mean, *OFFENDING)
    verdict = ("ON-MOON (user hypothesis CONFIRMED)" if s_moon < 0.5
               else "OFFENDING ~101,-5 (hypothesis REFUTED)" if s_off < 1.5
               else "OTHER")
    print(f"    RA={ra_mean:.3f}+-{np.degrees(np.std(ra_un)):.3f}  "
          f"Dec={dec_mean:.3f}+-{np.std(pdec):.3f}  "
          f"(per-integ RMS {np.median(pres):.1f} m, {nint} integ, "
          f"ITRF lon {lon_site:.3f})")
    print(f"    sep TRUE MOON {TRUE_MOON} = {s_moon:.2f} deg")
    print(f"    sep OFFENDING {OFFENDING} = {s_off:.2f} deg")
    print(f"    -> {verdict}")
    print(f"\n    MS kept at {MS} for further inspection.")


if __name__ == "__main__":
    main()
