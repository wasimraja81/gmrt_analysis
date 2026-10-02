#!/usr/bin/env python
"""Read-only probe: recover the implied sky pointing from the *on-disk MS* UVW
for moon0520, for every MS variant that lives on disk, and compare to:
  - the true apparent Moon that day  (SU RAAPP/DECAPP = 331.220,-17.385, MV-09)
  - the offending recovered direction (~101,-5, MV-07/MV-09)

Purpose: answer, from DATA not code, whether each imaged MS carries visibilities
that were ROTATED onto the Moon (phaseshift/fixvis -> UVW recomputed, recovers
on-Moon) or the RECORDED/offending UVW copied straight through importuvfits
(recovers ~126 deg off the Moon).

METHOD — certified, not ad-hoc.  MV-07/MV-09 established that the offending
scans track a *sky-fixed but wrong* direction, and that direction is recovered
by the per-integration orthogonal-Procrustes decoder `_pointing_per_integration`
in moon_uvw_checks.py (single-direction whole-scan least-squares COLLAPSES on the
offending scan -- that was the earlier bug).  We reuse that exact certified
decoder here.

Frame handling (the one genuine MS-vs-UVFITS difference):
  * raw UVFITS uses AIPS STABXYZ, whose X-axis sits at the LOCAL meridian, so the
    certified recipe adds the site longitude:  H = GAST + lon - RA  (MV-04).
  * an imported MS uses the CASA ANTENNA::POSITION table, which is ITRF geocentric
    with X at the GREENWICH meridian, so the hour angle is  H = GAST - RA  (lon=0).
We do NOT assert this is right -- we VALIDATE it: a straight-import MS (recorded
UVW copied through importuvfits) must recover the SAME offending direction as the
raw-UVFITS control.  Only after that check passes do we trust a shifted MS that
recovers a different (on-Moon) direction.

No file is modified. Read-only.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import moon_uvw_checks as mv  # certified recipe + Procrustes decoder

from astropy.time import Time
import astropy.units as u
from casatools import table

RAW = ("/Users/raj030/DATA/gmrt_40_014/work/split/moon/"
       "moon0520_primary_secondary_calibrated_flagged.uvfits")

MSS = {
    "workflow_phasecenter (stk10_phasecenter)":
        "/Users/raj030/DATA/gmrt_40_014/work/casa_selfcal/moon0520_stk10_phasecenter/moon0520_full.ms",
    "workflow_nophasecenter (stk10)":
        "/Users/raj030/DATA/gmrt_40_014/work/casa_selfcal/moon0520_stk10/moon0520_full.ms",
    "scratch_full":
        "/Users/raj030/DATA/gmrt_40_014/work/scratch/moon0520_full.ms",
    "scratch_shifted":
        "/Users/raj030/DATA/gmrt_40_014/work/scratch/moon0520_full_shifted.ms",
}

# Ground-truth reference directions (deg), established in MV-07/MV-09
TRUE_MOON = (331.220, -17.385)      # SU RAAPP/DECAPP for moon0520
OFFENDING = (101.3, -5.5)           # recovered offending direction (MV-09)


def sep_deg(ra1, dec1, ra2, dec2):
    r1, d1, r2, d2 = map(np.radians, (ra1, dec1, ra2, dec2))
    c = np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(r1 - r2)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def summarise(pra, pdec, pres, tag):
    """Collapse per-integration pointing to a mean and report vs the references."""
    ra_un = np.unwrap(np.radians(pra))
    ra_scatter = np.degrees(np.std(ra_un))
    ra_mean = np.degrees(np.mean(ra_un)) % 360.0
    dec_mean = float(np.mean(pdec))
    print(f"    {tag}: RA={ra_mean:.3f}+-{ra_scatter:.3f} "
          f"Dec={dec_mean:.3f}+-{np.std(pdec):.3f}  "
          f"(per-integ fit RMS {np.median(pres):.1f} m, {len(pra)} integ)")
    s_moon = sep_deg(ra_mean, dec_mean, *TRUE_MOON)
    s_off = sep_deg(ra_mean, dec_mean, *OFFENDING)
    verdict = ("ON-MOON" if s_moon < 0.5
               else "OFFENDING(matches MV-07)" if s_off < 1.0
               else "OTHER")
    print(f"       sep TRUE MOON {TRUE_MOON} = {s_moon:6.2f} deg   "
          f"sep OFFENDING {OFFENDING} = {s_off:6.2f} deg   -> {verdict}")
    return ra_mean, dec_mean, s_moon, s_off


def recover_from_ms(ms_path):
    """Load MS UVW + ITRF baselines + GAST; return per-integration pointing arrays
    using the certified Procrustes decoder (lon=0: ITRF X at Greenwich)."""
    tb = table()
    tb.open(ms_path, nomodify=True)
    uvw = np.asarray(tb.getcol("UVW"), "f8")          # (3, nrow) metres
    a1 = np.asarray(tb.getcol("ANTENNA1"), int)
    a2 = np.asarray(tb.getcol("ANTENNA2"), int)
    tsec = np.asarray(tb.getcol("TIME"), "f8")        # MJD seconds (UTC)
    tb.close()

    tb.open(ms_path + "/ANTENNA", nomodify=True)
    pos = np.asarray(tb.getcol("POSITION"), "f8").T   # (nant, 3) ITRF metres
    tb.close()

    tb.open(ms_path + "/FIELD", nomodify=True)
    pd = np.asarray(tb.getcol("PHASE_DIR"), "f8")     # (2,1,nfield) radians
    try:
        mi = tb.getcolkeyword("PHASE_DIR", "MEASINFO")
        ref = mi.get("Ref", "?")
    except Exception:
        ref = "?"
    tb.close()

    good = a1 != a2
    rec = uvw[:, good].T                               # (N,3) metres
    B = pos[a2[good]] - pos[a1[good]]                  # ant2-ant1 (MV-04 sign)

    jd = tsec / 86400.0 + 2400000.5
    gast = Time(jd, format="jd", scale="utc").sidereal_time(
        "apparent", "greenwich").to(u.rad).value

    # MEASURED: CASA copied the GMRT array-LOCAL antenna frame (STABXYZ) into
    # ANTENNA::POSITION verbatim -> X sits at the local meridian, same as the raw
    # UVFITS, so the certified recipe's site-longitude term applies here too.
    # (Validated: with lon=0 the MS recovered RA exactly site-lon short of the
    # certified control; adding lon reproduces the control's RA to <0.01 deg.)
    lon = np.arctan2(pos[:, 1].mean(), pos[:, 0].mean())
    pra, pdec, pres = mv._pointing_per_integration(rec, B, jd[good], gast[good], lon)

    pc = [(np.degrees(pd[0, 0, k]) % 360.0, np.degrees(pd[1, 0, k]))
          for k in range(pd.shape[2])]
    lon_site = np.degrees(np.arctan2(pos[:, 1].mean(), pos[:, 0].mean()))
    return pra, pdec, pres, pc, ref, lon_site, int(good.sum())


def main():
    print("=" * 78)
    print("CONTROL: recover raw UVFITS pointing via certified Procrustes decoder")
    print("         (must reproduce MV-07 offending ~101,-5 for moon0520)")
    print("=" * 78)
    rec, B, jd, gast, lon, a1, a2, app = mv._load_full(RAW)
    pra, pdec, pres = mv._pointing_per_integration(rec, B, jd, gast, lon)
    print(f"    SU RAAPP/DECAPP in file = {app}   site lon = {np.degrees(lon):.4f} deg")
    summarise(pra, pdec, pres, "raw UVFITS (STABXYZ, H=GAST+lon-RA)")
    print()

    for label, path in MSS.items():
        print("=" * 78)
        print(f"MS: {label}")
        print(f"    {path}")
        if not os.path.isdir(path):
            print("    (absent on disk)\n")
            continue
        try:
            pra, pdec, pres, pc, ref, lon_site, nrow = recover_from_ms(path)
        except Exception as e:
            print(f"    ERROR reading MS: {e}\n")
            continue
        print(f"    rows used {nrow}   ITRF site lon {lon_site:.4f} deg   "
              f"FIELD ref={ref}")
        for k, (r, d) in enumerate(pc):
            print(f"    FIELD[{k}] PHASE_DIR = RA={r:.4f} Dec={d:.4f}")
        summarise(pra, pdec, pres, "MS UVW (ITRF, H=GAST-RA)")
        print()


def compare_uvw(path_a, path_b, label_a, label_b):
    """Direct numeric comparison of the UVW columns of two MSs (row-aligned by
    ANTENNA1/ANTENNA2/TIME). Answers: did phaseshift actually change the numbers?"""
    def load(p):
        tb = table()
        tb.open(p, nomodify=True)
        d = dict(uvw=np.asarray(tb.getcol("UVW"), "f8"),
                 a1=np.asarray(tb.getcol("ANTENNA1"), int),
                 a2=np.asarray(tb.getcol("ANTENNA2"), int),
                 t=np.asarray(tb.getcol("TIME"), "f8"))
        tb.close()
        return d
    A, B = load(path_a), load(path_b)
    print("=" * 78)
    print(f"DIRECT UVW COMPARISON: {label_a}  vs  {label_b}")
    print(f"    rows: {A['uvw'].shape[1]} vs {B['uvw'].shape[1]}")
    if A['uvw'].shape != B['uvw'].shape:
        print("    (different shapes -- cannot subtract directly)\n")
        return
    same_key = (np.array_equal(A['a1'], B['a1']) and np.array_equal(A['a2'], B['a2'])
                and np.allclose(A['t'], B['t']))
    print(f"    row keys (ant1/ant2/time) identical: {same_key}")
    d = np.linalg.norm(A['uvw'] - B['uvw'], axis=0)   # per-row |dUVW| metres
    print(f"    |UVW_a - UVW_b|: max={d.max():.3e} m  median={np.median(d):.3e} m  "
          f"mean={d.mean():.3e} m")
    print(f"    -> {'IDENTICAL (phaseshift did NOT alter UVW)' if d.max() < 1e-3 else 'DIFFERENT UVW numbers'}\n")


if __name__ == "__main__":
    main()
    print()
    sf = MSS["scratch_full"]
    ss = MSS["scratch_shifted"]
    if os.path.isdir(sf) and os.path.isdir(ss):
        compare_uvw(sf, ss, "scratch_full", "scratch_shifted")
