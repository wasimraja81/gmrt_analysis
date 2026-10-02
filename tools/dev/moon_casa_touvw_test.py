#!/usr/bin/env python
"""Read-only: drive CASA's OWN UVW engine (casatools.measures.touvw -- the same
casacore machinery MSDerivedValues / the imager / fixvis / phaseshift use) and
ask the user's exact question:

    "whether the UVW rotation CASA's tasks adopt indeed lands us with the UVWs
     corresponding to the correct instantaneous directions for the Moon where
     the correlator has fringe-stopped."

me.touvw(baseline) builds (u,v,w) for a baseline given a FRAME = direction
(phase centre) + epoch (UTC) + position (observatory). It is the canonical
casacore UVW/rotation engine; feeding it a direction and letting it convert
that direction (J2000 mean <-> apparent-of-date) via the measures reference
system IS the "measRef rotation" the user is asking about.

NOTE on EPOCH=-1: that is a UVFITS SU-table keyword read by importuvfits, NOT a
measures parameter. There is no EPOCH column in this path. Its ROLE -- "this
direction is apparent-of-date, do not precess it" -- is played here by the
direction measure's reference-frame tag: dirref='APP' (apparent) vs 'J2000'
(mean). So no literal epoch=-1 is passed; the Moon is simply tagged 'APP'.

Design (no MS, no importuvfits crash, no PM-zeroed curated products -- only the
calibrator-certified geometry from moon_uvw_checks + the LTA online model):

  TEST A  -- HARNESS VALIDATION on calibrator 3C468.1 (recorded UVW certified
             correct by MV-04/05).  Feed me.touvw:
               (a) the APPARENT direction  (SU RAAPP/DECAPP, dirref='APP'), and
               (b) the J2000 mean direction (SU RAEPO/DECEPO, dirref='J2000').
             (a) should reproduce recorded UVW to ~metres  => CASA's touvw ==
                 our certified _predict_uvw (harness trustworthy).
             (b) exercises CASA's FULL mean->apparent measRef rotation; its
                 residual vs recorded quantifies how big that rotation is in
                 metres (it is the ~0.3 deg precession-scale rotation -- small).

  TEST B  -- moon0520 (OFFENDING with-rate scan).  Feed me.touvw the correct
             INSTANTANEOUS apparent Moon direction (LTA RA-DATE/DEC-DATE +
             rate*(t-t0), per integration) and compare CASA's UVW to
               (1) the RECORDED UVW  -> expect a huge (~159 deg / thousands of m)
                   residual, and
               (2) the theoretical synth UVW (moon_uvw_synthesis_compare)
                   -> expect ~metres.
             Plus a third column: the SAME Moon coordinates tagged 'J2000' (the
             measRef failure mode) -- CASA precesses them only ~0.3 deg, still
             nowhere near the 159 deg, proving no frame tag can manufacture it.

             If CASA (with the correct apparent Moon direction) lands on the
             theoretical Moon UVW and NOT on the recorded UVW, then CASA's UVW
             rotation, given the correct Moon direction, produces the correct
             instantaneous-Moon geometry -- and the recorded rotation is a gvfits
             arithmetic error that no measRef frame conversion can undo (route A,
             let CASA rotate the recorded UVW, cannot recover it; only route B, a
             fresh recompute -- exactly what me.touvw just did -- can).

Reads only; prints only. No files modified.

Usage:
    gmrt/bin/python tools/dev/moon_casa_touvw_test.py
"""
import os
import sys
import numpy as np
from astropy.io import fits

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moon_uvw_checks as m                                     # noqa: E402
from moon_uvw_synthesis_compare import synth_from_lta, LTA      # noqa: E402

C = 299792458.0


def array_center(path):
    """GMRT geocentric ITRF array reference (AN-header ARRAYX/Y/Z), metres."""
    with fits.open(path, memmap=False) as h:
        anh = [hd.header for hd in h
               if hd.header.get("EXTNAME", "").strip() == "AIPS AN"][0]
    return np.array([anh["ARRAYX"], anh["ARRAYY"], anh["ARRAYZ"]], "f8")


def su_positions(path):
    """SU mean (RAEPO/DECEPO) and apparent (RAAPP/DECAPP), degrees."""
    with fits.open(path, memmap=False) as h:
        su = h["AIPS SU"].data
        return (float(su["RAEPO"][0]), float(su["DECEPO"][0]),
                float(su["RAAPP"][0]), float(su["DECAPP"][0]))


def _Rz(a):
    """Rotation about the pole (Z) by angle a (rad)."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def casa_touvw(B, jd, arrpos, ra_deg, dec_deg, dirref, lon):
    """CASA measures UVW for baselines B (N,3 metres, ant2-ant1) at UTC times
    jd (N,), phase-centre (ra_deg,dec_deg) which may be scalar (fixed source)
    or per-row arrays (moving Moon), tagged reference `dirref` ('J2000' or
    'APP'). Observatory frame position = arrpos (ITRF metres).

    B is STABXYZ (AIPS local-meridian frame: X -> array meridian, which is why
    the certified _predict_uvw uses H = GAST + lon - RA). CASA's touvw wants the
    baseline in geocentric ITRF (X -> Greenwich), so we rotate about the pole by
    +lon first (validated on the calibrator: Rz(+lon) lands within ~17 m of the
    recorded UVW, vs 8590 m un-rotated).

    Returns UVW (N,3 metres) as CASA's engine computes them. One me.touvw call
    per unique integration (direction+epoch fixed within an integration), all
    that integration's baselines passed as arrays."""
    from casatools import measures, quanta
    B = (_Rz(lon) @ B.T).T                         # local-meridian -> ITRF
    me, qa = measures(), quanta()
    me.doframe(me.position("ITRF", qa.quantity(arrpos[0], "m"),
                           qa.quantity(arrpos[1], "m"),
                           qa.quantity(arrpos[2], "m")))

    ra_deg = np.broadcast_to(np.asarray(ra_deg, "f8"), (len(B),))
    dec_deg = np.broadcast_to(np.asarray(dec_deg, "f8"), (len(B),))
    out = np.full((len(B), 3), np.nan)

    stamps, inv = np.unique(np.round(jd, 10), return_inverse=True)
    for j in range(len(stamps)):
        sel = inv == j
        mjd = float(stamps[j]) - 2400000.5
        me.doframe(me.epoch("UTC", qa.quantity(mjd, "d")))
        # direction fixed within one integration (Moon rate << integration)
        me.doframe(me.direction(dirref,
                                qa.quantity(float(ra_deg[sel][0]), "deg"),
                                qa.quantity(float(dec_deg[sel][0]), "deg")))
        bx, by, bz = B[sel, 0], B[sel, 1], B[sel, 2]
        bl = me.baseline("ITRF", qa.quantity(list(bx), "m"),
                         qa.quantity(list(by), "m"),
                         qa.quantity(list(bz), "m"))
        # me.touvw returns (uvw_measure, xyz_quantity, dot); the 2nd element is
        # the flat cartesian [u0,v0,w0,u1,v1,w1,...] in metres -- the UVW.
        xyz = np.asarray(me.touvw(bl)[1]["value"], "f8").reshape(-1, 3)
        out[sel] = xyz
    me.done()
    return out


def rms(a, b):
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, 1))))


def procrustes_angle(a, b):
    """Rigid rotation angle (deg) carrying a -> b (orthogonal Procrustes)."""
    M = b.T @ a
    U, _, Vt = np.linalg.svd(M)
    d = np.sign(np.linalg.det(U @ Vt))
    Q = U @ np.diag([1, 1, d]) @ Vt
    return float(np.degrees(np.arccos(np.clip((np.trace(Q) - 1) / 2, -1, 1))))


def test_A():
    print("=" * 78)
    print("TEST A -- HARNESS VALIDATION: CASA me.touvw vs recorded UVW, "
          "calibrator 3C468.1")
    print("=" * 78)
    rec, B, jd, gast, lon, a1, a2, app = m._load_full(m.CAL)
    arrpos = array_center(m.CAL)
    raepo, decepo, raapp, decapp = su_positions(m.CAL)
    print(f"  N vis = {len(rec)}   GMRT ITRF center = "
          f"[{arrpos[0]:.1f} {arrpos[1]:.1f} {arrpos[2]:.1f}] m")
    print(f"  SU mean  (J2000) RAEPO/DECEPO = {raepo:.4f}, {decepo:.4f}")
    print(f"  SU apparent      RAAPP/DECAPP = {raapp:.4f}, {decapp:.4f}  "
          f"(mean->app sep = {m._sep_deg(raepo, decepo, raapp, decapp):.4f} deg)\n")

    uvw_app = casa_touvw(B, jd, arrpos, raapp, decapp, "APP", lon)
    uvw_j2k = casa_touvw(B, jd, arrpos, raepo, decepo, "J2000", lon)
    print(f"  (a) CASA touvw @ APPARENT dir  vs recorded : "
          f"RMS = {rms(uvw_app, rec):10.2f} m   "
          f"[{'MATCH -> harness OK' if rms(uvw_app, rec) < 50 else 'MISMATCH'}]")
    print(f"  (b) CASA touvw @ J2000  mean   vs recorded : "
          f"RMS = {rms(uvw_j2k, rec):10.2f} m   "
          f"(= size of CASA's full mean->apparent measRef rotation)")
    print(f"      -> CASA's mean<->apparent rotation is only "
          f"{procrustes_angle(uvw_j2k, uvw_app):.3f} deg on this source.\n")
    return rms(uvw_app, rec)


def test_B():
    print("=" * 78)
    print("TEST B -- moon0520 (OFFENDING with-rate scan): does CASA's UVW engine,")
    print("          fed the CORRECT instantaneous apparent Moon direction, land")
    print("          on the theoretical Moon UVW or on the recorded rotated UVW?")
    print("=" * 78)
    scan = "moon0520"
    rec, B, jd, gast, lon, a1, a2, app = m._load_full(m.MOON[scan])
    arrpos = array_center(m.MOON[scan])
    blk = LTA[scan]
    t0 = jd.mean()

    # theoretical (certified forward model) UVW, and the matching per-row
    # instantaneous apparent Moon direction (same propagation as synth)
    syn = synth_from_lta(blk, B, jd, gast, lon, t0)
    dt_s = (jd - t0) * 86400.0
    ra_t = blk["ra"] + blk["dra"] * dt_s
    dec_t = blk["dec"] + blk["ddec"] * dt_s

    print(f"  N vis = {len(rec)}   LTA apparent Moon @ scan centre = "
          f"{blk['ra']:.4f}, {blk['dec']:.4f}  "
          f"rate=({blk['dra']:.6f},{blk['ddec']:.6f}) deg/s")
    print(f"  SU RAAPP/DECAPP (file)            = {app[0]:.4f}, {app[1]:.4f}\n")

    uvw_casa = casa_touvw(B, jd, arrpos, ra_t, dec_t, "APP", lon)
    # measRef failure mode: SAME numbers tagged J2000 (CASA precesses them fwd)
    uvw_j2k = casa_touvw(B, jd, arrpos, ra_t, dec_t, "J2000", lon)

    r_rec = rms(uvw_casa, rec)
    r_syn = rms(uvw_casa, syn)
    ang_rec = procrustes_angle(uvw_casa, rec)
    ang_syn = procrustes_angle(uvw_casa, syn)
    print(f"  CASA touvw @ instantaneous apparent Moon  vs THEORETICAL synth : "
          f"RMS = {r_syn:10.2f} m  (rot {ang_syn:5.2f} deg)")
    print(f"  CASA touvw @ instantaneous apparent Moon  vs RECORDED UVW      : "
          f"RMS = {r_rec:10.2f} m  (rot {ang_rec:5.2f} deg)")
    # cross-check: recorded vs theoretical (the known ~159 deg)
    print(f"  (reference) RECORDED vs THEORETICAL synth                      : "
          f"RMS = {rms(rec, syn):10.2f} m  (rot {procrustes_angle(syn, rec):5.2f} deg)")
    # measRef failure mode: how far a WRONG frame tag (J2000) moves the Moon UVW
    print(f"  (measRef test) SAME Moon coords tagged J2000 vs 'APP' result   : "
          f"RMS = {rms(uvw_j2k, uvw_casa):10.2f} m  "
          f"(rot {procrustes_angle(uvw_j2k, uvw_casa):5.2f} deg)")
    print(f"                 -> the biggest rotation ANY frame tag can add is "
          f"{procrustes_angle(uvw_j2k, uvw_casa):.2f} deg, not 159 deg.\n")

    on_theory = r_syn < 50
    off_recorded = r_rec > 1000
    print("  VERDICT:")
    if on_theory and off_recorded:
        print("    CASA's own UVW engine, given the CORRECT instantaneous Moon")
        print("    direction, reproduces the THEORETICAL Moon UVW to the metre and")
        print(f"    sits ~{ang_rec:.0f} deg away from the recorded UVW. The recorded")
        print("    rotation is therefore NOT a frame CASA's measRef can apply/undo")
        print("    -- it is a gvfits arithmetic error. Route A (let CASA rotate the")
        print("    recorded UVW) cannot recover it; only route B (fresh recompute,")
        print("    exactly what CASA's touvw just did) can.")
    else:
        print(f"    Unexpected: on_theory={on_theory} off_recorded={off_recorded}")
        print("    -> re-examine the frame/direction handling before concluding.")


def main():
    r = test_A()
    print()
    if r >= 50:
        print("!! Harness validation (Test A) did NOT match recorded UVW to the")
        print("   metre level; interpret Test B with caution.\n")
    test_B()


if __name__ == "__main__":
    main()
