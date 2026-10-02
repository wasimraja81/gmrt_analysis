#!/usr/bin/env python
"""Read-only: DERIVE the 159 deg rigid UVW rotation of the offending Moon scans
from the UVFITS UVW convention + the two sky directions alone.

The recorded UVW are built by the standard convention M(H,dec) (TMS 4.1, exactly
moon_uvw_checks._predict_uvw):

    u = sinH.Bx + cosH.By
    v = -sin(dec)cosH.Bx + sin(dec)sinH.By + cos(dec).Bz
    w =  cos(dec)cosH.Bx - cos(dec)sinH.By + sin(dec).Bz

which factors as   M(H,dec) = A(dec) . Rz(-H)   with
    A(dec) = [[0, 1, 0], [-sin dec, 0, cos dec], [cos dec, 0, sin dec]]
    Rz(a)  = [[cos a, -sin a, 0], [sin a, cos a, 0], [0, 0, 1]].

If gvfits built the frame for a WRONG phase centre (RA2,Dec2) instead of the true
apparent Moon (RA1,Dec1), the rigid rotation carrying theoretical -> recorded is,
in closed form,

    Q = M(H2,Dec2) . M(H1,Dec1)^T = A(Dec2) . Rz(H1 - H2) . A(Dec1)^T
      = A(Dec2) . Rz(RA2 - RA1) . A(Dec1)^T          (H = GAST + lon - RA, so
                                                       GAST and lon CANCEL).

So Q depends only on the two sky directions -- not on the array location or the
time -- which is why one constant rotation fits the whole scan.

This script (1) measures the empirical rotation straight off the recorded UVW,
(2) recovers the wrong direction the recorded UVW point at, (3) confirms a single
fixed wrong direction actually fits the recorded UVW, (4) rebuilds Q from the
closed form using the two directions ALONE and checks it reproduces the measured
angle+axis, and (5) decomposes the 159 deg into the pointing swing (the W-axis
angle between the two directions) plus the line-of-sight roll of the U,V axes.

Reads only; prints only. No files modified.

Usage:
    gmrt/bin/python tools/dev/moon_uvw_rotation_derivation.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moon_uvw_checks as m                                # noqa: E402
from moon_uvw_synthesis_compare import synth_from_lta, LTA  # noqa: E402

C = 299792458.0


def A(dec):
    s, c = np.sin(dec), np.cos(dec)
    return np.array([[0.0, 1.0, 0.0], [-s, 0.0, c], [c, 0.0, s]])


def Rz(a):
    s, c = np.sin(a), np.cos(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def procrustes(rec, syn):
    """Proper rotation Q with rec ~= Q @ syn (row-wise)."""
    M = rec.T @ syn
    U, _, Vt = np.linalg.svd(M)
    d = np.sign(np.linalg.det(U @ Vt))
    Q = U @ np.diag([1, 1, d]) @ Vt
    resid = np.sqrt(np.mean(np.sum((syn @ Q.T - rec) ** 2, 1)))
    return Q, resid


def angle_axis(Q):
    ang = np.degrees(np.arccos(np.clip((np.trace(Q) - 1) / 2, -1, 1)))
    ax = np.array([Q[2, 1] - Q[1, 2], Q[0, 2] - Q[2, 0], Q[1, 0] - Q[0, 1]])
    n = np.linalg.norm(ax)
    return ang, (ax / n if n > 1e-9 else np.full(3, np.nan)), float(np.linalg.det(Q))


def sep_deg(ra1, d1, ra2, d2):
    ra1, d1, ra2, d2 = np.radians([ra1, d1, ra2, d2])
    cs = np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(ra1 - ra2)
    return np.degrees(np.arccos(np.clip(cs, -1, 1)))


def main():
    scan = "moon0520"
    print(f"=== Deriving the rigid UVW rotation of {scan} from convention + "
          f"sky angles ===\n")

    rec, B, jd, gast, lon, a1, a2, app = m._load_full(m.MOON[scan])
    syn = synth_from_lta(LTA[scan], B, jd, gast, lon, jd.mean())   # theoretical
    Q_emp, resid = procrustes(rec, syn)
    ang_e, ax_e, det_e = angle_axis(Q_emp)
    print("(1) Empirical rotation theoretical -> recorded (orthogonal Procrustes")
    print("    straight off the recorded UVW):")
    print(f"      angle = {ang_e:.2f} deg   det = {det_e:+.3f}   fit RMS = "
          f"{resid:.1f} m")
    print(f"      axis (U,V,W) = [{ax_e[0]:+.4f} {ax_e[1]:+.4f} {ax_e[2]:+.4f}]\n")

    # (2) recover the wrong direction the recorded UVW point at (per-integration
    #     W-axis, robust; the global fit can fall into a pole local-minimum)
    pra, pdec, pres = m._pointing_per_integration(rec, B, jd, gast, lon)
    ra2, dec2 = float(pra.mean()), float(pdec.mean())
    ra1, dec1 = LTA[scan]["ra"], LTA[scan]["dec"]
    swing = sep_deg(ra1, dec1, ra2, dec2)
    print(f"(2) True apparent Moon (LTA)          : RA1={ra1:8.4f}  Dec1={dec1:8.4f}")
    print(f"    Wrong direction from recorded UVW : RA2={ra2:8.4f}  Dec2={dec2:8.4f}"
          f"  (+-{pra.std():.3f},{pdec.std():.3f})")
    print(f"    Sky separation (W-axis swing)     : {swing:.2f} deg\n")

    # (3) does a SINGLE fixed wrong direction actually fit the recorded UVW?
    H2 = gast + lon - np.radians(ra2)
    pred = m._predict_uvw(B, H2, np.radians(dec2) * np.ones(len(B)))
    rms_dir = np.sqrt(np.mean(np.sum((pred - rec) ** 2, 1)))
    print(f"(3) Predict UVW at that ONE wrong (RA2,Dec2), tracked with "
          f"H=GAST+lon-RA2:")
    print(f"      RMS vs recorded = {rms_dir:.1f} m over the whole scan  "
          f"-> {'a fixed wrong pointing FITS' if rms_dir < 50 else 'does NOT fit'}\n")

    # (4) closed-form Q from the two directions ALONE
    d1r, d2r = np.radians(dec1), np.radians(dec2)
    Q_cf = A(d2r) @ Rz(np.radians(ra2 - ra1)) @ A(d1r).T
    ang_c, ax_c, det_c = angle_axis(Q_cf)
    print("(4) Closed-form Q = A(Dec2).Rz(RA2-RA1).A(Dec1)^T  (directions only;")
    print("    no array location, no time -- both cancel):")
    print(f"      angle = {ang_c:.2f} deg   det = {det_c:+.3f}")
    print(f"      axis (U,V,W) = [{ax_c[0]:+.4f} {ax_c[1]:+.4f} {ax_c[2]:+.4f}]")
    print(f"    vs empirical: d(angle) = {abs(ang_c - ang_e):.2f} deg, "
          f"axis dot = {abs(float(np.dot(ax_c, ax_e))):+.4f}\n")

    # (5) decompose 159 deg = pointing swing (W-axis) + line-of-sight roll
    #     swing S: shortest rotation taking theoretical W=(0,0,1) to recorded W
    w_rec = Q_emp[:, 2]                       # recorded W in theoretical-UVW coords
    theta = np.arccos(np.clip(w_rec[2], -1, 1))
    n = np.cross([0, 0, 1.0], w_rec)
    n /= np.linalg.norm(n)
    K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    S = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)  # Rodrigues
    roll = Q_emp @ S.T                        # residual rotation about recorded W
    ang_roll, _, _ = angle_axis(roll)
    print(f"(5) Decomposition of the {ang_e:.1f} deg rotation:")
    print(f"      pointing swing (W-axis, true->wrong) = "
          f"{np.degrees(theta):.2f} deg")
    print(f"      line-of-sight roll of U,V about W    = {ang_roll:.2f} deg")
    print("    The pointing swing is the 126 deg the beam is off the Moon; the")
    print("    extra roll is the twist of the U,V axes that no single (RA,Dec)")
    print("    label removes -- together they make the 159 deg rotation.\n")

    # (6) which convention 'knobs' reproduce Q? map each single-parameter error
    #     to its rotation axis, and try a two-knob fit dec-tilt(about U) + roll(W)
    axis_dec = np.array([1.0, 0.0, 0.0])                    # wrong Dec  -> about U
    axis_H = np.array([0.0, np.cos(d1r), np.sin(d1r)])      # wrong H/RA -> about (0,cosd,sind)
    print("(6) Which convention error is it? (axis alignment, |dot| = 1 is exact)")
    print(f"      wrong-Dec  rotates about U           : |dot| = "
          f"{abs(float(np.dot(ax_e, axis_dec))):.3f}")
    print(f"      wrong-H/RA rotates about (0,cosD,sinD): |dot| = "
          f"{abs(float(np.dot(ax_e, axis_H))):.3f}")
    print(f"      measured axis V-component            : {ax_e[1]:+.3f}  "
          f"(=0 => no hour-angle-axis error; axis lies in the U-W plane)\n")

    def Ru(a):
        s, c = np.sin(a), np.cos(a)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def Rw(g):
        s, c = np.sin(g), np.cos(g)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    from scipy.optimize import least_squares
    for order, f in (("Ru(a).Rw(g)", lambda a, g: Ru(a) @ Rw(g)),
                     ("Rw(g).Ru(a)", lambda a, g: Rw(g) @ Ru(a))):
        sol = least_squares(lambda x: (f(x[0], x[1]) - Q_emp).ravel(),
                            [np.radians(160), np.radians(160)])
        fro = np.sqrt(np.mean(sol.fun ** 2))
        print(f"      fit Q = {order}: dec-tilt a={np.degrees(sol.x[0]) % 360:6.1f} deg, "
              f"uv-roll g={np.degrees(sol.x[1]) % 360:6.1f} deg  "
              f"(rms {fro:.3f})")

    # (7) IS THE ROTATION A CLEAN MULTIPLE OF THE RATE?  Three with-rate scans
    #     carry three different rates; if gvfits consumes the rate with a wrong
    #     sign/unit factor k, then rot_angle (or swing) = k * rate for ONE
    #     constant k across all three. Measure and check.
    print("\n(7) Rate-proportionality test across the three with-rate scans")
    print("    (if the defect is rate consumed with a wrong sign/unit factor,")
    print("     angle/|rate| and swing/|rate| are constant across scans):\n")
    hdr = (f"    {'scan':9s} {'|rate|':>9s} {'rot_ang':>8s} {'swing':>7s} "
           f"{'HA':>7s} {'parallac':>8s} {'180-par':>8s} {'roll':>7s}")
    print(hdr)
    from moon_uv_coverage_corrected_movie import LTA as LTA5  # all 5 scan blocks
    from astropy.io import fits
    for sc in ("moon0520", "moon0545", "moon0605"):
        r, Bs, j, g, lo, _, _, _ = m._load_full(m.MOON[sc])
        sy = synth_from_lta(LTA5[sc], Bs, j, g, lo, j.mean())
        Qs, _ = procrustes(r, sy)
        aa, axs, _ = angle_axis(Qs)
        pr, pd, _ = m._pointing_per_integration(r, Bs, j, g, lo)
        sw = sep_deg(LTA5[sc]["ra"], LTA5[sc]["dec"], float(pr.mean()), float(pd.mean()))
        dra, ddec = LTA5[sc]["dra"], LTA5[sc]["ddec"]
        rmag = np.hypot(dra, ddec)
        # geometry: HA, parallactic angle at scan centre
        with fits.open(m.MOON[sc], memmap=False) as h:
            anh = [hd.header for hd in h
                   if hd.header.get("EXTNAME", "").strip() == "AIPS AN"][0]
        phi = np.arctan2(anh["ARRAYZ"], np.hypot(anh["ARRAYX"], anh["ARRAYY"]))
        dr = np.radians(LTA5[sc]["dec"])
        Hs = (g.mean() + lo - np.radians(LTA5[sc]["ra"]) + np.pi) % (2 * np.pi) - np.pi
        parr = np.degrees(np.arctan2(np.sin(Hs),
                          np.tan(phi) * np.cos(dr) - np.sin(dr) * np.cos(Hs)))
        # los roll = residual about recorded W after the shortest W-swing
        wr = Qs[:, 2]
        th = np.arccos(np.clip(wr[2], -1, 1))
        nn = np.cross([0, 0, 1.0], wr)
        nn = nn / np.linalg.norm(nn)
        Kk = np.array([[0, -nn[2], nn[1]], [nn[2], 0, -nn[0]], [-nn[1], nn[0], 0]])
        Sm = np.eye(3) + np.sin(th) * Kk + (1 - np.cos(th)) * (Kk @ Kk)
        rollang, _, _ = angle_axis(Qs @ Sm.T)
        print(f"    {sc:9s} {rmag:9.6f} {aa:8.2f} {sw:7.2f} "
              f"{np.degrees(Hs):7.2f} {parr:8.2f} {180 - parr:8.2f} {rollang:7.2f}")
    print("\n    If swing ~= 180 - parallactic across all three, the pointing")
    print("    offset is a parallactic (horizon-vs-equatorial) frame rotation,")
    print("    which is built from tan(latitude) -- i.e. cos(lat) IS involved.")

    # (8) THE SIGN-FLIP TEST (user hypothesis): can flipping the sign of the RA
    #     rate, the Dec rate, or both -- (++,+-,-+,--) -- reproduce the recorded
    #     UVW?  The rate enters as phase-centre = (RA + s_ra*dra*dt, Dec +
    #     s_dec*ddec*dt). For each of the 4 sign combos we let the time-lever dt
    #     float FREELY (not just the +-450 s scan window) and find the dt that
    #     best matches the recorded UVW -- the most generous possible test.
    print("\n(8) SIGN-FLIP TEST for moon0520: reproduce recorded UVW by flipping")
    print("    rate signs (++,+-,-+,--), time-lever dt free to best-fit:\n")
    dra0, ddec0 = LTA5[scan]["dra"], LTA5[scan]["ddec"]
    # sky displacement true(RA1,Dec1) -> wrong(RA2,Dec2), degrees (RA un-wrapped)
    dRA_disp = ((ra2 - ra1 + 180) % 360) - 180
    dDec_disp = dec2 - dec1
    print(f"    sky displacement true->wrong : dRA={dRA_disp:+.2f} deg, "
          f"dDec={dDec_disp:+.2f} deg  (dRA/dDec = {dRA_disp/dDec_disp:+.2f})")
    print(f"    rate vector                  : dRA/dt/dDec/dt = "
          f"{dra0/ddec0:+.2f}  (sign flips only change this to +-{dra0/ddec0:.2f})\n")

    def rms_at(s_ra, s_dec, dt_s):
        ra = np.radians(ra1 + s_ra * dra0 * dt_s)
        dec = np.radians(dec1 + s_dec * ddec0 * dt_s)
        H = gast + lon - ra
        pr = m._predict_uvw(B, H, dec * np.ones(len(B)))
        return np.sqrt(np.mean(np.sum((pr - rec) ** 2, 1)))

    for s_ra in (+1, -1):
        for s_dec in (+1, -1):
            sol = least_squares(lambda x: rms_at(s_ra, s_dec, x[0]), [0.0])
            dt_best = sol.x[0]
            rms_best = rms_at(s_ra, s_dec, dt_best)
            sig = {(+1, +1): "++", (+1, -1): "+-",
                   (-1, +1): "-+", (-1, -1): "--"}[(s_ra, s_dec)]
            print(f"      {sig}: best-fit dt = {dt_best:12.1f} s "
                  f"({dt_best/86400:8.2f} d) -> RMS vs recorded = {rms_best:10.1f} m "
                  f"-> {'REPRODUCES' if rms_best < 50 else 'does NOT'}")
    print("\n    (recall the true rigid rotation fits to 2.0 m; a fixed wrong")
    print("     pointing to ~14000 m. If all 4 combos stay ~km, a rate sign flip")
    print("     in the position path is DISPROVED as the mechanism.)")

    # (9) WHERE COULD A 159 deg FRAME ROTATION COME FROM? test closed-form
    #     rotations built from the real observing geometry (latitude, HA, Dec,
    #     parallactic angle) against the measured Q -- is cos(latitude) in it?
    from astropy.io import fits
    with fits.open(m.MOON[scan], memmap=False) as h:
        anh = [hd.header for hd in h
               if hd.header.get("EXTNAME", "").strip() == "AIPS AN"][0]
    ax_, ay_, az_ = anh["ARRAYX"], anh["ARRAYY"], anh["ARRAYZ"]
    phi = np.arctan2(az_, np.hypot(ax_, ay_))               # geocentric latitude
    H1 = float((gast.mean() + lon - np.radians(ra1)))
    H1 = (H1 + np.pi) % (2 * np.pi) - np.pi
    el = np.arcsin(np.sin(phi) * np.sin(d1r) +
                   np.cos(phi) * np.cos(d1r) * np.cos(H1))
    par = np.arctan2(np.sin(H1),
                     np.tan(phi) * np.cos(d1r) - np.sin(d1r) * np.cos(H1))
    print("\n(9) Observing geometry at scan centre, and closed-form rotation tests:")
    print(f"      site latitude phi = {np.degrees(phi):+.3f} deg   "
          f"(cos phi = {np.cos(phi):.4f})")
    print(f"      Moon  HA H1 = {np.degrees(H1):+.2f} deg   Dec1 = "
          f"{np.degrees(d1r):+.2f} deg   elevation = {np.degrees(el):+.2f} deg")
    print(f"      parallactic angle = {np.degrees(par):+.2f} deg\n")
    print(f"      measured Q: angle = {ang_e:.2f} deg, "
          f"axis = [{ax_e[0]:+.3f} {ax_e[1]:+.3f} {ax_e[2]:+.3f}]\n")

    def Ry(a):
        s, c = np.sin(a), np.cos(a)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    cands = {
        "frame built for -Dec  A(-d).A(d)^T": A(-d1r) @ A(d1r).T,
        "frame built for -HA   A(d).Rz(2H).A(d)^T": A(d1r) @ Rz(2 * H1) @ A(d1r).T,
        "Ru(2*phi)         (dec axis, 2 lat)": Ru(2 * phi),
        "Ru(phi - Dec1)    (dec axis)": Ru(phi - d1r),
        "Ru(2*Dec1)        (dec axis)": Ru(2 * d1r),
        "Ry(2*phi)         (V axis, 2 lat)": Ry(2 * phi),
        "Rw(2*parallactic) (los roll)": Rw(2 * par),
        "Rw(pi)            (uv sign flip)": Rw(np.pi),
    }
    print(f"      {'candidate rotation':40s} {'angle':>8s} {'axis.dot':>9s}")
    for name, Qc in cands.items():
        ac, axc, _ = angle_axis(Qc)
        dot = abs(float(np.dot(axc, ax_e))) if np.all(np.isfinite(axc)) else np.nan
        print(f"      {name:40s} {ac:8.2f} {dot:9.3f}")
    print("\n    A candidate MATCHES only if angle ~= 159 AND axis.dot ~= 1.")


if __name__ == "__main__":
    main()
