#!/usr/bin/env python
"""Read-only verification checks for the Moon UVW investigation.

One subcommand per ticket in moon_uvw_verification_tickets.md. Reads only;
writes nothing except stdout. Run e.g.:

    gmrt/bin/python tools/dev/moon_uvw_checks.py mv01
    gmrt/bin/python tools/dev/moon_uvw_checks.py mv02
"""
import sys
import numpy as np
from astropy.io import fits

C = 299792458.0  # m/s
DATA = "/Users/raj030/DATA/gmrt_40_014/work/split"
CAL = f"{DATA}/3c468.1/3c468.1_primary_secondary_calibrated_flagged.uvfits"
MOON = {s: f"{DATA}/moon/{s}_primary_secondary_calibrated_flagged.uvfits"
        for s in ("moon0520", "moon0545", "moon0605", "moon0625", "moon0635")}
ALL = {"3c468.1": CAL, **MOON}


def ptype_index(hdr, prefix):
    return [k for k in range(hdr["PCOUNT"]) if hdr[f"PTYPE{k+1}"].startswith(prefix)]


def load_uvw_baseline(path):
    """Return dict with uu/vv/ww (seconds, scaled), baseline codes, and header."""
    with fits.open(path, memmap=False) as h:
        g = h[0]
        hdr = g.header
        pt = [hdr[f"PTYPE{k+1}"] for k in range(hdr["PCOUNT"])]
        scal = [hdr.get(f"PSCAL{k+1}", 1.0) for k in range(hdr["PCOUNT"])]
        zero = [hdr.get(f"PZERO{k+1}", 0.0) for k in range(hdr["PCOUNT"])]

        def par(prefix):
            i = ptype_index(hdr, prefix)[0]
            return np.asarray(g.data.par(i), "f8")  # astropy applies PSCAL/PZERO

        uu, vv, ww = par("UU"), par("VV"), par("WW")
        bl_i = ptype_index(hdr, "BASELINE")
        baseline = np.asarray(g.data.par(bl_i[0]), "f8") if bl_i else None
        # antenna table
        an = h["AIPS AN"].data
        nosta = np.asarray(an["NOSTA"], int)
        stabxyz = np.asarray(an["STABXYZ"], "f8")  # metres
        xyz = {int(n): stabxyz[k] for k, n in enumerate(nosta)}
    return dict(uu=uu, vv=vv, ww=ww, baseline=baseline, xyz=xyz,
                pt=pt, scal=scal, zero=zero, path=path)


def decode_baseline(bl):
    """AIPS: baseline = ant1*256 + ant2 (+ 0.01*subarray). Return (a1, a2)."""
    b = np.floor(bl + 0.001).astype(int)  # drop subarray fraction
    a1 = b // 256
    a2 = b % 256
    return a1, a2


def mv01():
    print("=== MV-01: UVW unit + |UVW| = |B| magnitude check ===\n")
    for name, path in ALL.items():
        d = load_uvw_baseline(path)
        # report unit metadata for UU
        i = ptype_index(fits.getheader(path), "UU")[0]
        print(f"[{name}]  PTYPE={d['pt'][i]!r}  PSCAL={d['scal'][i]:.6e}  "
              f"PZERO={d['zero'][i]:.6e}  (astropy applies these -> physical units)")
        uvw_sec = np.sqrt(d["uu"]**2 + d["vv"]**2 + d["ww"]**2)
        uvw_m = uvw_sec * C
        a1, a2 = decode_baseline(d["baseline"])
        # baseline length from antenna table
        miss = 0
        bl_m = np.full(len(a1), np.nan)
        for k in range(len(a1)):
            p1, p2 = d["xyz"].get(a1[k]), d["xyz"].get(a2[k])
            if p1 is None or p2 is None:
                miss += 1
                continue
            bl_m[k] = np.linalg.norm(p2 - p1)
        good = np.isfinite(bl_m) & (bl_m > 0)  # drop autocorr / missing
        diff = np.abs(uvw_m[good] - bl_m[good])
        print(f"    rows={len(a1)}  autocorr/missing={(~good).sum()}  "
              f"median|UVW·c - |B|| = {np.median(diff):.4f} m   "
              f"max = {diff.max():.4f} m   "
              f"(|B| range {bl_m[good].min():.1f}..{bl_m[good].max():.1f} m)\n")


def mv02():
    print("=== MV-02: measured RAEPO/DECEPO vs RAAPP/DECAPP separation ===\n")
    print(f"{'source':10s} {'RAEPO':>10s} {'DECEPO':>9s}  {'RAAPP':>10s} "
          f"{'DECAPP':>9s}  {'sep':>10s}")
    for name, path in ALL.items():
        with fits.open(path, memmap=False) as h:
            su = h["AIPS SU"].data
            raepo, decepo = float(su["RAEPO"][0]), float(su["DECEPO"][0])
            raapp, decapp = float(su["RAAPP"][0]), float(su["DECAPP"][0])
            epoch = h["AIPS SU"].header.get("EPOCH", "?")
        # great-circle separation
        r1, d1, r2, d2 = np.radians([raepo, decepo, raapp, decapp])
        cs = np.sin(d1)*np.sin(d2) + np.cos(d1)*np.cos(d2)*np.cos(r1-r2)
        sep = np.degrees(np.arccos(np.clip(cs, -1, 1)))
        print(f"{name:10s} {raepo:10.4f} {decepo:9.4f}  {raapp:10.4f} "
              f"{decapp:9.4f}  {sep*60:8.2f}'  (EPOCH={epoch})")


# --- MS phase-centre paths (imported, pre-phaseshift where available) -------
MS = {
    "3c468.1_scan05": "/Users/raj030/DATA/gmrt_40_014/work/casa_selfcal/3c468.1_scan05_stk1/3c468.1_full.ms",
    "3c468.1_scan06": "/Users/raj030/DATA/gmrt_40_014/work/casa_selfcal/3c468.1_scan06_stk1/3c468.1_full.ms",
    "moon0520_full": "/Users/raj030/DATA/gmrt_40_014/work/scratch/moon0520_full.ms",
    "moon0520_shifted": "/Users/raj030/DATA/gmrt_40_014/work/scratch/moon0520_full_shifted.ms",
    "moon0520_phasecenter": "/Users/raj030/DATA/gmrt_40_014/work/casa_selfcal/moon0520_stk10_phasecenter/moon0520_full.ms",
}


def _sep_deg(r1, d1, r2, d2):
    r1, d1, r2, d2 = np.radians([r1, d1, r2, d2])
    cs = np.sin(d1)*np.sin(d2) + np.cos(d1)*np.cos(d2)*np.cos(r1-r2)
    return np.degrees(np.arccos(np.clip(cs, -1, 1)))


def mv03():
    print("=== MV-03: MS FIELD::PHASE_DIR vs SU RAEPO/RAAPP ===\n")
    from casatools import table, measures
    tb = table()
    # reference SU positions
    def su(path):
        with fits.open(path, memmap=False) as h:
            s = h["AIPS SU"].data
            return (float(s["RAEPO"][0]), float(s["DECEPO"][0]),
                    float(s["RAAPP"][0]), float(s["DECAPP"][0]))
    cal_epo_ra, cal_epo_dec, cal_app_ra, cal_app_dec = su(CAL)
    moon_epo_ra, moon_epo_dec, moon_app_ra, moon_app_dec = su(MOON["moon0520"])
    for name, path in MS.items():
        tb.open(f"{path}/FIELD")
        pd = tb.getcol("PHASE_DIR")          # rad, shape (2,1,nfield)
        names = list(tb.getcol("NAME"))
        ci = tb.getcolkeyword("PHASE_DIR", "MEASINFO")
        tb.close()
        # rows 0..n-2 can be empty placeholders; pick the populated source field
        fi = int(np.argmax(pd[0, 0, :]**2 + pd[1, 0, :]**2))
        ra = np.degrees(pd[0, 0, fi]) % 360.0
        dec = np.degrees(pd[1, 0, fi])
        ref = ci.get("Ref", "?")
        name = names[fi] if fi < len(names) else "?"
        print(f"    (field[{fi}]={name!r} of {pd.shape[2]} fields)")
        if name.startswith("3c468.1"):
            e = _sep_deg(ra, dec, cal_epo_ra, cal_epo_dec)*60
            a = _sep_deg(ra, dec, cal_app_ra, cal_app_dec)*60
        else:
            e = _sep_deg(ra, dec, moon_epo_ra, moon_epo_dec)*60
            a = _sep_deg(ra, dec, moon_app_ra, moon_app_dec)*60
        print(f"[{name:22s}] PHASE_DIR={ra:10.5f},{dec:9.5f} ref={ref:6s} "
              f"| to EPO {e:7.3f}'  to APP {a:7.3f}'")


def _array_lon_rad(hdr_an):
    """Site east longitude from AN-header ARRAYX/Y/Z (geocentric)."""
    ax, ay = hdr_an.get("ARRAYX", 0.0), hdr_an.get("ARRAYY", 0.0)
    return np.arctan2(ay, ax)


def _gast_rad(jd):
    from astropy.time import Time
    import astropy.units as u
    t = Time(np.asarray(jd), format="jd", scale="utc")
    return t.sidereal_time("apparent", "greenwich").to(u.rad).value


def _predict_uvw(B, H, dec):
    """(u,v,w) metres for baseline B (Greenwich earth-fixed), hour angle H, dec.
    B: (N,3); H, dec: (N,) radians. TMS eq 4.1."""
    sH, cH, sD, cD = np.sin(H), np.cos(H), np.sin(dec), np.cos(dec)
    Bx, By, Bz = B[:, 0], B[:, 1], B[:, 2]
    u = sH*Bx + cH*By
    v = -sD*cH*Bx + sD*sH*By + cD*Bz
    w = cD*cH*Bx - cD*sH*By + sD*Bz
    return np.column_stack([u, v, w])


def mv04():
    print("=== MV-04: frame/convention sweep on calibrator 3C468.1 ===")
    print("Predict UVW from antenna XYZ + GAST; compare to recorded (metres RMS).")
    print("Sweep: baseline sign, position (EPO/APP), hour-angle longitude term.\n")
    with fits.open(CAL, memmap=False) as h:
        g = h[0]
        hdr = g.header
        pt = [hdr[f"PTYPE{k+1}"] for k in range(hdr["PCOUNT"])]

        def par(prefix):
            i = [k for k, q in enumerate(pt) if q.startswith(prefix)][0]
            return np.asarray(g.data.par(i), "f8")
        uu, vv, ww = par("UU"), par("VV"), par("WW")
        di = [k for k, q in enumerate(pt) if q == "DATE"]
        jd = np.asarray(g.data.par(di[0]), "f8")
        if len(di) > 1:
            jd = jd + np.asarray(g.data.par(di[1]), "f8")
        bl = par("BASELINE")
        an = h["AIPS AN"].data
        anh = h["AIPS AN"].header
        nosta = np.asarray(an["NOSTA"], int)
        stab = np.asarray(an["STABXYZ"], "f8")
        xyz = {int(n): stab[k] for k, n in enumerate(nosta)}
        su = h["AIPS SU"].data
        pos = {"EPO": (float(su["RAEPO"][0]), float(su["DECEPO"][0])),
               "APP": (float(su["RAAPP"][0]), float(su["DECAPP"][0]))}

    lon = _array_lon_rad(anh)
    print(f"array east-longitude from ARRAYX/Y/Z = {np.degrees(lon):.4f} deg "
          f"(|STABXYZ| median {np.median(np.linalg.norm(stab,axis=1)):.1f} m)\n")
    a1, a2 = decode_baseline(bl)
    rec = np.column_stack([uu, vv, ww]) * C          # metres
    # baseline vectors both sign conventions
    B21 = np.array([xyz[int(a2[k])] - xyz[int(a1[k])] for k in range(len(a1))])
    B12 = -B21
    good = np.array([int(a1[k]) != int(a2[k]) for k in range(len(a1))])
    gast = _gast_rad(jd)

    for signname, B in (("ant2-ant1", B21), ("ant1-ant2", B12)):
        for posname, (ra, dec) in pos.items():
            for lonname, Huse in (("GAST-RA", gast - np.radians(ra)),
                                  ("GAST+lon-RA", gast + lon - np.radians(ra))):
                pred = _predict_uvw(B, Huse, np.radians(dec) * np.ones(len(B)))
                resid = np.linalg.norm(pred[good] - rec[good], axis=1)
                print(f"  sign={signname:9s} pos={posname:3s} H={lonname:12s} "
                      f"RMS={np.sqrt(np.mean(resid**2)):10.2f} m  "
                      f"median={np.median(resid):10.2f} m")
        print()


def _load_for_recovery(path):
    with fits.open(path, memmap=False) as h:
        g = h[0]
        hdr = g.header
        pt = [hdr[f"PTYPE{k+1}"] for k in range(hdr["PCOUNT"])]

        def par(prefix):
            i = [k for k, q in enumerate(pt) if q.startswith(prefix)][0]
            return np.asarray(g.data.par(i), "f8")
        uu, vv, ww = par("UU"), par("VV"), par("WW")
        di = [k for k, q in enumerate(pt) if q == "DATE"]
        jd = np.asarray(g.data.par(di[0]), "f8")
        if len(di) > 1:
            jd = jd + np.asarray(g.data.par(di[1]), "f8")
        bl = par("BASELINE")
        an = h["AIPS AN"].data
        anh = h["AIPS AN"].header
        nosta = np.asarray(an["NOSTA"], int)
        stab = np.asarray(an["STABXYZ"], "f8")
        xyz = {int(n): stab[k] for k, n in enumerate(nosta)}
        su = h["AIPS SU"].data
        posmap = {"EPO": (float(su["RAEPO"][0]), float(su["DECEPO"][0])),
                  "APP": (float(su["RAAPP"][0]), float(su["DECAPP"][0]))}
    a1, a2 = decode_baseline(bl)
    rec = np.column_stack([uu, vv, ww]) * C
    B = np.array([xyz[int(a2[k])] - xyz[int(a1[k])] for k in range(len(a1))])  # ant2-ant1 (MV-04)
    good = np.array([int(a1[k]) != int(a2[k]) for k in range(len(a1))])
    lon = _array_lon_rad(anh)
    return rec[good], B[good], _gast_rad(jd)[good], lon, posmap


def _recover_radec(rec, B, gast, lon):
    """Least-squares (RA,Dec) from recorded UVW using certified recipe
    (sign ant2-ant1, H = GAST + lon - RA). Grid seed then refine."""
    from scipy.optimize import least_squares

    def resid(x):
        ra, dec = x
        H = gast + lon - ra
        pred = _predict_uvw(B, H, dec * np.ones(len(B)))
        return (pred - rec).ravel()
    # coarse grid seed (avoid seeding at the answer)
    best, bx = np.inf, (0.0, 0.0)
    for ra in np.radians(np.arange(0, 360, 10)):
        for dec in np.radians(np.arange(-80, 81, 10)):
            r = resid((ra, dec))
            s = np.mean(r**2)
            if s < best:
                best, bx = s, (ra, dec)
    # bounded fit: Dec physically confined to [-90,90] so it cannot flip to the
    # antipode; RA free (wrapped afterwards)
    sol = least_squares(resid, bx, method="trf",
                        bounds=([-4*np.pi, -np.pi/2], [4*np.pi, np.pi/2]))
    ra, dec = sol.x
    rms = np.sqrt(np.mean(sol.fun**2))
    return np.degrees(ra) % 360.0, np.degrees(dec), rms


def mv05():
    print("=== MV-05 GATE: recover 3C468.1 RA/Dec from its own recorded UVW ===")
    print("Recipe (certified by MV-04): sign=ant2-ant1, H=GAST+lon-RA. "
          "Seeded from a coarse grid, not the answer.\n")
    rec, B, gast, lon, posmap = _load_for_recovery(CAL)
    ra, dec, rms = _recover_radec(rec, B, gast, lon)
    print(f"recovered:  RA={ra:.4f}  Dec={dec:.4f}   (fit RMS {rms:.2f} m)")
    for k, (r0, d0) in posmap.items():
        print(f"  vs {k}: catalogue {r0:.4f},{d0:.4f}  ->  "
              f"sep = {_sep_deg(ra, dec, r0, d0)*60:.3f}'  "
              f"({_sep_deg(ra, dec, r0, d0):.4f} deg)")
    gate = min(_sep_deg(ra, dec, *posmap["EPO"]), _sep_deg(ra, dec, *posmap["APP"]))
    print(f"\nGATE {'PASS' if gate < 0.1 else 'FAIL'}: "
          f"closest catalogue separation {gate:.4f} deg (threshold 0.1)")


def mv07():
    print("=== MV-07: apply certified recipe to the Moon (file's own RAAPP) ===")
    print("Predict UVW at SU RAAPP/DECAPP; compare to recorded. Also recover "
          "direction from recorded UVW and compare to the true apparent Moon.\n")
    for name, path in MOON.items():
        rec, B, gast, lon, posmap = _load_for_recovery(path)
        appra, appdec = posmap["APP"]
        H = gast + lon - np.radians(appra)
        pred = _predict_uvw(B, H, np.radians(appdec) * np.ones(len(B)))
        r = np.linalg.norm(pred - rec, axis=1)
        ra, dec, rms = _recover_radec(rec, B, gast, lon)
        sep = _sep_deg(ra, dec, appra, appdec)
        print(f"[{name}] predict@APP: RMS={np.sqrt(np.mean(r**2)):9.1f} m "
              f"median={np.median(r):9.1f} m | recovered {ra:.3f},{dec:.3f} "
              f"vs APP {appra:.3f},{appdec:.3f} -> sep {sep:.3f} deg "
              f"({'ON-MOON' if sep < 0.2 else 'OFFENDING'})")


def _load_full(path):
    """rec (N,3 m), B (N,3, ant2-ant1), jd, gast, lon, a1,a2, RAAPP/DECAPP."""
    with fits.open(path, memmap=False) as h:
        g = h[0]
        hdr = g.header
        pt = [hdr[f"PTYPE{k+1}"] for k in range(hdr["PCOUNT"])]

        def par(prefix):
            i = [k for k, q in enumerate(pt) if q.startswith(prefix)][0]
            return np.asarray(g.data.par(i), "f8")
        uu, vv, ww = par("UU"), par("VV"), par("WW")
        di = [k for k, q in enumerate(pt) if q == "DATE"]
        jd = np.asarray(g.data.par(di[0]), "f8")
        if len(di) > 1:
            jd = jd + np.asarray(g.data.par(di[1]), "f8")
        bl = par("BASELINE")
        an = h["AIPS AN"].data
        anh = h["AIPS AN"].header
        nosta = np.asarray(an["NOSTA"], int)
        stab = np.asarray(an["STABXYZ"], "f8")
        xyz = {int(n): stab[k] for k, n in enumerate(nosta)}
        su = h["AIPS SU"].data
        app = (float(su["RAAPP"][0]), float(su["DECAPP"][0]))
    a1, a2 = decode_baseline(bl)
    good = a1 != a2
    rec = (np.column_stack([uu, vv, ww]) * C)[good]
    B = np.array([xyz[int(a2[k])] - xyz[int(a1[k])] for k in range(len(a1))])[good]
    return (rec, B, jd[good], _gast_rad(jd)[good], _array_lon_rad(anh),
            a1[good], a2[good], app)


def _pointing_per_integration(rec, B, jd, gast, lon):
    """Per-timestamp orthogonal Procrustes B->uvw; decode the w-axis into the
    instantaneous phase-centre (RA,Dec). Returns arrays over integrations."""
    stamps, inv = np.unique(np.round(jd, 8), return_inverse=True)
    ras, decs, res = [], [], []
    for j in range(len(stamps)):
        m = inv == j
        if m.sum() < 10:
            continue
        Bi, ri = B[m], rec[m]
        M = ri.T @ Bi
        U, _, Vt = np.linalg.svd(M)
        d = np.sign(np.linalg.det(U @ Vt))
        R = U @ np.diag([1, 1, d]) @ Vt
        res.append(np.sqrt(np.mean(np.sum((Bi @ R.T - ri)**2, 1))))
        wx, wy, wz = R[2]                       # w-axis, earth-fixed (X=meridian)
        dec = np.arcsin(np.clip(wz, -1, 1))
        H = np.arctan2(-wy, wx)
        lst = gast[m].mean() + lon
        ras.append(np.degrees((lst - H)) % 360.0)
        decs.append(np.degrees(dec))
    return np.array(ras), np.array(decs), np.array(res)


def mv09():
    print("=== MV-09: benign-hypothesis sweep on the offending scans ===\n")
    from scipy.optimize import least_squares
    for name in ("moon0520", "moon0545", "moon0605", "moon0625", "moon0635"):
        rec, B, jd, gast, lon, a1, a2, (appra, appdec) = _load_full(MOON[name])
        stamps = np.unique(np.round(jd, 8))
        dur_min = (stamps.max() - stamps.min()) * 24 * 60

        # (1) frozen-delay test: temporal spread of UVW per baseline over the scan
        # pick one representative long baseline present in most integrations
        code = a1.astype(np.int64) * 256 + a2.astype(np.int64)
        uc, inv, cnt = np.unique(code, return_inverse=True, return_counts=True)
        # mean |u| per unique baseline; pick a long, well-sampled one
        mean_u = np.array([np.abs(rec[inv == j, 0]).mean() for j in range(len(uc))])
        cand = np.argmax(cnt * (mean_u > 100))
        sel = inv == cand
        u_spread = rec[sel, 0].max() - rec[sel, 0].min()   # metres over the scan

        # (2) tracking-apparent-Moon model (= MV-07)
        H_trk = gast + lon - np.radians(appra)
        pred_trk = _predict_uvw(B, H_trk, np.radians(appdec) * np.ones(len(B)))
        rms_trk = np.sqrt(np.mean(np.sum((pred_trk - rec)**2, 1)))

        # (3) SINGLE-ROTATION (frozen-geometry) test via orthogonal Procrustes:
        #     best constant R mapping B -> rec over the WHOLE scan. Low residual
        #     => the uvw are one fixed rotation of the baselines (delay model not
        #     updating). High residual => geometry is genuinely time-varying.
        M = rec.T @ B
        U, _, Vt = np.linalg.svd(M)
        d = np.sign(np.linalg.det(U @ Vt))
        R = U @ np.diag([1, 1, d]) @ Vt
        rms_frozen = np.sqrt(np.mean(np.sum((B @ R.T - rec)**2, 1)))

        # (4) how much SHOULD the geometry move? compare to a correct tracking
        #     model's own uvw motion over the scan (apparent Moon), same baseline
        pm = _predict_uvw(B[sel], H_trk[sel],
                          np.radians(appdec) * np.ones(sel.sum()))
        moon_move = pm[:, 0].max() - pm[:, 0].min()

        # (5) PER-INTEGRATION pointing: recover the instantaneous phase-centre
        #     (RA,Dec) at each timestamp via Procrustes on that integration's
        #     baselines. If the scan tracks ONE (wrong or right) sky direction,
        #     these are consistent across the scan (small scatter) and drift in
        #     RA at ~0 (sky-fixed). Large scatter => not any single direction.
        pra, pdec, pres = _pointing_per_integration(rec, B, jd, gast, lon)
        # RA scatter measured as std of (RA - sidereal drift removed): for a
        # sky-fixed target RA is constant; report circular std directly.
        ra_un = np.unwrap(np.radians(pra))
        ra_scatter = np.degrees(np.std(ra_un))
        dec_scatter = np.std(pdec)

        verdict = "ON-MOON" if rms_trk < 100 else "OFFENDING"
        print(f"[{name}] {dur_min:4.1f} min, {len(stamps):3d} integ  [{verdict}]")
        print(f"    long-baseline u motion: recorded {u_spread:8.1f} m  vs  "
              f"apparent-Moon-tracking {moon_move:8.1f} m")
        print(f"    tracking-apparent-Moon model RMS : {rms_trk:11.1f} m")
        print(f"    single fixed rotation (Procrustes) RMS : {rms_frozen:9.1f} m "
              f"(det R={np.linalg.det(R):+.2f})")
        print(f"    per-integration pointing: RA={pra.mean():.3f}+-{ra_scatter:.3f} "
              f"Dec={pdec.mean():.3f}+-{dec_scatter:.3f}  "
              f"(vs true Moon APP {appra:.3f},{appdec:.3f}; "
              f"per-integ fit RMS {np.median(pres):.1f} m)")
        print(f"    -> sep(per-integ mean, Moon) = "
              f"{_sep_deg(pra.mean(), pdec.mean(), appra, appdec):.3f} deg\n")


if __name__ == "__main__":
    {"mv01": mv01, "mv02": mv02, "mv03": mv03, "mv04": mv04,
     "mv05": mv05, "mv07": mv07, "mv09": mv09}[sys.argv[1]]()
