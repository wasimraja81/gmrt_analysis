#!/usr/bin/env python
"""Read-only: synthesise the theoretical UVW from the LTA ONLINE MODEL and
compare, axis-by-axis, to the recorded UVFITS values -- the user's recipe,
done correctly.

KEY METHOD POINT (user correction): the inputs to the theoretical UVW MUST
come from the LTA (the online model gvfits was handed), NOT from the UVFITS
SU RAAPP/DECAPP -- those are clean values gvfits *also* wrote to the file and
re-deriving UVW from them is circular. So here:

  * phase centre  <- LTA RA-DATE/DEC-DATE  +  rate DRA/DT, DDEC/DT propagated
                     over time (the phase centre MOVES for a rate scan),
  * per-integration UTC time (for GAST)  <- UVFITS DATE (verified below to be
                     the correct UTC, ~Jul 26; the UT/IST/stale-MJD conflict is
                     resolved: true UTC anchor MJD_REF=59419.770833=Jul24 18:30,
                     DATE-OBS is IST labelled as UTC, MJD_SRC=59420 is ~1 day
                     stale),
  * antenna geometry (STABXYZ), baseline sign ant2-ant1, H=GAST+lon-RA  <-
                     the calibrator-certified recipe from moon_uvw_checks.

Compare recorded vs synthesised, per axis (U-U/V-V/W-W):
  * 3C468.1 (rate 0) + moon0625 (rate 0)  -> straight y=x (certifies code),
  * moon0520 (rate != 0)                  -> the departure = the clue.

Then, on moon0520, test whether feeding a WRONG time base (the stale/IST
anchors) into GAST reproduces the recorded 126-deg-off UVW.

No files modified except PNG plots under moon_uvw_synthesis/.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moon_uvw_checks as m  # noqa: E402  (validated forward model + loaders)

C = 299792458.0
OUTDIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "..", "..", "moon_uvw_synthesis"))

# LTA online-model source blocks (read-only `strings` of the GSB LTA).
# ra/dec in deg (apparent of date); dra/ddec in deg/s.
LTA = {
    "3c468.1":  dict(ra=357.995635, dec=64.791581, dra=0.000000, ddec=0.000000),
    "moon0625": dict(ra=331.664774, dec=-17.116175, dra=0.000000, ddec=0.000000),
    "moon0520": dict(ra=331.218408, dec=-17.384986, dra=0.000103, ddec=0.000067),
}
SOURCES = [
    ("3c468.1", m.CAL, "CONTROL (fixed calibrator, rate 0)"),
    ("moon0625", m.MOON["moon0625"], "GOOD Moon (rate 0)"),
    ("moon0520", m.MOON["moon0520"], "OFFENDING Moon (WITH rate)"),
]


def load_geometry(path):
    """Antenna geometry + per-integration UTC from the UVFITS (NOT the SU pos)."""
    rec, B, jd, gast, lon, a1, a2, app = m._load_full(path)
    return dict(rec=rec, B=B, jd=jd, gast=gast, lon=lon, uvfits_app=app)


def synth_from_lta(block, B, jd_utc, gast, lon, t_epoch_jd):
    """Theoretical UVW from the LTA model. Phase centre = RA-DATE/DEC-DATE +
    rate*(t - epoch); geometry via the certified H = GAST + lon - RA recipe."""
    dt_s = (jd_utc - t_epoch_jd) * 86400.0
    ra = np.radians(block["ra"] + block["dra"] * dt_s)
    dec = np.radians(block["dec"] + block["ddec"] * dt_s)
    H = gast + lon - ra
    return m._predict_uvw(B, H, dec)


def per_axis_stats(rec, syn):
    out = {}
    for k, ax in enumerate("UVW"):
        x, y = syn[:, k], rec[:, k]
        A = np.column_stack([x, np.ones_like(x)])
        (slope, intercept), *_ = np.linalg.lstsq(A, y, rcond=None)
        out[ax] = dict(slope=slope, intercept=intercept,
                       corr=np.corrcoef(x, y)[0, 1],
                       rms=np.sqrt(np.mean((y - x) ** 2)))
    return out


def rigid_rotation(rec, syn):
    """Proper rotation Q with rec ~= Q @ syn (orthogonal Procrustes)."""
    M = rec.T @ syn
    U, _, Vt = np.linalg.svd(M)
    d = np.sign(np.linalg.det(U @ Vt))
    Q = U @ np.diag([1, 1, d]) @ Vt
    resid = np.sqrt(np.mean(np.sum((syn @ Q.T - rec) ** 2, 1)))
    angle = np.degrees(np.arccos(np.clip((np.trace(Q) - 1) / 2, -1, 1)))
    ax = np.array([Q[2, 1] - Q[1, 2], Q[0, 2] - Q[2, 0], Q[1, 0] - Q[0, 1]])
    n = np.linalg.norm(ax)
    axis = ax / n if n > 1e-9 else np.array([np.nan] * 3)
    return dict(resid=resid, angle=angle, axis=axis, det=np.linalg.det(Q))


def gast_at(jd, dt_hours):
    from astropy.time import Time
    import astropy.units as u
    t = Time(jd + dt_hours / 24.0, format="jd", scale="utc")
    return t.sidereal_time("apparent", "greenwich").to(u.rad).value


def plot_source(name, blurb, rec, syn, stats, rot):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for k, ax in enumerate("UVW"):
        x, y = syn[:, k] / 1000.0, rec[:, k] / 1000.0
        a = axs[k]
        a.scatter(x, y, s=3, alpha=0.3, color="C0", rasterized=True)
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        a.plot([lo, hi], [lo, hi], "r-", lw=1, label="y = x (perfect)")
        s = stats[ax]
        a.set_title(f"{ax}  slope={s['slope']:.3f}  r={s['corr']:.4f}\n"
                    f"RMS(rec-syn)={s['rms']:.1f} m")
        a.set_xlabel(f"synthesised {ax} (km)  [from LTA model]")
        a.set_ylabel(f"recorded {ax} (km)")
        a.legend(loc="upper left", fontsize=8)
        a.set_aspect("equal", "box")
    fig.suptitle(f"{name} -- {blurb}   |   rigid rotation syn->rec: "
                 f"{rot['angle']:.1f} deg, fit RMS {rot['resid']:.1f} m",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(OUTDIR, exist_ok=True)
    fp = os.path.join(OUTDIR, f"uvw_synth_vs_recorded_{name}.png")
    fig.savefig(fp, dpi=110)
    plt.close(fig)
    return fp


def main():
    from astropy.time import Time
    print("Theoretical UVW synthesised from the LTA online model vs recorded "
          "UVFITS.\nInputs: LTA RA-DATE/DEC-DATE + rate; times = UVFITS DATE "
          "(correct UTC).")
    print(f"Output plots -> {OUTDIR}\n")
    for name, path, blurb in SOURCES:
        g = load_geometry(path)
        rec, B, jd, gast, lon = g["rec"], g["B"], g["jd"], g["gast"], g["lon"]
        t_epoch = jd.mean()  # propagate rate from scan centre (rate*dt < 0.12 deg)
        syn = synth_from_lta(LTA[name], B, jd, gast, lon, t_epoch)
        stats = per_axis_stats(rec, syn)
        rot = rigid_rotation(rec, syn)
        fp = plot_source(name, blurb, rec, syn, stats, rot)

        t0, t1 = Time([jd.min(), jd.max()], format="jd", scale="utc").iso
        print(f"=== {name}  [{blurb}] ===")
        print(f"  N vis={len(rec)}  UTC {t0} .. {t1}  (from UVFITS DATE)")
        print(f"  LTA phase centre: RA={LTA[name]['ra']:.4f} Dec={LTA[name]['dec']:.4f}  "
              f"rate=({LTA[name]['dra']:.6f},{LTA[name]['ddec']:.6f}) deg/s")
        for ax in "UVW":
            s = stats[ax]
            print(f"    {ax}: slope={s['slope']:+.4f} intercept={s['intercept']:+9.1f} m "
                  f"corr={s['corr']:+.5f}  RMS(rec-syn)={s['rms']:9.1f} m")
        print(f"  rigid rotation syn->rec: angle={rot['angle']:.2f} deg det={rot['det']:+.3f} "
              f"fitRMS={rot['resid']:.1f} m  axis(U,V,W)="
              f"[{rot['axis'][0]:+.3f} {rot['axis'][1]:+.3f} {rot['axis'][2]:+.3f}]")
        print(f"  plot: {fp}\n")

    # --- wrong-time-base test on the offending scan ---------------------------
    print("=== moon0520: does a WRONG time base reproduce the recorded UVW? ===")
    g = load_geometry(m.MOON["moon0520"])
    rec, B, jd, lon = g["rec"], g["B"], g["jd"], g["lon"]
    blk = LTA["moon0520"]
    t_epoch = jd.mean()

    def resid_for_shift(dt_h):
        gast = gast_at(jd, dt_h)
        syn = synth_from_lta(blk, B, jd, gast, lon, t_epoch)
        return np.sqrt(np.mean(np.sum((syn - rec) ** 2, 1)))

    dts = np.linspace(-26.0, 26.0, 1041)
    rr = np.array([resid_for_shift(dt) for dt in dts])
    ibest = int(np.argmin(rr))
    print(f"  correct time (dt=0) residual : {resid_for_shift(0.0):9.1f} m")
    print(f"  best-fit time shift over +-26 h: dt={dts[ibest]:+.3f} h "
          f"(={dts[ibest]*15:+.1f} deg HA), residual {rr[ibest]:9.1f} m")
    for label, dt_h in [("+5.5 h (IST)", 5.5), ("-5.5 h", -5.5),
                        ("+24 h", 24.0), ("-24 h", -24.0)]:
        print(f"    dt={label:12s}: residual {resid_for_shift(dt_h):9.1f} m")
    print("  Interpretation: if NO time shift drives the residual to the "
          "~metre level\n  that the good scans reach, then a wrong time/RA "
          "alone cannot explain it\n  and the Dec axis is genuinely rotated "
          "too (a 2-axis rotation, not a clock error).")


if __name__ == "__main__":
    main()
