#!/usr/bin/env python
"""Per-snapshot UV-coverage movie for a Moon scan, colorized by frequency.

Reads a UVFITS once, builds one frame per unique timestamp (snapshot), and
draws the u,v samples of every baseline across the band (each baseline is a
radial streak over the 128 channels because UU/VV are stored in SECONDS and
uv_lambda = |uvw_sec| * nu). Four reference circles are drawn: 54 lambda
(half-flux), 108 lambda (unresolved 1/theta), 100 lambda (total-flux
threshold), 132 lambda (first null, 1.22/theta) for the Moon disk
theta = 0.531 deg on 2021-07-26.

Usage:
    gmrt/bin/python tools/dev/make_uv_coverage_movie.py moon0625

Writes moon_uv_movies/uv_coverage_<scan>.mp4 (and a _render_<scan>.done marker).
Uses only the repo tree for output; no /tmp, no root writes.
"""
import sys
import time
import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

# --- config ---------------------------------------------------------------
BASE = "/Users/raj030/DATA/gmrt_40_014/work/split/moon"
OUT = "moon_uv_movies"
HALF, DISK, TOTAL, FIRST_NULL = 54.0, 108.0, 100.0, 132.0  # lambda
CHSTEP = 8            # plot every 8th channel (counts still use all 128)
OFFENDING = {"moon0520", "moon0545", "moon0605"}


def main(scan):
    t0 = time.time()
    path = f"{BASE}/{scan}_primary_secondary_calibrated_flagged.uvfits"

    with fits.open(path, memmap=False) as h:
        g = h[0]
        hdr = g.header
        ptypes = [hdr[f"PTYPE{k+1}"] for k in range(hdr["PCOUNT"])]

        def par(prefix):
            i = [k for k, q in enumerate(ptypes) if q.startswith(prefix)][0]
            return np.asarray(g.data.par(i), "f8")

        uu, vv = par("UU"), par("VV")                       # seconds
        di = [k for k, q in enumerate(ptypes) if q == "DATE"]
        jd = np.asarray(g.data.par(di[0]), "f8") + np.asarray(g.data.par(di[1]), "f8")

        # frequency axis
        fi = [i for i in range(1, hdr["NAXIS"] + 1)
              if hdr.get(f"CTYPE{i}", "").strip().startswith("FREQ")][0]
        nch = hdr[f"NAXIS{fi}"]
        crv, cd, crp = hdr[f"CRVAL{fi}"], hdr[f"CDELT{fi}"], hdr[f"CRPIX{fi}"]

    nu = crv + (np.arange(1, nch + 1) - crp) * cd           # Hz, all channels
    nu_plot = nu[::CHSTEP]                                    # plotted channels
    nu_mean = nu.mean()

    # --- build per-snapshot frames ---------------------------------------
    stamps = np.unique(np.round(jd, 10))
    frames = []
    for tv in stamps:
        m = np.isclose(jd, tv, atol=1e-9)
        us, vs = uu[m], vv[m]
        r = np.hypot(np.outer(us, nu), np.outer(vs, nu))     # all channels
        n_total = int((r < TOTAL).sum())
        n_null = int((r < FIRST_NULL).sum())
        n_bl = int((np.hypot(us * nu_mean, vs * nu_mean) < TOTAL).sum())
        up = np.outer(us, nu_plot).ravel()
        vp = np.outer(vs, nu_plot).ravel()
        col = np.tile(nu_plot / 1e6, len(us))
        # plot conjugate points too
        off = np.column_stack([np.r_[up, -up], np.r_[vp, -vp]])
        frames.append((off, np.r_[col, col], n_total, n_null, n_bl))

    # --- figure scaffold (persistent artists) ----------------------------
    tag = "OFFENDING (UVW ~126-141 deg off Moon)" if scan in OFFENDING \
        else "on-Moon (correctly fringe-stopped)"
    amax = max(np.abs(np.concatenate([f[0].ravel() for f in frames])))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 6.2))
    fig.subplots_adjust(left=0.07, right=0.9, bottom=0.09, top=0.9, wspace=0.25)

    scs = []
    for ax, lim, title in ((axL, amax, "full uv coverage"),
                           (axR, 300, "total-flux region (zoom)")):
        sc = ax.scatter(frames[0][0][:, 0], frames[0][0][:, 1], c=frames[0][1],
                        s=4, cmap="turbo", vmin=nu.min() / 1e6, vmax=nu.max() / 1e6,
                        alpha=0.8, edgecolors="none")
        scs.append(sc)
        for rad, st in ((HALF, ":"), (DISK, "-."), (TOTAL, "-"), (FIRST_NULL, "--")):
            ax.add_patch(Circle((0, 0), rad, fill=False, ec="k", ls=st, lw=1.2))
        ax.set_aspect("equal")
        ax.axhline(0, color="grey", lw=0.4)
        ax.axvline(0, color="grey", lw=0.4)
        ax.set_xlabel(r"u ($\lambda$)")
        ax.set_ylabel(r"v ($\lambda$)")
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

    leg = [Line2D([0], [0], color="k", ls=":", label=r"54$\lambda$ half-flux"),
           Line2D([0], [0], color="k", ls="-.", label=r"108$\lambda$ unresolved"),
           Line2D([0], [0], color="k", ls="-", label=r"100$\lambda$ total-flux"),
           Line2D([0], [0], color="k", ls="--", label=r"132$\lambda$ first null")]
    axR.legend(handles=leg, loc="upper right", fontsize=7, framealpha=0.9)
    cb = fig.colorbar(scs[0], ax=[axL, axR], fraction=0.025, pad=0.02)
    cb.set_label("frequency (MHz)", fontsize=9)
    sup = fig.suptitle("", fontsize=11)

    # --- render ----------------------------------------------------------
    w = FFMpegWriter(fps=8, bitrate=2000)
    outfile = f"{OUT}/uv_coverage_{scan}.mp4"
    with w.saving(fig, outfile, dpi=95):
        for k, (off, cc, n_total, n_null, n_bl) in enumerate(frames):
            for sc in scs:
                sc.set_offsets(off)
                sc.set_array(cc)
            sup.set_text(f"{scan}  [{tag}]    snapshot {k+1}/{len(frames)}    "
                         f"in 100$\\lambda$: {n_total} ({n_bl} distinct BL)    "
                         f"in 132$\\lambda$: {n_null}")
            w.grab_frame()
    plt.close(fig)

    with open(f"{OUT}/_render_{scan}.done", "w") as fh:
        fh.write(f"{len(frames)} frames, {time.time() - t0:.0f}s")
    print(f"wrote {outfile}  ({len(frames)} frames, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: make_uv_coverage_movie.py <scan>  "
                 "(e.g. moon0520 moon0545 moon0605 moon0625 moon0635)")
    main(sys.argv[1])
