#!/usr/bin/env python
"""Read-only: per-snapshot UV-coverage movie for a Moon scan, RECORDED vs
CORRECTED UVW side by side, colorized by frequency (all 128 channels).

The recorded UVW of the with-rate scans (moon0520/0545/0605) carry a constant
159 deg rigid rotation (see moon_fringe_stopping_investigation.md), so a movie
made from them shows the wrong coverage -- and, because the rotated direction
sits low on the sky, baselines foreshorten and pile up spuriously at short uv.
The corrected UVW is synthesised from the GSB LTA online model (RA-DATE/DEC-DATE
+ rate, propagated over the scan) with the calibrator-certified forward model
(moon_uvw_checks / moon_uvw_synthesis_compare) -- i.e. the coverage the scan
WOULD have with correct geometry, recoverable by a fresh UVW recompute.

Two columns: LEFT = recorded (rotated) UVW, RIGHT = corrected UVW, zoomed to the
Moon-disk region (|uv| < 300 lambda). All 128 channels turn every baseline into a
short radial streak (uv_lambda = (uvw_metres / c) * nu; the 314-331 MHz band spans
only ~5% in radius). ONE reference circle marks the single boundary that dictates
whether the Moon's total flux is recoverable: the first null at 132 lambda =
1.22/theta (theta = 0.531 deg on 2021-07-26), the Bessel zero where the uniform
disk's visibility main lobe ends -- baselines beyond it have resolved the disk out
and carry no total-flux information. The in-null sample count is annotated per
column so the recovered short-spacing coverage is quantified live.

Rendering: mp4 via FFMpegWriter (seconds), then a palette-based ffmpeg pass to a
GIF that renders inline in any markdown viewer. No source/pipeline/data files
are modified; all output stays under moon_uv_movies/.

Usage:
    gmrt/bin/python tools/dev/moon_uv_coverage_corrected_movie.py moon0520
"""
import os
import sys
import time
import shutil
import argparse
import subprocess
import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moon_uvw_checks as m                      # noqa: E402  certified geometry
from moon_uvw_synthesis_compare import synth_from_lta  # noqa: E402

C = 299792458.0
OUT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "..", "moon_uv_movies"))
# Single physical boundary: the first null of the 0.531 deg uniform-disk
# visibility, 1.22/theta = 132 lambda. Inside it the disk's visibility main lobe
# carries total-flux information; beyond it the disk is resolved out. This is the
# one non-arbitrary boundary that dictates whether the Moon's total flux is
# recoverable (no arbitrary 100-lambda rule of thumb).
FIRST_NULL = 132.0                                          # lambda (1.22/theta)
ZOOM = 300.0                                                # lambda (uv half-window)

# GSB LTA online-model source blocks (apparent-of-date deg; rates deg/s).
LTA = {
    "moon0520": dict(ra=331.218408, dec=-17.384986, dra=0.000103, ddec=0.000067),
    "moon0545": dict(ra=331.383452, dec=-17.283068, dra=0.000108, ddec=0.000069),
    "moon0605": dict(ra=331.521323, dec=-17.200152, dra=0.000112, ddec=0.000070),
    "moon0625": dict(ra=331.664774, dec=-17.116175, dra=0.000000, ddec=0.000000),
    "moon0635": dict(ra=331.738688, dec=-17.073822, dra=0.000000, ddec=0.000000),
}
WITH_RATE = {"moon0520", "moon0545", "moon0605"}


def freq_axis(path):
    with fits.open(path, memmap=False) as h:
        hdr = h[0].header
        fi = [i for i in range(1, hdr["NAXIS"] + 1)
              if hdr.get(f"CTYPE{i}", "").strip().startswith("FREQ")][0]
        nch = hdr[f"NAXIS{fi}"]
        crv, cd, crp = hdr[f"CRVAL{fi}"], hdr[f"CDELT{fi}"], hdr[f"CRPIX{fi}"]
    return crv + (np.arange(1, nch + 1) - crp) * cd          # Hz


def frame_masks(jd):
    """Boolean row-mask per snapshot (cheap; built once). jd is a full Julian
    Date (~2.46e6): all rows in one integration share it exactly, and
    np.isclose's default rtol would span ~24 days, so match exactly."""
    return [jd == tv for tv in np.unique(jd)]


def one_frame(uu, vv, nu, mask):
    """Build a single snapshot's plot layers on demand (keeps memory to one
    frame -- precomputing all 113 x 2 datasets OOM-kills)."""
    nu_mean = nu.mean()
    us, vs = uu[mask], vv[mask]
    r = np.hypot(np.outer(us, nu), np.outer(vs, nu))
    n_null = int((r < FIRST_NULL).sum())
    n_bl = int((np.hypot(us * nu_mean, vs * nu_mean) < FIRST_NULL).sum())
    up = np.outer(us, nu).ravel()
    vp = np.outer(vs, nu).ravel()
    col = np.tile(nu / 1e6, len(us))
    off = np.column_stack([np.r_[up, -up], np.r_[vp, -vp]])
    return off, np.r_[col, col], n_null, n_bl


def main(scan, fps):
    if scan not in LTA:
        sys.exit(f"unknown scan {scan!r}; choose from {sorted(LTA)}")
    t0 = time.time()
    path = m.MOON[scan]

    # geometry + per-integration UTC (autocorr already dropped in _load_full)
    rec_m, B, jd, gast, lon, a1, a2, app = m._load_full(path)
    syn_m = synth_from_lta(LTA[scan], B, jd, gast, lon, jd.mean())  # corrected
    nu = freq_axis(path)                                            # all 128 ch

    uu_rec, vv_rec = rec_m[:, 0] / C, rec_m[:, 1] / C
    uu_cor, vv_cor = syn_m[:, 0] / C, syn_m[:, 1] / C
    masks = frame_masks(jd)
    nframes = len(masks)

    # Zoom to the Moon-disk region (0-300 lambda): the whole point is the disk
    # imprint and where the disk resolves out at the 132 lambda first null.
    amax = ZOOM

    mode = "with rate" if scan in WITH_RATE else "no rate"
    cols = [("recorded UVW  (as in UVFITS, rotated)", uu_rec, vv_rec, "C3"),
            ("corrected UVW  (true-Moon geometry)", uu_cor, vv_cor, "C2")]

    f0 = [one_frame(uu, vv, nu, masks[0]) for _, uu, vv, _ in cols]
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 6.6))
    fig.subplots_adjust(left=0.06, right=0.9, bottom=0.09, top=0.88, wspace=0.2)
    scs = []
    for ax, (title, _, _, tc), fr in zip(axs, cols, f0):
        sc = ax.scatter(fr[0][:, 0], fr[0][:, 1], c=fr[1],
                        s=12, cmap="turbo", vmin=nu.min() / 1e6, vmax=nu.max() / 1e6,
                        alpha=0.85, edgecolors="none")
        scs.append(sc)
        ax.add_patch(Circle((0, 0), FIRST_NULL, fill=False, ec="k", ls="--", lw=1.2))
        ax.set_aspect("equal")
        ax.axhline(0, color="grey", lw=0.4)
        ax.axvline(0, color="grey", lw=0.4)
        ax.set_xlabel(r"u ($\lambda$)")
        ax.set_ylabel(r"v ($\lambda$)")
        ax.set_title(title, fontsize=10, color=tc)
        ax.set_xlim(-amax, amax)
        ax.set_ylim(-amax, amax)

    leg = [Line2D([0], [0], color="k", ls="--",
                  label=r"132$\lambda$ first null (1.22/$\theta$) — total-flux limit")]
    axs[1].legend(handles=leg, loc="upper right", fontsize=7, framealpha=0.9)
    cb = fig.colorbar(scs[0], ax=list(axs), fraction=0.025, pad=0.02)
    cb.set_label("frequency (MHz)", fontsize=9)
    sup = fig.suptitle("", fontsize=11)
    ann = [axs[0].text(0.02, 0.98, "", transform=axs[0].transAxes, va="top",
                       fontsize=8.5, color="C3"),
           axs[1].text(0.02, 0.98, "", transform=axs[1].transAxes, va="top",
                       fontsize=8.5, color="C2")]

    os.makedirs(OUT, exist_ok=True)
    mp4 = os.path.join(OUT, f"uv_coverage_corrected_{scan}.mp4")
    w = FFMpegWriter(fps=fps, bitrate=2600)
    with w.saving(fig, mp4, dpi=95):
        for k in range(nframes):
            frames = [one_frame(uu, vv, nu, masks[k]) for _, uu, vv, _ in cols]
            for sc, fr in zip(scs, frames):
                sc.set_offsets(fr[0])
                sc.set_array(fr[1])
            for a, fr in zip(ann, frames):
                _, _, n_null, n_bl = fr
                a.set_text(f"nVis (per channel) < 132$\\lambda$: {n_null}\n"
                           f"from {n_bl} distinct baselines "
                           f"($\\times${len(nu)} ch)")
            sup.set_text(f"{scan}  [{mode}]   snapshot {k+1}/{nframes}   "
                         f"recorded vs corrected UVW  (Moon-disk region, "
                         f"|uv| < {ZOOM:.0f}$\\lambda$; all 128 ch, "
                         f"{nu.min()/1e6:.0f}-{nu.max()/1e6:.0f} MHz)")
            w.grab_frame()
    plt.close(fig)

    # mp4 -> GIF via a two-pass palette (fast, good quality) for .md embedding
    gif = os.path.join(OUT, f"uv_coverage_corrected_{scan}.gif")
    ff = shutil.which("ffmpeg")
    if ff:
        pal = os.path.join(OUT, f"_palette_{scan}.png")
        vf = f"fps={fps},scale=1000:-1:flags=lanczos"
        subprocess.run([ff, "-y", "-i", mp4, "-vf", f"{vf},palettegen", pal],
                       check=True, capture_output=True)
        subprocess.run([ff, "-y", "-i", mp4, "-i", pal, "-lavfi",
                        f"{vf}[x];[x][1:v]paletteuse", gif],
                       check=True, capture_output=True)
        os.remove(pal)
        sz = os.path.getsize(gif) / 1e6
        print(f"wrote {gif}  ({nframes} frames, {sz:.1f} MB, "
              f"{time.time() - t0:.0f}s)   [mp4 also at {mp4}]")
    else:
        print(f"wrote {mp4}  ({nframes} frames, {time.time() - t0:.0f}s); "
              f"ffmpeg not found -> no GIF made")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan", help="moon0520 moon0545 moon0605 moon0625 moon0635")
    ap.add_argument("--fps", type=int, default=8, help="frames per second")
    a = ap.parse_args()
    main(a.scan, a.fps)
