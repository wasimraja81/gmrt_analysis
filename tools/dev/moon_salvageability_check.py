#!/usr/bin/env python
"""Read-only salvageability probe for the GMRT 40_014 Moon scans.

Question: for a WITH-rate (offending) scan whose recorded UVW point ~126 deg off
the Moon, were the *visibilities* fringe-stopped on the Moon (only the UVW
metadata is wrong -> salvageable) or fringe-stopped 126 deg off (decorrelated ->
not salvageable)?

Test: on the shortest PHYSICAL baselines (where the near-unresolved Moon should
dominate), compare the coherent (vector-averaged) visibility to the scalar-
averaged amplitude. A source at the phase centre averages coherently
(coherence ~ 1); a source 126 deg away winds in both time and frequency and
averages to ~0 (coherence ~ 0) even though its power is still present (scalar
amplitude high). Compare a WITH-rate scan (moon0520) against a NO-rate scan
(moon0625) that is known to be correctly stopped on the Moon.

No files are modified. Reads UVFITS only.
"""
import sys
import numpy as np
from astropy.io import fits

C = 299792458.0

FILES = {
    "moon0520 (WITH rate, offending)":
        "/Users/raj030/DATA/gmrt_40_014/work/split/moon/moon0520_primary_secondary_calibrated_flagged.uvfits",
    "moon0625 (NO rate, good/control)":
        "/Users/raj030/DATA/gmrt_40_014/work/split/moon/moon0625_primary_secondary_calibrated_flagged.uvfits",
}


def load(path):
    h = fits.open(path, memmap=True)
    g = h[0]
    hdr = g.header
    # frequency axis (AX4): freq(ch) = CRVAL + (ch+1 - CRPIX)*CDELT
    crval = hdr["CRVAL4"]; crpix = hdr["CRPIX4"]; cdelt = hdr["CDELT4"]
    nch = hdr["NAXIS4"]
    freqs = crval + (np.arange(nch) + 1 - crpix) * cdelt

    # antenna positions
    an = None
    for hd in h:
        if hd.name == "AIPS AN":
            an = hd.data
            break
    nosta = np.array(an["NOSTA"])
    xyz = np.array(an["STABXYZ"])  # metres, geocentric-equatorial local frame
    pos = {int(n): xyz[i] for i, n in enumerate(nosta)}

    par = g.data.parnames
    def col(name):
        return np.array(g.data.par(name))
    bl = col("BASELINE")
    a1 = (bl // 256).astype(int)
    a2 = (bl % 256).astype(int)

    data = np.array(g.data.data)  # (N,1,1,1,nch,2,3)
    data = data[:, 0, 0, 0, :, :, :]  # (N, nch, 2pol, 3)
    re = data[..., 0]; im = data[..., 1]; wt = data[..., 2]
    vis = re + 1j * im
    h.close()
    return dict(freqs=freqs, a1=a1, a2=a2, pos=pos, vis=vis, wt=wt)


def analyse(label, path, nshort=15, maxlambda=132.0):
    d = load(path)
    freqs = d["freqs"]; fc = freqs.mean()
    lam_c = C / fc
    a1, a2, pos = d["a1"], d["a2"], d["pos"]
    vis, wt = d["vis"], d["wt"]  # (N,nch,2)

    # physical baseline length (m) per row from antenna positions
    keys = list(zip(a1.tolist(), a2.tolist()))
    ukeys = sorted(set(k for k in keys if k[0] in pos and k[1] in pos and k[0] != k[1]))
    blen = {}
    for k in ukeys:
        b = pos[k[0]] - pos[k[1]]
        blen[k] = float(np.linalg.norm(b))
    short = sorted(ukeys, key=lambda k: blen[k])[:nshort]

    print(f"\n=== {label} ===")
    print(f"  band centre {fc/1e6:.3f} MHz  (lambda_c = {lam_c:.3f} m); "
          f"{len(ukeys)} baselines, showing shortest {nshort}")
    print(f"  {'ant1-ant2':>10} {'|B| (m)':>8} {'|B| (lam)':>9} {'Nint':>5} "
          f"{'scalarAmp':>10} {'|vecAvg|':>9} {'coherence':>9}")

    print(f"    (freqCoh = per-integration coherence across the 33 MHz band = delay/freq-slope test;")
    print(f"     timeCoh = per-channel coherence across the scan = fringe-rate/time-slope test)")
    pol = 0  # RR (stokes -1)
    rows_scal = []; rows_vec = []; cohs = []; lams = []
    for k in short:
        sel = (a1 == k[0]) & (a2 == k[1])
        if sel.sum() == 0:
            continue
        v = vis[sel][:, :, pol]   # (nint, nch)
        w = wt[sel][:, :, pol]    # (nint, nch)
        m = w > 0
        if m.sum() == 0:
            continue
        vv = v[m]; ww = w[m]
        scalar = np.average(np.abs(vv), weights=ww)
        vec = np.abs(np.sum(ww * vv) / np.sum(ww))
        coh = vec / scalar if scalar > 0 else np.nan

        # frequency-only coherence: collapse each integration across channels,
        # isolating the delay (phase-vs-frequency) term
        wm = np.where(m, w, 0.0)
        vm = np.where(m, v, 0.0)
        num_f = np.abs(np.sum(wm * vm, axis=1))          # per integration
        den_f = np.sum(wm * np.abs(vm), axis=1)
        good_f = den_f > 0
        freq_coh = np.median(num_f[good_f] / den_f[good_f]) if good_f.any() else np.nan
        # time-only coherence: collapse each channel across integrations,
        # isolating the fringe-rate (phase-vs-time) term
        num_t = np.abs(np.sum(wm * vm, axis=0))          # per channel
        den_t = np.sum(wm * np.abs(vm), axis=0)
        good_t = den_t > 0
        time_coh = np.median(num_t[good_t] / den_t[good_t]) if good_t.any() else np.nan

        blam = blen[k] / lam_c
        print(f"  {k[0]:>3d}-{k[1]:<3d}   {blen[k]:8.1f} {blam:9.1f} "
              f"{sel.sum():5d} {scalar:10.4f} {vec:9.4f} {coh:9.3f}"
              f"  freqCoh={freq_coh:6.3f} timeCoh={time_coh:6.3f}")
        rows_scal.append(scalar); rows_vec.append(vec); cohs.append(coh); lams.append(blam)

    rows_scal = np.array(rows_scal); rows_vec = np.array(rows_vec); cohs = np.array(cohs)
    lams = np.array(lams)
    # focus on the genuinely short (<maxlambda) baselines for the summary
    fsel = lams < maxlambda
    print(f"  -- summary over baselines < {maxlambda:.0f} lambda "
          f"({fsel.sum()} of {len(lams)}):")
    if fsel.sum():
        print(f"     median scalar amp = {np.median(rows_scal[fsel]):.4f}")
        print(f"     median |vec avg|  = {np.median(rows_vec[fsel]):.4f}")
        print(f"     median coherence  = {np.median(cohs[fsel]):.3f}")
    print(f"  -- summary over all {len(lams)} shortest baselines shown:")
    print(f"     median scalar amp = {np.median(rows_scal):.4f}")
    print(f"     median |vec avg|  = {np.median(rows_vec):.4f}")
    print(f"     median coherence  = {np.median(cohs):.3f}")
    return dict(coh=cohs, lam=lams, scalar=rows_scal, vec=rows_vec)


if __name__ == "__main__":
    for label, path in FILES.items():
        analyse(label, path)
    print("\nInterpretation:")
    print("  coherence ~1 on short baselines  -> visibilities stopped on Moon (salvageable)")
    print("  coherence ~0 with high scalar amp -> Moon power present but stopped off-Moon (decorrelated)")
