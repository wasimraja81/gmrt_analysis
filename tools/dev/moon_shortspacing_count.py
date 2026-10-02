#!/usr/bin/env python
"""Read-only: count short-spacing coverage for the Moon scans at the CORRECTED
(true-Moon) geometry, making the per-channel visibility count explicit.

For each scan we synthesise the corrected UVW from the LTA online model (as in
moon_uvw_synthesis_compare / the coverage movie), turn every baseline into its
128 per-channel uv points (uv_lambda = |uvw_metres/c| * nu over 314-331 MHz),
and count how many *visibilities* (baseline x channel) fall inside a given
radius over the whole track -- versus how many *distinct baselines* that is.

This is the number the §5 table and the movie annotations report; running it at
both 100 and 132 lambda shows how the count moves with the boundary choice.
Nothing is modified; prints only.

Usage:
    gmrt/bin/python tools/dev/moon_shortspacing_count.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moon_uvw_checks as m                       # noqa: E402  certified geometry
from moon_uvw_synthesis_compare import synth_from_lta  # noqa: E402
from moon_uv_coverage_corrected_movie import LTA, freq_axis, frame_masks  # noqa: E402

C = 299792458.0
RADII = (100.0, 132.0)   # lambda: the old rule-of-thumb vs the 1.22/theta null


def main():
    print("Corrected (true-Moon) short-spacing coverage, per-channel visibilities.")
    print("uv_lambda = |uvw_metres/c| * nu over 128 channels (314-331 MHz).\n")
    hdr = (f"{'scan':9s} {'min(min)':>7s} {'snaps':>6s} {'nchan':>6s} "
           f"{'minuv_lam':>9s}")
    for R in RADII:
        hdr += f" | <{R:.0f}lam: nVis  (BL/snap, chan-vis)"
    print(hdr)
    for scan in ("moon0520", "moon0545", "moon0605", "moon0625", "moon0635"):
        path = m.MOON[scan]
        rec_m, B, jd, gast, lon, a1, a2, app = m._load_full(path)
        syn_m = synth_from_lta(LTA[scan], B, jd, gast, lon, jd.mean())
        nu = freq_axis(path)
        nchan = len(nu)
        uu, vv = syn_m[:, 0] / C, syn_m[:, 1] / C          # seconds (corrected)
        ru, rv = rec_m[:, 0] / C, rec_m[:, 1] / C          # seconds (recorded)
        masks = frame_masks(jd)
        nsnap = len(masks)
        dur_min = (jd.max() - jd.min()) * 24 * 60
        # per-channel radius for every (row, channel)
        r_all = np.hypot(np.outer(uu, nu), np.outer(vv, nu))   # (nrow, nchan)
        r_bl = np.hypot(uu * nu.mean(), vv * nu.mean())        # per-row (mean nu)
        r_bl_rec = np.hypot(ru * nu.mean(), rv * nu.mean())    # recorded per-row
        minuv = r_all.min()
        line = (f"{scan:9s} {dur_min:7.1f} {nsnap:6d} {nchan:6d} {minuv:9.1f}")
        for R in RADII:
            nvis = int((r_all < R).sum())                  # per-channel visibilities
            nbl_rows = int((r_bl < R).sum())               # baseline-integrations
            nbl_rec = int((r_bl_rec < R).sum())            # recorded (offending)
            line += (f" | {nvis:7d}  (cor {nbl_rows/nsnap:4.1f}, "
                     f"rec {nbl_rec/nsnap:4.1f} BL/snap)")
        print(line)
    print("\nPer boundary: nVis = baseline-integrations x channels inside R over the "
          "whole track (corrected geometry);\n  cor/rec BL/snap = mean baseline-"
          "integrations per snapshot inside R for the corrected vs recorded UVW\n  "
          "(recorded > corrected = the spurious low-elevation short-spacing pile-up).")


if __name__ == "__main__":
    main()
