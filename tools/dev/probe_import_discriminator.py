#!/usr/bin/env python
"""Read-only probe: which SU-table / primary-header fields discriminate the
CASA importuvfits success/failure across the calibrator + 5 Moon split files.

Prints, per file: PMRA/PMDEC (SU table), VELDEF/SPECSYS (primary + SU),
EPOCH/EQUINOX, and the RA/DEC axis CTYPEs. No files are modified.
"""
import numpy as np
from astropy.io import fits

BASE = "/Users/raj030/DATA/gmrt_40_014/work/split"
FILES = [
    ("cal      ", f"{BASE}/3c468.1/3c468.1_primary_secondary_calibrated_flagged.uvfits", "OK"),
    ("moon0520 ", f"{BASE}/moon/moon0520_primary_secondary_calibrated_flagged.uvfits", "FAIL"),
    ("moon0545 ", f"{BASE}/moon/moon0545_primary_secondary_calibrated_flagged.uvfits", "FAIL"),
    ("moon0605 ", f"{BASE}/moon/moon0605_primary_secondary_calibrated_flagged.uvfits", "FAIL"),
    ("moon0625 ", f"{BASE}/moon/moon0625_primary_secondary_calibrated_flagged.uvfits", "OK"),
    ("moon0635 ", f"{BASE}/moon/moon0635_primary_secondary_calibrated_flagged.uvfits", "untested"),
]


def getcol(tab, name):
    try:
        return np.asarray(tab.data[name]).ravel()
    except KeyError:
        return None


print(f"{'scan':9} {'import':9} {'PMRA':>10} {'PMDEC':>10} "
      f"{'VELDEF(pri)':>12} {'VELDEF(SU)':>11} {'EPOCH':>8}")
print("-" * 82)

for tag, path, status in FILES:
    with fits.open(path, memmap=False) as h:
        pri = h[0].header
        su = None
        for hdu in h[1:]:
            if hdu.header.get("EXTNAME", "").strip() == "AIPS SU":
                su = hdu
                break
        pmra = getcol(su, "PMRA") if su is not None else None
        pmdec = getcol(su, "PMDEC") if su is not None else None
        epoch = getcol(su, "EPOCH") if su is not None else None
        veldef_pri = pri.get("VELDEF", "-")
        veldef_su = su.header.get("VELDEF", "-") if su is not None else "-"
        pr = f"{pmra[0]:10.4f}" if pmra is not None else f"{'none':>10}"
        pd = f"{pmdec[0]:10.4f}" if pmdec is not None else f"{'none':>10}"
        ep = f"{epoch[0]:8.1f}" if epoch is not None else f"{'-':>8}"
        print(f"{tag} {status:9} {pr} {pd} {str(veldef_pri):>12} "
              f"{str(veldef_su):>11} {ep}")
