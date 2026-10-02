#!/usr/bin/env python
"""Read-only: verify the GSB LTA per-scan rate values and settle their UNITS.

Anchors from the data (moon0520, MJD_SRC = 59420.0):
  SU RAEPO/DECEPO (J2000) = 330.924, -17.490
  SU RAAPP/DECAPP (apparent of date) = 331.220, -17.385
  LTA RA-DATE/DEC-DATE = 331.218408, -17.384986
  LTA DRA/DT, DDEC/DT = 0.000103, 0.000067   (no unit token in file)

Step 1: reproduce the Moon's position in several frames (geocentric vs
topocentric; J2000 vs apparent-of-date) and see which matches the anchors,
to pick the correct frame. Step 2: compute the rate in that frame and settle
the unit. Nothing is modified.
"""
import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import (EarthLocation, TETE, GCRS, PrecessedGeocentric,
                                  get_body, solar_system_ephemeris)

GMRT = EarthLocation(lat=19.0931 * u.deg, lon=74.0507 * u.deg, height=656 * u.m)

MJD0 = 59420.0
ANCHOR = dict(raepo=330.924, decepo=-17.490, raapp=331.220, decapp=-17.385)


def positions(t):
    out = {}
    # geocentric GCRS (~J2000 axes, geocentric direction)
    mg = get_body("moon", t)                    # geocentric
    out["GCRS geo"] = (mg.ra.deg, mg.dec.deg)
    # topocentric GCRS
    mt = get_body("moon", t, GMRT)              # topocentric
    out["GCRS topo"] = (mt.ra.deg, mt.dec.deg)
    # apparent of date, geocentric and topocentric
    out["TETE geo"] = _rd(mg.transform_to(TETE(obstime=t)))
    out["TETE topo"] = _rd(mt.transform_to(TETE(obstime=t, location=GMRT)))
    return out


def _rd(c):
    return (c.ra.deg, c.dec.deg)


def rate(frame, t, dt_s=120.0, topo=False):
    t0 = Time(t, format="mjd", scale="utc")
    tm, tp = t0 - dt_s/2*u.s, t0 + dt_s/2*u.s
    loc = GMRT if topo else None
    def rd(tt):
        m = get_body("moon", tt, loc) if topo else get_body("moon", tt)
        if frame == "GCRS":
            return m.ra.deg, m.dec.deg
        return _rd(m.transform_to(TETE(obstime=tt, location=loc) if topo else TETE(obstime=tt)))
    ra0, dec0 = rd(tm); ra1, dec1 = rd(tp)
    dra = ((ra1 - ra0 + 180) % 360) - 180
    return dra/dt_s, (dec1 - dec0)/dt_s


if __name__ == "__main__":
    solar_system_ephemeris.set("builtin")
    t = Time(MJD0, format="mjd", scale="utc")
    print(f"Anchors: RAEPO(J2000)={ANCHOR['raepo']}  RAAPP(app)={ANCHOR['raapp']}  "
          f"DECAPP={ANCHOR['decapp']}\n")
    print("Moon position by frame at MJD 59420.0:")
    for k, (ra, dec) in positions(t).items():
        dra_epo = ((ra - ANCHOR['raepo'] + 180) % 360) - 180
        dra_app = ((ra - ANCHOR['raapp'] + 180) % 360) - 180
        print(f"  {k:11}: RA={ra:9.4f} Dec={dec:8.4f}   "
              f"(RA-RAEPO={dra_epo:+.3f}, RA-RAAPP={dra_app:+.3f})")

    print("\nRate by frame (deg/s) and vs LTA 0.000103 / 0.000067:")
    for label, (fr, topo) in {"GCRS geo": ("GCRS", False),
                               "GCRS topo": ("GCRS", True),
                               "TETE geo": ("TETE", False),
                               "TETE topo": ("TETE", True)}.items():
        dra, ddec = rate(fr, MJD0, topo=topo)
        print(f"  {label:11}: dRA/dt={dra:.6g} dDec/dt={ddec:.6g} deg/s "
              f"| LTA/true: dRA={0.000103/dra:5.3f} dDec={0.000067/ddec:5.3f} "
              f"| dRA={dra*3600*3600:6.0f} dDec={ddec*3600*3600:6.0f} asec/hr")

    print("\nUnit test (best-matching frame): does 0.000103 == the true rate in deg/s?")
    dra_geo, _ = rate("TETE", MJD0, topo=False)
    for label, val in {"deg/s": 0.000103,
                       "rad/hr->deg/s": 0.000103*(180/np.pi)/3600,
                       "deg/hr->deg/s": 0.000103/3600}.items():
        print(f"   LTA as {label:14}: {val:.6g} deg/s (true_geo/this={dra_geo/val:8.2f})")
