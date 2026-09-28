"""Checking a file's timestamps against its own u, v, w.

A correlator computes u, v, w for each integration from the antenna
positions, the source direction and the time. Predicting u, v, w the same way
from the AN table's positions and the SU table's coordinates, for the row's
timestamp shifted by `dt`, and fitting `dt` to the stored values, measures
recorded time - UTC as the correlator used it. The file's keywords declare
the same offset (`TimeReference.recorded_minus_utc_s`); `check_timestamps`
compares the two.

Each of `N_INTEGRATIONS` integrations spread over the file is fitted on its
own, in the J2000 frame and in the apparent frame of date, with both
baseline signs (ant2 - ant1 and ant1 - ant2); integrations whose best fit
leaves more than `RMS_LIMIT_M` (a moving source such as the Moon, whose SU
coordinates are not what the correlator tracked) are left out. The result is
the median over the rest, in the frame most of them fit.

What this measures is the time the correlator's u, v, w belong to; whether
the correlator's own clock was right is outside what the u, v, w can show.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import erfa
import numpy as np

from data_io.astrometry import DEFAULT_UT1, Ut1Provider

SPEED_OF_LIGHT_M_PER_S = 299_792_458.0
N_INTEGRATIONS = 12
# An integration counts only if its best fit leaves at most this rms: a correct model fits the
# stored float32 u, v, w to millimetres, while 1 s of time moves a 25 km baseline's u, v by ~1.8 m.
RMS_LIMIT_M = 1.0
# Measured and declared offsets further apart than this disagree: on the GWB file the per-integration
# fits spread over 0.06 s, so 0.1 s is just above what the method resolves.
DEFAULT_TOLERANCE_S = 0.1
MIN_INTEGRATIONS = 3
SEARCH_S = 120.0  # offsets tried: -SEARCH_S to +SEARCH_S
FRAMES = ("J2000", "apparent")


@dataclass(frozen=True)
class TimestampCheck:
    declared_s: float  # recorded - UTC by the file's keywords
    measured_s: float | None  # recorded - UTC implied by the u, v, w; None: not measurable
    spread_s: float | None  # max - min of the per-integration fits used
    rms_m: float | None  # median rms residual at the fits used
    frame: str | None  # the frame the stored u, v, w fit: "J2000" or "apparent"
    n_used: int  # integrations that fit within RMS_LIMIT_M, in that frame
    n_tried: int
    tolerance_s: float

    @property
    def agrees(self) -> bool | None:
        """Whether measured and declared agree within the tolerance; None if
        the offset could not be measured or the file declares none."""
        if self.measured_s is None or not np.isfinite(self.declared_s):
            return None
        return abs(self.measured_s - self.declared_s) <= self.tolerance_s

    def summary(self) -> str:
        declared = (f"the file declares {self.declared_s:g} s" if np.isfinite(self.declared_s)
                    else "the file's keywords give no offset")
        if self.measured_s is None:
            return (f"timestamps: recorded - UTC could not be measured from u, v, w ({self.n_used} of "
                    f"{self.n_tried} integrations fit within {RMS_LIMIT_M:g} m); {declared}")
        measured = (f"timestamps: u, v, w imply recorded - UTC = {self.measured_s:.3f} s ({self.n_used} "
                    f"integrations, spread {self.spread_s:.3f} s, {self.frame} frame, rms {self.rms_m:.3f} m); "
                    f"{declared}")
        if not np.isfinite(self.declared_s):
            return measured
        gap = abs(self.measured_s - self.declared_s)
        verdict = "within" if self.agrees else "above"
        return f"{measured}: {gap:.3f} s apart, {verdict} the {self.tolerance_s:g} s tolerance"


def check_timestamps(index, antennas, sources, declared_s: float, n_integrations: int = N_INTEGRATIONS,
                     tolerance_s: float = DEFAULT_TOLERANCE_S, ut1: Ut1Provider | None = None) -> TimestampCheck:
    """Measure recorded time - UTC from `index`'s u, v, w (a `RowIndex`),
    `antennas` (`read_antenna_table`) and `sources` (`read_source_table`),
    and compare it with `declared_s`."""
    ut1 = ut1 or DEFAULT_UT1
    positions = {a.station_number: np.array([a.x_m, a.y_m, a.z_m]) for a in antennas}
    bounds = np.asarray(index.integration_boundaries)
    n_total = len(bounds) - 1
    picks = np.unique(np.linspace(0, n_total - 1, min(n_integrations, n_total)).astype(int)) if n_total > 0 else []
    fits = []
    for i in picks:
        fit = _fit_integration(index, positions, sources, int(bounds[i]), int(bounds[i + 1]), ut1)
        if fit is not None:
            fits.append(fit)
    good = [f for f in fits if f[0] <= RMS_LIMIT_M]
    frame = Counter(f[2] for f in good).most_common(1)[0][0] if good else None
    used = [f for f in good if f[2] == frame]
    if len(used) < MIN_INTEGRATIONS:
        return TimestampCheck(declared_s, None, None, None, None, len(used), len(picks), tolerance_s)
    offsets = np.array([f[1] for f in used])
    return TimestampCheck(
        declared_s=declared_s, measured_s=float(np.median(offsets)), spread_s=float(offsets.max() - offsets.min()),
        rms_m=float(np.median([f[0] for f in used])), frame=frame, n_used=len(used), n_tried=len(picks),
        tolerance_s=tolerance_s,
    )


def _fit_integration(index, positions, sources, start: int, stop: int, ut1) -> tuple[float, float, str] | None:
    """(rms in m, recorded - UTC in s, frame) of the best fit for one
    integration's cross-correlations, or None if it cannot be fitted."""
    rows = np.arange(start, stop)
    rows = rows[np.asarray(index.ant1)[rows] != np.asarray(index.ant2)[rows]]
    rows = np.array([r for r in rows if int(index.ant1[r]) in positions and int(index.ant2[r]) in positions])
    source = sources.get(int(index.source_id[start]))
    if len(rows) < 3 or source is None:
        return None
    stored = np.stack([np.asarray(index.uu_sec)[rows], np.asarray(index.vv_sec)[rows],
                       np.asarray(index.ww_sec)[rows]], axis=1).astype(np.float64)
    baseline = np.stack([positions[int(index.ant2[r])] - positions[int(index.ant1[r])] for r in rows])
    jd = float(index.jd[start])
    coords = {"J2000": (source.ra_epoch_deg, source.dec_epoch_deg),
              "apparent": (source.ra_apparent_deg, source.dec_apparent_deg)}
    best = None
    for frame in FRAMES:
        ra_deg, dec_deg = coords[frame]
        if not (np.isfinite(ra_deg) and np.isfinite(dec_deg)):
            continue
        ra, dec = np.radians(ra_deg), np.radians(dec_deg)

        def rms_by_offset(offsets_s):
            predicted = uvw_seconds(jd - offsets_s / 86400.0, baseline, ra, dec, frame, ut1)
            # both baseline signs: the stored values or their negation
            rms = [np.sqrt(np.mean((sign * predicted - stored) ** 2, axis=(1, 2))) for sign in (1.0, -1.0)]
            return np.minimum(*rms) * SPEED_OF_LIGHT_M_PER_S

        coarse = np.arange(-SEARCH_S, SEARCH_S + 0.5, 1.0)
        centre = coarse[np.argmin(rms_by_offset(coarse))]
        fine = centre + np.arange(-1.5, 1.505, 0.01)
        offset = _parabola_minimum(fine, rms_by_offset(fine) ** 2)
        rms = float(rms_by_offset(np.array([offset]))[0])
        if best is None or rms < best[0]:
            best = (rms, offset, frame)
    return best


def _parabola_minimum(x: np.ndarray, y: np.ndarray) -> float:
    """The minimum of `y`, refined between grid points by the parabola
    through the lowest point and its neighbours (the squared residual is
    quadratic in the offset where the prediction is linear in it)."""
    k = int(np.clip(np.argmin(y), 1, len(y) - 2))
    y0, y1, y2 = y[k - 1], y[k], y[k + 1]
    curvature = y0 - 2 * y1 + y2
    if curvature <= 0:
        return float(x[k])
    step = x[k + 1] - x[k]
    return float(x[k] + 0.5 * step * (y0 - y2) / curvature)


def uvw_seconds(jd_utc, baseline_m, ra_rad: float, dec_rad: float, frame: str, ut1: Ut1Provider) -> np.ndarray:
    """u, v, w in seconds, shaped (times, baselines, 3), for baselines
    (ITRS vectors, m) towards (ra, dec) at the UTC Julian dates `jd_utc`:
    in the J2000 frame (the baseline rotated into GCRS by IAU 2006/2000A,
    polar motion ignored) or the apparent frame of date (rotated by
    Greenwich apparent sidereal time). TT and UT1 come from erfa's
    leap-second table and `ut1` directly, as two-part Julian dates."""
    jd_utc = np.atleast_1d(np.asarray(jd_utc, dtype=np.float64))
    (tt1, tt2), (ut11, ut12) = _tt_and_ut1(jd_utc, ut1)
    b = np.asarray(baseline_m, dtype=np.float64)
    if frame == "apparent":
        gast = erfa.gst06a(ut11, ut12, tt1, tt2)
        c, s = np.cos(gast)[:, None], np.sin(gast)[:, None]
        bx, by = c * b[None, :, 0] - s * b[None, :, 1], s * b[None, :, 0] + c * b[None, :, 1]
        bz = np.broadcast_to(b[None, :, 2], bx.shape)
    elif frame == "J2000":
        rc2t = erfa.c2t06a(tt1, tt2, ut11, ut12, 0.0, 0.0)  # ITRS = rc2t @ GCRS
        rotated = np.einsum("tji,bj->tbi", rc2t, b)
        bx, by, bz = rotated[..., 0], rotated[..., 1], rotated[..., 2]
    else:
        raise ValueError(f"unknown frame {frame!r}")
    sin_ra, cos_ra, sin_dec, cos_dec = np.sin(ra_rad), np.cos(ra_rad), np.sin(dec_rad), np.cos(dec_rad)
    u = -sin_ra * bx + cos_ra * by
    v = -sin_dec * cos_ra * bx - sin_dec * sin_ra * by + cos_dec * bz
    w = cos_dec * cos_ra * bx + cos_dec * sin_ra * by + sin_dec * bz
    return np.stack([u, v, w], axis=-1) / SPEED_OF_LIGHT_M_PER_S


_MJD_ZERO = 2400000.5
TT_MINUS_TAI_S = 32.184


def _tt_and_ut1(jd_utc: np.ndarray, ut1: Ut1Provider):
    """(TT, UT1) as two-part Julian dates (2400000.5 + days) for UTC
    Julian dates: TT = UTC + (TAI - UTC) + 32.184 s, UT1 = UTC + (UT1 - UTC)."""
    days = jd_utc - _MJD_ZERO
    year, month, day, fraction = erfa.jd2cal(_MJD_ZERO, days)
    tai_minus_utc = erfa.dat(year, month, day, fraction)
    tt = days + (tai_minus_utc + TT_MINUS_TAI_S) / 86400.0
    ut = days + np.asarray(ut1.ut1_minus_utc_s(jd_utc), dtype=np.float64).reshape(days.shape) / 86400.0
    return (_MJD_ZERO, tt), (_MJD_ZERO, ut)
