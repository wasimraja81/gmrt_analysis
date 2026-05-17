#!/usr/bin/env python3
"""
stack_moon_snapshots.py

Registration-aware stacking of per-stack Moon selfcal images.

⚠️  GMRT-SPECIFIC CODE: This script recomputes Moon position using hardcoded GMRT
location (C02 ITRF: 1656342.30, 5797947.77, 2073243.16 m). For other telescopes
(ASKAP, VLA, etc.), modification of gmrt_location() is required.

Each stack was imaged at its own (drifting) phase centre, so the Moon
appears at slightly different pixel positions across stacks.  This script:

  1. Loads all *_final.fits images from a selfcal output directory.
  2. Reads observation time (JDMID or JDAVG) from FITS header.
  3. Recomputes Moon topocentric position at that time for GMRT location.
  4. Optionally derives inter-stack shifts relative to a reference image
      (middle stack by default) using phase cross-correlation.
  5. Optionally applies shifts via Fourier-domain phase ramps
      (no interpolation loss).
  6. Stacks the (optionally registered) images (mean or median).
  7. Writes the result as a FITS image with the reference stack's WCS header.

Usage:
    python experimental/stack_moon_snapshots.py \
        --selfcal-dir ~/DATA/gmrt_40_014/work/casa_selfcal/moon0520_stk10 \
        --output ~/DATA/gmrt_40_014/work/casa_selfcal/moon0520_stk10_stack.fits

Optional:
    --method median        # mean (default) or median
    --upsample 10          # sub-pixel precision; 1 = integer shifts only
    --registration-mode derive  # derive shifts (default) or none for direct co-add
    --registration-method phase-correlation  # shift solver when deriving
    --reference-idx 0      # which stack to use as reference (default: middle)
    --plot-shifts          # print shift table
    --glob '*_final.fits'  # glob pattern for FITS files within selfcal-dir
    --destripe-iters 4     # iterative FFT destriping passes (0 disables)
    --moon-mask-radius-arcmin 20.0  # keep original pixels inside this radius
"""

import argparse
from collections import Counter
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from astropy.io import fits
from astropy.coordinates import EarthLocation, get_body
from astropy.time import Time
from astropy.wcs import WCS


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


# ── helpers ──────────────────────────────────────────────────────────────────

def load_2d(path: str) -> np.ndarray:
    """Load the first HDU, squeeze degenerate Stokes/freq axes → (Ny, Nx)."""
    with fits.open(path) as hdul:
        data = hdul[0].data
    if data is None:
        raise ValueError(f'No data in primary HDU: {path}')
    return np.squeeze(data).astype(np.float32)


def load_header(path: str):
    with fits.open(path) as hdul:
        return hdul[0].header.copy()


def gmrt_location() -> EarthLocation:
    """Return GMRT reference EarthLocation (C02 ITRF reference).
    
    Hardcoded GMRT-specific location used to recompute Moon topocentric position
    from observation timestamps (JDMID/JDAVG) stored in FITS headers.
    
    For other telescopes: replace with the appropriate observatory location.
    
    Returns:
        EarthLocation: GMRT C02 reference point in ITRF coordinates (metres).
    """
    return EarthLocation.from_geocentric(
        1656342.30,
        5797947.77,
        2073243.16,
        unit='m',
    )


def moon_radec_at_jd(jd: float, location: EarthLocation) -> tuple[float, float]:
    """Return apparent topocentric (ra_deg, dec_deg) of Moon at JD(UTC)."""
    t = Time(jd, format='jd', scale='utc')
    moon = get_body('moon', t, location=location)
    return float(moon.ra.deg), float(moon.dec.deg)


def moon_center_from_ephem(header, shape: tuple[int, int]) -> tuple[float, float]:
    """Compute moon center pixel (row, col) using representative FITS timestamp.
    
    ⚠️  GMRT-SPECIFIC: Uses hardcoded GMRT location via gmrt_location().
    Moon position is recomputed from FITS timestamp for GMRT observatory.

    Time priority (most representative first):
      1) JDMID (sample-centred)  2) JDAVG
      3) MJD-AVG  4) DATE-AVG  5) DATE-OBS
    """
    if header.get('JDMID') is not None:
        jd = float(header['JDMID'])
    elif header.get('JDAVG') is not None:
        jd = float(header['JDAVG'])
    elif header.get('MJD-AVG') is not None:
        jd = Time(float(header['MJD-AVG']), format='mjd', scale='utc').jd
    elif header.get('DATE-AVG'):
        jd = Time(header['DATE-AVG'], format='isot', scale='utc').jd
    elif header.get('DATE-OBS'):
        jd = Time(header['DATE-OBS'], format='isot', scale='utc').jd
    else:
        raise ValueError('No usable time keyword found (JDMID/JDAVG/MJD-AVG/DATE-AVG/DATE-OBS).')

    ra_deg, dec_deg = moon_radec_at_jd(jd, gmrt_location())

    wcs = WCS(header).celestial
    x_pix, y_pix = wcs.world_to_pixel_values(ra_deg, dec_deg)

    ny, nx = shape
    row = float(np.clip(y_pix, 0.0, ny - 1.0))
    col = float(np.clip(x_pix, 0.0, nx - 1.0))
    return row, col


def representative_jd_from_header(header) -> float:
    """Return representative JD using the same priority as moon ephemeris usage.

    Priority: JDMID > JDAVG > MJD-AVG > DATE-AVG > DATE-OBS
    """
    if header.get('JDMID') is not None:
        return float(header['JDMID'])
    if header.get('JDAVG') is not None:
        return float(header['JDAVG'])
    if header.get('MJD-AVG') is not None:
        return Time(float(header['MJD-AVG']), format='mjd', scale='utc').jd
    if header.get('DATE-AVG'):
        return Time(header['DATE-AVG'], format='isot', scale='utc').jd
    if header.get('DATE-OBS'):
        return Time(header['DATE-OBS'], format='isot', scale='utc').jd
    raise ValueError('No usable time keyword found (JDMID/JDAVG/MJD-AVG/DATE-AVG/DATE-OBS).')


def phase_cross_correlate(ref: np.ndarray, mov: np.ndarray,
                          upsample: int = 10) -> tuple[float, float]:
    """
    Return (row_shift, col_shift) to align *mov* onto *ref*.
    Uses skimage if available, falls back to numpy FFT integer-only.
    """
    try:
        from skimage.registration import phase_cross_correlation
        shift, _, _ = phase_cross_correlation(ref, mov,
                                              upsample_factor=upsample,
                                              normalization=None)
        return float(shift[0]), float(shift[1])
    except ImportError:
        # Integer-only fallback via np.fft
        F = np.fft.fft2(ref) * np.conj(np.fft.fft2(mov))
        cc = np.fft.ifft2(F / (np.abs(F) + 1e-30)).real
        idx = np.unravel_index(np.argmax(cc), cc.shape)
        dr = idx[0] if idx[0] < ref.shape[0] // 2 else idx[0] - ref.shape[0]
        dc = idx[1] if idx[1] < ref.shape[1] // 2 else idx[1] - ref.shape[1]
        return float(dr), float(dc)


def apply_shift_fourier(img: np.ndarray, dr: float, dc: float) -> np.ndarray:
    """
    Apply sub-pixel shift (dr, dc) via Fourier phase ramp.
    Zero-shift returns the original unchanged.
    """
    if dr == 0.0 and dc == 0.0:
        return img
    ny, nx = img.shape
    # Build frequency grids
    fr = np.fft.fftfreq(ny)[:, np.newaxis]  # (ny, 1)
    fc = np.fft.fftfreq(nx)[np.newaxis, :]  # (1, nx)
    phase_ramp = np.exp(-2j * np.pi * (dr * fr + dc * fc))
    shifted = np.fft.ifft2(np.fft.fft2(img) * phase_ramp).real
    return shifted.astype(np.float32)


def robust_mad_sigma(values: np.ndarray) -> float:
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    return float(1.4826 * mad + 1e-12)


def build_circular_mask(shape: tuple[int, int], center_rc: tuple[float, float],
                        radius_pix: float) -> np.ndarray:
    ny, nx = shape
    rr, cc = np.indices((ny, nx), dtype=np.float32)
    r0, c0 = center_rc
    return (rr - r0) ** 2 + (cc - c0) ** 2 <= radius_pix ** 2


def make_notch_multiplier(shape: tuple[int, int], peaks: list[tuple[int, int]],
                          notch_sigma_pix: float, notch_depth: float) -> np.ndarray:
    ny, nx = shape
    rr, cc = np.indices((ny, nx), dtype=np.float32)
    mult = np.ones((ny, nx), dtype=np.float32)
    for pr, pc in peaks:
        d2 = (rr - pr) ** 2 + (cc - pc) ** 2
        mult *= (1.0 - notch_depth * np.exp(-0.5 * d2 / (notch_sigma_pix ** 2)))
    return np.clip(mult, 0.0, 1.0)


def select_band_centers(power_shift: np.ndarray,
                        axis: str,
                        core_radius_pix: float,
                        sigma_thresh: float,
                        max_bands: int,
                        min_separation_pix: int,
                        min_run_pix: int) -> list[int]:
    ny, nx = power_shift.shape
    cy, cx = ny // 2, nx // 2
    rr, cc = np.indices((ny, nx))
    r2 = (rr - cy) ** 2 + (cc - cx) ** 2
    outside_core = r2 >= core_radius_pix ** 2

    if axis == 'row':
        n = ny
        center = cy
        profile = np.full(ny, np.nan, dtype=np.float64)
        for r in range(ny):
            vals = power_shift[r, outside_core[r, :]]
            if vals.size:
                profile[r] = np.median(vals)
    elif axis == 'col':
        n = nx
        center = cx
        profile = np.full(nx, np.nan, dtype=np.float64)
        for c in range(nx):
            vals = power_shift[outside_core[:, c], c]
            if vals.size:
                profile[c] = np.median(vals)
    else:
        raise ValueError(f"axis must be 'row' or 'col', got {axis!r}")

    valid = np.isfinite(profile)
    if not np.any(valid):
        return []

    idx = np.arange(n)
    dist_to_center = np.minimum(np.abs(idx - center), n - np.abs(idx - center))
    valid &= dist_to_center >= core_radius_pix

    vals = profile[valid]
    if vals.size == 0:
        return []
    med = np.median(vals)
    sigma = robust_mad_sigma(vals)
    thresh = med + sigma_thresh * sigma

    candidates = np.where((profile > thresh) & valid)[0]
    if candidates.size == 0:
        return []

    runs: list[np.ndarray] = []
    start = 0
    for i in range(1, candidates.size + 1):
        if i == candidates.size or candidates[i] != candidates[i - 1] + 1:
            run = candidates[start:i]
            if run.size >= max(1, min_run_pix):
                runs.append(run)
            start = i

    if not runs:
        return []

    peaks = [int(run[np.argmax(profile[run])]) for run in runs]
    peaks = sorted(peaks, key=lambda p: profile[p], reverse=True)

    selected: list[int] = []
    for p in peaks:
        if all(min(abs(p - q), n - abs(p - q)) >= min_separation_pix for q in selected):
            selected.append(p)
        if len(selected) >= max_bands:
            break

    out: list[int] = []
    seen = set()
    for p in selected:
        for q in (p, (2 * center - p) % n):
            if q not in seen:
                seen.add(q)
                out.append(int(q))
    return out


def make_band_multiplier(shape: tuple[int, int],
                         row_bands: list[int],
                         col_bands: list[int],
                         band_sigma_pix: float,
                         band_depth: float) -> np.ndarray:
    ny, nx = shape
    rr, cc = np.indices((ny, nx), dtype=np.float32)
    mult = np.ones((ny, nx), dtype=np.float32)

    for r in row_bands:
        d2 = (rr - float(r)) ** 2
        mult *= (1.0 - band_depth * np.exp(-0.5 * d2 / (band_sigma_pix ** 2)))

    for c in col_bands:
        d2 = (cc - float(c)) ** 2
        mult *= (1.0 - band_depth * np.exp(-0.5 * d2 / (band_sigma_pix ** 2)))

    return np.clip(mult, 0.0, 1.0)


def make_radial_bandpass_multiplier(shape: tuple[int, int],
                                    inner_radius_pix: float,
                                    outer_radius_pix: float,
                                    inner_taper_pix: float,
                                    outer_taper_pix: float) -> np.ndarray:
    ny, nx = shape
    rr, cc = np.indices((ny, nx), dtype=np.float32)
    cy, cx = ny / 2.0, nx / 2.0
    radius = np.sqrt((rr - cy) ** 2 + (cc - cx) ** 2)

    inner_taper = max(0.25, float(inner_taper_pix))
    outer_taper = max(0.25, float(outer_taper_pix))

    high_pass = 1.0 / (1.0 + np.exp(-(radius - float(inner_radius_pix)) / inner_taper))
    low_pass = 1.0 / (1.0 + np.exp((radius - float(outer_radius_pix)) / outer_taper))
    mult = high_pass * low_pass
    return np.clip(mult.astype(np.float32), 0.0, 1.0)


def make_radial_shell_multiplier(shape: tuple[int, int],
                                 ring_radius_pix: float,
                                 ring_sigma_pix: float) -> np.ndarray:
    ny, nx = shape
    rr, cc = np.indices((ny, nx), dtype=np.float32)
    cy, cx = ny / 2.0, nx / 2.0
    radius = np.sqrt((rr - cy) ** 2 + (cc - cx) ** 2)
    sigma = max(0.25, float(ring_sigma_pix))
    r0 = max(0.0, float(ring_radius_pix))
    mult = np.exp(-0.5 * ((radius - r0) / sigma) ** 2)
    maxv = float(np.max(mult))
    if maxv > 0:
        mult = mult / maxv
    return np.clip(mult.astype(np.float32), 0.0, 1.0)


def fft_scale_values(power_before: np.ndarray,
                     power_after: np.ndarray,
                     fft_axis: str = 'pix',
                     fft_u_klambda: Optional[np.ndarray] = None,
                     fft_v_klambda: Optional[np.ndarray] = None,
                     fft_max_klambda: Optional[float] = None) -> np.ndarray:
    if (
        fft_axis == 'klambda'
        and fft_max_klambda is not None
        and fft_max_klambda > 0
        and fft_u_klambda is not None
        and fft_v_klambda is not None
        and fft_u_klambda.size == power_before.shape[1]
        and fft_v_klambda.size == power_before.shape[0]
    ):
        mask = (
            (np.abs(np.asarray(fft_v_klambda, dtype=np.float64))[:, None] <= fft_max_klambda)
            & (np.abs(np.asarray(fft_u_klambda, dtype=np.float64))[None, :] <= fft_max_klambda)
        )
        vals_before = power_before[mask]
        vals_after = power_after[mask]
        vals = np.concatenate([vals_before.ravel(), vals_after.ravel()])
        if vals.size > 0:
            return vals
    return np.concatenate([power_before.ravel(), power_after.ravel()])


def select_fft_peaks(power_shift: np.ndarray,
                     core_radius_pix: float,
                     sigma_thresh: float,
                     max_peaks: int,
                     min_separation_pix: int) -> list[tuple[int, int]]:
    ny, nx = power_shift.shape
    cy, cx = ny // 2, nx // 2
    rr, cc = np.indices((ny, nx))
    r2 = (rr - cy) ** 2 + (cc - cx) ** 2
    outside_core = r2 >= core_radius_pix ** 2

    vals = power_shift[outside_core]
    med = np.median(vals)
    sigma = robust_mad_sigma(vals)
    thresh = med + sigma_thresh * sigma

    cand = np.argwhere((power_shift > thresh) & outside_core)
    if cand.size == 0:
        return []

    scores = power_shift[cand[:, 0], cand[:, 1]]
    order = np.argsort(scores)[::-1]
    selected: list[tuple[int, int]] = []
    taken = np.zeros((ny, nx), dtype=bool)

    for idx in order:
        r, c = int(cand[idx, 0]), int(cand[idx, 1])
        if taken[r, c]:
            continue
        selected.append((r, c))
        if len(selected) >= max_peaks:
            break

        r0, r1 = max(0, r - min_separation_pix), min(ny, r + min_separation_pix + 1)
        c0, c1 = max(0, c - min_separation_pix), min(nx, c + min_separation_pix + 1)
        taken[r0:r1, c0:c1] = True

        # Also suppress the conjugate-symmetric partner neighborhood
        rc = (2 * cy - r) % ny
        ccj = (2 * cx - c) % nx
        r0, r1 = max(0, rc - min_separation_pix), min(ny, rc + min_separation_pix + 1)
        c0, c1 = max(0, ccj - min_separation_pix), min(nx, ccj + min_separation_pix + 1)
        taken[r0:r1, c0:c1] = True

    # include conjugate peaks explicitly so notches are symmetric
    out: list[tuple[int, int]] = []
    seen = set()
    for r, c in selected:
        partners = [(r, c), ((2 * cy - r) % ny, (2 * cx - c) % nx)]
        for pr, pc in partners:
            key = (int(pr), int(pc))
            if key not in seen:
                seen.add(key)
                out.append(key)
    return out


def iterative_fft_destripe(img: np.ndarray,
                           moon_mask: np.ndarray,
                           n_iter: int,
                           strategy: str,
                           core_radius_pix: float,
                           sigma_thresh: float,
                           max_peaks: int,
                           min_separation_pix: int,
                           notch_sigma_pix: float,
                           notch_depth: float,
                           radial_inner_radius_pix: float = 20.0,
                           radial_outer_radius_pix: float = -1.0,
                           radial_inner_taper_pix: float = 4.0,
                           radial_outer_taper_pix: float = 6.0,
                           shell_radius_pix: float = 0.0,
                           shell_sigma_pix: float = 10.0,
                           band_mode: bool = False,
                           band_sigma_thresh: float = 4.0,
                           band_max_runs: int = 4,
                           band_min_run_pix: int = 3,
                           band_sigma_pix: float = 3.0,
                           band_depth: float = 0.35,
                           debug_callback: Optional[Callable[[int, np.ndarray, list[tuple[int, int]], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], None]] = None) -> tuple[np.ndarray, list[int], list[float]]:
    original = img.astype(np.float32)
    work = original.copy()
    peaks_per_iter: list[int] = []
    rms_outside_moon_per_iter: list[float] = []

    raw_rms_outside = float(np.sqrt(np.mean(original[~moon_mask] ** 2)))
    rms_outside_moon_per_iter.append(raw_rms_outside)

    for iter_idx in range(max(0, n_iter)):
        fft_shift = np.fft.fftshift(np.fft.fft2(work))
        power = np.log10(np.abs(fft_shift) + 1e-12)

        peaks: list[tuple[int, int]] = []
        if strategy == 'legacy-notch':
            peaks = select_fft_peaks(
                power_shift=power,
                core_radius_pix=core_radius_pix,
                sigma_thresh=sigma_thresh,
                max_peaks=max_peaks,
                min_separation_pix=min_separation_pix,
            )
            peaks_per_iter.append(len(peaks))
            if not peaks:
                break

            mult = make_notch_multiplier(
                shape=work.shape,
                peaks=peaks,
                notch_sigma_pix=notch_sigma_pix,
                notch_depth=notch_depth,
            )

            if band_mode:
                row_bands = select_band_centers(
                    power_shift=power,
                    axis='row',
                    core_radius_pix=core_radius_pix,
                    sigma_thresh=band_sigma_thresh,
                    max_bands=band_max_runs,
                    min_separation_pix=min_separation_pix,
                    min_run_pix=band_min_run_pix,
                )
                col_bands = select_band_centers(
                    power_shift=power,
                    axis='col',
                    core_radius_pix=core_radius_pix,
                    sigma_thresh=band_sigma_thresh,
                    max_bands=band_max_runs,
                    min_separation_pix=min_separation_pix,
                    min_run_pix=band_min_run_pix,
                )
                if row_bands or col_bands:
                    band_mult = make_band_multiplier(
                        shape=work.shape,
                        row_bands=row_bands,
                        col_bands=col_bands,
                        band_sigma_pix=band_sigma_pix,
                        band_depth=band_depth,
                    )
                    mult = np.clip(mult * band_mult, 0.0, 1.0)
        elif strategy == 'radial-bandpass':
            peaks_per_iter.append(0)
            ny, nx = work.shape
            outer_default = 0.48 * min(ny, nx)
            outer_radius = radial_outer_radius_pix if radial_outer_radius_pix > 0 else outer_default
            mult = make_radial_bandpass_multiplier(
                shape=work.shape,
                inner_radius_pix=max(0.0, radial_inner_radius_pix),
                outer_radius_pix=max(1.0, outer_radius),
                inner_taper_pix=radial_inner_taper_pix,
                outer_taper_pix=radial_outer_taper_pix,
            )
        elif strategy == 'radial-shell':
            peaks_per_iter.append(0)
            mult = make_radial_shell_multiplier(
                shape=work.shape,
                ring_radius_pix=shell_radius_pix,
                ring_sigma_pix=shell_sigma_pix,
            )
        else:
            raise ValueError(f'Unknown destripe strategy: {strategy!r}')

        power_after = np.log10(np.abs(fft_shift * mult) + 1e-12)

        filtered = np.fft.ifft2(np.fft.ifftshift(fft_shift * mult)).real.astype(np.float32)

        # Outside moon: always use filtered (stripe-cleaned) values.
        # Inside moon: keep original where it is non-negative (real emission);
        #              replace negative pixels (stripe artifacts) with filtered.
        hybrid = filtered.copy()
        moon_positive = moon_mask & (original >= 0)
        hybrid[moon_positive] = original[moon_positive]
        
        # Track RMS outside moon mask
        rms_outside = float(np.sqrt(np.mean(hybrid[~moon_mask] ** 2)))
        rms_outside_moon_per_iter.append(rms_outside)
        
        if debug_callback is not None:
            debug_callback(iter_idx, power, peaks, mult, power_after, work, filtered, hybrid)
        work = hybrid

    return work, peaks_per_iter, rms_outside_moon_per_iter


def save_combined_debug_plot(power_before: np.ndarray,
                             peaks: list[tuple[int, int]],
                             multiplier: np.ndarray,
                             power_after: np.ndarray,
                             core_radius_pix: float,
                             image_before: np.ndarray,
                             image_filtered: np.ndarray,
                             image_hybrid: np.ndarray,
                             moon_mask: np.ndarray,
                             iter_idx: int,
                             out_png: Path,
                             fft_vmin: Optional[float] = None,
                             fft_vmax: Optional[float] = None,
                             img_vmin: Optional[float] = None,
                             img_vmax: Optional[float] = None,
                             fft_axis: str = 'pix',
                             fft_u_klambda: Optional[np.ndarray] = None,
                             fft_v_klambda: Optional[np.ndarray] = None,
                             fft_max_klambda: Optional[float] = None,
                             fft_scale_mode: str = 'global',
                             fft_scale_low: float = 1.0,
                             fft_scale_high: float = 99.0) -> None:
    """2-row × 3-col debug panel with optional fixed color scales.
    Row 0: FFT log-power before | notch multiplier | FFT log-power after
    Row 1: Image before iter    | filtered (IFFT)  | hybrid for next iter
    
    Args:
        fft_vmin, fft_vmax: Fixed color scale for FFT panels (if None, auto-scaled each iter)
        img_vmin, img_vmax: Fixed color scale for image panels (if None, auto-scaled each iter)
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
    except Exception:
        return

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

    # ── Row 0: Fourier space ────────────────────────────────────────────────
    fft_row = axes[0]
    ny, nx = power_before.shape
    cy, cx = ny / 2.0, nx / 2.0
    u_axis = np.array([], dtype=np.float64)
    v_axis = np.array([], dtype=np.float64)
    show_klambda = (
        fft_axis == 'klambda'
        and fft_u_klambda is not None
        and fft_v_klambda is not None
        and fft_u_klambda.size == nx
        and fft_v_klambda.size == ny
    )

    if show_klambda:
        u_axis = np.asarray(fft_u_klambda, dtype=np.float64)
        v_axis = np.asarray(fft_v_klambda, dtype=np.float64)
        x_extent = [float(u_axis[0]), float(u_axis[-1])]
        y_extent = [float(v_axis[0]), float(v_axis[-1])]
        du_kl = float(np.median(np.abs(np.diff(u_axis)))) if nx > 1 else 0.0
        dv_kl = float(np.median(np.abs(np.diff(v_axis)))) if ny > 1 else 0.0
        r_core_kl = core_radius_pix * (du_kl + dv_kl) * 0.5
        fft_extent = (x_extent[0], x_extent[1], y_extent[0], y_extent[1])
    else:
        du_kl = 0.0
        dv_kl = 0.0
        r_core_kl = 0.0
        fft_extent = None
    
    # Auto-scale FFT if not provided
    if fft_vmin is None or fft_vmax is None:
        fft_combined = fft_scale_values(
            power_before=power_before,
            power_after=power_after,
            fft_axis=fft_axis if fft_scale_mode != 'global' else 'pix',
            fft_u_klambda=fft_u_klambda,
            fft_v_klambda=fft_v_klambda,
            fft_max_klambda=fft_max_klambda if fft_scale_mode != 'global' else None,
        )
        fft_vmin = float(np.percentile(fft_combined, fft_scale_low))
        fft_vmax = float(np.percentile(fft_combined, fft_scale_high))

    fft_panels = [
        (fft_row[0], power_before, 'FFT log-power (before)'),
        (fft_row[1], multiplier,   'Notch multiplier'),
        (fft_row[2], power_after,  'FFT log-power (after)'),
    ]
    for ax, arr, title in fft_panels:
        # Use fixed scale for before/after, auto for multiplier
        if title == 'Notch multiplier':
            im = ax.imshow(
                arr,
                origin='lower',
                cmap='viridis',
                extent=fft_extent,
            )
        else:
            im = ax.imshow(
                arr,
                origin='lower',
                cmap='magma',
                vmin=fft_vmin,
                vmax=fft_vmax,
                extent=fft_extent,
            )
        ax.set_title(title, fontsize=9)
        if show_klambda:
            ax.set_xlabel('u [kλ]', fontsize=8)
            ax.set_ylabel('v [kλ]', fontsize=8)
            if fft_max_klambda is not None and fft_max_klambda > 0:
                ax.set_xlim(-fft_max_klambda, fft_max_klambda)
                ax.set_ylim(-fft_max_klambda, fft_max_klambda)
            ax.axvline(0.0, color='white', lw=0.6, alpha=0.45)
            ax.axhline(0.0, color='white', lw=0.6, alpha=0.45)
            if du_kl > 0 and dv_kl > 0:
                for sign in (-1.0, 1.0):
                    ax.axvline(sign * du_kl, color='cyan', lw=0.6, alpha=0.45, linestyle=':')
                    ax.axhline(sign * dv_kl, color='cyan', lw=0.6, alpha=0.45, linestyle=':')
        else:
            ax.set_xlabel('k_x pixel', fontsize=8)
            ax.set_ylabel('k_y pixel', fontsize=8)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)
        core_center = (0.0, 0.0) if show_klambda else (cx, cy)
        core_radius = r_core_kl if show_klambda else core_radius_pix
        core = Circle(core_center, core_radius, fill=False,
                      edgecolor='cyan', linewidth=1.2, linestyle='--')
        ax.add_patch(core)

    for idx, (r, c) in enumerate(peaks, start=1):
        if show_klambda:
            u = float(u_axis[c])
            v = float(v_axis[r])
            for fax in (fft_row[0], fft_row[2]):
                fax.plot(u, v, marker='o', markersize=8, markerfacecolor='none',
                         markeredgecolor='lime', markeredgewidth=1.2)
            du_txt = 2.0 * du_kl if du_kl > 0 else 0.5
            dv_txt = 2.0 * dv_kl if dv_kl > 0 else 0.5
            fft_row[0].text(u + du_txt, v + dv_txt, str(idx), color='white', fontsize=7,
                            bbox=dict(facecolor='black', alpha=0.45, pad=1.0, edgecolor='none'))
        else:
            for fax in (fft_row[0], fft_row[2]):
                fax.add_patch(Circle((c, r), 6.0, fill=False, edgecolor='lime', linewidth=1.2))
            fft_row[0].text(c + 5, r + 5, str(idx), color='white', fontsize=7,
                            bbox=dict(facecolor='black', alpha=0.45, pad=1.0, edgecolor='none'))

    # ── Row 1: image space ──────────────────────────────────────────────────
    img_row = axes[1]
    
    # Auto-scale image if not provided
    if img_vmin is None or img_vmax is None:
        img_combined = np.concatenate([image_before[np.isfinite(image_before)].ravel(),
                                       image_filtered[np.isfinite(image_filtered)].ravel(),
                                       image_hybrid[np.isfinite(image_hybrid)].ravel()])
        if img_combined.size > 0:
            img_vmin = float(np.percentile(img_combined, 2.0))
            img_vmax = float(np.percentile(img_combined, 99.8))
        else:
            img_vmin, img_vmax = -1.0, 1.0

    img_panels = [
        (img_row[0], image_before,    'Image before iteration'),
        (img_row[1], image_filtered,  'Filtered (IFFT after notch)'),
        (img_row[2], image_hybrid,    'Hybrid for next iter'),
    ]
    for ax, arr, title in img_panels:
        im = ax.imshow(arr, origin='lower', cmap='magma', vmin=img_vmin, vmax=img_vmax)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('x pixel', fontsize=8)
        ax.set_ylabel('y pixel', fontsize=8)
        try:
            ax.contour(moon_mask.astype(np.uint8), levels=[0.5],
                       colors='cyan', linewidths=0.7)
        except Exception:
            pass
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)

    if show_klambda:
        fig.suptitle(
            f'Iteration {iter_idx}  |  peaks cut: {len(peaks)}  |  '
            f'FFT axis=kλ (Δu≈{du_kl:.3f}, Δv≈{dv_kl:.3f})  |  '
            f'cyan circle=FFT core exclusion',
            fontsize=10,
        )
    else:
        fig.suptitle(
            f'Iteration {iter_idx}  |  peaks cut: {len(peaks)}  |  '
            f'cyan = moon mask  |  lime circles = notched peaks',
            fontsize=10,
        )
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def save_destripe_convergence_plot(rms_outside_moon_list: list[float],
                                   stem: str,
                                   debug_root: Path) -> None:
    """Save RMS convergence plot for destriping iterations.
    
    Shows both absolute RMS and normalized RMS (RMS/RMS₀).
    Uses robust MAD-based RMS for resistance to outliers.
    
    Args:
        rms_outside_moon_list: RMS values per iteration (outside moon mask)
        stem: Filename stem for output PNG
        debug_root: Directory to save plot
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    
    if not rms_outside_moon_list or len(rms_outside_moon_list) < 2:
        return
    
    iterations = np.arange(len(rms_outside_moon_list)) - 1
    rms_arr = np.array(rms_outside_moon_list)
    rms_norm = rms_arr / rms_arr[0]  # Normalized to raw

    # For readability, exclude the dramatic raw->iter0 jump from line plots
    # and report it in text annotations instead.
    if len(rms_arr) >= 3:
        plot_mask = iterations >= 1  # plot from iter1 onward
    else:
        plot_mask = np.ones_like(iterations, dtype=bool)
    iter_plot = iterations[plot_mask]
    rms_plot = rms_arr[plot_mask]
    rms_norm_plot = rms_norm[plot_mask]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'FFT Destripe Convergence: {stem}', fontsize=13, fontweight='bold')
    
    # Panel 1: Absolute RMS
    ax = axes[0]
    ax.plot(iter_plot, rms_plot, 'o-', linewidth=3, markersize=10,
            color='steelblue', alpha=0.8, markerfacecolor='lightblue',
            markeredgewidth=2, markeredgecolor='steelblue')
    ax.axhline(rms_arr[-1], color='crimson', linestyle='--', linewidth=2, 
               label=f'Asymptote = {rms_arr[-1]:.3e}', alpha=0.7)
    ax.set_xlabel('Iteration (raw = -1)', fontsize=12, fontweight='bold')
    ax.set_ylabel('RMS (outside moon mask)', fontsize=12, fontweight='bold')
    ax.set_title('Absolute RMS', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10, loc='upper right')
    
    # Panel 2: Normalized RMS
    ax2 = axes[1]
    ax2.plot(iter_plot, rms_norm_plot, 's-', linewidth=3, markersize=10,
             color='darkorange', alpha=0.8, markerfacecolor='navajowhite',
             markeredgewidth=2, markeredgecolor='darkorange')
    ax2.axhline(1.0, color='grey', linestyle=':', linewidth=2, label='Raw (1.0)')
    ax2.axhline(rms_norm[-1], color='crimson', linestyle='--', linewidth=2,
                label=f'Final = {rms_norm[-1]:.3f}', alpha=0.7)
    ax2.set_xlabel('Iteration (raw = -1)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('RMS / RMS₀  (normalized)', fontsize=12, fontweight='bold')
    ax2.set_title('Normalized RMS', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=10, loc='upper right')
    
    # Calculate improvement
    improvement_pct = (rms_arr[0] - rms_arr[-1]) / rms_arr[0] * 100
    asymptotic_reduction = (1 - rms_norm[-1]) * 100
    first_iter_drop_pct = 0.0
    first_iter_rms = None
    if len(rms_arr) > 1:
        first_iter_rms = rms_arr[1]
        first_iter_drop_pct = (rms_arr[0] - rms_arr[1]) / rms_arr[0] * 100
    
    fig.text(0.5, 0.02, 
             f'Raw→iter0: {rms_arr[0]:.2e}→{first_iter_rms:.2e} ({first_iter_drop_pct:.1f}% drop)  |  '
             f'Raw→final: {improvement_pct:.1f}% reduction  |  '
             f'Final level: {asymptotic_reduction:.1f}% below raw',
             ha='center', fontsize=10, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    out_png = debug_root / f'{stem}_converge_rms.png'
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[destripe] Saved convergence plot: {out_png}')


def save_destripe_debug_animation(debug_img_dir: Path,
                                  stem: str,
                                  fps: float = 2.0) -> None:
    """Build animated debug products from per-iteration PNGs.

    Creates an animated GIF whenever Pillow is available and attempts an MP4
    when `ffmpeg` is installed on the system. This is a best-effort helper for
    debug mode and should never fail the main destriping workflow.
    """
    frame_paths = sorted(debug_img_dir.glob('iter_*_debug.png'))
    if len(frame_paths) < 2:
        return

    safe_fps = max(float(fps), 0.1)
    gif_path = debug_img_dir / f'{stem}_iter_progression.gif'
    mp4_path = debug_img_dir / f'{stem}_iter_progression.mp4'

    try:
        from PIL import Image

        frames = []
        for frame_path in frame_paths:
            with Image.open(frame_path) as img:
                frames.append(img.convert('RGB').copy())

        frame_duration_ms = max(80, int(round(1000.0 / safe_fps)))
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration_ms,
            loop=0,
            optimize=False,
        )
        print(f'[destripe] Saved debug GIF: {gif_path}')
    except Exception as exc:
        print(f'[destripe] WARNING: could not build debug GIF for {stem}: {exc}')

    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        print(f'[destripe] NOTE: ffmpeg not found; skipping MP4 movie for {stem}.')
        return

    pattern = str(debug_img_dir / 'iter_%02d_debug.png')
    ffmpeg_cmd = [
        ffmpeg,
        '-y',
        '-loglevel', 'error',
        '-framerate', f'{safe_fps:.3f}',
        '-i', pattern,
        '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
        '-pix_fmt', 'yuv420p',
        str(mp4_path),
    ]
    try:
        proc = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=False)
        if proc.returncode == 0:
            print(f'[destripe] Saved debug MP4: {mp4_path}')
        else:
            stderr = (proc.stderr or '').strip()
            print(f'[destripe] WARNING: ffmpeg MP4 build failed for {stem}: {stderr}')
    except Exception as exc:
        print(f'[destripe] WARNING: could not build debug MP4 for {stem}: {exc}')


def stack_registered_images(images: list[np.ndarray],
                            ref_img: np.ndarray,
                            valid_paths: list[str],
                            ref_idx: int,
                            upsample: int,
                            method: str,
                            registration_mode: str,
                            registration_method: str,
                            px_scale_arcsec: float,
                            print_shift_table: bool = True) -> np.ndarray:
    """Stack images with optional registration to a reference frame."""
    shifted = []
    for i, img in enumerate(images):
        fname = os.path.basename(valid_paths[i])
        if registration_mode == 'none':
            dr, dc = 0.0, 0.0
        else:
            if i == ref_idx:
                dr, dc = 0.0, 0.0
            elif registration_method == 'phase-correlation':
                dr, dc = phase_cross_correlate(ref_img, img, upsample=upsample)
            else:
                raise ValueError(f'Unsupported registration method: {registration_method}')

        if print_shift_table:
            print(f'{i:>4}  {fname:<45}  {dr:>8.3f}  {dc:>8.3f}  '
                  f'{dr * px_scale_arcsec:>10.3f}  {dc * px_scale_arcsec:>10.3f}  '
                  f'{np.hypot(dr, dc) * px_scale_arcsec / 60.0:>8.3f}')

        shifted.append(apply_shift_fourier(img, dr, dc))

    stack = np.array(shifted)
    if method == 'mean':
        return np.nanmean(stack, axis=0)
    return np.nanmedian(stack, axis=0)


def save_stack_preview_png(image: np.ndarray,
                           out_png: Path,
                           title: str) -> None:
    """Save a quick-look PNG for a final stacked image."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    finite = image[np.isfinite(image)]
    if finite.size > 0:
        vmin = float(np.percentile(finite, 2.0))
        vmax = float(np.percentile(finite, 99.8))
    else:
        vmin, vmax = -1.0, 1.0

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 6.5))
    im = ax.imshow(image, origin='lower', cmap='magma', vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('x pixel')
    ax.set_ylabel('y pixel')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Flux', fontsize=10)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[stack] Saved preview PNG: {out_png}')


def main():
    p = argparse.ArgumentParser(
        description='Cross-correlation stack of per-stack Moon selfcal images.')
    p.add_argument('--selfcal-dir', required=True,
                   help='Directory containing per-stack selfcal output dirs.')
    p.add_argument('--output', required=True,
                   help='Output stacked FITS path.')
    p.add_argument('--glob', default='*_final.fits',
                   dest='glob_pat',
                   help='Glob pattern for FITS files within selfcal-dir '
                        '(default: *_final.fits).')
    p.add_argument('--method', choices=['mean', 'median'], default='mean',
                   help='Stacking method (default: mean).')
    p.add_argument('--upsample', type=int, default=10,
                   help='Sub-pixel upsampling factor for cross-correlation '
                        '(default: 10 → 0.1 pixel precision).')
    p.add_argument('--registration-mode', '--phase-shift',
                   choices=['derive', 'none'],
                   dest='registration_mode',
                   default='derive',
                   help='Image registration mode before stacking: '
                        'derive = estimate+apply shifts; '
                        'none = direct pixel-by-pixel co-add with no shifts.')
    p.add_argument('--registration-method', '--phase-shift-derive-method',
                   choices=['phase-correlation'],
                   dest='registration_method',
                   default='phase-correlation',
                   help='Shift estimation method used when --registration-mode=derive.')
    p.add_argument('--reference-idx', type=int, default=None,
                   help='Index of the reference stack (default: middle).')
    p.add_argument('--plot-shifts', action='store_true',
                   help='Print per-stack shift table to stdout.')
    p.add_argument('--destripe-iters', type=int, default=0,
                   help='Iterative FFT destriping passes before cross-correlation '
                        '(default: 0 = disabled).')
    p.add_argument('--destripe-strategy', choices=['legacy-notch', 'radial-bandpass', 'radial-shell'],
                   default='legacy-notch',
                   help='Destriping strategy: legacy peak/band notch filtering or '
                        'smooth radial filtering (default: legacy-notch).')
    p.add_argument('--moon-mask-radius-arcmin', type=float, default=20.0,
                   help='Moon protection radius used during destriping. '
                        'Pixels inside this radius are restored from original image '
                        '(default: 20 arcmin).')
    p.add_argument('--moon-mask-center', choices=['center', 'peak', 'ephem'], default='ephem',
                   help='How to set moon-mask center for destriping: '
                        '"peak" finds brightest pixel after mild smoothing; '
                        '"center" uses image center; '
                        '"ephem" computes Moon RA/Dec at FITS DATE-OBS and projects with WCS '
                        '(default: ephem).')
    p.add_argument('--destripe-core-radius-pix', type=float, default=20.0,
                   help='Ignore low-frequency FFT core within this radius when '
                        'detecting stripe peaks (default: 20).')
    p.add_argument('--destripe-sigma-thresh', type=float, default=5.0,
                   help='FFT peak threshold in robust sigma above median '
                        '(default: 5).')
    p.add_argument('--destripe-max-peaks', type=int, default=24,
                   help='Max independent FFT peaks per iteration '
                        '(default: 24; conjugates added automatically).')
    p.add_argument('--destripe-min-separation-pix', type=int, default=10,
                   help='Minimum separation in FFT pixels between selected peaks '
                        '(default: 10).')
    p.add_argument('--destripe-notch-sigma-pix', type=float, default=2.5,
                   help='Gaussian notch width in FFT pixels (default: 2.5).')
    p.add_argument('--destripe-notch-depth', type=float, default=1.0,
                   help='Notch depth [0,1]; 1 fully suppresses peak center '
                        '(default: 1).')
    p.add_argument('--destripe-debug-dir', default='',
                   help='If set, save Fourier-domain debug PNGs (with circled '
                        'and numbered cut peaks) for each image/iteration, plus '
                        'an animated GIF and MP4 progression when possible.')
    p.add_argument('--destripe-debug-animation-fps', type=float, default=2.0,
                   help='Frame rate for destripe debug GIF/MP4 animations '
                        '(default: 2.0 fps).')
    p.add_argument('--destripe-debug-fft-axis', choices=['pix', 'klambda'], default='pix',
                   help='Debug FFT axis units: pixel index (pix) or kλ (klambda). '
                        'Only affects debug plot axes; destriping math is unchanged.')
    p.add_argument('--destripe-debug-fft-max-klambda', type=float, default=None,
                   help='If set and --destripe-debug-fft-axis=klambda, zoom FFT debug '
                        'panels to ±this value in kλ (display only).')
    p.add_argument('--destripe-debug-fft-scale-mode', choices=['global', 'windowed', 'per-iter'],
                   default='global',
                   help='How to scale FFT log-power debug panels: '
                        'global = fixed from full FFT at iter 0; '
                        'windowed = fixed from displayed FFT window at iter 0; '
                        'per-iter = recompute each iteration using displayed window if set.')
    p.add_argument('--destripe-debug-fft-scale-low', type=float, default=1.0,
                   help='Lower percentile for FFT debug color scaling (default: 1.0).')
    p.add_argument('--destripe-debug-fft-scale-high', type=float, default=99.0,
                   help='Upper percentile for FFT debug color scaling (default: 99.0).')
    p.add_argument('--destripe-save-npz', action='store_true',
                   help='Save per-iteration Fourier arrays (.npz) in debug dir.')
    p.add_argument('--destripe-write-dir', default='',
                   help='If set, write per-frame destriped FITS files (final '
                        'iteration output) to this directory.')
    p.add_argument('--destripe-band-mode', action='store_true',
                   help='Enable additional soft row/column band notches for smooth '
                        'ripple-like FFT lines (default: off).')
    p.add_argument('--destripe-band-sigma-thresh', type=float, default=4.0,
                   help='Robust sigma threshold for detecting row/column FFT band outliers '
                        '(default: 4.0).')
    p.add_argument('--destripe-band-max-runs', type=int, default=4,
                   help='Maximum independent row and column bands to notch per iteration '
                        '(default: 4; conjugates added automatically).')
    p.add_argument('--destripe-band-min-run-pix', type=int, default=3,
                   help='Minimum contiguous band width in FFT bins to consider a band '
                        '(default: 3).')
    p.add_argument('--destripe-band-sigma-pix', type=float, default=6.0,
                   help='Gaussian half-width (sigma) of soft band notch in FFT pixels '
                        '(default: 6.0).')
    p.add_argument('--destripe-band-depth', type=float, default=0.35,
                   help='Soft band notch depth [0,1] (default: 0.35).')
    p.add_argument('--destripe-radial-inner-pix', type=float, default=20.0,
                   help='For radial-bandpass strategy: inner UV cutoff radius in FFT pixels '
                        '(suppresses very low spatial frequencies).')
    p.add_argument('--destripe-radial-outer-pix', type=float, default=-1.0,
                   help='For radial-bandpass strategy: outer UV cutoff radius in FFT pixels '
                        '(suppresses very high spatial frequencies). '
                        'Use <=0 for auto (~0.48*min(Nx,Ny)).')
    p.add_argument('--destripe-radial-inner-taper-pix', type=float, default=4.0,
                   help='For radial-bandpass strategy: smooth taper width at inner cutoff '
                        'in FFT pixels (default: 4.0).')
    p.add_argument('--destripe-radial-outer-taper-pix', type=float, default=6.0,
                   help='For radial-bandpass strategy: smooth taper width at outer cutoff '
                        'in FFT pixels (default: 6.0).')
    p.add_argument('--destripe-shell-radius-pix', type=float, default=0.0,
                   help='For radial-shell strategy: ring peak radius in FFT pixels '
                        '(0 gives Gaussian low-pass centered at DC).')
    p.add_argument('--destripe-shell-sigma-pix', type=float, default=10.0,
                   help='For radial-shell strategy: Gaussian shell width (sigma) in FFT pixels.')
    p.add_argument('--provenance-dir', default='',
                   help='Write run provenance files (.cmd and .log) to this directory. '
                        'If not set, defaults to ./provenance_logs/')
    args = p.parse_args()

    # Generate provenance every time with timestamps
    prov_dir_arg = args.provenance_dir.strip()
    if prov_dir_arg:
        prov_dir = Path(prov_dir_arg).expanduser().resolve()
    else:
        # Default: create ./provenance_logs/ in current directory
        prov_dir = Path.cwd() / 'provenance_logs'
    
    prov_dir.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_stem = Path(args.output).stem if args.output else 'stack_moon_snapshots'
    cmd_file = prov_dir / f'run_stack_moon_snapshots_{out_stem}_{run_ts}.cmd'
    log_file = prov_dir / f'run_stack_moon_snapshots_{out_stem}_{run_ts}.log'

    cmd = ['python', str(Path(__file__).resolve()), *sys.argv[1:]]
    with open(cmd_file, 'w', encoding='utf-8') as fcmd:
        fcmd.write(f'# timestamp={run_ts}\n')
        fcmd.write(f'# cwd={Path.cwd()}\n')
        fcmd.write(' '.join([shlex.quote(c) for c in cmd]))
        fcmd.write('\n')

    log_handle = open(log_file, 'w', encoding='utf-8')
    sys.stdout = TeeStream(sys.stdout, log_handle)
    sys.stderr = TeeStream(sys.stderr, log_handle)
    print(f'[provenance] cmd: {cmd_file}')
    print(f'[provenance] log: {log_file}')

    selfcal_dir = Path(args.selfcal_dir).expanduser().resolve()
    if not selfcal_dir.is_dir():
        sys.exit(f'ERROR: selfcal-dir not found: {selfcal_dir}')

    fits_paths = sorted(
        str(p) for p in selfcal_dir.rglob(args.glob_pat)
    )
    if not fits_paths:
        sys.exit(f'ERROR: no files matching {args.glob_pat!r} under {selfcal_dir}')

    if args.destripe_write_dir.strip():
        name_counts = Counter(os.path.basename(path) for path in fits_paths)
        duplicate_names = sorted(name for name, count in name_counts.items() if count > 1)
        if duplicate_names:
            print('ERROR: duplicate FITS basenames found for destripe-write-dir output. '
                  'This would overwrite files in the output directory.')
            print('       Use a stricter --glob that uniquely selects source stacks.')
            for name in duplicate_names[:10]:
                print(f'       duplicate: {name}')
            if len(duplicate_names) > 10:
                print(f'       ... and {len(duplicate_names) - 10} more')
            sys.exit(2)

    print(f'[stack] Found {len(fits_paths)} FITS files under {selfcal_dir}')

    requested_ref_idx = args.reference_idx

    # Load all images
    images = []
    loaded_paths: list[str] = []
    for path in fits_paths:
        try:
            images.append(load_2d(path))
            loaded_paths.append(path)
        except Exception as e:
            print(f'[stack] WARNING: skipping {path}: {e}')
    if not images:
        sys.exit('ERROR: no valid images loaded.')

    if requested_ref_idx is not None and (requested_ref_idx < 0 or requested_ref_idx >= len(fits_paths)):
        sys.exit(f'ERROR: reference-idx {requested_ref_idx} is out of range for {len(fits_paths)} discovered files.')

    # Validate shapes
    shape = images[0].shape
    valid = [(i, img, loaded_paths[i]) for i, img in enumerate(images) if img.shape == shape]
    skipped = len(images) - len(valid)
    if skipped:
        print(f'[stack] WARNING: {skipped} image(s) have mismatched shape and will be skipped.')
    idxs, images, valid_paths = zip(*valid)
    valid_paths = list(valid_paths)
    images = list(images)

    # Choose reference frame: explicit index if given, else closest to mean JD.
    if requested_ref_idx is not None:
        requested_ref_path = fits_paths[requested_ref_idx]
        if requested_ref_path in valid_paths:
            new_ref_idx = valid_paths.index(requested_ref_path)
            print(f'[stack] Reference stack from --reference-idx: [{new_ref_idx}] {os.path.basename(valid_paths[new_ref_idx])}')
        else:
            print(f'[stack] WARNING: requested reference not in valid-shape set; falling back to mean-JD reference.')
            requested_ref_idx = None

    if requested_ref_idx is None:
        jds = []
        for path in valid_paths:
            hdr = load_header(path)
            jds.append(representative_jd_from_header(hdr))
        jd_arr = np.array(jds, dtype=np.float64)
        jd_mean = float(np.mean(jd_arr))
        new_ref_idx = int(np.argmin(np.abs(jd_arr - jd_mean)))
        print(f'[stack] Reference stack from mean JD: [{new_ref_idx}] {os.path.basename(valid_paths[new_ref_idx])} '
              f'(JD={jd_arr[new_ref_idx]:.8f}, mean={jd_mean:.8f})')

    raw_ref_img = images[new_ref_idx]
    ref_header = load_header(valid_paths[new_ref_idx])

    raw_images = list(images)

    # Optional iterative Fourier destriping before registration
    if args.destripe_iters > 0:
        print(f'\n[stack] Destriping enabled: {args.destripe_iters} iterations/image')
        print(f'[stack] Moon-mask radius: {args.moon_mask_radius_arcmin:.2f} arcmin')
        print(f'[stack] Destripe strategy: {args.destripe_strategy}')
        if args.destripe_strategy == 'legacy-notch' and args.destripe_band_mode:
            print(f'[stack] Band-notch mode: on '
                  f'(thr={args.destripe_band_sigma_thresh:.2f}σ, '
                  f'max_runs={args.destripe_band_max_runs}, '
                  f'min_run={args.destripe_band_min_run_pix}, '
                  f'sigma={args.destripe_band_sigma_pix:.2f}, '
                  f'depth={args.destripe_band_depth:.2f})')
        if args.destripe_strategy == 'radial-bandpass':
            radial_outer_txt = (
                f'{args.destripe_radial_outer_pix:.2f}'
                if args.destripe_radial_outer_pix > 0 else 'auto(0.48*min(Nx,Ny))'
            )
            print(f'[stack] Radial band-pass: inner={args.destripe_radial_inner_pix:.2f}px '
                  f'(taper={args.destripe_radial_inner_taper_pix:.2f}), '
                  f'outer={radial_outer_txt}px '
                  f'(taper={args.destripe_radial_outer_taper_pix:.2f})')
        if args.destripe_strategy == 'radial-shell':
            print(f'[stack] Radial shell: r0={args.destripe_shell_radius_pix:.2f}px '
                  f'sigma={args.destripe_shell_sigma_pix:.2f}px')
        debug_root = None
        if args.destripe_debug_dir.strip():
            debug_root = Path(args.destripe_debug_dir).expanduser().resolve()
            debug_root.mkdir(parents=True, exist_ok=True)
            print(f'[stack] Fourier debug outputs: {debug_root}')
        px_scale_arcsec = abs(ref_header.get('CDELT2', 0.0004166666666667)) * 3600.0
        moon_radius_pix = (args.moon_mask_radius_arcmin * 60.0) / px_scale_arcsec
        destripe_write_dir = None
        if args.destripe_write_dir.strip():
            destripe_write_dir = Path(args.destripe_write_dir).expanduser().resolve()
            destripe_write_dir.mkdir(parents=True, exist_ok=True)
            print(f'[stack] Destriped per-frame FITS output: {destripe_write_dir}')
        destriped = []
        for i, img in enumerate(images):
            fname = os.path.basename(valid_paths[i])
            stem = Path(fname).stem
            ny, nx = img.shape
            img_header = load_header(valid_paths[i])

            if args.moon_mask_center == 'center':
                center_rc = (0.5 * (ny - 1), 0.5 * (nx - 1))
            elif args.moon_mask_center == 'ephem':
                center_rc = moon_center_from_ephem(img_header, img.shape)
            else:
                # Mild blur via FFT low-pass for robust peak location
                ff = np.fft.fft2(img)
                rr, cc = np.indices(img.shape)
                cy, cx = ny // 2, nx // 2
                r2 = (rr - cy) ** 2 + (cc - cx) ** 2
                lp = np.exp(-0.5 * r2 / (12.0 ** 2))
                smooth = np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(ff) * lp)).real
                pr, pc = np.unravel_index(np.argmax(smooth), smooth.shape)
                center_rc = (float(pr), float(pc))

            moon_mask = build_circular_mask(img.shape, center_rc, moon_radius_pix)
            fixed_fft_vmin: Optional[float] = None
            fixed_fft_vmax: Optional[float] = None
            fixed_img_vmin: Optional[float] = None
            fixed_img_vmax: Optional[float] = None
            fft_u_klambda: Optional[np.ndarray] = None
            fft_v_klambda: Optional[np.ndarray] = None

            if args.destripe_debug_fft_axis == 'klambda':
                cdelt1 = float(abs(img_header.get('CDELT1', ref_header.get('CDELT1', 0.0))))
                cdelt2 = float(abs(img_header.get('CDELT2', ref_header.get('CDELT2', 0.0))))
                if cdelt1 > 0 and cdelt2 > 0:
                    dx_rad = np.deg2rad(cdelt1)
                    dy_rad = np.deg2rad(cdelt2)
                    fft_u_klambda = np.fft.fftshift(np.fft.fftfreq(nx, d=dx_rad)) / 1e3
                    fft_v_klambda = np.fft.fftshift(np.fft.fftfreq(ny, d=dy_rad)) / 1e3
                else:
                    print(f'[destripe] WARNING: {stem} missing CDELT1/2; fallback FFT debug axis to pixel units.')

            def _debug_callback(iter_idx: int,
                                power_before: np.ndarray,
                                peaks: list[tuple[int, int]],
                                multiplier: np.ndarray,
                                power_after: np.ndarray,
                                image_before: np.ndarray,
                                image_filtered: np.ndarray,
                                image_hybrid: np.ndarray) -> None:
                nonlocal fixed_fft_vmin, fixed_fft_vmax, fixed_img_vmin, fixed_img_vmax
                if debug_root is None:
                    return

                if iter_idx == 0:
                    if args.destripe_debug_fft_scale_mode in ('global', 'windowed'):
                        scale_axis = 'pix' if args.destripe_debug_fft_scale_mode == 'global' else args.destripe_debug_fft_axis
                        scale_max_kl = None if args.destripe_debug_fft_scale_mode == 'global' else args.destripe_debug_fft_max_klambda
                        fft_combined = fft_scale_values(
                            power_before=power_before,
                            power_after=power_after,
                            fft_axis=scale_axis,
                            fft_u_klambda=fft_u_klambda,
                            fft_v_klambda=fft_v_klambda,
                            fft_max_klambda=scale_max_kl,
                        )
                        fixed_fft_vmin = float(np.percentile(fft_combined, args.destripe_debug_fft_scale_low))
                        fixed_fft_vmax = float(np.percentile(fft_combined, args.destripe_debug_fft_scale_high))

                    img_combined = np.concatenate([
                        image_before[np.isfinite(image_before)].ravel(),
                        image_filtered[np.isfinite(image_filtered)].ravel(),
                        image_hybrid[np.isfinite(image_hybrid)].ravel(),
                    ])
                    if img_combined.size > 0:
                        fixed_img_vmin = float(np.percentile(img_combined, 2.0))
                        fixed_img_vmax = float(np.percentile(img_combined, 99.8))
                    else:
                        fixed_img_vmin, fixed_img_vmax = -1.0, 1.0

                    fft_scale_msg = (
                        f'FFT=[{fixed_fft_vmin:.3f}, {fixed_fft_vmax:.3f}]'
                        if fixed_fft_vmin is not None and fixed_fft_vmax is not None
                        else 'FFT=per-iter auto'
                    )
                    print(f'[destripe] {stem} fixed scales: {fft_scale_msg}, '
                          f'IMG=[{fixed_img_vmin:.6f}, {fixed_img_vmax:.6f}]')

                fft_plot_vmin = fixed_fft_vmin if args.destripe_debug_fft_scale_mode != 'per-iter' else None
                fft_plot_vmax = fixed_fft_vmax if args.destripe_debug_fft_scale_mode != 'per-iter' else None

                img_dir = debug_root / stem
                img_dir.mkdir(parents=True, exist_ok=True)
                combined_png = img_dir / f'iter_{iter_idx:02d}_debug.png'
                save_combined_debug_plot(
                    power_before=power_before,
                    peaks=peaks,
                    multiplier=multiplier,
                    power_after=power_after,
                    core_radius_pix=args.destripe_core_radius_pix,
                    image_before=image_before,
                    image_filtered=image_filtered,
                    image_hybrid=image_hybrid,
                    moon_mask=moon_mask,
                    iter_idx=iter_idx,
                    out_png=combined_png,
                    fft_vmin=fft_plot_vmin,
                    fft_vmax=fft_plot_vmax,
                    img_vmin=fixed_img_vmin,
                    img_vmax=fixed_img_vmax,
                    fft_axis=args.destripe_debug_fft_axis,
                    fft_u_klambda=fft_u_klambda,
                    fft_v_klambda=fft_v_klambda,
                    fft_max_klambda=args.destripe_debug_fft_max_klambda,
                    fft_scale_mode=args.destripe_debug_fft_scale_mode,
                    fft_scale_low=args.destripe_debug_fft_scale_low,
                    fft_scale_high=args.destripe_debug_fft_scale_high,
                )

                if args.destripe_save_npz:
                    np.savez_compressed(
                        img_dir / f'iter_{iter_idx:02d}_fft_arrays.npz',
                        power_before=power_before.astype(np.float32),
                        power_after=power_after.astype(np.float32),
                        multiplier=multiplier.astype(np.float32),
                        peaks=np.array(peaks, dtype=np.int32),
                        image_before=image_before.astype(np.float32),
                        image_filtered=image_filtered.astype(np.float32),
                        image_hybrid=image_hybrid.astype(np.float32),
                        moon_mask=moon_mask.astype(np.uint8),
                    )

            out_img, peaks_hist, rms_hist = iterative_fft_destripe(
                img=img,
                moon_mask=moon_mask,
                n_iter=args.destripe_iters,
                strategy=args.destripe_strategy,
                core_radius_pix=args.destripe_core_radius_pix,
                sigma_thresh=args.destripe_sigma_thresh,
                max_peaks=args.destripe_max_peaks,
                min_separation_pix=args.destripe_min_separation_pix,
                notch_sigma_pix=args.destripe_notch_sigma_pix,
                notch_depth=float(np.clip(args.destripe_notch_depth, 0.0, 1.0)),
                radial_inner_radius_pix=max(0.0, args.destripe_radial_inner_pix),
                radial_outer_radius_pix=args.destripe_radial_outer_pix,
                radial_inner_taper_pix=max(0.25, args.destripe_radial_inner_taper_pix),
                radial_outer_taper_pix=max(0.25, args.destripe_radial_outer_taper_pix),
                shell_radius_pix=max(0.0, args.destripe_shell_radius_pix),
                shell_sigma_pix=max(0.25, args.destripe_shell_sigma_pix),
                band_mode=args.destripe_band_mode,
                band_sigma_thresh=args.destripe_band_sigma_thresh,
                band_max_runs=max(0, args.destripe_band_max_runs),
                band_min_run_pix=max(1, args.destripe_band_min_run_pix),
                band_sigma_pix=max(0.5, args.destripe_band_sigma_pix),
                band_depth=float(np.clip(args.destripe_band_depth, 0.0, 1.0)),
                debug_callback=_debug_callback,
            )
            destriped.append(out_img)

            if destripe_write_dir is not None:
                out_frame = destripe_write_dir / fname
                out_header = img_header.copy()
                out_header['HISTORY'] = (
                    f'Iterative FFT destripe applied; n_iter={args.destripe_iters}, '
                    f'band_mode={int(args.destripe_band_mode)}'
                )
                out_data = out_img[np.newaxis, np.newaxis, :, :]
                fits.writeto(str(out_frame), out_data, out_header, overwrite=True)
            
            # Save RMS convergence plot
            if debug_root is not None:
                save_destripe_convergence_plot(rms_hist, stem, debug_root / stem)
                save_destripe_debug_animation(debug_root / stem, stem,
                                             fps=args.destripe_debug_animation_fps)
            
            print(f'[destripe] {fname}: center=({center_rc[0]:.1f},{center_rc[1]:.1f}) '
                  f'r={moon_radius_pix:.1f}px peaks/iter={peaks_hist} rms/iter={[f"{r:.2e}" for r in rms_hist]}')
        images = destriped

    proc_ref_img = images[new_ref_idx]

    px_scale_arcsec = abs(ref_header.get('CDELT2', 0.0004166666666667)) * 3600.0
    print(f'[stack] Registration mode: {args.registration_mode}')
    if args.registration_mode == 'derive':
        print(f'[stack] Registration method: {args.registration_method} '
              f'(upsample={args.upsample})')
    else:
        print('[stack] Registration method: <none> (no image shifts will be applied)')
        print('[stack] Shift table values are fixed to 0.0 pix (direct co-add mode).')
    print(f'[stack] Shift conversion: 1 pix = {px_scale_arcsec:.4f} arcsec')
    print(f'\n{"#":>4}  {"file":<45}  {"dr_pix":>8}  {"dc_pix":>8}  {"dr_arcsec":>10}  {"dc_arcsec":>10}  {"r_arcmin":>8}')
    print('-' * 114)
    result = stack_registered_images(
        images=images,
        ref_img=proc_ref_img,
        valid_paths=valid_paths,
        ref_idx=new_ref_idx,
        upsample=args.upsample,
        method=args.method,
        registration_mode=args.registration_mode,
        registration_method=args.registration_method,
        px_scale_arcsec=px_scale_arcsec,
        print_shift_table=True,
    )

    print(f'\n[stack] Stacked {len(images)} images using {args.method}.')
    print(f'[stack] Result shape: {result.shape}   '
          f'peak={result.max():.4f}  rms={result.std():.4f}')

    # Write output — restore degenerate axes to match CASA convention (1,1,Ny,Nx)
    out_data = result[np.newaxis, np.newaxis, :, :]
    out_hdr = ref_header.copy()
    # Update history
    out_hdr['HISTORY'] = (f'Stack of {len(images)} moon snapshots '
                          f'({args.method}, ref_idx={new_ref_idx}, '
                          f'upsample={args.upsample}, '
                          f'reg_mode={args.registration_mode}, '
                          f'reg_method={args.registration_method})')

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fits.writeto(str(out_path), out_data, out_hdr, overwrite=True)
    print(f'[stack] Written: {out_path}')

    if args.destripe_iters > 0:
        original_result = stack_registered_images(
            images=raw_images,
            ref_img=raw_ref_img,
            valid_paths=valid_paths,
            ref_idx=new_ref_idx,
            upsample=args.upsample,
            method=args.method,
            registration_mode=args.registration_mode,
            registration_method=args.registration_method,
            px_scale_arcsec=px_scale_arcsec,
            print_shift_table=False,
        )
        original_png = out_path.parent / f'{out_path.stem}_original.png'
        destriped_png = out_path.parent / f'{out_path.stem}_destriped.png'
        save_stack_preview_png(original_result, original_png,
                               f'Final Stack (original, no destripe): {out_path.stem}')
        save_stack_preview_png(result, destriped_png,
                               f'Final Stack (destriped): {out_path.stem}')
    else:
        final_png = out_path.parent / f'{out_path.stem}.png'
        save_stack_preview_png(result, final_png,
                               f'Final Stack: {out_path.stem}')


if __name__ == '__main__':
    main()
