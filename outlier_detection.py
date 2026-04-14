"""
outlier_detection.py — Standalone per-channel visibility outlier detector
==========================================================================

Usage
-----
    # Run with CONFIG-block defaults:
    python3 outlier_detection.py

    # Override the most common parameters from the command line:
    python3 outlier_detection.py \\
        --fits  /data/40_014.FITS \\
        --bp    /work/3c48_bp.npz \\
        --cal   3C48 \\
        --chans 64-191 \\
        --out   /work/outlier_detection

    # Single channel (like Fortran my_uvflg.f chan_num):
    python3 outlier_detection.py --chans 128

    # Dry run (print what would be written, no files):
    python3 outlier_detection.py --dry-run

    # Write output files:
    python3 outlier_detection.py --no-dry-run

CLI arguments override CONFIG-block defaults; all are optional.
Edit the CONFIG block for persistent settings.

Overview
--------
1. Load (or build) the row index from the FITS file.
2. Load calibrator visibilities and apply the bandpass solution.
3. For each metric (RR, LL, V) and each channel, compute a per-row test quantity
   and mark rows that exceed the threshold as bad.
4. Cluster bad rows conservatively:
       Tier-0 — Whole scan   (≥ WHOLE_SCAN_BAD_FRACTION bad → flag entire scan)
       Level 1 — Antenna     (global: bad fraction ≥ MIN_CLUSTER_FRACTION;
                               per-scan: methods 1 + 2 from Fortran my_uvflg.f)
       Level 2 — Baseline    (global: bad fraction ≥ MIN_CLUSTER_FRACTION;
                               per-scan: count ≥ BASELINE_MAX_BAD_SAMPLES)
       Level 3 — Time window (many baselines bad simultaneously → RFI burst)
5. Merge adjacent time intervals separated by less than max_gap_minutes.
6. Output:
       • JSON flag table   (bad_antennas, bad_baselines, bad_timeranges)
       • CASA flagdata commands text file
7. Optionally transfer flags to a target source (antenna/baseline flags always
   transfer; time-range flag transfer is controlled by TRANSFER_TIMERANGE_FLAGS).

Calibrator → target transfer rationale
---------------------------------------
All flagging is fundamentally time-dependent: every bad row was measured at a
specific time on a specific baseline.  This script distinguishes two tiers:

  WHOLESALE flags (no time qualifier in JSON output):
    A coordinate (antenna or baseline) was bad for ≥ MIN_CLUSTER_FRACTION
    (default 80%) of all calibrator rows.  So pervasive that collapsing the
    time dimension is justified.  When applied to the target, the coordinate
    is flagged for the entire target scan — assuming persistent hardware trouble.

  TIME-RANGE flags (UTC-bounded in JSON output):
    A coordinate was bad only during specific UTC windows on the calibrator.
    TRANSFER_TIMERANGE_FLAGS controls whether those windows are also applied to
    target data (True = copy UTC windows; False = drop, only wholesale transfers).
"""

from __future__ import annotations

# =============================================================================
# ── CONFIG ────────────────────────────────────────────────────────────────────
# Edit this block to match your observation.  Every value here can also be
# overridden from the command line (run with --help to see all options).
# =============================================================================

import argparse
import sys
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# ── File paths ────────────────────────────────────────────────────────────────
FITS_PATH       = Path('/Users/raj030/DATA/gmrt_40_014/data/40_014_25jul2021_gsb.FITS')
INDEX_CACHE     = Path('/Users/raj030/DATA/gmrt_40_014/work/40_014_25jul2021_gsb.row_index_cache.npz')
BANDPASS_NPZ    = Path('/Users/raj030/DATA/gmrt_40_014/work/3c48_bandpass_25jul_gsb.npz')
OUT_DIR         = Path('/Users/raj030/DATA/gmrt_40_014/work/outlier_detection')

# ── Sources ───────────────────────────────────────────────────────────────────
CALIBRATOR_SOURCE = '3C48'    # Source used to identify bad data (known flux)
TARGET_SOURCES    = []        # e.g. ['J0319+4130'] — flags will be copied here;
                              # leave empty [] to skip target transfer

# ── Known dud antennas (excluded from all counts and solves) ──────────────────
DUD_ANTENNA_NAMES = ['C07', 'S05']

# ── Channel selection ─────────────────────────────────────────────────────────
CHAN_RANGE = (64, 191)        # 0-based inclusive channel indices (FITS convention)

# ── Outlier detection — single-correlator mode ────────────────────────────────
#
# DETECTION_CORR selects which correlator product is used in a given run.
# ONE value per run.  The required STOKES labels are loaded automatically.
#
# The test quantity is always an amplitude (Jy).  Rows exceeding
# DETECTION_THRESHOLD_JY are marked bad and passed to the clustering step.
#
# ┌──────┬──────────────────────────────────────────────────────────────────────┐
# │ CORR │ Test quantity                                                       │
# ├──────┼──────────────────────────────────────────────────────────────────────┤
# │ 'V'  │ |RR − LL| / 2     Stokes-V proxy                  [loads RR, LL] │
# │ 'Q'  │ |XX − YY| / 2     Stokes-Q proxy                  [loads XX, YY] │
# │ 'I'  │ (|RR| + |LL|) / 2 mean parallel-hand amplitude     [loads RR, LL] │
# │ 'RR' │ |RR|               amplitude                        [loads RR]     │
# │ 'LL' │ |LL|               amplitude                        [loads LL]     │
# │ 'XX' │ |XX|               amplitude                        [loads XX]     │
# │ 'YY' │ |YY|               amplitude                        [loads YY]     │
# │ 'RL' │ |RL|               cross-hand amplitude             [loads RL]     │
# │ 'LR' │ |LR|               cross-hand amplitude             [loads LR]     │
# │ 'XY' │ |XY|               cross-hand amplitude             [loads XY]     │
# │ 'YX' │ |YX|               cross-hand amplitude             [loads YX]     │
# └──────┴──────────────────────────────────────────────────────────────────────┘
#
# Default: 'V' with 5.0 Jy — works at any calibration stage.
DETECTION_CORR         = 'V'
DETECTION_THRESHOLD_JY = 5.0

# ── Clustering parameters ─────────────────────────────────────────────────────
# Fraction of a coordinate's rows that must be bad before it is flagged WHOLESALE
# (no time restriction).  Set conservatively high: a coordinate flagged wholesale
# is treated as persistently bad and transferred to the target with no UTC limit.
# Coordinates whose bad fraction is below this threshold receive TIME-RANGE flags
# only for the specific UTC windows where they were actually bad.
MIN_CLUSTER_FRACTION        = 0.80   # 80% — flag whole coordinate only if bad for most of observation

# Minimum number of distinct baselines that must be bad for an antenna to be
# flagged whole.  Prevents one noisy baseline from condemning an antenna.
MIN_DISTINCT_BASELINES_FOR_ANT = 3

# Minimum fraction of active baselines that must be simultaneously bad in a
# time window for a global time-burst flag to be raised.
MIN_BURST_BASELINE_FRACTION = 0.50   # 50%

# ── Fortran-inspired per-scan thresholds (ported from my_uvflg.f, 2011) ───────
# These thresholds apply per scan, supplementing the global wholesale detection.
#
# Tier-0: flag an entire scan if this fraction of all vis in that scan are bad.
# Mirrors Fortran: ibad_scan >= 0.7 × nbase × nsamp_per_scan
WHOLE_SCAN_BAD_FRACTION = 0.70

# Per-scan antenna method 1: an antenna is flagged for the full scan if its
# bad-vis fraction within that scan exceeds this threshold.
# Mirrors Fortran: ibad_ante >= 0.6 × nante × scan_secs / tsamp_sec
PER_SCAN_ANT_FRACTION = 0.60

# Per-scan antenna method 2: an antenna is flagged for the full scan if it
# has > BASELINE_MAX_BAD_SAMPLES bad vis on > PER_SCAN_ANT_BL_FRACTION of
# its baselines within that scan.  Requires BASELINE_MAX_BAD_SAMPLES != None.
# Mirrors Fortran: ibad_ante > maxbad_allowed on > 50% of baselines
PER_SCAN_ANT_BL_FRACTION = 0.50

# Absolute bad-sample count threshold per scan per baseline.
# A baseline is treated as wholly bad within a scan if its bad count >= this.
# Also used as the per-baseline cutoff for antenna method 2 above.
# Set to None to disable the absolute-count backstop entirely.
# Mirrors Fortran: maxbad_allowed parameter in the parfile
BASELINE_MAX_BAD_SAMPLES = 100       # set to None to disable

# Integration-time-aware gap tolerance for time-range merging.
# Adjacent bad intervals separated by <= MAX_GAP_SAMPLES × (integration time)
# are merged into one.  The effective gap = max(this, MAX_GAP_MINUTES), so
# MAX_GAP_MINUTES still acts as a useful calendar floor.
# Mirrors Fortran: tol_sec / tsamp_sec — sample-count-based gap tolerance.
MAX_GAP_SAMPLES = 2                  # merge if gap <= 2 integration periods

# Merge adjacent bad time intervals separated by less than this many minutes.
MAX_GAP_MINUTES = 30.0

# ── Elevation cut ─────────────────────────────────────────────────────────────
ELEVATION_MIN_DEG = 25.0

# ── Max rows to load ──────────────────────────────────────────────────────────
MAX_ROWS = 150_000

# ── Flag transfer ─────────────────────────────────────────────────────────────
# Calibrator → target flag transfer.
#
# Two classes of flags are produced for each coordinate (antenna or baseline):
#
#   1. WHOLESALE flag (no time qualifier):
#      The coordinate was bad for ≥ MIN_CLUSTER_FRACTION (default 80%) of ALL
#      calibrator rows.  This pervasive badness is assumed to reflect a persistent
#      hardware or standing-wave problem that exists throughout the observation.
#      It is ALWAYS transferred to the target with NO time restriction — every
#      target row on that antenna/baseline is flagged.
#
#   2. TIME-RANGE flag:
#      The coordinate was bad only during specific UTC windows on the calibrator.
#      These windows correspond to events (RFI burst, ionospheric glitch, etc.)
#      that occurred at those real times.  Whether the same UTC windows should
#      be flagged in the target is controlled by TRANSFER_TIMERANGE_FLAGS:
#
#        True  → copy UTC windows to target flag table
#                 (use when cal and target are interleaved and RFI events are
#                  synchronous — the same burst would contaminate both)
#        False → discard time-range flags for the target;
#                 only wholesale (persistent) flags are transferred
#                 (use when cal and target are observed at very different times
#                  and time-range flags are specific to the cal scan geometry)
TRANSFER_TIMERANGE_FLAGS = True

# ── Scan detection ────────────────────────────────────────────────────────────
# Consecutive timestamps separated by more than this gap (in minutes) are
# treated as belonging to different scans.
SCAN_GAP_MINUTES = 2.0

# ── Output ────────────────────────────────────────────────────────────────────
# Dry-run: compute proposals but do not write any files.
DRY_RUN = True

# =============================================================================
# ── END OF CONFIG ─────────────────────────────────────────────────────────────
# =============================================================================


# ---------------------------------------------------------------------------
# Correlator / Stokes registry
# ---------------------------------------------------------------------------

# Maps each DETECTION_CORR name → the STOKES labels that must be loaded.
_CORR_STOKES_NEEDED: Dict[str, Tuple[str, ...]] = {
    'V':  ('RR', 'LL'),
    'Q':  ('XX', 'YY'),
    'I':  ('RR', 'LL'),
    'RR': ('RR',),
    'LL': ('LL',),
    'XX': ('XX',),
    'YY': ('YY',),
    'RL': ('RL',),
    'LR': ('LR',),
    'XY': ('XY',),
    'YX': ('YX',),
}


# ---------------------------------------------------------------------------
# Scan helpers
# ---------------------------------------------------------------------------

def detect_scans(jd: np.ndarray, gap_minutes: float = 2.0) -> List[Tuple[int, int]]:
    """Return list of (start_idx, end_idx_exclusive) for each scan.

    A new scan begins whenever the JD gap between consecutive rows exceeds
    *gap_minutes*.
    """
    if len(jd) == 0:
        return []
    gap_jd = gap_minutes / (24.0 * 60.0)
    breaks = np.where(np.diff(jd) > gap_jd)[0] + 1
    starts = np.concatenate([[0], breaks])
    ends   = np.concatenate([breaks, [len(jd)]])
    return list(zip(starts.tolist(), ends.tolist()))


def jd_to_iso(jd: float) -> str:
    """Convert a Julian Date scalar to an ISO 8601 UTC string."""
    try:
        from astropy.time import Time
        return Time(jd, format='jd', scale='utc').iso
    except ImportError:
        # Fallback: approximate conversion
        # JD 2451545.0 = J2000.0 = 2000-01-01 12:00:00 UTC
        days_since_j2000 = jd - 2451545.0
        import datetime
        t = datetime.datetime(2000, 1, 1, 12, 0, 0) + datetime.timedelta(days=days_since_j2000)
        return t.strftime('%Y-%m-%d %H:%M:%S')


def jd_to_casa(jd: float) -> str:
    """Return a CASA-format timerange string component: YYYY/MM/DD/HH:MM:SS."""
    iso = jd_to_iso(jd)
    # iso: '2021-07-25 19:05:30.0'
    date_part, time_part = iso.split(' ')
    date_part = date_part.replace('-', '/')
    time_part = time_part.split('.')[0]   # drop sub-second
    return f'{date_part}/{time_part}'


# ---------------------------------------------------------------------------
# Test-quantity computation  (single-correlator)
# ---------------------------------------------------------------------------

def compute_test_quantity(
    vis: dict,
    corr: str,
) -> np.ndarray:
    """Return an amplitude-like test-quantity array of shape (nrows, nchans).

    This is purely a clustering input: rows that exceed DETECTION_THRESHOLD_JY
    are marked bad and passed to the multi-tier clustering step.
    No flux model or sky model is involved.

    Parameters
    ----------
    vis : dict
        Output of :func:`apply_bandpass_solution` — must contain
        ``vis_complex_corrected`` and ``stokes_labels``.
    corr : str
        One of the keys in ``_CORR_STOKES_NEEDED``.  See the CONFIG block.

    Returns
    -------
    float32 ndarray of shape (nrows, nchans).
    """
    if corr not in _CORR_STOKES_NEEDED:
        raise ValueError(
            f"Unknown DETECTION_CORR '{corr}'. "
            f"Supported values: {sorted(_CORR_STOKES_NEEDED)}"
        )

    stokes_labels = vis['stokes_labels']
    vc = vis['vis_complex_corrected']          # (nrows, nchans, npols)

    def _get(label: str) -> Optional[np.ndarray]:
        if label in stokes_labels:
            return vc[:, :, stokes_labels.index(label)]
        return None

    if corr == 'V':
        rr, ll = _get('RR'), _get('LL')
        if rr is None or ll is None:
            raise RuntimeError("DETECTION_CORR='V' requires both RR and LL in the loaded data.")
        tq = np.abs(rr - ll) / 2.0

    elif corr == 'Q':
        xx, yy = _get('XX'), _get('YY')
        if xx is None or yy is None:
            raise RuntimeError("DETECTION_CORR='Q' requires both XX and YY in the loaded data.")
        tq = np.abs(xx - yy) / 2.0

    elif corr == 'I':
        rr, ll = _get('RR'), _get('LL')
        if rr is None or ll is None:
            raise RuntimeError("DETECTION_CORR='I' requires both RR and LL in the loaded data.")
        tq = (np.abs(rr) + np.abs(ll)) / 2.0

    elif corr in ('RR', 'LL', 'XX', 'YY', 'RL', 'LR', 'XY', 'YX'):
        pol = _get(corr)
        if pol is None:
            raise RuntimeError(f"DETECTION_CORR='{corr}' not found in loaded data.")
        tq = np.abs(pol)

    else:
        raise ValueError(f"Unhandled corr '{corr}'")

    return tq.astype(np.float32)


def build_bad_mask(
    tq: np.ndarray,
    threshold_jy: float,
    threshold_low: Optional[float] = None,
) -> np.ndarray:
    """Return a boolean row mask for a single-channel test-quantity slice.

    Parameters
    ----------
    tq : ndarray of shape (nrows, 1)
        Single-channel slice from :func:`compute_test_quantity`.
    threshold_jy : float
        Flag a row if the test quantity exceeds this value for the channel.
    threshold_low : float or None
        If given, also flag rows whose test quantity falls *below* this value.
        Useful for catching dead antennas with anomalously low power on RR/LL
        correlations.  Has no meaningful use for Stokes-V (which is ~0 for
        clean data).  Default is ``None`` (no lower-bound clip).

    Returns
    -------
    boolean ndarray of shape (nrows,).
    """
    bad = (tq > threshold_jy) | (~np.isfinite(tq))   # (nrows, 1)
    if threshold_low is not None:
        bad = bad | (tq < threshold_low)
    return bad.any(axis=1)                             # (nrows,)



# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def cluster_flags(
    vis: dict,
    bad_mask: np.ndarray,
    antenna_name_map: Dict[int, str],
    min_cluster_fraction: float,
    min_distinct_baselines_for_ant: int,
    min_burst_baseline_fraction: float,
    max_gap_minutes: float,
    scan_gap_minutes: float,
    *,
    whole_scan_bad_fraction: float = 0.70,
    per_scan_ant_fraction: float = 0.60,
    per_scan_ant_bl_fraction: float = 0.50,
    baseline_max_bad_samples: Optional[int] = None,
    max_gap_samples: int = 2,
) -> dict:
    """Multi-tier flag clustering (validated thresholds ported from Fortran my_uvflg.f).

    Tier-0  — whole-scan bad (>= whole_scan_bad_fraction of scan rows).  These
               scans are covered by bad_scan_timeranges and skipped in lower tiers.
    Level 1 — global per-antenna wholesale (bad fraction >= min_cluster_fraction
               across the full dataset, spread across >= min_distinct_baselines).
    Level 1b— per-scan per-antenna:
               Method 1 (Fortran): fraction of antenna vis in scan >= per_scan_ant_fraction.
               Method 2 (Fortran): count > baseline_max_bad_samples on >=
               per_scan_ant_bl_fraction of baselines in scan.
    Level 2 — global per-baseline wholesale (bad fraction >= min_cluster_fraction,
               not covered by antenna flag).
    Level 2b— per-scan per-baseline: absolute count >= baseline_max_bad_samples.
    Level 3 — RFI burst (many baselines simultaneously bad in a time bin).
    Time-range — remaining sub-threshold bad intervals compressed to UTC windows.
               Merge gap = max(max_gap_samples * integration_time, max_gap_minutes),
               mirroring Fortran tol_sec sample-count-based gap tolerance.

    Returns
    -------
    dict with keys:
        'bad_scan_timeranges'    : list of [iso_start, iso_end]
                                   scans flagged wholesale (Tier-0)
        'bad_antennas'           : list of antenna names flagged wholly
                                   (bad fraction >= min_cluster_fraction,
                                    spread across >= min_distinct_baselines_for_ant)
        'bad_baselines'          : list of [ant1_name, ant2_name] pairs
                                   (bad fraction >= min_cluster_fraction,
                                    not already covered by an antenna flag)
        'bad_antenna_timeranges' : dict  ant_name -> list of [iso_start, iso_end]
                                   per-scan + fine-grained time-range flags
        'bad_baseline_timeranges': dict 'ant1::ant2' -> list of [iso_start, iso_end]
                                   per-scan + fine-grained time-range flags
        'bad_burst_timeranges'   : list of [iso_start, iso_end]
                                   global RFI bursts (many baselines simultaneously bad)
    """
    ant1 = vis['ant1']
    ant2 = vis['ant2']
    jd   = vis['jd']
    nrows = len(jd)

    all_ant_ids = sorted(set(ant1.tolist()) | set(ant2.tolist()))
    name_of     = {a: antenna_name_map.get(a, str(a)) for a in all_ant_ids}

    # Integration time: median positive JD gap between consecutive rows.
    # Used for sample-count-aware gap tolerance (mirrors Fortran tol_sec).
    _sorted_jd = np.sort(jd)
    _pos_diffs = np.diff(_sorted_jd)
    _pos_diffs = _pos_diffs[_pos_diffs > 0]
    dt_int_jd  = float(np.median(_pos_diffs)) if len(_pos_diffs) >= 2 else 8.0 / 86400.0
    # Effective merge gap: larger of sample-count-based and calendar-based floor.
    effective_gap_minutes = max(max_gap_samples * dt_int_jd * 24.0 * 60.0, max_gap_minutes)

    # Scan list computed once; reused by all per-scan loops below.
    scans = detect_scans(jd, gap_minutes=scan_gap_minutes)

    # ------------------------------------------------------------------
    # Tier-0 — whole-scan bad detection
    # (Fortran: ibad_scan >= 0.7 × nbase × nsamp_per_scan)
    # ------------------------------------------------------------------
    # A scan is flagged entirely when >= whole_scan_bad_fraction of its rows
    # are bad.  Any such scan is added to bad_scan_timeranges and recorded in
    # bad_scan_indices so all lower tiers can skip it.
    bad_scan_timeranges: List[Tuple[str, str]] = []
    bad_scan_indices: set = set()   # scan loop indices to skip in lower tiers

    for _t0_idx, (_t0_start, _t0_end) in enumerate(scans):
        _t0_n = _t0_end - _t0_start
        if _t0_n == 0:
            continue
        _t0_nbad = int(bad_mask[_t0_start:_t0_end].sum())
        if _t0_nbad / _t0_n >= whole_scan_bad_fraction:
            bad_scan_timeranges.append((
                jd_to_iso(float(jd[_t0_start])),
                jd_to_iso(float(jd[_t0_end - 1])),
            ))
            bad_scan_indices.add(_t0_idx)

    # ------------------------------------------------------------------
    # Level 1 — per-antenna bad fraction
    # ------------------------------------------------------------------
    bad_antennas: List[str] = []
    ant_explained: set = set()   # antenna ids fully covered by antenna flags

    for a in all_ant_ids:
        rows_for_a = (ant1 == a) | (ant2 == a)
        total = int(rows_for_a.sum())
        if total == 0:
            continue
        bad   = int((bad_mask & rows_for_a).sum())
        frac  = bad / total

        if frac < min_cluster_fraction:
            continue

        # Check spread: distinct baselines that are bad
        bad_rows_a = np.where(bad_mask & rows_for_a)[0]
        partners = set()
        for r in bad_rows_a:
            other = int(ant2[r]) if int(ant1[r]) == a else int(ant1[r])
            if other != a:
                partners.add(other)

        if len(partners) >= min_distinct_baselines_for_ant:
            bad_antennas.append(name_of[a])
            ant_explained.add(a)

    # ------------------------------------------------------------------
    # Level 2 — per-baseline bad fraction (not explained by antenna flags)
    # ------------------------------------------------------------------
    bad_baselines: List[Tuple[str, str]] = []
    baseline_explained: set = set()  # frozenset pairs covered by baseline flags

    all_pairs = set(
        (min(int(a), int(b)), max(int(a), int(b)))
        for a, b in zip(ant1.tolist(), ant2.tolist())
    )

    for (a, b) in sorted(all_pairs):
        # Skip if at least one antenna already flagged wholly
        if a in ant_explained or b in ant_explained:
            continue
        rows_for_bl = ((ant1 == a) & (ant2 == b)) | ((ant1 == b) & (ant2 == a))
        total = int(rows_for_bl.sum())
        if total == 0:
            continue
        bad   = int((bad_mask & rows_for_bl).sum())
        frac  = bad / total
        if frac >= min_cluster_fraction:
            bad_baselines.append((name_of[a], name_of[b]))
            baseline_explained.add(frozenset([a, b]))

    # ------------------------------------------------------------------
    # Level 3 — per-scan RFI burst detection
    # ------------------------------------------------------------------
    n_all_baselines = len(all_pairs)
    burst_intervals_jd: List[Tuple[float, float]] = []

    for _burst_idx, (scan_start, scan_end) in enumerate(scans):
        if _burst_idx in bad_scan_indices:
            continue   # Tier-0 already covers this scan
        scan_jd   = jd[scan_start:scan_end]
        scan_bad  = bad_mask[scan_start:scan_end]
        scan_a1   = ant1[scan_start:scan_end]
        scan_a2   = ant2[scan_start:scan_end]
        if not np.any(scan_bad):
            continue
        # Bin into 10-second windows
        dt_bin_jd = 10.0 / 86400.0
        if len(scan_jd) == 0:
            continue
        bins = np.arange(scan_jd[0], scan_jd[-1] + dt_bin_jd, dt_bin_jd)
        for i in range(len(bins) - 1):
            t0, t1  = bins[i], bins[i + 1]
            in_bin  = (scan_jd >= t0) & (scan_jd < t1)
            bad_bin = scan_bad & in_bin
            if not np.any(in_bin):
                continue
            # Count distinct baselines bad in this bin
            bad_pairs = set(
                (min(int(a), int(b)), max(int(a), int(b)))
                for a, b in zip(scan_a1[bad_bin].tolist(), scan_a2[bad_bin].tolist())
            )
            n_active = len(set(
                (min(int(a), int(b)), max(int(a), int(b)))
                for a, b in zip(scan_a1[in_bin].tolist(), scan_a2[in_bin].tolist())
            ))
            if n_active == 0:
                continue
            frac = len(bad_pairs) / n_active
            if frac >= min_burst_baseline_fraction:
                burst_intervals_jd.append((t0, t1))

    burst_intervals_jd = _merge_jd_intervals(burst_intervals_jd, effective_gap_minutes)

    # ------------------------------------------------------------------
    # Per-scan antenna detection — Fortran methods 1 + 2
    # Per-scan baseline detection — absolute-count backstop
    # ------------------------------------------------------------------
    # These supplement the global wholesale flags.  Antennas/baselines that
    # do not qualify for a global wholesale flag may still be wholly bad within
    # a single scan; those appear as scan-spanning entries in the time-range
    # dicts below.  Tier-0 scans (bad_scan_indices) are skipped here.
    _per_scan_ant_ivs: Dict[int, List[Tuple[float, float]]] = {}      # ant_id → JD intervals
    _per_scan_bl_ivs:  Dict[Tuple[int, int], List[Tuple[float, float]]] = {}  # (a,b) → JD intervals

    for _ps_idx, (_ps_start, _ps_end) in enumerate(scans):
        if _ps_idx in bad_scan_indices:
            continue
        _ps_n = _ps_end - _ps_start
        if _ps_n == 0:
            continue
        _ps_a1  = ant1[_ps_start:_ps_end]
        _ps_a2  = ant2[_ps_start:_ps_end]
        _ps_bad = bad_mask[_ps_start:_ps_end]
        _ps_jd  = jd[_ps_start:_ps_end]
        _t0_sc  = float(_ps_jd[0])
        _t1_sc  = float(_ps_jd[-1])
        _scan_pairs_ps = set(
            (min(int(_a), int(_b)), max(int(_a), int(_b)))
            for _a, _b in zip(_ps_a1.tolist(), _ps_a2.tolist())
        )

        # Method 1 + 2 per-scan antenna check
        for a in all_ant_ids:
            if name_of[a] in bad_antennas:
                continue   # already globally wholesale
            _rows_a  = (_ps_a1 == a) | (_ps_a2 == a)
            _total_a = int(_rows_a.sum())
            if _total_a == 0:
                continue
            _bad_a   = int((_ps_bad & _rows_a).sum())
            _flagged = (_bad_a / _total_a) >= per_scan_ant_fraction  # method 1

            # Method 2 (requires baseline_max_bad_samples):
            # flag if count > threshold on >= per_scan_ant_bl_fraction of baselines.
            if not _flagged and baseline_max_bad_samples is not None:
                _partner_ids = sorted(set(
                    int(_ps_a2[r]) if int(_ps_a1[r]) == a else int(_ps_a1[r])
                    for r in np.where(_rows_a)[0]
                ))
                _n_partners = len(_partner_ids)
                if _n_partners > 0:
                    _n_over = sum(
                        1 for p in _partner_ids
                        if int((_ps_bad & (
                            ((_ps_a1 == a) & (_ps_a2 == p)) |
                            ((_ps_a1 == p) & (_ps_a2 == a))
                        )).sum()) > baseline_max_bad_samples
                    )
                    if _n_over / _n_partners >= per_scan_ant_bl_fraction:
                        _flagged = True

            if _flagged:
                _per_scan_ant_ivs.setdefault(a, []).append((_t0_sc, _t1_sc))

        # Per-baseline absolute-count backstop (per scan)
        if baseline_max_bad_samples is not None:
            for (a, b) in _scan_pairs_ps:
                if a in ant_explained or b in ant_explained:
                    continue
                if frozenset([a, b]) in baseline_explained:
                    continue
                _rows_bl = ((_ps_a1 == a) & (_ps_a2 == b)) | ((_ps_a1 == b) & (_ps_a2 == a))
                if int((_ps_bad & _rows_bl).sum()) >= baseline_max_bad_samples:
                    _per_scan_bl_ivs.setdefault((a, b), []).append((_t0_sc, _t1_sc))

    # ------------------------------------------------------------------
    # Per-antenna time-ranges (for antennas NOT wholesale-flagged)
    # ------------------------------------------------------------------
    # Wholesale-flagged antennas need no time restriction.
    # All others: merge fine-grained per-row intervals with per-scan intervals.
    bad_antenna_timeranges: Dict[str, List[Tuple[str, str]]] = {}
    for a in all_ant_ids:
        if name_of[a] in bad_antennas:
            continue   # wholesale flag already covers this antenna
        rows_for_a = np.where((ant1 == a) | (ant2 == a))[0]
        bad_rows_a = rows_for_a[bad_mask[rows_for_a]]
        # Combine fine-grained per-row intervals with coarse per-scan intervals.
        _all_ant_ivs: List[Tuple[float, float]] = []
        if len(bad_rows_a) > 0:
            _all_ant_ivs.extend(_rows_to_intervals(jd, bad_rows_a, effective_gap_minutes, jd))
        if a in _per_scan_ant_ivs:
            _all_ant_ivs.extend(_per_scan_ant_ivs[a])
        if not _all_ant_ivs:
            continue
        _merged_ant = _merge_jd_intervals(_all_ant_ivs, effective_gap_minutes)
        bad_antenna_timeranges[name_of[a]] = [
            (jd_to_iso(t0), jd_to_iso(t1)) for t0, t1 in _merged_ant
        ]

    # ------------------------------------------------------------------
    # Per-baseline time-ranges (for baselines NOT wholesale-flagged)
    # ------------------------------------------------------------------
    # Same logic: wholesale-flagged baselines need no time restriction.
    # Baselines with some bad rows but below the threshold get time-range flags.
    bad_baseline_timeranges: Dict[str, List[Tuple[str, str]]] = {}
    bad_baselines_set = {frozenset(pair) for pair in bad_baselines}
    for (a, b) in sorted(all_pairs):
        # Skip if covered by a wholesale baseline flag or a wholesale antenna flag
        if frozenset([name_of.get(a, str(a)), name_of.get(b, str(b))]) in bad_baselines_set:
            continue
        if a in ant_explained or b in ant_explained:
            continue
        rows_for_bl = np.where(
            ((ant1 == a) & (ant2 == b)) | ((ant1 == b) & (ant2 == a))
        )[0]
        bad_rows_bl = rows_for_bl[bad_mask[rows_for_bl]]
        # Combine fine-grained per-row intervals with per-scan absolute-count intervals.
        _all_bl_ivs: List[Tuple[float, float]] = []
        if len(bad_rows_bl) > 0:
            _all_bl_ivs.extend(_rows_to_intervals(jd, bad_rows_bl, effective_gap_minutes, jd))
        _bl_key_int = (min(a, b), max(a, b))
        if _bl_key_int in _per_scan_bl_ivs:
            _all_bl_ivs.extend(_per_scan_bl_ivs[_bl_key_int])
        if not _all_bl_ivs:
            continue
        _merged_bl = _merge_jd_intervals(_all_bl_ivs, effective_gap_minutes)
        key = f'{name_of[a]}::{name_of[b]}'
        bad_baseline_timeranges[key] = [
            (jd_to_iso(t0), jd_to_iso(t1)) for t0, t1 in _merged_bl
        ]

    return {
        'bad_scan_timeranges':    bad_scan_timeranges,
        'bad_antennas':           bad_antennas,
        'bad_baselines':          [[a, b] for a, b in bad_baselines],
        'bad_antenna_timeranges': bad_antenna_timeranges,
        'bad_baseline_timeranges': bad_baseline_timeranges,
        'bad_burst_timeranges':   [(jd_to_iso(t0), jd_to_iso(t1))
                                   for t0, t1 in burst_intervals_jd],
    }


def _rows_to_intervals(
    jd_all: np.ndarray,
    bad_row_indices: np.ndarray,
    max_gap_minutes: float,
    source_jd: np.ndarray,
) -> List[Tuple[float, float]]:
    """Convert a set of bad row indices to merged JD intervals.

    Each bad timestamp is extended by half the typical integration time on each
    side to build a proper interval, then adjacent intervals within
    max_gap_minutes are merged.
    """
    if len(bad_row_indices) == 0:
        return []
    bad_jd = np.sort(jd_all[bad_row_indices])
    # Estimate integration time (median gap between consecutive timestamps)
    if len(source_jd) >= 2:
        dt_jd = float(np.median(np.diff(np.sort(source_jd)))) * 0.5
    else:
        dt_jd = 5.0 / 86400.0   # 5-second fallback
    raw_intervals = [(t - dt_jd, t + dt_jd) for t in bad_jd.tolist()]
    return _merge_jd_intervals(raw_intervals, max_gap_minutes)


def _merge_jd_intervals(
    intervals: List[Tuple[float, float]],
    max_gap_minutes: float,
) -> List[Tuple[float, float]]:
    """Merge overlapping or close JD intervals.

    Pairs separated by less than *max_gap_minutes* are merged into one; the gap
    between them is filled even if no bad samples fall within it.
    """
    if not intervals:
        return []
    gap_jd = max_gap_minutes / (24.0 * 60.0)
    sorted_iv = sorted(intervals, key=lambda x: x[0])
    merged: List[Tuple[float, float]] = [sorted_iv[0]]
    for t0, t1 in sorted_iv[1:]:
        last_t0, last_t1 = merged[-1]
        if t0 <= last_t1 + gap_jd:
            merged[-1] = (last_t0, max(last_t1, t1))
        else:
            merged.append((t0, t1))
    return merged


# ---------------------------------------------------------------------------
# Scan flag matrices
# ---------------------------------------------------------------------------

def build_scan_flag_matrices(
    vis: dict,
    bad_mask: np.ndarray,
    antenna_name_map: Dict[int, str],
    scan_gap_minutes: float,
) -> Tuple[Dict[int, dict], dict]:
    """Build per-scan and summary antenna×antenna bad-fraction matrices.

    Returns
    -------
    scan_matrices : dict
        Keys are scan indices (0-based).  Each value is a dict:
            'jd_start', 'jd_end', 'iso_start', 'iso_end',
            'matrix' : ndarray (nants, nants) — fraction of rows flagged,
            'ant_ids', 'ant_names'
    summary_matrix : dict
        'matrix', 'ant_ids', 'ant_names' — aggregated across all scans.
    """
    jd   = vis['jd']
    ant1 = vis['ant1']
    ant2 = vis['ant2']

    all_ant_ids = sorted(set(ant1.tolist()) | set(ant2.tolist()))
    ant_idx     = {a: i for i, a in enumerate(all_ant_ids)}
    ant_names   = [antenna_name_map.get(a, str(a)) for a in all_ant_ids]
    nants       = len(all_ant_ids)

    scans = detect_scans(jd, gap_minutes=scan_gap_minutes)
    scan_matrices: Dict[int, dict] = {}
    total_bad    = np.zeros((nants, nants), dtype=np.float64)
    total_counts = np.zeros((nants, nants), dtype=np.float64)

    for scan_i, (s_start, s_end) in enumerate(scans):
        s_jd  = jd[s_start:s_end]
        s_a1  = ant1[s_start:s_end]
        s_a2  = ant2[s_start:s_end]
        s_bad = bad_mask[s_start:s_end]

        mat_bad   = np.zeros((nants, nants), dtype=np.float64)
        mat_count = np.zeros((nants, nants), dtype=np.float64)

        for r in range(len(s_jd)):
            i = ant_idx.get(int(s_a1[r]), -1)
            j = ant_idx.get(int(s_a2[r]), -1)
            if i < 0 or j < 0:
                continue
            mat_count[i, j] += 1
            mat_count[j, i] += 1
            total_counts[i, j] += 1
            total_counts[j, i] += 1
            if s_bad[r]:
                mat_bad[i, j] += 1
                mat_bad[j, i] += 1
                total_bad[i, j] += 1
                total_bad[j, i] += 1

        with np.errstate(invalid='ignore', divide='ignore'):
            frac = np.where(mat_count > 0, mat_bad / mat_count, np.nan)

        scan_matrices[scan_i] = {
            'jd_start':  float(s_jd[0])  if len(s_jd) else float('nan'),
            'jd_end':    float(s_jd[-1]) if len(s_jd) else float('nan'),
            'iso_start': jd_to_iso(float(s_jd[0]))  if len(s_jd) else '',
            'iso_end':   jd_to_iso(float(s_jd[-1])) if len(s_jd) else '',
            'n_rows':    int(s_end - s_start),
            'matrix':    frac,
            'ant_ids':   all_ant_ids,
            'ant_names': ant_names,
        }

    with np.errstate(invalid='ignore', divide='ignore'):
        summary_frac = np.where(total_counts > 0, total_bad / total_counts, np.nan)

    summary_matrix = {
        'matrix':    summary_frac,
        'ant_ids':   all_ant_ids,
        'ant_names': ant_names,
    }

    return scan_matrices, summary_matrix


def plot_scan_flag_matrix(
    matrix_dict: dict,
    title: str = '',
    cmap: str = 'YlOrRd',
    figsize: Tuple[float, float] = (12, 10),
) -> object:
    """Plot a triangular antenna×antenna flag-fraction heatmap.

    Parameters
    ----------
    matrix_dict : dict
        One entry from the *scan_matrices* dict, or the *summary_matrix* dict.
    title : str
        Plot title.
    cmap : str
        Matplotlib colormap name.
    figsize : (w, h) inches.

    Returns
    -------
    matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    mat = matrix_dict['matrix'].copy()
    ant_names = matrix_dict['ant_names']
    nants = len(ant_names)

    # Show only upper triangle; mask lower triangle
    lower_mask = np.tril(np.ones((nants, nants), dtype=bool), k=-1)
    mat[lower_mask] = np.nan

    fig, ax = plt.subplots(figsize=figsize)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color='#e0e0e0')

    im = ax.imshow(mat, aspect='auto', cmap=cmap_obj,
                   vmin=0.0, vmax=1.0, interpolation='nearest')

    ax.set_xticks(range(nants))
    ax.set_yticks(range(nants))
    ax.set_xticklabels(ant_names, rotation=90, fontsize=7)
    ax.set_yticklabels(ant_names, fontsize=7)

    # Annotate cells with percentage
    for i in range(nants):
        for j in range(i, nants):
            v = mat[i, j]
            if np.isfinite(v):
                pct = int(round(v * 100))
                color = 'white' if v > 0.5 else 'black'
                ax.text(j, i, f'{pct}%', ha='center', va='center',
                        fontsize=5, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Fraction of rows flagged', rotation=270, labelpad=14)

    if 'iso_start' in matrix_dict:
        subtitle = f"  {matrix_dict['iso_start']}  →  {matrix_dict['iso_end']}"
    else:
        subtitle = ''
    ax.set_title(f'{title}{subtitle}', fontsize=11)
    ax.set_xlabel('Antenna')
    ax.set_ylabel('Antenna')
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def build_output_flag_table(
    proposal: dict,
    source: str,
    notes: str = '',
    obs_jd_start: Optional[float] = None,
    obs_jd_end: Optional[float] = None,
    elevation_min_deg: Optional[float] = None,
) -> dict:
    """Build a JSON-serialisable flag table dict from a clustering proposal.

    Parameters
    ----------
    obs_jd_start, obs_jd_end : float, optional
        JD boundaries of the rows that were actually analysed (after any
        elevation cut).  When supplied, CASA wholesale flags are bounded to
        this window so they only fire during the examined period.
    elevation_min_deg : float, optional
        Elevation cut applied when loading visibilities.  Stored in the JSON
        for provenance and annotated in the CASA flag file header.
    """
    obs_tr = (
        [jd_to_iso(obs_jd_start), jd_to_iso(obs_jd_end)]
        if obs_jd_start is not None else None
    )
    return {
        'kind':    'ugmrt_flag_table',
        'version': 2,
        'source':  source,
        'notes':   notes,
        # Provenance — data-selection context so CASA commands can be bounded
        # to the exact time window that was actually analysed:
        'obs_timerange':     obs_tr,
        'elevation_min_deg': elevation_min_deg,
        # Whole-scan flags (calibrator-specific; not transferred to target):
        'bad_scan_timeranges':    proposal.get('bad_scan_timeranges', []),
        # Time-agnostic flags (apply to all sources):
        'bad_antennas':  proposal['bad_antennas'],
        'bad_baselines': proposal['bad_baselines'],
        # Time-specific flags (apply only within these windows):
        'bad_antenna_timeranges':  proposal['bad_antenna_timeranges'],
        'bad_baseline_timeranges': proposal['bad_baseline_timeranges'],
        'bad_burst_timeranges':    proposal['bad_burst_timeranges'],
    }


def build_target_flag_table(
    cal_proposal: dict,
    target_source: str,
    transfer_timeranges: bool = True,
    notes: str = '',
) -> dict:
    """Build a flag table for a target source by transferring calibrator flags.

    Parameters
    ----------
    cal_proposal : dict
        Output of :func:`build_output_flag_table` for the calibrator.
    target_source : str
        Name of the target source.
    transfer_timeranges : bool
        If True, time-range flags are also copied.  If False, only
        antenna/baseline (time-agnostic) flags are transferred.
    """
    out = {
        'kind':    'ugmrt_flag_table',
        'version': 2,
        'source':  target_source,
        'notes':   notes,
        # Whole-scan flags are calibrator-scan-specific; never transferred.
        'bad_scan_timeranges':    [],
        'bad_antennas':  list(cal_proposal['bad_antennas']),
        'bad_baselines': list(cal_proposal['bad_baselines']),
        'bad_antenna_timeranges':  {},
        'bad_baseline_timeranges': {},
        'bad_burst_timeranges':    [],
    }
    if transfer_timeranges:
        out['bad_antenna_timeranges']  = dict(cal_proposal['bad_antenna_timeranges'])
        out['bad_baseline_timeranges'] = dict(cal_proposal['bad_baseline_timeranges'])
        out['bad_burst_timeranges']    = list(cal_proposal['bad_burst_timeranges'])
    return out


def format_casa_commands(flag_table: dict, reason_prefix: str = 'outlier_det') -> List[str]:
    """Return a list of CASA ``flagdata`` command strings.

    The commands use ``mode='manual'`` and are suitable for pasting into a
    CASA script or passing to ``flagdata(vis=..., mode='list', inpfile=...)``.

    Wholesale antenna/baseline flags (no time qualifier) are bounded to
    ``obs_timerange`` when the flag table carries that field.  This ensures
    flags only fire during the window that was actually examined, respecting
    any elevation cut applied during data loading.
    """
    obs_tr = flag_table.get('obs_timerange')   # [iso_start, iso_end] or None
    el_min = flag_table.get('elevation_min_deg')

    cmds: List[str] = [
        f"# Flag table for source: {flag_table.get('source', '')}",
        f"# Generated by outlier_detection.py",
        f"# Notes: {flag_table.get('notes', '')}",
    ]
    if obs_tr:
        el_str = f'{el_min}°' if el_min is not None else 'unknown'
        cmds += [
            f"# Observation window analysed (elevation >= {el_str}):",
            f"#   {obs_tr[0]}  \u2192  {obs_tr[1]}",
            "# Wholesale antenna/baseline flags below are bounded to this window.",
            "# Data outside it was never examined, so no detections were made there.",
        ]
    cmds.append('')

    # CASA timerange string to constrain wholesale flags (empty string = no constraint)
    if obs_tr:
        obs_casa_t0 = jd_to_casa(_iso_to_jd(obs_tr[0]))
        obs_casa_t1 = jd_to_casa(_iso_to_jd(obs_tr[1]))
        obs_tr_str  = f" timerange='{obs_casa_t0}~{obs_casa_t1}'"
    else:
        obs_tr_str = ''

    for ant in flag_table.get('bad_antennas', []):
        cmds.append(
            f"mode='manual' antenna='{ant}'"
            f"{obs_tr_str}"
            f" reason='{reason_prefix}:antenna'"
        )

    for bl in flag_table.get('bad_baselines', []):
        a1, a2 = bl[0], bl[1]
        cmds.append(
            f"mode='manual' antenna='{a1}&&{a2}'"
            f"{obs_tr_str}"
            f" reason='{reason_prefix}:baseline'"
        )

    def _spw_str(chanrange):
        """CASA spw selector for an absolute chanrange [c0, c1], or empty str."""
        if chanrange is None:
            return ''
        return f" spw='0:{chanrange[0]}~{chanrange[1]}'"

    def _iter_tr_entries(entries):
        """Yield (intervals, chanrange) for v1 or v2 timerange entry lists."""
        for entry in entries:
            if isinstance(entry, dict):
                yield entry['intervals'], entry.get('chanrange')
            else:
                yield [list(entry)], None   # v1: entry is [t0, t1]

    for ant, entries in flag_table.get('bad_antenna_timeranges', {}).items():
        for intervals, chanrange in _iter_tr_entries(entries):
            spw = _spw_str(chanrange)
            for (t0, t1) in intervals:
                casa_t0 = jd_to_casa(_iso_to_jd(t0))
                casa_t1 = jd_to_casa(_iso_to_jd(t1))
                cmds.append(
                    f"mode='manual' antenna='{ant}' "
                    f"timerange='{casa_t0}~{casa_t1}'{spw} "
                    f"reason='{reason_prefix}:ant_timerange'"
                )

    for bl_key, entries in flag_table.get('bad_baseline_timeranges', {}).items():
        a1, a2 = bl_key.split('::', 1)
        for intervals, chanrange in _iter_tr_entries(entries):
            spw = _spw_str(chanrange)
            for (t0, t1) in intervals:
                casa_t0 = jd_to_casa(_iso_to_jd(t0))
                casa_t1 = jd_to_casa(_iso_to_jd(t1))
                cmds.append(
                    f"mode='manual' antenna='{a1}&&{a2}' "
                    f"timerange='{casa_t0}~{casa_t1}'{spw} "
                    f"reason='{reason_prefix}:bl_timerange'"
                )

    for (t0, t1) in flag_table.get('bad_scan_timeranges', []):
        casa_t0 = jd_to_casa(_iso_to_jd(t0))
        casa_t1 = jd_to_casa(_iso_to_jd(t1))
        cmds.append(
            f"mode='manual' "
            f"timerange='{casa_t0}~{casa_t1}' "
            f"reason='{reason_prefix}:whole_scan'"
        )

    for (t0, t1) in flag_table.get('bad_burst_timeranges', []):
        casa_t0 = jd_to_casa(_iso_to_jd(t0))
        casa_t1 = jd_to_casa(_iso_to_jd(t1))
        cmds.append(
            f"mode='manual' "
            f"timerange='{casa_t0}~{casa_t1}' "
            f"reason='{reason_prefix}:burst'"
        )

    return cmds


def _iso_to_jd(iso_str: str) -> float:
    """Convert ISO UTC string to Julian Date."""
    try:
        from astropy.time import Time
        return float(Time(iso_str, format='iso', scale='utc').jd)
    except ImportError:
        import datetime
        t_str = iso_str.split('.')[0]
        t = datetime.datetime.strptime(t_str, '%Y-%m-%d %H:%M:%S')
        j2000 = datetime.datetime(2000, 1, 1, 12, 0, 0)
        days_since_j2000 = (t - j2000).total_seconds() / 86400.0
        return 2451545.0 + days_since_j2000


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(proposal: dict, label: str = 'Calibrator') -> None:
    """Print a human-readable summary of the outlier-detection result."""
    width = 70
    print()
    print('─' * width)
    print(f'  Outlier Detection Summary — {label}')
    print('─' * width)

    bad_ants = proposal['bad_antennas']
    bad_bls  = proposal['bad_baselines']
    print(f'  Bad antennas  ({len(bad_ants):>3d}): {", ".join(bad_ants) or "none"}')
    print(f'  Bad baselines ({len(bad_bls):>3d}): '
          + (', '.join(f'{a}–{b}' for a, b in bad_bls) or 'none'))

    def _count_intervals(entries):
        """Count total time-range intervals across v1 or v2 entries."""
        n = 0
        for entry in entries:
            if isinstance(entry, dict):
                n += len(entry['intervals'])
            else:
                n += 1
        return n

    n_scans   = len(proposal.get('bad_scan_timeranges', []))
    n_ant_tr  = sum(_count_intervals(v) for v in proposal['bad_antenna_timeranges'].values())
    n_bl_tr   = sum(_count_intervals(v) for v in proposal['bad_baseline_timeranges'].values())
    n_burst   = len(proposal['bad_burst_timeranges'])
    print(f'  Whole-scan flags         : {n_scans}')
    print(f'  Antenna time-range flags : {n_ant_tr}')
    print(f'  Baseline time-range flags: {n_bl_tr}')
    print(f'  Burst time-range flags   : {n_burst}')

    def _print_tr_entries(label, entries):
        for entry in entries:
            if isinstance(entry, dict):
                ch = f"  chan={entry['chanrange']}" if entry.get('chanrange') else ''
                for t0, t1 in entry['intervals']:
                    print(f'    {label}{ch}  {t0} – {t1}')
            else:
                t0, t1 = entry
                print(f'    {label}  {t0} – {t1}')

    for ant, entries in proposal['bad_antenna_timeranges'].items():
        _print_tr_entries(f'ant {ant}', entries)
    for bl_key, entries in proposal['bad_baseline_timeranges'].items():
        _print_tr_entries(f'bl  {bl_key}', entries)
    for t0, t1 in proposal.get('bad_scan_timeranges', []):
        print(f'    scan  {t0} – {t1}')
    for t0, t1 in proposal['bad_burst_timeranges']:
        print(f'    burst  {t0} – {t1}')
    print('─' * width)


# ---------------------------------------------------------------------------
# Inspection / survey mode
# ---------------------------------------------------------------------------

def _inspect_mode(
    row_index: dict,
    source: str,
    vis_jd: np.ndarray,
    vis_a1: np.ndarray,
    vis_a2: np.ndarray,
    fits_path: Path,
    chan_range: Tuple[int, int],
    scan_gap_minutes: float,
) -> None:
    """Print observation structure for *source* and suggest CONFIG parameters.

    Derives everything the old Fortran parfile required manually:
      - number of scans and their UTC extents
      - integration time (sampling time)
      - active antenna / baseline count
      - suggested threshold values for all clustering parameters

    No flagging is performed.  Exits after printing.
    """
    width = 72
    bar   = '\u2550' * width

    # ── integration time from median JD gap ────────────────────────────
    sorted_jd = np.sort(vis_jd)
    pos_diffs = np.diff(sorted_jd)
    pos_diffs = pos_diffs[pos_diffs > 0]
    dt_int_jd = float(np.median(pos_diffs)) if len(pos_diffs) >= 2 else 8.0 / 86400.0
    dt_int_s  = dt_int_jd * 86400.0

    # ── scans ──────────────────────────────────────────────────────────
    scans = detect_scans(vis_jd, gap_minutes=scan_gap_minutes)
    scan_lengths_s = []
    scan_rows      = []
    for s, e in scans:
        sc_jd = vis_jd[s:e]
        if len(sc_jd) >= 2:
            scan_lengths_s.append((float(sc_jd[-1]) - float(sc_jd[0])) * 86400.0)
        else:
            scan_lengths_s.append(0.0)
        scan_rows.append(e - s)
    median_scan_s   = float(np.median(scan_lengths_s)) if scan_lengths_s else 0.0
    median_scan_rows = int(np.median(scan_rows))        if scan_rows       else 0

    # ── antennas / baselines ───────────────────────────────────────────
    active_ants = sorted(set(vis_a1.tolist()) | set(vis_a2.tolist()))
    n_ant  = len(active_ants)
    n_base = n_ant * (n_ant - 1) // 2

    # ── parameter suggestions ─────────────────────────────────────────
    # BASELINE_MAX_BAD_SAMPLES: Fortran used maxbad_allowed=5 at tsamp=2s
    # (≈10 s of bad data).  Scale to current integration time.
    sug_max_bad = max(3, round(10.0 / dt_int_s))

    # MAX_GAP_SAMPLES: Fortran tol_sec=60 s.  Scale to integration periods.
    sug_gap_samp = max(2, round(60.0 / dt_int_s))

    # BASELINE_MAX_BAD_SAMPLES for scan-level check:
    # rows per scan per baseline ≈ median_scan_rows / n_base; flag if >10% of those
    rows_per_bl = max(1, median_scan_rows // max(n_base, 1))
    sug_max_bad_scan = max(sug_max_bad, round(0.10 * rows_per_bl))

    # SCAN_GAP_MINUTES: half the shortest inter-scan gap, floored at 1 min
    inter_gaps: List[float] = []
    if len(scans) >= 2:
        for i in range(len(scans) - 1):
            gap_s = (vis_jd[scans[i+1][0]] - vis_jd[scans[i][1] - 1]) * 86400.0
            if gap_s > 0:
                inter_gaps.append(gap_s)
    sug_scan_gap = max(1.0, min(inter_gaps) / 2.0 / 60.0) if inter_gaps else scan_gap_minutes

    # MAX_GAP_MINUTES: half the median scan length, floored at 1 min
    sug_gap_min = max(1.0, median_scan_s / 2.0 / 60.0)

    # ── print report ───────────────────────────────────────────────────────
    print()
    print(bar)
    print(f'  INSPECTION REPORT — {source}')
    print(f'  File : {fits_path}')
    print(f'  Chans: {chan_range[0]}–{chan_range[1]}  (used only for row-count estimates)')
    print(bar)
    print(f'  Active antennas  : {n_ant}')
    print(f'  Baselines        : {n_base}')
    print(f'  Total rows loaded: {len(vis_jd):,}')
    print(f'  Integration time : {dt_int_s:.2f} s  (median JD gap)')
    print(f'  Scans detected   : {len(scans)}  (SCAN_GAP_MINUTES = {scan_gap_minutes})')
    print()

    # scan table
    hdr = f"  {'Scan':>4}  {'Start (UTC)':^19}  {'End (UTC)':^19}  {'Rows':>6}  {'Len (min)':>9}"
    print(hdr)
    print('  ' + '─' * (len(hdr) - 2))
    for i, (s, e) in enumerate(scans):
        iso0 = jd_to_iso(float(vis_jd[s]))
        iso1 = jd_to_iso(float(vis_jd[e - 1]))
        nrows_sc = e - s
        len_min  = scan_lengths_s[i] / 60.0
        print(f'  {i+1:>4}  {iso0:^19}  {iso1:^19}  {nrows_sc:>6,}  {len_min:>9.2f}')
    print()
    print(f'  Median scan length : {median_scan_s:.1f} s  ({median_scan_s/60:.2f} min)')
    print(f'  Median rows / scan : {median_scan_rows:,}')
    print()

    # suggested parameters
    print('  ── Suggested CONFIG parameters ─────────────────────────────────────')
    print(f'  # Integration time {dt_int_s:.1f} s detected automatically at runtime')
    print(f'  # — no parfile entry needed (Fortran required manual tsamp_sec entry)')
    print()
    print(f'  SCAN_GAP_MINUTES         = {sug_scan_gap:.1f}')
    print(f'  # Half the shortest inter-scan gap observed ({min(inter_gaps, default=0)/60:.1f} min gap).')
    print()
    print(f'  MAX_GAP_SAMPLES          = {sug_gap_samp}')
    print(f'  # ≈ 60 s tolerance ({sug_gap_samp} × {dt_int_s:.1f} s = {sug_gap_samp*dt_int_s:.0f} s).')
    print(f'  # Fortran tol_sec = 60 s; scales with your actual integration time.')
    print()
    print(f'  MAX_GAP_MINUTES          = {sug_gap_min:.1f}')
    print(f'  # Half the median scan length — prevents bridging across separate events.')
    print()
    print(f'  BASELINE_MAX_BAD_SAMPLES = {sug_max_bad}')
    print(f'  # ≈10 s of bad data ({sug_max_bad} × {dt_int_s:.1f} s) per baseline per scan.')
    print(f'  # Fortran maxbad_allowed = 5 at tsamp = 2 s (≈10 s).')
    print()
    print(f'  WHOLE_SCAN_BAD_FRACTION  = 0.70   # Fortran: 0.70 (tested; leave as is)')
    print(f'  PER_SCAN_ANT_FRACTION    = 0.60   # Fortran: 0.60 (tested; leave as is)')
    print(f'  PER_SCAN_ANT_BL_FRACTION = 0.50   # Fortran: 0.50 (tested; leave as is)')
    print(f'  MIN_CLUSTER_FRACTION     = 0.80   # Global wholesale: bad ≥80% of obs')
    print(f'  MIN_BURST_BASELINE_FRACTION = 0.50')
    print()
    print(f'  CHAN_RANGE = ({chan_range[0]}, {chan_range[1]})   # adjust as needed')
    print()
    print('  ── Run command ────────────────────────────────────────────────────────')
    print(f'  python3 outlier_detection.py --cal {source} --chans {chan_range[0]}-{chan_range[1]} --no-dry-run')
    print(bar)


# ---------------------------------------------------------------------------
# Library API — in-memory clustering detection (importable from notebook /
# preprocess_ugmrt.py without running main())
# ---------------------------------------------------------------------------

def _build_chan_aware_proposals(
    per_chan_proposals: List[dict],
    chan_indices: List[int],
    max_gap_minutes: float,
) -> dict:
    """Compress per-channel cluster proposals into channel-annotated flag entries.

    Wholesale flags (``bad_antennas``, ``bad_baselines``) are the union across
    all channels — a coordinate flagged wholesale in *any* channel is treated
    as pervasively bad across the full band.  These carry no channel
    restriction.

    Time-range flags (``bad_antenna_timeranges``, ``bad_baseline_timeranges``)
    are channel-annotated.  For each coordinate, the per-channel interval
    lists are grouped: entries whose interval sets are identical across a
    contiguous run of channel indices are compressed into one entry with
    ``{"intervals": [[t0,t1],...], "chanrange": [c_min, c_max]}``.
    A ``chanrange`` of ``None`` means "all loaded channels".

    Parameters
    ----------
    per_chan_proposals : list of dict
        One entry per channel, as returned by :func:`cluster_flags`.
    chan_indices : list of int
        Absolute 0-based FITS channel index for each entry in
        *per_chan_proposals*.
    max_gap_minutes : float
        Carried through for provenance; interval merging was already done
        inside :func:`cluster_flags` for each channel.

    Returns
    -------
    dict
        Same top-level keys as :func:`cluster_flags` but with **v2 schema**
        for time-range entries: each value is a list of
        ``{"intervals": [[t0_iso, t1_iso], ...], "chanrange": [c0, c1] | None}``.
    """
    # ── Wholesale flags: union across ALL channels ──────────────────────────
    all_ants:     set  = set()
    all_bases:    set  = set()   # tuple((a_name, b_name)) sorted
    all_scan_tr:  list = []
    all_burst_tr: list = []

    for p in per_chan_proposals:
        for a in p.get('bad_antennas', []):
            all_ants.add(a)
        for bl in p.get('bad_baselines', []):
            all_bases.add(tuple(sorted(bl)))
        all_scan_tr.extend(p.get('bad_scan_timeranges', []))
        all_burst_tr.extend(p.get('bad_burst_timeranges', []))

    # ── Time-range flags: collect per channel ──────────────────────────────
    # Structure: {key → {abs_chan_idx → [(t0_iso, t1_iso), ...]}}}
    bl_tr_per_chan:  Dict[str, Dict[int, List]] = {}
    ant_tr_per_chan: Dict[str, Dict[int, List]] = {}

    for proposal, abs_chan in zip(per_chan_proposals, chan_indices):
        for bl_key, trs in proposal.get('bad_baseline_timeranges', {}).items():
            bl_tr_per_chan.setdefault(bl_key, {})[abs_chan] = list(trs)
        for ant_name, trs in proposal.get('bad_antenna_timeranges', {}).items():
            ant_tr_per_chan.setdefault(ant_name, {})[abs_chan] = list(trs)

    n_chans_total = len(chan_indices)
    c_min_loaded  = min(chan_indices) if chan_indices else 0
    c_max_loaded  = max(chan_indices) if chan_indices else 0

    def _compress(per_chan: Dict[int, List]) -> List[dict]:
        """per_chan: {abs_chan_idx: [(t0_iso, t1_iso), ...]}
        Returns list of {"intervals": [[t0,t1],...], "chanrange": [c0,c1]|None}
        Channels with identical canonical interval sets are grouped into
        contiguous chanrange entries.
        """
        if not per_chan:
            return []

        def _canon(trs) -> tuple:
            return tuple(sorted((str(t[0]), str(t[1])) for t in trs))

        # Group channels by identical canonical interval set
        canon_to_chans: Dict[tuple, List[int]] = {}
        for c, trs in per_chan.items():
            canon_to_chans.setdefault(_canon(trs), []).append(c)

        entries = []
        for _canon_key, chans in canon_to_chans.items():
            chans_sorted = sorted(chans)
            orig_trs = [[str(t[0]), str(t[1])] for t in per_chan[chans_sorted[0]]]

            # Find contiguous runs of channel indices
            runs: List[tuple] = []
            start = prev = chans_sorted[0]
            for c in chans_sorted[1:]:
                if c == prev + 1:
                    prev = c
                else:
                    runs.append((start, prev))
                    start = prev = c
            runs.append((start, prev))

            for c0, c1 in runs:
                all_loaded = (c0 == c_min_loaded and c1 == c_max_loaded)
                chanrange = None if (n_chans_total <= 1 or all_loaded) else [c0, c1]
                entries.append({'intervals': orig_trs, 'chanrange': chanrange})
        return entries

    bad_baseline_timeranges = {k: _compress(v) for k, v in bl_tr_per_chan.items()}
    bad_antenna_timeranges  = {k: _compress(v) for k, v in ant_tr_per_chan.items()}

    return {
        'bad_antennas':            sorted(all_ants),
        'bad_baselines':           [[a, b] for (a, b) in sorted(all_bases)],
        'bad_scan_timeranges':     sorted({tuple(t) for t in all_scan_tr}),
        'bad_burst_timeranges':    sorted({tuple(t) for t in all_burst_tr}),
        'bad_antenna_timeranges':  bad_antenna_timeranges,
        'bad_baseline_timeranges': bad_baseline_timeranges,
    }


def _merge_flag_tables_in_memory(ft_a: dict, ft_b: dict, notes: str = '') -> dict:
    """Merge two in-memory flag tables, unioning all six v2 sections.

    This is the in-memory equivalent of
    :func:`ugmrt_query.merge_flag_table_into_file`.  It is used by
    :func:`run_clustering_detection` to union flag tables produced by
    independent per-corr detection runs.
    """
    # ── Wholesale antennas: union with dedup by name ──────────────────────
    seen_ants: set = set()
    merged_ants: list = []
    for ant in list(ft_a.get('bad_antennas', [])) + list(ft_b.get('bad_antennas', [])):
        key = str(ant).strip().upper()
        if key not in seen_ants:
            seen_ants.add(key)
            merged_ants.append(ant)

    # ── Wholesale baselines: union with dedup by sorted name pair ─────────
    seen_bls: set = set()
    merged_bls: list = []
    for bl in list(ft_a.get('bad_baselines', [])) + list(ft_b.get('bad_baselines', [])):
        if isinstance(bl, (list, tuple)) and len(bl) == 2:
            key = tuple(sorted([str(bl[0]).strip().upper(), str(bl[1]).strip().upper()]))
        else:
            key = str(bl)
        if key not in seen_bls:
            seen_bls.add(key)
            merged_bls.append(bl)

    # ── Time-range dicts: merge per key; append new entries ───────────────
    def _merge_tr_dict(section_key: str) -> dict:
        old_d: dict = dict(ft_a.get(section_key, {}))
        new_d: dict = dict(ft_b.get(section_key, {}))
        for coord_key, new_entries in new_d.items():
            if coord_key not in old_d:
                old_d[coord_key] = list(new_entries)
            else:
                old_d[coord_key] = list(old_d[coord_key]) + list(new_entries)
        return old_d

    # ── Scan / burst timeranges: union by string key ──────────────────────
    def _merge_tr_list(section_key: str) -> list:
        old_l = [tuple(str(t) for t in x) for x in ft_a.get(section_key, [])]
        new_l = [tuple(str(t) for t in x) for x in ft_b.get(section_key, [])]
        seen: set = set(old_l)
        merged_out: list = list(ft_a.get(section_key, []))
        for item, key in zip(ft_b.get(section_key, []), new_l):
            if key not in seen:
                seen.add(key)
                merged_out.append(item)
        return merged_out

    all_notes = ' | '.join(filter(None, [
        str(ft_a.get('notes', '')).strip(),
        str(ft_b.get('notes', '')).strip(),
        notes,
    ]))

    return {
        'kind':    'ugmrt_flag_table',
        'version': 2,
        'source':  ft_b.get('source', ft_a.get('source', '')),
        'notes':   all_notes,
        'bad_antennas':            merged_ants,
        'bad_baselines':           merged_bls,
        'bad_antenna_timeranges':  _merge_tr_dict('bad_antenna_timeranges'),
        'bad_baseline_timeranges': _merge_tr_dict('bad_baseline_timeranges'),
        'bad_scan_timeranges':     _merge_tr_list('bad_scan_timeranges'),
        'bad_burst_timeranges':    _merge_tr_list('bad_burst_timeranges'),
    }


def run_clustering_detection(
    vis_corr:    dict,
    ant_name_map: dict,
    *,
    corr:                            Union[str, List[str]]                   = 'V',
    threshold_jy:                    Union[float, Dict[str, float]]          = 5.0,
    threshold_low_jy:                Union[float, Dict[str, float], None]   = None,
    min_cluster_fraction:            float          = 0.80,
    min_distinct_baselines_for_ant:  int            = 3,
    min_burst_baseline_fraction:     float          = 0.50,
    whole_scan_bad_fraction:         float          = 0.70,
    per_scan_ant_fraction:           float          = 0.60,
    per_scan_ant_bl_fraction:        float          = 0.50,
    baseline_max_bad_samples:        Optional[int]  = 100,
    max_gap_samples:                 int            = 2,
    max_gap_minutes:                 float          = 30.0,
    scan_gap_minutes:                float          = 2.0,
    source:                          str            = '',
    obs_jd_start:                    Optional[float] = None,
    obs_jd_end:                      Optional[float] = None,
    elevation_min_deg:               Optional[float] = None,
    verbose:                         bool           = True,
) -> dict:
    """Run per-channel clustering outlier detection on bandpass-corrected visibilities.

    Parameters
    ----------
    vis_corr : dict
        Already-loaded, already-bandpass-corrected visibility dict as returned
        by :func:`ugmrt_query.apply_bandpass_solution`.  Must contain
        ``'vis_complex'``, ``'jd'``, ``'ant1'``, ``'ant2'``.
    ant_name_map : dict
        Mapping ``{antenna_id: antenna_name}`` used to name flag entries.
        Typically built from ``row_index['antennas']``.
    corr : str or list of str
        Correlator product(s) for the test statistic.  A single string (e.g.
        ``'V'``) reproduces the original single-corr behaviour.  A list (e.g.
        ``['V', 'RR', 'LL']``) runs detection independently per corr and
        unions the resulting flag tables before returning.
    threshold_jy : float or dict
        Upper flag threshold.  A scalar applies the same value to all corrs.
        A dict (e.g. ``{'V': 8.0, 'RR': 500.0}``) sets per-corr thresholds;
        missing corrs fall back to 5.0 Jy.
    threshold_low_jy : float, dict, or None
        Lower flag threshold.  Rows whose test quantity falls *below* this
        value are also flagged.  Useful for catching dead antennas via
        anomalously low RR/LL power.  Not meaningful for Stokes-V.  Same
        scalar/dict/None semantics as *threshold_jy*.
    verbose : bool
        Print per-channel progress and final summary.

    Returns
    -------
    dict with keys:
        ``'flag_table'``          — merged ``ugmrt_flag_table`` dict (union
                                    across all corrs), compatible with
                                    ``q.apply_flag_tables_to_vis`` and
                                    ``q.save_flag_table``.
        ``'per_chan_proposals'``  — per-channel proposals for the first corr
                                    (backward-compatible).
        ``'n_channels'``         — number of channels processed.
        ``'n_bad_channels'``     — channels where at least one corr flagged.
        ``'per_corr_results'``   — dict keyed by corr with full per-corr
                                    breakdown (flag_table, per_chan_proposals,
                                    n_bad_channels).

    Notes
    -----
    Each channel is processed independently.  Per-channel proposals are
    compressed by :func:`_build_chan_aware_proposals` into a **v2 flag table**
    where every time-range entry carries a ``chanrange`` field recording which
    contiguous block of channels the flag applies to.  Wholesale
    ``bad_antennas`` / ``bad_baselines`` entries (bad fraction >=
    *min_cluster_fraction*) are still all-channel since they are considered
    pervasively bad.  Time-range entries are channel-selective.
    """
    # ── Normalise corr and thresholds to lists / dicts ────────────────────
    if isinstance(corr, str):
        corr_list: List[str] = [corr]
    else:
        corr_list = list(corr)
    for _c in corr_list:
        if _c not in _CORR_STOKES_NEEDED:
            raise ValueError(
                f"Unknown corr '{_c}'. "
                f"Supported values: {sorted(_CORR_STOKES_NEEDED)}"
            )

    if isinstance(threshold_jy, (int, float)):
        thr_high: Dict[str, float] = {_c: float(threshold_jy) for _c in corr_list}
    else:
        thr_high = {_c: float(threshold_jy.get(_c, 5.0)) for _c in corr_list}

    if threshold_low_jy is None:
        thr_low: Dict[str, Optional[float]] = {_c: None for _c in corr_list}
    elif isinstance(threshold_low_jy, (int, float)):
        thr_low = {_c: float(threshold_low_jy) for _c in corr_list}
    else:
        thr_low = {
            _c: (float(threshold_low_jy[_c]) if _c in threshold_low_jy else None)
            for _c in corr_list
        }

    # ── Shared channel meta (identical across corrs from same vis_corr) ───
    _raw_ci = vis_corr.get('chan_indices')
    _tq_tmp = compute_test_quantity(vis_corr, corr_list[0])
    nrows, nchans = _tq_tmp.shape
    del _tq_tmp
    if _raw_ci is not None:
        chan_indices: List[int] = [int(_c) for _c in _raw_ci]
    else:
        chan_indices = list(range(nchans))

    obs_jd_start_ = obs_jd_start if obs_jd_start is not None else float(vis_corr['jd'].min())
    obs_jd_end_   = obs_jd_end   if obs_jd_end   is not None else float(vis_corr['jd'].max())

    # ── Run per-corr detection; union flag tables across corrs ────────────
    combined_ft:     Optional[dict]      = None
    primary_per_chan: Optional[List[dict]] = None
    per_corr_results: Dict[str, dict]    = {}

    for _c in corr_list:
        _thr_h = thr_high[_c]
        _thr_l = thr_low[_c]
        tq_arr = compute_test_quantity(vis_corr, _c)

        if verbose:
            _low_str = f'  low={_thr_l} Jy' if _thr_l is not None else ''
            print(f'[clustering] corr={_c}  thr={_thr_h} Jy{_low_str}  '
                  f'{nchans} channel(s)  chans={chan_indices[0]}-{chan_indices[-1]}  '
                  f'{nrows:,} rows')

        per_chan_proposals: List[dict] = []
        for c_local in range(nchans):
            tq_chan  = tq_arr[:, c_local:c_local + 1]
            bad_mask = build_bad_mask(tq_chan, _thr_h, threshold_low=_thr_l)

            proposal = cluster_flags(
                vis_corr,
                bad_mask,
                ant_name_map,
                min_cluster_fraction,
                min_distinct_baselines_for_ant,
                min_burst_baseline_fraction,
                max_gap_minutes,
                scan_gap_minutes,
                whole_scan_bad_fraction  = whole_scan_bad_fraction,
                per_scan_ant_fraction    = per_scan_ant_fraction,
                per_scan_ant_bl_fraction = per_scan_ant_bl_fraction,
                baseline_max_bad_samples = baseline_max_bad_samples,
                max_gap_samples          = max_gap_samples,
            )
            per_chan_proposals.append(proposal)

            if verbose and nchans > 1:
                n_a = len(proposal['bad_antennas'])
                n_b = len(proposal['bad_baselines'])
                print(f'\r  chan {c_local+1:>3d}/{nchans}: '
                      f'{int(bad_mask.sum()):,}/{nrows:,} bad rows  '
                      f'→ {n_a} ant(s) {n_b} bl(s)    ',
                      end='', flush=True)

        if verbose and nchans > 1:
            print()   # newline after \r progress

        merged = _build_chan_aware_proposals(per_chan_proposals, chan_indices, max_gap_minutes)

        _notes = (
            f'run_clustering_detection; corr={_c}; thr={_thr_h} Jy'
            + (f'/low={_thr_l} Jy' if _thr_l is not None else '')
            + f'; {nchans} channel(s) chans={chan_indices[0]}-{chan_indices[-1]}'
        )
        ft = build_output_flag_table(
            merged,
            source            = source,
            notes             = _notes,
            obs_jd_start      = obs_jd_start_,
            obs_jd_end        = obs_jd_end_,
            elevation_min_deg = elevation_min_deg,
        )

        n_bad_c = sum(
            1 for p in per_chan_proposals
            if p['bad_antennas'] or p['bad_baselines']
        )
        per_corr_results[_c] = {
            'flag_table':         ft,
            'per_chan_proposals':  per_chan_proposals,
            'n_bad_channels':     n_bad_c,
        }
        if primary_per_chan is None:
            primary_per_chan = per_chan_proposals
        combined_ft = ft if combined_ft is None else _merge_flag_tables_in_memory(combined_ft, ft)

        if verbose:
            print(f'[clustering] {_c}: '
                  f'{len(merged["bad_antennas"])} antenna(s) flagged, '
                  f'{len(merged["bad_baselines"])} baseline(s) flagged '
                  f'(from {n_bad_c}/{nchans} active channels)')

    # ── Summary n_bad_channels: any corr flagged in that channel ─────────
    n_bad_chans = sum(
        1 for ci in range(nchans)
        if any(
            per_corr_results[_c]['per_chan_proposals'][ci]['bad_antennas']
            or per_corr_results[_c]['per_chan_proposals'][ci]['bad_baselines']
            for _c in corr_list
        )
    )

    if verbose and len(corr_list) > 1:
        _comb_ants  = combined_ft.get('bad_antennas', [])
        _comb_bases = combined_ft.get('bad_baselines', [])
        print(f'[clustering] combined {corr_list}: '
              f'{len(_comb_ants)} antenna(s) flagged, '
              f'{len(_comb_bases)} baseline(s) flagged '
              f'(from {n_bad_chans}/{nchans} active channels)')

    return {
        'flag_table':         combined_ft,
        'per_chan_proposals':  primary_per_chan,
        'n_channels':         nchans,
        'n_bad_channels':     n_bad_chans,
        'per_corr_results':   per_corr_results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # global declarations MUST come before any reference to these names in
    # this function scope (Python requires this to avoid SyntaxError).
    global FITS_PATH, INDEX_CACHE, BANDPASS_NPZ, OUT_DIR
    global CALIBRATOR_SOURCE, TARGET_SOURCES, DUD_ANTENNA_NAMES
    global CHAN_RANGE, DETECTION_CORR, DETECTION_THRESHOLD_JY, MIN_CLUSTER_FRACTION, DRY_RUN
    global ELEVATION_MIN_DEG

    # ------------------------------------------------------------------
    # CLI argument parsing  (all args optional; CONFIG-block is the default)
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        prog='outlier_detection.py',
        description=(
            'Per-channel uGMRT visibility outlier detector.\n'
            'CLI arguments override CONFIG-block defaults.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Examples:\n'
            '  python3 outlier_detection.py --inspect --cal 3C48\n'
            '  python3 outlier_detection.py --chans 64-191 --cal 3C48 --no-dry-run\n'
            '  python3 outlier_detection.py --chans 128\n'
            '  python3 outlier_detection.py --fits /data/obs.FITS --bp /work/bp.npz'
        ),
    )

    # File paths
    grp_files = parser.add_argument_group('file paths')
    grp_files.add_argument('--fits',  metavar='PATH', help=f'FITS file  [default: {FITS_PATH}]')
    grp_files.add_argument('--index', metavar='PATH', help=f'Row-index cache .npz  [default: {INDEX_CACHE}]')
    grp_files.add_argument('--bp',    metavar='PATH', help=f'Bandpass solution .npz  (not needed for --inspect)  [default: {BANDPASS_NPZ}]')
    grp_files.add_argument('--out',   metavar='DIR',  help=f'Output directory  [default: {OUT_DIR}]')

    # Source selection
    grp_src = parser.add_argument_group('source selection')
    grp_src.add_argument('--cal',    metavar='NAME', help=f'Calibrator source name  [default: {CALIBRATOR_SOURCE}]')
    grp_src.add_argument('--target', metavar='NAME', action='append', default=None,
                         help=f'Target source(s) to transfer flags to (repeat for multiple)  [default: {TARGET_SOURCES or "none"}]')
    grp_src.add_argument('--dud',    metavar='ANT',  action='append', default=None,
                         help=f'Dud antenna name to exclude (repeat for multiple, replaces CONFIG list)  [default: {DUD_ANTENNA_NAMES or "none"}]')

    # Data selection
    grp_sel = parser.add_argument_group('data selection')
    grp_sel.add_argument(
        '--chans', metavar='RANGE',
        help=(
            f'Channel range as START-END (inclusive, 0-based FITS convention) '
            f'or a single channel number.  [default: {CHAN_RANGE[0]}-{CHAN_RANGE[1]}]'
        ),
    )
    grp_sel.add_argument('--elev-min', type=float, metavar='DEG',
                         help=f'Minimum elevation (degrees); rows below this are excluded from '
                              f'detection, and wholesale CASA flag commands are bounded to the '
                              f'examined window  [default: {ELEVATION_MIN_DEG}]')

    # Detection thresholds
    grp_thr = parser.add_argument_group('detection thresholds')
    grp_thr.add_argument(
        '--corr', metavar='CORR',
        help=(
            f'Correlator product for outlier detection (one per run).  '
            f'V=|RR−LL|/2  Q=|XX−YY|/2  I=(|RR|+|LL|)/2  '
            f'RR/LL/XX/YY/RL/LR/XY/YX=amplitude.  '
            f'[default: {DETECTION_CORR}]'
        ),
    )
    grp_thr.add_argument('--thr', type=float, metavar='JY',
                         help=f'Detection threshold in Jy for the chosen --corr  [default: {DETECTION_THRESHOLD_JY}]')
    grp_thr.add_argument('--cluster-frac', type=float, metavar='F',
                         help=f'Wholesale cluster fraction 0–1  [default: {MIN_CLUSTER_FRACTION}]')

    # Output control
    grp_out = parser.add_argument_group('output control')
    grp_out.add_argument('--dry-run', dest='dry_run', action=argparse.BooleanOptionalAction,
                         default=None,
                         help=f'Print proposed flags without writing files; '
                              f'use --no-dry-run to write JSON + CASA flag files  '
                              f'[default: {DRY_RUN}]')
    grp_out.add_argument('--inspect', action='store_true', default=False,
                         help=(
                             'Survey mode: print scan structure, integration time, '
                             'antenna count, and suggested CONFIG parameters for '
                             'the calibrator source.  No flagging is performed.'
                         ))

    args = parser.parse_args()

    if args.fits:         FITS_PATH          = Path(args.fits)
    if args.index:        INDEX_CACHE        = Path(args.index)
    if args.bp:           BANDPASS_NPZ       = Path(args.bp)
    if args.out:          OUT_DIR            = Path(args.out)
    if args.cal:          CALIBRATOR_SOURCE  = args.cal
    if args.target:       TARGET_SOURCES     = args.target
    if args.dud:          DUD_ANTENNA_NAMES  = args.dud
    if args.dry_run is not None: DRY_RUN              = args.dry_run
    if args.cluster_frac is not None: MIN_CLUSTER_FRACTION = args.cluster_frac
    if args.corr is not None:     DETECTION_CORR          = args.corr.strip()
    if args.thr  is not None:     DETECTION_THRESHOLD_JY  = args.thr
    if args.chans:
        raw = args.chans.strip()
        if '-' in raw:
            parts = raw.split('-', 1)
            CHAN_RANGE = (int(parts[0]), int(parts[1]))
        else:
            ch = int(raw)
            CHAN_RANGE = (ch, ch)
    if args.elev_min is not None:
        ELEVATION_MIN_DEG = args.elev_min

    # ------------------------------------------------------------------
    # Insert the script's own directory so ugmrt_query is importable.
    # ------------------------------------------------------------------
    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    import ugmrt_query as q

    # ------------------------------------------------------------------
    # 1. Row index
    # ------------------------------------------------------------------
    print('[1] Loading row index ...')
    row_index = q.get_or_build_row_index(
        FITS_PATH,
        cache_path         = INDEX_CACHE,
        force_rebuild      = False,
        validation_mode    = 'fast+sha',
        write_cache        = True,
        override_dud_names = DUD_ANTENNA_NAMES,
    )
    ant_name_map: Dict[int, str] = {
        int(a['antenna_no']): str(a['name'])
        for a in row_index['antennas']
    }
    file_stokes: List[str] = list(row_index['stokes_labels'])
    print(f'   Active antennas : {len(row_index.get("active_antennas", row_index["antennas"]))}')
    print(f'   Correlators in file: {file_stokes}')

    # Validate DETECTION_CORR against what the file actually recorded.
    # Do this before any vis I/O so the user gets a clear error immediately.
    if DETECTION_CORR not in _CORR_STOKES_NEEDED:
        print(
            f'\nERROR: Unknown DETECTION_CORR={DETECTION_CORR!r}.\n'
            f'       Supported values: {sorted(_CORR_STOKES_NEEDED)}\n'
        )
        sys.exit(1)
    _needed: Tuple[str, ...] = _CORR_STOKES_NEEDED[DETECTION_CORR]
    _missing = [s for s in _needed if s not in file_stokes]
    if _missing:
        print(
            f'\nERROR: DETECTION_CORR={DETECTION_CORR!r} needs correlator(s) {_missing} '
            f'but the FITS file only contains: {file_stokes}\n'
            f'       Choose one of the corrs whose inputs are all present in the file.\n'
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # --inspect mode: load timestamps only, print scan report, exit
    # ------------------------------------------------------------------
    if args.inspect:
        print(f'[inspect] Loading timestamps for {CALIBRATOR_SOURCE} '
              f'(chan {CHAN_RANGE[0]} only — timestamps are channel-independent) ...')
        _vis_ts = q.load_vis_for_source(
            row_index,
            source            = CALIBRATOR_SOURCE,
            chan_range         = (CHAN_RANGE[0], CHAN_RANGE[0]),   # single channel — minimal I/O
            stokes             = list(_CORR_STOKES_NEEDED[DETECTION_CORR]),
            max_rows           = None,
            elevation_min_deg  = ELEVATION_MIN_DEG,
            flag_all_corrs_if_any_rawvis_flagged = False,
        )
        _inspect_mode(
            row_index      = row_index,
            source         = CALIBRATOR_SOURCE,
            vis_jd         = _vis_ts['jd'],
            vis_a1         = _vis_ts['ant1'],
            vis_a2         = _vis_ts['ant2'],
            fits_path      = FITS_PATH,
            chan_range      = CHAN_RANGE,
            scan_gap_minutes = SCAN_GAP_MINUTES,
        )
        if args.dry_run is not None:
            print('[warn] --inspect exits before any file I/O; --dry-run / --no-dry-run has no effect')
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 2. Load calibrator visibilities
    # ------------------------------------------------------------------
    _needed_stokes = list(_CORR_STOKES_NEEDED[DETECTION_CORR])
    print(f'[2/4] Loading calibrator visibilities ({CALIBRATOR_SOURCE}) '
          f'[corr={DETECTION_CORR}, stokes={_needed_stokes}] ...')
    vis_raw = q.load_vis_for_source(
        row_index,
        source            = CALIBRATOR_SOURCE,
        chan_range         = CHAN_RANGE,
        stokes             = _needed_stokes,
        max_rows           = MAX_ROWS,
        elevation_min_deg  = ELEVATION_MIN_DEG,
        flag_all_corrs_if_any_rawvis_flagged = True,
    )
    print(f'   Shape: {vis_raw["amp"].shape}  '
          f'({vis_raw["amp"].shape[0]} rows × {vis_raw["amp"].shape[1]} chans × {vis_raw["amp"].shape[2]} pols)')

    vis_corr = q.apply_bandpass_solution(vis_raw, q.load_bandpass_solution(BANDPASS_NPZ))
    print('   Bandpass solution applied (normalises channel amplitudes).')

    # Record the JD bounds of the (elevation-filtered) data that was loaded.
    # These are used to bound wholesale CASA flag commands so they only apply
    # during the window that was actually examined.
    obs_jd_start = float(vis_raw['jd'].min())
    obs_jd_end   = float(vis_raw['jd'].max())

    # ------------------------------------------------------------------
    # 3. Outlier detection
    # ------------------------------------------------------------------
    print(f'[3/4] Computing test quantities  [corr={DETECTION_CORR}, thr={DETECTION_THRESHOLD_JY} Jy] ...')

    test_quantity_arr = compute_test_quantity(vis_corr, DETECTION_CORR)
    # test_quantity_arr: (nrows, nchans)

    nchans_loaded = test_quantity_arr.shape[1]
    chan_indices   = list(range(nchans_loaded))   # 0-based within loaded block
    n_total_chans  = len(chan_indices)
    print(f'   Ready: {n_total_chans} channel(s), {test_quantity_arr.shape[0]:,} rows '
          f'(FITS chans {CHAN_RANGE[0]}–{CHAN_RANGE[1]}).')

    # ------------------------------------------------------------------
    # Per-channel loop
    # ------------------------------------------------------------------
    for c_local in chan_indices:
        c_global = CHAN_RANGE[0] + c_local   # absolute FITS channel number
        chan_label = f'chan{c_global:04d}'

        # Single-channel test-quantity slice: (nrows, 1)
        tq_chan  = test_quantity_arr[:, c_local:c_local + 1]

        bad_mask = build_bad_mask(tq_chan, DETECTION_THRESHOLD_JY)
        n_bad    = int(bad_mask.sum())
        n_rows   = len(bad_mask)

        if n_total_chans > 1:
            print(f'\r   Chan {c_global} ({c_local+1}/{n_total_chans}): '
                  f'{n_bad:,}/{n_rows:,} bad rows ({100*n_bad/max(n_rows,1):.1f}%)  ',
                  end='', flush=True)
        else:
            print(f'   Chan {c_global}: {n_bad:,}/{n_rows:,} bad rows '
                  f'({100*n_bad/max(n_rows,1):.1f}%)')

        cal_proposal = cluster_flags(
            vis_corr,
            bad_mask,
            ant_name_map,
            MIN_CLUSTER_FRACTION,
            MIN_DISTINCT_BASELINES_FOR_ANT,
            MIN_BURST_BASELINE_FRACTION,
            MAX_GAP_MINUTES,
            SCAN_GAP_MINUTES,
            whole_scan_bad_fraction  = WHOLE_SCAN_BAD_FRACTION,
            per_scan_ant_fraction    = PER_SCAN_ANT_FRACTION,
            per_scan_ant_bl_fraction = PER_SCAN_ANT_BL_FRACTION,
            baseline_max_bad_samples = BASELINE_MAX_BAD_SAMPLES,
            max_gap_samples          = MAX_GAP_SAMPLES,
        )

        if n_total_chans == 1:
            print_summary(cal_proposal, label=f'{CALIBRATOR_SOURCE} chan {c_global}')

        # ── 5. Scan matrices ───────────────────────────────────────────
        scan_matrices, summary_matrix = build_scan_flag_matrices(
            vis_corr, bad_mask, ant_name_map, SCAN_GAP_MINUTES,
        )

        # ── 6. Save outputs ────────────────────────────────────────────
        cal_flag_table = build_output_flag_table(
            cal_proposal,
            source = CALIBRATOR_SOURCE,
            notes  = (f'outlier_detection.py; chan={c_global}; '
                      f'corr={DETECTION_CORR}; thr={DETECTION_THRESHOLD_JY} Jy; '
                      f'min_cluster_frac={MIN_CLUSTER_FRACTION}; '
                      f'elevation_min_deg={ELEVATION_MIN_DEG}'),
            obs_jd_start      = obs_jd_start,
            obs_jd_end        = obs_jd_end,
            elevation_min_deg = ELEVATION_MIN_DEG,
        )
        cal_json_path = OUT_DIR / f'{src_slug}_{chan_label}_outlier_flags.json'
        cal_casa_path = OUT_DIR / f'{src_slug}_{chan_label}_outlier_flags.casa.txt'

        if not DRY_RUN:
            cal_json_path.write_text(json.dumps(cal_flag_table, indent=2, sort_keys=True),
                                     encoding='utf-8')
            casa_lines = format_casa_commands(cal_flag_table, reason_prefix='outlier_det')
            cal_casa_path.write_text('\n'.join(casa_lines) + '\n', encoding='utf-8')
            if n_total_chans == 1:
                print(f'   Saved: {cal_json_path}')
                print(f'   Saved: {cal_casa_path}')
        else:
            if n_total_chans == 1:
                print(f'   [dry-run] would write: {cal_json_path}')
                print(f'   [dry-run] would write: {cal_casa_path}')
                print()
                print(f'   ── CASA commands for {CALIBRATOR_SOURCE} chan {c_global} ──')
                for line in format_casa_commands(cal_flag_table):
                    print(f'   {line}')

        # ── Per-scan matrix plots ──────────────────────────────────────
        try:
            import matplotlib
            matplotlib.use('Agg')
            for scan_i, sm in scan_matrices.items():
                fig = plot_scan_flag_matrix(
                    sm,
                    title=f'{CALIBRATOR_SOURCE} {chan_label} — Scan {scan_i + 1}',
                )
                plot_path = OUT_DIR / f'{src_slug}_{chan_label}_scan{scan_i + 1:02d}_flag_matrix.png'
                if not DRY_RUN:
                    fig.savefig(plot_path, dpi=100, bbox_inches='tight')
                    if n_total_chans == 1:
                        print(f'   Saved: {plot_path}')
                import matplotlib.pyplot as plt
                plt.close(fig)

            fig_sum = plot_scan_flag_matrix(
                summary_matrix,
                title=f'{CALIBRATOR_SOURCE} {chan_label} — All Scans',
            )
            sum_plot_path = OUT_DIR / f'{src_slug}_{chan_label}_all_scans_flag_matrix.png'
            if not DRY_RUN:
                fig_sum.savefig(sum_plot_path, dpi=100, bbox_inches='tight')
            plt.close(fig_sum)
        except Exception as exc:
            print(f'   WARNING: plots failed for chan {c_global}: {exc}')

        # ── Target transfer ────────────────────────────────────────────
        for target in TARGET_SOURCES:
            tgt_slug      = target.lower().replace(' ', '_')
            tgt_flag_table = build_target_flag_table(
                cal_flag_table,
                target_source       = target,
                transfer_timeranges = TRANSFER_TIMERANGE_FLAGS,
                notes               = (f'Transferred from {CALIBRATOR_SOURCE} '
                                       f'chan={c_global}; '
                                       f'transfer_timeranges={TRANSFER_TIMERANGE_FLAGS}'),
            )
            tgt_json_path = OUT_DIR / f'{tgt_slug}_{chan_label}_outlier_flags.json'
            tgt_casa_path = OUT_DIR / f'{tgt_slug}_{chan_label}_outlier_flags.casa.txt'
            if not DRY_RUN:
                tgt_json_path.write_text(json.dumps(tgt_flag_table, indent=2, sort_keys=True),
                                         encoding='utf-8')
                casa_lines = format_casa_commands(tgt_flag_table,
                                                  reason_prefix='outlier_det_transfer')
                tgt_casa_path.write_text('\n'.join(casa_lines) + '\n', encoding='utf-8')

    # End of per-channel loop
    if n_total_chans > 1:
        print()
        print(f'   Done. {n_total_chans} channel(s) processed.')
        if DRY_RUN:
            print(f'   [dry-run] would write {n_total_chans} JSON + CASA file(s) to {OUT_DIR}/')

    print()
    print('[4/4] Done.')


if __name__ == '__main__':
    main()
