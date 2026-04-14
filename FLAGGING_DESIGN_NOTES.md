# Flagging System — Design Principles & Known Issues
*Human-taught notes. Do not modify without understanding the reasoning below.*

---

## Fundamental Data Layout (FITS UVFITS)

- One **row** = one **(baseline, time)** sample.
- Shape per row after extraction: `(nChans, nPols, 3)` where the last axis is
  `[real, imag, weight]`.
- For this dataset: `128 channels × 2 pols × 3 = 768 float32 values per row`.
- `nRows = nIntegrations × nActiveBaselines`
- Channels and polarisations are packed **within** a row — they are NOT
  separate rows. This is fundamental: flagging a "row" drops all channels and
  all pols for that (baseline, time) sample simultaneously.

---

## Active Baselines — Correct Denominator

- The theoretical baseline count is `nAnt × (nAnt-1) / 2`.
- **DUD antennas** (hardware-dead, never correlated) have no rows in the FITS
  file at all. Their baselines simply do not exist.
- Therefore the correct denominator for any flagging fraction or coverage
  statistic is:
  ```
  n_active_baselines = len(unique (ant1, ant2) pairs in vis_raw)
  ```
  NOT `nAnt*(nAnt-1)/2` from the antenna table.
- `DUD_ANTENNA_NAMES` (user config) gives the *expected* absent antennas.
  Cross-check: if `n_observed_active != n_expected_active`, warn — an antenna
  not listed as DUD has no data.
- **All coverage/flagging-% log lines must use the observed count.**

---

## TODO #1 — Wholesale→All-Channel Promotion Bug

**File:** `outlier_detection.py` → `_build_chan_aware_proposals()`

**Problem:**  
Clustering runs per-channel. `cluster_flags()` returns a `bad_antennas` list
for each channel independently. `_build_chan_aware_proposals()` currently takes
the **union across all channels** and promotes any antenna flagged wholesale in
*any single channel* into the top-level `bad_antennas` list — which carries no
channel restriction.

`apply_flag_tables_to_vis` then **drops every row** touching that antenna
across **all channels**.

**Example of the damage:**  
1 fully-FITS-flagged channel (wt≤0 for all rows) → `tq = NaN` for that channel
→ `build_bad_mask` (before the NaN fix) flagged all rows → `cluster_flags`
proposed all antennas as wholesale bad → `_build_chan_aware_proposals` promoted
them to all-band `bad_antennas` → 100% of data lost (128× amplification).

**Correct semantics:**  
"Wholesale bad in channel C" means: this antenna/baseline was bad for the
*entire time range of channel C*. It should be stored as a
`bad_antenna_timeranges` entry with `chanrange: [C, C]` covering the full
observation time — **not** as an all-band row deletion.

Only promote to true `bad_antennas` (no chanrange) when the antenna is
wholesale-bad across **all loaded channels** (or a configurable majority
fraction thereof).

Same logic applies to `bad_baselines`.

---

## TODO #2 — Flagging Stats Denominator

**Files:** `run_clustering.py`, `preprocess_ugmrt.py`, anywhere
`Coverage:` or flagging-% is logged.

**Problem:**  
Current code reports `X/496 baselines` or `X/32 antennas` derived from the
FITS antenna table. This overcounts when DUD antennas are present.

**Fix:**  
- Compute `n_active_baselines = len(set(zip(vis_raw['ant1'], vis_raw['ant2'])))`
  after loading.
- Compute `n_active_antennas = len(set(vis_raw['ant1']) | set(vis_raw['ant2']))`
- Cross-check against `DUD_ANTENNA_NAMES`-corrected expected counts.
- Use observed counts in all log lines.

---

## NaN Propagation Chain (commit 42d0e18 + fix 0d5d7df)

- `load_vis_for_source` now sets `vis_complex[flagged] = nan+0j` for all
  natively-FITS-flagged cells (wt≤0) AND exact-zero cells (hardware dropouts).
- `apply_bandpass_solution` now sets `corrected[corrected_flagged] = nan+nanj`
  so `vis_complex_corrected` is consistent.
- `compute_test_quantity` therefore produces `tq = NaN` for every
  natively-flagged (row, chan).
- **`build_bad_mask` must NOT treat NaN as an outlier.** NumPy comparisons
  against NaN return False, so `(tq > threshold)` already skips them safely.
  The old `~np.isfinite(tq)` term was removed in `0d5d7df` for this reason.
  NaN = "no measurement", not "bad measurement".

---

## Key Principle

> A flag should only remove data at the **(baseline, channel, time)** coordinates
> where the data is actually bad. Wholesale antenna flags are appropriate only
> when the antenna is bad across the **full band and full time range**.
> Channel-selective badness must be expressed as `chanrange`-restricted
> time-range entries, not as all-band row deletions.
