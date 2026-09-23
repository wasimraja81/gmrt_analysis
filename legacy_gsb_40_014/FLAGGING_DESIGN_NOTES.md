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

## Flag Array Shape & Fine-Grained Statistics

### Canonical flag array shape

User-defined flag directives (wholesale antennas, baselines, timerange entries
with optional `chanrange`) all resolve to `(row, channel)` cells. Polarisation
is never an independent flag axis — `FLAG_ALL_CORRS_IF_ANY_RAWVIS_FLAGGED=True`
is the only supported mode, so every flag covers all corrs implicitly.

```
flag_array shape:  (nRows, nChans)  =  (57456, 128)   bool
```

The pol axis is handled by broadcasting this mask over `vis_complex`
`(nRows, nChans, nPols)` at apply time, not by storing it.

### 3-D reshape for statistics

Because DUD antennas have **zero rows** (not sparse/missing entries), the
active-baseline grid is fully regular: every integration contributes exactly
the same `nActiveBaselines` rows in the same order. This makes a lossless
reshape valid:

```
flags_3d shape:  (nIntegrations, nActiveBaselines, nChans)
               = (nIntegrations, 448, 128)          for this dataset
```

All statistics are then single numpy reductions:

| Statistic | Operation | Output shape |
|---|---|---|
| % flagged per baseline | `flags_3d.mean(axis=(0,2))` | `(nActiveBaselines,)` |
| % flagged per channel  | `flags_3d.mean(axis=(0,1))` | `(nChans,)` |
| % flagged per integration | `flags_3d.mean(axis=(1,2))` | `(nIntegrations,)` |
| total % flagged | `flags_3d.mean()` | scalar |

### Per-antenna statistics

A row involves **two** antennas (`ant1`, `ant2`), so antenna stats cannot be
read off a single axis. An auxiliary lookup table is required:

```python
bl_ant_table: (nActiveBaselines, 2)   # bl_ant_table[b] = (ant1_idx, ant2_idx)
```

Then for antenna `i`:

```python
bl_mask = (bl_ant_table[:, 0] == i) | (bl_ant_table[:, 1] == i)
pct_flagged_ant_i = flags_3d[:, bl_mask, :].mean()
```

### Implication for TODO #2

Once `flags_3d` exists, all Coverage/flagging-% log lines reduce to axis
reductions on it — there is no need to track separate counters. The correct
denominators (observed `nActiveBaselines`, `nIntegrations`, `nChans`) fall
out automatically as dimension sizes of `flags_3d`.

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

## TODO #3 — Verify & Wire `FLAG_ALL_CORRS_IF_ANY_RAWVIS_FLAGGED` End-to-End

**Files:** `ugmrt_query.py`, `run_clustering.py`, `preprocess_ugmrt.py`,
`tools/dev/patch_thresholds.py`, `apply_flag_tables_to_vis` (wherever called).

### Current state

`FLAG_ALL_CORRS_IF_ANY_RAWVIS_FLAGGED` is a config variable that controls
whether natively-FITS-flagged samples are symmetrised across all correlations
at load time. **Only `True` is supported.** Every production entry point
hard-codes `True`:

- `run_clustering.py` line 80: `FLAG_ALL_CORRS_IF_ANY_RAWVIS_FLAGGED: bool = True`
- `preprocess_ugmrt.cfg` line 66: `FLAG_ALL_CORRS_IF_ANY_RAWVIS_FLAGGED = True`
- `tools/dev/patch_thresholds.py` line 86: `FLAG_ALL_CORRS_IF_ANY_RAWVIS_FLAGGED = True`

The underlying function signatures default to `False` (e.g. `ugmrt_query.py`
lines 2171, 3573, 3987, 4994) — that default is **never** exercised in any
real pipeline run. The `False` code path is untested and unsupported.

### Step 1 — Verify the current `True` behaviour ✅ VERIFIED

**1. Symmetrisation in `load_vis_for_source`** (`ugmrt_query.py` ~line 2399):
```python
shared_flagged = np.any(flagged, axis=2, keepdims=True)  # OR across corr axis
flagged = np.broadcast_to(shared_flagged, flagged.shape).copy()  # back to all corrs
wt_[flagged] = 0.0
```
A post-condition assertion is now present confirming
`flagged.all(axis=2) == flagged.any(axis=2)` (uniformity across corrs).
A diagnostic count log reports how many `(row, chan)` asymmetric cells were corrected.

**2. Flag-table entries carry no per-corr field** — verified:
- `expand_flag_table_to_mask` returns `(nRows, nChans)` — no pol axis.
  The pol axis is handled at apply time by broadcasting over `vis_complex`
  `(nRows, nChans, nPols)`. A `# TODO #3 sanity` comment marks this invariant.
- `apply_flag_tables_to_vis` drops whole rows (all corrs simultaneously).
- The JSON schema (`bad_antennas`, `bad_baselines`, `bad_antenna_timeranges`,
  `bad_baseline_timeranges`) has no `corr` or `stokes` field — all-corr by
  construction.

**3. Outlier detection is corr-free** — verified:
- `compute_test_quantity` collapses all corrs into a single scalar per
  `(row, chan)` before any threshold decision (Stokes-V: `|RR−LL|/2`;
  Stokes-I: `(|RR|+|LL|)/2`; single-pol: `|pol|`). Output shape:
  `(nrows, nchans)` — no corr index.
- `build_bad_mask` operates on that scalar — no per-corr state.
- `_build_chan_aware_proposals` maps `(row, chan)` cells into flag-table entries
  that carry no corr field. ✅

### Step 2 — WISHLIST: wire consistently end-to-end

Ultimately `FLAG_ALL_CORRS_IF_ANY_RAWVIS_FLAGGED` should be a single
authoritative switch, honoured consistently for **both**:

1. **Native FITS flags** — symmetrisation at load time (current usage).
2. **Clustering-detected outlier flags** written to the flag table.

For case 2: once TODO #1 (chanrange-restricted entries) is implemented,
ensure new `chanrange`-qualified entries still carry no per-corr field (the
flag table schema must remain all-corr). The variable should be read at
flag-application time in `apply_flag_tables_to_vis`, not only at load-parse
time in `load_vis_for_source`.

When `False` support is eventually added it must be tested with a synthetic
dataset containing deliberately asymmetric RR/LL hardware flags.

---

## TODO #4 — Flag Array Shape, Reshape, and Fine-Grained Statistics ✅ IMPLEMENTED

**Implemented:** `ugmrt_query.materialise_flag_stats` (after `expand_flag_table_to_mask`);
called from `run_clustering.py` after `expand_flag_table_to_mask` on `vis_raw`.

**Files:** `ugmrt_query.py` (new function), `run_clustering.py` (detailed stats log).

### Flag array canonical shape

User-defined flag directives (wholesale antennas, baselines, timerange entries
with optional `chanrange`) each resolve to a set of `(row, channel)` cells.
Pols are **never** differentiated — the flag table has no per-corr field and
`FLAG_ALL_CORRS_IF_ANY_RAWVIS_FLAGGED=True` is the only supported mode.

**Canonical shape of the materialised flag array:**
```
(nRows, nChans)  →  bool
```
For this dataset: `(57456, 128)`.

This is then **broadcast across the pol axis** when applied to
`vis_complex` which is `(57456, 128, 2)`.

Do NOT give it a pol axis — doing so would silently imply per-corr
differentiation, which is unsupported (see TODO #3).

### Reshape for statistics

Because DUD antennas have zero rows and the active-baseline set is fully
regular (every integration contributes exactly `nActiveBaselines` rows in
the same order), the flat flag array can be losslessly reshaped to:

```
flags_3d = flags.reshape(nIntegrations, nActiveBaselines, nChans)  # bool
```

For this dataset: `(N_int, N_bl, 128)` where `N_bl` = observed unique
`(ant1,ant2)` pairs (see TODO #2).

### Fine-grained statistics via axis reductions

`flags_3d` is boolean: `True=1, False=0`. `.mean()` along any axis gives
the flagging *fraction* (multiply ×100 for percentage). All reductions are
O(1) numpy operations on the already-materialised array:

| Statistic | Expression | Output shape |
|---|---|---|
| % flagged per baseline | `flags_3d.mean(axis=(0,2)) * 100` | `(nActiveBaselines,)` |
| % flagged per channel | `flags_3d.mean(axis=(0,1)) * 100` | `(nChans,)` |
| % flagged per integration | `flags_3d.mean(axis=(1,2)) * 100` | `(nIntegrations,)` |
| total % flagged | `flags_3d.mean() * 100` | scalar |

**Per-antenna** requires a companion lookup table built once after loading:
```python
bl_ant_table  # shape (nActiveBaselines, 2)  — columns: ant1_idx, ant2_idx
# % flagged for antenna i:
bl_mask = (bl_ant_table[:, 0] == i) | (bl_ant_table[:, 1] == i)
pct_ant_i = flags_3d[:, bl_mask, :].mean() * 100
```

This replaces all current ad-hoc flagging-fraction counters scattered across
`run_clustering.py` and `preprocess_ugmrt.py` with a single materialised
boolean array plus the five reductions above.

---

## Key Principle

> A flag should only remove data at the **(baseline, channel, time)** coordinates
> where the data is actually bad. Wholesale antenna flags are appropriate only
> when the antenna is bad across the **full band and full time range**.
> Channel-selective badness must be expressed as `chanrange`-restricted
> time-range entries, not as all-band row deletions.
