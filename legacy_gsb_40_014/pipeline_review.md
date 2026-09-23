# uGMRT Pipeline — Comprehensive Review

**Branch:** `40_014`  
**Date:** 2025-07  
**Files reviewed:** `ugmrt_query.py` (6234 L), `outlier_detection.py` (1974 L),
`preprocess_ugmrt.py` (1171 L), `run_clustering.py` (529 L),
`run_bandpass.sh` (381 L), `run_preprocess.sh` (58 L)

---

## 1. Flag Table Schema and Terminology

The codebase uses a **v2 JSON flag table** with six top-level arrays.

| Key | Meaning | Scope |
|-----|---------|-------|
| `bad_antennas` | List of antenna names flagged for the entire obs | All times, all chans |
| `bad_baselines` | List of `[ant1, ant2]` pairs | All times, all chans |
| `bad_antenna_timeranges` | `{ant_name: [{intervals: [[t0,t1],...], chanrange: [c0,c1]|null}, ...]}` | Selective time + chan |
| `bad_baseline_timeranges` | `{ant1::ant2: [{intervals: ..., chanrange: ...}, ...]}` | Selective time + chan |
| `bad_scan_timeranges` | List of `[t0, t1]` for whole-scan outages | All antennas, all chans |
| `bad_burst_timeranges` | List of `[t0, t1]` for global RFI bursts | All antennas, all chans |

**v1 back-compat rule** (both `_iter_entries` in `outlier_detection.py` and
`_iter_entries` in `ugmrt_query.expand_flag_table_to_mask`): if a timerange
entry is a bare `[t0, t1]` list rather than a dict, it is treated as
`{intervals: [[t0,t1]], chanrange: null}`. Both decoders handle this correctly —
no v1/v2 compatibility gap.

---

## 2. Flag → Array Translation: Full Audit

Three distinct functions exist for converting a flag table into a visibility
mask. They differ significantly in completeness.

### 2a. `apply_flag_tables_to_vis` (`ugmrt_query.py:2711`)

**What it does:** Drops entire visibility *rows* for `bad_antennas` and
`bad_baselines` only. Returns a sub-selected `vis` dict with those rows
removed.

**What it ignores completely:**
- `bad_antenna_timeranges`
- `bad_baseline_timeranges`
- `bad_scan_timeranges`
- `bad_burst_timeranges`
- `chanrange` annotations on any entry

**Used by:**  
- `derive_point_source_bandpass` (the stefCal solve, `ugmrt_query.py:3400`)  
- `_run_final_clustering_step` in `preprocess_ugmrt.py:784`  
- `run_clustering.py` when loading vis before Phase 2 detection  

**Impact:** Any flag that is not a *global wholesale* antenna or baseline
exclusion is silently ignored during the bandpass solve and (by extension) all
downstream corrected-vis products derived from that solve.

Concretely: if clustering finds that baseline C06-C09 was bad only during a
10-minute RFI event (stored in `bad_baseline_timeranges`), **that event's rows
are still included in the stefCal solution**. The solution quality is degraded
to the extent those rows are not down-weighted by any other mechanism.

### 2b. `count_vis_rows_flagged` (`ugmrt_query.py:2840`)

Counts (does not drop) rows hit by all six flag types, including time-range
intervals. However it is **row-only** (no chanrange dimension). Used only for
reporting/diagnostics. Correct for what it does.

### 2c. `expand_flag_table_to_mask` (`ugmrt_query.py:2940`)

**What it does:** Returns a `(nrows, nchans)` boolean array. Processes all
six flag types including chanrange-selective time-range entries. This is the
authoritative full v2 decoder.

**Used by:** Only the AFTER-correction diagnostic plot in
`run_clustering.py:_vis_plot` (since commit `8646130`). Not used in any
solve path.

### Summary table

| Use site | Function | bad_antennas | bad_baselines | timeranges | burst/scan | chanrange |
|----------|----------|:---:|:---:|:---:|:---:|:---:|
| stefCal solve | `apply_flag_tables_to_vis` | ✅ | ✅ | ❌ | ❌ | ❌ |
| Final clustering vis load | `apply_flag_tables_to_vis` | ✅ | ✅ | ❌ | ❌ | ❌ |
| Phase 2 AFTER plot | `expand_flag_table_to_mask` | ✅ | ✅ | ✅ | ✅ | ✅ |
| Row-count reporting | `count_vis_rows_flagged` | ✅ | ✅ | ✅ | ✅ | ❌ (row-only) |

**Recommendation:** Either extend `apply_flag_tables_to_vis` with a
`apply_timeranges: bool = True` option that NaN-fills the corresponding rows
(rather than dropping them, since dropping changes the array shape), or add a
wrapper that calls `expand_flag_table_to_mask` and then sets
`vis['vis_complex'][mask] = np.nan + 0j` and `vis['weight'][mask] = 0` before
passing to the solver. The stefCal solver already handles `weight=0` rows
correctly (they contribute nothing to the normal equations).

---

## 3. Workflow Steps and Their Code Locations

The intended pipeline has 10 steps (i–x). Below is the actual code for each.

### Phase 1 — `run_preprocess.sh` → `preprocess_ugmrt.py`

| Step | Intent | Code |
|------|--------|------|
| i | stefCal bandpass solve | `step_2_bandpass` → `derive_bandpass_iteration` → `derive_point_source_bandpass` |
| ii | Verify via self-cal (corrected vis vs model + V=0) | `step_3_diagnostics` → `run_bandpass_diagnostics`; BEFORE/AFTER plots |
| iii | Identify outlier antennas/baselines | `step_4_propose_flags` → `propose_flag_updates_from_diagnostics` |
| iv | Iterate until convergence | `run_auto` loop or interactive `run_manual` |
| v | Expand: cluster bad vis channel-by-channel | `_run_final_clustering_step` (gated by `RUN_FINAL_CLUSTERING`) |
| vi | Record compact flag coords | `_run_final_clustering_step` calls `update_flag_table` writing wholesale flags |

### Phase 2 — `run_bandpass.sh --phase=audit` → `run_clustering.py`

| Step | Intent | Code |
|------|--------|------|
| vii | Translate to machine-readable flag arrays | BEFORE: `apply_flag_tables_to_vis` (row-only); AFTER: `expand_flag_table_to_mask` (2D) |
| viii | Apply advanced flags and check plots | `run_clustering.py:_vis_plot` with BEFORE/AFTER panels |
| **ix** | **Re-derive bandpass with advanced flags applied** | **`--refit` flag in `run_clustering.py`** (see §4) |
| **x** | **Save re-derived solution as canonical `BANDPASS_OUT`** | **MISSING** (see §4) |

---

## 4. Missing Step (x): Final Bandpass Re-derivation

### Current state

`run_clustering.py` has a `--refit` flag. When used, it calls
`derive_bandpass_iteration` with the clustering flag table appended to
`flag_table=`. This re-derives the bandpass. However:

1. **`--refit` is not the default.** The normal `--phase=audit` path in
   `run_bandpass.sh` does not pass `--refit`, so it is dead code for the
   standard workflow.

2. **The re-derived solution is NOT saved as `BANDPASS_OUT`.** The `--refit`
   path in `run_clustering.py` saves to a separate path (or an in-memory dict
   that is discarded). The canonical `BANDPASS_OUT` still contains the Phase 1
   convergence solution, which was derived without the clustering flags.

3. **Even with `--refit`, `apply_flag_tables_to_vis` is used** (see §2a), so
   time-range flags from the clustering result still do not feed into the new
   solve.

4. **`_run_final_clustering_step`** (Phase 1's embedded clustering step,
   `preprocess_ugmrt.py:725`) does run a "final bandpass solve" tagged
   `final_clustering`. But it persists only the `bad_antennas` and
   `bad_baselines` from the clustering result to `FLAG_TABLE_SESSION` —
   discarding all `bad_antenna_timeranges`, `bad_baseline_timeranges`,
   `bad_scan_timeranges`, and `bad_burst_timeranges`. The per-channel
   chanrange-annotated flags produced by `run_clustering_detection` are thrown
   away before persistence.

### What is needed

After Phase 2 audit flags are approved (written to `FLAG_TABLE_SESSION`), a
clean re-solve should:

1. Load vis with `FLAG_TABLE_PATHS` + `FLAG_TABLE_SESSION` active.
2. Apply all flag types (not just wholesale) by NaN-filling plus weight=0 on
   flagged cells before the solve.
3. Run `derive_point_source_bandpass`.
4. Save the result as (or overwrite) `BANDPASS_OUT`.
5. Save a plot alongside with a `_final` tag so the before/after is visible.

The simplest implementation is a new `--phase=rederive-final` phase in
`run_bandpass.sh` that calls `run_preprocess.sh` with a dedicated step tag and
with all current session flags already on disk.

---

## 5. Double-Apply Risk: `FLAG_TABLE_PATHS` vs `FLAG_TABLE_SESSION`

`_rederive_source_paths` (line 239) constructs:

```python
FLAG_TABLE_PATHS = [p for p in [FLAG_TABLE_BASE, FLAG_TABLE_SESSION] if Path(p).exists()]
```

So after the session file is created, both base and session flags are always
loaded together. This means approved session flags are included in every
subsequent Phase 1 bandpass solve — which is correct and intentional.

**Potential double-apply:** `step_2_bandpass` passes `flag_table_path=FLAG_TABLE_PATHS`.
If `FLAG_TABLE_SESSION` is already in `FLAG_TABLE_PATHS` and the caller also
passes `flag_tables_extra=[session_flag_table]`, the session flags would be
applied twice. Looking at the actual call sites:

- `step_2_bandpass` does not pass any `flag_tables_extra` unless the caller
  supplies it — and all actual callers (`run_single_iteration`, `run_auto`) do
  not. **No double-apply in the normal path.**

- `_run_final_clustering_step` constructs `active_disk` from `FLAG_TABLE_PATHS`
  at runtime and separately appends `session_path` only if it is not already in
  the list. **No double-apply here either.**

The logic is correct. It is, however, easy to accidentally double-apply if a
future caller passes both `FLAG_TABLE_PATHS` (which already contains the session
file) and the session dict directly. Adding a deduplication doc-comment to
`step_2_bandpass` would help future maintainers.

---

## 6. `_run_final_clustering_step` Flags Truncation

When Phase 1's embedded clustering step writes new flags, only wholesaleantennas and baselines are persisted:

```python
# preprocess_ugmrt.py:820
if not dry_run and (new_ants or new_bases):
    q.update_flag_table(
        FLAG_TABLE_SESSION,
        add_antennas  = new_ants,
        add_baselines = new_bases,
        ...
    )
```

The `cluster_result['flag_table']` also contains:
- `bad_antenna_timeranges`
- `bad_baseline_timeranges`
- `bad_scan_timeranges`
- `bad_burst_timeranges`

None of these are passed to `update_flag_table`. They are silently discarded.

The Phase 2 standalone path (`run_clustering.py`) saves the full flag table via
`save_flag_table`, so Phase 2 preserves all six sections. Phase 1's inline
clustering is less complete.

**Recommendation:** Either pass the full flag table dict to a dedicated
`update_flag_table_from_flag_table(path, ft_dict)` helper that merges all six
sections, or at minimum log a warning that time-range and burst flags from
the embedded clustering step are not persisted.

---

## 7. Dual Code Paths: `outlier_detection.py::main()` vs `run_clustering.py`

Two separate CLI entry points perform Phase 2:

| | `outlier_detection.py --cal ...` | `run_clustering.py --fits ...` |
|-|----------------------------------|-------------------------------|
| Config | Global CONFIG block (edit file) | CLI args + `preprocess_ugmrt.cfg` loaded at runtime |
| Bandpass load | Reads `.npz` from `BANDPASS_NPZ` | Reads from `BANDPASS_OUT` path |
| Flag table input | `FLAG_TABLE_BASE` / `FLAG_TABLE_SESSION` from CONFIG | `--flag-table` CLI arg |
| Flag table output | `OUT_DIR/{source}_flag_table.json` | `FLAG_TABLE_SESSION` (merges) |
| Plot output | `OUT_DIR/{source}_*.png` | `WORK_DIR/{source}_clustering_*.png` |
| `--refit` | Not present | Present but incomplete (see §4) |

Both call the same `outlier_detection.run_clustering_detection()` library
function, so detection logic is shared. But the config surface, output
destinations, and flag-table merge behaviour diverge.

**Risk:** An operator running `outlier_detection.py` directly (likely during
interactive notebook exploration) would write flags to a different path than
`run_bandpass.sh --phase=audit` expects. Any flags written by the standalone
script are invisible to subsequent `--phase=derive` calls unless manually
copied to `FLAG_TABLE_SESSION`.

**Recommendation:** Deprecate `outlier_detection.py::main()` as a standalone
entry point and route all Phase 2 invocations through `run_bandpass.sh --phase=audit`.
Alternatively, add a `--flag-table-out` arg to `outlier_detection.py` that
defaults to the same path convention as `run_clustering.py`.

---

## 8. Robustness and Usability Gaps

### 8a. Silent timerange-flag loss in Phase 1 clustering (§6)

Severity: medium. The most precise flags (per-baseline burst intervals) are
quietly discarded. There is no log warning.

### 8b. Time-range flags ignored in all solve paths (§2a)

Severity: medium-high. After Phase 2 produces channel-selective time-range
flags, the next bandpass solve (whether triggered by `--refit` or manually)
still ignores them. Affected data is included in the solution with full weight.

### 8c. `FLAG_TABLE_PATHS` stale until script restart

`FLAG_TABLE_PATHS` is computed once by `_rederive_source_paths`. If
`FLAG_TABLE_SESSION` is created mid-run (e.g. first iteration), the cached
list won't include it until the next script invocation. In interactive
`run_manual` this could cause an iteration to run without session flags.
`run_auto` is not affected (it reinvokes the script). The `_rederive_source_paths`
function is called at startup only.

**Recommendation:** Call `_rederive_source_paths` at the start of each
`run_single_iteration` call.

### 8d. `--refit` dry-run default

`run_clustering.py` `--refit` defaults to `dry_run=True`. This is correct for
safety but means `--refit --no-dry-run` must be passed explicitly. The flag
is not wired through `run_bandpass.sh`, so there is currently no shell-level
way to trigger a real refit. The `_DERIVE_DRY_RUN_FLAG` is passed to
`run_preprocess.sh` but not to the audit path.

### 8e. `bad_burst_timeranges` never transferred to target sources

`outlier_detection.build_target_flag_table` (line ~1000) copies target flags
from calibrator flags but conditionally gates time-ranges on
`transfer_timeranges=True`. Burst timeranges are copied when this is True.
However the function is never called in `preprocess_ugmrt.py` or
`run_clustering.py` — there is no automated calibrator→target flag transfer
path in the pipeline scripts.

### 8f. `update_flag_table` creates v1-style session files

`q.update_flag_table` adds to `bad_antennas` / `bad_baselines` only. If the
session file was previously written by `save_flag_table` (which writes a full
v2 dict), `update_flag_table` must be checked to not overwrite the v2 fields
with empty defaults. This is low risk if the function merges rather than
replaces, but worth verifying.

### 8g. Logging verbosity in `_run_final_clustering_step`

`od.run_clustering_detection` always prints tabular progress to stdout
(`verbose=True`). In a batch run captured to a log file this is fine, but
there is no way to suppress it from `preprocess_ugmrt.py` (which uses the
`logging` module rather than print). Minor cosmetic issue.

---

## 9. Summary and Priority Actions

| Priority | Issue | Location | Action |
|----------|-------|----------|--------|
| 🔴 High | Time-range flags silently ignored in stefCal solve | `derive_point_source_bandpass` | NaN/weight=0 fill before solve via `expand_flag_table_to_mask` |
| 🔴 High | Step (x) missing: no path to save re-derived bandpass as canonical `BANDPASS_OUT` | `run_bandpass.sh`, `run_clustering.py` | Add `--phase=rederive-final` |
| 🟠 Medium | Phase 1 inline clustering discards all time-range and burst flags | `_run_final_clustering_step` line 820 | Persist full flag table, not just wholesale entries |
| 🟠 Medium | `FLAG_TABLE_PATHS` stale within an interactive run | `preprocess_ugmrt.py` line 239 | Refresh inside `run_single_iteration` |
| 🟡 Low | Dual CLI entry points may write flags to different paths | `outlier_detection.py::main` vs `run_clustering.py` | Deprecate standalone `main()`; route through `run_bandpass.sh` |
| 🟡 Low | `--refit` not wired into `run_bandpass.sh` and dry-run by default | `run_bandpass.sh`, `run_clustering.py` | Wire `_AUDIT_REFIT_FLAG` and `_AUDIT_DRY_RUN_FLAG` |
| 🟡 Low | No calibrator→target flag transfer in pipeline scripts | Both scripts | Call `build_target_flag_table` and save for each `TARGET_SOURCES` entry |
| ℹ️ Info | `count_vis_rows_flagged` is row-only (no chanrange), used in reporting | `ugmrt_query.py:2840` | Acceptable for totals; note in docstring that chanrange selective flags inflate estimate |

---

## 10. What is Working Well

- The v2 flag schema is well-designed and forward-compatible with v1.
- `expand_flag_table_to_mask` is a correct and complete 2D decoder.
- `cluster_flags` / `run_clustering_detection` / `_build_chan_aware_proposals`
  faithfully implement the Fortran UVFLG tiers with modern Python clustering.
- Per-channel `chanrange` annotations are correctly compressed into contiguous
  runs by `_build_chan_aware_proposals`.
- `format_casa_commands` exports all six flag types including `spw` selectors
  for channel ranges — CASA flagging is fully correct.
- The iterative Phase 1 loop, config override system (`--set`), and logging
  are solid.
- The five bug fixes in commits `44bed09`–`732d918` are all correct and
  well-targeted.
