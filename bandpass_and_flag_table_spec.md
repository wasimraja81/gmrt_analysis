# uGMRT Bandpass and Flag Table Specification

This document describes the structure and usage of the non-destructive calibration products:
1. Bandpass solution table (`.npz`)
2. Flag table (`.json`)

All products are external sidecars. FITS files are never modified.

## 1. Bandpass Solution Table

### 1.1 File format
- Container: NumPy compressed archive (`np.savez_compressed`)
- Typical file: `work/3c48_bandpass_25jul_gsb.npz`
- Iteration-tagged output form: `work/3c48_bandpass_25jul_gsb_iterXX.npz`

### 1.2 Arrays stored in the table
- `freqs_hz`: float64, shape `(nchan,)`
  - Channel center frequencies used for solve and apply.
- `chan_indices`: int32, shape `(nchan,)`
  - Original FITS channel indices used in solve.
- `antenna_ids`: int32, shape `(nant,)`
  - Numeric antenna IDs participating in the solve.
- `antenna_names`: unicode, shape `(nant,)`
  - Human-readable antenna labels from AN table (for example `C11:11`).
- `stokes_labels`: unicode, shape `(npol,)`
  - Polarizations solved, typically `RR`, `LL`.
- `gains`: complex128, shape `(nchan, nant, npol)`
  - Per-channel complex antenna gains.
- `valid`: bool, shape `(nchan, nant, npol)`
  - Validity mask for each gain sample.
- `residual_rms`: float64, shape `(nchan, npol)`
  - Post-solve channel residual metric.
- `baseline_counts`: int32, shape `(nchan, npol)`
  - Number of surviving baselines per channel/pol in solve.
- `input_flagged_counts`: int32, shape `(nchan, npol)`
  - Input FITS-flagged sample counts by channel/pol.
- `used_sample_counts`: int32, shape `(nchan, npol)`
  - Samples effectively used in solve by channel/pol.
- `skipped_channel_mask`: bool, shape `(nchan, npol)`
  - Channels skipped due to insufficient usable data.
- `iterations`: int32, shape `(nchan, npol)`
  - Iterations used by solver.
- `flux_model_jy`: float64, shape `(nchan,)`
  - Flux model values used in normalization.
- `excluded_antenna_ids`: int32, shape `(n_ex_ant,)`
  - Antennas excluded via flag table during solve.
- `excluded_baseline_pairs`: int32, shape `(n_ex_base, 2)`
  - Baseline ID pairs excluded via flag table during solve.

### 1.3 Metadata stored in `metadata_json`
- `kind`: should be `point_source_bandpass`
- `source_name`, `source_file`
- `reference_antenna`
- `smooth_window`, `max_rows`, `min_baselines`
- `flag_table_applied` (bool)
- `flag_table_path` (string or null)
- `flag_table_paths` (list of strings)
- `flag_table_count` (int)
- `flag_table_notes` (string)
- `solve_input_rows` (int)
- `solve_dropped_rows_by_flag_table` (int)
- `bad_data_policy`
- `notes`

### 1.4 Dry-run solve behavior
In `derive_bandpass_iteration(...)`:
- `dry_run=True` computes and returns the full solution in memory.
- `dry_run=True` does not write the `.npz` bandpass table to disk.
- `bandpass_out` still returns the tagged target path string for traceability.

## 2. Flag Table

### 2.1 File format
- Container: JSON
- Typical file: `work/3c48_flag_table.json`

### 2.2 Required top-level structure
```json
{
  "kind": "ugmrt_flag_table",
  "version": 1,
  "bad_antennas": ["C11:11", "E04:17", 26],
  "bad_baselines": [
    ["E04:17", "W02:26"],
    {"ant1": "S02:21", "ant2": 26}
  ],
  "notes": "Reason for exclusions"
}
```

### 2.3 Antenna selectors accepted
- Exact antenna label from solution, for example `C11:11`
- Base tag label, for example `C11` (resolved by normalization)
- Numeric antenna ID, for example `11`

### 2.4 Baseline selectors accepted
- Two-item list/tuple: `["E04:17", "W02:26"]`
- Object: `{ "ant1": "E04:17", "ant2": "W02:26" }`
- Hyphenated string: `"E04:17-W02:26"`

Baselines are canonicalized to sorted `(min_id, max_id)` antenna-ID pairs.

## 3. Solve-Time Usage

`derive_point_source_bandpass(...)` supports:
- `flag_table_path=Path(...)` to load JSON from disk
- `flag_table_path=[Path(...), Path(...)]` to load and merge multiple JSON files on-the-fly
- `flag_table={...}` to pass table directly
- `flag_table=[{...}, {...}]` to pass and merge multiple in-memory tables
- `strict_flag_table=True` to fail on unresolved selectors

If one or more flag tables are provided, exclusions are merged on-the-fly and applied before gain solving.

The merged application records in solution metadata:
- `flag_table_count`
- `flag_table_paths`
- unioned `excluded_antenna_ids` and `excluded_baseline_pairs`

In notebook workflow, solve-time merged inputs can come from:
- On-disk list: `FLAG_TABLE_PATHS`
- In-memory list from previous dry-run iteration: `PENDING_FLAG_TABLES`
- These are merged in one solve call.

## 4. Diagnostics-Time Usage (New)

`run_bandpass_diagnostics(...)` supports optional merged flag-table filtering before computing residual statistics:
- `apply_flag_tables` (bool, default `False`)
- `flag_table_path` (single or multiple JSON paths)
- `flag_table` (single or multiple in-memory tables)
- `strict_flag_table` (bool)

When enabled, diagnostics reports:
- `diagnostics_flag_table_applied` (bool)
- `diagnostics_flag_table_count` (int)
- `diagnostics_flag_table_paths` (list)
- `diagnostics_dropped_rows_by_flag_table` (int)

This allows remnant-badness detection after currently known exclusions are applied.

## 5. Apply-Time Usage

`apply_bandpass_solution(...)` applies solved gains to visibilities in memory.
- Matching is by frequency value, not channel index.
- Full-band solutions can be applied to channel subsets if frequencies match.

For pre-apply visibility filtering in memory:
- `apply_flag_table_to_vis(...)` applies one table
- `apply_flag_tables_to_vis(...)` applies multiple tables with merged exclusions

## 6. Notebook Controls and Recommended Pattern

Current notebook controls:
- `USE_PENDING_FLAG_TABLES`: include previous dry-run proposals in next solve.
- `CLEAR_PENDING_FLAG_TABLES`: clear carried in-memory proposals.
- `DIAG_APPLY_FLAGS_ON_THE_FLY`: apply merged flag tables before diagnostics statistics.
- `DIAG_SAVE_UNFLAGGED_COMPARISON`: save additional unflagged diagnostics plot.

Iteration-tagged plot outputs are generated via `tagged_output_path(...)`.

## 7. Recommended Operational Pattern

1. Run diagnostics (for example Step 16) and identify outliers.
2. Update `work/3c48_flag_table.json` with bad antennas/baselines.
3. Re-run Step 12 to derive a new solution with exclusions applied.
4. Re-run validation steps (Step 14 to Step 16) and iterate.

Prefer antenna-level exclusions first when one antenna dominates many bad baselines. Use baseline-level exclusions for isolated pair-specific failures.
