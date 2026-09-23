# Unified Calibration Plan (Primary Bandpass + Secondary Phase/Delay)

## Goal
Use one solver and one applicator with mode-based behavior.

## Solver modes
- `primary_bandpass`
  - Derive complex gains per antenna and per channel.
  - Anchor amplitude scale using a source with trusted flux model (e.g. 3C48).
- `phase_only`
  - Derive time-dependent phase corrections from compact secondary calibrators.
  - Keep amplitudes fixed.
- `delay_phase`
  - Derive time-dependent phase-offset and delay terms.
  - Per-antenna model: `phi(t, nu) = phi0(t) + 2*pi*tau(t)*nu`.

## Canonical calibration table
All tables should carry enough metadata to be applied uniformly.

Required metadata:
- `kind`: `unified_calibration_table`
- `mode`: `primary_bandpass | phase_only | delay_phase`
- `source_name`
- `source_file`
- `reference_antenna`
- `antenna_ids`
- `antenna_names`
- `stokes_labels`
- `freqs_hz`
- `time_centers_jd`
- `time_intervals_jd`

Payload options:
- Full complex form:
  - `gains[ntime, nchan, nant, npol]` complex
  - `valid[ntime, nchan, nant, npol]` bool
- Compact delay/phase form:
  - `phi0_rad[ntime, nant, npol]`
  - `tau_s[ntime, nant, npol]`
  - Optional residual terms if needed later

## Applicator behavior
- Accept one or more tables in sequence.
- For each row/time in vis:
  - pick nearest or interpolated time solution
  - evaluate per-channel complex correction
  - apply chain in order (primary first, secondary next)

## Initial implementation milestones
1. Add mode-aware solver entrypoint (`cal_solver.py`).
2. Implement table writer/reader in `ugmrt_query.py` for unified table schema.
3. Add mode-aware applicator entrypoint (`cal_apply.py`).
4. Add `delay_phase` fit utility from solved channel phases.
5. Add validation plots (`phase residual`, `coherence ratio`).
