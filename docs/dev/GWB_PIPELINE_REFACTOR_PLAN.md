# GWB Pipeline Rebuild — Plan

**Status line:** T0-T4, T5a done (2026-09-23), Phase A complete. T5b (GMRT data adapter,
`src/instruments/gmrt/`) next.

## Objective

A reproducible pipeline that ingests raw GMRT GWB data and produces science outputs
(calibrated visibilities, Moon/3C468.1 imaging products), together with professional
book-keeping of every interim stage — data curation, calibration, flagging, imaging. This
is the north star every ticket below serves; if a design choice doesn't serve
reproducibility, book-keeping, or correctness, it's scope creep.

The calibration/imaging algorithms themselves should be written generic enough to run on
any interferometric dataset, not hardcoded to GMRT — with GMRT's own quirks (see
standing rule 8) supplied as data/config to that generic core, not woven into it.

## Standing rules (apply to every ticket, no exceptions)

1. **Raw input data is never modified.** Always opened read-only. `src/data_io/` is the only
   place raw data is touched, and it enforces this centrally. Any derived-output path is
   checked against the raw input path before writing, enforced automatically inside
   `RunManifest.add_output`. This guarantee only holds if every engine module routes raw
   reads through `src/data_io/` — nothing currently stops a future stage from calling
   `astropy`/`numpy` directly instead. Each engine ticket (T5+) must confirm this at
   review time.
2. **Names are chosen deliberately, every time.** Even where a new file/function does
   exactly what its GSB-era equivalent did, its name is reviewed on its own merits — never
   inherited by habit from `legacy_gsb_40_014/`. See
   `[[feedback_unambiguous_naming]]`-style discipline: no generic/overloaded names — see
   the `CLUSTERING_V_THRESHOLD` vs `CLUSTERING_THRESHOLD_JY` confusion that caused a crash
   during GSB-era GWB tuning.
3. **No ticket is done without a passing test that verifies the stage's products.**
   Not "the script ran without error" — "the expected output files exist and are
   scientifically sane" (e.g. bandpass converged, flag counts are plausible, split file
   has the expected row/channel shape). Tests live in `tests/`.
4. **Every invocation is reproducible**, end-to-end and per-stage. A later run — by this
   user or a collaborator — must be able to reproduce exactly what an earlier run did, via
   the manifest `src/provenance/` writes (git state, resolved parameters, input identity),
   not by remembering what flags were passed.
5. **Dev-test-dev-test at the microscopic level.** One focused change, then verify, before
   the next. Don't batch untested changes across a ticket.
6. **Venv-only, no system installs.** All Python work runs inside `gmrt/` at the repo
   root; see `ENVIRONMENT_SETUP.md` for rebuild instructions and `config/requirements.txt`
   for the current dependency list.
7. **No ticket is done without its user-guide section written.** `../user/GWB_USER_GUIDE.md`
   gets a new section, written for the person running the pipeline (not the code), the
   same turn a ticket lands — what it does, what you get out of it, where to find the
   output. It must never fall behind what's built, the way `pipeline_review.md` fell
   behind the GSB-era engine.
8. **Generic core, GMRT specifics kept separate — a required process, not a one-off
   decision.** `src/engine/` holds telescope-agnostic algorithms: no hardcoded antenna
   counts, names, or FITS-format assumptions. GMRT-specific knowledge (antenna table
   quirks, DUD-antenna handling, channel-to-frequency mapping, calibrator naming
   conventions) lives in `src/instruments/gmrt/` and is supplied to the generic core as
   data, not baked into its logic. Before writing any new piece of engine/data-handling
   code: (1) check how the archived GSB code did it — grep `legacy_gsb_40_014/`, don't
   assume; (2) understand *why* it did it that way (read `FLAGGING_DESIGN_NOTES.md` and
   similar docs, not just the code); (3) classify what was found as a general
   interferometry concern (any telescope would need this) or a GMRT-specific quirk
   (learned the hard way, specific to this instrument/observation); (4) design so the
   generic core handles the general case and `src/instruments/gmrt/` supplies the
   specifics — never skip straight to writing code from general algorithmic knowledge
   alone, since that knowledge has no way of knowing what a real telescope's data
   does (concrete example: DUD antennas — GMRT antennas present in the antenna
   table but hardware-dead for this observation, contributing zero rows to the raw FITS
   file; not something derivable from any calibration textbook, and not optional to
   handle correctly for real GMRT data — see T5b).

## Reference material

- `legacy_gsb_40_014/ARCHIVE_INDEX.md` — categorized index of everything archived
  (dead / duplicate / superseded / orphaned-research / active-reference).
- `legacy_gsb_40_014/pipeline_review.md` — prior code review of the GSB-era engine; partly
  stale (two of its "High priority" findings are already fixed upstream — see this ticket
  doc's history in the 2026-09-22/23 conversation record for specifics before assuming a
  finding still applies).
- The published gh-pages report (`origin/gh-pages`, `40_014/latest/index.html`) — the
  collaborator-facing output taxonomy to keep aiming at: primary bandpass diagnostics,
  primary self-check, primary transfer check, clustering diagnostics, full calibration QA,
  per-scan Moon target QA, Moon imaging diagnostics, provenance & reproducibility, Moon
  post-selfcal imaging/stacking, per-scan 3C468.1 post-selfcal imaging, flag overview.

## Tickets

### T0 — Restructure: archive GSB legacy, stand up empty GWB tree — DONE (2026-09-23)

Moved all existing code/docs into `legacy_gsb_40_014/` intact (git mv, history preserved),
categorized in `ARCHIVE_INDEX.md`, and created the empty skeleton (`bin/`, `src/engine/`,
`src/data_io/`, `src/provenance/`, `src/cli/`, `config/`, `tests/`, `docs/dev/`,
`docs/user/`) with per-directory READMEs stating intent. `.gitignore` updated for the new
legacy paths. `src/instruments/gmrt/` added later, T5b, per standing rule 8 (not part of
the original T0 skeleton — the generic-core/instrument-specific split wasn't decided yet
at that point).

### Phase A — Foundation

- **T1 — Provenance/manifest library — DONE (2026-09-23).** `src/provenance/manifest.py`,
  `RunManifest`: per-invocation manifest with git commit + dirty-diff, resolved
  parameter set (not just overrides), `run_id`, input-data identity (path + size + mtime,
  not a checksum — see rationale in the user guide), outcome (success/failed +
  exception). Written automatically on every run via a context manager; `add_output`
  calls `data_io.guard_output_path` automatically (added 2026-09-23 after the user
  guide's raw-data-safety claim was checked and found not yet enforced — see
  T2). `manifest.logger` (wired in T3) is the stage's logger, sharing this run's
  `run_id` with its log file. 5 tests in `tests/test_provenance_manifest.py`.
  User-facing: `../user/GWB_USER_GUIDE.md#run-records-provenance`.
- **T2 — Read-only data-access layer — DONE (2026-09-23).** `src/data_io/raw_data_access.py`:
  `open_fits_readonly`/`open_raw_memmap` (no `mode` parameter exposed — a writable open
  isn't a code path that exists), `guard_output_path` (resolves symlinks/relative paths
  before comparing, raises `RawDataProtectionError` on a match). Named `data_io`, not
  `io`, to avoid shadowing the stdlib module. 9 tests in
  `tests/test_data_io_raw_data_access.py`, including one against the real 389GB GWB FITS
  file (skipped where that file isn't present). User-facing:
  `../user/GWB_USER_GUIDE.md#reading-raw-data-safely`.
- **T3 — Unified logging — DONE (2026-09-23).** `src/provenance/logging_setup.py`:
  `setup_stage_logger` (per-run file at DEBUG + console at a configurable level + a
  `<stage>_latest.log` symlink) and `close_logger`. Wired directly into
  `RunManifest.__init__`/`__exit__`, sharing the manifest's `run_id` as the log filename
  and logging the stage's success/failure automatically. This makes the "no bare
  `print()`" half of this ticket a property of using `RunManifest` at all for future
  engine code (T5+); it doesn't retroactively touch the archived GSB engine. 8 tests
  across `tests/test_provenance_logging_setup.py` and the `manifest_logger_*` tests in
  `tests/test_provenance_manifest.py`. User-facing:
  `../user/GWB_USER_GUIDE.md#finding-out-what-a-run-did-logs`.
- **T4 — Run index — DONE (2026-09-23).** `src/provenance/run_index.py`:
  `append_to_run_index`, one JSON-lines file at `<work_dir>/runs_index.jsonl` spanning
  every stage and run. Wired into `RunManifest.__exit__` (a compact entry: run_id, stage,
  timestamps, git commit/dirty, parameters, outcome status, plus pointers to the full
  manifest and log file — not a duplicate of the full manifest). Queried directly with
  `grep`/`jq`, no bespoke API. 4 tests (`tests/test_provenance_run_index.py` +
  `manifest_run_index_*` in `tests/test_provenance_manifest.py`). Closes Phase A.
  User-facing: `../user/GWB_USER_GUIDE.md#finding-past-runs-the-run-index`.

### Phase B — Primary Calibration (3C48)

- **T5a — Core per-channel gain solve (generic, `src/engine/bandpass_solve.py`) — DONE
  (2026-09-23).** StefCal-style alternating-least-squares antenna gain solve, per channel.
  Telescope-agnostic: takes visibility/model/antenna-index arrays, returns gains — no
  GMRT-specific assumptions (antenna count, names, FITS structure) anywhere in it.
  Correctness gate: a synthetic recovery test (inject known gains, corrupt visibilities,
  verify the solver recovers them), not just "runs on real data without crashing" — this
  caught a derivation error (a conjugate placed wrong in one update term) before it
  went anywhere. Two preconditions on its input, both enforced or documented rather than
  assumed:
  - **Cross-correlations only.** An autocorrelation (`ant1 == ant2`) is a different
    physical quantity (total power, no phase) that doesn't fit `V_ab = g_a·conj(g_b)·M_ab`
    at all — including one would silently corrupt the solve, not raise an error, so
    `solve_channel_gains` now raises `ValueError` if it finds one. Confirmed: the
    archived engine tracks autocorrelation row counts as a statistic separate from
    baseline counts (`ugmrt_query.py:791-793`), and its own solve path already filters
    them explicitly via `ant1 != ant2` (`_build_channel_visibility_matrix`,
    `ugmrt_query.py:3547`) — a previously-validated pattern, not a guess.
  - **An antenna with zero baselines in a call keeps its untouched initial gain**, never
    solved or phase-rotated. This is a robustness fallback for whatever hands it a
    zero-baseline antenna — it is *not* DUD-antenna handling by itself. DUD antennas
    must be excluded from the antenna list before `n_ant` or any baseline count is
    computed anywhere in the pipeline (see T5b) — every independent piece of code that
    computes a baseline-count denominator (flagging percentages, coverage stats, etc.)
    would otherwise be wrong on its own, not just the solve.
  5 tests in `tests/test_engine_bandpass_solve.py`. No user-guide section yet (rule 7):
  this is a library function with no directly observable output of its own until T5c/a
  `bin/` script wires it into something runnable — adding one now would describe nothing
  a reader could go do.
- **T5b — GMRT data adapter (`src/instruments/gmrt/`) — NEW, not yet started.** Antenna
  table reading, DUD-antenna resolution (GMRT antennas present in the antenna table but
  hardware-dead for this observation, contributing zero rows — see standing rule 8; same
  list confirmed by the user to apply to both GSB and GWB), row-index building, GWB
  channel-range/frequency mapping, vis loading for one source. This is where GMRT-specific
  FITS-format knowledge belongs — not in `src/engine/`. Produces the vis/model/
  antenna-index arrays T5a's solver consumes.

  **Design requirement, not optional:** the active (non-DUD) antenna list is resolved
  once, immediately after reading the antenna table, before anything else is computed
  from it — matching the archived code's own pattern (`n_ant = len(active_antennas)`
  computed *before* `cross_th = n_ant*(n_ant-1)//2`, `ugmrt_query.py:752-753`). Every
  downstream consumer (the solver, flagging statistics, coverage plots) uses this one
  resolved set — never independently re-derives "the antenna list" or "the baseline
  count" from the raw antenna table, which is exactly how one consumer excluding DUDs
  correctly and another not doing so would produce silently inconsistent statistics.

  **Two distinct counts, kept explicitly separate, not conflated:**
  - *Cross-correlation baselines*: `n_ant × (n_ant-1) / 2` — for anything about
    interferometric fringes: the solver's input, UV coverage, visibility-based flagging
    statistics.
  - *Total correlations including autocorrelations*: cross-baselines + `n_ant`
    autocorrelation rows — for anything about raw data volume/row counts per
    integration, since GMRT records both. Confirmed present in this dataset (see T5a's
    autocorrelation note above); needed for any expected-row-count check, not for
    anything solver-facing.

  **Sanity checks in scope for this ticket** (load-bearing for T5b's own correctness —
  if antenna/DUD resolution is wrong, everything downstream is silently wrong too):
  - `TELESCOP == 'GMRT'` in the primary header (catches pointing the pipeline at the
    wrong file).
  - Antenna count is plausible, not just present (0 or an absurd value fails loudly).
  - The DUD-antenna cross-check `FLAGGING_DESIGN_NOTES.md` already specified: after
    excluding the configured DUD list, the *observed* active-baseline set must match the
    *expected* one — fail if some antenna not on the DUD list still has zero data (a
    stale config or a hardware problem — either way, fail loudly rather than work
    around it silently).
  - Frequency axis (`AIPS FQ` table) falls inside the expected band for the active
    profile (GSB vs GWB) — catches a channel-range/profile mismatch immediately, not
    three stages downstream.
  - Expected calibrators/targets are present in the `AIPS SU` table by name.
  - Row-count consistency: total `GCOUNT` decomposes cleanly into
    `n_integrations × (cross-baselines + autocorrelations)` — the two-count distinction
    above, made executable.
- **T5c — Outer iteration loop** (solve → diagnose → propose flags → re-solve), wrapping
  T5a via T5b's data adapter. Was originally scoped together with T5a as one ticket;
  split out so the core solve could be tested and verified on its own first.
- **T5d — Data-content sanity checks (`src/data_io/`, GMRT-specific thresholds supplied
  by `src/instruments/gmrt/`) — scoped now, not deferred silently, after being raised
  2026-09-23.** Broader statistical checks deliberately left out of T5b's scope so T5b
  isn't gold-plated before it has a first working version, but tracked here rather than
  left as a verbal intention:
  - No all-NaN/all-zero visibility blocks (a corrupted or truncated read).
  - Weight column has some nonzero values (there's unflagged data to work with).
  - UV coverage isn't degenerate — baselines aren't all sitting at zero (an
    antenna-position/geometry bug).
  Generic structural checks (`GroupsHDU` present, `AN`/`FQ`/`SU` tables present,
  `GCOUNT`/`PCOUNT`/`NAXIS` internally consistent) belong in `src/data_io/` too, but are
  basic enough to fold into T2's existing module rather than warrant their own ticket —
  add them there when first needed, not necessarily as part of T5d.
- **T6 — Two distinct threshold knobs, named so they can't be confused**: the coarse
  per-iteration wholesale antenna/baseline flagging threshold, and the per-channel
  clustering threshold, as two differently named config keys.
- **T7 — Final-solution naming.** The clustering-refined bandpass becomes a first-class
  named artifact, not a filename-suffix convention only the caller has to know about.
- **T8 — GWB threshold validation.** Settle and record (with reasoning, not just a number)
  the working per-channel clustering threshold for GWB.

### Phase C — Primary Transfer & Split

- **T9 — 3C48 split + diagnostics.**
- **T10 — 3C468.1 primary-only split + diagnostics.**

### Phase D — Advanced Flagging (Clustering, 3C468.1)

- **T11 — Full six-section flag-table clustering.**

### Phase E — Secondary Calibration

- **T12 — Phase-only/delay-phase secondary solve + final split.**

### Phase F — Target (Moon) Processing

- **T13 — Split/plot moon scans** (raw/primary/primary+secondary variants).
- **T14 — Selfcal imaging** (phasecentre / no-phasecentre).
- **T15 — Post-selfcal artifacts** (destriping, stacking, movies, cumulative RMS).

### Phase G — 3C468.1 Self-cal Imaging

- **T16 — Per-scan selfcal imaging.**
- **T17 — Post-selfcal stacking/movies/flag diagnostics.**

### Phase H — Publishing

- **T18 — Redesigned gh-pages "Provenance & reproducibility" section**, surfacing the
  manifest/run-index from Phase A, keeping the rest of the existing section taxonomy
  intact.

## Sequencing

T0 → A (T1-T4) → B (T5a-T5d, T6-T8, the live pain point) → C → D → E → F → G → H, with a
dev-test checkpoint after each ticket. Each ticket gets its own commit(s); nothing merges to
`develop` without its test passing against real (or realistic fixture) data.
