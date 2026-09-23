# GWB Pipeline Rebuild — Plan

**Status line:** T0-T4 done (2026-09-23), Phase A (Foundation) complete. Phase B (T5) next.

## Objective

A reproducible pipeline that ingests raw GMRT GWB data and produces science outputs
(calibrated visibilities, Moon/3C468.1 imaging products), together with professional
book-keeping of every interim stage — data curation, calibration, flagging, imaging. This
is the north star every ticket below serves; if a design choice doesn't serve
reproducibility, book-keeping, or correctness, it's scope creep.

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
legacy paths.

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

- **T5 — Bandpass solve + iteration loop**, GWB channel range/paths/venv.
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

T0 → A (T1-T4) → B (T5-T8, the live pain point) → C → D → E → F → G → H, with a dev-test
checkpoint after each ticket. Each ticket gets its own commit(s); nothing merges to
`develop` without its test passing against real (or realistic fixture) data.
