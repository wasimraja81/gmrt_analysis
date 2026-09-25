# GWB Pipeline Rebuild — Plan

**Status line:** T0-T4, T5a, T5b, T5c done, Phase A complete (2026-09-24). 102 tests
passing. `bin/run_gwb_pipeline.sh` + the `build_index` stage ran against the real 389GB
GWB file (2026-09-25), producing a validated row index — see Phase C (T5c) for details
and the hardening that followed a run getting killed mid-scan. Two originally-scoped
sanity checks (frequency-band, expected-calibrators) were dropped after review found them
mis-scoped, not deferred — see Phase C for why.

Phases re-ordered 2026-09-25 around the pipeline's own high-level stages (raw data →
index → know your data → curation → cal solve → split-with-cal-applied → imaging); see
Phase B, added the same day, and the `## Sequencing` note below. Next: T19 (Phase B,
visPlot app — buildable now, no dependency on calibration existing first), or T5d (Phase
C, outer solve/diagnose/flag loop) — either is a reasonable next step; T19 doesn't block
on T5d or the reverse.

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

   **Hardened (2026-09-25), after a real run was killed mid-scan** (session teardown, not a
   code bug) **and left no trace at all** — no manifest, no run-index entry, just a
   one-line log fragment. `RunManifest` now writes a manifest immediately on `__enter__`
   (`outcome.status = "started"`), before any of the stage's own work runs, and overwrites
   the same file in place at `__exit__` with the final outcome — a manifest whose status is
   still `"started"` means the run never finished, instead of nothing existing at all. Both
   the manifest and `save_row_index`'s `.npz` output are now written atomically (temp file
   in the same directory, then `os.replace`, cleaned up on any exception) — a process
   killed mid-write leaves the previous complete file (or none), never a truncated one that
   looks present but isn't valid. The `build_index` stage is also idempotent by default: if
   an index already exists at its target path, the expensive full-file scan is skipped
   (still writes a fresh manifest recording the skip) — `build_index.force_rebuild: true`
   forces a rebuild.
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
   handle correctly for real GMRT data — see T5c).

   **A stricter reading of step (1), added 2026-09-23 after being caught reasoning too
   narrowly:** "check how the archived code did it" means checking every consumer of a
   shared layer before designing it, not just the one ticket currently in front of you.
   Designing T5c's vis-loading around only what the solver needs — cross-correlations —
   would have broken `plotVis.py`, which plots whatever the general loader returns and
   has no autocorrelation filtering anywhere in it; autocorrelations reach it today
   simply because nothing excludes them. Confirmed by reading `plotVis.py` directly, not
   assumed. Before designing a shared piece (a loader, an index, a config schema), list
   every place in `legacy_gsb_40_014/` that currently uses the equivalent thing, not just
   the one motivating the current ticket. See T5b for where this went further: the
   cached index used to check "does this dataset have autocorrelations" may itself have
   been built by code that drops them, which would make that check say nothing reliable
   about the raw file at all.
9. **Visibility data is indexed by antenna identity, never by assumed position.** No stage
   may assume every integration has the same antennas present, the same row count, or that
   baseline N always means the same antenna pair — an antenna can be dead for the whole
   observation, or (in principle, on a file this pipeline hasn't seen yet) drop out or come
   back mid-run. Every stage reads a row's own `ant1`/`ant2` and looks up by that identity,
   the way `solve_channel_gains` (T5a) already does by taking explicit `ant1_idx`/`ant2_idx`
   arrays rather than assuming a dense fixed layout. A stage that needs a fixed, dense,
   flagged-for-gaps shape — because it's handing data to an external tool that expects
   conventional UVFITS — builds that as an explicit, opt-in export step, never as this
   pipeline's internal default. Confirmed directly (2026-09-24) that GWB's own data doesn't
   need this in practice — all 10,476 integrations in the real file have the same 28
   antennas and the same 378-row length throughout — but the code must not rely on that
   holding for a different file.
10. **Re-evaluate design at each micro-stage, not just at ticket boundaries.** Added
    2026-09-25, naming a practice that was already happening rather than introducing a new
    one: the dead-this-observation redesign, the axis-selection genericity fix, and the
    provenance hardening after a real run was killed mid-scan all came from stepping back
    mid-stream, not from a ticket's own acceptance criteria. Before starting the next
    piece of work, check whether finishing the last one revealed something already built
    that should change — not only whether the next thing is ready to start.

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
legacy paths. `src/instruments/gmrt/` added later, T5c, per standing rule 8 (not part of
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

### Phase B — Know Your Data

**Added 2026-09-25, from a high-level re-framing of the pipeline's own stages** (raw data
→ index → know your data → curation → cal solve → split-with-cal-applied → imaging):
nothing in the original ticket list covered looking at the data *before* calibration
starts, even though `select_rows`/`read_visibility_data` already give everything this
needs — source list, UV-coverage, frequency sampling, integration counts, uncalibrated
amplitude/phase checks — with no dependency on Phase C (calibration) existing first.
Buildable now, ahead of Phase C.

- **T19 — visPlot app: meaningful defaults + interactive exploration.** Reuse and improve
  the archived `legacy_gsb_40_014/src/plotVis.py` (1492 lines, "generalized plotVis
  utility ... single multi-panel figure with selectable products and selectors") per
  standing rule 8 — this is not written from scratch; check what it already does and why
  before designing its replacement. Requirements from the user (2026-09-25), fine-grained
  design deferred to its own conversation:
  - A meaningful default set of plots/products generated with no configuration, so a
    first look at any observation needs no setup.
  - An interactive "scratch phase": the user explores the data live with the tool, then,
    once satisfied, asks for specific additional views on top of the defaults (zoomed
    selections, particular filters) rather than needing to script that from the start.
  - Antenna layout plotting.
  - Source listing with positions and other attributes.
  - HA range plotting.
  - Az/El range plotting.

### Phase C — Primary Calibration (3C48)

- **T5a — Core per-channel gain solve (generic, `src/engine/bandpass_solve.py`) — DONE
  (2026-09-23).** StefCal-style alternating-least-squares antenna gain solve, per channel.
  Telescope-agnostic: takes visibility/model/antenna-index arrays, returns gains — no
  GMRT-specific assumptions (antenna count, names, FITS structure) anywhere in it.
  Correctness gate: a synthetic recovery test (inject known gains, corrupt visibilities,
  verify the solver recovers them), not just "runs on real data without crashing" — this
  caught a derivation error (a conjugate placed wrong in one update term) before it
  went anywhere. Two preconditions on its input, both enforced or documented rather than
  assumed:
  - **Cross-correlations only.** `solve_channel_gains` solves `V_ab = g_a·conj(g_b)·M_ab`
    — a cross-correlation equation. An autocorrelation (`ant1_idx[k] == ant2_idx[k]`)
    doesn't fit this equation: it measures total power, with no phase information. If the
    function is ever given one, it raises an error immediately, since including it would
    silently produce wrong gains.

    This filtering happens late, just before the solve — not in general data loading.
    Checked directly in the archived engine: its general loader, `load_vis_for_source`,
    filters by antenna/time-range/UV-distance/elevation, nothing about autocorrelations
    at all. The cross-only filter lives in a separate, later function,
    `_build_channel_visibility_matrix` (`ugmrt_query.py:3547`), which is called
    specifically to prepare input for the solve. T5c should follow the same split: load
    everything for a source (cross and auto together, if autocorrelations exist), and
    filter to cross-only only when preparing input for `solve_channel_gains`. This keeps
    autocorrelation data available for whatever else needs it — including applying the
    solved gains to it afterward (a separate, later step, Phase D, not built yet — the
    gains are per-antenna, so they apply to autocorrelation visibilities too, even though
    they weren't solved from them).

    Unrelated to any of this: checking the two cached index files for this observation on
    2026-09-23 showed neither file's data contains any autocorrelation rows.
  - **An antenna with zero baselines in a call keeps its untouched initial gain**, never
    solved or phase-rotated. This is a robustness fallback for whatever hands it a
    zero-baseline antenna — it is *not* DUD-antenna handling by itself. DUD antennas
    must be excluded from the antenna list before `n_ant` or any baseline count is
    computed anywhere in the pipeline (see T5c) — every independent piece of code that
    computes a baseline-count denominator (flagging percentages, coverage stats, etc.)
    would otherwise be wrong on its own, not just the solve.
  5 tests in `tests/test_engine_bandpass_solve.py`. No user-guide section yet (rule 7):
  this is a library function with no directly observable output of its own until T5d/a
  `bin/` script wires it into something runnable — adding one now would describe nothing
  a reader could go do.
- **T5b — Correlation-type (auto vs. cross) audit across the archived pipeline — DONE
  (2026-09-24).** For every archived function/script that operates on baseline-level
  visibility data, what it does with correlation type today, checked by reading the code
  directly, with a file:line citation for each:

  | Function/script | What it does today | What T5c should do |
  |---|---|---|
  | `load_vis_for_source`, general loader (`ugmrt_query.py:2217`) | Filters by antenna/time/UV-distance/elevation only. Both correlation types pass through if present. | Same: a general loader, filtering on those same terms only. |
  | `_build_channel_visibility_matrix`, solve-prep (`ugmrt_query.py:3529-3577`) | Takes an `ignore_autos` parameter (default `True`) that filters `ant1 != ant2` when set. Threaded through `derive_point_source_bandpass` (:3703) and the outer workflow (:5339); every call site in the production chain leaves it at the default. | Filter to cross-only immediately before calling `solve_channel_gains`, and only there. |
  | `build_row_index` (`ugmrt_query.py:222-389`) + `_decode_baseline_array` (:213-219) | Indexes every one of `gcount` rows unconditionally, decoding `ant1`/`ant2` for all of them. | Same: index everything. This also settles the earlier open question — the "zero autocorrelation rows in the cached GWB/GSB index" finding is now established as a fact about the raw data itself, since the code that built that cache never filters anything out. |
  | `visSplit.py` (`:304-355`) | Passes `ant1`/`ant2` straight from the index to the output UVFITS file. | Same, so autocorrelation data survives into split output for later use. |
  | `outlier_detection.py` / `run_clustering.py` | Baseline/antenna selection queries operate on whatever rows are present, with no correlation-type filter. | Same. |
  | `plotVis.py` | Has no correlation-type filter anywhere in the file — confirms the concern that started this ticket. | Same. |

  One consistent pattern across the whole archived pipeline: every consumer except the
  solve itself is correlation-type-agnostic; only the stefCal solve excludes
  autocorrelations, applied right at the point of calling it. T5c's design (drafted
  under T5a before this audit ran) already matches that shape — a general loader, with
  the cross-only filter applied immediately before `solve_channel_gains` and nowhere
  else — now confirmed against the whole pipeline.

  This ticket's correctness gate is the file:line citations above, each checked directly
  against the code. Its output is the audit finding itself; there is no test file for it.

  Resolved (2026-09-24), checked directly against the row index: 378 rows/integration
  is not scan-specific, it holds for every integration of every one of the 13 sources
  in this file, and no autocorrelations are recorded anywhere in it — 378 = C(28,2),
  cross-only, 28 antennas. The two antennas that never appear in `ant1`/`ant2` anywhere
  in the file are station 4 (`C03:04`) and station 10 (`C10:10`) — not the `C07`/`S05`
  pair the pipeline's DUD config carries, and this file's AN table has no antenna named
  `C07` or `S05` at all (the C-arm sequence in it runs ...C06, C08... with no C07).
  Resolved (2026-09-24, confirmed by the user): not a naming difference, not a DUD at
  all in the structural sense. `C03:04`/`C10:10` were dead for this specific
  observation only — valid AN-table positions, will have real data again once
  repaired — and were deliberately omitted from the raw file at the correlator to
  reduce data size, distinct from `C07`/`S05`'s permanent, structural AN-table
  placeholder-position quirk. Two exclusion concepts now kept separate rather than
  conflated into one DUD list — see T5c below.
- **T5c — GMRT data adapter (`src/instruments/gmrt/`) — DONE (2026-09-24).** The
  index/select/read stack, the GMRT orchestration layer, and every sanity check that
  survived scrutiny are built and tested (89 tests passing). Two items originally
  scoped here were dropped, not deferred — see below for why. Channel-range/frequency
  mapping needed no dedicated GMRT code in the end: it's fully self-contained in the
  UVFITS metadata already (`chan_freqs_hz`, generic), so any caller turns a frequency
  range into channel indices directly and passes them to `read_visibility_data`'s
  `axis_selection`. This is where GMRT-specific FITS-format knowledge belongs — not in
  `src/engine/`. Produces the vis/model/antenna-index arrays T5a's solver consumes.
  T5b's audit is done: the row-index builder loads everything unconditionally (no
  correlation-type filtering), matching every other consumer except the solve
  itself — that's the design followed here and in `select_rows`.

  **First slice done (2026-09-23): `src/instruments/gmrt/antenna_table.py`**
  (`read_antenna_table`, `resolve_active_antennas`). Matches a configured DUD name
  ("C07") against the table's "code:station" naming ("C07:08") by exact match or
  `"<name>:"` prefix — deliberately narrower than the archived engine's own matcher,
  which also accepted a bare substring prefix with no colon required (risking matching
  more than one antenna for a short name); an unmatched configured name is reported, not
  silently dropped. 8 tests, including real-data checks against **both** raw files, which
  surfaced something not previously known:
  - **GSB's AN table has 32 rows** — the 30 real antennas, plus two placeholder entries
    appended at the end, `C07:31` and `S05:32`, matching `FLAGGING_DESIGN_NOTES.md`'s
    description exactly (present in the table, zero data rows).
  - **GWB's AN table has only 30 rows** — its correlator doesn't emit placeholder
    entries for C07/S05 at all; they're simply absent, not present-with-zero-data.
    Resolving the same DUD list against it correctly reports both names as unmatched,
    which is the expected, benign outcome for GWB, not a stale-config warning sign the
    way an unmatched name would be for GSB.
  Same physical antennas, same DUD list, **same code with no GSB/GWB branching** —
  verified handling both table layouts correctly, not assumed. "DUD antennas are the
  same for both" (confirmed by the user) is true of which antennas are dead; it is not
  true that the two correlators represent that fact in the data the same way.

  **Second slice done (2026-09-24): `src/data_io/uvfits_group_params.py`,
  `src/data_io/row_index.py`, `src/data_io/row_selection.py`.** Telescope-agnostic, not
  GMRT-specific — the GMRT-specific pieces (antenna table, DUD resolution) stay in
  `src/instruments/gmrt/`, built on top of these.
  - `uvfits_group_params.py`: reads a random-groups file's parameter columns (BASELINE,
    SOURCE, UU/VV, DATE, ...) without touching the much larger visibility data. Reads in
    memory-bounded chunks (a fraction of the host's total RAM, default 20%, `os.sysconf`
    stdlib-only so this works on the Raspberry Pi target too) rather than one unbounded
    pass — the archived code's own equivalent pattern held the whole file's touched span
    resident (22GB+ on the real 389GB GWB file, confirmed directly), which this avoids.
    Verified against the real file: output matches the pre-existing cached index
    (`40_014_25jul2021_2.6s_gwb.index.npz`) exactly. An explicit bulk-`file.read()`
    alternative was built and benchmarked against the same file too, on the theory that
    one big sequential read might beat many small memmap-triggered page faults — it
    didn't: 2086.5s vs. the chunked memmap approach's 1764.9s, ~18% slower, with identical
    output. Kept the memmap approach; the bulk-read variant isn't part of the pipeline.
  - `row_index.py`: `RowIndex`/`build_row_index` — the one unavoidable full-file pass
    (every row's SOURCE/BASELINE lives only in that row, confirmed no `AIPS NX` scan
    table exists in either real file to shortcut it). Reads every data axis by its
    `CTYPE` label (`STOKES`, `FREQ`, ...), not by assumed position — the real GWB header
    has `NAXIS=7` (`COMPLEX, STOKES, FREQ, IF, RA, DEC`), not the 4 axes the
    random-groups convention is often described with; a fixed-position read (what the
    archived code also does, safely, only because RA/DEC are both length 1 in this data)
    would silently mis-locate STOKES/FREQ on a file where that isn't true. Also adds
    `integration_boundaries` (a new integration starts wherever SOURCE or DATE changes),
    for T5d's per-integration iteration.
  - `row_selection.py`: `select_rows()` — a generic "any subset of any number of
    sources" row selector, replacing the shape of the archived `load_vis_for_source`
    (which required exactly one source as its entry point, confirmed still the shared
    engine behind `cal_solver.py`, `cal_apply.py`, `plotVis.py`, `preprocess_ugmrt.py` —
    all its filters, including `uvrange`/`elevation`, are real production requirements,
    not just diagnostic conveniences). `sources=None` selects from the whole file.
    `correlation_type` defaults to `"cross"` per the pipeline-wide default. No `max_rows`
    silent downsampling — `every_nth`/`random_subset_n` are explicit, named, and
    `random_subset_n` requires a `random_seed` (no seed-less path exists). Cheap and
    file-I/O-free: works purely from `RowIndex`'s in-memory arrays; the row bytes
    themselves are read separately, by `read_visibility_data`.
  - `visibility_data.py`: `read_visibility_data()` reads the actual visibility bytes for
    a `select_rows()` result. Every axis except `COMPLEX` (mandatorily decoded into
    real/imag/weight — that's what the FITS convention defines that axis to mean) is
    handled by one symmetric mechanism, selectable by CTYPE name via `axis_selection`,
    defaulting to "select everything" when not named — an early version instead hardcoded
    "only COMPLEX/STOKES/FREQ have selection logic, anything else must have length 1",
    which would have raised on any file with a genuine multi-IF or multi-pointing axis;
    corrected after review caught it. Verified against three properties directly, not
    assumed: (1) known per-row/channel/Stokes values recovered exactly; (2) a second,
    unrelated axis (`RA`, not `IF`) with length > 1 handled by the same code path, proving
    the fix isn't secretly IF-specific; (3) shuffled axis order (`FREQ` before `STOKES`,
    reversed from this file's own order) and a nonstandard CTYPE name both read correctly
    — found by label via `find_axis`, never assumed position, anywhere.
  - Real-data finding, checked directly against the built-and-verified index: every one
    of this file's 10,476 integrations, across all 13 sources, has exactly 378 rows and
    the same 28 active antennas — no variation anywhere in the file. Resolves the "worth
    a look later" note below. The two antennas absent from every row are station 4
    (`C03:04`) and station 10 (`C10:10`) — not the `C07`/`S05` pair the DUD config
    carries (see the note below for the open question this raises).

  **Third slice done (2026-09-24): `src/instruments/gmrt/row_index.py`,
  `src/instruments/gmrt/sanity_checks.py`.** `build_gmrt_row_index()` composes the
  generic pieces above with `antenna_table.py`'s DUD resolution, unmodified — no
  reimplementation of scanning, selecting, or reading. Adds the sanity checks that
  survived scrutiny (see the revised list below): `TELESCOP == 'GMRT'` and
  antenna-count plausibility run first, cheap and header-only, before the expensive
  full-file scan starts; the DUD cross-check and row-count consistency run once the
  index is built. `strict=True` (default) raises `GmrtDudConfigMismatch` or
  `GmrtRowCountMismatch` rather than letting either pass silently; `strict=False`
  proceeds anyway with both recorded in the returned `GmrtAntennaResolution`.
  Confirmed against the real file (via the already-verified saved index, no need to
  re-run the full scan): with neither exclusion list covering them, the check reports
  precisely `C03:04` and `C10:10` as the mismatch; with
  `dead_this_observation_names=["C03","C10"]`, row-count consistency reports a clean
  match — 378 rows/integration for 28 active antennas, exactly as predicted. Returns
  the plain, unmodified `RowIndex` — downstream code calls
  `select_rows`/`read_visibility_data` directly on it, no GMRT-specific wrapper needed
  for either. 17 new tests (`test_instruments_gmrt_row_index.py`,
  `test_instruments_gmrt_sanity_checks.py`).

  **Revised (2026-09-24), after the user corrected a conflation:** the single
  `dud_names` parameter was wrong — it conflated two genuinely different things.
  `antenna_table.GMRT_STRUCTURAL_DUD_NAMES` (`C07`, `S05`, hardcoded, not a caller
  parameter) is a permanent fact about GMRT's AN-table conventions: these two carry
  invalid/placeholder positions in GSB's table regardless of which observation is
  being read, so excluding them can't depend on a per-run argument. `C03:04`/`C10:10`
  are not that — they have valid AN-table positions, were merely dead for *this*
  observation (deliberately omitted from the raw file at the correlator to reduce
  data size), and will carry real data again in a future file once repaired. Treating
  them as permanent DUDs would have been wrong the same way a stale config is wrong.
  `build_gmrt_row_index()` initially took `dead_this_observation_names` as a caller
  parameter, separate from `structural_dud_names`. Corrected again the same day, at
  the user's prompting: requiring the caller to name the dead-this-observation
  antennas up front just recreated the exact stale-config risk that made `C07`/`S05`
  wrong for GWB in the first place. Once the row index is built, which nominal,
  non-structural-DUD antennas have zero rows is already known — nothing to predict in
  advance. The parameter is gone; `dead_this_observation_antennas` is now derived,
  not supplied, and `GmrtDudConfigMismatch` (which checked a caller's list against
  reality) is gone with it — there's no longer a list that could disagree with
  reality. `GmrtRowCountMismatch` remains: given the *derived* active set, every
  integration's row count is still checked against the dense-all-pairs prediction,
  now a check with nothing to go stale rather than one comparing two independently
  fallible sources.

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
    integration, if this observation's raw output includes autocorrelations at all.
    T5b settles that; don't use this formula against real row counts until it does.

  **Sanity checks — built (2026-09-24), in `sanity_checks.py`** (load-bearing for T5c's
  own correctness — if antenna/DUD resolution is wrong, everything downstream is
  silently wrong too):
  - `check_telescope_is_gmrt`: `TELESCOP == 'GMRT'` in the primary header (catches
    pointing the pipeline at the wrong file).
  - `check_antenna_count_is_plausible`: rejects zero antennas, and rejects more than
    the AIPS baseline encoding could ever represent (2047 — the format's own ceiling,
    not a GMRT-specific number; see `decode_baseline`).
  - `check_row_count_consistency` (the DUD-antenna cross-check `FLAGGING_DESIGN_NOTES.md`
    specified, made executable, plus the row-count formula check): after excluding the
    configured DUD list, the *observed* active-baseline set must match the *expected*
    one — fails if some antenna not on the DUD list still has zero data, and separately
    reports (without assuming it must hold, per standing rule 9) whether every
    integration's row count matches what the resolved active-antenna count predicts.

  **Two items dropped from this ticket, not deferred** — examined properly rather than
  left as "needs more info," and found to be mis-scoped from the start:
  - *"Frequency axis falls inside the expected band for the active profile (GSB vs
    GWB)"* — a category error. GSB/GWB are correlator backends, not frequency bands;
    GWB alone can sit anywhere from ~100 MHz to ~1800 MHz depending on which GMRT
    receiver was selected for a given observation, so there is no single "expected GWB
    band" to check a file against. The frequency axis itself is already fully
    self-contained in the UVFITS metadata (`chan_freqs_hz`, generic, CTYPE-driven) —
    there's nothing external to validate it against at this layer.
  - *"Expected calibrators/targets present in the AIPS SU table"* — assumed a
    per-observation reference (an observing proposal, a source-list config) that
    doesn't exist anywhere in this pipeline's design; the archived config's
    `SOURCE='3C48'` is one calibrator for one processing run, not a manifest of an
    observation's intended source list. Neither frequency range nor calibrator choice
    stops mattering for analysis — they just aren't unit-testable properties of a raw
    UVFITS file at the data-adapter layer; if either belongs anywhere, it's at the
    observing-proposal or per-run-config level, scoped properly if it's ever built.

- **Pipeline driver (`bin/run_gwb_pipeline.sh`, `src/cli/pipeline_stages.py`) — first
  stage runnable, 2026-09-25.** A fixed, ordered list of stages (`STAGE_ORDER`), each
  independently switched on/off by a per-observation YAML config's `stages:` block
  (`config/40_014_25jul2021_gwb.yaml`). Adding a future stage (split, primary
  calibration, ...) means writing one function and adding one line to the list, not
  restructuring the driver. Stage 1, `build_index`, wraps `build_gmrt_row_index` +
  `save_row_index` in exactly the existing `RunManifest`/logging machinery — no new
  provenance mechanism, just the first real caller. This is also what makes T5a/T5c's
  antenna resolution and indexing genuinely runnable for the first time, so the
  user-guide deferral noted under T5a ("no directly observable output until a `bin/`
  script wires it into something runnable") is now due.
  Index files always live adjacent to the raw file they index (`<fits_path>.idx.npz`,
  confirmed with the user as the standing rule, regardless of telescope), never under
  `work_dir`. `work_dir` itself is deliberately not on `/data1` (the raw data disk,
  confirmed spinning, not SSD) — it's `/scratch/gmrt/40_014_25JUL2021/work_gwb/`,
  mirroring the raw data's directory structure on the fast disk instead.

- **T5d — Outer iteration loop** (solve → diagnose → propose flags → re-solve), wrapping
  T5a via T5c's data adapter. Was originally scoped together with T5a as one ticket;
  split out so the core solve could be tested and verified on its own first.

  **Design requirement, decided 2026-09-24, before this ticket starts:** T5d must call
  `solve_channel_gains` with `n_ant = len(nominal_antennas)` (the fixed 30, from the AN
  table) and build `ant1_idx`/`ant2_idx` from raw station number
  (`station_number - 1`) — never `len(active_antennas)` with a compacted index. Verified
  directly against `solve_channel_gains`'s own code: an antenna with zero baselines
  (`has_any_data[i] = False`) is left at its untouched initial gain (`1+0j`) for the
  whole solve — never divided by zero, never phase-rotated — so this already gives
  exactly the "look for all 30, treat not-found as flagged" behavior agreed earlier,
  with no new mechanism needed. A compacted index would work for one observation but
  make antenna index position mean a different physical antenna in a different
  observation if a different antenna happened to be dead that day — the exact failure
  mode standing rule 9 exists to prevent.
- **T5e — Data-content sanity checks (`src/data_io/`, GMRT-specific thresholds supplied
  by `src/instruments/gmrt/`) — scoped now, not deferred silently, after being raised
  2026-09-23.** Broader statistical checks deliberately left out of T5c's scope so T5c
  isn't gold-plated before it has a first working version, but tracked here rather than
  left as a verbal intention:
  - No all-NaN/all-zero visibility blocks (a corrupted or truncated read).
  - Weight column has some nonzero values (there's unflagged data to work with).
  - UV coverage isn't degenerate — baselines aren't all sitting at zero (an
    antenna-position/geometry bug).
  Generic structural checks (`GroupsHDU` present, `AN`/`FQ`/`SU` tables present,
  `GCOUNT`/`PCOUNT`/`NAXIS` internally consistent) belong in `src/data_io/` too, but are
  basic enough to fold into T2's existing module rather than warrant their own ticket —
  add them there when first needed, not necessarily as part of T5e.
- **T6 — Two distinct threshold knobs, named so they can't be confused**: the coarse
  per-iteration wholesale antenna/baseline flagging threshold, and the per-channel
  clustering threshold, as two differently named config keys.
- **T7 — Final-solution naming.** The clustering-refined bandpass becomes a first-class
  named artifact, not a filename-suffix convention only the caller has to know about.
- **T8 — GWB threshold validation.** Settle and record (with reasoning, not just a number)
  the working per-channel clustering threshold for GWB.

### Phase D — Primary Transfer & Split

**Design requirement for both tickets below, decided 2026-09-24:** split and plot each
take an explicit correlation-type option (cross, auto, or both), defaulting to cross.
Selecting autocorrelations is a user choice, always available, never something to
reconstruct by enumerating antenna pairs by hand the way the archived `plotVis.py`
required — see T5b's audit, which found neither archived tool has a selector for this.

- **T9 — 3C48 split + diagnostics.**
- **T10 — 3C468.1 primary-only split + diagnostics.**

### Phase E — Advanced Flagging (Clustering, 3C468.1)

- **T11 — Full six-section flag-table clustering.**

### Phase F — Secondary Calibration

- **T12 — Phase-only/delay-phase secondary solve + final split.**

### Phase G — Target (Moon) Processing

- **T13 — Split/plot moon scans** (raw/primary/primary+secondary variants).
- **T14 — Selfcal imaging** (phasecentre / no-phasecentre).
- **T15 — Post-selfcal artifacts** (destriping, stacking, movies, cumulative RMS).

### Phase H — 3C468.1 Self-cal Imaging

- **T16 — Per-scan selfcal imaging.**
- **T17 — Post-selfcal stacking/movies/flag diagnostics.**

### Phase I — Publishing

- **T18 — Redesigned gh-pages "Provenance & reproducibility" section**, surfacing the
  manifest/run-index from Phase A, keeping the rest of the existing section taxonomy
  intact.

## Sequencing

T0 → A (T1-T4) → B (T19, know your data) → C (T5a-T5e, T6-T8, the live pain point) → D → E
→ F → G → H → I, with a dev-test checkpoint after each ticket. Each ticket gets its own
commit(s); nothing merges to `develop` without its test passing against real (or
realistic fixture) data.
