# GWB Pipeline Rebuild — Plan

**Status line:** T0-T4, T5a, T5b, T5c done, Phase A complete (2026-09-24). Phase B: T19
and T22 (visPlot, streaming) done 2026-09-27; T23-T25, T27-T30, T34 done 2026-09-28, T31
2026-09-29, T36, T37, T39 and T41 2026-09-29, T33, T43 and T44 2026-09-30, T38 and
T47-T49 2026-10-01, 487 tests passing; T26, T32 in progress; T40, T42, T45, T46, point
D, T20, T21 open (order in Phase B); T35 (Moon scans' u, v, w) open, parked until the
Moon is imaged.
`bin/run_gwb_pipeline.sh` + the `build_index` stage ran against the archival 389GB GWB file
(2026-09-25), producing a validated row index — see Phase C (T5c) for details
and the hardening that followed a run getting killed mid-scan. Two originally-scoped
sanity checks (frequency-band, expected-calibrators) were dropped after review found them
mis-scoped, not deferred — see Phase C for why.

Phases re-ordered 2026-09-25 around the pipeline's own high-level stages (raw data →
index → know your data → curation → cal solve → split-with-cal-applied → imaging); see
Phase B, added the same day, and the `## Sequencing` note below.

T19 (Phase B, visPlot app) is done, and T22 rebuilt its plotting to stream the selection
through fixed pixel grids, so memory no longer depends on how much is selected (924
million samples plotted in 1.40 GB; 2026-09-27). T20 (observation metadata aggregator),
T21 (density mode) and T5d (Phase C, outer solve/diagnose/flag loop) are the open next
steps; none blocks another.

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
starts, even though `select_rows` and the visibility reader already give everything this
needs — source list, UV-coverage, frequency sampling, integration counts, uncalibrated
amplitude/phase checks — with no dependency on Phase C (calibration) existing first.
Buildable now, ahead of Phase C.

- **T19 — visPlot app: meaningful defaults + interactive exploration — DONE
  (started 2026-09-26, completed 2026-09-27).** Reuse and improve the archived
  `legacy_gsb_40_014/src/plotVis.py` (1492 lines, "generalized plotVis utility ... single
  multi-panel figure with selectable products and selectors") per standing rule 8 —
  checked directly before designing its replacement (its own "any panel, any product"
  design is the precedent for the generic Y-vs-X plot below, arrived at independently
  before rereading it). CLI-first per the user (2026-09-25): no Jupyter; a future Qt GUI
  reuses the same `src/visplot/` classes from its own widget callbacks.

  **Done:**
  - Foundation: `data_io/source_table.py` (AIPS SU table, including a corrected
    understanding of `CALCODE` as an EVLA-convention scan-intent code, not a VLA-specific
    resolved/unresolved flag), `data_io/astrometry.py` (hour angle, Az/El, parallactic
    angle, generic against any `EarthLocation`), `select_rows` extended with
    `ha_range_hours`/`az_range_deg`/`el_range_deg`/`parallactic_angle_range_deg` filters
    and exact (not reference-frequency-approximated) `u/v/w_range_klambda`/
    `uvdist_range_klambda` filters.
  - Fixed a real, previously undetected bug found while building the antenna layout plot:
    `read_antenna_table` was adding `STABXYZ` directly to `ARRAYX/Y/Z` as if both were in
    the same ECEF frame; AIPS Memo 117 defines `STABXYZ` in ECEF rotated to the array's own
    local meridian, and the fix (confirmed via independent geodetic height, and via web
    search of the AIPS memo) was necessary for every antenna position in the codebase, not
    just this plot.
  - Added the W coordinate (`ww_sec`) to `RowIndex`/`VisibilityBlock` — never read before.
  - `src/visplot/`: `antenna_layout` (density-based compact-core detection, not
    centroid-distance, since GMRT's own layout shows those disagree; a dynamically
    positioned, exact-collision-avoiding-label inset; a title with optional
    telescope/source-file provenance), `source_listing`, and a generic "pick two named
    quantities and plot them" engine (with per-category coloring, flag handling, and a
    `mirror` option for a measurement-and-its-conjugate pair like UV coverage) that
    replaced the two originally-built specific functions (`amplitude_phase`,
    `uv_coverage`). First built as `derived_quantities` + `scatter_xy`, with separate
    `hour_angle_range`/`az_el_range`/`parallactic_angle_range` functions; rebuilt as a
    streaming design in T22, where the geometry plots became presets of the generic
    plot.
  - `src/visplot/range_spec.py` (`"1:5,10,12:14"` set/range grammar, plus
    `astropy.units`-based conversion for genuine physical units — not kilo-wavelengths,
    which isn't one), `channel_selection.py` (index or frequency band, resolved against a
    file's real, not assumed-ordered, channel frequencies), `antenna_selection.py` (id or
    full name, mirroring `sources`'s own name-or-id), and
    `instruments/gmrt/antenna_selection.py` (a bare GMRT code prefix, e.g. "C00").
  - `src/visplot/time_range.py` (relative hours / absolute JD / ISO "start/end", resolved
    against the row index's own JD range).
  - `src/visplot/plot_title.py` — a shared title helper (source list, plus optional
    telescope/file provenance), wired into all three geometry-range plots so a
    single-source plot still names its source even with no legend to carry it.
  - `src/cli/visplot.py` (the CLI entry point, dispatched by `bin/visplot.sh`; renamed
    `run_visplot.py` in T30 so it no longer shares the `visplot` package's name) and
    `src/cli/visplot_args.py` (its pure, directly-testable argument resolution; moved to
    `src/visplot/request_args.py` in T32, shared with the GUI) — wiring
    every filter, resolver, and plot function above into one tool. Verified end to end
    against the real GWB file with a narrow, contiguous `3C286` selection; a naive
    selection that stacks a wide channel range and full Stokes axis onto a generic scatter
    plot is not fast (10206 rows x 4 pol x 2048 chan is 83 million points for matplotlib
    to render, independent of file I/O) — `--channels`/`--stokes` narrow that per plot.
  - Follow-ups after first use (2026-09-27): axis labels carry units, with
    amplitude/real/imag taken from the file's own `BUNIT`; titles without units; `-h`
    rewritten (every option described, defaults stated, a plot-names section, examples)
    and plot names checked before the file is opened; the kλ row filters now use the
    band selected by `--channels` (previously the whole band); saved PDFs draw points as
    an image (150 and 600 dpi), since vector points made a 520k-point page take ~20 s to
    render in poppler and MuPDF alike; dense plots get smaller, square markers;
    `read_visibility_data` reads in bounded chunks, so its memory follows the selection
    (before, it held the longest contiguous run of full rows). Memory still grew with
    the selection until T22.

  **Remaining:**
  - T20 (below): the AIPS FQ table and AN-table polarization columns, and DATE-OBS /
    per-source on-source time for richer plot titles and axis context.

- **T20 — Observation metadata aggregator ("listObs") — NOT STARTED (added 2026-09-27).**
  A stock-take of what a UVFITS file actually carries, against what this codebase reads,
  found: the AIPS FQ table (`FRQSEL`/`IF FREQ`/`CH WIDTH`/`TOTAL BANDWIDTH`/`SIDEBAND`) is
  never consulted at all — `chan_freqs_hz` is computed purely from the primary header's
  `CRVAL4`/`CDELT4`/`CRPIX4`, correct for the real GWB file only because it happens to have
  a single frequency setup (confirmed directly: its one FQ-table row's `CH WIDTH` exactly
  matches `CDELT4`) — a file with more than one frequency setup would be read wrong, since
  each row's own `FREQSEL` (also unread) is never used to pick the matching FQ entry; the
  AN table's polarization/feed columns (`POLTYA`/`POLAA`/`POLCALA`/`POLTYB`/`POLAB`/
  `POLCALB`, plus `MNTSTA`/`STAXOF`) are unread, directly relevant given this project's
  pol-cal interest; the SU table's `QUAL`/`FREQOFF`/`BANDWIDTH`/`LSRVEL`/`RESTFREQ`/
  `PMRA`/`PMDEC` are unread (all zero/trivial in the real GWB file, but real gaps); no
  integration-time header keyword exists in this file at all (the "2.6s" in its filename
  is not recorded metadata anywhere) — it has to be a derived quantity, from the row
  index's own integration boundaries, not a table read.

  Proposed shape (generic-core/GMRT-wrapper split, as everywhere else in this codebase):
  - `data_io/observation_header.py` — the primary header's descriptive keywords
    (`TELESCOP`, `OBSERVER`, `OBJECT`, `DATE_OBS`, `INSTRUME`, `BUNIT`).
  - `data_io/frequency_table.py` — the AIPS FQ table, and the fix for the FQ-table gap
    above.
  - A separate `AntennaPolarization` dataclass + reader, keyed by station number (like
    `read_source_table`'s dict-by-id), rather than extending `Antenna` itself — `Antenna`
    is already used everywhere, and adding more required fields would mean another round
    of test-fixture churn (as happened when `x_m`/`y_m`/`z_m` were added) for data most
    callers don't need.
  - `Source` extended directly with its missing SU-table columns (no legacy fixture
    burden — built fresh this session).
  - `data_io/observation_summary.py`: `list_obs(fits_path, index=None) -> ObservationSummary`,
    bundling `header`, `antennas`, `antenna_polarization`, `sources`, `frequency_setups`,
    and (only given a `RowIndex`, since building one is the expensive full-file scan)
    derived scan/timing info — integration time, per-source on-source time, observation
    start/end JD. This is both a genuine standalone "what's in this file" report and what
    plotting tools query for titles/axis context, rather than each one re-reading header
    metadata separately.
  - `instruments/gmrt/observation_summary.py` — a thin wrapper adding active-vs-DUD
    antenna resolution on top, same split as `instruments/gmrt/row_index.py`.
  - A shared `visplot` title helper reading `TELESCOP`/`DATE_OBS` (and the array's known
    location, if useful) off an `ObservationSummary`, replacing `antenna_layout`'s current
    ad hoc `telescope`/`source_path` parameters — and used by every other plot's title too.

- **T21 — visPlot density mode — NOT STARTED (added 2026-09-27).** A plot style for dense
  generic Y-vs-X plots that colors each pixel by the number of samples landing in it —
  the approach of Datashader and `mpl-scatter-density`, and close to AIPS UVPLT's own
  "array method" (which fills an in-memory pixel array and displays it as an image).
  Plain points overplot: at hundreds of thousands of samples, later points hide earlier
  ones and the plot shows only coverage. Proposed, on T22's design: a `--density` style
  option backed by a counting reducer (samples per pixel, e.g. uint32) fed by the same
  `run_stream`, drawn by `XYFigure` with a logarithmic color scale; no new dependency.
  Deferred by the user behind the PDF-output and marker work (2026-09-27).
  The user (2026-09-30): a 2D density plot, showing which x-y regions hold many samples
  and which few; for colouring by category, each category its own hue with its own
  density, and one category at a time where that is too cluttered. Datashader's
  categorical shading does this: a pixel's hue mixes its categories' colours by their
  counts, its intensity follows the total count. Where categories overlap (RR and LL of
  an unpolarised source, almost everywhere) the pixel shows the mixed hue.

- **T22 — visPlot streaming plots — DONE (added and built 2026-09-27).**
  *Problem.* A generic Y-vs-X plot held the whole selection in memory at once: the
  reader's output (complex128 + float64 weight, 24 bytes per sample, twice the file's
  float32 12), then each plotted quantity as float64, masks and filtered copies, and
  matplotlib's own float64 offsets — a measured peak of ~110 bytes per sample (checked at
  122M samples: 13.7 GB). 3C468.1, RR, all channels is 924 million samples, ~101.6 GB —
  more than this 67 GB host, for data that is 11.1 GB on disk (44.3 GB read from disk,
  since Stokes are interleaved within each channel). An interim guard refused such
  selections; the user rejected it as hiding the problem, and asked for plotting whose
  memory does not depend on the data size, on a Raspberry Pi as on a large machine.

  *Design* (one path for every streamed plot; the user asked for a single scalable
  design over several code paths):
  - One reader, `data_io.visibility_data.iter_visibility_chunks`, yielding one bounded
    chunk of rows at a time (`read_visibility_data`, the whole-selection reader, is
    removed). Chunk size follows index building's budget convention (`ram_fraction` of
    RAM); `read_data=False` yields metadata-only chunks with no disk read.
  - One quantity registry, `visplot/quantities.py`: name, display name, unit (fixed, or
    the file's `BUNIT`), whether it needs visibility data, whether it is a category, and
    one evaluate function. Values keep size-1 axes where they don't vary, so a pair
    broadcasts only to the shape it needs (hour angle vs time: one value per row).
    Includes the geometry quantities (ha_h, az_deg, el_deg, pa_deg) and the categories
    stokes and source.
  - One stream runner, `visplot/stream.py`: `run_stream` feeds each chunk to reducers and
    drops it. `RangeReducer` is the axis-range pre-pass; `GridReducer` marks which
    pixels of a fixed grid the samples fall in, one int16 per pixel storing the top
    layer in paint order (category code + 1, flagged on top), whatever the number of
    categories. Re-adding the same samples leaves a grid unchanged, so restarting an
    interrupted pass is harmless.
  - One renderer, `visplot/xy_figure.py`: the grid drawn as an image on vector axes, with
    markers stamped at occupied pixels (size and shape chosen from the sample count),
    titles naming sources, telescope and file, and units on both axes.
  - Saved output: one plotting pass at a 600 dpi grid; the 150 dpi PNG and PDF use its
    exact 4x4 reduction.
  - Windows (`visplot/xy_interactive.py`): each plot fills as chunks arrive, with rows
    read shown; after a zoom or pan settles, that plot re-streams the selection over the
    new limits at the window's resolution.
  - Axis ranges: a `--x-range`/`--y-range` is used as given; otherwise one pre-pass over
    the selection, reading visibility data only for a visibility axis (amp, real, imag,
    phase) — the user chose an exact min/max pre-pass over a percentile estimate.
  - Named geometry plots (`ha-range`, `az-el-range`, `parallactic-angle-range`) became
    presets of the generic plot, colored by source, reading no visibility data;
    `geometry_range.py`, `scatter_xy.py` and `derived_quantities.py` are removed.

  *Measured* (2026-09-27, archival GWB file, amp-vs-uvdist_klambda, 256 MiB chunks):
  - RR, 3C286, 21M samples: 1.34 GB peak (old design 2.48 GB).
  - RR, 3C468.1, 122M samples: 1.33 GB peak (old design 13.7 GB).
  - All four Stokes, 3C286, 84M samples: 3.73 GB peak, against 0.23 GB for a run that
    reads no visibilities — ~13 bytes of working memory per chunk byte, the worst case;
    `STREAM_MEMORY_PER_CHUNK_BYTE = 14` sizes chunks from it.
  - RR, 3C468.1, all channels, 924M samples (the selection the old design refused):
    1.40 GB peak; 505 s for two passes of 44.3 GB each, while a second process was
    reading the same disk.

  *Behavior changes against the first visPlot:* `az-el-range` is two pages (elevation,
  azimuth) where it was one figure with two panels sharing the time axis; flagged
  samples are light-red markers of the plot's own shape, where they were crosses;
  overlapping markers paint in a fixed order (higher category on top, flagged above
  all), where they were 0.7-transparent; `--colorize-by` takes a category (stokes,
  source), where it took any quantity; `--linewidths` is removed, since stamped markers
  have no separate edge; zooming in a window re-reads the selection, where matplotlib
  zoomed into points already in memory. The transit (HA = 0) and horizon (el = 0) lines
  were dropped in the first cut of this ticket and restored as preset reference lines.

  *Open:* re-streaming on zoom re-reads the whole selection; a quantity could tell the
  reader which rows or channels fall outside the new limits so they are skipped.
  `astrometry.local_sidereal_time_hours` takes ~10 s on first use per process (astropy
  looking up UT1 from IERS tables), which geometry quantities inherit.

- **T23 — visPlot streaming speed and per-plot memory — DONE (2026-09-28).** From the T22
  review (user's points A and E). Before: one pass over 3C286 RR (6.8 GB, in the OS cache)
  took 33.2 s (205 MB/s), CPU-bound: `np.take` copying full chunks along unselected axes,
  `np.unique` sorting every chunk's samples, pixel arithmetic, dtype conversions.
  Changes:
  - the reader copies each selected Stokes/channel range straight from the file into
    the chunk's buffer (complex64 / float32, the file's precision), slicing where a
    selection is contiguous; amplitude and phase are computed in float64, so binning
    matches the earlier float64 results pixel for pixel (checked: 266,386 pixels both);
  - samples carry no per-sample category or flag arrays when a plot has none; pixel
    indices are computed in place; categories are counted with `np.bincount`;
  - reducers split into `compute` (thread-safe) and `apply` (one thread, row order): a
    chunk's row blocks are computed by worker threads (default 6, `--threads`), results
    applied in order, so the grid is identical for any thread count (tested);
  - a reader thread reads the next chunk while the current one is computed (at most
    one chunk waiting, in memory; nothing written to disk);
  - within a chunk, each plot's samples are released once its reducers have them.
  Measured (same pass, cached): 9.5 s single-threaded, 7.1 s with the reader overlap,
  4.5 s with 6 threads (1.5 GB/s), identical samples and pixels. Uncached (3C345, 8.7
  GB): 181 MB/s, the disk's rate, so the CPU no longer sets the pace. Worst-case working
  memory (all Stokes, colored, 6 threads): 2.66 GB peak for 256 MiB chunks (was 3.73 GB),
  ~9.2 per chunk byte; `STREAM_MEMORY_PER_CHUNK_BYTE = 10`.

- **T24 — visPlot range cache on disk — DONE (2026-09-28).** From the review (point B).
  `--cache-dir DIR` saves each axis range the range pass finds (min, max, and the value
  histogram, so a later percentile run also reuses it) and reuses it when the same
  selection is plotted again, skipping that pass. Keyed by the file (path, size,
  modification time), rows, channel/Stokes selection, quantity, flag handling, mirroring
  and log-axis handling. The user asked for no trash without an easy cleanup: nothing is
  written without `--cache-dir`; files are `visplot-cache_<FITS stem>_<key>.npz`, written
  as `<name>.<pid>.partial` and renamed when complete; the next run removes partials whose
  process is gone; an interrupted pass saves nothing; each run prints the cache's file
  count and size with the removal command; `--clear-cache DIR` removes every
  `visplot-cache_*` file there and nothing else. On 3C286 RR, a repeat run took 6.4 s
  against 13.9 s. Per-row summaries (for skipping rows on zoom) are left to review point
  D, on hold.

- **T25 — visPlot axis range modes and scaling — DONE (2026-09-28).** From the review
  (point C). `--x-range-mode`/`--y-range-mode`: `minmax` (default; outliers visible) or
  `percentile` (`--range-percentiles`, default 0.1:99.9), from a histogram of values in
  fixed log-magnitude bins (`visplot/value_histogram.py`: 1000 per decade, 0.23% value
  resolution, ~0.5 MB) filled during the existing range pass. `--x-scale`/`--y-scale`:
  linear, log, symlog, asinh (`--scale-linear-width` for the last two); samples are binned
  evenly in the scale's own coordinate using matplotlib's transform for that scale, and
  a test checks pixels line up with the drawn axis to 1e-9 of its length. Samples outside
  an axis range, or not positive on a log axis, are counted and stated on the plot.
  Checked on 3C286 RR amplitude vs uv distance: a log y-axis shows ten decades; the
  percentile range leaves 138,636 of 141.7M samples (0.098%) outside, stated on the plot.

- **T26 — visPlot inspection window — IN PROGRESS (added 2026-09-28).** From the review
  (point F). User's decisions (2026-09-28):
  - Toolkit: Qt (PySide6-Essentials 6.11.2, added to the venv and
    `config/requirements.txt`). The Qt window replaces the matplotlib windows of
    `xy_interactive.py`, keeping one interactive path; saved output is unchanged.
  - Build order: locate and export first, then pages (per baseline/antenna/source/
    Stokes), then flagging, then the review's layout items (multi-panel pages sharing an
    axis, e.g. elevation and azimuth; legends outside the axes; distinct colors beyond 10
    categories).
  - Flags: atomic, a visibility being a function of time, channel, Stokes and baseline.
    The archived pipeline's format (legacy `bandpass_and_flag_table_spec.md`, decoded by
    `ugmrt_query.expand_flag_table_to_mask`) is JSON `ugmrt_flag_table` v1/v2:
    `bad_antennas`, `bad_baselines`, and `bad_antenna_timeranges`/
    `bad_baseline_timeranges` entries of ISO-UTC `intervals` with an optional absolute
    `chanrange`; it has no polarisation field by design (the archived notes record
    all-correlation flagging as the only supported mode). Proposed: the same kind, version
    3, with an optional per-entry `stokes` list (absent = all Stokes, as before) and
    `reason`/`created` fields; an atomic flag is one entry (baseline, time interval,
    channel range, Stokes). The archived decoder ignores the added keys, so it reads a v3
    Stokes-specific entry as all Stokes (it over-flags, never under-flags). A converter
    writes an AIPS FG table: FG rows carry an antenna pair, time range, channel range and
    per-Stokes `PFLAGS`, one row per entry; how AIPS loads a standalone FG table to be
    checked when the converter is built.
  Plan: locate (drag a box; stream the selection once; table of samples: baseline by
  antenna names, UTC time, channel and frequency, Stokes, values, weight; capped, with
  totals; saved as CSV) and export (re-stream the current view at a chosen dpi, save
  PNG/PDF/SVG) first.

  Done (2026-09-28), awaiting the user's trial with a display: `visplot/qt_inspector.py`,
  one window with a tab per plot; each streamed plot fills as chunks arrive (status bar:
  pass, rows, GB read, elapsed) and redraws after a zoom or pan settles; Locate (drag a
  box: `LocateReducer` lists up to 10,000 samples with baseline by antenna names, UTC
  time, channel, frequency, Stokes, values, weight, and counts all by baseline; saved as
  CSV) and Export (re-reads the view at a chosen dpi; PNG, PDF, SVG, EPS, TIFF, JPEG).
  Streams run one at a time on a background thread; drawing reads locked grid snapshots;
  a locate or export interrupts drawing, which resumes. Replaces the matplotlib windows
  (`xy_interactive.py`, removed). Tested headlessly (Qt offscreen): drawing, zoom
  redraw, locate with CSV, export size. On 3C286 RR, headless: 141.7M samples drawn in
  9.1 s; a locate at 36-37.5 kλ found 30,045 samples, all on E06:19-S06:24, in 6.8 s;
  an export at 300 dpi took 5.9 s.
  First trial by the user: export worked; Locate did nothing but zoom. Cause: matplotlib's
  zoom and pan modes lock the canvas, and the box selector ignores events while another
  tool holds the lock, so with zoom still on the drag zoomed. The headless test had
  called the locate method directly, bypassing the toolbar and mouse path. Fixed: Locate
  is a mode exclusive with zoom and pan, with status-bar guidance, and the CSV button is
  enabled once there is a result; a test now drives the toolbar button and a simulated
  mouse drag with zoom left on.
  Next: pages, then flagging, then the layout items.
  Flagging here means writing flags: drag a box and record entries for those samples in a
  flag file of its own, the raw file untouched (the T26 proposal of 2026-09-28, point 2).
  The user (2026-09-30): writing flags is to be discussed before it is built; reading
  flag and calibration tables to apply while plotting is T42. The user's direction, the
  same day: Locate is the listing (like AIPS UVLIST: each visibility as the file records
  it, with its flag), and flagging is editing that listing -- flag or unflag chosen
  visibilities -- written to a flag file that this code, AIPS and CASA can read; the input
  file is never modified. Checked (2026-09-30): AIPS UVFLG reads a text file of commands
  (INTEXT; adverbs per entry ending in "/", up to 40,000 entries: ANTENNAS/BASELINE,
  TIMERANG as day, h, m, s, BCHAN/ECHAN, STOKES as names or a bit mask, REASON, OPCODE
  'FLAG' or 'UFLG', the latter removing FG entries that match exactly); CASA flagdata
  mode='list' reads a text file of one command per line (antenna='A&B',
  timerange='YYYY/MM/DD/hh:mm:ss~...', spw='0:5~61', correlation='RR', mode='manual' or
  'unflag'). The two syntaxes differ, so the proposal is a flag file of this code's own as
  the record, written out as a UVFLG INTEXT file and a flagdata list file. To settle in the
  design discussion: Locate lists one row per visibility (a u-v point now stands for all
  selected Stokes) and lists flagged ones for unflagging; whether AIPS and CASA can unflag
  what the file flags by its weights (UVFLG's help describes UFLG for FG entries only);
  each package's time convention for this file (recorded times are IAT, T34); merging a
  box's visibilities into ranges.
  Terms (user, 2026-09-30: "we need to design our definition consistently with
  conventions used in radio astronomy data processing packages"): a row is one baseline
  at one time, numbered as in the file, its data and flags of shape [nchan x nStokes] (the
  MS's main-table row, DATA and FLAG [ncorr, nchan]); a visibility is one (channel,
  Stokes) element of a row, the unit a flag applies to; a point is what a plot draws --
  one visibility on a per-Stokes plot, one (row, channel) standing for every selected
  Stokes on a u-v plot. The editor shows each selected row with its [nchan x nStokes]
  flag grid, the selected visibilities marked; a flag or unflag applies to a cell, a
  channel across Stokes, a Stokes across channels, or the row. Flag file entries use the
  same terms: time range, baseline, channel range, Stokes set.

- **T27 — Astrometry without network access — DONE (2026-09-28).** From the review (point
  G). `local_sidereal_time_hours` asked astropy for UT1, which tried to download IERS
  tables (10 s timeout per process here, and a stall on an offline machine). Now
  `astrometry.Ut1Provider` supplies UT1 - UTC: astropy's bundled IERS-B tables (final
  values; this astropy's cover 1962-01-01 to 2026-08-14) when they cover the dates, the
  online IERS tables (5 s timeout) for later dates, and otherwise UT1 = UTC with a
  warning that hour angle may be off by up to 0.9 s of time (0.00025 h). Sidereal time
  is Greenwich apparent sidereal time relative to the TIO plus the site longitude, which
  needs no further table lookup; it agrees with astropy's own computation to 0.001 ms
  of time. The CLI prints the UT1 source for any geometry plot or filter before
  reading, and on a fallback prints the warning and puts it on every geometry plot.
  First use now takes 0.48 s (was 10 s).

- **T19 review point D — interactive responsiveness — NOT STARTED (on hold since
  2026-09-28).** Four items from the T22 review, distinct from the locate/zoom fix:
  progressive chunk order (read chunks in a spread-out order, so an early window shows a
  sparse sample of the whole selection; in time order it shows only the first part);
  instant zoom preview
  (show the existing image enlarged at once, sharpened when the re-read finishes); zoom
  pruning (skip rows or channels outside the new limits for time, u/v/w, uv distance and
  frequency); re-bin on window resize. To be built with the GUI (T32), where they matter.
  Measured first (2026-10-01; GWB 3C286, RR, amp vs uv distance: 69,174 rows, 6.80 GB,
  141.7M samples): with the rows in the page cache a range pass took 3.8 s and a draw
  4.5 s; from disk, at T23's 181 MB/s, about 38 s each; every zoom re-read all 6.80 GB.
  A zoom to 0-2 kλ can reach 18,189 rows (26%), to one minute 8,694 (12.6%). The user
  chose all four, in the order: zoom pruning, progressive order, re-bin on resize, zoom
  preview.
  1. Zoom pruning — DONE (2026-10-01). Every pass reading visibility data reads only the
  rows that can reach its view (`xy_session.rows_in_views`, `stream.ViewRowsReducer`):
  window draws (a zoom or pan), exports, Locate and its CSV, and the command line's
  saves (with --x-range/--y-range) and --locate; the window and the command line prune
  alike, so an export and its command's repeat stay pixel for pixel the same. The rows
  come from one pass over row metadata, no visibility read, on the band's two edge
  channels alone: every row-metadata quantity is either independent of frequency or
  proportional to it, so a row's values reach their extremes at the band's edges (a
  test checks the rows equal those every channel gives); a row is kept when on each
  metadata axis its values reach the view, or for a mirrored plot their negation on
  both. A grid drawn from those rows equals the one from every row (tests; on GWB the
  0-2 kλ zoom: 30,628,181 samples either way). The metadata pass took 0.1 s on GWB's
  69,174 rows; the zoomed draw reads 18,189 rows, 1.79 GB instead of 6.80. The Drawn line
  says it: "... from 18,189 rows of 69,174 (the others lie outside the view)".
  2. Progressive order — DONE (2026-10-01). The window's draws read in spread order
  (`iter_visibility_chunks(spread=True)`): units of consecutive integrations holding at
  least 32 MiB of the selection's rows (one GWB integration, about 14 GSB ones) taken in
  bit-reversed order (`spread_order`: 0, n/2, n/4, 3n/4, ...), each chunk's rows
  ascending; every row read once, the grid unchanged (test). Range passes, saves, exports
  and Locate keep file order (the CSV is in row order). Disk cost, measured on GWB rows
  not read before: 165 MB/s in file order against 162 spread on one contiguous run of 18
  integrations (0.67 GB each), 156 against 161 on a sparse selection. In the window,
  GWB 3C468.1 RR amp vs time (450,954 rows): after 8 s, 3% read, samples cover the whole
  6.5 hours; in file order they would cover its first minutes.
  4. Re-bin on resize — DONE (2026-10-01). A window's grid was made at the axes' pixel
  size once, and a resize stretched its image. Now once a resize settles (0.3 s, as a
  zoom does) and the axes differ from the grid by more than 2 px, the view is drawn
  again at the new size (pruned to the view and in spread order, items 1 and 2). On GWB
  3C286 RR amp vs uv distance a resize took the grid from 329 x 806 to 579 x 1116 pixels
  in 6.4 s (rows in the page cache), the window idle afterwards. Test: a resized window
  draws the same samples on a grid of its new size.

- **T28 — Locate: save every located sample — DONE (2026-09-28).** User's question: the
  table and the CSV stopped at the first 10,000 samples. `LocateReducer` now works in
  columns (one array per field per chunk) with an optional sink; `visplot/locate_csv.py`
  writes every located sample as it is found (memory bounded by the chunk), as
  `<name>.<pid>.partial` renamed when complete, with readable and exact columns
  (baseline names and station numbers, UTC and JD, channel and frequency, Stokes, x, y,
  weight, mirrored, row, source) and the totals by baseline at the end. The window's
  "Save all as CSV" writes from memory when every sample was kept, otherwise reads the
  selection again; the table keeps the first 10,000. CLI parity: `--locate
  XLO:XHI,YLO:YHI --locate-csv FILE` (one streamed plot; no window). On 3C286 RR, a CLI
  locate at 36-37.5 kλ wrote all 30,045 samples (E06:19-S06:24) in 3 s.

- **T29 — Plot aspect — DONE (2026-09-28).** `--aspect auto|equal|free`; `auto` (default)
  is equal when x and y are the same kind of quantity, by an `aspect_group` in the
  quantity registry (u/v/w in s, u/v/w in kλ, real/imag), on linear axes, and free
  otherwise (a shared unit is not enough: hour angle and time are both in hours). Equal
  scale widens one axis's range about its centre, never narrowing either, computed here:
  matplotlib's own `adjustable="datalim"` narrowed x from ±10 to ±2.68 in a test, which
  would have cut data off. The grid is sized to the limits after the aspect applies. The
  window has an "Equal aspect" toggle per plot (to be recorded in the reproducing
  command by T31). `--aspect equal` requires linear axes.
  Amended 2026-09-28 at the user's request (u and v at one scale but different spans were
  hard to read): equal aspect now draws a square axes box and widens the shorter range to
  the longer's span, so both axes show one scale and one span (a mirrored u-v plot: both
  +-R); a window re-applies this after each zoom.

- **T30 — Units per axis — DONE (added and built 2026-09-28).** The user chooses the unit
  an axis is shown in (`--x-unit`/`--y-unit`; the GUI's unit selectors come with T32).
  Quantities are now `real, imag, amp, phase, time, u, v, w, uvdist, ha, az, el, pa, freq`
  (and the categories); the old names that carried a unit (`u_klambda`, `time_h`,
  `freq_mhz`, ...) stay as aliases of a quantity and unit, and an alias with a different
  `--x-unit` is rejected. Units, by quantity: time in h, min, s since the first selected
  integration, or clock time: UTC, local observatory time, LST; u, v, w, uv distance in
  λ, kλ, Mλ at each channel's frequency, or m, km (one value per row; s, µs, ns dropped
  at the user's request, 2026-09-28, `u_sec` etc. now rejected with the alternatives); phase,
  azimuth, elevation, parallactic angle in deg or rad; hour angle in h, deg, rad;
  frequency in Hz, kHz, MHz, GHz; real, imag, amp in the file's BUNIT, plus Jy, mJy, µJy
  when BUNIT is a flux density (AIPS's 'JY' read as Jy; 'UNCALIB' allows no conversion).
  Design: each quantity has base evaluations (u: a length in m, the file's seconds times c,
  and a length in λ); a unit is a base times a positive factor (from astropy.units). Ranges are
  found and cached in the base and converted, so a range found in kλ serves Mλ, and one
  in h serves min, without another pass. Clock axes are ticked at whole clock steps
  (labels: see the amendment below); LST counts on past 24 h across 0h LST; clock time
  needs a linear scale and no mirroring.
  Presets take the units too, with their fixed ranges and reference lines converted
  (`ha-range --y-unit deg`: ±180°). Axis labels, the located-samples table and CSV header
  state each axis's unit; elapsed time names its origin ("Time since 2021-07-25 16:48:00
  UTC (h)"). Visibility units are checked against BUNIT and local time's zone before any
  row is selected; every plot then carries its units explicitly (for T31's command).
  Preset output files are now named by quantity alone (`az-el-range_el.png`, was
  `..._el_deg.png`), since the unit can change.
  Local time added at the user's request (2026-09-28: night and day follow local time,
  which matters for polarimetry): the zone comes from the file's TELESCOP by a table in
  `instruments/observatory_time_zones.py` (GMRT: Asia/Kolkata, IST), or `--time-zone`
  (an IANA name); the file itself has no zone. The axis uses the zone's UTC offset at the
  first selected integration, named in the label ("Time (IST, UTC+05:30; day 0 =
  2021-07-24)");
  if the offset differs at the last (a daylight-saving change), the CLI warns and notes
  it on the plot. Checked on the GWB file: 3C286 at 22:13-22:21 IST with hour angle
  4.4 h (LST 17.9 h at GMRT, RA 13.5 h); the whole file's hour angle against LST from
  17:50 to 03:40 (+1d); 3C286's uv coverage in km at equal scale.

  Amended the same day at the user's request: the default time axis is the time as
  recorded (the DATE parameters as the file holds them, labelled with the file's TIMSYS),
  every clock axis (recorded, UTC, local, LST) is labelled dd:hh:mm:ss, days counted from
  the file's reference date (RDATE, else DATE-OBS) as AIPS does, so the GWB file (RDATE
  2021-07-24) starts on day 01; u, v, w and uv distance lost s, µs, ns (`u_sec` etc. are
  rejected with the alternatives). `src/visplot/__init__.py` removed (its module list is
  now `src/visplot/README.md`); that exposed `src/cli/visplot.py`, run as a script,
  shadowing the `visplot` package, so the entry script is now `src/cli/run_visplot.py`
  (as `run_gwb_pipeline.py`), with a test running `bin/visplot.sh`. The Equal aspect
  toggle is a check box and off returns to the data's range (it kept the widened limits).

- **T34 — Timestamps: declared time system against the file's u, v, w — DONE
  (2026-09-28).** The user asked whether the recorded time is UT. The GWB file's AN table
  has TIMSYS = 'IAT', IATUTC = 35, DATUTC = 0. AIPS Memo 117 (rev. 2025-10-01) defines
  DATUTC as the data's time system minus UTC and IATUTC as IAT - UTC on RDATE; 35 s was
  the leap-second count of 2012-07 to 2015-06 (37 s on 2021-07-24, erfa), so the keywords
  disagree with each other and with the date. Measured from the data
  (`data_io/timestamp_check.py`): u, v, w predicted from the AN positions, SU coordinates
  and each timestamp shifted by an offset, the offset fitted per integration: the stored
  u, v, w match timestamps 34.078 s after UTC (11 of 12 integrations over the night,
  spread 0.050 s, J2000 frame, baseline = ant2 - ant1, rms 5 mm; 10 m rms read as UTC).
  The 0.93 s between 34.08 and 35 is not explained (half the 2.683 s integration is
  1.34 s; UT1 - UTC is -0.146 s). User's decision (option C): UTC = recorded - IATUTC
  when TIMSYS is 'IAT' (`TimeReference.recorded_minus_utc_s`), applied wherever a
  timestamp is read as UTC (UTC, local, LST axes; hour angle, Az/El, parallactic angle;
  the geometry filters; absolute --time-range bounds, taken as UTC; the located samples'
  time_utc, with the JD column renamed jd_recorded), and the check run by the pipeline's
  build_index stage (logged) and by visplot whenever it reads UTC, warning when measured
  and declared differ by more than 0.1 s (the method's own spread is 0.05 s) or the offset
  cannot be measured. On the GWB file it warns: 0.922 s apart.

- **T35 — Moon scans' u, v, w point 130-143 degrees from the Moon — OPEN, found
  2026-09-28; parked until the Moon is imaged (2026-09-29).** The user saw projected
  spacings below the 45 m dish diameter in MOON0520 (v vs u, λ, 300-316 MHz): 684 rows, all on C05:06-C06:07 (103 m; 9.4-15.1 m in the file)
  and C01:02-C02:03 (329 m; 26.9-40.2 m), unflagged. Fitting the direction each
  integration's stored u, v, w were computed for (w = b·s over 378 baselines, exact to
  0.000 m): 10 of 13 sources match their own SU coordinates to 0.000 degrees (3C286,
  3C345, 3C303, 3C468.1, Cas-A, B1929+10, 3C48, MOON0625, MOON0635, DA240), so rows are
  labelled correctly; MOON0520, MOON0545, MOON0605 match directions 130-143 degrees from
  the Moon (RA 109.8/116.4/130.5, Dec -11.2/-10.7/-13.3; hour angle about -6 h, just
  below the eastern horizon), fixed in RA/Dec within a scan apart from a Moon-like drift
  (about 0.1 degrees per 15 min), jumping between scans; their SU coordinates match the
  Moon's topocentric position to 0.1 degrees. The two short baselines are level and run
  east-west (azimuth 285 and 277 degrees), within 6-8 degrees of that wrong direction,
  hence the short projections; towards the Moon they project to 84-87 m and 249-261 m.
  visplot plots the stored u, v, w as they are (3C286's shortest is 50.67 λ at 300 MHz).
  The user's u-v plot of MOON0605 and MOON0625 (same target, 6 min apart) showed different
  coverages; predicted for the Moon's direction from the antenna positions, SU position and
  time, the two continue each other, and the file's u, v match that prediction to 0.0 m
  rms for MOON0625 (44,604 rows) but differ by 12.3 km rms for MOON0605 (124,740 rows).
  The GSB file of the same night (`40_014_25jul2021_gsb.FITS`, 8.1 GB, 1,312,794 rows,
  8.05 s integrations, RR/LL, 256 channels 306-339 MHz; row index built in memory, nothing
  written beside it) behaves the same (2026-09-28): the same three Moon scans' stored u,
  v, w point 125.6 / 131.2 / 141.1 degrees from their SU positions (GWB: 130.2 / 135.7 /
  143.2), MOON0625/0635 and the other 8 sources match to 0.000 degrees. Its timestamps
  follow the same rule as GWB's: TIMSYS 'IAT', DATUTC 0, IATUTC 34 (GWB: 35), and the u,
  v, w put recorded - UTC at 33.078 s (GWB: 34.078 s) -- in both files IATUTC - 0.922 s,
  so the two files' timestamps differ by 1 s. The archived GSB Moon trajectory plots
  checked the SU/FIELD position, which is right in both files; the u, v, w are a separate
  record in each row.
  Fringe-stopping test (GSB, 2026-09-29): median raw amplitude by physical baseline length,
  channels 64-191, RR/LL, unflagged, MOON0520 / MOON0625: 0.91 (0-300 m), 1.23
  (0.3-1 km), 1.12 (1-3 km), 0.56 (3-10 km), 0.37 (10-30 km); MOON0605 / MOON0625: 0.98,
  1.27, 1.19, 0.82, 0.64. Short baselines carry the Moon's flux in all scans (the dishes
  pointed at the Moon), and 1-3 km baselines are not decorrelated, as they would be
  (to about 1-30%) had fringes been stopped 125-143 degrees away: the correlator tracked
  the Moon; only the u, v, w written for MOON0520/0545/0605 are wrong, and must be
  recomputed (AN positions, SU phase centre, time) before those scans are plotted in u, v
  or imaged. The weaker long baselines in MOON0520 are not yet explained. The archived
  movie (`uv_coverage_movie.py`, e88ae13 on develop; the user's MOON0520 movie) plots the
  split file's stored u, v -- copied unchanged by `visSplit.py` -- without the four
  baselines under 70 λ, which were flagged in the split, hence its "shortest ~72 λ".
  SU table (2026-09-29): the three scans are exactly the SU entries with non-zero proper
  motion, the same in both files: PMRA / PMDEC (deg/day) 8.856 / 5.796 (MOON0520),
  9.288 / 5.940 (MOON0545), 9.684 / 6.012 (MOON0605); MOON0625, MOON0635 and every other
  source have 0 / 0. The three were set up as a moving source, the last two as fixed
  positions; the fitted direction's drift within a scan (about 0.1 degrees per 15 min,
  about 10 degrees/day) is of the order of PMRA. How the u, v, w writer used PMRA/PMDEC is
  not known.
  Open puzzle, parked until imaging (user, 2026-09-29): the spacings below 45 m are what
  the stored u, v, w give, yet the archived Moon imaging from the split data worked. The
  user will pull the split, flagged data and image it; of the candidate u, v, w (stored;
  recomputed towards the fixed SU position; recomputed towards a centre moving at
  PMRA/PMDEC), the right one shows the Moon's disk. Also open: the weaker long baselines
  in MOON0520; whether the correlator stopped fringes towards a moving centre in these
  three scans; a per-scan u, v, w consistency check at indexing (extending T34; a non-zero
  PMRA/PMDEC in the SU table as the first signal); recomputing u, v, w for such scans --
  the user's decisions.

- **T31 — Provenance for every plot — DONE (added 2026-09-28, built 2026-09-29).**
  Every GUI "Plot" click and every CLI run records the explicit
  CLI-equivalent command (every option including defaults, absolute paths, view ranges,
  figure size, dpi, theme) through `provenance/manifest.py`'s RunManifest (stage
  "visplot"): host, UTC time, git commit and branch with the dirty diff, Python and key
  package versions, fingerprints of the FITS file and its row index, and the outputs. The
  user: "absolute reproducibility required! No compromise. Record the host name as well."
  Records go to `./visplot_runs/` (the directory visplot starts from) by default, shown in
  the GUI and printed by the CLI, changeable with `--provenance-dir` (user's choice,
  2026-09-28).
  Built (`visplot/records.py`; RunManifest gained a `details` entry for what a tool adds):
  - `PlotRecord`, one per command-line run (action save, locate, window), GUI Plot, and
    file a plot window saves (Locate's CSV, Export): the request as parameters, its
    command line, the GUI session, package versions; the run's messages in its log; the
    files written as outputs; failed with the reason when the run stops (a request that
    cannot run, a missing index, an interrupt). The CLI prints the record's run id and
    path first.
  - `SessionRecord`, one per GUI window (stage "visplot_session"): its log keeps every
    message the window shows (the Messages list keeps the newest 5000), each plot's and
    save's run id and command among them.
  - `WindowProvenance`: the plot window's saves are recorded by the same code for a
    window the CLI opened and a GUI tab: the request narrowed to the action (a CSV adds
    `--locate` with the box and `--locate-csv`; an export fixes `--x-range`/`--y-range` to
    the exported view), the parent run's id, dpi and figure size. A save or plot whose
    record cannot be written is not made, and the window says why.
  - A located-samples CSV names its command and record in its header. The GUI shows each
    plot's run id in its tab and in History, and has a Records field (`--provenance-dir`)
    naming the session's log. Tests: `test_visplot_records.py`, and GUI tests for the
    plot, export and session records and for a plot refused when its record fails; test
    records go to `tmp/pytest/visplot_runs` (conftest). The theme is a window preference;
    plots are drawn the same in either (T33 brings plot themes as a request option).
  Checked on the GWB file (2026-09-29, `ha-range`, 3C286, 69,174 rows): the record lists
  both inputs (the 389 GB FITS by size and time only, and its index), the two outputs,
  the git diff beside it, and the time-system and timestamp-check messages in its log.
  Exports (closed 2026-09-29, the user's decision): an export's dpi and figure size were
  in its record's details only, since the command line saved at 150 and 600 dpi at its
  own figure size. `--dpi N` (50-2400, the Export dialog's range; default 150) sets the
  PNGs' and the low-resolution PDF's resolution, the high-resolution PDF being at the
  smallest whole multiple of N that is at least 600 (600 for 150), so the PNGs stay exact
  reductions of the one plotting pass; `--figure-size W,H` (inches, default 8,6) sizes
  each streamed plot's saved figure (the antenna layout and source listing keep theirs).
  An export's command fixes the view's ranges, `--dpi`, `--figure-size`, and `--aspect`
  when the window's Equal aspect box overrode the request's (on linear axes, where it
  takes effect); a GUI test runs the command of an export from a window-sized figure
  with `--output-dir` and gets the export's PNG pixel for pixel.

- **T32 — visplot GUI — IN PROGRESS (added 2026-09-28).** A plotms-style front end on the
  existing engine: file and selection controls, x/y from the quantity registry with units
  and scales, display options, Plot/Clear; `bin/visplot.sh` with no arguments opens it.
  The CLI stays first-class (programmatic probes and batch generation). User's brief: "a
  professional design ... robust, efficient, fool-proof, good-looking", and (2026-09-28)
  "design it in a way that cli and gui can never diverge". Folds in T19 point D and T26's
  remaining controls.
  Built (2026-09-28), on one code path for both:
  - `visplot/request.py`: the command-line parser (moved from `cli/`) defines every option
    once; `PlotRequest` holds a value per parser option, by the option's own name, and
    `to_argv`/`from_argv` convert it to and from a complete command line (every option,
    defaults included, paths absolute; a value starting with '-' written `--opt=value`).
  - `visplot/run.py`: `check_request`, `open_file`, `select`/`count_selection`, `prepare`,
    `run_locate`, `save_outputs` -- what `cli/visplot.py`'s `main` did, now shared; the CLI
    (`cli/run_visplot.py`) only parses, prints the run's reports and picks save, locate or
    window. `cli/visplot_args.py` moved to `visplot/request_args.py`.
  - `visplot/gui/`: the form (`form.py`) has one control per request option, bound by
    name; choices from the parser, quantities and units from the registry, tooltips from
    each option's `--help`; fields checked as typed by the same resolvers. The window
    (`main_window.py`): Data (file, summary: `visplot/file_summary.py`), Axes, Selection,
    Display, Performance sections; live counts (`count_selection`, off the UI thread);
    the form's command line; Plot runs `prepare` and opens a tab with the plot window
    and the plot's command; Messages and History docks (double-click loads a plot's
    request into the form); light and dark window themes.
  - Tests that keep them together: every request option is a form field or a named GUI
    action (`ACTION_OPTIONS`: locate, save); a request shown in the form comes back
    unchanged; a request's command line reads back as the same request; a GUI plot and
    the command line's run of the same request select the same rows and give the same
    pixels.
  Measured on 3C286 RR (141.7M samples, 6.8 GB): the GUI's Plot drew it in 12 s.
  Done since: provenance records from Plot, Locate's CSV and Export (T31, 2026-09-29).
  File > Save plots as files (2026-09-29, `gui/save_dialog.py`): folder, filename prefix,
  `--dpi`, `--figure-size` and the high-resolution PDF, with the command it runs and the
  files it would replace, checked by `check_request`; it runs `prepare` and
  `save_outputs` on a background thread, with the save's record (`records.recorded_run`,
  now also around the CLI's run and the GUI's Plot) and progress in the status bar; the
  last save's options are offered again. Test: the GUI's save and the command line's run
  of the same request write the same files, pixel for pixel, for a streamed plot and the
  antenna layout. For the save off the main thread, the antenna layout and source listing
  are now bare Figures with their own Agg canvas (pyplot's figures are Qt objects under
  the Qt backend); the GWB file's antenna layout renders identically before and after.
  Stop for a save (user's request, 2026-09-29): one save runs at a time, with a "Stop
  saving" button in the status bar; the save stops at its next chunk, writes nothing
  (`save_outputs` raises `run.Stopped` before any file is opened), and its record is
  failed with where it stopped; closing the window stops a running save and waits for
  its record. Found on the way: `resolve_extents` returned ranges from the part of the
  selection read when its pass was stopped; it now returns None, as the plot window's
  range job already expected.
  Opening a file (2026-09-29): the user asked for a progress bar by the File field while a
  large file loads. Timed on the GWB file: reading the index and tables took 0.18 s, the
  file summary 6.9 s, of which 6.2 s was `np.unique(..., axis=0)` over 4.0M antenna
  pairs; counting the pairs as one integer each gives the same 378 baselines in 0.2 s,
  and opening now takes 0.7 s (a test checks the summary's counts). The progress bar
  (under the File field; row counts where a step has them, e.g. building an index)
  comes with the Build index button.
  Build index (user's request, 2026-09-28; built 2026-09-30), with the progress bar the
  user asked for by the File field: opening a file without its row index shows what
  building one takes (the file's size to pass over) and a Build index button, which runs
  the pipeline's own build_index stage on a background thread (its record and log under
  the Records folder), then opens the file. The full-file scan
  (`read_all_param_columns`) gained `on_chunk(rows_done, rows_total)`, passed through
  `build_row_index`, `build_gmrt_row_index` and `run_build_index_stage`: the GUI's thin
  bar shows rows, percent, time elapsed and left, and Stop ends the scan at its next
  block (ScanStopped: nothing saved, the stage's record failed). The stage also takes
  `build_index.max_chunk_bytes` (recorded in its parameters): the GUI reads 256 MB
  blocks, since the default, 20% of this host's 67 GB, is 13.5 GB -- the 8.1 GB GSB file
  would be one block, the bar never moving and Stop taking effect only at the end.
  Opening a file shows the same bar, as a busy indicator. The note beside Build index
  was clipped in the user's window (a wrapped label beside a button in one form row);
  it is a read-only box of three lines, scrolled when longer, as the user suggested.
  Tests: the stage's progress
  per block and a stop saving nothing; the GUI building a missing index and opening the
  file; Stop ending a build with nothing built and saying so.
  Next: point D and T26's controls.

- **T33 — Themes — DONE (added 2026-09-28, built 2026-09-30).** Light and dark themes for
  Qt and matplotlib; a color too close to the background (e.g. `k` on dark) is flipped in
  lightness; saved files light unless chosen otherwise. The Qt window had its light and
  dark themes (View menu) already; T33 themes the plots. `visplot/plot_theme.py`: the
  light theme is matplotlib's default look -- the GSB file's 3C286 plots (amplitude vs uv
  distance colored by Stokes, hour angle, antenna layout, source listing) render pixel
  for pixel as before themes, apart from the record id; the dark theme uses the GUI's
  dark window colors. The contrast rule, the user's choice (2026-09-30) between it
  applying to both themes or the dark one only: dark only. On the dark axes a marker
  color below 3:1 (WCAG's minimum for graphics) is mirrored in lightness, hue kept, and
  lightened further where the mirror falls short -- tab10's brown (2.85:1), `k` (1.24:1)
  and `0.3` (1.98:1) among the colors offered; on white, tab10's orange (2.53:1), olive,
  cyan, light coral and 11 of tab20's 16 are below 3:1 and stay as they are.
  `--plot-theme light|dark` (default light; the GUI's "Plot theme") colors a plot on
  screen and saved, as its command states; the View menu themes the window. First built
  with the plots on screen following the window's theme and only saved files and exports
  following --plot-theme; the user's trial (2026-09-30): "View-> Light/Dark does
  something. But Plot theme: Light/Dark has no effect" -- a new tab took the window's
  theme, so the field changed nothing on screen. Now a plot is drawn in its request's
  theme wherever it is shown, and the tab's command reproduces what it shows. Tests: the
  contrast ratio and the rule; a dark figure's colors, on screen and saved; the source
  listing's table; in the GUI, the plot theme coloring the plot and its export whatever
  the window's theme, and a table plot drawn in it.

- **T36 — Antenna layout: the core inset covered S04 — DONE (found and fixed
  2026-09-29).** On the GWB file's layout the inset half covered S04:23's marker: the
  corner search counted marker centres inside the inset box, and S04's centre sat 2 px
  outside it. `_choose_inset_box` now counts a marker as covered when its disc plus a
  5 pt clearance reaches the inset (in display pixels, core antennas included), keeps the
  inset off the core's zoom rectangle, and shrinks the inset from 0.46 of the axes in
  steps to 0.30 until a corner (upper right, upper left, lower right, lower left) is
  clear. GMRT's inset is now 0.44, lower right, touching no antenna (test against the
  file).

- **T37 — A plot states its Stokes and the parameters it shows — DONE (user, 2026-09-29;
  built the same day, below).** "For many of the y-axis quantities, what is that
  quantity for (which Stokes) is a fundamental attribution"; u, v, w do not need it. A
  plot named its Stokes only in the legend, when colored by Stokes. The user's decisions:
  the axis labels stay as they are (no added clutter); "it is more fundamental than the
  axis label. Flagging for RR and LL can in principle be different", so plots of
  visibility quantities get a Stokes legend, keeping the plot's look; channels do not go on the
  axes, and a text block outside the plot area, as the AIPS TV has, may record the
  critical parameters shown. Design to be agreed before building; mock (2026-09-29,
  `tmp/t37_mock/`, 3C286 amp vs freq, colored by Stokes): a key and parameter panel
  below the axes. Found in the mock: colored by Stokes on the GWB file's four products,
  LR takes the category palette's red (tab10's fourth color), next to the red of flagged
  samples. The user on the mock: "looks good in a general sense". Decisions (2026-09-29):
  - one color dimension, the one --colorize-by names, with its key in the panel below
    the axes (outside the plot area, so it covers no data, in the window and in saved
    files alike); the other dimension stated as text in the panel; both separated by one
    plot per value (T26's pages);
  - no --colorize-by: one color, as now, the panel naming what was plotted;
  - default figure 8 x 7 in, the panel taking the extra inch;
  - red is reserved for flagged samples: category palettes without reds or pinks (tab10
    has 8 colors left; tab20 16, for more categories); flagged samples drawn as "x" (size:
    see the revision below), kept light coral #f08080 (the user; the mock's key swatch
    was pure red by mistake). Go-ahead given 2026-09-29.
  Markers measured (150 dpi PNGs): above 1M samples one pixel, so dot and square are the
  same; 100k-1M a 3x3 square (a disc at that radius is a 5-pixel plus sign); up to 100k
  discs (13 px, 69 px).
  First build (`visplot/plot_panel.py`, `xy_figure.py`; revised below): `panel_facts`
  records what the selection holds -- Stokes, channels and their frequency span, baselines,
  autocorrelations and antennas, sources, the UTC span, the row filters as text; the
  panel shows it in two columns under a rule: Stokes (the key when colored by Stokes;
  otherwise the one color's swatch, or "each in its source's color"), Channels,
  Baselines, Drawn (what was drawn, or the drawing's progress, and the flagged cross,
  "flagged: none" when none are drawn) | Sources (the key when colored by source),
  Time, Selection, Record (the provenance record of the run showing or saving the
  figure; an export names its own). The key is fixed before drawing: colors go by rank
  among the categories the selection holds (before, by rank among those drawn so far,
  so a color could change mid-draw), and a long key wraps between entries. The panel is
  laid out in inches from the figure's bottom (text widths from the font's metrics), the
  axes above it, and a window re-lays it out on resize. The legend inside the axes, the
  status line and the note at the figure's corners are gone; the caveat note is the
  panel's last line. Default figure 8 x 7 in. A u-v plot keeps its Stokes row: with
  flags applied it draws a sample per Stokes (flags come from per-Stokes weights), so
  the Stokes selected change the coverage drawn. Checked on the GWB file: amp vs freq
  colored by Stokes (LR now purple; the light-coral count in the plot area is 0, so the
  bottom columns the mock showed in red were LR samples, and that selection has no
  flagged samples), v vs u, amp vs time colored by five sources (key on two lines). The
  GUI's export and save tests compare whole PNGs, panel included, with the same record
  id.
  Revised (2026-09-29, the user on the first build):
  - Counts: a plot whose quantities do not vary along Stokes (u, v, w, uv distance,
    time, hour angle, ...) drew one sample per Stokes when flags applied, the samples
    broadcast to the weights' shape: the GWB u-v render's "28,344,320 samples" was
    1,730 rows x 2,048 channels x 4 Stokes x 2 (mirrored), four times its 7,086,080
    points (and per-row quantities, e.g. u in m, were multiplied by the channels too);
    the auto marker size went by that count. The picture was right (duplicates share a
    pixel). The user: "keeping the row does not get me the correct statistics".
  - The flag rule (the user's): flags have the data's shape, one per visibility's time,
    baseline, channel and Stokes. A point whose quantities do not vary with Stokes (u-v)
    combines its visibility's selected Stokes and is flagged if any of them is; one
    Stokes selected, its own flag decides; flagged points are left out unless
    --show-flagged draws them as crosses. Channels are never combined: a per-row
    quantity (u in m, time) is one point per channel's visibility. (A first version
    combined channels too, a point spanning channels flagged only if every channel was;
    the user: "a single visibility has a unique channel much like it has a unique Stokes,
    a unique time ... why are you conflating channels again? The dimension of the flags
    in a UVFITS file is exactly the same shape as that of the visibility data array!";
    removed.) `stream.combine_flags` applies the rule where samples are built and in the
    Locate tool, whose counts now match the plot's; located points get a "flagged" column
    (a point combining Stokes has no one weight). Tests: the rule on a block with chosen
    flags, and `varies_along` (the panel's knowledge of what a point combines) against
    the arrays the stream builds, for every quantity and unit.
  - Flagged crosses the size of the markers (the user: "the same/similar size as the
    plot markers, not huge!"), at least 3 x 3 px, the smallest "x" (the first build drew
    at least 5 x 5 px).
  - The panel, "a bit too cluttered ... the human has to LOOK FOR information": 9 pt
    (was 7.5); the key on its own first line (swatches, the flagged cross); two columns
    below (Stokes, Channels, Baselines | Time, Selection, Drawn); a Flags line stating
    the rule when points combine samples; Sources only as the key (the title lists
    them); the record id as a small grey footer; each line anchored at its own height.
    Font, as the user asked, from the packages of CERN/AIPS/NASA: Helvetica (CERN ROOT's
    default text font; PGPLOT and AIPS draw Hershey stroke fonts, which matplotlib does
    not render), through its metric clones TeX Gyre Heros or Nimbus Sans where
    installed, else Liberation Sans, else DejaVu Sans (matplotlib's own). A panel font
    from the system can differ between machines, so the same command can give different
    pixels elsewhere: the user wants portability and reproducibility, and a font chooser
    (T41). The panel's design is to be thought afresh (T40).

- **T38 — Flag rule option: the lenient rule — DONE (user, 2026-09-29; built
  2026-10-01).** T37's rule flags a point combining several Stokes if any of them is
  flagged (the user's default). An option for the other rule -- such a point shown unless
  flagged in every selected Stokes (coverage wherever any product has data) -- e.g.
  `--combine-flags any|all`, stated in the panel's Flags line.
  Built: `--combine-flags any|all` (default any; the GUI's "Flag rule" in Display), a
  `PlotSpec` field that `stream.combine_flags` applies; a point of one Stokes is that
  visibility's own flag under either rule. The panel's Flags line reads "a point is
  flagged if all of RR, LL are" for `all`. Ranges need no change: the range pass keeps a
  value wherever any of its samples is unflagged, so a range, cached or found by a pass,
  holds under both rules. Neither raw file carries flags where checked: u-v points, RR and LL, were
  drawn for every row and channel of GSB's 3C286 (5,806,080) and of GWB's every 200th
  integration over all sources (41,029,632), under both rules; the rule shows on flagged
  data, e.g. the user's split files. Tests: the rule on constructed flags (a point
  flagged in one Stokes, in both, a per-Stokes plot unchanged, an unknown rule refused);
  the Flags line; the request's round trip.

- **T39 — --every-nth skips whole baselines — DONE (found and built 2026-09-29).** The
  user asked why the panel of a 28-antenna plot said "189 baselines": the test renders used
  `--every-nth 40`, and the GWB file has exactly 378 rows per integration, one per
  baseline (no autocorrelation rows), in the same order in all 10,476 integrations
  (checked), so row k is baseline slot k mod 378. A row stride steps through the
  baseline order within each integration, so a stride sharing a factor with 378 never selects some
  baselines: on 3C286, every 40th row covers 189 baselines (gcd 2), every 7th 54 (gcd
  7), every 41st all 378 (gcd 1). The panel's count was right for the rows selected; the
  sampling misleads. The user's decision (2026-09-29), both: say so in the run's report
  and the panel when a row stride covers fewer baselines than the selection has, and an
  integration stride (every Nth integration, all its baselines) beside the row stride.
  Built: `select_rows` gains `every_nth_integration` (every Nth of the selected
  integrations, with all their selected rows; one of the three ways of thinning at a
  time, checked before any file is read) and reports what a row stride kept
  (`RowSelection.stride_pairs`: antenna pairs kept, and in the selection without it).
  `--every-nth-integration N` (and the GUI's "Every Nth integration"); a row stride that
  kept fewer is warned in the run's messages ("--every-nth 40 keeps 189 of the
  selection's 378 baselines: rows are in baseline order within each integration (378
  rows here) ...") and on the panel's Baselines line, in dark red; `--every-nth`'s help
  says why. On the GWB file, 3C286 u-v: every 40th row, 1,730 rows and 189 baselines;
  every 40th integration, 1,890 rows (5 integrations) and all 378.

- **T40 — The panel under a plot, designed afresh — OPEN (user, 2026-09-29).** "Lets
  discuss the beautification/aesthetics of the text panel at the bottom in a separate
  ticket. We need to step back on this and think afresh - we could borrow some ideas from
  professional displays such as from medical fraternity." T37's panel stays as built
  until then.

- **T41 — Fonts: portable, reproducible, chosen — DONE (user, 2026-09-29; built the
  same day).** The panel's font came from the system (TeX Gyre Heros, Nimbus Sans,
  Liberation Sans, else DejaVu Sans), so the same command could draw different pixels on
  another machine. The user: "I want portability and reproducibility", a font chooser
  (a drop-down in the GUI), the axes' fonts fine as they are, and "we want the fonts
  installed in the venv, not in system". Shipping font files in the repository was set
  aside: TeX Gyre's licence (the LPPL) counts distributing part of the work as a
  modification, a reading better not guessed; installing from the publisher avoids
  redistributing at all. pip cannot install them (no TeX Gyre or Liberation package on
  PyPI among the names tried).
  Built: `config/fonts.txt` pins TeX Gyre Heros 2.004 (GUST's OpenType package, its URL
  and SHA-256 755954b7...265d); `bin/install_fonts.sh` (`src/cli/install_fonts.py`)
  downloads it into `tmp/`, checks the SHA-256 (a mismatch installs nothing), unpacks
  the eight files unmodified into `gmrt/share/fonts/tex-gyre-heros/` with a SOURCE.txt,
  deletes the download, and leaves a font installed from the same pin as it is. Tied to
  the venv's build (the user: "tie the font installation to the venv building
  process"): `bin/build_venv.sh` builds the venv in one run -- creates it (with `--clear`,
  or when missing), installs `config/requirements.txt`, installs `config/fonts.txt`, and
  checks both (`install_fonts.py --check`); `docs/dev/ENVIRONMENT_SETUP.md` now builds
  the venv with it. `visplot/fonts.py` loads the panel's font from its file (TeX Gyre
  Heros from the venv; DejaVu Sans from matplotlib);
  `--panel-font tex-gyre-heros|dejavu-sans` (default tex-gyre-heros) and the
  GUI's "Panel font" drop-down choose it; the command records it, and the run's record
  names the font file and its SHA-256. A font not installed stops the run before the
  file is read, saying to run `bin/build_venv.sh` or choose dejavu-sans. Tests: the
  installer on a pinned test zip (once, then "already installed"; a wrong checksum
  refused; `--check`), the repository's pin against the installed font, the panel
  drawing with each font's file and its record, a missing font stopping the run.
  `bin/build_venv.sh` run on the existing venv (2026-09-29): every requirement already
  satisfied, the font already installed, both checks passing.

- **T42 — Flag and calibration tables read and applied while plotting — OPEN (user,
  2026-09-30).** "We should in fact allow reading of gains/leakage etc tables as well so
  that we can 'apply' calibration on the fly when plotting." Flags from a flag table
  combine with the file's own; gains and bandpass divide each visibility by its two
  antennas' gains per channel and polarisation; leakage (D-terms) mixes the four
  correlations, so a leakage-corrected RR needs RL, LR and LL read as well (the GSB file
  has RR and LL only). The formats come from Phase C's solver (the archived pipeline's
  are the bandpass `.npz` and the `ugmrt_flag_table` JSON of
  `legacy_gsb_40_014/bandpass_and_flag_table_spec.md`); one apply path serves plotting
  and the calibrated split (stage 6 of the user's outline, 2026-09-25).

- **T45 — Averaging, scalar and vector, of the selected visibilities — OPEN (user,
  2026-09-30).** "I find it useful to check visibilities of point sources - vector
  averaging across baseline and time directly shows the spectra of the source! Very
  handy as diagnostic." Averaging over time (an interval, or all), baselines and channels
  (a bin width), per Stokes: vector (amplitude and phase of the weighted mean complex
  visibility) or scalar (the weighted mean amplitude), flagged visibilities left out; as
  AIPS's POSSM and plotms's averaging. In the streaming design: a reducer accumulating
  weighted sums per output bin (memory set by the number of bins), the averages then
  plotted as any Y vs X. Across baselines, a vector average of uncalibrated visibilities
  decorrelates (each baseline carries its two antennas' instrumental phases); with T42's
  calibration applied it gives the source's spectrum. The user asked whether to club it
  with T42: a ticket of its own, since averaging serves raw data now (in time per
  baseline, scalar across baselines), and the two are stages of one path -- select,
  apply calibration (T42), average (T45), plot. To be designed with the user.

- **T46 — Gain tables plotted — OPEN (user, 2026-09-30).** "we want our tool to also be
  able to plot gain tables - so we shall discuss the layout for gain tables." The layout
  to be discussed with the user. The archived pipeline's bandpass `.npz`
  (`legacy_gsb_40_014/bandpass_and_flag_table_spec.md`) can be read now; the new
  solver's tables come with Phase C, as T42's do.

- **T47 — Elevation below the horizon or the elevation limit, marked — DONE (user,
  2026-10-01; built the same day).** "Do we need negative elevations? Below elevation
  limit is a red flag already. So we may not need anything below 0, but if the
  calculation show el < 0 (or elLimit when present), we should show some warning (red
  Cross?) for those points ? The markers can remain at el=0 line." The `az-el-range`
  preset drew -90 to 90 degrees with a dashed horizon at 0; no elevation limit was known
  to the code.
  The user's decisions: GMRT's hardware limits, 15 degrees low and 110 high; one floor
  (the limit, the horizon only for a telescope with none known); "using el values as
  they appear ..., but have provision to warn if they exceed 110, or go below 15!
  Simple"; ▼ in the point's own color, red staying with flags. The elevations are
  computed (source position, array position, time), so they lie within -90 to 90 and
  only the low limit can mark one; neither file has antenna pointing (PRIMARY, AN, FQ, SU;
  u, v, w, baseline, date, source, frequency setup).
  Built: `instruments/elevation_limits.py` (TELESCOP to (low, high); GMRT 15, 110; else
  0, 90); `run.with_elevation_limits` gives any plot with elevation on an axis a
  `PlotSpec.limits` in that axis's unit, drawn as dashed lines. `GridReducer` keeps the
  unflagged points beyond them at their values in two more grids, placed on the grid's
  edge where they lie beyond it, counted by category with their extreme value;
  `XYFigure` draws them as ▼ (below) or ▲ (above) in their own color, under flagged
  crosses; the panel's Limits line reads e.g. "below 25°: 8,694 points (MOON0605 1,890,
  DA240 1,890, 3C468.1 1,512, and 4 more sources), lowest 17.9°", in the warning color,
  else "none below 15° or above 110°"; a save prints it as a warning, and the window
  reports it once per plot. Every elevation axis shows the sky, 0 to 90 degrees, unless
  a range is given (the user: "why not make el axis show from 0 upwards? And mark that
  dashed line if the el limit is known"; then the top at the zenith), the dashed lines
  drawn where the telescope's limits are known; a fixed range holds from the start (the
  empty axes had stretched to the 110-degree line before the first drawing, which the
  user saw as a 0-100 view). On the GSB file every
  source stays above 15 degrees (lowest about 18, 3C48 and MOON0635); a test limit of 25
  degrees drew and warned of seven sources. Tests: the table; the grids, edge placement
  and counts; the panel's text and color and the triangles; the limits in an axis's
  unit, on x or y.

- **T48 — Time tick labels: tilted, and a choice of formats — DONE (user, 2026-10-01;
  built the same day).** "On the time axis - can we tilt the time string - that adds to
  the aesthetics as well as uses space optimally"; "Time string format specifier: Can we
  have some standard defaults as user-choosable option for the time format?" The clock
  axis (`visplot/clock_axis.py`) labelled every tick dd:hh:mm:ss, the day counted from
  the file's reference date, and the axis label names day 0.
  The user's decisions: the formats dd/hh:mm:ss, dd:hh:mm:ss, hh:mm:ss and iso; the
  default AIPS's dd/hh:mm:ss ("DD:HH:MM:SS caught me since I am not used to seeing the
  DD in a time. The "/" tells the brain that ... that is the DAY part"). Built:
  `--time-format` (the GUI's "Time format" in Display), for every clock axis and the
  cursor readout; hh:mm:ss gives the first tick and the first of each later day their
  day (1/16:44:00, 17:00:00, ..., 2/00:00:00); iso writes day 0's date plus the day,
  refused for LST (sidereal days have no calendar date). The GUI's data panel uses the
  default. Clock tick labels on the x axis are tilted 30 degrees (matplotlib's
  `labelrotation_mode="xtick"`, kept by ticks a zoom makes), up to 10 of them; the
  axes leave room for the widest label the format writes, two digits of seconds
  included. The user asked why day 0 is the day before the data: both files declare it
  (DATE-OBS and the AN table's RDATE 2021-07-24; the first row is 2021-07-25 16:43,
  recorded IAT), and AIPS counts days from RDATE, so AIPS writes this file's first time
  1/16:43:30 too; kept, so times match AIPS's listings and the TIMERANG day numbers of
  UVFLG flag files (T26). Tests: every format, the day marks of hh:mm:ss, iso refused
  for LST, the tilt, and the axes' room for iso's labels.

- **T49 — Plot title: the source, or "multiple sources" — DONE (user, 2026-10-01;
  built the same day).** "Since we are labeling the sources in the bottom panel, do we
  need them all in the plot title? If a single source is plotted for, we can add the
  Source name. But if more than a source is plotted for, we can simply say (multiple
  sources)." The panel named the sources only in its key, when colored by source; it now
  has a Sources line otherwise ("13 sources: 3C286, 3C345, ..." cut to its column).
  Tests: the title for one source and several; the Sources line, and none beside a key
  of sources.

- **T43 — visplot leaves the structural DUD entries in — DONE (found and fixed
  2026-09-30).** The
  GSB file's antenna layout drew 32 antennas and put its inset on three points 1 m apart,
  W06:30, C07:31 and S05:32: C07 and S05 are GMRT's structural DUD entries
  (`GMRT_STRUCTURAL_DUD_NAMES`), placeholders appended after the 30 stations. The user:
  "You forgot DUD antennas! GMRT has only 30 antennas." visplot's `open_file` reads the
  whole AN table, so the layout, `--antennas` matching and the GUI's antenna check see
  32; standing rule 8 excludes DUDs once, before anything downstream. The file summary
  counts antennas from the rows and is right. Fix: `open_file` leaves out the
  telescope's structural DUD entries (GMRT's from `instruments.gmrt`), kept as
  `dud_antennas`; the layout draws the 30 and names the entries left out.
  Built: `instruments/structural_duds.py` maps TELESCOP to its structural DUD names (as
  `instruments/observatory_time_zones.py` maps it to a time zone) and splits a table
  with `resolve_active_antennas`; `open_file` keeps `antennas` without them and
  `dud_antennas`; the layout's footnote reads "Left out: C07:31, S05:32, structural DUD
  entries of the AN table". Neither file's rows reference stations 31 or 32 (checked
  against both row indexes), so no selection changes. The GSB layout now draws 30
  antennas with its inset on the central square; the GWB layout (no DUD entries in its
  table) is pixel for pixel as before. `--antennas C07` on the GSB file is refused as no
  antenna of the file. Tests: the lookup by TELESCOP; a GMRT table losing its DUD
  entries, another telescope's keeping every entry; the GSB file opened with 30
  antennas, C07 refused, the layout drawing 30 with its footnote.

- **T44 — Antenna selection: the convention, and an empty selection said — DONE (user,
  2026-09-30; built the same day).** The user: one antenna selected, amp vs uv distance,
  cross or auto: "I get nothing ... I would want some message to be printed"; and with
  C1, C2, C3 and
  cross, "cross only between the 3? Or cross of those 3 with all others?" The code
  (`select_rows`) keeps a row when both its antennas are in `--antennas`: only the
  baselines among those listed, so one antenna selects nothing with cross; while
  `--antennas`' help says "keep baselines involving these antennas". Neither file has
  autocorrelation rows (GSB 0 of 1,312,794; GWB, T39), so auto selects nothing. An empty
  selection reports "selected 0 rows" as information and draws an empty plot. The
  conventions, checked 2026-09-30: CASA's `antenna='1,2,3'` selects all baselines
  including those antennas, `'1,2,3&'` only those among them; AIPS's BASELINE help
  defines ANTENNAS with BASELINE=0 as "(I in ANTE) .OR. (J in ANTE)" (its example 2's
  wording, "baselines between antennas 1 through 6", reads the other way; example 3,
  the negated list, fits the rule). `--exclude-antennas` already drops a baseline with
  either antenna listed.
  The user's decisions: `--antennas` follows the convention (every baseline with one of
  them); for a narrower selection, the user's design, AIPS's ANTENNAS x BASELINE and
  CASA's 'A&B': `--baselines-with` keeps only the cross-correlation baselines between
  one of `--antennas` and one of its own -- the same list in both gives the baselines
  among them; it needs `--antennas`. An autocorrelation has one antenna, chosen by
  `--antennas`; `--correlation-type` stays the one switch for autocorrelations, and with
  `auto`, `--baselines-with` is refused (the user: "if we select corr="auto", the
  baselines-with should be frozen"; the GUI's field is disabled and its text left out of
  the request). All nC2 baselines remain the default (no antenna options). A plot
  whose selection is empty stops before reading, naming the filter that emptied it: the
  filters applied one at a time in the command line's order, e.g. "no rows selected:
  --antennas C00 --baselines-with C00 leaves none of the 22,680 rows selected before it",
  or "this file has no autocorrelation rows (--correlation-type auto)"; table plots,
  which read no rows, still plot. The GUI shows it under the count and disables Plot and
  Save with it. On the GSB file, 3C286: `--antennas C00`, 1,620 rows (27 baselines x 60
  integrations); with `--baselines-with C01,C02`, 120; `--antennas C00,C01,C02`, 4,680
  (78 baselines); with the same list as `--baselines-with`, 180 (3). Tests: the rules in
  `select_rows`; the counts, the messages and the refusals on the synthetic file; the
  GUI's frozen field and disabled Plot.

- **T20 fits here too:** its file summary (channel width, integration time, sources,
  dates) belongs in the GUI's data panel. The user (2026-09-30): the data panel already
  lists much of it; a listObs of its own, perhaps a new tab with controls choosing what
  to list, designed with the user before it is built.

  Agreed order (2026-09-28): T28 and T29, then T30 and T31 (the GUI's unit selectors and
  its Plot button need them), then T32 with T33 and point D, then T26's remaining items
  (pages, flagging, layouts) as GUI controls with CLI options; T20 alongside T32; T21
  after.
  Order to finish the GUI (user, 2026-09-30, easier first): T33, T38, point D, T21, T26
  pages; then T20 (a listObs tab) and T26's flag editing, each designed with the user;
  T40 later; T42 with Phase C; then the data analysis.

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
  `axis_selection` (since replaced by `iter_visibility_chunks`, T22). This is where
  GMRT-specific FITS-format knowledge belongs, outside `src/engine/`. Produces the
  vis/model/antenna-index arrays T5a's solver consumes.
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
    themselves are read separately, by `read_visibility_data` (since replaced by
    `iter_visibility_chunks`, T22).
  - `visibility_data.py` (reader since replaced by `iter_visibility_chunks`, T22):
    `read_visibility_data()` reads the visibility bytes for
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
  `select_rows`/`read_visibility_data` (now `iter_visibility_chunks`) directly on it, no
  GMRT-specific wrapper needed
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
