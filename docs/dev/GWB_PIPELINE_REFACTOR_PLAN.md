# GWB Pipeline Rebuild — Plan

**Status line:** T0-T4, T5a, T5b, T5c done, Phase A complete (2026-09-24). Phase B: T19
and T22 (visPlot, streaming) done 2026-09-27, 277 tests passing; T20, T21 not started.
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
  - `src/cli/visplot.py` (the CLI entry point, dispatched by `bin/visplot.sh`) and
    `src/cli/visplot_args.py` (its pure, directly-testable argument resolution) — wiring
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

- **T28 — Locate: save every located sample — NOT STARTED (added 2026-09-28).** User's
  question: the table and the CSV stop at the first 10,000 samples (the CSV adds only
  the total and per-baseline counts). Add "Save all as CSV": the locate pass writes every
  located sample to the chosen file as it finds them (memory bounded); the table keeps
  the first 10,000.

- **T29 — Plot aspect — NOT STARTED (added 2026-09-28).** Equal aspect when x and y share
  a unit (u vs v, real vs imag), free otherwise (amp vs time, amp vs uv distance); a
  toolbar toggle and `--aspect equal|auto`; the pixel grid sized after the aspect applies.

- **T30 — Units per axis — NOT STARTED (added 2026-09-28).** The user chooses the unit an
  axis is shown in (`--x-unit`/`--y-unit`, and a unit selector per axis in the GUI), via
  astropy.units: frequency Hz..GHz; time s/min/h since the first integration, or UTC or
  LST clock time; u, v, w and uv distance in s/ns, m/km or λ/kλ/Mλ (so `u_klambda` and
  `u_sec` become one quantity `u` with units; the old names stay as aliases); phase and
  angles deg/rad, hour angle h/deg. Amplitude keeps the file's BUNIT; conversion only
  where BUNIT is a flux unit (e.g. Jy to mJy).

- **T31 — Provenance for every plot — NOT STARTED (added 2026-09-28).** Every GUI "Plot"
  click and every CLI run records the explicit CLI-equivalent command (every option
  including defaults, absolute paths, view ranges, figure size, dpi, theme) through
  `provenance/manifest.py`'s RunManifest (stage "visplot"): host, UTC time, git commit
  and branch with the dirty diff, Python and key package versions, fingerprints of the
  FITS file and its row index, and the outputs. The user: "absolute reproducibility
  required! No compromise. Record the host name as well." Where records go (a directory
  the user names, with a visible default) to be settled with the GUI design.

- **T32 — visplot GUI — DESIGN (added 2026-09-28).** A plotms-style front end on the
  existing engine: file and selection controls, x/y from the quantity registry with units
  and scales, display options, Plot/Clear; `bin/visplot.sh` with no arguments opens it.
  The CLI stays first-class (programmatic probes and batch generation); GUI and CLI share
  one request model, and every GUI setting has a CLI option. User's brief: "a
  professional design ... robust, efficient, fool-proof, good-looking". Design presented
  for review before building. Folds in T19 point D and T26's remaining controls.

- **T33 — Themes — NOT STARTED (added 2026-09-28).** Light and dark themes for Qt and
  matplotlib; a color too close to the background (e.g. `k` on dark) is flipped in
  lightness; saved files light unless chosen otherwise.

- **T20 fits here too:** its file summary (channel width, integration time, sources,
  dates) belongs in the GUI's data panel.

  Agreed order (2026-09-28): T28 and T29, then T30 and T31 (the GUI's unit selectors and
  its Plot button need them), then T32 with T33 and point D, then T26's remaining items
  (pages, flagging, layouts) as GUI controls with CLI options; T20 alongside T32; T21
  after.

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
