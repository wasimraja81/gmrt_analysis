# Legacy GSB/40_014 Archive — Index and Rationale

This directory holds the entire pre-rewrite repository content, moved here intact on
2026-09-23 when the GWB pipeline build restarted from an empty tree (see
`../docs/GWB_PIPELINE_REFACTOR_PLAN.md`). Nothing here was deleted. Everything below is
categorized so a later decision to reuse or consult a file doesn't require re-deriving
whether it was safe to ignore.

Path-relative assumptions in these scripts (e.g. `REPO_ROOT` resolving to `../..` from a
script's own location, or `${REPO_ROOT}/gmrt/bin/python`) no longer resolve correctly from
this location — the venv (`gmrt/`) stays at the true repo root. Treat everything here as
reference material to read, not to run in place without adjusting paths first.

## Categories

**Dead** — zero incoming references anywhere in the repo at time of archival; already
unused before this move, not just unused after it.

- `experimental/gainPlots_example.sh` — calls `pipeline_cli.py solplot`, which has no `bin/`
  wrapper and no driver step; both sides of this pair were dead.
- `experimental/backfill_moon_fits_time_headers.py` — one-off retroactive header-stamping
  utility, no production caller.
- `tools/dev/patch_notebook_interactive.py`, `tools/dev/rebuild_notebook.py`,
  `tools/dev/patch_thresholds.py`, `tools/dev/_patch_cross_scan_stack.py` — all target
  `preprocess_ugmrt.ipynb`, which does not exist anywhere in the repo; these would crash on
  invocation.
- `tools/dev/migrate_pages_layout.py` — one-off migration for a `diagnostics_out/` naming
  scheme that predates the current layout; hardcoded to a machine path
  (`/Users/raj030/...`) that no longer applies.

**Duplicate** — a near-identical fork of a file also archived here, kept only for its own
sake (both copies are equally archived; the distinction just says which one was the real
one in production).

- `experimental/outlier_detection.py` — near-identical fork of `src/modules/outlier_detection.py`
  (the one dynamically imported by `src/preprocess_ugmrt.py` and `src/run_clustering.py`);
  differs only in a 4-line import shim.
- `experimental/gainPlots.py` — near-identical fork of `src/gainPlots.py`; differs by one
  import line.

**Superseded** — an earlier version of something the production driver came to call a
different way; not a byte-for-byte duplicate, but functionally replaced.

- `experimental/visSplit_example.sh`, `experimental/visSplit_moon_example.sh`,
  `experimental/plotVis_example.sh`, `experimental/plotVis_moon0520_dev.sh`,
  `experimental/clustering_example.sh` — GSB-era, bare `python <script>.py` invocations
  (relative paths, no venv selection, no machine-profile handling); superseded by the
  `bin/*_example.sh` wrappers with `$REPO_ROOT`/`$PYTHON_CMD`/profile handling.
- `experimental/run_bandpass.sh`, `experimental/run_preprocess.sh`,
  `experimental/run_clustering.sh` — thin compat wrappers that themselves dispatch into
  `pipeline_cli.py`; self-documented as legacy in `pipeline_cli.py`'s own docstring.
- `experimental/casa_moon_imaging.py` (+ `casa_moon_imaging_example.sh`),
  `experimental/casa_3c468_imaging_example.sh` — earlier single-shot (non-selfcal) CASA
  imaging examples, superseded by the iterative selfcal engines
  (`src/gmrt_selfcal_dev.py`, `src/moon_selfcal_dev.py`).

**Orphaned-research** — standalone exploratory/diagnostic tools, never wired into the
production driver, not a fork of anything currently live. Not proven wrong or obsolete —
just never promoted. Worth a skim before assuming any specific problem needs solving from
scratch (destriping, ripple calibration, and UV-sampling diagnostics in particular have
real prior thought behind them).

`experimental/clustering_moon0520_dev.sh`, `tools/dev/test_smooth_window.py`,
`experimental/baseline_fringe_report.py`, `experimental/baseline_info_from_diagnostic.py`,
`experimental/casa_plotms_suspicious_baselines.py`, `experimental/plotms_moon_suspicious_example.sh`,
`experimental/moon_elevation_report.py`, `experimental/moon_visibility_model_fit.py`,
`experimental/moon_phasecenter_trajectory_analysis.py`, `experimental/quick_moon_snapshot_native.py`,
`experimental/quick_3c468_snapshot_native.py`, `experimental/convolve_moon_snapshot.py`,
`experimental/moon_snapshot_and_convolve_int110.py`, `experimental/ripple_calibration.py`,
`experimental/ripple_calibration_test.py`, `experimental/fit_rrll_sinusoid.py`,
`experimental/plot_bandpass_tool.py`, `experimental/plot_transfer_summary_panels.py`,
`experimental/stripe_all_planes.py`, `experimental/stripe_baseline_diagnose.py`,
`experimental/stripe_baseline_diagnostic.py`, `experimental/stripe_baseline_diagnostic_example.sh`,
`experimental/stripe_baseline_mean.py`, `experimental/stripe_baseline_multi.py`,
`experimental/v-based-outlier-detection.sh`, `experimental/uv_sampling.py`,
`experimental/uv_sampling_movie_example.sh`, `experimental/uvfits_ms_metadata_audit.py`,
`experimental/write_moon_ds9_annotations.py`, `experimental/smoketest_tclean_vs_phaseshift.py`,
`experimental/run_all11_destripe_no_shift_compare.sh`, `experimental/run_moon_selfcal_nomask_dev.sh`,
`experimental/run_plot_tool.sh`, `experimental/test_radial_shell_stk0000.sh`,
`experimental/tmp_run_stack_compare.sh`, `experimental/fits_time_headers.py`.

Note on the last one: `fits_time_headers.py` was genuinely load-bearing at archival time
(imported by `src/gmrt_selfcal_dev.py`), just misplaced under `experimental/`. It is
archived as-is per the archive-only decision — the new tree gets a freshly named/written
equivalent rather than a copy-forward, per the file/function naming-review requirement.

**Active-reference** — was real, working production code/docs for GSB. Not dead, not
duplicated; archived only because the pipeline is being rebuilt from scratch, not because
anything here is wrong. This is the material to read for scientific/behavioral intent when
writing the new equivalent — never to import or copy by filename into the new tree.

- All of `bin/` (driver + every stage wrapper script), `src/` and `src/modules/` (the
  stefCal solver, flag application, clustering, secondary-cal, split/plot/selfcal engines),
  `provenance_logs/` (the `.cmd`-capture precedent for the new provenance library).
- Design/domain docs: `GMRT_40_014_WORKFLOW.md`, `pipeline_review.md` (partially stale — see
  the 2026-09-22/23 conversation record for what's since been fixed upstream),
  `3c48_bandpass_gain_strategy.md`, `bandpass_and_flag_table_spec.md`,
  `FLAGGING_DESIGN_NOTES.md`, `moon_imaging_notes.md`, `ripple_convergence_todos.md`,
  `experimental/docs/*` (workflow flowcharts, `unified_calibration_plan.md`,
  `clustering_mode_config_matrix.html`, `api_reference.html`).
- `preprocess_ugmrt.cfg` — GSB-tuned config defaults; the new tree gets its own config
  written from first principles, not inherited defaults.
- `gwb_workflow_action_plan.html`, `stk0010_mask.reg`, `moon_int0112.reg` — not GSB-specific
  logic (a forward-looking planning doc and sky-coordinate region files for the same
  field), but archived per the archive-only decision rather than copied forward; re-derive
  or re-author these fresh in the new tree if/when needed.
- `src/gainPlots.py`, `src/moon_selfcal_dev.py` — real code, reachable only manually
  (`pipeline_cli.py solplot`) or via a non-default engine flag (`MOON_SELFCAL_ENGINE=legacy`)
  respectively; not exercised by the default driver, but not dead in the sense of "unused
  code with no purpose" — orphaned-but-functional. Treated as active-reference since they
  represent real design intent (an alternate/predecessor moon selfcal engine, a bandpass
  solution plotting tool) that may inform whether the new tree wants an equivalent.
