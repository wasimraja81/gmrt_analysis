# src/visplot/

Plotting for exploring a UVFITS observation, driven by `bin/visplot.sh` (`src/cli/run_visplot.py`).

- `request`: the plot request -- every option of a run, defined once by the command-line
  parser; `request_args`: the grammar of its values (ranges, antennas, channels, ...).
- `run`: running a request (check, open the file, select, prepare, locate, save) -- the one
  code path of the command line and the GUI.
- `records`: provenance records (T31) under `--provenance-dir` (default `./visplot_runs`):
  one per command-line run, GUI Plot and file a plot window saves, with the command that
  reproduces it; one per GUI session, whose log keeps every message the window showed.
- `quantities`: every quantity a plot can show, with the units it can be shown in, in one
  registry.
- `plot_spec`: what one plot shows (y vs x, units, coloring, flags, ranges), and the named
  presets.
- `stream`: one loop feeding chunks of a selection to reducers (axis ranges, pixel grids,
  located samples); memory does not grow with the selection.
- `xy_session`: chunk size, range pre-pass and plotting pass.
- `xy_figure`: a pixel grid drawn as an image on vector axes; `plot_panel`: the panel under
  each plot (color key, Stokes, channels, baselines, sources, time, filters, what was
  drawn, the provenance record); `clock_axis`: ticks and labels for clock-time axes
  (dd:hh:mm:ss).
- `axis_scale`, `value_histogram`, `range_cache`: axis scales, percentile ranges, and axis
  ranges cached on disk.
- `qt_inspector`: the Qt window: plots fill as the data streams in and re-stream on zoom;
  locate samples in a box (`locate_csv` writes them); export at any dpi.
- `gui/`: the plotting GUI (T32): a form holding exactly the request's options (`gui/form.py`),
  plot tabs, messages and history (`gui/main_window.py`); `file_summary`: what a file holds.
- `antenna_layout`, `source_listing`: plots of the file's tables.

Row selection is `data_io.row_selection.select_rows`; reading is
`data_io.visibility_data.iter_visibility_chunks`.
