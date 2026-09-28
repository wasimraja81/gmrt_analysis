"""Plotting for exploring a UVFITS observation.

- `quantities`: every quantity a plot can show, in one registry.
- `plot_spec`: what one plot shows (y vs x, coloring, flags, ranges), and the
  named presets.
- `stream`: one loop feeding chunks of a selection to reducers (axis
  ranges, pixel grids); memory does not grow with the selection.
- `xy_session`: chunk size, range pre-pass and plotting pass.
- `xy_figure`: a pixel grid drawn as an image on vector axes.
- `qt_inspector`: the Qt inspection window: plots fill as the data streams
  in and re-stream on zoom; locate samples in a box; export at any dpi.
- `antenna_layout`, `source_listing`: plots of the file's tables.

Row selection is `data_io.row_selection.select_rows`; reading is
`data_io.visibility_data.iter_visibility_chunks`. `cli/visplot.py` assembles
inputs and decides what to show or save; a future Qt GUI can drive the same
session and figure classes from its own callbacks.
"""
