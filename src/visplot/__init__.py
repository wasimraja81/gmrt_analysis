"""Reusable plotting functions for exploring a UVFITS observation.

Each function here takes already-loaded data and returns a plain
`matplotlib.figure.Figure` -- no file I/O, no argument parsing, no
selection/filtering logic (that's `data_io.row_selection.select_rows` and
`data_io.visibility_data.read_visibility_data`, reused as-is). `bin/visplot.py`
is the only place that assembles inputs and decides what to show or save; a
future Qt GUI calls these same functions from its own callbacks instead of
reimplementing the plots.
"""
