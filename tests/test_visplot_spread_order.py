"""A window draws in spread order (T19 point D): the grid it ends with equals
the one drawn in file order."""

import numpy as np

import data_io.visibility_data as visibility_data
from conftest import make_scratch_dir
from test_cli_visplot_output import _make_synthetic_file
from visplot.request import PlotRequest
from visplot.run import open_file, prepare
from visplot.stream import GridReducer
from visplot.xy_session import resolve_extents


def test_a_grid_drawn_in_spread_order_equals_one_drawn_in_file_order(monkeypatch):
    path = _make_synthetic_file(make_scratch_dir("spread_order_grid") / "obs.fits")
    run = prepare(PlotRequest(str(path), "amp-vs-time", colorize_by="stokes"), open_file(path))
    (plot,) = run.xy_plots
    source = run.source
    monkeypatch.setattr(visibility_data, "SPREAD_UNIT_BYTES", 3 * source.row_bytes)  # one integration per unit
    source.chunk_bytes = 6 * source.row_bytes  # several chunks
    x, y = resolve_extents(source, [plot])[plot]
    grids = [GridReducer(plot, x, y, 30, 40) for _ in range(2)]
    source.stream([grids[0]], read_data=True)
    source.stream([grids[1]], read_data=True, spread=True)
    assert np.array_equal(grids[0].layers, grids[1].layers) and grids[0].n_samples == grids[1].n_samples > 0
