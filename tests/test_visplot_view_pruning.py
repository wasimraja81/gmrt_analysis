"""A pass reading visibility data reads only the rows that can reach its view
(T19 point D): found from row metadata over the band's two edge channels, the
grid drawn from them identical to one drawn from every row."""

import numpy as np

from conftest import make_scratch_dir
from test_cli_visplot_output import _make_synthetic_file
from visplot.request import PlotRequest
from visplot.run import open_file, prepare
from visplot.stream import ViewRowsReducer
from visplot.xy_session import plot_grids, resolve_extents, rows_in_views


def _run(name, plots, **options):
    path = _make_synthetic_file(make_scratch_dir(name) / "obs.fits")  # u, v grow with the row: 40 rows
    run = prepare(PlotRequest(str(path), plots, **options), open_file(path))
    (plot,) = run.xy_plots
    return run.source, plot, resolve_extents(run.source, [plot])[plot]


def _same_grid(source, plot, view, rows):
    full, _ = plot_grids(source, [plot], {plot: view}, {plot: (40, 60)})
    pruned, _ = plot_grids(source.subset(rows), [plot], {plot: view}, {plot: (40, 60)})
    return np.array_equal(full[plot].layers, pruned[plot].layers) and full[plot].n_samples == pruned[plot].n_samples


def test_a_zoom_reads_only_the_rows_that_reach_it_and_draws_the_same_grid():
    source, plot, (x_full, y_full) = _run("prune_uvdist", "amp-vs-uvdist")
    lo, hi = x_full
    view = ((lo, lo + (hi - lo) / 4), y_full)  # the shortest quarter of the uv distances
    rows = rows_in_views(source, {plot: view})
    assert 0 < rows.size < source.n_rows
    assert _same_grid(source, plot, view, rows)
    assert rows_in_views(source, {plot: (x_full, y_full)}).size == source.n_rows  # the full view: every row


def test_a_mirrored_plot_keeps_the_rows_whose_mirror_reaches_the_view():
    source, plot, (x_full, y_full) = _run("prune_mirror", "v-vs-u", mirror=True)
    view = ((x_full[0], x_full[0] / 2), (y_full[0], y_full[0] / 2))  # the far negative corner: mirrored points
    rows = rows_in_views(source, {plot: view})
    assert rows.size > 0 and _same_grid(source, plot, view, rows)


def test_the_band_edges_find_the_rows_every_channel_finds():
    source, plot, (x_full, y_full) = _run("prune_edges", "amp-vs-uvdist")
    view = ((x_full[0] + (x_full[1] - x_full[0]) * 0.4, x_full[1]), y_full)
    every_channel = ViewRowsReducer(plot, *view, source.row_indices)
    source.stream([every_channel], read_data=False)  # every channel's values, as a plot evaluates them
    assert np.array_equal(rows_in_views(source, {plot: view}), source.row_indices[every_channel.keep])
