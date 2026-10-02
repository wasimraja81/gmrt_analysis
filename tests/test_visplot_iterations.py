"""One plot per baseline, antenna, source or Stokes product (T26): the
iterations of a selection, and pages of them in a grid."""

import re
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from conftest import make_scratch_dir
from data_io.row_index import default_row_index_path, load_row_index
from data_io.row_selection import select_rows
from test_cli_visplot_output import _make_synthetic_file
from visplot.iterations import Iteration, iteration_source, list_iterations, page_layout
from visplot.plot_spec import PlotSpec
from visplot.quantities import QuantityContext
from visplot.request import PlotRequest
from visplot.run import (RequestError, check_request, draw_pages, highres_dpi, open_file, page_batches, prepare,
                         save_outputs)
from visplot.stream import GridReducer, RangeReducer
from visplot.xy_figure import density_opacity
from visplot.xy_session import XYSource, plot_grids, resolve_extents, rows_in_views


def _source(name):
    path = _make_synthetic_file(make_scratch_dir(name) / "obs.fits")
    index = load_row_index(default_row_index_path(path))
    ctx = QuantityContext(time_reference_jd=float(index.jd.min()), bunit="UNCALIB", stokes_labels=("RR", "LL"),
                          antenna_names={1: "C00:01", 2: "C01:02", 3: "C02:03"}, source_names={1: "3C286", 2: "3C48"})
    return XYSource(path, index, select_rows(index).row_indices, None, ctx, chunk_bytes=64 * 1024)


def test_the_iterations_are_the_values_the_selection_holds_in_order():
    source = _source("iterations_list")
    index, rows, ctx = source.index, source.row_indices, source.ctx
    assert [it.label for it in list_iterations("baseline", index, rows, ctx.stokes_labels, ctx)] == [
        "C00:01-C01:02", "C00:01-C02:03", "C01:02-C02:03"]
    assert [it.label for it in list_iterations("antenna", index, rows, ctx.stokes_labels, ctx)] == [
        "C00:01", "C01:02", "C02:03"]
    assert [it.label for it in list_iterations("source", index, rows, ctx.stokes_labels, ctx)] == ["3C286", "3C48"]
    assert [it.label for it in list_iterations("stokes", index, rows, ctx.stokes_labels, ctx)] == ["RR", "LL"]
    first_half = rows[:20]  # 3C286 only
    assert [it.label for it in list_iterations("source", index, first_half, ctx.stokes_labels, ctx)] == ["3C286"]
    with pytest.raises(ValueError, match="one plot per one of"):
        list_iterations("scan", index, rows, ctx.stokes_labels, ctx)


def test_an_iteration_narrows_the_selection_to_its_rows_or_its_stokes():
    source = _source("iterations_narrow")
    baseline = iteration_source(source, Iteration("baseline", (1, 3), "C00:01-C02:03"))
    np.testing.assert_array_equal(baseline.row_indices, np.arange(1, 40, 3))  # every third row, from row 1
    antenna = iteration_source(source, Iteration("antenna", 3, "C02:03"))  # 1-3 and 2-3: two of every three rows
    assert antenna.n_rows == 26
    assert iteration_source(source, Iteration("source", 2, "3C48")).n_rows == 20
    stokes = iteration_source(source, Iteration("stokes", "LL", "LL"))
    assert stokes.n_rows == 40 and list(stokes.axis_selection["STOKES"]) == [1]
    assert stokes.ctx.stokes_labels == ("LL",) and source.ctx.stokes_labels == ("RR", "LL")


@pytest.mark.parametrize("by", ["baseline", "antenna", "source", "stokes"])
@pytest.mark.parametrize("y, x", [("amp", "freq_mhz"), ("v", "u")])
def test_one_pass_fills_every_iteration_as_a_pass_over_each_alone(by, y, x):
    source = _source(f"iterations_pass_{by}_{y}")
    plot = PlotSpec(y=y, x=x, colorize_by="stokes", show_flagged=True)
    iterations = list_iterations(by, source.index, source.row_indices, source.ctx.stokes_labels, source.ctx)
    extent = ((-1e9, 1e9), (-1e9, 1e9)) if y == "v" else ((399.5, 403.5), (0.0, 41.0))
    together = {it: GridReducer(replace(plot, iteration=it), *extent, 20, 30) for it in iterations}
    ranges = {it: RangeReducer(replace(plot, iteration=it), "y", source.ctx) for it in iterations}
    source.stream([*together.values(), *ranges.values()], read_data=True)
    for it in iterations:
        alone = iteration_source(source, it)
        grid, extent_alone = GridReducer(plot, *extent, 20, 30), RangeReducer(plot, "y", alone.ctx)
        alone.stream([grid, extent_alone], read_data=True)
        np.testing.assert_array_equal(together[it].layers, grid.layers)
        assert together[it].n_samples == grid.n_samples > 0
        assert (ranges[it].lo, ranges[it].hi) == (extent_alone.lo, extent_alone.hi)
        np.testing.assert_array_equal(rows_in_views(source, {replace(plot, iteration=it): extent}),
                                      rows_in_views(alone, {plot: extent}))


def test_a_page_holds_the_default_grid_or_the_smallest_within_it_for_fewer():
    default = (5, 6)
    assert page_layout(378, None, default) == (5, 6)
    assert page_layout(30, None, default) == (5, 6)
    assert page_layout(28, None, default) == (5, 6)
    assert page_layout(13, None, default) == (3, 5)
    assert page_layout(4, None, default) == (2, 2)
    assert page_layout(3, None, default) == (1, 3)
    assert page_layout(1, None, default) == (1, 1)
    assert page_layout(4, (1, 1), default) == (1, 1)  # as given
    assert page_layout(4, (2, 3), default) == (2, 3)


def _run(name, **options):
    path = _make_synthetic_file(make_scratch_dir(name) / "obs.fits")
    out = make_scratch_dir(f"{name}_out")
    request = PlotRequest(str(path), "amp-vs-freq", output_dir=str(out), no_highres_pdf=True, **options)
    return prepare(request, open_file(path)), out


def _pdf_pages(path) -> int:
    return len(re.findall(rb"/Type\s*/Page\b", path.read_bytes()))


def test_the_plots_go_on_one_grid_page_or_one_page_each():
    run, out = _run("iterations_grid", one_plot_per="baseline")
    written = save_outputs(run)
    assert [p.name for p in written if p.suffix == ".png"] == ["visplot_amp-vs-freq_page01.png"]  # 3 in 1 x 3
    assert _pdf_pages(out / "visplot_lowres.pdf") == 1
    run, out = _run("iterations_one_each", one_plot_per="baseline", page_grid="1,1")
    written = save_outputs(run)
    assert sorted(p.name for p in written if p.suffix == ".png") == [
        "visplot_amp-vs-freq_C00_01-C01_02.png", "visplot_amp-vs-freq_C00_01-C02_03.png",
        "visplot_amp-vs-freq_C01_02-C02_03.png"]
    assert _pdf_pages(out / "visplot_lowres.pdf") == 3
    run, out = _run("iterations_two_pages", one_plot_per="baseline", page_grid="1,2")
    written = save_outputs(run)
    assert [p.name for p in written if p.suffix == ".png"] == ["visplot_amp-vs-freq_page01.png",
                                                               "visplot_amp-vs-freq_page02.png"]
    plt.close("all")


def test_a_grid_page_names_its_plots_and_shares_its_axes():
    run, _ = _run("iterations_grid_page", one_plot_per="baseline", colorize_by="stokes")
    ((page, figure, grids),) = list(draw_pages(run))
    assert figure.page.grid == (1, 3) and figure.page.shared == (True, True)  # every plot's range: the default
    assert figure.fig.texts and figure.title.get_text().splitlines()[0] == (
        "Amplitude vs Frequency: 3 baselines, multiple sources")
    assert figure.panel.facts.filters == "3 baselines: C00:01-C01:02 to C01:02-C02:03"
    corners = [[t.get_text() for t in cell.ax.texts] for cell in figure.cells]
    assert corners == [["C00:01-C01:02"], ["C00:01-C02:03"], ["C01:02-C02:03"]]
    left_labels = [cell.ax.yaxis.get_tick_params()["labelleft"] for cell in figure.cells]
    assert left_labels == [True, False, False]  # one y range: tick labels on the first column only
    assert len({grid.y_extent for grid in grids.values()}) == 1
    assert sum(grid.n_samples for grid in grids.values()) == 40 * 4 * 2 - 1  # row 0's first RR flagged
    assert figure.status.get_text() == "319 samples from 40 rows"
    run, _ = _run("iterations_grid_rows_left_empty", one_plot_per="baseline", page_grid="4,1")
    ((_, figure, _),) = list(draw_pages(run))
    lowest = figure.cells[-1].ax.get_position().y0  # the third of four rows
    label_y = figure.xlabel.get_position()[1]
    assert (figure.panel.height_in + 0.1) / figure.fig.get_size_inches()[1] < label_y < lowest  # under the plots
    plt.close("all")


def test_each_plot_takes_its_own_range_or_one_from_every_plot():
    each, _ = _run("iterations_each", one_plot_per="source", y_range_from="each")
    ((_, figure, grids),) = list(draw_pages(each))
    own = {plot.iteration.label: grid.y_extent for plot, grid in grids.items()}
    assert own["3C286"][0] < own["3C48"][0] and own["3C286"][1] < own["3C48"][1]  # amp 1-20, then 21-40
    assert [cell.ax.yaxis.get_tick_params()["labelleft"] for cell in figure.cells] == [True, True]
    every, _ = _run("iterations_every", one_plot_per="source")
    ((_, _, grids),) = list(draw_pages(every))
    (plot,) = every.xy_plots
    whole = resolve_extents(every.source, [plot])[plot][1]
    assert all(grid.y_extent == pytest.approx(whole) for grid in grids.values())
    one_each, _ = _run("iterations_one_each_range", one_plot_per="source", page_grid="1,1")
    one_y = {page.plots[0].iteration.label: next(iter(grids.values())).y_extent
             for page, _, grids in draw_pages(one_each)}
    assert one_y["3C286"] == pytest.approx(own["3C286"])  # one plot a page: its own range by default
    plt.close("all")


def test_an_iterations_plot_draws_what_the_request_narrowed_by_its_filter_draws():
    run, _ = _run("iterations_equal", one_plot_per="source", page_grid="1,1")
    plots = {page.plots[0].iteration.label: grids[page.plots[0]] for page, _, grids in draw_pages(run)}
    narrowed, _ = _run("iterations_equal_narrowed", sources="3C48")
    (plot,) = narrowed.xy_plots
    figure = narrowed.xy_figures[plot]
    view = figure.set_view(*resolve_extents(narrowed.source, [plot])[plot])
    h, w = figure.grid_shape(150)
    factor = highres_dpi(150) // 150
    grids, _ = plot_grids(narrowed.source, [plot], {plot: view}, {plot: (h * factor, w * factor)})
    mine = plots["3C48"]
    assert (mine.x_extent, mine.y_extent, mine.height, mine.width) == (
        grids[plot].x_extent, grids[plot].y_extent, grids[plot].height, grids[plot].width)
    np.testing.assert_array_equal(mine.layers, grids[plot].layers)
    plt.close("all")


def test_a_one_plot_page_names_its_iteration_in_the_title_and_the_selection():
    run, _ = _run("iterations_titles", one_plot_per="baseline", colorize_by="stokes", page_grid="1,1")
    page, figure, grids = next(draw_pages(run))
    assert figure.ax.get_title().splitlines()[0] == "Amplitude vs Frequency: baseline C00:01-C01:02, multiple sources"
    assert figure.panel.facts.filters == "baseline C00:01-C01:02"
    (grid,) = grids.values()
    assert grid.n_samples == 14 * 4 * 2 - 1  # rows 0, 3, ..., 39 x 4 channels x RR, LL; row 0's first RR flagged
    _, left, _, _ = figure.panel._lines((), 8.0)
    assert dict(left)["Baselines"] == ["1 baseline, 2 antennas"]
    run, _ = _run("iterations_titles_filtered", one_plot_per="baseline", antennas="C00", page_grid="1,1")
    _, figure, _ = next(draw_pages(run))
    assert figure.panel.facts.filters == "baseline C00:01-C01:02, antennas C00"  # first: kept when cut
    run, _ = _run("iterations_titles_source", one_plot_per="source", page_grid="1,1")
    _, figure, _ = next(draw_pages(run))
    assert figure.ax.get_title().splitlines()[0] == "Amplitude vs Frequency: 3C286"  # the source names its plot
    plt.close("all")


def test_pages_drawn_in_several_batches_are_drawn_as_in_one(monkeypatch):
    import visplot.run as run_module

    run, _ = _run("iterations_batches", one_plot_per="baseline", page_grid="1,1")
    one = {page.name: grids[page.plots[0]].layers.copy() for page, _, grids in draw_pages(run)}
    monkeypatch.setattr(run_module, "host_total_memory_bytes", lambda: 1)
    reports = []
    several = {page.name: grids[page.plots[0]].layers
               for page, _, grids in draw_pages(run, report=lambda _, text: reports.append(text))}
    assert any(text.startswith("3 pages of 1 x 1, drawn in 3 batches of up to 1 ") for text in reports)
    for name, layers in one.items():
        np.testing.assert_array_equal(layers, several[name])
    assert page_batches(list("abcde"), 10, 25) == [["a", "b"], ["c", "d"], ["e"]]
    assert page_batches(list("ab"), 10, 5) == [["a"], ["b"]]  # one page at least
    plt.close("all")


def test_a_density_scale_can_come_from_other_counts():
    count = np.array([[1.0, 10.0, 0.0]])
    alone, scale_alone = density_opacity(count, "log", 100.0)
    shared, scale_shared = density_opacity(count, "log", 100.0, reference=np.array([1.0, 10.0, 100.0]))
    assert scale_alone.top == 10 and scale_shared.top == 100
    assert alone[0, 1] == 1.0 and shared[0, 1] < 1.0  # 10 is the top alone, halfway (log) on the shared scale


def test_page_options_are_checked():
    for options in (dict(page_grid="5,6"), dict(y_range_from="each")):
        with pytest.raises(RequestError, match="needs --one-plot-per"):
            check_request(PlotRequest("obs.fits", "amp-vs-freq", output_dir="out", **options))
    with pytest.raises(RequestError, match="--page-grid takes 'ROWS,COLS'"):
        check_request(PlotRequest("obs.fits", "amp-vs-freq", one_plot_per="baseline", page_grid="5x6",
                                  output_dir="out"))
    with pytest.raises(RequestError, match="each plot holds one Stokes product"):
        check_request(PlotRequest("obs.fits", "amp-vs-freq", one_plot_per="stokes", colorize_by="stokes",
                                  output_dir="out"))
    check_request(PlotRequest("obs.fits", "amp-vs-freq", one_plot_per="baseline"))  # the window shows pages
    with pytest.raises(RequestError, match="--locate lists the samples of one plot"):
        check_request(PlotRequest("obs.fits", "amp-vs-freq", one_plot_per="baseline", locate="0:1,0:1",
                                  locate_csv="a.csv", output_dir="out"))


def test_an_iteration_names_itself_and_its_file():
    iteration = Iteration("baseline", (1, 2), "C00:01-C01:02")
    assert iteration.text == "baseline C00:01-C01:02" and iteration.file_label == "C00_01-C01_02"
    assert Iteration("stokes", "RR", "RR").text == "Stokes RR"


def test_a_save_reports_its_writing_page_by_page_and_can_be_stopped_there():
    from visplot.run import Stopped

    def recorder(texts, stop_at=None):
        def factory(progress):
            def on_chunk(done):
                texts.append(progress.text(done))
                return not (stop_at and stop_at in texts[-1])
            return on_chunk
        return factory

    run, _ = _run("iterations_writing", one_plot_per="baseline", page_grid="1,1")
    texts = []
    save_outputs(run, progress=recorder(texts))
    writing = [t for t in texts if t.startswith("writing the pages")]
    assert [t.split(", ")[0] for t in writing] == [
        "writing the pages (PNG and PDF): 1 / 3 pages (33%)", "writing the pages (PNG and PDF): 2 / 3 pages (67%)",
        "writing the pages (PNG and PDF): 3 / 3 pages (100%)"]
    run, _ = _run("iterations_writing_one_plot")
    texts = []
    save_outputs(run, progress=recorder(texts))
    assert any(t.startswith("writing the PNGs and visplot_lowres.pdf: 1 / 1 plots (100%)") for t in texts)
    run, _ = _run("iterations_writing_stopped", one_plot_per="baseline", page_grid="1,1")
    with pytest.raises(Stopped, match="stopped while writing the pages"):
        save_outputs(run, progress=recorder([], stop_at="2 / 3 pages"))
    plt.close("all")
