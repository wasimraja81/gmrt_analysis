"""Pages (T26): one plot per baseline, antenna, source or Stokes product of
the selection."""

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
from visplot.pages import Page, list_pages, page_source
from visplot.plot_spec import PlotSpec
from visplot.quantities import QuantityContext
from visplot.request import PlotRequest
from visplot.run import (RequestError, check_request, draw_pages, highres_dpi, open_file, page_batches, prepare,
                         save_outputs)
from visplot.stream import GridReducer, RangeReducer
from visplot.xy_session import XYSource, plot_grids, resolve_extents, rows_in_views


def _source(name):
    path = _make_synthetic_file(make_scratch_dir(name) / "obs.fits")
    index = load_row_index(default_row_index_path(path))
    ctx = QuantityContext(time_reference_jd=float(index.jd.min()), bunit="UNCALIB", stokes_labels=("RR", "LL"),
                          antenna_names={1: "C00:01", 2: "C01:02", 3: "C02:03"}, source_names={1: "3C286", 2: "3C48"})
    return XYSource(path, index, select_rows(index).row_indices, None, ctx, chunk_bytes=64 * 1024)


def test_the_pages_are_the_values_the_selection_holds_in_order():
    source = _source("pages_list")
    index, rows, ctx = source.index, source.row_indices, source.ctx
    assert [p.label for p in list_pages("baseline", index, rows, ctx.stokes_labels, ctx)] == [
        "C00:01-C01:02", "C00:01-C02:03", "C01:02-C02:03"]
    assert [p.label for p in list_pages("antenna", index, rows, ctx.stokes_labels, ctx)] == [
        "C00:01", "C01:02", "C02:03"]
    assert [p.label for p in list_pages("source", index, rows, ctx.stokes_labels, ctx)] == ["3C286", "3C48"]
    assert [p.label for p in list_pages("stokes", index, rows, ctx.stokes_labels, ctx)] == ["RR", "LL"]
    first_half = rows[:20]  # 3C286 only
    assert [p.label for p in list_pages("source", index, first_half, ctx.stokes_labels, ctx)] == ["3C286"]
    with pytest.raises(ValueError, match="pages are by one of"):
        list_pages("scan", index, rows, ctx.stokes_labels, ctx)


def test_a_page_narrows_the_selection_to_its_rows_or_its_stokes():
    source = _source("pages_narrow")
    baseline = page_source(source, Page("baseline", (1, 3), "C00:01-C02:03"))
    np.testing.assert_array_equal(baseline.row_indices, np.arange(1, 40, 3))  # every third row, from row 1
    antenna = page_source(source, Page("antenna", 3, "C02:03"))  # 1-3 and 2-3: two of every three rows
    assert antenna.n_rows == 26
    assert page_source(source, Page("source", 2, "3C48")).n_rows == 20
    stokes = page_source(source, Page("stokes", "LL", "LL"))
    assert stokes.n_rows == 40 and list(stokes.axis_selection["STOKES"]) == [1]
    assert stokes.ctx.stokes_labels == ("LL",) and source.ctx.stokes_labels == ("RR", "LL")


@pytest.mark.parametrize("by", ["baseline", "antenna", "source", "stokes"])
@pytest.mark.parametrize("y, x", [("amp", "freq_mhz"), ("v", "u")])
def test_one_pass_fills_every_page_as_a_pass_over_each_page_alone(by, y, x):
    source = _source(f"pages_pass_{by}_{y}")
    plot = PlotSpec(y=y, x=x, colorize_by="stokes", show_flagged=True)
    pages = list_pages(by, source.index, source.row_indices, source.ctx.stokes_labels, source.ctx)
    extent = ((-1e9, 1e9), (-1e9, 1e9)) if y == "v" else ((399.5, 403.5), (0.0, 41.0))
    together = {page: GridReducer(replace(plot, page=page), *extent, 20, 30) for page in pages}
    ranges = {page: RangeReducer(replace(plot, page=page), "y", source.ctx) for page in pages}
    source.stream([*together.values(), *ranges.values()], read_data=True)
    for page in pages:
        alone = page_source(source, page)
        grid, extent_alone = GridReducer(plot, *extent, 20, 30), RangeReducer(plot, "y", alone.ctx)
        alone.stream([grid, extent_alone], read_data=True)
        np.testing.assert_array_equal(together[page].layers, grid.layers)
        assert together[page].n_samples == grid.n_samples > 0
        assert (ranges[page].lo, ranges[page].hi) == (extent_alone.lo, extent_alone.hi)
        np.testing.assert_array_equal(rows_in_views(source, {replace(plot, page=page): extent}),
                                      rows_in_views(alone, {plot: extent}))


def _run(name, **options):
    path = _make_synthetic_file(make_scratch_dir(name) / "obs.fits")
    out = make_scratch_dir(f"{name}_out")
    request = PlotRequest(str(path), "amp-vs-freq", output_dir=str(out), no_highres_pdf=True, **options)
    return prepare(request, open_file(path)), out


def test_pages_save_a_png_per_page_and_a_pdf_page_each():
    run, out = _run("pages_save", pages_by="baseline")
    written = save_outputs(run)
    assert sorted(p.name for p in written if p.suffix == ".png") == [
        "visplot_amp-vs-freq_C00_01-C01_02.png", "visplot_amp-vs-freq_C00_01-C02_03.png",
        "visplot_amp-vs-freq_C01_02-C02_03.png"]
    assert len(re.findall(rb"/Type\s*/Page\b", (out / "visplot_lowres.pdf").read_bytes())) == 3
    plt.close("all")


def test_each_page_names_its_page_in_the_title_and_the_selection():
    run, _ = _run("pages_titles", pages_by="baseline", colorize_by="stokes")
    plot, figure, grid = next(draw_pages(run))
    assert figure.ax.get_title().splitlines()[0] == "Amplitude vs Frequency: baseline C00:01-C01:02, multiple sources"
    assert figure.panel.facts.filters == "page: baseline C00:01-C01:02"
    assert grid.n_samples == 14 * 4 * 2 - 1  # rows 0, 3, ..., 39 x 4 channels x RR, LL; row 0's first RR flagged
    _, left, _, _ = figure.panel._lines((), 8.0)
    assert dict(left)["Baselines"] == ["1 baseline, 2 antennas"]
    run, _ = _run("pages_titles_filtered", pages_by="baseline", antennas="C00")
    _, figure, _ = next(draw_pages(run))
    assert figure.panel.facts.filters == "page: baseline C00:01-C01:02, antennas C00"  # first: kept when cut
    run, _ = _run("pages_titles_source", pages_by="source")
    _, figure, _ = next(draw_pages(run))
    assert figure.ax.get_title().splitlines()[0] == "Amplitude vs Frequency: 3C286"  # the source names its page


def test_a_page_draws_what_the_request_narrowed_by_its_filter_draws():
    run, _ = _run("pages_equal", pages_by="source")
    pages = {plot.page.label: grid for plot, _, grid in draw_pages(run)}
    narrowed, _ = _run("pages_equal_narrowed", sources="3C48")
    (plot,) = narrowed.xy_plots
    figure = narrowed.xy_figures[plot]
    view = figure.set_view(*resolve_extents(narrowed.source, [plot])[plot])
    h, w = figure.grid_shape(150)
    factor = highres_dpi(150) // 150
    grids, _ = plot_grids(narrowed.source, [plot], {plot: view}, {plot: (h * factor, w * factor)})
    page = pages["3C48"]
    assert (page.x_extent, page.y_extent, page.height, page.width) == (
        grids[plot].x_extent, grids[plot].y_extent, grids[plot].height, grids[plot].width)
    np.testing.assert_array_equal(page.layers, grids[plot].layers)
    plt.close("all")


def test_pages_take_their_own_ranges_or_one_common_range():
    own, _ = _run("pages_own", pages_by="source")
    own_y = {plot.page.label: grid.y_extent for plot, _, grid in draw_pages(own)}
    assert own_y["3C286"][0] < own_y["3C48"][0] and own_y["3C286"][1] < own_y["3C48"][1]  # amp 1-20, then 21-40
    common, _ = _run("pages_common", pages_by="source", y_page_range="common")
    common_y = {plot.page.label: grid.y_extent for plot, _, grid in draw_pages(common)}
    (plot,) = common.xy_plots
    whole = resolve_extents(common.source, [plot])[plot][1]
    assert common_y["3C286"] == common_y["3C48"] == pytest.approx(whole)
    plt.close("all")


def test_pages_drawn_in_several_batches_are_drawn_as_in_one(monkeypatch):
    import visplot.run as run_module

    run, _ = _run("pages_batches", pages_by="baseline")
    one = {plot.page: grid.layers.copy() for plot, _, grid in draw_pages(run)}
    monkeypatch.setattr(run_module, "host_total_memory_bytes", lambda: 1)
    reports = []
    several = {plot.page: grid.layers for plot, _, grid in draw_pages(run, report=lambda _, text: reports.append(text))}
    assert any(text.startswith("3 pages drawn in 3 batches of up to 1 ") for text in reports)
    for page, layers in one.items():
        np.testing.assert_array_equal(layers, several[page])
    assert page_batches(list("abcde"), 10, 25) == [["a", "b"], ["c", "d"], ["e"]]
    assert page_batches(list("ab"), 10, 5) == [["a"], ["b"]]  # one page at least
    plt.close("all")


def test_page_options_are_checked():
    with pytest.raises(RequestError, match="needs --pages-by"):
        check_request(PlotRequest("obs.fits", "amp-vs-freq", y_page_range="common", output_dir="out"))
    with pytest.raises(RequestError, match="the window's pages are not built yet"):
        check_request(PlotRequest("obs.fits", "amp-vs-freq", pages_by="baseline"))
    with pytest.raises(RequestError, match="--locate lists the samples of one plot"):
        check_request(PlotRequest("obs.fits", "amp-vs-freq", pages_by="baseline", locate="0:1,0:1",
                                  locate_csv="a.csv", output_dir="out"))


def test_a_page_names_itself_and_its_file():
    page = Page("baseline", (1, 2), "C00:01-C01:02")
    assert page.text == "baseline C00:01-C01:02" and page.file_label == "C00_01-C01_02"
    assert Page("stokes", "RR", "RR").text == "Stokes RR"
