"""Density plots (T21): each pixel counts the samples landing in it, its hue
the mix of its categories' colors by their counts, its opacity rising with
log(count); the panel gives the scale."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from conftest import make_scratch_dir
from test_cli_visplot_output import _make_synthetic_file
from visplot.plot_panel import DENSITY_MIN_ALPHA, DensityScale
from visplot.plot_spec import PlotSpec
from visplot.quantities import QuantityContext
from visplot.request import PlotRequest
from visplot.run import open_file, prepare, save_outputs
from visplot.stream import GridReducer
from visplot.xy_figure import XYFigure, density_opacity, density_rgba, downsample_counts

CTX = QuantityContext(time_reference_jd=2459421.0, bunit="UNCALIB", stokes_labels=("RR", "LL"))
RED, BLUE = (1.0, 0.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0)


class _Samples:
    def __init__(self, x, y, code=None, flagged=None):
        self.arrays = (np.asarray(x, float), np.asarray(y, float), None if code is None else np.asarray(code),
                       None if flagged is None else np.asarray(flagged))

    def samples(self, plot):
        return self.arrays


def test_a_density_grid_counts_the_samples_in_each_pixel_by_category():
    plot = PlotSpec(y="amp", x="freq_mhz", colorize_by="stokes", style="density", show_flagged=True)
    grid = GridReducer(plot, (0.0, 4.0), (0.0, 4.0), height=4, width=4)
    grid.update(_Samples([0.5, 0.5, 0.5, 2.5, 2.5], [0.5, 0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1, 0],
                         [False, False, False, False, True]))
    assert grid.counts[0][0] == 2 and grid.counts[1][0] == 1 and grid.counts[1][2] == 1
    assert grid.flagged_counts[2] == 1 and grid.n_samples == 5
    assert grid.snapshot().counts[0][0] == 2
    np.testing.assert_array_equal(downsample_counts(grid.layers_2d(grid.counts[0]), 2), [[2, 0], [0, 0]])


def test_hues_mix_by_count_and_opacity_rises_with_log_count():
    shape = (1, 3)
    counts = {0: np.array([[1, 50, 0]]), 1: np.array([[0, 50, 100]])}
    rgba, scale = density_rgba(shape, counts, {0: RED, 1: BLUE}, top_percentile=100.0)
    assert (scale.kind, scale.top, scale.most) == ("log", 100, 100)
    assert tuple(rgba[0, 0]) == (255, 0, 0, round(DENSITY_MIN_ALPHA * 255))  # one sample: the least opacity
    assert tuple(rgba[0, 1, :3]) == (128, 0, 128) and rgba[0, 1, 3] == 255  # half and half: mixed, the top
    assert tuple(rgba[0, 2, :3]) == (0, 0, 255)
    flagged_rgba, _ = density_rgba(shape, counts, {0: RED, 1: BLUE}, np.array([[0, 0, 5]]), (0.0, 1.0, 0.0, 1.0))
    assert tuple(flagged_rgba[0, 2]) == (0, 255, 0, 255)  # flagged over the rest, on their own scale


def _floored():
    """Like amp vs uv distance: most pixels hold a few samples, a crowded floor holds nearly all."""
    count = np.ones((1, 200))
    count[0, :100] = 10  # half the pixels hold 10, half 1
    count[0, :10] = 1_000_000  # 5% of them, the floor
    return count


def test_the_log_scale_tops_out_at_the_chosen_percentile_and_beyond_it_is_full():
    count = _floored()
    alpha, scale = density_opacity(count, "log", 90.0)
    assert (scale.top, scale.most) == (10, 1_000_000)
    assert alpha[0, 50] == 1.0 and alpha[0, 0] == 1.0 and alpha[0, 150] == pytest.approx(DENSITY_MIN_ALPHA)
    _, at_most = density_opacity(count, "log", 100.0)
    assert at_most.top == 1_000_000


def test_the_histogram_scale_follows_each_pixels_rank():
    alpha, scale = density_opacity(_floored(), "histogram")
    assert (scale.kind, scale.median, scale.p90, scale.most) == ("histogram", 5, 10, 1_000_000)
    assert alpha[0, 150] == pytest.approx(DENSITY_MIN_ALPHA) and alpha[0, 0] == 1.0
    assert DENSITY_MIN_ALPHA < alpha[0, 50] < 1.0  # the middle rank: between
    assert scale.text() == ("1,000,000", "samples per pixel, histogram-equalized: half the pixels hold ≤ 5, "
                                         "90% ≤ 10")


def test_the_panel_states_the_density_scale():
    plot = PlotSpec(y="amp", x="freq_mhz", style="density")
    figure = XYFigure(plot, CTX)
    grid = GridReducer(plot, (0.0, 4.0), (0.0, 4.0), height=4, width=4)
    grid.update(_Samples([0.5] * 7 + [2.5], [0.5] * 7 + [2.5]))
    figure.show(grid, display_dpi=100)
    assert figure.panel.density_peak.get_text() == "7"  # the 95th percentile of 7 and 1, rounded up
    assert figure.panel.density_note.get_text() == "samples per pixel, log scale"
    labels = [label for label, _ in figure.panel._lines((), 8.0)[0]]
    assert labels == ["Density"]
    figure.panel.set_density_scale(DensityScale("log", 2_423_881, top=315, percentile=95.0))
    assert figure.panel.density_note.get_text() == (
        "samples per pixel, log scale; the 95th percentile, 2,423,881 at most")
    plt.close(figure.fig)


def test_a_density_plot_saves_from_the_command_lines_code():
    path = _make_synthetic_file(make_scratch_dir("density_save") / "obs.fits")
    out = make_scratch_dir("density_save_out")
    run = prepare(PlotRequest(str(path), "amp-vs-freq", colorize_by="stokes", style="density", output_dir=str(out),
                              no_highres_pdf=True), open_file(path))
    written = save_outputs(run)
    assert any(p.suffix == ".png" for p in written)
    (figure,) = run.xy_figures.values()
    assert int(figure.panel.density_peak.get_text().replace(",", "")) >= 1
    assert figure.status.get_text().startswith("319 samples")  # one of 320 flagged, left out
    plt.close("all")
