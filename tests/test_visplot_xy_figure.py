import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pytest

from visplot.plot_spec import PlotSpec
from visplot.quantities import QuantityContext
from visplot.stream import FLAGGED_LAYER, GridReducer
from visplot.xy_figure import (
    DENSE_MARKER_THRESHOLD,
    XYFigure,
    auto_point_size,
    auto_square_marker,
    downsample_layers,
    draw_markers,
    layers_to_rgba,
    marker_offsets,
    marker_radius_px,
)

CTX = QuantityContext(time_reference_jd=2459421.0, bunit="UNCALIB", stokes_labels=("RR", "LL"))


def test_auto_point_size_shrinks_with_sample_count():
    assert [auto_point_size(n) for n in (10, 1_000, 1_001, 100_000, 1_000_000, 1_000_001)] == [20, 20, 4, 4, 1, 0.25]


def test_markers_are_square_only_when_dense():
    assert not auto_square_marker(DENSE_MARKER_THRESHOLD)
    assert auto_square_marker(DENSE_MARKER_THRESHOLD + 1)


def test_marker_radius_follows_the_scatter_area_convention():
    assert marker_radius_px(4.0, 72) == 1.0  # diameter sqrt(4) = 2 points = 2 pixels at 72 dpi


def test_marker_offsets_disk_and_square():
    assert marker_offsets(0.5, square=False) == [(0, 0)]
    assert len(marker_offsets(1.0, square=False)) == 5
    assert len(marker_offsets(1.0, square=True)) == 9


def test_draw_markers_stamps_each_occupied_pixel_higher_layer_wins():
    layers = np.zeros((5, 5), dtype=np.int16)
    layers[2, 1] = 1
    layers[2, 3] = 2
    out = draw_markers(layers, marker_offsets(1.0, square=True))
    assert out[2, 2] == 2  # both markers cover (2, 2); layer 2 paints over layer 1
    assert out[1, 0] == 1 and out[3, 4] == 2
    assert out[0, 0] == 0


def test_downsample_takes_the_top_layer_of_each_block():
    layers = np.zeros((4, 4), dtype=np.int16)
    layers[0, 0] = 1
    layers[1, 1] = 3
    layers[3, 2] = FLAGGED_LAYER
    np.testing.assert_array_equal(downsample_layers(layers, 2), [[3, 0], [0, FLAGGED_LAYER]])


def test_layers_to_rgba_colors_each_layer_and_leaves_empty_transparent():
    layers = np.array([[0, 1]], dtype=np.int16)
    rgba = layers_to_rgba(layers, {1: (1.0, 0.0, 0.0, 1.0)})
    assert rgba[0, 0, 3] == 0
    assert tuple(rgba[0, 1]) == (255, 0, 0, 255)


def test_xy_figure_labels_title_and_grid_shape():
    plot = PlotSpec(y="amp", x="freq_mhz")
    fig = XYFigure(plot, CTX, sources=["3C286"], telescope="GMRT", source_path="/d/obs.fits")
    assert fig.ax.get_xlabel() == "Frequency (MHz)"
    assert fig.ax.get_ylabel() == "Amplitude (UNCALIB)"
    assert fig.ax.get_title() == "Amplitude vs Frequency: 3C286\n(GMRT, file: obs.fits)"
    h150, w150 = fig.grid_shape(150)
    h600, w600 = fig.grid_shape(600)
    assert abs(h600 - 4 * h150) <= 2 and abs(w600 - 4 * w150) <= 2
    plt.close(fig.fig)


def test_xy_figure_shows_a_grid_as_an_image_with_its_extent():
    plot = PlotSpec(y="amp", x="stokes", colorize_by="stokes")
    grid = GridReducer(plot, (-0.5, 1.5), (0.0, 10.0), height=4, width=4)
    grid.layers[:] = 0
    grid.layers[0] = 1
    grid.layers[3] = 2
    grid.seen_codes = {0, 1}
    grid.n_samples = 2
    fig = XYFigure(plot, CTX)
    fig.show(grid, display_dpi=100)
    assert fig.image.get_extent() == [-0.5, 1.5, 0.0, 10.0]
    assert fig.ax.get_legend() is None  # the key is in the panel under the plot
    assert [label for label, _ in fig.panel.key_entries(grid.seen_codes)] == ["RR", "LL"]
    assert [t.get_text() for t in fig.ax.get_xticklabels()] == ["RR", "LL"]
    plt.close(fig.fig)


def test_xy_figure_non_linear_axis_uses_the_scale_and_pins_the_image_to_the_axes():
    plot = PlotSpec(y="amp", x="freq_mhz", y_scale="log")
    grid = GridReducer(plot, (100.0, 200.0), (1.0, 1000.0), height=4, width=4)
    grid.layers[5] = 1
    grid.n_samples = 1
    fig = XYFigure(plot, CTX)
    fig.show(grid, display_dpi=100)
    assert fig.ax.get_yscale() == "log"
    assert fig.image.get_transform() == fig.ax.transAxes
    assert fig.image.get_extent() == [0.0, 1.0, 0.0, 1.0]
    fig.ax.set_ylim(10.0, 100.0)  # a zoom: the pinned image no longer matches
    fig.hide_if_view_moved(grid)
    assert not fig.image.get_visible()
    plt.close(fig.fig)


def test_grid_summary_mentions_samples_left_out():
    from visplot.xy_figure import grid_summary

    grid = GridReducer(PlotSpec(y="amp", x="freq_mhz"), (0.0, 1.0), (0.0, 1.0), height=1, width=1)
    grid.n_samples, grid.n_outside = 1200, 3
    assert grid_summary(grid, 40) == "1,200 samples from 40 rows; 3 outside the axis ranges, left out"


def test_auto_aspect_is_equal_only_for_the_same_kind_of_quantity_on_linear_axes():
    assert PlotSpec(y="v_klambda", x="u_klambda").equal_aspect
    assert PlotSpec(y="imag", x="real").equal_aspect
    assert not PlotSpec(y="amp", x="uvdist_klambda").equal_aspect
    assert not PlotSpec(y="ha_h", x="time_h").equal_aspect  # both in hours, different kinds
    assert not PlotSpec(y="v_klambda", x="u_klambda", y_scale="symlog").equal_aspect
    assert PlotSpec(y="amp", x="time_h", aspect="equal").equal_aspect
    assert not PlotSpec(y="v_klambda", x="u_klambda", aspect="free").equal_aspect


def test_set_view_with_equal_aspect_gives_equal_pixels_per_unit():
    plot = PlotSpec(y="v_klambda", x="u_klambda")
    fig = XYFigure(plot, CTX, figsize=(8, 6))
    (x0, x1), (y0, y1) = fig.set_view((-10.0, 10.0), (-2.0, 2.0))
    bbox = fig.ax.get_window_extent()
    assert (x1 - x0) / bbox.width == pytest.approx((y1 - y0) / bbox.height, rel=1e-3)
    assert (x0, x1) == pytest.approx((-10.0, 10.0))  # never narrowed: no data cut off
    assert (y1 - y0) > 4.0  # the y range widened instead
    free = XYFigure(PlotSpec(y="amp", x="uvdist_klambda"), CTX)
    assert free.set_view((0.0, 40.0), (0.0, 5.0)) == ((0.0, 40.0), (0.0, 5.0))
    plt.close(fig.fig)


def test_axes_are_labelled_in_their_units_and_clock_time_as_a_time_of_day():
    ctx = QuantityContext(time_reference_jd=2459421.2, bunit="UNCALIB", time_zone="Asia/Kolkata")
    fig = XYFigure(PlotSpec(y="phase", x="time", x_unit="local", y_unit="rad"), ctx)
    assert fig.ax.get_xlabel() == "Time (IST, UTC+05:30; day 0 = 2021-07-25)"
    assert fig.ax.get_ylabel() == "Phase (rad)"
    fig.set_view((22.0, 26.0), (-3.2, 3.2))
    fig.fig.canvas.draw()
    labels = fig.ax.get_xticklabels()
    assert [t.get_text() for t in labels] == [  # AIPS's day/time form, tilted, up to 10 ticks
        "0/22:00:00", "0/22:30:00", "0/23:00:00", "0/23:30:00", "1/00:00:00", "1/00:30:00", "1/01:00:00",
        "1/01:30:00", "1/02:00:00"]
    assert all(t.get_rotation() == 30.0 for t in labels)
    plt.close(fig.fig)


def test_the_time_format_writes_the_clock_axis_and_room_is_left_for_tilted_labels():
    ctx = QuantityContext(time_reference_jd=2459421.2, bunit="UNCALIB", reference_date_jd=2459419.5)  # 2021-07-24
    iso = XYFigure(PlotSpec(y="amp", x="time"), ctx, time_format="iso")
    iso.set_view((40.0, 44.0), (0.0, 1.0))
    iso.fig.canvas.draw()
    assert iso.ax.get_xticklabels()[0].get_text() == "2021-07-25 16:00:00"
    tod = XYFigure(PlotSpec(y="amp", x="time"), ctx, time_format="hh:mm:ss")
    tod.set_view((40.0, 44.0), (0.0, 1.0))
    tod.fig.canvas.draw()
    assert [t.get_text() for t in tod.ax.get_xticklabels()][:2] == ["1/16:00:00", "16:30:00"]
    # the axes rise by the tilted labels' extra height: more for iso's longer labels
    level = XYFigure(PlotSpec(y="amp", x="freq"), ctx)
    assert level.ax.get_position().y0 < tod.ax.get_position().y0 < iso.ax.get_position().y0
    for figure in (iso, tod, level):
        plt.close(figure.fig)


def test_equal_aspect_gives_a_square_box_with_the_same_span_on_both_axes():
    fig = XYFigure(PlotSpec(y="v", x="u", mirror=True), CTX, figsize=(8, 6))
    (x0, x1), (y0, y1) = fig.set_view((-20.0, 20.0), (-35.0, 35.0))
    fig.fig.canvas.draw()
    box = fig.ax.get_window_extent()
    assert box.width == pytest.approx(box.height, rel=1e-3)
    assert (x0, x1) == pytest.approx((-35.0, 35.0)) and (y0, y1) == pytest.approx((-35.0, 35.0))
    height, width = fig.grid_shape(150)
    assert height == width
    fig.equal_override = False  # off: the data's own ranges in the full-width box
    assert fig.set_view(*fig.view_request) == ((-20.0, 20.0), (-35.0, 35.0))
    fig.fig.canvas.draw()
    assert fig.ax.get_window_extent().width > fig.ax.get_window_extent().height
    plt.close(fig.fig)
