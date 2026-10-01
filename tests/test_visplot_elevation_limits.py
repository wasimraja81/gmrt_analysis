"""Points beyond a plot's elevation limits (T47): kept at their values,
drawn as ▼ (below) or ▲ (above) in their own color, placed on the grid's
edge where they lie beyond it, and stated in the panel's Limits line."""

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from instruments.elevation_limits import elevation_limits_deg
from visplot.plot_spec import PlotSpec
from visplot.quantities import QuantityContext
from visplot.run import with_elevation_limits
from visplot.stream import GridReducer
from visplot.xy_figure import XYFigure

CTX = QuantityContext(time_reference_jd=2459421.0, source_names={0: "3C286", 1: "MOON0520", 2: "DA240"})
PLOT = PlotSpec(y="el", x="time_h", colorize_by="source", apply_flags=False, y_range=(0.0, 90.0),
                limits=("y", 15.0, 110.0))


class _Samples:
    """Stands in for a chunk's values: the samples a plot draws."""

    def __init__(self, x, y, code):
        self.arrays = (np.asarray(x, float), np.asarray(y, float), np.asarray(code), None)

    def samples(self, plot):
        return self.arrays


def _grid():
    grid = GridReducer(PLOT, (0.0, 10.0), (0.0, 90.0), height=90, width=10)
    # one point inside the limits; two below (one also below the horizon, under the grid); one above 110
    grid.update(_Samples([1.5, 3.5, 5.5, 7.5], [50.0, 10.0, -5.0, 120.0], [0, 1, 1, 2]))
    return grid


def test_the_limits_come_from_the_telescope():
    assert elevation_limits_deg("GMRT") == (15.0, 110.0)
    assert elevation_limits_deg("other") == (0.0, 90.0)  # the horizon and the zenith


def test_points_beyond_the_limits_have_grids_of_their_own_at_the_edge_where_beyond_it():
    grid = _grid()
    assert grid.n_samples == 4 and grid.n_outside == 0
    below, above = grid.layers_2d(grid.below), grid.layers_2d(grid.above)
    assert below[10, 3] == 2 and below[0, 5] == 2  # MOON0520 (code 1): at 10 degrees, and -5 on the bottom edge
    assert above[89, 7] == 3  # DA240 at 120 degrees, on the top edge
    assert grid.layers_2d()[50, 1] == 1 and grid.layers_2d().sum() == 1  # 3C286 alone in the main grid
    assert grid.beyond == {"below": {1: [2, -5.0]}, "above": {2: [1, 120.0]}}
    assert grid.snapshot().beyond == grid.beyond


def test_the_panel_states_them_in_the_warning_color_and_the_triangles_are_drawn():
    figure = XYFigure(PLOT, CTX)
    assert figure.limits_warning is None and figure.panel.limits.get_text() == "none below 15° or above 110°"
    figure.set_view((0.0, 10.0), (0.0, 90.0))
    figure.show(_grid(), display_dpi=100)
    assert figure.limits_warning == (
        "below 15°: 2 points (MOON0520 2), lowest -5.0°; above 110°: 1 point (DA240 1), highest 120.0°")
    assert figure.panel.limits.get_children()[0].get_color() == figure.theme.warning
    alpha = figure.image.get_array()[..., 3]
    assert alpha[:6, 5].sum() > 255  # the ▼ on the bottom edge stands above it, over several rows
    plt.close(figure.fig)


def test_an_elevation_axis_takes_the_limits_in_its_unit_and_draws_them():
    el = with_elevation_limits(PlotSpec(y="el", x="time", y_unit="rad"), "GMRT")
    assert el.limits == pytest.approx(("y", math.radians(15.0), math.radians(110.0)))
    assert [(a, round(v, 6)) for a, v in el.reference_lines] == [("y", round(math.radians(15.0), 6)),
                                                                 ("y", round(math.radians(110.0), 6))]
    assert with_elevation_limits(PlotSpec(y="amp", x="el", x_unit="deg"), "GMRT").limits == ("x", 15.0, 110.0)
    assert with_elevation_limits(PlotSpec(y="amp", x="time"), "GMRT").limits is None
    other = with_elevation_limits(PlotSpec(y="el", x="time", y_unit="deg"), "other")
    assert other.limits == ("y", 0.0, 90.0) and other.reference_lines == ()  # lines only for known limits


def test_an_elevation_axis_shows_the_sky_unless_a_range_is_given():
    assert with_elevation_limits(PlotSpec(y="el", x="time", y_unit="deg"), "GMRT").y_range == (0.0, 90.0)
    rad = with_elevation_limits(PlotSpec(y="el", x="time", y_unit="rad"), "GMRT")
    assert rad.y_range == pytest.approx((0.0, math.pi / 2))
    amp_el = with_elevation_limits(PlotSpec(y="amp", x="el", x_unit="deg"), "GMRT")
    assert amp_el.x_range == (0.0, 90.0) and amp_el.y_range is None  # the other axis from the data
    given = with_elevation_limits(PlotSpec(y="el", x="time", y_unit="deg", y_range=(10.0, 40.0)), "GMRT")
    assert given.y_range == (10.0, 40.0)


def test_a_fixed_elevation_range_holds_before_the_first_drawing():
    from visplot.plot_spec import expand_plot_name

    el, _ = expand_plot_name("az-el-range", PlotSpec(y="", x=""))
    figure = XYFigure(with_elevation_limits(el, "GMRT"), CTX)
    figure.fig.canvas.draw()
    assert figure.ax.get_ylim() == (0.0, 90.0)  # the 110-degree line outside it does not stretch the empty axes
    plt.close(figure.fig)
