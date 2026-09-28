import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pytest

from visplot.axis_scale import SCALE_NAMES, AxisScale


@pytest.mark.parametrize("name", SCALE_NAMES)
def test_binning_coordinate_lines_up_with_the_axis_matplotlib_draws(name):
    scale = AxisScale(name, linear_width=0.5)
    lo, hi = (0.01, 1000.0) if name == "log" else (-300.0, 1000.0)
    values = np.array([0.02, 0.3, 2.0, 40.0, 700.0]) if name == "log" else np.array([-250.0, -0.2, 0.0, 0.4, 3.0, 900.0])

    fig, ax = plt.subplots()
    ax.set_xscale(**scale.mpl_kwargs())
    ax.set_xlim(lo, hi)
    fig.canvas.draw()
    display = ax.transData.transform(np.column_stack([values, np.ones_like(values)]))
    axes_fraction = ax.transAxes.inverted().transform(display)[:, 0]

    t = scale.forward(values)
    t_lo, t_hi = scale.forward(np.array([lo, hi]))
    np.testing.assert_allclose((t - t_lo) / (t_hi - t_lo), axes_fraction, atol=1e-9)
    plt.close(fig)


@pytest.mark.parametrize("name", SCALE_NAMES)
def test_inverse_undoes_forward(name):
    scale = AxisScale(name)
    values = np.array([0.5, 3.0, 1e4]) if name == "log" else np.array([-1e4, -2.0, 0.0, 0.7, 5e3])
    np.testing.assert_allclose(scale.inverse(scale.forward(values)), values, rtol=1e-12, atol=1e-12)


def test_log_scale_marks_non_positive_values_invalid():
    np.testing.assert_array_equal(AxisScale("log").valid(np.array([-1.0, 0.0, 2.0])), [False, False, True])
    assert AxisScale("symlog").valid(np.array([-1.0])) is None


def test_rejects_unknown_scales_and_non_positive_widths():
    with pytest.raises(ValueError, match="unknown axis scale"):
        AxisScale("sqrt")
    with pytest.raises(ValueError, match="positive"):
        AxisScale("symlog", linear_width=0.0)
