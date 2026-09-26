import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np

from visplot.scatter_xy import scatter_xy


def test_scatter_xy_plots_all_points_by_default():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])

    fig = scatter_xy(x, y, xlabel="X", ylabel="Y")
    ax = fig.axes[0]

    assert len(ax.collections) == 1
    np.testing.assert_allclose(sorted(ax.collections[0].get_offsets()[:, 0]), x)
    assert ax.get_xlabel() == "X"
    assert ax.get_ylabel() == "Y"
    plt.close(fig)


def test_scatter_xy_excludes_flagged_points_by_default():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])
    weight = np.array([1.0, -1.0, 1.0])  # row 1 flagged

    fig = scatter_xy(x, y, weight=weight)
    ax = fig.axes[0]

    assert len(ax.collections[0].get_offsets()) == 2
    plt.close(fig)


def test_scatter_xy_shows_flagged_points_when_requested():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])
    weight = np.array([1.0, -1.0, 1.0])

    fig = scatter_xy(x, y, weight=weight, show_flagged=True)
    ax = fig.axes[0]

    assert len(ax.collections) == 2  # good points + flagged points
    assert len(ax.collections[1].get_offsets()) == 1
    assert ax.get_legend() is not None
    plt.close(fig)


def test_scatter_xy_colorize_by_creates_one_series_per_category_with_legend():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    labels = np.array(["RR", "RR", "LL", "LL"])

    fig = scatter_xy(x, y, colorize_by=labels)
    ax = fig.axes[0]

    assert len(ax.collections) == 2  # one scatter call per category
    legend_labels = {t.get_text() for t in ax.get_legend().get_texts()}
    assert legend_labels == {"RR", "LL"}
    plt.close(fig)


def test_scatter_xy_draws_into_a_given_axes_without_creating_a_new_figure():
    fig, ax = plt.subplots()
    x = np.array([1.0, 2.0])
    y = np.array([3.0, 4.0])

    returned_fig = scatter_xy(x, y, ax=ax)

    assert returned_fig is fig
    assert len(fig.axes) == 1  # no extra axes/figure created
    plt.close(fig)
