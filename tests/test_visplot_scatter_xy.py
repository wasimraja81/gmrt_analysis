import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np

from visplot.scatter_xy import DENSE_MARKER_THRESHOLD, auto_marker, auto_point_size, scatter_xy


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


def test_scatter_xy_mirror_also_plots_the_negated_points_without_a_duplicate_legend_entry():
    x = np.array([1.0, 2.0])
    y = np.array([3.0, 4.0])

    fig = scatter_xy(x, y, mirror=True, color="tab:blue")
    ax = fig.axes[0]

    assert len(ax.collections) == 2  # original points + mirrored points
    np.testing.assert_allclose(sorted(ax.collections[0].get_offsets()[:, 0]), sorted(x))
    np.testing.assert_allclose(sorted(ax.collections[1].get_offsets()[:, 0]), sorted(-x))
    assert ax.get_legend() is None  # no label was given, so no legend either


def test_scatter_xy_mirror_with_colorize_by_keeps_one_legend_entry_per_category():
    x = np.array([1.0, 2.0])
    y = np.array([3.0, 4.0])
    labels = np.array(["RR", "LL"])

    fig = scatter_xy(x, y, colorize_by=labels, mirror=True)
    ax = fig.axes[0]

    assert len(ax.collections) == 4  # 2 categories x (original + mirrored)
    legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert legend_labels == ["LL", "RR"]  # not doubled by the mirrored points (sorted, not insertion order)


def test_scatter_xy_draws_into_a_given_axes_without_creating_a_new_figure():
    fig, ax = plt.subplots()
    x = np.array([1.0, 2.0])
    y = np.array([3.0, 4.0])

    returned_fig = scatter_xy(x, y, ax=ax)

    assert returned_fig is fig
    assert len(fig.axes) == 1  # no extra axes/figure created
    plt.close(fig)


def test_auto_point_size_shrinks_with_point_count():
    assert auto_point_size(10) == 20.0
    assert auto_point_size(1_000) == 20.0
    assert auto_point_size(1_001) == 4.0
    assert auto_point_size(100_000) == 4.0
    assert auto_point_size(1_000_000) == 1.0
    assert auto_point_size(1_000_001) == 0.25


def test_scatter_xy_picks_marker_size_from_point_count_when_not_given():
    x = np.arange(5000, dtype=float)

    fig = scatter_xy(x, x)

    assert fig.axes[0].collections[0].get_sizes()[0] == auto_point_size(5000)
    plt.close(fig)


def test_scatter_xy_counts_mirrored_points_when_picking_marker_size():
    x = np.arange(800, dtype=float)  # 800 drawn, 1600 with the mirror

    fig = scatter_xy(x, x, mirror=True)

    assert fig.axes[0].collections[0].get_sizes()[0] == auto_point_size(1600)
    plt.close(fig)


def test_scatter_xy_explicit_point_size_overrides_the_automatic_one():
    x = np.arange(5000, dtype=float)

    fig = scatter_xy(x, x, point_size=30.0)

    assert fig.axes[0].collections[0].get_sizes()[0] == 30.0
    plt.close(fig)


def test_auto_marker_is_square_only_above_the_dense_threshold():
    assert auto_marker(DENSE_MARKER_THRESHOLD) == "o"
    assert auto_marker(DENSE_MARKER_THRESHOLD + 1) == "s"


def test_scatter_xy_uses_squares_for_a_dense_plot_and_crosses_for_flagged_points():
    n = DENSE_MARKER_THRESHOLD + 1
    x = np.arange(n, dtype=float)
    weight = np.ones(n)
    weight[0] = -1.0

    sparse_fig = scatter_xy(x[:10], x[:10])
    dense_fig = scatter_xy(x, x, weight=weight, show_flagged=True)

    def _vertex_count(collection):
        return len(collection.get_paths()[0].vertices)

    circle = sparse_fig.axes[0].collections[0]
    square, flagged = dense_fig.axes[0].collections
    assert _vertex_count(square) < _vertex_count(circle)  # a square has far fewer vertices than a circle
    assert _vertex_count(flagged) == 4  # "x": two line segments
    plt.close(sparse_fig)
    plt.close(dense_fig)
