"""A single generic scatter plot: any x against any y, with optional
per-point category coloring and flagged-point handling. This is the one
function every "quantity vs quantity" visibility plot (amp vs freq, uvdist
vs freq, real vs u, u vs v, w vs time, ...) is built from -- the quantities
themselves come from `derived_quantities.compute_quantity`.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Marker area (the scatter "s" parameter, points^2) by number of points
# drawn: (max_points, size), first match wins. Denser plots get smaller
# markers so points don't paint over each other; 0.25 points^2 is a ~0.5pt
# marker, about one pixel at 150 dpi, below which nothing is gained.
AUTO_POINT_SIZE_STEPS = ((1_000, 20.0), (100_000, 4.0), (1_000_000, 1.0))
AUTO_POINT_SIZE_FLOOR = 0.25
# Above this many points, markers are squares (circles below it): at the
# sizes used there (1 points^2 or less) a square reads as a point, and it
# renders faster (a filled rectangle is cheaper to draw than a curved outline) --
# measured ~40% faster in a PDF viewer at 520k points. CASA plotms's
# "autoscaling" symbol shape switches to pixels at high counts likewise.
DENSE_MARKER_THRESHOLD = 100_000
DEFAULT_LINEWIDTHS = 0.0
DEFAULT_COLOR = "tab:blue"
# A qualitative (categorical) palette -- matplotlib's own default cycle,
# good up to 10 distinct categories, comfortably covering the 2-4
# polarizations/correlations one plot would ever colorize by.
CATEGORY_COLORMAP = "tab10"
FLAGGED_COLOR = "lightcoral"


def auto_point_size(n_points: int) -> float:
    """Marker area (points^2) for a scatter of `n_points` -- see
    `AUTO_POINT_SIZE_STEPS`."""
    for max_points, size in AUTO_POINT_SIZE_STEPS:
        if n_points <= max_points:
            return size
    return AUTO_POINT_SIZE_FLOOR


def auto_marker(n_points: int) -> str:
    """Square above `DENSE_MARKER_THRESHOLD` points, circle otherwise."""
    return "s" if n_points > DENSE_MARKER_THRESHOLD else "o"


def scatter_xy(
    x: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray | None = None,
    colorize_by: np.ndarray | None = None,
    show_flagged: bool = False,
    mirror: bool = False,
    point_size: float | None = None,
    linewidths: float = DEFAULT_LINEWIDTHS,
    color: str = DEFAULT_COLOR,
    xlabel: str = "",
    ylabel: str = "",
    title: str | None = None,
    ax=None,
) -> Figure:
    """Scatter `y` against `x` (any two arrays of the same shape, typically
    from `derived_quantities.compute_quantity`).

    `weight`, if given, applies the AIPS flagged convention (<=0 means
    flagged): flagged points are excluded by default, or shown in a muted
    color via `show_flagged=True` -- never included as if they were good.

    `colorize_by`, if given, is a per-point category label (e.g. a Stokes
    label per cell): each distinct category gets its own color from a
    qualitative palette and a legend entry, and `color` is ignored.

    `mirror`, if set, also plots (-x, -y) for every point plotted (same
    style, no extra legend entry) -- for a quantity pair where a
    measurement and its conjugate are both physically sampled, e.g. a UV
    plane point and (-u, -v).

    `point_size` (the marker area in points^2; `None` picks one from the
    number of points drawn via `auto_point_size`) and `linewidths` (marker
    edge width) apply to every point drawn. The marker is a square for a
    dense plot and a circle otherwise (`auto_marker`); flagged points are
    always crosses. Draws into `ax` if given, else creates a new figure."""
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    if weight is not None:
        good = np.asarray(weight).ravel() > 0
    else:
        good = np.ones(x.shape, dtype=bool)

    n_drawn = int(good.sum()) + (int((~good).sum()) if show_flagged and weight is not None else 0)
    n_drawn *= 2 if mirror else 1
    if point_size is None:
        point_size = auto_point_size(n_drawn)
    point_marker = auto_marker(n_drawn)

    def _plot(px, py, **kwargs):
        kwargs.setdefault("marker", point_marker)
        ax.scatter(px, py, s=point_size, linewidths=linewidths, alpha=0.7, **kwargs)
        if mirror:
            kwargs.pop("label", None)
            ax.scatter(-px, -py, s=point_size, linewidths=linewidths, alpha=0.7, **kwargs)

    if colorize_by is not None:
        labels = np.asarray(colorize_by).ravel()
        categories = sorted(set(labels[good].tolist()))
        cmap = plt.get_cmap(CATEGORY_COLORMAP)
        for i, category in enumerate(categories):
            mask = good & (labels == category)
            _plot(x[mask], y[mask], color=cmap(i % cmap.N), label=str(category))
        if len(categories) > 1:
            ax.legend(fontsize=8, markerscale=2, loc="best")
    else:
        _plot(x[good], y[good], color=color)

    if show_flagged and weight is not None:
        _plot(x[~good], y[~good], color=FLAGGED_COLOR, marker="x", label="flagged")
        ax.legend(fontsize=8, markerscale=2, loc="best")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig if fig is not None else ax.figure
