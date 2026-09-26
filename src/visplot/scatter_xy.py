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

# A reasonable range for point_size (the scatter "s" parameter, in points^2):
# small enough that a dense, all-baseline/all-channel scatter (tens of
# thousands of points) doesn't paint over itself, large enough that a
# handful of points are actually visible. 4.0 suits the dense case; a
# sparser selection reads better with something nearer 20-40.
POINT_SIZE_RANGE = (1.0, 60.0)
DEFAULT_POINT_SIZE = 4.0
DEFAULT_LINEWIDTHS = 0.0
DEFAULT_COLOR = "tab:blue"
# A qualitative (categorical) palette -- matplotlib's own default cycle,
# good up to 10 distinct categories, comfortably covering the 2-4
# polarizations/correlations one plot would ever colorize by.
CATEGORY_COLORMAP = "tab10"
FLAGGED_COLOR = "lightcoral"


def scatter_xy(
    x: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray | None = None,
    colorize_by: np.ndarray | None = None,
    show_flagged: bool = False,
    point_size: float = DEFAULT_POINT_SIZE,
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

    `point_size` (the marker area in points^2; see `POINT_SIZE_RANGE` for a
    sensible range) and `linewidths` (marker edge width) apply to every
    point drawn. Draws into `ax` if given, else creates a new figure."""
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    if weight is not None:
        good = np.asarray(weight).ravel() > 0
    else:
        good = np.ones(x.shape, dtype=bool)

    if colorize_by is not None:
        labels = np.asarray(colorize_by).ravel()
        categories = sorted(set(labels[good].tolist()))
        cmap = plt.get_cmap(CATEGORY_COLORMAP)
        for i, category in enumerate(categories):
            mask = good & (labels == category)
            ax.scatter(
                x[mask], y[mask], s=point_size, linewidths=linewidths,
                color=cmap(i % cmap.N), alpha=0.7, label=str(category),
            )
        if len(categories) > 1:
            ax.legend(fontsize=8, markerscale=2, loc="best")
    else:
        ax.scatter(x[good], y[good], s=point_size, linewidths=linewidths, color=color, alpha=0.7)

    if show_flagged and weight is not None:
        ax.scatter(
            x[~good], y[~good], s=point_size, linewidths=linewidths,
            color=FLAGGED_COLOR, alpha=0.5, marker="x", label="flagged",
        )
        ax.legend(fontsize=8, markerscale=2, loc="best")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig if fig is not None else ax.figure
