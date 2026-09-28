"""Interactive windows for generic plots, filled as the data streams in.

Each plot's window shows its grid after every refresh interval while the
selection streams through, with the rows read so far. After a zoom or pan
settles (mouse released and the limits unchanged for `settle_s`), that plot
re-streams the selection over the new limits at the window's own pixel
resolution. A zoom during a stream interrupts it; plots whose pass did not
finish restart, which is harmless since adding the same samples to a grid
again changes nothing.
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt

from visplot.stream import GridReducer
from visplot.xy_figure import XYFigure
from visplot.xy_session import PassProgress, XYSource


class _View:
    """Tracks one plot's axes limits, to tell when a zoom/pan has settled."""

    def __init__(self, figure: XYFigure):
        self.figure = figure
        self.mouse_down = False
        self.last_limits = self.limits()
        self.changed_at = time.monotonic()
        canvas = figure.fig.canvas
        canvas.mpl_connect("button_press_event", lambda e: setattr(self, "mouse_down", True))
        canvas.mpl_connect("button_release_event", lambda e: setattr(self, "mouse_down", False))

    def limits(self):
        return (tuple(self.figure.ax.get_xlim()), tuple(self.figure.ax.get_ylim()))

    def settled_change(self, grid: GridReducer, settle_s: float):
        """The new limits if the view differs from what `grid` covers and has
        stopped changing; otherwise None."""
        now = time.monotonic()
        current = self.limits()
        if current != self.last_limits:
            self.last_limits, self.changed_at = current, now
            return None
        if self.mouse_down or now - self.changed_at < settle_s:
            return None
        if current == (grid.x_extent, grid.y_extent):
            return None
        return current


def _window_grid(figure: XYFigure, plot, x_extent, y_extent) -> GridReducer:
    bbox = figure.ax.get_window_extent()
    return GridReducer(plot, x_extent, y_extent, max(1, int(bbox.height)), max(1, int(bbox.width)))


def run_interactive(source: XYSource, figures: dict, extents: dict, first_pass_label: str = "drawing",
                    refresh_s: float = 0.5, settle_s: float = 0.3, keep_running=lambda: True) -> None:
    """Show `figures` (plot -> XYFigure) and stream into them until every
    window is closed (or `keep_running()` returns False). Each window's status
    line names the pass (`first_pass_label`, then "re-drawing the zoomed
    view" after a zoom) with rows, data read and time elapsed."""
    plt.ion()
    for figure in figures.values():
        figure.fig.show()
    plt.pause(0.001)

    grids = {p: _window_grid(f, p, *extents[p]) for p, f in figures.items()}
    views = {p: _View(f) for p, f in figures.items()}
    pending = set(figures)  # plots whose current grid has not had a full pass
    drawn_once = set()  # plots that have completed at least one pass

    def refresh(plots, status):
        for p in plots:
            figure = figures[p]
            figure.show(grids[p], display_dpi=figure.fig.dpi)
            figure.set_status(status)
            figure.fig.canvas.draw_idle()

    def restart_zoomed() -> bool:
        restarted = False
        for p, view in views.items():
            if not plt.fignum_exists(figures[p].fig.number):
                continue
            new = view.settled_change(grids[p], settle_s)
            if new is not None:
                grids[p] = _window_grid(figures[p], p, *new)
                pending.add(p)
                restarted = True
        return restarted

    while keep_running() and any(plt.fignum_exists(f.fig.number) for f in figures.values()):
        open_pending = [p for p in pending if plt.fignum_exists(figures[p].fig.number)]
        if not open_pending:
            plt.pause(0.1)
            restart_zoomed()
            continue

        last_refresh = time.monotonic()
        read_data = any(p.needs_data for p in open_pending)
        label = "re-drawing the zoomed view" if drawn_once.issuperset(open_pending) else first_pass_label
        progress = PassProgress(label, source.n_rows, source.row_bytes if read_data else 0)

        def on_chunk(rows_done):
            nonlocal last_refresh
            if time.monotonic() - last_refresh >= refresh_s:
                refresh(open_pending, progress.text(rows_done))
                plt.pause(0.001)
                last_refresh = time.monotonic()
                if restart_zoomed() or not keep_running():
                    return False
            return True

        stream_grids = [grids[p] for p in open_pending]
        completed = source.stream(stream_grids, read_data=read_data, on_chunk=on_chunk)
        if completed:
            pending.difference_update(open_pending)
            drawn_once.update(open_pending)
            for p in open_pending:
                refresh([p], f"{grids[p].n_samples:,} samples plotted from {source.n_rows:,} rows")
            plt.pause(0.001)
