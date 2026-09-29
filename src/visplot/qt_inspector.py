"""The visplot inspection window (Qt).

One window, a tab per plot. Each streamed plot fills in as the selection is
read, redraws its region after a zoom or pan settles, and has two tools
beside matplotlib's own zoom/pan/save:

- Locate: drag a box; the selection is read once and every sample inside is
  listed (baseline, UTC time, channel and frequency, Stokes, values,
  weight), the first `LOCATE_LIMIT` in the table and all counted by baseline;
  "Save all as CSV" writes every one of them (`locate_csv`).
- Export: the current view re-read at a chosen dpi and saved (PNG, PDF,
  SVG, EPS, TIFF, JPEG), since the window's own image is at screen
  resolution.

Given a `visplot.records.WindowProvenance`, each saved CSV and export gets
its provenance record (T31), with the request that reproduces it; a save
whose record cannot be written is not made. Given a `report(level, text)`
callback, the window says what it saved and what failed through it.

Streams run one at a time on a background thread (they share the disk);
the window polls the running job and draws from locked snapshots of the
grids. A locate or export interrupts drawing, which then resumes.
"""

from __future__ import annotations

import csv
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
from astropy.time import Time
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.widgets import RectangleSelector
from PySide6 import QtCore, QtGui, QtWidgets

from visplot.locate_csv import LocateCsvWriter, write_kept
from visplot.plot_spec import PlotSpec
from visplot.quantities import utc_jd
from visplot.request_args import DPI_LIMITS
from visplot.stream import GridReducer, LocateReducer
from visplot.xy_figure import XYFigure, grid_summary
from visplot.xy_session import PassProgress, XYSource, range_pass_axes, range_pass_reads_data, resolve_extents

if TYPE_CHECKING:
    from visplot.records import PlotRecord, WindowProvenance

LOCATE_LIMIT = 10_000
POLL_MS = 150
REFRESH_S = 0.5
SETTLE_S = 0.3
EXPORT_FILTERS = "PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;EPS (*.eps);;TIFF (*.tif);;JPEG (*.jpg)"


class _Job:
    """One stream on a background thread. `run(on_chunk)` returns whether it
    ran to the end; the window reads `rows_done`, `done`, `completed` and
    `error`, and sets `stop` to end it early."""

    def __init__(self, kind: str, label: str, run: Callable, on_done: Callable, bytes_per_row: int, n_rows: int):
        self.kind = kind
        self.progress = PassProgress(label, n_rows, bytes_per_row)
        self.on_done = on_done
        self.rows_done = 0
        self.done = False
        self.completed = False
        self.error: str | None = None
        self.stop = False
        self.started = False
        self._run = run
        self._thread = threading.Thread(target=self._body, name=f"visplot-{kind}", daemon=True)

    def start(self) -> None:
        self.started = True
        self._thread.start()

    def _body(self) -> None:
        def on_chunk(rows_done: int) -> bool:
            self.rows_done = rows_done
            return not self.stop

        try:
            self.completed = bool(self._run(on_chunk))
        except Exception:  # shown in the window; the next job still runs
            self.error = traceback.format_exc()
        self.done = True

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)


@dataclass
class _Panel:
    """One streamed plot's tab."""

    plot: PlotSpec
    figure: XYFigure
    canvas: FigureCanvasQTAgg
    grid: GridReducer | None = None
    drawn: bool = False  # the current grid has had a full pass
    mouse_down: bool = False
    last_limits: tuple | None = None
    changed_at: float = 0.0
    selector: RectangleSelector | None = None
    locate_action: QtGui.QAction | None = None
    toolbar: NavigationToolbar2QT | None = None
    aspect_box: QtWidgets.QCheckBox | None = None
    extent: tuple | None = None


class InspectorWindow(QtWidgets.QMainWindow):
    def __init__(self, source: XYSource, figures: list[tuple[str, object]], labels=("finding data ranges", "drawing"),
                 cache=None, cached=None, provenance: WindowProvenance | None = None,
                 report: Callable[[str, str], None] | None = None):
        super().__init__()
        self.source = source
        self.labels = labels
        self.cache = cache
        self.cached = cached or {}
        self.provenance = provenance
        self.report = report
        self.setWindowTitle("visplot")
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setTabBarAutoHide(True)  # one plot: no tab bar
        self.setCentralWidget(self.tabs)
        self.panels: dict[PlotSpec, _Panel] = {}
        self.jobs: list[_Job] = []  # queued, the first running
        self.extents: dict | None = None
        self.last_refresh = 0.0
        self.to_draw: set[PlotSpec] = set()  # plots whose current grid still needs a full pass
        self.locate: LocateReducer | None = None

        for name, item in figures:
            if isinstance(item, XYFigure):
                self._add_panel(name, item)
            else:
                self._add_static_tab(name, item)
        self._build_locate_dock()
        self.statusBar().showMessage("starting")

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._poll)
        self.timer.start(POLL_MS)
        if self.panels:
            self._queue_ranges()

    # ---- tabs ---------------------------------------------------------------

    def _add_static_tab(self, name: str, fig) -> None:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        canvas = FigureCanvasQTAgg(fig)
        layout.addWidget(NavigationToolbar2QT(canvas, widget))
        layout.addWidget(canvas)
        self.tabs.addTab(widget, name)

    def _add_panel(self, name: str, figure: XYFigure) -> None:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        canvas = FigureCanvasQTAgg(figure.fig)
        toolbar = NavigationToolbar2QT(canvas, widget)
        panel = _Panel(figure.plot, figure, canvas, toolbar=toolbar)
        panel.locate_action = toolbar.addAction("Locate")
        panel.locate_action.setCheckable(True)
        panel.locate_action.setToolTip("Drag a box to list the samples inside it")
        panel.locate_action.toggled.connect(lambda on, p=panel: self._toggle_locate(p, on))
        # A check box, so its state shows: equal scale on or off.
        panel.aspect_box = QtWidgets.QCheckBox("Equal aspect")
        panel.aspect_box.setChecked(figure.plot.equal_aspect)
        panel.aspect_box.setToolTip("One unit the same length on both axes (widens one axis; off returns to the view)")
        panel.aspect_box.toggled.connect(lambda on, p=panel: self._toggle_aspect(p, on))
        toolbar.addWidget(panel.aspect_box)
        export_action = toolbar.addAction("Export…")
        export_action.setToolTip("Re-read the current view at a chosen dpi and save it")
        export_action.triggered.connect(lambda _=False, p=panel: self._export(p))
        layout.addWidget(toolbar)
        layout.addWidget(canvas)
        canvas.mpl_connect("button_press_event", lambda e, p=panel: setattr(p, "mouse_down", True))
        canvas.mpl_connect("button_release_event", lambda e, p=panel: setattr(p, "mouse_down", False))
        self.panels[figure.plot] = panel
        self.tabs.addTab(widget, name)

    def _build_locate_dock(self) -> None:
        dock = QtWidgets.QDockWidget("Located samples", self)
        body = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(body)
        self.locate_summary = QtWidgets.QLabel("Use Locate on a plot's toolbar, then drag a box.")
        self.locate_summary.setWordWrap(True)
        self.locate_table = QtWidgets.QTableWidget(0, len(_LOCATE_COLUMNS))
        self.locate_table.setHorizontalHeaderLabels([c for c, _ in _LOCATE_COLUMNS])
        self.locate_table.setSortingEnabled(True)
        self.save_locate_button = QtWidgets.QPushButton("Save all as CSV…")
        self.save_locate_button.setToolTip("Every located sample (the table shows the first 10,000)")
        self.save_locate_button.setEnabled(False)  # until something is located
        self.save_locate_button.clicked.connect(self._save_locate_csv)
        layout.addWidget(self.locate_summary)
        layout.addWidget(self.locate_table)
        layout.addWidget(self.save_locate_button)
        dock.setWidget(body)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, dock)
        dock.hide()  # shown when Locate is first used, so the plot has the room until then
        self.locate_dock = dock

    # ---- jobs -----------------------------------------------------------------

    def _queue(self, job: _Job, front: bool = False) -> None:
        """Add a job; `front` puts it next, stopping a running draw (whose
        plots stay waiting and are drawn again afterwards)."""
        if front and self.jobs:
            if self.jobs[0].kind == "draw":
                self.jobs[0].stop = True
            self.jobs.insert(1, job)
        elif front:
            self.jobs.insert(0, job)
        else:
            self.jobs.append(job)

    def _queue_ranges(self) -> None:
        plots = list(self.panels)
        pending = [k for k in range_pass_axes(plots) if k not in self.cached]
        read_data = range_pass_reads_data(plots, self.cached)

        def run(on_chunk):
            self.extents = resolve_extents(self.source, plots, on_chunk=on_chunk, cache=self.cache)
            return self.extents is not None

        def done(job):
            if job.completed:
                for plot, panel in self.panels.items():
                    panel.extent = panel.figure.set_view(*self.extents[plot])
                self.request_draw(list(self.panels))

        label = self.labels[0] if pending else "reading ranges from the cache"
        self._queue(_Job("ranges", label, run, done, self.source.row_bytes if read_data else 0, self.source.n_rows))

    def _window_grid(self, panel: _Panel, extent) -> GridReducer:
        panel.canvas.draw()
        bbox = panel.figure.ax.get_window_extent()
        return GridReducer(panel.plot, extent[0], extent[1], max(1, int(bbox.height)), max(1, int(bbox.width)))

    def request_draw(self, plots: list[PlotSpec]) -> None:
        """Mark plots for drawing over their current extent (a new grid where
        the extent changed). A running draw is stopped so the next one covers
        every plot waiting; unfinished plots stay waiting."""
        for plot in plots:
            panel = self.panels[plot]
            if panel.grid is None or (panel.grid.x_extent, panel.grid.y_extent) != tuple(panel.extent):
                panel.grid = self._window_grid(panel, panel.extent)
            panel.drawn = False
            self.to_draw.add(plot)
        running = self.jobs[0] if self.jobs else None
        if running is not None and running.kind == "draw":
            running.stop = True

    def _start_draw(self) -> None:
        plots = list(self.to_draw)
        grids = {p: self.panels[p].grid for p in plots}
        read_data = any(p.needs_data for p in plots)
        first = not any(self.panels[p].figure.image is not None for p in plots)
        label = self.labels[1] if first else "re-drawing the view"

        def run(on_chunk):
            return self.source.stream(list(grids.values()), read_data=read_data, on_chunk=on_chunk)

        def done(job):
            if not job.completed:
                return
            for plot, grid in grids.items():
                panel = self.panels[plot]
                if panel.grid is grid:  # not replaced by a zoom meanwhile
                    panel.drawn = True
                    self.to_draw.discard(plot)
                    self._render(panel, final=True)

        self._queue(_Job("draw", label, run, done, self.source.row_bytes if read_data else 0, self.source.n_rows))

    def _poll(self) -> None:
        now = time.monotonic()
        if not self.jobs and self.to_draw and self.extents is not None:
            self._start_draw()
        if self.jobs:
            job = self.jobs[0]
            if not job.started:
                job.start()
            if job.done:
                self.jobs.pop(0)
                if job.error:
                    if self.report is None:
                        self.statusBar().showMessage(f"{job.progress.label} failed; see the terminal")
                        print(job.error)
                    else:
                        self.statusBar().showMessage(f"{job.progress.label} failed; see the messages")
                        self.report("warning", f"{job.progress.label} failed:\n{job.error}")
                else:
                    self.statusBar().showMessage(job.progress.text(job.rows_done))
                job.on_done(job)
            else:
                self.statusBar().showMessage(job.progress.text(job.rows_done))
                if job.kind == "draw" and now - self.last_refresh >= REFRESH_S:
                    for plot in self.to_draw:
                        self._render(self.panels[plot])
                    self.last_refresh = now
        elif self.statusBar().currentMessage().endswith("elapsed"):
            self.statusBar().showMessage("ready")
        self._check_views(now)

    def _render(self, panel: _Panel, final: bool = False) -> None:
        snapshot = panel.grid.snapshot()
        panel.figure.show(snapshot, display_dpi=panel.figure.fig.dpi)
        if final:
            panel.figure.set_status(grid_summary(snapshot, self.source.n_rows))
        else:
            job = self.jobs[0] if self.jobs else None
            panel.figure.set_status(job.progress.text(job.rows_done) if job else "")
        panel.canvas.draw_idle()

    def _check_views(self, now: float) -> None:
        """After a zoom or pan settles, re-draw that plot over the new limits.
        Zoom or pan turned on while Locate is on turns Locate off."""
        for plot, panel in self.panels.items():
            if panel.selector is not None and panel.toolbar.mode.name != "NONE":
                panel.locate_action.setChecked(False)
            if panel.grid is None or panel.figure.image is None:
                continue
            limits = (tuple(panel.figure.ax.get_xlim()), tuple(panel.figure.ax.get_ylim()))
            if limits != panel.last_limits:
                panel.last_limits, panel.changed_at = limits, now
                panel.figure.hide_if_view_moved(panel.grid)
                continue
            if panel.mouse_down or now - panel.changed_at < SETTLE_S:
                continue
            if limits != (panel.grid.x_extent, panel.grid.y_extent) and limits != tuple(panel.extent):
                # the zoomed view, kept as the view the aspect toggle returns to; with equal
                # aspect, widened again so both axes keep one scale and one span
                panel.extent = panel.figure.set_view(*limits)
                self.request_draw([plot])

    def _toggle_aspect(self, panel: _Panel, equal: bool) -> None:
        """Equal scale on: widen the requested view so a unit is the same
        length on both axes; off: back to the requested view (the data's
        range, or the last zoom)."""
        panel.figure.equal_override = equal
        if panel.extent is None:
            return  # applied when the ranges are known
        panel.extent = panel.figure.set_view(*panel.figure.view_request)
        panel.last_limits = (tuple(panel.extent[0]), tuple(panel.extent[1]))
        self.request_draw([panel.plot])

    # ---- locate -------------------------------------------------------------

    def _toggle_locate(self, panel: _Panel, on: bool) -> None:
        """Locate is a mode like zoom and pan: turning it on turns them off.
        (Zoom and pan lock the canvas, and the box selector ignores every
        event while they hold the lock.)"""
        if on:
            self.locate_dock.show()
            if panel.toolbar.mode.name == "ZOOM":
                panel.toolbar.zoom()
            elif panel.toolbar.mode.name == "PAN":
                panel.toolbar.pan()
            panel.selector = RectangleSelector(
                panel.figure.ax, lambda press, release, p=panel: self._locate(p, press, release),
                useblit=True, button=[1], interactive=False, minspanx=2, minspany=2, spancoords="pixels",
            )
            self.statusBar().showMessage("Locate: drag a box on the plot to list the samples inside it")
        elif panel.selector is not None:
            panel.selector.set_active(False)
            panel.selector = None

    def _locate(self, panel: _Panel, press, release) -> None:
        box_x = (press.xdata, release.xdata)
        box_y = (press.ydata, release.ydata)
        if None in box_x or None in box_y:
            return
        locate = LocateReducer(panel.plot, box_x, box_y, limit=LOCATE_LIMIT)
        read_data = panel.plot.needs_data
        self.locate_summary.setText(
            f"reading the selection to find the samples in x {min(box_x):.4g} to {max(box_x):.4g}, "
            f"y {min(box_y):.4g} to {max(box_y):.4g}…")
        self.locate_dock.show()
        self.locate_dock.raise_()

        def run(on_chunk):
            return self.source.stream([locate], read_data=read_data, on_chunk=on_chunk)

        def done(job):
            if job.completed:
                self.locate = locate
                self._show_locate(locate)
                self.save_locate_button.setEnabled(locate.n_found > 0)
                self.statusBar().showMessage(
                    f"located {locate.n_found:,} samples; listed in the table below")
            else:
                self.locate_summary.setText("locate stopped before the end of the selection")

        self._queue(_Job("locate", "locating samples", run, done,
                         self.source.row_bytes if read_data else 0, self.source.n_rows), front=True)

    def _show_locate(self, locate: LocateReducer) -> None:
        self.locate_dock.show()
        names = self.source.ctx.antenna_names or {}
        shown = len(locate.records)
        top = sorted(locate.by_baseline.items(), key=lambda kv: -kv[1])[:10]
        baselines = ", ".join(f"{_baseline(a, b, names)}: {n:,}" for (a, b), n in top)
        more = f" (listing the first {shown:,})" if shown < locate.n_found else ""
        self.locate_summary.setText(
            f"{locate.n_found:,} samples in the box{more}, on {len(locate.by_baseline):,} "
            f"baseline{'s' if len(locate.by_baseline) != 1 else ''}. "
            f"Most: {baselines}" if locate.n_found else "no samples in the box")
        rows = locate_rows(locate.records, self.source.ctx)
        table = self.locate_table
        headers = [c for c, _ in _LOCATE_COLUMNS]
        for axis in ("x", "y"):
            headers[headers.index(axis)] = _axis_header(locate.plot, axis, self.source.ctx)
        table.setHorizontalHeaderLabels(headers)
        table.setSortingEnabled(False)
        table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, value in enumerate(row):
                item = QtWidgets.QTableWidgetItem()
                item.setData(QtCore.Qt.DisplayRole, value)
                table.setItem(i, j, item)
        table.setSortingEnabled(True)
        self.locate_dock.show()

    def _save_locate_csv(self) -> None:
        if self.locate is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save located samples", "located_samples.csv",
                                                        "CSV (*.csv)")
        if path:
            self.save_locate_csv(path)

    def save_locate_csv(self, path: str) -> None:
        """Every located sample to `path`: from memory when all were kept,
        otherwise by reading the selection again and writing each sample as
        it is found."""
        located = self.locate
        started, record = self._start_record("locate", located.plot, path, box=(located.x_box, located.y_box))
        if not started:
            return
        header = dict(fits_path=self.source.fits_path, command=record and record.command,
                      record=record and record.describe())
        if located.kept_all:
            try:
                written = write_kept(path, located, self.source.ctx, **header)
            except OSError as err:
                self._finish_record(record, "locate", None, f"could not write {path}: {err}")
                return
            self._finish_record(record, "locate", written, None)
            self.statusBar().showMessage(f"saved {located.n_found:,} located samples to {path}")
            return
        again = LocateReducer(located.plot, located.x_box, located.y_box, limit=0)
        try:
            writer = LocateCsvWriter(path, again, self.source.ctx, **header)
        except OSError as err:
            self._finish_record(record, "locate", None, f"could not write {path}: {err}")
            return
        again.sink = writer
        read_data = located.plot.needs_data

        def run(on_chunk):
            return self.source.stream([again], read_data=read_data, on_chunk=on_chunk)

        def done(job):
            written = writer.close(again, completed=job.completed and job.error is None)
            self._finish_record(record, "locate", written,
                                None if written else "stopped before the end of the selection; no file written")
            self.statusBar().showMessage(
                f"saved {again.n_found:,} located samples to {written}" if written
                else "saving the located samples stopped before the end; no file written")

        self._queue(_Job("locate-csv", f"writing all {located.n_found:,} located samples", run, done,
                         self.source.row_bytes if read_data else 0, self.source.n_rows), front=True)

    # ---- provenance of saved files -----------------------------------------------

    def _report(self, level: str, text: str) -> None:
        if self.report is not None:
            self.report(level, text)

    def _start_record(self, kind: str, plot: PlotSpec, path, **details) -> tuple[bool, PlotRecord | None]:
        """(whether to go ahead with the save, its record). Without
        provenance there is no record; a record that cannot be written stops
        the save."""
        if self.provenance is None:
            return True, None
        try:
            return True, self.provenance.start(kind, plot.name, str(Path(path).resolve()), **details)
        except Exception as err:
            text = f"{kind} not saved: its provenance record could not be written ({type(err).__name__}: {err})"
            self._report("warning", text)
            self.statusBar().showMessage(text)
            return False, None

    def _finish_record(self, record: PlotRecord | None, kind: str, written, error: str | None) -> None:
        if error is not None:
            self.statusBar().showMessage(f"{kind} failed: {error}")
        if record is None:
            if error is not None:
                self._report("warning", f"{kind} failed: {error}")
            return
        if written is not None:
            record.add_output(written)
        record.finish(error)
        if error is None:
            self._report("info", f"{kind} {record.run_id}: saved {written}; command: {record.command}")
        else:
            self._report("warning", f"{kind} {record.run_id} failed: {error}")

    # ---- export ---------------------------------------------------------------

    def _export(self, panel: _Panel) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export plot", f"{panel.plot.name}.png", EXPORT_FILTERS)
        if not path:
            return
        dpi, ok = QtWidgets.QInputDialog.getInt(self, "Export resolution", "dots per inch:", 600, *DPI_LIMITS)
        if ok:
            self.export(panel.plot, path, dpi)

    def export(self, plot: PlotSpec, path: str, dpi: int) -> None:
        """Re-read the plot's current view at `dpi` and save it to `path`."""
        panel = self.panels[plot]
        limits = (tuple(panel.figure.ax.get_xlim()), tuple(panel.figure.ax.get_ylim()))
        # the Equal aspect box's override takes effect on linear axes only
        equal = panel.figure.equal_override if panel.figure.linear else None
        started, record = self._start_record("export", plot, path, view=limits, dpi=dpi,
                                             figure_size_in=tuple(panel.figure.fig.get_size_inches()), equal=equal)
        if not started:
            return
        height, width = panel.figure.grid_shape(dpi)
        grid = GridReducer(plot, limits[0], limits[1], height, width)
        read_data = plot.needs_data

        def run(on_chunk):
            return self.source.stream([grid], read_data=read_data, on_chunk=on_chunk)

        def done(job):
            if not job.completed:
                self._finish_record(record, "export", None, "stopped before the end of the selection; nothing saved")
                return
            panel.figure.show(grid, display_dpi=dpi)
            status = panel.figure.status.get_text()
            panel.figure.set_status(grid_summary(grid, self.source.n_rows))
            try:
                panel.figure.fig.savefig(path, dpi=dpi)
                error = None
            except (OSError, ValueError) as err:
                error = f"could not save {path}: {err}"
            panel.figure.set_status(status)
            if panel.grid is not None:  # back to the window's own image
                panel.figure.show(panel.grid.snapshot(), display_dpi=panel.figure.fig.dpi)
            panel.canvas.draw_idle()
            self._finish_record(record, "export", None if error else path, error)
            if error is None:
                self.statusBar().showMessage(f"exported {path} at {dpi} dpi")

        self._queue(_Job("export", f"exporting at {dpi} dpi", run, done,
                         self.source.row_bytes if read_data else 0, self.source.n_rows), front=True)

    # ---- closing ----------------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt's name)
        for job in self.jobs:
            job.stop = True
        for job in self.jobs[:1]:
            job.join(timeout=5)
        self.timer.stop()
        super().closeEvent(event)

    def idle(self) -> bool:
        return not self.jobs


_LOCATE_COLUMNS = [
    ("baseline", str), ("time (UTC)", str), ("channel", str), ("freq (MHz)", float), ("Stokes", str),
    ("x", float), ("y", float), ("weight", float), ("mirrored", str), ("row", int), ("source", str),
]


def _axis_header(plot: PlotSpec, axis: str, ctx) -> str:
    """A located-samples column header with the axis's unit, e.g. "x (kλ)";
    clock time is in hours, e.g. "x (h, UTC)"."""
    unit = plot.unit(axis, ctx)
    if unit.clock:
        return f"{axis} (h, {unit.label})"
    return f"{axis} ({unit.label})" if unit.label else axis


def _baseline(a: int, b: int, names: dict) -> str:
    return f"{names.get(a, a)}-{names.get(b, b)}"


def locate_rows(records: list[dict], ctx) -> list[list]:
    """Located samples as table rows, in `_LOCATE_COLUMNS` order."""
    if not records:
        return []
    names = ctx.antenna_names or {}
    sources = ctx.source_names or {}
    times = Time(utc_jd(ctx, np.array([r["jd"] for r in records])), format="jd", scale="utc").isot
    rows = []
    for r, t in zip(records, times):
        rows.append([
            _baseline(r["ant1"], r["ant2"], names), t,
            "all" if r["channel"] is None else str(r["channel"]),
            float("nan") if r["channel"] is None else r["freq_hz"] / 1e6,
            "all" if r["stokes"] is None else r["stokes"],
            r["x"], r["y"], r["weight"], "yes" if r["mirrored"] else "no", r["row"],
            sources.get(r["source_id"], str(r["source_id"])),
        ])
    return rows


def run_inspector(source: XYSource, figures: list[tuple[str, object]], labels, cache=None, cached=None,
                  provenance: WindowProvenance | None = None, report: Callable[[str, str], None] | None = None) -> int:
    """Open the inspection window and run until it is closed."""
    import signal

    signal.signal(signal.SIGINT, signal.SIG_DFL)  # Ctrl-C in the terminal closes the window
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = InspectorWindow(source, figures, labels=labels, cache=cache, cached=cached, provenance=provenance,
                             report=report)
    window.resize(1100, 900)
    window.show()
    return app.exec()
