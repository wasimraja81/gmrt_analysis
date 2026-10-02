"""The visplot inspection window (Qt).

One window, a tab per plot. Each streamed plot fills in as the selection is
read, redraws its region after a zoom or pan settles, and has two tools
beside matplotlib's own zoom/pan/save (below). With --one-plot-per (T26) a
plot's tab shows its pages one at a time (◀ ▶, the pages by name, Page Up
and Page Down), drawing the page shown from the rows its plots hold; a zoom
on a page's shared axis zooms every plot of it and carries to the next
page, and the tools act on the plot boxed (Locate) or the whole page
(Export).

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
from visplot.quantities import QUANTITIES, utc_jd
from visplot.request_args import DPI_LIMITS
from visplot.run import PageOfPlots, PlotPages, rows_of_iterations, stack_extents
from visplot.stream import GridReducer, LocateReducer
from visplot.xy_figure import PlotAxes, XYFigure, grids_summary, same_view
from visplot.xy_session import (PassProgress, XYSource, range_pass_axes, resolve_axis_ranges, resolve_extents,
                                source_in_views)

if TYPE_CHECKING:
    from visplot.records import PlotRecord, WindowProvenance

LOCATE_LIMIT = 10_000
LOCATE_PANEL_SIZE = QtCore.QSize(640, 400)  # the Located samples panel, when it first opens
POLL_MS = 150
REFRESH_S = 0.5
SETTLE_S = 0.3
RESIZE_TOLERANCE_PX = 2  # a grid this close to the axes' pixel size is kept (rounding)
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
    """One streamed plot in the window: a tab's plot, or one plot of the
    page a tab shows (T26), which shares the tab's figure, canvas and
    toolbar with the page's other plots."""

    plot: PlotSpec
    figure: XYFigure
    canvas: FigureCanvasQTAgg
    cell: PlotAxes | None = None  # the plot's axes in the figure
    tab: _Tab | None = None  # the tab of pages showing it, if any
    grid: GridReducer | None = None
    drawn: bool = False  # the current grid has had a full pass
    mouse_down: bool = False
    last_limits: tuple | None = None
    changed_at: float = 0.0
    last_shape: tuple | None = None  # the axes' size in pixels when last looked at, and since when
    resized_at: float = 0.0
    selector: RectangleSelector | None = None
    locate_action: QtGui.QAction | None = None
    toolbar: NavigationToolbar2QT | None = None
    aspect_box: QtWidgets.QCheckBox | None = None
    extent: tuple | None = None
    rows_read: int | None = None  # by the current grid's draw: the rows that can reach its view


@dataclass
class _Tab:
    """A tab of pages (T26): one streamed plot's pages (`PlotPages`), one
    shown at a time, its plots the window's panels while shown."""

    plot: PlotSpec
    pages_of: PlotPages
    layout: QtWidgets.QVBoxLayout
    previous: QtWidgets.QToolButton
    next: QtWidgets.QToolButton
    chooser: QtWidgets.QComboBox
    counter: QtWidgets.QLabel
    body: QtWidgets.QWidget  # the page's toolbar and canvas (a placeholder until the ranges are known)
    pages: list[PageOfPlots] = field(default_factory=list)  # once the ranges from every plot are known
    index: int = 0
    figure: XYFigure | None = None
    panels: list[_Panel] = field(default_factory=list)
    view: dict[str, tuple] = field(default_factory=dict)  # a shared axis's zoom, kept from page to page
    rows: int = 0  # the selection's rows the page's plots hold
    rows_read: int | None = None  # by the page's draw: the rows that can reach its views

    @property
    def page(self) -> PageOfPlots | None:
        return self.pages[self.index] if self.pages else None


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

        self.paged: list[_Tab] = []  # tabs of pages (T26)
        self.figure_names: dict[int, str] = {}  # a tab's figure (by id): its --plots entry, for saves
        self.laying_out: set[int] = set()  # tabs whose page shown waits for Qt to lay its canvas out
        for name, item in figures:
            if isinstance(item, XYFigure):
                self._add_panel(name, item)
            elif isinstance(item, PlotPages):
                self._add_paged_tab(name, item)
            else:
                self._add_static_tab(name, item)
        self._build_locate_dock()
        self.statusBar().showMessage("starting")

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._poll)
        self.timer.start(POLL_MS)
        if self.panels or self.paged:
            self._queue_ranges()

    # ---- tabs ---------------------------------------------------------------

    def _add_static_tab(self, name: str, fig) -> None:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        canvas = FigureCanvasQTAgg(fig)
        layout.addWidget(NavigationToolbar2QT(canvas, widget))
        layout.addWidget(canvas)
        self.tabs.addTab(widget, name)

    def _plot_toolbar(self, parent, canvas, panels: Callable[[], list[_Panel]], equal: bool):
        """The toolbar of a figure of streamed plots: matplotlib's, with
        Locate, Equal aspect and Export acting on `panels()` (one plot, or
        the plots of the page shown). Returns (toolbar, locate action,
        aspect box)."""
        toolbar = NavigationToolbar2QT(canvas, parent)
        locate_action = toolbar.addAction("Locate")
        locate_action.setCheckable(True)
        locate_action.setToolTip("Drag a box on a plot to list the samples inside it")
        locate_action.toggled.connect(lambda on: self._toggle_locate(panels(), on))
        # A check box, so its state shows: equal scale on or off.
        aspect_box = QtWidgets.QCheckBox("Equal aspect")
        aspect_box.setChecked(equal)
        aspect_box.setToolTip("One unit the same length on both axes (widens one axis; off returns to the view)")
        aspect_box.toggled.connect(lambda on: self._toggle_aspect(panels(), on))
        toolbar.addWidget(aspect_box)
        export_action = toolbar.addAction("Export…")
        export_action.setToolTip("Re-read the current view at a chosen dpi and save it")
        export_action.triggered.connect(lambda _=False: self._export(panels()))
        return toolbar, locate_action, aspect_box

    def _track_mouse(self, canvas, panels: Callable[[], list[_Panel]], figure: XYFigure) -> None:
        """A button held on `canvas` holds its plots' redraws (a drag in
        progress); a resize lays the figure out again (the panel keeps its
        size)."""
        def held(down: bool) -> None:
            for panel in panels():
                panel.mouse_down = down

        canvas.mpl_connect("button_press_event", lambda e: held(True))
        canvas.mpl_connect("button_release_event", lambda e: held(False))
        canvas.mpl_connect("resize_event", lambda e, f=figure: f.relayout())

    def _add_panel(self, name: str, figure: XYFigure) -> None:
        """A tab of one figure: one plot, or a stack of two sharing x (T26),
        each plot a panel of the window, the toolbar acting on them all."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        canvas = FigureCanvasQTAgg(figure.fig)
        panels = [_Panel(cell.plot, figure, canvas, cell=cell) for cell in figure.cells]
        toolbar, locate_action, aspect_box = self._plot_toolbar(
            widget, canvas, lambda ps=panels: list(ps), figure.cells[0].plot.equal_aspect)
        for panel in panels:
            panel.toolbar, panel.locate_action, panel.aspect_box = toolbar, locate_action, aspect_box
            self.panels[panel.plot] = panel
        layout.addWidget(toolbar)
        layout.addWidget(canvas)
        self._track_mouse(canvas, lambda ps=panels: list(ps), figure)
        self.figure_names[id(figure)] = name
        self.tabs.addTab(widget, name)

    def _add_paged_tab(self, name: str, pages_of: PlotPages) -> None:
        """A tab of a streamed plot's pages (T26): ◀ ▶, the pages by name and
        Page Up / Page Down above the page shown; its first page shown once
        the ranges from every plot are known."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        bar = QtWidgets.QHBoxLayout()
        previous, following = QtWidgets.QToolButton(text="◀"), QtWidgets.QToolButton(text="▶")
        previous.setToolTip("The previous page (Page Up)")
        following.setToolTip("The next page (Page Down)")
        chooser, counter = QtWidgets.QComboBox(), QtWidgets.QLabel("")
        chooser.setToolTip("Go to a page")
        chooser.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)  # its pages' names, once known
        for control in (previous, following, chooser):
            control.setEnabled(False)  # until the pages are known
            bar.addWidget(control)
        bar.addWidget(counter)
        bar.addStretch(1)
        layout.addLayout(bar)
        body = QtWidgets.QLabel("finding the ranges of the plots…")
        body.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(body, 1)
        tab = _Tab(pages_of.plot, pages_of, layout, previous, following, chooser, counter, body)
        previous.clicked.connect(lambda _=False, t=tab: self.show_page(t, t.index - 1))
        following.clicked.connect(lambda _=False, t=tab: self.show_page(t, t.index + 1))
        chooser.activated.connect(lambda i, t=tab: self.show_page(t, i))
        for key, step in ((QtCore.Qt.Key_PageUp, -1), (QtCore.Qt.Key_PageDown, 1)):
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), widget)
            shortcut.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
            shortcut.activated.connect(lambda t=tab, d=step: self.show_page(t, t.index + d))
        self.paged.append(tab)
        self.tabs.addTab(widget, name)

    def show_page(self, tab: _Tab, index: int) -> None:
        """Show page `index` of `tab`: its figure in place of the last
        page's, whose plots leave the window. A shared axis keeps the view a
        zoom gave it (`_Tab.view`); a plot's own range is found for the page
        by a pass over its rows."""
        if not tab.pages:
            return
        index = max(0, min(index, len(tab.pages) - 1))
        if tab.figure is not None and index == tab.index:
            return
        locating = bool(tab.panels and tab.panels[0].locate_action.isChecked())
        for panel in tab.panels:
            self.panels.pop(panel.plot, None)
            self.to_draw.discard(panel.plot)
            if panel.selector is not None:
                panel.selector.set_active(False)
        tab.index = index
        page = tab.page
        figure = tab.pages_of.run.page_figure(page)
        body = QtWidgets.QWidget()
        box = QtWidgets.QVBoxLayout(body)
        box.setContentsMargins(0, 0, 0, 0)
        canvas = FigureCanvasQTAgg(figure.fig)
        toolbar, locate_action, aspect_box = self._plot_toolbar(
            body, canvas, lambda t=tab: list(t.panels), figure.cells[0].plot.equal_aspect)
        box.addWidget(toolbar)
        box.addWidget(canvas, 1)
        self._track_mouse(canvas, lambda t=tab: list(t.panels), figure)
        tab.layout.replaceWidget(tab.body, body)
        tab.body.deleteLater()
        tab.body, tab.figure = body, figure
        tab.panels = [_Panel(cell.plot, figure, canvas, cell=cell, tab=tab, toolbar=toolbar,
                             locate_action=locate_action, aspect_box=aspect_box) for cell in figure.cells]
        for panel in tab.panels:
            self.panels[panel.plot] = panel
        iterations = [plot.iteration for plot in page.plots]
        tab.rows = len(rows_of_iterations(self.source, iterations))
        tab.counter.setText(f"page {index + 1:,} of {len(tab.pages):,}")
        tab.chooser.setCurrentIndex(index)
        tab.previous.setEnabled(index > 0)
        tab.next.setEnabled(index < len(tab.pages) - 1)
        if locating:
            locate_action.setChecked(True)
        if not range_pass_axes(page.plots):  # every range known: from every plot, or given
            # once Qt has laid the new canvas out, so the plots' grids take its pixels
            extents = {plot: (plot.x_range, plot.y_range) for plot in page.plots}
            self.laying_out.add(id(tab))

            def views(t=tab, shown=page):
                self.laying_out.discard(id(t))
                if t.page is shown:
                    self._set_page_views(t, extents)
            QtCore.QTimer.singleShot(0, views)
            return
        part = self.source.subset(rows_of_iterations(self.source, iterations))
        read_data = any(QUANTITIES[p.x if axis == "x" else p.y].needs_data for p, axis in range_pass_axes(page.plots))
        found = {}

        def run(on_chunk):
            found["extents"] = resolve_extents(part, page.plots, on_chunk=on_chunk, cache=self.cache)
            return found["extents"] is not None

        def done(job):
            if job.completed and tab.page is page:
                self._set_page_views(tab, found["extents"])

        self._queue(_Job("ranges", f"finding the ranges of page {index + 1}", run, done,
                         self.source.row_bytes if read_data else 0, part.n_rows), front=True)

    def _set_page_views(self, tab: _Tab, extents: dict) -> None:
        """The page's plots shown over `extents` ({plot: (x, y)}: their
        ranges), a shared axis over the view a zoom on an earlier page gave
        it, and drawn. With such a zoom, the toolbar's Home and Back return
        to the page's own ranges (the user, 2026-10-02: Home and Back did not
        undo a zoom carried from another page)."""
        shared = tab.figure.page.shared if tab.figure.page is not None else (False, False)
        zoomed = {plot: (tab.view.get("x", x) if shared[0] else x, tab.view.get("y", y) if shared[1] else y)
                  for plot, (x, y) in extents.items()}
        limits = self._page_views(tab, extents)
        if zoomed != extents:
            toolbar = tab.panels[0].toolbar
            toolbar.push_current()  # Home: the page's own ranges
            limits = self._page_views(tab, zoomed)
            toolbar.push_current()  # the zoom carried over, one step on from Home (Back returns)
        for panel in tab.panels:
            panel.extent = panel.last_limits = limits[panel.plot]
        self.request_draw([panel.plot for panel in tab.panels])

    @staticmethod
    def _page_views(tab: _Tab, extents: dict) -> dict:
        """Set the page's plots' views to `extents`; the limits they take."""
        if tab.figure.page is None:
            (plot,) = extents
            return {plot: tab.figure.set_view(*extents[plot])}
        return tab.figure.set_views(extents)

    def _set_view(self, panel: _Panel, x_extent, y_extent) -> tuple:
        """`XYFigure.set_view` for the panel's plot (in a page or a stack, its axes)."""
        if panel.figure.page is None:
            return panel.figure.set_view(x_extent, y_extent)
        return panel.cell.set_view(x_extent, y_extent)

    def _figure_panels(self, figure: XYFigure) -> list[_Panel]:
        return [panel for panel in self.panels.values() if panel.figure is figure]

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
        # A window of its own, so the plot never gives up room to it; its title
        # bar docks it (drag or double-click), and then it stays docked.
        dock.setFloating(True)
        dock.hide()  # shown when a box is first drawn
        self.locate_dock = dock
        self.locate_placed = False

    def _show_locate_panel(self) -> None:
        """Show the Located samples panel: the first time beside the window,
        afterwards wherever the user moved or docked it."""
        dock = self.locate_dock
        if not self.locate_placed and dock.isFloating():
            top = self.window()
            dock.setGeometry(beside(top.frameGeometry(), top.screen().availableGeometry(), LOCATE_PANEL_SIZE))
        self.locate_placed = True
        dock.show()
        dock.raise_()

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
        """The first pass: the ranges the plots take from the data (a plot's
        own, or for a tab of pages, those from every plot), then each tab's
        pages planned and its first page shown."""
        plots = list(self.panels)
        pairs = range_pass_axes(plots) + [pair for tab in self.paged for pair in tab.pages_of.from_all]
        pending = [pair for pair in pairs if pair not in self.cached]
        read_data = any(QUANTITIES[p.x if axis == "x" else p.y].needs_data for p, axis in pending)
        found = {}

        def run(on_chunk):
            found["ranged"] = resolve_axis_ranges(self.source, pairs, on_chunk=on_chunk, cache=self.cache)
            if found["ranged"] is None:
                return False
            ranged = found["ranged"]
            self.extents = {p: (p.x_range if p.x_range is not None else ranged[(p, "x")],
                                p.y_range if p.y_range is not None else ranged[(p, "y")]) for p in plots}
            return True

        def done(job):
            if not job.completed:
                return
            figures = {id(self.panels[plot].figure): self.panels[plot].figure for plot in plots}
            extents = stack_extents(self.extents, [figure.plots for figure in figures.values()])
            for figure in figures.values():  # a stack's plots on one x extent
                for plot, limits in figure.set_views({p: extents[p] for p in figure.plots}).items():
                    self.panels[plot].extent = limits
            self.request_draw(plots)
            for tab in self.paged:
                tab.pages = tab.pages_of.pages(found["ranged"])
                tab.chooser.addItems([page.panel_text for page in tab.pages])
                tab.chooser.setEnabled(len(tab.pages) > 1)
                self.show_page(tab, 0)

        label = self.labels[0] if pending else "reading ranges from the cache"
        self._queue(_Job("ranges", label, run, done, self.source.row_bytes if read_data else 0, self.source.n_rows))

    @staticmethod
    def _pixel_shape(panel: _Panel) -> tuple[int, int]:
        """The axes' (height, width) in the window's pixels: a window grid's shape."""
        bbox = panel.cell.ax.get_window_extent()
        return max(1, int(bbox.height)), max(1, int(bbox.width))

    @classmethod
    def _resized(cls, panel: _Panel) -> bool:
        """Whether the axes' pixel size differs from the grid's (by more than
        rounding): the window was resized since the grid was made."""
        height, width = cls._pixel_shape(panel)
        return max(abs(height - panel.grid.height), abs(width - panel.grid.width)) > RESIZE_TOLERANCE_PX

    def _window_grid(self, panel: _Panel, extent) -> GridReducer:
        """A grid of the axes' pixels over `extent` (the canvas drawn already,
        so the axes' size is current)."""
        return GridReducer(panel.plot, extent[0], extent[1], *self._pixel_shape(panel))

    def request_draw(self, plots: list[PlotSpec]) -> None:
        """Mark plots for drawing over their current extent (a new grid where
        the extent or the window's size changed). A running draw is stopped so
        the next one covers every plot waiting; unfinished plots stay
        waiting."""
        drawn_canvases = set()
        for plot in plots:
            panel = self.panels[plot]
            if id(panel.canvas) not in drawn_canvases:  # once per figure, however many of its plots (a page's)
                panel.canvas.draw()
                drawn_canvases.add(id(panel.canvas))
            if (panel.grid is None or not same_view((panel.grid.x_extent, panel.grid.y_extent), panel.extent)
                    or self._resized(panel)):
                panel.cell.begin_redraw()  # the last complete image stays under the new one meanwhile
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
        first = not any(self.panels[p].cell.image is not None for p in plots)
        label = self.labels[1] if first else "re-drawing the view"
        tabs = {id(self.panels[p].tab): self.panels[p].tab for p in plots if self.panels[p].tab is not None}

        def run(on_chunk):
            source = self._in_view({p: (g.x_extent, g.y_extent) for p, g in grids.items()}, job, read_data)
            for plot in plots:
                self.panels[plot].rows_read = source.n_rows
            for tab in tabs.values():
                tab.rows_read = len(rows_of_iterations(source, [p.plot.iteration for p in tab.panels]))
            # spread over the time range, so each refresh shows the whole view filling in
            return source.stream(list(grids.values()), read_data=read_data, on_chunk=on_chunk, spread=True)

        def done(job):
            if not job.completed:
                return
            finished = {}
            for plot, grid in grids.items():
                panel = self.panels.get(plot)
                if panel is not None and panel.grid is grid:  # still shown, its grid the one this pass filled
                    panel.drawn = True
                    self.to_draw.discard(plot)
                    panel.cell.end_redraw()
                    finished[id(panel.figure)] = panel
            for panel in finished.values():  # each figure once, its plots together
                self._render(panel, final=all(p.drawn for p in self._figure_panels(panel.figure)))
                if first and self.report is not None and panel.figure.limits_warning:  # once: the whole view
                    name = panel.tab.page.name if panel.tab is not None else panel.plot.name
                    self.report("warning", f"{name}: {panel.figure.limits_warning}")

        job = _Job("draw", label, run, done, self.source.row_bytes if read_data else 0, self.source.n_rows)
        self._queue(job)

    def _in_view(self, views: dict, job: _Job, read_data: bool) -> XYSource:
        """For a pass reading visibility data: the selection narrowed to the
        rows that can hold a point inside the views ({plot: (x limits, y
        limits)}), found from row metadata alone (`rows_in_views`), the job's
        progress counting those rows; for one reading none, the selection."""
        source = source_in_views(self.source, views, read_data)
        job.progress.total_rows = source.n_rows
        return source

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
                    rendered = set()
                    for plot in self.to_draw:
                        panel = self.panels.get(plot)
                        if panel is not None and id(panel.figure) not in rendered:  # each figure once
                            rendered.add(id(panel.figure))
                            self._render(panel)
                    self.last_refresh = now
        elif self.statusBar().currentMessage().endswith("elapsed"):
            self.statusBar().showMessage("ready")
        self._check_views(now)

    def _render(self, panel: _Panel, final: bool = False) -> None:
        """Show the panel's figure from its plots' grids (a page's together);
        `final`: every plot of it drawn, so its status says what was."""
        shown = [p for p in self._figure_panels(panel.figure) if p.grid is not None]
        snapshots = {p.plot: p.grid.snapshot() for p in shown}
        panel.figure.show_page(snapshots, display_dpi=panel.figure.fig.dpi)
        if final and panel.tab is not None:
            panel.figure.set_status(grids_summary(list(snapshots.values()), panel.tab.rows_read or panel.tab.rows,
                                                  panel.tab.rows))
        elif final:
            panel.figure.set_status(grids_summary(list(snapshots.values()), panel.rows_read or self.source.n_rows,
                                                  self.source.n_rows))
        else:
            job = self.jobs[0] if self.jobs else None
            panel.figure.set_status(job.progress.text(job.rows_done) if job else "")
        panel.canvas.draw_idle()

    def _check_views(self, now: float) -> None:
        """After a zoom or pan settles, re-draw that plot over the new limits
        (a page's shared axis: kept for its next pages); after a resize
        settles, re-draw it at the window's new pixel size (T19 point D:
        re-bin, where the image was stretched). Zoom or pan turned on while
        Locate is on turns Locate off."""
        redraw = []
        for plot, panel in list(self.panels.items()):
            if panel.selector is not None and panel.toolbar.mode.name != "NONE":
                panel.locate_action.setChecked(False)
            if panel.grid is None or panel.cell.image is None:
                continue
            limits = (tuple(panel.cell.ax.get_xlim()), tuple(panel.cell.ax.get_ylim()))
            if not same_view(limits, panel.last_limits):
                panel.last_limits, panel.changed_at = limits, now
                panel.cell.hide_if_view_moved(panel.grid)
                continue
            if panel.mouse_down or now - panel.changed_at < SETTLE_S:
                continue
            if (not same_view(limits, (panel.grid.x_extent, panel.grid.y_extent))
                    and not same_view(limits, panel.extent)):
                # the zoomed view, kept as the view the aspect toggle returns to; with equal
                # aspect, widened again so both axes keep one scale and one span
                panel.extent = self._set_view(panel, *limits)
                if panel.tab is not None and panel.figure.page is not None:
                    for axis, shared, extent in zip(("x", "y"), panel.figure.page.shared, panel.extent):
                        if shared:
                            panel.tab.view[axis] = tuple(extent)
                redraw.append(plot)
                continue
            shape = self._pixel_shape(panel)
            if shape != panel.last_shape:
                panel.last_shape, panel.resized_at = shape, now
                continue
            if now - panel.resized_at >= SETTLE_S and self._resized(panel):
                redraw.append(plot)  # a new grid at the window's size
        if redraw:  # together: a page's plots, zoomed or resized at once, in one request and one pass
            self.request_draw(redraw)

    def _toggle_aspect(self, panels: list[_Panel], equal: bool) -> None:
        """Equal scale on: widen the requested view so a unit is the same
        length on both axes; off: back to the requested view (the data's
        range, or the last zoom). On a page, for each of its plots."""
        for panel in panels:
            panel.cell.equal_override = equal
        known = [panel for panel in panels if panel.extent is not None]  # the rest: when the ranges are known
        for panel in known:
            panel.extent = self._set_view(panel, *panel.cell.view_request)
            panel.last_limits = (tuple(panel.extent[0]), tuple(panel.extent[1]))
        if known:
            self.request_draw([panel.plot for panel in known])

    # ---- locate -------------------------------------------------------------

    def _toggle_locate(self, panels: list[_Panel], on: bool) -> None:
        """Locate is a mode like zoom and pan: turning it on turns them off.
        (Zoom and pan lock the canvas, and the box selector ignores every
        event while they hold the lock.) On a page, a box on any of its
        plots locates that plot's samples."""
        if not panels:
            return
        if on:
            toolbar = panels[0].toolbar
            if toolbar.mode.name == "ZOOM":
                toolbar.zoom()
            elif toolbar.mode.name == "PAN":
                toolbar.pan()
            for panel in panels:
                panel.selector = RectangleSelector(
                    panel.cell.ax, lambda press, release, p=panel: self._locate(p, press, release),
                    useblit=True, button=[1], interactive=False, minspanx=2, minspany=2, spancoords="pixels",
                )
            self.statusBar().showMessage("Locate: drag a box on the plot to list the samples inside it")
            return
        for panel in panels:
            if panel.selector is not None:
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
        self._show_locate_panel()

        def run(on_chunk):
            source = self._in_view({panel.plot: (box_x, box_y)}, job, read_data)
            return source.stream([locate], read_data=read_data, on_chunk=on_chunk)

        def done(job):
            if job.completed:
                self.locate = locate
                self._show_locate(locate)
                self.save_locate_button.setEnabled(locate.n_found > 0)
                self.statusBar().showMessage(
                    f"located {locate.n_found:,} samples; listed in the Located samples panel")
            else:
                self.locate_summary.setText("locate stopped before the end of the selection")

        job = _Job("locate", "locating samples", run, done, self.source.row_bytes if read_data else 0,
                   self.source.n_rows)
        self._queue(job, front=True)

    def _show_locate(self, locate: LocateReducer) -> None:
        self._show_locate_panel()
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
            source = self._in_view({located.plot: (located.x_box, located.y_box)}, job, read_data)
            return source.stream([again], read_data=read_data, on_chunk=on_chunk)

        def done(job):
            written = writer.close(again, completed=job.completed and job.error is None)
            self._finish_record(record, "locate", written,
                                None if written else "stopped before the end of the selection; no file written")
            self.statusBar().showMessage(
                f"saved {again.n_found:,} located samples to {written}" if written
                else "saving the located samples stopped before the end; no file written")

        job = _Job("locate-csv", f"writing all {located.n_found:,} located samples", run, done,
                   self.source.row_bytes if read_data else 0, self.source.n_rows)
        self._queue(job, front=True)

    # ---- provenance of saved files -----------------------------------------------

    def _report(self, level: str, text: str) -> None:
        if self.report is not None:
            self.report(level, text)

    def _start_record(self, kind: str, plot: PlotSpec, path, name: str | None = None,
                      **details) -> tuple[bool, PlotRecord | None]:
        """(whether to go ahead with the save, its record), the save named
        `name` (default: the plot's). Without provenance there is no record;
        a record that cannot be written stops the save."""
        if self.provenance is None:
            return True, None
        try:
            return True, self.provenance.start(kind, name or plot.name, str(Path(path).resolve()), **details)
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

    def _export(self, panels: list[_Panel]) -> None:
        if not panels:
            return
        panel = panels[0]
        name = (panel.tab.page.name if panel.tab is not None
                else self.figure_names.get(id(panel.figure), panel.plot.name))
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export plot", f"{name}.png", EXPORT_FILTERS)
        if not path:
            return
        dpi, ok = QtWidgets.QInputDialog.getInt(self, "Export resolution", "dots per inch:", 600, *DPI_LIMITS)
        if ok:
            self.export(panel.plot, path, dpi)

    def export(self, plot: PlotSpec, path: str, dpi: int) -> None:
        """Re-read the plot's current view at `dpi` and save it to `path`; a
        plot of a page: the whole page, each plot over its view."""
        panel = self.panels[plot]
        panels = self._figure_panels(panel.figure)  # one plot, a stack's, or a page's
        figure = panel.figure
        views = {p.plot: (tuple(p.cell.ax.get_xlim()), tuple(p.cell.ax.get_ylim())) for p in panels}
        # the Equal aspect box's override takes effect on linear axes only
        equal = panel.cell.equal_override if panel.cell.linear else None
        name = panel.tab.page.name if panel.tab is not None else self.figure_names.get(id(figure), plot.name)
        started, record = self._start_record("export", plot, path, view=views[plot], dpi=dpi,
                                             figure_size_in=tuple(figure.fig.get_size_inches()), equal=equal,
                                             name=name)
        if not started:
            return
        shapes = figure.grid_shapes(dpi)
        grids = {p.plot: GridReducer(p.plot, *views[p.plot], *shapes[p.plot]) for p in panels}
        read_data = any(p.plot.needs_data for p in panels)
        iterations = [p.plot.iteration for p in panels] if panel.tab is not None else None
        rows_read = []

        def run(on_chunk):
            source = self._in_view(views, job, read_data)
            rows_read.append(len(rows_of_iterations(source, iterations)) if iterations else source.n_rows)
            return source.stream(list(grids.values()), read_data=read_data, on_chunk=on_chunk)

        def done(job):
            if not job.completed:
                self._finish_record(record, "export", None, "stopped before the end of the selection; nothing saved")
                return
            figure.show_page(grids, display_dpi=dpi)
            status, shown_record = figure.status.get_text(), figure.record_id
            of_rows = panel.tab.rows if panel.tab is not None else self.source.n_rows
            figure.set_status(grids_summary(list(grids.values()), rows_read[0], of_rows))
            if record is not None:
                figure.set_record(record.run_id)  # the file names the export's own record
            previews = [p.cell.preview for p in panels if p.cell.preview is not None]  # a redraw's: out of the file
            for preview in previews:
                preview.set_visible(False)
            try:
                figure.fig.savefig(path, dpi=dpi)
                error = None
            except (OSError, ValueError) as err:
                error = f"could not save {path}: {err}"
            for preview in previews:
                preview.set_visible(True)
            figure.set_status(status)
            figure.set_record(shown_record)
            shown = {p.plot: p.grid.snapshot() for p in panels if p.grid is not None}
            if shown:  # back to the window's own image
                figure.show_page(shown, display_dpi=figure.fig.dpi)
            panel.canvas.draw_idle()
            self._finish_record(record, "export", None if error else path, error)
            if error is None:
                self.statusBar().showMessage(f"exported {path} at {dpi} dpi")

        job = _Job("export", f"exporting at {dpi} dpi", run, done, self.source.row_bytes if read_data else 0,
                   self.source.n_rows)
        self._queue(job, front=True)

    # ---- closing ----------------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt's name)
        for job in self.jobs:
            job.stop = True
        for job in self.jobs[:1]:
            job.join(timeout=5)
        self.timer.stop()
        super().closeEvent(event)

    def idle(self) -> bool:
        """No job running or waiting, and no page waiting to be laid out."""
        return not self.jobs and not self.laying_out


_LOCATE_COLUMNS = [
    ("baseline", str), ("time (UTC)", str), ("channel", str), ("freq (MHz)", float), ("Stokes", str),
    ("x", float), ("y", float), ("weight", float), ("flagged", str), ("mirrored", str), ("row", int),
    ("source", str),
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
            r["x"], r["y"], r["weight"], r["flagged"], "yes" if r["mirrored"] else "no", r["row"],
            sources.get(r["source_id"], str(r["source_id"])),
        ])
    return rows


def beside(window: QtCore.QRect, screen: QtCore.QRect, size: QtCore.QSize) -> QtCore.QRect:
    """Where a panel of `size` opens: right of `window`, its top level with
    the window's, if the screen has the room; else against the screen's
    right edge, over the window. Kept on the screen either way."""
    width, height = min(size.width(), screen.width()), min(size.height(), screen.height())
    x = window.x() + window.width()
    if x + width > screen.x() + screen.width():
        x = screen.x() + screen.width() - width
    y = min(max(window.y(), screen.y()), screen.y() + screen.height() - height)
    return QtCore.QRect(x, y, width, height)


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
