"""The visplot GUI's main window.

Left: the request form (`gui.form`), and under it the live counts of what
the form selects, the form's own command line, and Plot / Clear. Right: a
tab per plot, each the plot window (`visplot.qt_inspector`) with the
command that reproduces it. Bottom: messages (what a run reports, as the
command line prints it) and the history of plots made.

Plot builds a `PlotRequest` from the form and runs it through
`visplot.run` -- the same `check_request`, `prepare` and plot window the
command line runs -- so a GUI plot and its command line are one run.
Opening a file, counting a selection and preparing a plot run on
background threads; the window polls them.

Provenance (T31, `visplot.records`): the window's session has a record
whose log keeps every message the window shows; each Plot, and each CSV or
export a plot tab saves, has its own record (under the request's
--provenance-dir), naming the session. A plot's record is complete when
its tab opens.
"""

from __future__ import annotations

import threading
import time
import traceback
from typing import Callable

from PySide6 import QtCore, QtGui, QtWidgets

from visplot.file_summary import summarize
from visplot.gui.form import RequestForm
from visplot.gui.widgets import CommandLine
from visplot.gui.style import THEMES, apply_theme
from visplot.records import PlotRecord, SessionRecord, WindowProvenance
from visplot.request import build_arg_parser
from visplot.run import RequestError, check_request, count_selection, open_file, prepare

POLL_MS = 100
COUNT_DELAY_MS = 400
# The Messages list keeps the newest this many (about 600 bytes each: a few MB at most);
# History keeps this many plots. Every message stays in the session's log.
MAX_MESSAGES = 5000
MAX_HISTORY = 1000


class _Task:
    """`fn()` on a background thread; the window polls `done` and then
    calls `on_done(result)` or `on_error(message)` on its own thread."""

    def __init__(self, fn: Callable, on_done: Callable, on_error: Callable):
        self.fn, self.on_done, self.on_error = fn, on_done, on_error
        self.result = None
        self.error: str | None = None
        self.done = False
        self.thread = threading.Thread(target=self._body, daemon=True)

    def _body(self):
        try:
            self.result = self.fn()
        except RequestError as err:
            self.error = str(err)
        except Exception:
            self.error = traceback.format_exc()
        self.done = True


class VisplotWindow(QtWidgets.QMainWindow):
    """`provenance_dir`: where the session's record goes (default: the
    --provenance-dir default); each plot's record goes to its request's
    --provenance-dir, the form's Records field."""

    def __init__(self, fits_path: str | None = None, theme: str = "light", provenance_dir: str | None = None):
        super().__init__()
        self.setWindowTitle("visplot")
        self.theme = theme
        self.opened = None
        self._tasks: list[_Task] = []
        self._pending_reports: list[tuple[str, str]] = []
        self._reports_lock = threading.Lock()
        self._count_generation = 0
        self.session = SessionRecord(provenance_dir or build_arg_parser().get_default("provenance_dir"))

        self.form = RequestForm()
        self.form.open_button.clicked.connect(self._choose_file)
        self.form.path_edit.returnPressed.connect(lambda: self.open_path(self.form.path_edit.text().strip()))
        self.form.cache_button.clicked.connect(self._choose_cache_dir)
        self.form.provenance_button.clicked.connect(self._choose_provenance_dir)
        self.form.show_session(self.session.session_id, self.session.log_path)
        self.form.changed.connect(self._form_changed)

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(self._sidebar())
        splitter.addWidget(self._plot_area())
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([430, 1070])
        self.setCentralWidget(splitter)
        self._build_docks()
        self._build_menus()
        self._build_status_bar()

        self.count_timer = QtCore.QTimer(self, singleShot=True, interval=COUNT_DELAY_MS)
        self.count_timer.timeout.connect(self._count)
        self.poll_timer = QtCore.QTimer(self, interval=POLL_MS)
        self.poll_timer.timeout.connect(self._poll)
        self.poll_timer.start()
        self.report("info", f"session {self.session.session_id}; every message is kept in {self.session.log_path}")
        self._form_changed()
        if fits_path:
            self.open_path(fits_path)

    # ---- layout ----------------------------------------------------------------

    def _sidebar(self) -> QtWidgets.QWidget:
        scroll = QtWidgets.QScrollArea(objectName="sidebar")
        scroll.setWidget(self.form)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        panel = QtWidgets.QFrame()
        panel.setFrameShape(QtWidgets.QFrame.StyledPanel)
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(6)
        self.counts = QtWidgets.QLabel("Open a file to see what the selection holds.", objectName="counts")
        self.counts.setMinimumHeight(2 * self.counts.fontMetrics().lineSpacing() + 4)
        self.problem = QtWidgets.QLabel("", objectName="problem")
        self.problem.setWordWrap(True)
        self.form_command = CommandLine()
        copy = QtWidgets.QToolButton(text="Copy")
        copy.clicked.connect(lambda: QtGui.QGuiApplication.clipboard().setText(self.form_command.text()))
        self.plot_button = QtWidgets.QPushButton("Plot", objectName="primary")
        self.plot_button.setDefault(True)
        self.plot_button.clicked.connect(self._plot)
        clear = QtWidgets.QPushButton("Clear")
        clear.setToolTip("Close every plot tab")
        clear.clicked.connect(self._clear_plots)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(clear)
        buttons.addStretch(1)
        buttons.addWidget(self.plot_button)
        command_row = QtWidgets.QHBoxLayout()
        command_row.addWidget(self.form_command, 1)
        command_row.addWidget(copy)
        layout.addWidget(self.counts)
        layout.addWidget(self.problem)
        layout.addLayout(command_row)
        layout.addLayout(buttons)

        side = QtWidgets.QWidget()
        side_layout = QtWidgets.QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(0)
        side_layout.addWidget(scroll, 1)
        side_layout.addWidget(panel)
        side.setMinimumWidth(380)
        return side

    def _plot_area(self) -> QtWidgets.QWidget:
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.setMovable(True)
        self.tabs.tabCloseRequested.connect(self._close_tab)
        welcome = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(welcome)
        layout.addStretch(1)
        title = QtWidgets.QLabel("visplot", objectName="welcomeTitle", alignment=QtCore.Qt.AlignCenter)
        text = QtWidgets.QLabel(
            "Open a UVFITS file, choose what to plot on the left, and press Plot.\n"
            "Each plot opens in its own tab, with the command line that reproduces it;\n"
            "bin/visplot.sh runs the same request from a terminal.",
            objectName="welcomeText", alignment=QtCore.Qt.AlignCenter)
        layout.addWidget(title)
        layout.addWidget(text)
        layout.addStretch(2)
        self.tabs.addTab(welcome, "Start")
        self.tabs.tabBar().setTabButton(0, QtWidgets.QTabBar.RightSide, None)
        return self.tabs

    def _build_docks(self) -> None:
        self.messages = QtWidgets.QListWidget()
        self.messages.setWordWrap(True)
        messages_dock = QtWidgets.QDockWidget("Messages", self, objectName="messagesDock")
        messages_dock.setWidget(self.messages)
        self.history = QtWidgets.QTableWidget(0, 4)
        self.history.setHorizontalHeaderLabels(["time", "record", "plot", "command"])
        self.history.horizontalHeader().setStretchLastSection(True)
        self.history.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.history.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.history.setToolTip("Double-click a plot to load its request into the form")
        self.history.cellDoubleClicked.connect(self._load_history)
        self._history_requests = []
        history_dock = QtWidgets.QDockWidget("History", self, objectName="historyDock")
        history_dock.setWidget(self.history)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, messages_dock)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, history_dock)
        self.tabifyDockWidget(messages_dock, history_dock)
        messages_dock.raise_()
        self.resizeDocks([messages_dock], [150], QtCore.Qt.Vertical)
        self.docks = (messages_dock, history_dock)

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction("&Open UVFITS…", QtGui.QKeySequence.Open, self._choose_file)
        save = file_menu.addAction("Save plots as files…")
        save.setEnabled(False)
        save.setToolTip("The command line's --output-dir run from the form (next step)")
        file_menu.addSeparator()
        file_menu.addAction("&Quit", QtGui.QKeySequence.Quit, self.close)
        view = self.menuBar().addMenu("&View")
        group = QtGui.QActionGroup(self)
        for theme in THEMES:
            action = view.addAction(f"{theme.capitalize()} theme", lambda t=theme: self.set_theme(t))
            action.setCheckable(True)
            action.setChecked(theme == self.theme)
            group.addAction(action)
        view.addSeparator()
        for dock in self.docks:
            view.addAction(dock.toggleViewAction())
        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction("Command-line options…", self._show_cli_help)

    def _build_status_bar(self) -> None:
        self.file_label = QtWidgets.QLabel("no file")
        self.statusBar().addPermanentWidget(self.file_label)
        self.statusBar().showMessage("ready")

    # ---- files -------------------------------------------------------------------

    def _choose_file(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open UVFITS", "", "UVFITS (*.fits *.FITS *.uvfits);;All (*)")
        if path:
            self.open_path(path)

    def _choose_cache_dir(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Range cache folder")
        if path:
            self.form.cache_edit.setText(path)

    def _choose_provenance_dir(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Folder for provenance records")
        if path:
            self.form.provenance_edit.setText(path)

    def open_path(self, path: str) -> None:
        """Read the file's index, headers and tables in the background."""
        if not path:
            return
        self.form.path_edit.setText(path)
        self.statusBar().showMessage(f"opening {path} …")

        def done(result):
            self.opened, summary = result
            self.form.set_file(self.opened, summary)
            self.file_label.setText(path)
            self.statusBar().showMessage("file open")
            self.report("info", f"opened {path}: {summary.n_rows:,} rows, {len(summary.sources)} sources")
            self._form_changed()

        def failed(message):
            self.opened = None
            self.statusBar().showMessage("could not open the file")
            self.report("warning", message)
            self._form_changed()

        self._start(lambda: (lambda opened: (opened, summarize(opened)))(open_file(path)), done, failed)

    # ---- the form -------------------------------------------------------------------

    def _form_changed(self) -> None:
        """Check the form (field checks, then the request as the command line
        checks it), show its command, and recount the selection shortly."""
        problem, request = None, None
        invalid = self.form.invalid_fields()
        if invalid:
            problem = f"check {', '.join('--' + d.replace('_', '-') for d in invalid)} (red frame; its tooltip says why)"
        else:
            try:
                request = self.form.request()
            except (ValueError, TypeError) as err:
                problem = str(err)
        if request is None:
            self.form_command.set_text(f"(no command until the form is fixed: {problem})")
        else:
            self.form_command.set_text(request.command_line())
            if not request.fits_path:
                problem = "open a file first"
            else:
                try:
                    check_request(request)
                except RequestError as err:
                    problem = str(err)
                except Exception:  # a bug here must not freeze the form: say so
                    problem = "the request could not be checked (details in Messages)"
                    self.report("warning", traceback.format_exc())
        if self.opened is None and not problem:
            problem = "the file is still opening" if self._tasks else "open the file (Enter in the File field)"
        self.problem.setText(problem or "")
        self.plot_button.setEnabled(problem is None)
        self.plot_button.setToolTip(problem or "Plot this request in a new tab")
        if self.opened is not None and request is not None:
            self.count_timer.start()

    def _count(self) -> None:
        if self.opened is None:
            return
        self._count_generation += 1
        generation = self._count_generation
        try:
            request = self.form.request()
        except (ValueError, TypeError):
            return  # the form shows the problem
        opened = self.opened

        def done(count):
            if generation == self._count_generation:
                self.counts.setText(f"Selected: {count.rows:,} rows × {count.samples_per_row:,} = {count.samples:,} "
                                    f"samples\n{count.gigabytes:.1f} GB of visibilities to read for a data plot")

        def failed(message):
            if generation == self._count_generation:
                self.counts.setText(f"selection: {message}")

        self.counts.setText("counting …")
        self._start(lambda: count_selection(request, opened), done, failed)

    # ---- plotting --------------------------------------------------------------------

    def _plot(self) -> None:
        """Run the form's request as the command line does, into a new tab,
        with its provenance record (no record, no plot)."""
        request = self.form.request()
        opened = self.opened
        session_id = self.session.session_id
        self.plot_button.setEnabled(False)
        self.statusBar().showMessage("preparing the plot …")

        def work():
            try:
                record = PlotRecord(request, "plot", session_id=session_id)
            except Exception as err:
                raise RequestError(f"not plotted: the provenance record could not be written in "
                                   f"{request.provenance_dir} ({type(err).__name__}: {err})") from err
            self.report("info", f"plot {record.run_id}: {record.command}")

            def report(level: str, text: str) -> None:
                self.report(level, text)
                record.log(level, text)

            try:
                run = prepare(request, opened, report=report)
            except BaseException as err:
                record.finish(str(err) if isinstance(err, RequestError) else f"{type(err).__name__}: {err}")
                raise
            record.finish()
            return run, record

        def done(result):
            from visplot.qt_inspector import InspectorWindow

            run, record = result
            provenance = WindowProvenance(request, len(run.xy_plots), parent_run_id=record.run_id,
                                          session_id=session_id)
            panel = InspectorWindow(run.source, run.figures(), labels=run.labels, cache=run.cache, cached=run.cached,
                                    provenance=provenance, report=self.report)
            panel.setWindowFlags(QtCore.Qt.Widget)
            tab = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(tab)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(panel, 1)
            footer = QtWidgets.QHBoxLayout()
            footer.setContentsMargins(6, 2, 6, 4)
            command = CommandLine()
            command.set_text(request.command_line())
            copy = QtWidgets.QToolButton(text="Copy")
            copy.clicked.connect(lambda: QtGui.QGuiApplication.clipboard().setText(command.text()))
            record_label = QtWidgets.QLabel(f"record {record.run_id}", objectName="hint")
            record_label.setToolTip(str(record.path))
            record_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            footer.addWidget(QtWidgets.QLabel("Command:"))
            footer.addWidget(command, 1)
            footer.addWidget(copy)
            footer.addWidget(record_label)
            layout.addLayout(footer)
            tab.panel, tab.request, tab.run, tab.record = panel, request, run, record
            title = ", ".join(name for name, _ in run.figures())
            self.tabs.setCurrentIndex(self.tabs.addTab(tab, title))
            self._add_history(request, record, title)
            self.statusBar().showMessage(f"{title}: reading and drawing; progress is under the plot", 8000)
            self._form_changed()

        def failed(message):
            self.report("warning", message)
            self.statusBar().showMessage("the plot could not be prepared")
            self._form_changed()

        self._start(work, done, failed)

    def _close_tab(self, index: int) -> None:
        tab = self.tabs.widget(index)
        if hasattr(tab, "panel"):
            tab.panel.close()  # stops its reading
        self.tabs.removeTab(index)
        tab.deleteLater()

    def _clear_plots(self) -> None:
        for index in reversed(range(self.tabs.count())):
            if hasattr(self.tabs.widget(index), "panel"):
                self._close_tab(index)

    def _add_history(self, request, record: PlotRecord, title: str) -> None:
        if self.history.rowCount() >= MAX_HISTORY:
            self.history.removeRow(0)
            self._history_requests.pop(0)
        self._history_requests.append(request)
        row = self.history.rowCount()
        self.history.insertRow(row)
        for column, text in enumerate((time.strftime("%H:%M:%S"), record.run_id, title, request.command_line())):
            item = QtWidgets.QTableWidgetItem(text)
            if column == 1:
                item.setToolTip(str(record.path))
            self.history.setItem(row, column, item)

    def _load_history(self, row: int, _column: int) -> None:
        self.form.load(self._history_requests[row])
        self.report("info", "loaded a plot's request into the form")

    # ---- messages, tasks, themes ------------------------------------------------------

    def report(self, level: str, text: str) -> None:
        """A run's report callback: safe from any thread; shown on the next poll."""
        with self._reports_lock:
            self._pending_reports.append((level, text))

    def _start(self, fn, on_done, on_error) -> None:
        task = _Task(fn, on_done, on_error)
        self._tasks.append(task)
        task.thread.start()

    def _poll(self) -> None:
        self._show_reports()
        for task in [t for t in self._tasks if t.done]:
            self._tasks.remove(task)
            if task.error is None:
                task.on_done(task.result)
            else:
                task.on_error(task.error)

    def _show_reports(self) -> None:
        """The messages reported since the last poll: into the session's log
        and the Messages list."""
        with self._reports_lock:
            reports, self._pending_reports = self._pending_reports, []
        for level, text in reports:
            self.session.log(level, text)
            item = QtWidgets.QListWidgetItem(f"{time.strftime('%H:%M:%S')}  {'WARNING: ' if level == 'warning' else ''}{text}")
            if level == "warning":
                item.setForeground(QtGui.QColor("#c93c37"))
                self.docks[0].raise_()
            self.messages.addItem(item)
        while self.messages.count() > MAX_MESSAGES:
            self.messages.takeItem(0)
        if reports:
            self.messages.scrollToBottom()

    def set_theme(self, theme: str) -> None:
        self.theme = theme
        apply_theme(QtWidgets.QApplication.instance(), theme)

    def _show_cli_help(self) -> None:
        box = QtWidgets.QDialog(self)
        box.setWindowTitle("bin/visplot.sh --help")
        layout = QtWidgets.QVBoxLayout(box)
        text = QtWidgets.QPlainTextEdit(build_arg_parser().format_help(), readOnly=True)
        text.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        layout.addWidget(text)
        box.resize(900, 700)
        box.show()

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt's name)
        self._clear_plots()
        self.poll_timer.stop()
        self._show_reports()  # the last messages, into the session's log
        self.session.close()
        super().closeEvent(event)


def run_gui(fits_path: str | None = None) -> int:
    """Open the GUI (with `fits_path`, if given) and run until it is closed."""
    import signal

    signal.signal(signal.SIGINT, signal.SIG_DFL)  # Ctrl-C in the terminal closes the window
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    apply_theme(app, "light")
    window = VisplotWindow(fits_path)
    window.resize(1500, 950)
    window.show()
    return app.exec()
