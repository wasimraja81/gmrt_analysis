"""The Locate tab (T26 step 2b-3): build a run from --where conditions,
and a session panel listing a session's runs with their actions. Runs as
the command line's Locate does (`visplot.run.locate_by_conditions`,
`combine_runs`, `perform_run_operation`).

Undo, Restore, Replace, Intersect and Difference act on runs selected in
the session's table (1 row for undo/restore/replace, 2 for intersect/
difference, each row's own run number kept as its first column's data) --
no run number to remember or type.
"""

from __future__ import annotations

from typing import Callable

from PySide6 import QtCore, QtWidgets

from visplot.conditions import CONDITION_EXAMPLES, parse_condition
from visplot.gui.widgets import CheckedLineEdit, hint, row
from visplot.locate_where import SESSION_NAME

LOCATE_HELP = {
    "condition": "A condition on a data value or coordinate -- 'OPERAND CMP VALUE UNIT' or 'A < OPERAND < B'. "
                 "Examples:\n" + "\n".join(f"{text}: {meaning}" for text, meaning in CONDITION_EXAMPLES),
    "session": "The session this run is added to (letters, digits, '.', '_', '-'); default: the file's name.",
    "flagged": "Which of the file's already-flagged visibilities are candidates: exclude (default, searches only "
               "unflagged ones), only (a diagnosis of what is already flagged), or include both.",
}
# (header, the run entry's key) -- `entry["status"]` is added to each run by the window before `show_session`.
RUN_COLUMNS = [("Run", "run"), ("Status", "status"), ("Conditions", "conditions"), ("Cells found", "cells_found"),
              ("Rows whole", "rows_whole"), ("Written (UTC)", "written_utc")]


def _check_condition(text: str) -> str | None:
    parse_condition(text)
    return None


def _check_session(text: str) -> str | None:
    return None if SESSION_NAME.match(text) else "a name of letters, digits, '.', '_' and '-'"


class LocateTab(QtWidgets.QWidget):
    """`on_locate`/`on_replace`: a new run from `conditions()`/`session()`/
    `flagged()` (`on_replace` also the one selected run). `on_list_session`:
    refresh the table (`show_session`). `on_undo`/`on_restore`/
    `on_intersect`/`on_difference`: act on the table's selected run(s)."""

    def __init__(self, on_locate: Callable[[], None], on_replace: Callable[[], None],
                on_list_session: Callable[[], None], on_undo: Callable[[], None], on_restore: Callable[[], None],
                on_intersect: Callable[[], None], on_difference: Callable[[], None], parent=None):
        super().__init__(parent)
        self._is_open = False
        self._condition_rows: list[tuple[QtWidgets.QWidget, CheckedLineEdit]] = []
        layout = QtWidgets.QVBoxLayout(self)

        build = QtWidgets.QGroupBox("Build a run")
        build_layout = QtWidgets.QVBoxLayout(build)
        self.session_edit = CheckedLineEdit("the file's name", _check_session)
        self.session_edit.setToolTip(LOCATE_HELP["session"])
        self.session_edit.textChanged.connect(self._update_buttons)
        self.flagged_combo = QtWidgets.QComboBox()
        for text, value in (("exclude (default)", "exclude"), ("only", "only"), ("include", "include")):
            self.flagged_combo.addItem(text, value)
        self.flagged_combo.setToolTip(LOCATE_HELP["flagged"])
        build_layout.addWidget(row(QtWidgets.QLabel("Session"), self.session_edit, QtWidgets.QLabel("Flagged"),
                                   self.flagged_combo, stretches=(0, 1, 0, 1)))
        self.conditions_box = QtWidgets.QVBoxLayout()
        build_layout.addLayout(self.conditions_box)
        self._add_condition()
        add_condition = QtWidgets.QPushButton("+ Add condition")
        add_condition.clicked.connect(lambda: self._add_condition())
        self.locate_button = QtWidgets.QPushButton("Locate", objectName="primary")
        self.locate_button.setToolTip("Find every visibility of the selection meeting every condition, written to "
                                      "a run of the session")
        self.locate_button.clicked.connect(on_locate)
        locate_row = QtWidgets.QHBoxLayout()
        locate_row.addWidget(add_condition)
        locate_row.addStretch(1)
        locate_row.addWidget(self.locate_button)
        build_layout.addLayout(locate_row)
        self.build_status = hint("Open a file, add at least one condition on a data value, then Locate.")
        build_layout.addWidget(self.build_status)
        layout.addWidget(build)

        panel = QtWidgets.QGroupBox("Session")
        panel_layout = QtWidgets.QVBoxLayout(panel)
        self.table = QtWidgets.QTableWidget(0, len(RUN_COLUMNS))
        self.table.setHorizontalHeaderLabels([heading for heading, _ in RUN_COLUMNS])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._update_buttons)
        panel_layout.addWidget(self.table)
        actions = QtWidgets.QHBoxLayout()
        self.undo_button = QtWidgets.QPushButton("Undo")
        self.undo_button.setToolTip("Set the selected run aside: kept, restorable")
        self.undo_button.clicked.connect(on_undo)
        self.restore_button = QtWidgets.QPushButton("Restore")
        self.restore_button.setToolTip("Make the selected (aside) run active again")
        self.restore_button.clicked.connect(on_restore)
        self.replace_button = QtWidgets.QPushButton("Replace selected")
        self.replace_button.setToolTip("Runs the conditions above as a new run; the selected run is set aside once "
                                       "it is written")
        self.replace_button.clicked.connect(on_replace)
        self.intersect_button = QtWidgets.QPushButton("Intersect")
        self.intersect_button.setToolTip("The two selected runs, combined where both found a visibility; both set "
                                         "aside")
        self.intersect_button.clicked.connect(on_intersect)
        self.difference_button = QtWidgets.QPushButton("Difference (A − B)")
        self.difference_button.setToolTip("A: the first selected run (by row); B: the second -- A, less every "
                                          "visibility B also found; both set aside")
        self.difference_button.clicked.connect(on_difference)
        refresh = QtWidgets.QPushButton("Refresh")
        refresh.setToolTip("Re-read the session (another process may have changed it)")
        refresh.clicked.connect(on_list_session)
        for button in (self.undo_button, self.restore_button, self.replace_button, self.intersect_button,
                      self.difference_button):
            actions.addWidget(button)
        actions.addStretch(1)
        actions.addWidget(refresh)
        panel_layout.addLayout(actions)
        self.panel_summary = hint("Open a file to see its Locate session.")
        panel_layout.addWidget(self.panel_summary)
        self.panel_message = hint("")
        panel_layout.addWidget(self.panel_message)
        layout.addWidget(panel)
        layout.addStretch(1)
        self._update_buttons()

    # ---- condition lines ----------------------------------------------------

    def _add_condition(self) -> None:
        edit = CheckedLineEdit("e.g. amp(V) > 5", _check_condition)
        edit.setToolTip(LOCATE_HELP["condition"])
        edit.textChanged.connect(self._update_buttons)
        remove = QtWidgets.QToolButton(text="×")
        remove.setToolTip("Remove this condition")
        line = row(edit, remove, stretches=(1, 0))
        remove.clicked.connect(lambda: self._remove_condition(line))
        self.conditions_box.addWidget(line)
        self._condition_rows.append((line, edit))

    def _remove_condition(self, line: QtWidgets.QWidget) -> None:
        if len(self._condition_rows) <= 1:
            return  # always at least one row, empty if nothing else
        self._condition_rows = [(widget, edit) for widget, edit in self._condition_rows if widget is not line]
        self.conditions_box.removeWidget(line)
        line.deleteLater()
        self._update_buttons()

    def conditions(self) -> tuple[str, ...]:
        return tuple(edit.text().strip() for _, edit in self._condition_rows if edit.text().strip())

    def session(self) -> str | None:
        return self.session_edit.text().strip() or None

    def flagged(self) -> str:
        return self.flagged_combo.currentData()

    # ---- the session table ----------------------------------------------------

    def selected(self) -> list[tuple[int, str]]:
        """(run, status) of each selected row, in the table's own (run
        number) order."""
        rows = sorted({index.row() for index in self.table.selectedIndexes()})
        return [self.table.item(r, 0).data(QtCore.Qt.UserRole) for r in rows]

    def selected_run(self) -> int | None:
        selected = self.selected()
        return selected[0][0] if len(selected) == 1 else None

    def selected_pair(self) -> tuple[int, int] | None:
        selected = self.selected()
        return (selected[0][0], selected[1][0]) if len(selected) == 2 else None

    def show_session(self, name: str, runs: list[dict]) -> None:
        """`name`'s runs, each a log entry with "status" added (`run_status`,
        done by the window before calling this)."""
        self.table.setRowCount(len(runs))
        for i, entry in enumerate(runs):
            for col, (_, key) in enumerate(RUN_COLUMNS):
                value = entry.get(key, "")
                text = (" and ".join(value) if key == "conditions"
                       else f"{value:,}" if isinstance(value, int) else str(value))
                item = QtWidgets.QTableWidgetItem(text)
                if col == 0:
                    item.setData(QtCore.Qt.UserRole, (entry["run"], entry["status"]))
                self.table.setItem(i, col, item)
        if not runs:
            summary = f"session {name}: no runs yet"
        else:
            active = [entry["run"] for entry in runs if entry["status"] == "active"]
            found = (f"the union of run{'' if len(active) == 1 else 's'} {', '.join(map(str, active))}" if active
                    else "empty, no run active")
            summary = f"session {name}: {len(runs)} run(s); found set: {found}"
        self.panel_summary.setText(summary)
        self._update_buttons()

    # ---- messages and state ----------------------------------------------------

    def show_build_status(self, text: str) -> None:
        self.build_status.setText(text)

    def show_panel_message(self, text: str) -> None:
        self.panel_message.setText(text)

    def set_file_open(self, is_open: bool) -> None:
        self._is_open = is_open
        if not is_open:
            self.table.setRowCount(0)
            self.panel_summary.setText("Open a file to see its Locate session.")
        self._update_buttons()

    def _update_buttons(self, *_) -> None:
        valid = self.session_edit.valid and all(edit.valid for _, edit in self._condition_rows)
        can_locate = self._is_open and valid and bool(self.conditions())
        self.locate_button.setEnabled(can_locate)
        selected = self.selected()
        statuses = [status for _, status in selected]
        one_active, one_aside = statuses == ["active"], statuses == ["aside"]
        two_active = statuses == ["active", "active"]
        self.undo_button.setEnabled(self._is_open and one_active)
        self.restore_button.setEnabled(self._is_open and one_aside)
        self.replace_button.setEnabled(can_locate and one_active)
        self.intersect_button.setEnabled(self._is_open and two_active)
        self.difference_button.setEnabled(self._is_open and two_active)
