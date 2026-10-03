"""The Listing tab (T20, listObs): what the open file holds, as CASA listobs
and AIPS LISTR's scan listing list it -- the sections to list, the scan
rule's two limits, the whole file or the form's selection, and List, Copy
and Save as text; the listing itself in fixed-width text. The window runs
it as the command line's --listobs runs it (`visplot.run.listing_text`).
Flags, unticked at first, reads every selected visibility: its progress
shows under the buttons, with Stop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PySide6 import QtGui, QtWidgets

from data_io.observation_summary import DEFAULT_GAP_INTEGRATIONS, DEFAULT_LONGEST_SCAN_S
from visplot.gui.widgets import hint, row
from visplot.listing import DEFAULT_SECTIONS, SECTION_TITLES, SECTIONS

LISTING_HELP = {
    "sections": "The parts of the listing (on the command line: --listobs SECTIONS)",
    "flags": "The share of visibilities flagged (weight not positive), by Stokes, source, scan and antenna: reads "
             "every selected visibility, one pass, so it takes as long as a plot's pass (the sidebar gives the GB "
             "for the form's selection; Stop ends it)",
    "gap": "Scans are derived (the file has no scan table), as AIPS INDXR derives them: a gap in the data longer "
           "than this many integration times starts a new scan (--scan-gap)",
    "longest": "A scan on one source longer than this is cut, the rest a new scan (INDXR's 60 min; --scan-longest)",
    "selection": "List only what the form selects (its sources, correlations, antennas, time range and other "
                 "filters, Stokes and channels), as listobs's selectdata; unticked, the whole file",
}


@dataclass(frozen=True)
class ListingChoices:
    sections: tuple[str, ...]
    scan_gap: float  # integration times
    scan_longest_min: float
    selection_only: bool


class ListingTab(QtWidgets.QWidget):
    """`on_list()` runs a listing of the tab's choices (`choices()`);
    `on_save(path)` saves the one shown."""

    def __init__(self, on_list: Callable[[], None], on_save: Callable[[str], None], on_stop: Callable[[], None],
                 parent=None):
        super().__init__(parent)
        self.on_save = on_save
        self.shown_request = None  # the request of the listing shown, for Save
        layout = QtWidgets.QVBoxLayout(self)
        self.boxes = {}
        sections = QtWidgets.QHBoxLayout()
        sections.addWidget(QtWidgets.QLabel("List"))
        for name in SECTIONS:
            box = QtWidgets.QCheckBox(SECTION_TITLES[name])
            box.setChecked(name in DEFAULT_SECTIONS)
            box.setToolTip(LISTING_HELP.get(name, LISTING_HELP["sections"]))
            self.boxes[name] = box
            sections.addWidget(box)
        sections.addStretch(1)
        layout.addLayout(sections)

        self.gap = QtWidgets.QDoubleSpinBox(decimals=1, minimum=0.5, maximum=1000.0, value=DEFAULT_GAP_INTEGRATIONS)
        self.gap.setSuffix(" integrations")
        self.gap.setToolTip(LISTING_HELP["gap"])
        self.longest = QtWidgets.QDoubleSpinBox(decimals=1, minimum=0.1, maximum=10000.0,
                                                value=DEFAULT_LONGEST_SCAN_S / 60.0)
        self.longest.setSuffix(" min")
        self.longest.setToolTip(LISTING_HELP["longest"])
        self.selection_only = QtWidgets.QCheckBox("the form's selection only")
        self.selection_only.setToolTip(LISTING_HELP["selection"])
        self.list_button = QtWidgets.QPushButton("List", objectName="primary")
        self.list_button.clicked.connect(on_list)
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.setToolTip("Stop counting the flags; nothing is listed, and the record says so")
        self.stop_button.clicked.connect(on_stop)
        self.stop_button.hide()
        self.copy_button = QtWidgets.QPushButton("Copy")
        self.copy_button.clicked.connect(lambda: QtGui.QGuiApplication.clipboard().setText(self.text.toPlainText()))
        self.save_button = QtWidgets.QPushButton("Save as text…")
        self.save_button.clicked.connect(self._save)
        layout.addWidget(row(QtWidgets.QLabel("Scan gap"), self.gap, QtWidgets.QLabel("longest scan"), self.longest,
                             self.selection_only, QtWidgets.QWidget(), self.list_button, self.stop_button,
                             self.copy_button, self.save_button, stretches=(0, 0, 0, 0, 0, 1, 0, 0, 0, 0)))
        self.status = hint("Open a file, then List: reads the header, tables and row index; Flags also reads every "
                           "selected visibility.")
        layout.addWidget(self.status)
        self.text = QtWidgets.QPlainTextEdit(readOnly=True)
        self.text.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.text.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        layout.addWidget(self.text, 1)
        self.set_file_open(False)

    def choices(self) -> ListingChoices:
        return ListingChoices(tuple(name for name in SECTIONS if self.boxes[name].isChecked()), self.gap.value(),
                              self.longest.value(), self.selection_only.isChecked())

    def set_file_open(self, is_open: bool) -> None:
        self.list_button.setEnabled(is_open)
        has_text = bool(self.text.toPlainText())
        self.copy_button.setEnabled(has_text)
        self.save_button.setEnabled(has_text and self.shown_request is not None)

    def show_busy(self, reading: bool) -> None:
        """A listing running; `reading`: it counts the flags (Stop shown)."""
        self.list_button.setEnabled(False)
        self.status.setText("listing …")
        self.stop_button.setEnabled(True)
        self.stop_button.setVisible(reading)

    def show_progress(self, text: str) -> None:
        self.status.setText(text)

    def stopping(self) -> None:
        self.stop_button.setEnabled(False)
        self.status.setText("stopping at the next chunk …")

    def show_listing(self, text: str, request, record_id: str) -> None:
        self.shown_request = request
        self.text.setPlainText(text)
        self.status.setText(f"listed: record {record_id} (hover for its command)")
        self.status.setToolTip(request.command_line())
        self.stop_button.hide()
        self.set_file_open(True)

    def show_failed(self, message: str) -> None:
        self.status.setText(f"not listed: {message.splitlines()[-1] if message else ''}")
        self.stop_button.hide()
        self.list_button.setEnabled(True)

    def _save(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save the listing", "listobs.txt", "Text (*.txt)")
        if path:
            self.on_save(path)
