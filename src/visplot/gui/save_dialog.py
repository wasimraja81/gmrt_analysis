"""File > Save plots as files: the form's request run as the command line's
--output-dir run, with the options only a save takes (folder, filename
prefix, dpi, figure size, the high-resolution PDF) and the command it
runs, checked as the command line checks it."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6 import QtWidgets

from visplot.gui.widgets import CommandLine, form_layout, hint, row
from visplot.request import PlotRequest, build_arg_parser
from visplot.request_args import DPI_LIMITS, MAX_FIGURE_INCHES, resolve_figure_size_arg
from visplot.run import RequestError, check_request, highres_dpi


class SaveDialog(QtWidgets.QDialog):
    """`make_request(**actions)` is the form's request with the save's
    options (`actions()`); `previous` holds the last save's options, shown
    again."""

    def __init__(self, make_request: Callable[..., PlotRequest], previous: dict | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Save plots as files")
        self.make_request = make_request
        parser = build_arg_parser()
        previous = previous or {}

        self.folder = QtWidgets.QLineEdit(previous.get("output_dir") or "")
        self.folder.setPlaceholderText("a folder for the PNGs and PDFs")
        choose = QtWidgets.QPushButton("Choose…")
        choose.clicked.connect(self._choose_folder)
        self.prefix = QtWidgets.QLineEdit(previous.get("output_prefix") or "")
        self.prefix.setPlaceholderText(parser.get_default("output_prefix"))
        self.dpi = QtWidgets.QSpinBox()
        self.dpi.setRange(*DPI_LIMITS)
        self.dpi.setValue(previous.get("dpi", parser.get_default("dpi")))
        width, height = resolve_figure_size_arg(previous.get("figure_size", parser.get_default("figure_size")))
        self.width, self.height = QtWidgets.QDoubleSpinBox(), QtWidgets.QDoubleSpinBox()
        for box, value in ((self.width, width), (self.height, height)):
            box.setRange(0.5, MAX_FIGURE_INCHES)
            box.setDecimals(2)
            box.setSuffix(" in")
            box.setValue(value)
        self.highres = QtWidgets.QCheckBox()
        self.highres.setChecked(not previous.get("no_highres_pdf", False))
        self.command = CommandLine()
        self.problem = QtWidgets.QLabel("", objectName="problem")
        self.problem.setWordWrap(True)
        self.note = hint("")
        self.buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        body = QtWidgets.QWidget()
        form = form_layout(body)
        form.addRow("Folder", row(self.folder, choose, stretches=(1, 0)))
        form.addRow("Filename prefix", self.prefix)
        form.addRow("Resolution", row(self.dpi, QtWidgets.QLabel("dpi, the PNGs and PREFIX_lowres.pdf"),
                                      stretches=(0, 1)))
        form.addRow("Figure size", row(self.width, QtWidgets.QLabel("×"), self.height, QtWidgets.QWidget(),
                                       stretches=(0, 0, 0, 1)))
        form.addRow("", self.highres)
        form.addRow(hint("One PNG per plot, and PREFIX_lowres.pdf with a page per plot. The figure size is each "
                         "streamed plot's; the antenna layout and source listing keep their own."))
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(body)
        layout.addWidget(QtWidgets.QLabel("Command:"))
        layout.addWidget(self.command)
        layout.addWidget(self.problem)
        layout.addWidget(self.note)
        layout.addWidget(self.buttons)
        for signal in (self.folder.textChanged, self.prefix.textChanged, self.dpi.valueChanged,
                       self.width.valueChanged, self.height.valueChanged, self.highres.toggled):
            signal.connect(lambda *_: self._changed())
        self.resize(720, 0)
        self._changed()

    def _choose_folder(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Folder for the saved plots", self.folder.text())
        if path:
            self.folder.setText(path)

    def actions(self) -> dict:
        """The save's request options."""
        return dict(output_dir=self.folder.text().strip() or None,
                    output_prefix=self.prefix.text().strip() or self.prefix.placeholderText(),
                    dpi=self.dpi.value(), figure_size=f"{self.width.value():g},{self.height.value():g}",
                    no_highres_pdf=not self.highres.isChecked())

    def request(self) -> PlotRequest:
        return self.make_request(**self.actions())

    def _changed(self) -> None:
        """Show the command and what stops the save, if anything; name the
        files the save would replace."""
        self.highres.setText(f"also PREFIX_highres.pdf, at {highres_dpi(self.dpi.value())} dpi, for zooming in")
        problem, note = None, ""
        try:
            request = self.request()
        except (ValueError, TypeError) as err:
            request, problem = None, str(err)
            self.command.set_text(f"(no command: {problem})")
        if request is not None:
            self.command.set_text(request.command_line())
            try:
                check_request(request)
            except RequestError as err:
                problem = str(err)
        if problem is None and not request.output_dir:
            problem = "choose a folder"
        elif problem is None:
            existing = sorted(Path(request.output_dir).glob(f"{request.output_prefix}_*"))
            if existing:
                note = (f"{len(existing)} file(s) named {request.output_prefix}_* are in this folder already; "
                        "those with the same names are replaced")
        self.problem.setText(problem or "")
        self.note.setText(note)
        self.buttons.button(QtWidgets.QDialogButtonBox.Save).setEnabled(problem is None)
