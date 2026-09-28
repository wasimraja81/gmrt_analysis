"""Small widgets the GUI's panels are built from."""

from __future__ import annotations

import html
from typing import Callable

from PySide6 import QtCore, QtGui, QtWidgets


class CollapsibleSection(QtWidgets.QWidget):
    """A titled section whose body folds away under its header."""

    def __init__(self, title: str, expanded: bool = True, parent=None):
        super().__init__(parent)
        self.header = QtWidgets.QToolButton(text=title, checkable=True, checked=expanded)
        self.header.setObjectName("sectionHeader")
        self.header.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.header.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.header.toggled.connect(self.set_expanded)
        self.body = QtWidgets.QWidget()
        self.body.setObjectName("sectionBody")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.header)
        layout.addWidget(self.body)
        self.set_expanded(expanded)

    def set_expanded(self, expanded: bool) -> None:
        self.header.setArrowType(QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow)
        self.body.setVisible(expanded)


def form_layout(parent: QtWidgets.QWidget) -> QtWidgets.QFormLayout:
    form = QtWidgets.QFormLayout(parent)
    form.setContentsMargins(10, 6, 8, 10)
    form.setHorizontalSpacing(10)
    form.setVerticalSpacing(6)
    form.setLabelAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
    form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
    return form


def hint(text: str) -> QtWidgets.QLabel:
    """A small grey note under or beside a control."""
    label = QtWidgets.QLabel(text)
    label.setObjectName("hint")
    label.setWordWrap(True)
    return label


def row(*widgets, stretches: tuple[int, ...] | None = None) -> QtWidgets.QWidget:
    """Widgets side by side, as one form field; `stretches` gives each one's
    share of spare width (default: equal)."""
    box = QtWidgets.QWidget()
    layout = QtWidgets.QHBoxLayout(box)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(6)
    for widget, stretch in zip(widgets, stretches or (1,) * len(widgets)):
        layout.addWidget(widget, stretch)
    return box


class CheckedLineEdit(QtWidgets.QLineEdit):
    """A line edit checked as it is typed: `check(text)` returns None when
    the text is acceptable (empty is always acceptable: the option is not
    given) or the reason it is not, shown as the field's tooltip with a red
    frame. `valid` says whether the current text passes."""

    def __init__(self, placeholder: str = "", check: Callable[[str], str | None] | None = None, parent=None):
        super().__init__(parent)
        self.setPlaceholderText(placeholder)
        self.setClearButtonEnabled(True)
        self._check = check
        self._help = ""
        self.textChanged.connect(self._revalidate)

    def set_check(self, check: Callable[[str], str | None] | None) -> None:
        self._check = check
        self._revalidate(self.text())

    def set_help(self, text: str) -> None:
        self._help = text
        self._revalidate(self.text())

    @property
    def valid(self) -> bool:
        return self.property("invalid") is not True

    def _revalidate(self, text: str) -> None:
        problem = None
        if text.strip() and self._check is not None:
            try:
                problem = self._check(text.strip())
            except ValueError as err:
                problem = str(err)
        self.setProperty("invalid", problem is not None)
        self.setToolTip(problem or self._help)
        self.style().unpolish(self)
        self.style().polish(self)


def grouped_combo(groups: list[tuple[str, list[tuple[str, str]]]]) -> QtWidgets.QComboBox:
    """A combo box of (text, data) items under non-selectable group
    headings."""
    combo = QtWidgets.QComboBox()
    model = QtGui.QStandardItemModel(combo)
    heading_font = QtGui.QFont()
    heading_font.setBold(True)
    for heading, items in groups:
        head = QtGui.QStandardItem(heading)
        head.setFlags(QtCore.Qt.NoItemFlags)
        head.setFont(heading_font)
        model.appendRow(head)
        for text, data in items:
            item = QtGui.QStandardItem("    " + text)
            item.setData(data, QtCore.Qt.UserRole)
            model.appendRow(item)
    combo.setModel(model)
    return combo


def select_data(combo: QtWidgets.QComboBox, data) -> None:
    index = combo.findData(data, QtCore.Qt.UserRole)
    if index >= 0:
        combo.setCurrentIndex(index)


class CommandLine(QtWidgets.QPlainTextEdit):
    """A command on one line, shown from its start, read-only: the scroll bar
    slides along it, the text can be selected, and hovering shows it whole."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("command")
        self.setReadOnly(True)
        self.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setTabChangesFocus(True)
        font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        font.setPointSizeF(max(8.0, font.pointSizeF() - 1))
        self.setFont(font)
        line = QtGui.QFontMetrics(font).lineSpacing()
        self.setFixedHeight(line + self.horizontalScrollBar().sizeHint().height() + 2 * self.frameWidth() + 6)

    def set_text(self, text: str) -> None:
        self.setPlainText(text)
        self.horizontalScrollBar().setValue(0)
        self.setToolTip(f"<p>{html.escape(text)}</p>")  # rich text: the tooltip wraps it

    def text(self) -> str:
        return self.toPlainText()
