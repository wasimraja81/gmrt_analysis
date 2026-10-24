"""The GUI's look: Qt's Fusion style with matched light and dark palettes,
and one style sheet for section headers, the primary button, hints, the
command line and fields that fail their check. The theme is a window
preference: it does not change a plot's output (T33 brings themed plots, as
a request option)."""

from __future__ import annotations

from PySide6 import QtGui, QtWidgets

THEMES = ("light", "dark")

_COLORS = {
    "light": dict(window="#f3f4f6", base="#ffffff", alt="#f6f8fa", text="#1f2328", mid="#d0d7de", hint="#6e7781",
                  button="#f6f8fa", accent="#2f6fdb", accent_text="#ffffff", header="#e7eaee", error="#c93c37",
                  disabled="#a0a7b0"),
    "dark": dict(window="#24272c", base="#1b1d21", alt="#22252a", text="#e6e8eb", mid="#3b4048", hint="#9aa3ad",
                 button="#2d3137", accent="#4c8ef7", accent_text="#ffffff", header="#2f333a", error="#f07470",
                 disabled="#6b727c"),
}


def palette(theme: str) -> QtGui.QPalette:
    c = _COLORS[theme]
    p = QtGui.QPalette()
    role = QtGui.QPalette.ColorRole
    for r, key in ((role.Window, "window"), (role.Base, "base"), (role.AlternateBase, "alt"),
                   (role.Text, "text"), (role.WindowText, "text"), (role.ButtonText, "text"),
                   (role.Button, "button"), (role.Highlight, "accent"), (role.HighlightedText, "accent_text"),
                   (role.ToolTipBase, "base"), (role.ToolTipText, "text"), (role.PlaceholderText, "hint"),
                   (role.Mid, "mid"), (role.Link, "accent")):
        p.setColor(r, QtGui.QColor(c[key]))
    for r in (role.Text, role.WindowText, role.ButtonText):
        p.setColor(QtGui.QPalette.Disabled, r, QtGui.QColor(c["disabled"]))
    return p


def style_sheet(theme: str) -> str:
    c = _COLORS[theme]
    return f"""
    QToolButton#sectionHeader {{
        font-weight: 600; text-align: left; padding: 6px 8px; border: none;
        background: {c['header']}; border-top: 1px solid {c['mid']};
    }}
    QLabel#hint {{ color: {c['hint']}; font-size: 11px; }}
    QLabel#counts {{ font-size: 12px; }}
    QLabel#problem {{ color: {c['error']}; font-size: 11px; }}
    QPushButton#primary {{
        background: {c['accent']}; color: {c['accent_text']}; font-weight: 600;
        border: none; border-radius: 4px; padding: 7px 26px;
    }}
    QPushButton#primary:disabled {{ background: {c['mid']}; color: {c['disabled']}; }}
    QLineEdit[invalid="true"] {{ border: 1px solid {c['error']}; }}
    QPlainTextEdit#command {{ background: {c['alt']}; border: 1px solid {c['mid']}; }}
    QPlainTextEdit#note {{ background: {c['alt']}; border: 1px solid {c['mid']}; color: {c['hint']}; }}
    QProgressBar#fileProgress {{ border: none; border-radius: 3px; background: {c['mid']}; }}
    QProgressBar#fileProgress::chunk {{ border-radius: 3px; background: {c['accent']}; }}
    QLabel#welcomeTitle {{ font-size: 22px; font-weight: 600; }}
    QLabel#welcomeText {{ color: {c['hint']}; font-size: 13px; }}
    QScrollArea#sidebar {{ border: none; }}
    """


def apply_theme(app: QtWidgets.QApplication, theme: str) -> None:
    app.setStyle("Fusion")
    app.setPalette(palette(theme))
    app.setStyleSheet(style_sheet(theme))
