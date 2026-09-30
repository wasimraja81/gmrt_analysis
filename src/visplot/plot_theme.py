"""Plot themes (T33): the colors of a plot's figure, axes, text and panel,
light or dark, chosen by --plot-theme.

The light theme is matplotlib's own default look, the plots as they were
before themes. The dark theme uses the GUI's dark window colors, so a dark
plot matches the dark window; there a marker color with less than
MIN_CONTRAST against the axes' background (WCAG's ratio: 3:1 is its minimum
for graphics) is mirrored in lightness, its hue kept (`readable`), and
lightened further where the mirror still falls short. The light theme keeps
every color as given (the user, 2026-09-30: tab10's orange, olive and cyan,
and light coral, are below 3:1 on white, and stay).
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass

from matplotlib.colors import to_rgba

DEFAULT_PLOT_THEME = "light"
MIN_CONTRAST = 3.0


@dataclass(frozen=True)
class PlotTheme:
    name: str
    figure_face: str
    axes_face: str
    text: str  # title, axis labels, tick labels, ticks and spines
    grid: str  # drawn at alpha 0.3
    reference: str  # dashed reference lines (transit, horizon)
    panel_label: str
    panel_value: str
    panel_rule: str
    warning: str  # the panel's warnings and the plot's caveat note
    box_face: str  # antenna labels' boxes
    box_edge: str
    leader: str  # antenna labels' leader lines
    inset_face: str  # the antenna layout's core inset
    table_face: str  # source listing cells
    table_edge: str
    min_contrast: float | None  # markers below it against axes_face are made readable; None: kept as given


PLOT_THEMES = {
    "light": PlotTheme(
        name="light", figure_face="white", axes_face="white", text="black", grid="#b0b0b0", reference="0.5",
        panel_label="0.42", panel_value="0.08", panel_rule="0.75", warning="darkred", box_face="white",
        box_edge="0.6", leader="0.5", inset_face="aliceblue", table_face="white", table_edge="black",
        min_contrast=None,
    ),
    # the GUI's dark palette (visplot.gui.style): window, base, text, hint, mid, error
    "dark": PlotTheme(
        name="dark", figure_face="#24272c", axes_face="#1b1d21", text="#e6e8eb", grid="#8a929c", reference="0.5",
        panel_label="#9aa3ad", panel_value="#e6e8eb", panel_rule="#3b4048", warning="#f07470", box_face="#2d3137",
        box_edge="#6b727c", leader="#8a929c", inset_face="#1d2733", table_face="#1b1d21", table_edge="#6b727c",
        min_contrast=MIN_CONTRAST,
    ),
}


def plot_theme(name: str) -> PlotTheme:
    try:
        return PLOT_THEMES[name]
    except KeyError:
        raise ValueError(f"no plot theme {name!r}; choose from {', '.join(PLOT_THEMES)}") from None


def color_axes(ax, theme: PlotTheme, face: str | None = None) -> None:
    """`ax`'s background (`face`, default the theme's axes), title, axis
    labels, ticks (those made later by a zoom too) and spines in `theme`."""
    ax.set_facecolor(face or theme.axes_face)
    for text in (ax.title, ax.xaxis.label, ax.yaxis.label):
        text.set_color(theme.text)
    ax.tick_params(which="both", colors=theme.text)
    for spine in ax.spines.values():
        spine.set_edgecolor(theme.text)


def relative_luminance(color) -> float:
    """WCAG 2 relative luminance of a matplotlib color (alpha ignored)."""
    def channel(v: float) -> float:
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4

    r, g, b, _ = to_rgba(color)
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def contrast_ratio(a, b) -> float:
    """WCAG 2 contrast ratio of two colors, 1 to 21."""
    hi, lo = sorted((relative_luminance(a), relative_luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def readable(color, theme: PlotTheme) -> tuple:
    """`color` as RGBA, made readable on `theme`'s axes: unchanged where the
    theme keeps colors as given or the contrast is at least its minimum;
    otherwise mirrored in lightness (HLS), hue and saturation kept, and
    moved further from the background's lightness where the mirror still
    falls short (a color of middle lightness mirrors to itself)."""
    rgba = to_rgba(color)
    if theme.min_contrast is None or contrast_ratio(rgba, theme.axes_face) >= theme.min_contrast:
        return rgba
    hue, lightness, saturation = colorsys.rgb_to_hls(*rgba[:3])
    towards_light = relative_luminance(theme.axes_face) < 0.5
    candidate = 1.0 - lightness
    candidate = max(candidate, lightness) if towards_light else min(candidate, lightness)
    step = 0.01 if towards_light else -0.01
    while True:
        out = (*colorsys.hls_to_rgb(hue, candidate, saturation), rgba[3])
        if contrast_ratio(out, theme.axes_face) >= theme.min_contrast or not 0.0 < candidate < 1.0:
            return out
        candidate = min(1.0, max(0.0, candidate + step))
