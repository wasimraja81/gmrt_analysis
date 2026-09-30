import colorsys

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_rgba

from data_io.source_table import Source
from visplot.plot_panel import _PALETTE_LARGE, _PALETTE_SMALL, FLAGGED_COLOR
from visplot.plot_spec import PlotSpec
from visplot.plot_theme import PLOT_THEMES, contrast_ratio, plot_theme, readable
from visplot.quantities import QuantityContext
from visplot.source_listing import source_listing
from visplot.stream import FLAGGED_LAYER, GridReducer
from visplot.xy_figure import XYFigure

LIGHT, DARK = PLOT_THEMES["light"], PLOT_THEMES["dark"]
CTX = QuantityContext(time_reference_jd=2459421.0, bunit="UNCALIB", stokes_labels=("RR", "LL"))
EVERY_COLOR = [*_PALETTE_SMALL, *_PALETTE_LARGE, FLAGGED_COLOR, "tab:blue", "tab:orange", "tab:green", "tab:red",
               "k", "0.3", "white", "blue", "#0000ff", "navy", "yellow"]


def test_the_contrast_ratio_is_wcags():
    assert contrast_ratio("white", "black") == pytest.approx(21.0)
    assert contrast_ratio("tab:blue", "tab:blue") == pytest.approx(1.0)
    assert contrast_ratio("#777777", "white") == pytest.approx(4.48, abs=0.01)  # just under WCAG's 4.5:1 for text


def test_the_light_theme_keeps_every_color_as_given():
    for color in EVERY_COLOR:
        assert readable(color, LIGHT) == to_rgba(color)


def test_the_dark_theme_makes_every_color_readable_and_keeps_those_already_readable():
    for color in EVERY_COLOR:
        out = readable(color, DARK)
        assert contrast_ratio(out, DARK.axes_face) >= 3.0, color
        if contrast_ratio(color, DARK.axes_face) >= 3.0:
            assert out == to_rgba(color), color


def test_a_color_too_dark_is_mirrored_in_lightness_with_its_hue():
    brown = to_rgba(_PALETTE_SMALL[4])  # tab10's brown, 2.85:1 on the dark axes
    assert contrast_ratio(brown, DARK.axes_face) < 3.0
    out = readable(brown, DARK)
    hue_in, light_in, _ = colorsys.rgb_to_hls(*brown[:3])
    hue_out, light_out, _ = colorsys.rgb_to_hls(*out[:3])
    assert hue_out == pytest.approx(hue_in, abs=1e-6) and light_out == pytest.approx(1 - light_in, abs=1e-6)
    assert readable("k", DARK)[:3] == pytest.approx((1.0, 1.0, 1.0))  # black mirrors to white


def test_an_unknown_theme_is_refused_with_the_choices():
    assert plot_theme("dark") is DARK
    with pytest.raises(ValueError, match="light, dark"):
        plot_theme("sepia")


def _grid(plot):
    """An 8 x 8 grid with an unflagged sample at (1, 1) and a flagged one at
    (6, 6), apart by more than a marker (plots here use the smallest)."""
    grid = GridReducer(plot, (0.0, 1.0), (0.0, 1.0), height=8, width=8)
    grid.layers[1 * 8 + 1] = 1
    grid.layers[6 * 8 + 6] = FLAGGED_LAYER
    grid.n_samples = 2
    return grid


def _pixels(figure: XYFigure) -> np.ndarray:
    canvas = FigureCanvasAgg(figure.fig)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba()).copy()


def test_a_dark_figure_draws_axes_text_panel_and_markers_in_the_dark_colors():
    plot = PlotSpec(y="amp", x="freq_mhz", color="k", show_flagged=True, point_size=0.25)
    figure = XYFigure(plot, CTX, sources=["3C286"], theme="dark")
    figure.show(_grid(plot), display_dpi=100)
    _pixels(figure)  # drawn, so the tick labels exist
    assert figure.fig.get_facecolor() == to_rgba(DARK.figure_face)
    assert figure.ax.get_facecolor() == to_rgba(DARK.axes_face)
    assert figure.ax.title.get_color() == DARK.text and figure.ax.xaxis.label.get_color() == DARK.text
    assert all(t.get_color() == DARK.text for t in figure.ax.get_xticklabels())
    colors = figure.image.get_array()
    unflagged, flagged = colors[1, 1], colors[6, 6]
    assert tuple(unflagged) == tuple(np.round(np.asarray(readable("k", DARK)) * 255).astype(np.uint8))
    assert tuple(unflagged[:3]) == (255, 255, 255)  # black markers mirror to white on dark
    assert tuple(flagged) == tuple(np.round(np.asarray(to_rgba(FLAGGED_COLOR)) * 255).astype(np.uint8))
    assert figure.status.get_children()[0].get_color() == DARK.panel_value
    assert figure.panel.note.get_color() == DARK.warning


def test_the_dark_figure_is_saved_in_its_colors():
    plot = PlotSpec(y="amp", x="freq_mhz", colorize_by="stokes", show_flagged=True, point_size=0.25)
    grid = _grid(plot)
    grid.seen_codes = {0}
    light, dark = XYFigure(plot, CTX, sources=["3C286"]), XYFigure(plot, CTX, sources=["3C286"], theme="dark")
    for figure in (light, dark):
        figure.show(grid, display_dpi=100)
    light_pixels, dark_pixels = _pixels(light), _pixels(dark)
    assert tuple(light_pixels[2, 2, :3]) == (255, 255, 255)
    assert tuple(dark_pixels[2, 2, :3]) == tuple(int(round(v * 255)) for v in to_rgba(DARK.figure_face)[:3])


def test_the_source_listing_draws_its_table_in_the_theme():
    source = Source(id=1, name="3C286", ra_epoch_deg=202.78, dec_epoch_deg=30.51, ra_apparent_deg=202.78,
                    dec_apparent_deg=30.51, epoch_year=2000.0, calcode="", flux_i_jy=(0.0,), flux_q_jy=(),
                    flux_u_jy=(), flux_v_jy=())
    fig = source_listing({1: source}, theme="dark")
    assert fig.get_facecolor() == to_rgba(DARK.figure_face)
    cells = fig.axes[0].tables[0].get_celld().values()
    assert all(c.get_facecolor() == to_rgba(DARK.table_face) and c.get_text().get_color() == DARK.text for c in cells)
    assert fig.axes[0].title.get_color() == DARK.text
