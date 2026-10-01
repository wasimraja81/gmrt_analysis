"""The Qt inspection window, driven headlessly (Qt's offscreen platform)."""

import os
import time
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")
# matplotlib's Qt backend reads a Qt attribute that Qt 6 marks deprecated; not from this code.
pytestmark = pytest.mark.filterwarnings("ignore:Enum value 'Qt.*AA_UseHighDpiPixmaps' is marked as deprecated:DeprecationWarning")
from PySide6 import QtWidgets  # noqa: E402

from conftest import make_scratch_dir  # noqa: E402
from data_io.antenna_table import read_antenna_table, read_array_earth_location  # noqa: E402
from data_io.row_index import default_row_index_path, load_row_index  # noqa: E402
from data_io.row_selection import select_rows  # noqa: E402
from data_io.source_table import read_source_table  # noqa: E402
from test_cli_visplot_output import _make_synthetic_file  # noqa: E402
from visplot.plot_spec import PlotSpec  # noqa: E402
from visplot.qt_inspector import InspectorWindow  # noqa: E402
from visplot.quantities import context_from_source_table  # noqa: E402
from visplot.xy_figure import XYFigure  # noqa: E402
from visplot.xy_session import XYSource  # noqa: E402


def _window(name, plot):
    scratch = make_scratch_dir(name)
    path = _make_synthetic_file(scratch / "obs.fits")
    index = load_row_index(default_row_index_path(path))
    selection = select_rows(index)
    ctx = context_from_source_table(
        float(index.jd.min()), read_source_table(path), read_array_earth_location(path), "UNCALIB",
        index.stokes_labels, antenna_names={a.station_number: a.name for a in read_antenna_table(path)},
    )
    source = XYSource(path, index, selection.row_indices, None, ctx, chunk_bytes=64 * 1024)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    figure = XYFigure(plot, ctx)
    window = InspectorWindow(source, [("plot", figure)])
    window.resize(900, 700)
    window.show()
    return app, window, scratch


def _wait(app, window, timeout=30.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.processEvents()
        if window.idle() and not window.to_draw and window.extents is not None:
            return
        time.sleep(0.02)
    raise AssertionError("the window did not finish its jobs")


def test_window_draws_every_sample_and_reports_it():
    plot = PlotSpec(y="amp", x="freq_mhz", name="amp-vs-freq_mhz")
    app, window, _ = _window("qt_draw", plot)
    _wait(app, window)
    panel = window.panels[plot]
    assert panel.drawn and panel.grid.n_samples == 40 * 4 * 2 - 1  # one sample flagged
    assert panel.figure.status.get_text() == "319 samples from 40 rows"
    window.close()


def test_zoom_redraws_the_view_over_the_new_limits():
    plot = PlotSpec(y="amp", x="freq_mhz", name="amp-vs-freq_mhz")
    app, window, _ = _window("qt_zoom", plot)
    _wait(app, window)
    panel = window.panels[plot]
    panel.figure.ax.set_xlim(400.5, 402.5)  # channels 1 and 2 (401, 402 MHz)
    panel.figure.ax.set_ylim(0.0, 50.0)
    time.sleep(0.4)
    app.processEvents()
    time.sleep(0.4)
    _wait(app, window)
    assert panel.grid.x_extent == (400.5, 402.5)
    assert panel.grid.n_samples == 40 * 2 * 2  # rows x the two channels x Stokes
    window.close()


def test_locate_lists_the_samples_in_a_box_and_saves_them():
    plot = PlotSpec(y="amp", x="freq_mhz", name="amp-vs-freq_mhz")
    app, window, scratch = _window("qt_locate", plot)
    _wait(app, window)
    panel = window.panels[plot]
    press = SimpleNamespace(xdata=400.5, ydata=4.5)  # x 400.5-401.5: channel 1 (401 MHz)
    release = SimpleNamespace(xdata=401.5, ydata=6.5)  # y 4.5-6.5: amp 5, 6, i.e. rows 4, 5
    window._locate(panel, press, release)
    _wait(app, window)
    assert window.locate.n_found == 2 * 2  # rows 4, 5 x RR, LL at channel 1
    assert window.locate_table.rowCount() == 4
    assert "4 samples in the box" in window.locate_summary.text()
    path = scratch / "located.csv"
    window.save_locate_csv(str(path))  # all 4 were kept: written from memory
    rows = [line for line in path.read_text().splitlines() if not line.startswith("#")]
    assert rows[0].startswith("baseline,ant1,ant2,time_utc,jd_recorded,channel,freq_mhz,stokes")
    assert len(rows) == 1 + 4
    assert rows[1].split(",")[0] in {"C00:01-C01:02", "C00:01-C02:03", "C01:02-C02:03"}
    assert "# total: 4 samples in the box, all listed above" in path.read_text()
    window.close()


def test_saving_more_located_samples_than_the_table_keeps_reads_again_and_writes_all(monkeypatch):
    import visplot.qt_inspector as qi

    monkeypatch.setattr(qi, "LOCATE_LIMIT", 3)
    plot = PlotSpec(y="amp", x="freq_mhz", name="amp-vs-freq_mhz")
    app, window, scratch = _window("qt_locate_all", plot)
    _wait(app, window)
    panel = window.panels[plot]
    window._locate(panel, SimpleNamespace(xdata=399.5, ydata=0.0), SimpleNamespace(xdata=403.5, ydata=100.0))
    _wait(app, window)
    assert window.locate.n_found == 319 and window.locate_table.rowCount() == 3  # the table keeps 3
    path = scratch / "all.csv"
    window.save_locate_csv(str(path))
    _wait(app, window)
    rows = [line for line in path.read_text().splitlines() if not line.startswith("#")]
    assert len(rows) == 1 + 319  # header + every located sample
    assert not list(scratch.glob("*.partial"))
    window.close()


def test_export_re_reads_the_view_at_the_chosen_dpi():
    plot = PlotSpec(y="amp", x="freq_mhz", name="amp-vs-freq_mhz")
    app, window, scratch = _window("qt_export", plot)
    _wait(app, window)
    path = scratch / "exported.png"
    window.export(plot, str(path), dpi=200)
    _wait(app, window)
    assert path.exists() and path.stat().st_size > 0
    from matplotlib.image import imread

    height, width = imread(path).shape[:2]
    fig_w, fig_h = window.panels[plot].figure.fig.get_size_inches()
    assert (width, height) == (round(fig_w * 200), round(fig_h * 200))
    window.close()


def _drag(canvas, ax, start, end):
    """A left-button drag from `start` to `end` (data coordinates), as a user makes it."""
    from matplotlib.backend_bases import MouseEvent

    (x0, y0), (x1, y1) = ax.transData.transform([start, end])
    MouseEvent("button_press_event", canvas, x0, y0, button=1)._process()
    MouseEvent("motion_notify_event", canvas, (x0 + x1) / 2, (y0 + y1) / 2, button=1)._process()
    MouseEvent("motion_notify_event", canvas, x1, y1, button=1)._process()
    MouseEvent("button_release_event", canvas, x1, y1, button=1)._process()


def test_locate_through_the_toolbar_and_a_mouse_drag_after_zooming():
    plot = PlotSpec(y="amp", x="freq_mhz", name="amp-vs-freq_mhz")
    app, window, _ = _window("qt_locate_ui", plot)
    _wait(app, window)
    panel = window.panels[plot]
    ax = panel.figure.ax
    limits = (ax.get_xlim(), ax.get_ylim())

    panel.toolbar.zoom()  # zoom mode left on, as after zooming
    assert panel.toolbar.mode.name == "ZOOM"
    assert not window.save_locate_button.isEnabled()
    panel.locate_action.setChecked(True)  # the Locate button
    assert panel.toolbar.mode.name == "NONE"  # zoom turned off, so the drag reaches Locate

    _drag(panel.canvas, ax, (400.5, 4.5), (401.5, 6.5))
    _wait(app, window)
    assert (ax.get_xlim(), ax.get_ylim()) == limits  # the drag did not zoom
    assert window.locate is not None and window.locate.n_found == 4
    assert window.locate_table.rowCount() == 4
    assert window.save_locate_button.isEnabled()
    assert window.statusBar().currentMessage() == "located 4 samples; listed in the table below"

    panel.toolbar.pan()  # turning pan on turns Locate off
    app.processEvents()
    time.sleep(0.2)
    app.processEvents()
    assert not panel.locate_action.isChecked()
    window.close()


def test_equal_aspect_toggle_redraws_with_equal_scaling():
    plot = PlotSpec(y="amp", x="freq_mhz", name="amp-vs-freq_mhz")
    app, window, _ = _window("qt_aspect", plot)
    _wait(app, window)
    panel = window.panels[plot]
    ax = panel.figure.ax
    assert not panel.aspect_box.isChecked()
    panel.aspect_box.click()  # through the toolbar's check box
    _wait(app, window)
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    bbox = ax.get_window_extent()
    assert (x1 - x0) / bbox.width == pytest.approx((y1 - y0) / bbox.height, rel=1e-2)
    assert panel.drawn and (panel.grid.x_extent, panel.grid.y_extent) == ((x0, x1), (y0, y1))
    window.close()


def test_equal_aspect_off_returns_to_the_datas_range_and_on_widens_again():
    plot = PlotSpec(y="v", x="u", x_unit="m", y_unit="m", name="v-vs-u")  # auto: equal scale
    app, window, _ = _window("qt_aspect_off", plot)
    _wait(app, window)
    panel = window.panels[plot]
    ax = panel.figure.ax

    def view():
        return [*ax.get_xlim(), *ax.get_ylim()]

    data_view = [v for lim in window.extents[plot] for v in lim]
    assert panel.aspect_box.isChecked()
    widened = view()
    assert widened != pytest.approx(data_view)

    panel.aspect_box.click()  # off
    _wait(app, window)
    assert view() == pytest.approx(data_view)
    assert [*panel.grid.x_extent, *panel.grid.y_extent] == pytest.approx(data_view)

    panel.aspect_box.click()  # on again
    _wait(app, window)
    assert view() == pytest.approx(widened)
    window.close()


def test_a_resized_window_redraws_the_view_at_its_new_pixel_size():
    plot = PlotSpec(y="amp", x="freq_mhz", name="amp-vs-freq_mhz")
    app, window, _ = _window("qt_resize", plot)
    _wait(app, window)
    panel = window.panels[plot]
    before = (panel.grid.height, panel.grid.width)
    window.resize(1300, 1000)
    end = time.monotonic() + 30
    while time.monotonic() < end and (panel.grid.height, panel.grid.width) == before:
        app.processEvents()
        time.sleep(0.02)
    _wait(app, window)
    assert (panel.grid.height, panel.grid.width) == window._pixel_shape(panel) != before
    assert panel.drawn and panel.grid.n_samples == 40 * 4 * 2 - 1  # the same samples, re-binned
    window.close()
