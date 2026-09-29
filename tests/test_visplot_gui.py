"""The GUI, driven headlessly (Qt's offscreen platform): its form holds
exactly the command line's options, and its Plot runs the command line's
own code."""

import json
import os
import shlex
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib.pyplot as plt
import numpy as np
import pytest

pytest.importorskip("PySide6")
pytestmark = pytest.mark.filterwarnings("ignore:Enum value 'Qt.*AA_UseHighDpiPixmaps' is marked as deprecated:DeprecationWarning")
from PySide6 import QtWidgets  # noqa: E402

from conftest import make_scratch_dir  # noqa: E402
from test_cli_visplot_output import _make_synthetic_file  # noqa: E402
from visplot.gui.form import ACTION_OPTIONS, RequestForm  # noqa: E402
from visplot.gui.main_window import VisplotWindow  # noqa: E402
from visplot.gui.widgets import select_data  # noqa: E402
from visplot.records import action_request  # noqa: E402
from visplot.request import PlotRequest, request_option_names  # noqa: E402
from visplot.run import prepare, save_outputs  # noqa: E402
from visplot.xy_session import plot_grids  # noqa: E402


def _app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _window(name):
    app = _app()
    path = _make_synthetic_file(make_scratch_dir(name) / "obs.fits")
    window = VisplotWindow(str(path))
    window.resize(1200, 800)
    window.show()
    _settle(app, window, lambda: window.opened is not None)
    return app, window, path


def _settle(app, window, condition, timeout=60.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.processEvents()
        if condition() and not window._tasks and not window.count_timer.isActive():
            return
        time.sleep(0.02)
    raise AssertionError("the window did not settle")


def test_every_request_option_is_a_form_field_or_a_named_gui_action():
    _app()
    form = RequestForm()
    options = set(request_option_names())
    assert set(form.fields).isdisjoint(ACTION_OPTIONS)
    assert set(form.fields) | set(ACTION_OPTIONS) == options  # an option missing from the GUI fails here
    form.deleteLater()


def test_the_form_shows_a_request_and_gives_it_back_unchanged():
    app, window, path = _window("gui_round_trip")
    request = PlotRequest(
        str(path), "phase-vs-freq", sources="3C48", correlation_type="both", stokes="LL", channels="0:2",
        antennas="1:2", time_range="0:1", uvdist_range="0:5km", ha_range="-3:3", every_nth=2, colorize_by="stokes",
        show_flagged=True, mirror=False, x_unit="GHz", y_unit="rad", x_range="0.3:0.5", y_range_mode="percentile",
        range_percentiles="1:99", y_scale="symlog", aspect="free", scale_linear_width=0.5, point_size=2.5,
        color="k", time_zone="Asia/Kolkata", cache_dir=str(path.parent / "cache"), threads=2,
    )
    window.form.load(request)
    assert window.form.request() == request
    window.close()


def test_a_field_that_fails_its_check_stops_plot_and_says_why():
    app, window, _ = _window("gui_invalid_field")
    window.form.channels.setText("not-a-channel")
    _settle(app, window, lambda: True)
    assert not window.plot_button.isEnabled()
    assert "--channels" in window.problem.text()
    window.close()


def test_plot_runs_the_command_lines_code_and_draws_the_same_pixels():
    app, window, path = _window("gui_plot")
    select_data(window.form.y.quantity, "amp")
    select_data(window.form.x.quantity, "freq")
    window.form.sources.set("3C286")
    _settle(app, window, lambda: window.plot_button.isEnabled())
    request = window.form.request()
    window._plot()
    _settle(app, window, lambda: hasattr(window.tabs.currentWidget(), "panel")
            and window.tabs.currentWidget().panel.idle() and not window.tabs.currentWidget().panel.to_draw)
    tab = window.tabs.currentWidget()
    assert tab.request == request
    assert tab.run.request == request

    # The command line's run of the same request selects the same rows and plots,
    # and binned over the tab's view it gives the tab's pixels.
    cli_run = prepare(PlotRequest.from_argv(request.to_argv()))
    np.testing.assert_array_equal(cli_run.source.row_indices, tab.run.source.row_indices)
    assert cli_run.xy_plots == tab.run.xy_plots
    (plot,) = tab.run.xy_plots
    grid = tab.panel.panels[plot].grid
    cli_grids, _ = plot_grids(cli_run.source, [plot], {plot: (grid.x_extent, grid.y_extent)},
                              {plot: (grid.height, grid.width)})
    np.testing.assert_array_equal(cli_grids[plot].layers, grid.layers)
    assert grid.n_samples > 0
    window.close()


def test_the_command_line_with_no_arguments_or_a_file_alone_opens_the_gui(monkeypatch):
    import visplot.gui.main_window as main_window
    from cli.run_visplot import main

    opened = []
    monkeypatch.setattr(main_window, "run_gui", lambda path=None: opened.append(path) or 0)
    assert main(["visplot"]) == 0
    assert main(["visplot", "obs.fits"]) == 0
    assert opened == [None, "obs.fits"]


def test_an_emptied_field_with_a_default_means_the_default():
    app, window, _ = _window("gui_default_fields")
    window.form.percentiles.setText("5:95")
    window.form.percentiles.setText("")
    _settle(app, window, lambda: True)
    assert window.form.request().range_percentiles == "0.1:99.9"
    assert window.plot_button.isEnabled()
    window.close()


def test_text_that_is_not_a_number_stops_plot_without_breaking_the_form():
    app, window, _ = _window("gui_not_a_number")
    field = window.form.fields["every_nth"].widget
    field.setText("abc")
    _settle(app, window, lambda: True)
    assert not window.plot_button.isEnabled() and "--every-nth" in window.problem.text()
    field.setText("")
    _settle(app, window, lambda: True)
    assert window.plot_button.isEnabled()
    window.close()


def test_the_command_box_holds_the_whole_command_from_its_start():
    app, window, _ = _window("gui_command_box")
    command = window.form_command
    assert command.text() == window.form.request().command_line()
    assert command.text().startswith("bin/visplot.sh ")
    assert command.horizontalScrollBar().value() == 0
    window.close()


def _plot_amp_vs_freq(app, window):
    select_data(window.form.y.quantity, "amp")
    select_data(window.form.x.quantity, "freq")
    _settle(app, window, lambda: window.plot_button.isEnabled())
    window._plot()
    _settle(app, window, lambda: hasattr(window.tabs.currentWidget(), "panel")
            and window.tabs.currentWidget().panel.idle() and not window.tabs.currentWidget().panel.to_draw)
    return window.tabs.currentWidget()


def _records(folder, stage="visplot"):
    return [json.loads(p.read_text()) for p in sorted((folder / "provenance" / stage).glob("*.json"))]


def test_plot_and_export_are_recorded_with_the_session_whose_log_keeps_every_message():
    app, window, path = _window("gui_records")
    runs = path.parent / "runs"
    window.form.provenance_edit.setText(str(runs))
    tab = _plot_amp_vs_freq(app, window)

    (plot_record,) = _records(runs)
    assert plot_record["run_id"] == tab.record.run_id and plot_record["outcome"]["status"] == "success"
    assert plot_record["details"]["action"] == "plot"
    assert plot_record["details"]["session"] == window.session.session_id
    assert plot_record["parameters"] == tab.request.as_dict()
    assert window.history.item(0, 1).text() == tab.record.run_id

    (plot,) = tab.run.xy_plots
    ax = tab.panel.panels[plot].figure.ax
    view = (tuple(ax.get_xlim()), tuple(ax.get_ylim()))
    exported = path.parent / "export.png"
    tab.panel.export(plot, str(exported), 100)
    _settle(app, window, lambda: tab.panel.idle())
    export_record = next(r for r in _records(runs) if r["details"]["action"] == "export")
    assert exported.exists() and export_record["outputs"] == [str(exported.resolve())]
    assert export_record["details"]["from_run"] == tab.record.run_id
    assert export_record["details"]["session"] == window.session.session_id
    figure_size_in = tuple(tab.panel.panels[plot].figure.fig.get_size_inches())
    assert figure_size_in != (8.0, 6.0)  # the window's size, so the default figure size would not repeat it
    assert export_record["details"]["dpi"] == 100
    assert export_record["details"]["figure_size_in"] == list(figure_size_in)
    narrowed, _ = action_request(tab.request, 1, "export", view=view, dpi=100, figure_size_in=figure_size_in)
    assert export_record["parameters"] == narrowed.as_dict()

    # The export's command, run with --output-dir, saves the export's image, pixel for pixel.
    repeat = PlotRequest.from_argv(shlex.split(export_record["details"]["command"])[1:])
    saved = save_outputs(prepare(repeat.replace(output_dir=str(path.parent / "repeat"), no_highres_pdf=True)))
    (png,) = [p for p in saved if p.suffix == ".png"]
    np.testing.assert_array_equal(plt.imread(png), plt.imread(exported))

    session_log = window.session.log_path
    window.close()
    (session,) = [r for r in _records(session_log.parents[2], "visplot_session")
                  if r["run_id"] == window.session.session_id]
    assert session["outcome"]["status"] == "success"
    log = session_log.read_text()
    assert f"plot {tab.record.run_id}: " in log and f"export {export_record['run_id']}: saved " in log


def test_a_plot_whose_record_cannot_be_written_is_not_made():
    app, window, path = _window("gui_record_fails")
    blocker = path.parent / "a_file"
    blocker.write_text("")
    window.form.provenance_edit.setText(str(blocker))
    _settle(app, window, lambda: window.plot_button.isEnabled())
    n_tabs = window.tabs.count()
    window._plot()
    _settle(app, window, lambda: True)
    assert window.tabs.count() == n_tabs
    texts = [window.messages.item(i).text() for i in range(window.messages.count())]
    assert any("the provenance record could not be written in" in t and "nothing was run" in t for t in texts)
    window.close()


def test_save_writes_what_the_command_lines_save_writes_with_its_record():
    app, window, path = _window("gui_save")
    runs = path.parent / "runs"
    window.form.provenance_edit.setText(str(runs))
    # at 100 dpi: the streamed plot at --figure-size 5,4; the antenna layout at its own 11 x 11 in
    for plots, name, pixels in ((("amp", "freq"), "amp-vs-freq", (400, 500)),
                                ("antenna-layout", "antenna-layout", (1100, 1100))):
        if isinstance(plots, tuple):
            select_data(window.form.kind, "generic")
            select_data(window.form.y.quantity, plots[0])
            select_data(window.form.x.quantity, plots[1])
        else:
            select_data(window.form.kind, plots)
        _settle(app, window, lambda: window.plot_button.isEnabled() and window.save_action.isEnabled())
        gui_out, cli_out = path.parent / f"gui_{name}", path.parent / f"cli_{name}"
        request = window.form.request(output_dir=str(gui_out), dpi=100, figure_size="5,4", no_highres_pdf=True)
        window.save(request)
        _settle(app, window, lambda: gui_out.exists() and not window._tasks)

        (record,) = [r for r in _records(runs) if r["details"]["action"] == "save" and
                     r["parameters"]["output_dir"] == str(gui_out)]
        assert record["outcome"]["status"] == "success"
        assert record["details"]["session"] == window.session.session_id
        assert sorted(record["outputs"]) == sorted(str(p) for p in gui_out.iterdir())
        assert window.history.item(window.history.rowCount() - 1, 1).text() == record["run_id"]

        # The command line's run of the same request writes the same files, pixel for pixel.
        from cli.run_visplot import main

        assert main(["visplot", *request.replace(output_dir=str(cli_out)).to_argv()]) == 0
        assert sorted(p.name for p in gui_out.iterdir()) == sorted(p.name for p in cli_out.iterdir())
        png = f"visplot_{name}.png"
        assert plt.imread(gui_out / png).shape[:2] == pixels
        np.testing.assert_array_equal(plt.imread(gui_out / png), plt.imread(cli_out / png))
    window.close()


def test_the_save_dialog_builds_the_forms_request_with_the_save_options():
    from visplot.gui.save_dialog import SaveDialog

    app, window, path = _window("gui_save_dialog")
    dialog = SaveDialog(window.form.request)
    save_button = dialog.buttons.button(QtWidgets.QDialogButtonBox.Save)
    assert not save_button.isEnabled() and "choose a folder" in dialog.problem.text()
    dialog.folder.setText(str(path.parent / "out"))
    dialog.prefix.setText("run1")
    dialog.dpi.setValue(300)
    dialog.width.setValue(10.5)
    dialog.height.setValue(7.25)
    dialog.highres.setChecked(False)
    expected = window.form.request(output_dir=str(path.parent / "out"), output_prefix="run1", dpi=300,
                                   figure_size="10.5,7.25", no_highres_pdf=True)
    assert dialog.request() == expected and save_button.isEnabled()
    assert dialog.command.text() == expected.command_line()
    assert "at 600 dpi" in dialog.highres.text()

    again = SaveDialog(window.form.request, dialog.actions())  # the next save offers the same options
    assert again.request() == expected
    window.close()


def test_messages_keep_the_newest_up_to_the_limit(monkeypatch):
    import visplot.gui.main_window as main_window

    monkeypatch.setattr(main_window, "MAX_MESSAGES", 5)
    app, window, _ = _window("gui_message_cap")
    for i in range(12):
        window.report("info", f"message {i}")
    window._poll()
    texts = [window.messages.item(i).text() for i in range(window.messages.count())]
    assert len(texts) == 5 and texts[-1].endswith("message 11")
    window.close()
