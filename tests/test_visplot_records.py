"""Provenance records (T31): every command-line run records the request that
reproduces it, its inputs, its outputs and how it ended."""

import json
import shlex
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from conftest import make_scratch_dir
from cli.run_visplot import main
from data_io.row_index import default_row_index_path
from test_cli_visplot_output import _make_synthetic_file
from visplot.records import PlotRecord, action_request
from visplot.request import PlotRequest
from visplot.run import check_request


def _records(folder: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((folder / "provenance" / "visplot").glob("*.json"))]


def _command_request(record: dict) -> PlotRequest:
    """The request the record's command line runs."""
    return PlotRequest.from_argv(shlex.split(record["details"]["command"])[1:])


def test_a_record_holds_the_request_its_command_inputs_outputs_and_log():
    scratch = make_scratch_dir("records_plot_record")
    path = _make_synthetic_file(scratch / "obs.fits")
    request = PlotRequest(str(path), "amp-vs-freq", provenance_dir=str(scratch / "runs"))
    record = PlotRecord(request, "plot", session_id="S1", details={"note": 1})
    assert json.loads(record.path.read_text())["outcome"]["status"] == "started"
    record.add_output(scratch / "out.png")
    record.finish()

    (saved,) = _records(scratch / "runs")
    assert saved["run_id"] == record.run_id and saved["outcome"]["status"] == "success"
    assert saved["parameters"] == request.as_dict()
    assert _command_request(saved) == request
    assert saved["details"]["action"] == "plot" and saved["details"]["session"] == "S1"
    assert saved["details"]["note"] == 1
    assert {"numpy", "astropy", "matplotlib"} <= set(saved["details"]["packages"])
    assert saved["host"] and saved["git"]["commit"]
    assert [i["path"] for i in saved["inputs"]] == [str(path.resolve()), str(default_row_index_path(path).resolve())]
    assert saved["outputs"] == [str((scratch / "out.png").resolve())]
    log = (scratch / "runs" / "logs" / "visplot" / f"{record.run_id}.log").read_text()
    assert request.command_line() in log


def test_a_failed_run_is_recorded_as_failed_with_its_reason():
    scratch = make_scratch_dir("records_failed")
    path = _make_synthetic_file(scratch / "obs.fits")
    record = PlotRecord(PlotRequest(str(path), "amp-vs-freq", provenance_dir=str(scratch / "runs")), "plot")
    record.finish("no rows selected")
    record.finish()  # a second finish changes nothing
    (saved,) = _records(scratch / "runs")
    assert saved["outcome"]["status"] == "failed"
    assert saved["outcome"]["error_message"] == "no rows selected"


def test_a_window_action_narrows_the_request_to_its_box_or_view():
    request = PlotRequest("x.fits", "amp-vs-freq")
    located, full = action_request(request, 1, "locate", box=((2.0, 1.0), (-0.5, 3.0)), csv_path="a.csv")
    assert full and located.locate == "1.0:2.0,-0.5:3.0" and located.locate_csv == str(Path("a.csv").resolve())
    assert check_request(located).locate_box == ((1.0, 2.0), (-0.5, 3.0))
    assert PlotRequest.from_argv(located.to_argv()) == located

    exported, full = action_request(request, 1, "export", view=((0.1, 0.2), (-1.0, 1.0)))
    assert full and exported.x_range == "0.1:0.2" and exported.y_range == "-1.0:1.0"
    assert PlotRequest.from_argv(exported.to_argv()) == exported

    several = PlotRequest("x.fits", "az-el-range")
    unchanged, full = action_request(several, 2, "export", view=((0.0, 1.0), (0.0, 1.0)))
    assert unchanged == several and not full


def test_a_saving_run_records_its_outputs_and_its_command_reproduces_it(capsys):
    scratch = make_scratch_dir("records_cli_save")
    path = _make_synthetic_file(scratch / "obs.fits")
    args = [str(path), "--plots", "amp-vs-freq", "--output-dir", str(scratch / "out"), "--no-highres-pdf",
            "--provenance-dir", str(scratch / "runs")]
    assert main(["visplot", *args]) == 0
    plt.close("all")

    (record,) = _records(scratch / "runs")
    assert record["outcome"]["status"] == "success" and record["details"]["action"] == "save"
    assert sorted(record["outputs"]) == sorted(str(p) for p in (scratch / "out").iterdir())
    assert _command_request(record) == PlotRequest.from_argv(args)
    assert f"provenance record: run {record['run_id']}" in capsys.readouterr().out
    log = (scratch / "runs" / "logs" / "visplot" / f"{record['run_id']}.log").read_text()
    assert "selected " in log and f"saved {scratch / 'out'}" in log


def test_a_run_that_cannot_start_is_recorded_as_failed(capsys):
    scratch = make_scratch_dir("records_cli_failed")
    path = _make_synthetic_file(scratch / "obs.fits")
    with pytest.raises(SystemExit):
        main(["visplot", str(path), "--plots", "amp-vs-freq", "--locate", "0:1,0:1",
              "--provenance-dir", str(scratch / "runs")])
    (record,) = _records(scratch / "runs")
    assert record["outcome"]["status"] == "failed"
    assert "--locate needs --locate-csv" in record["outcome"]["error_message"]


def test_located_samples_name_their_command_and_record(capsys):
    scratch = make_scratch_dir("records_cli_locate")
    path = _make_synthetic_file(scratch / "obs.fits")
    csv_path = scratch / "located.csv"
    assert main(["visplot", str(path), "--plots", "amp-vs-freq_mhz", "--locate", "400.5:401.5,4.5:6.5",
                 "--locate-csv", str(csv_path), "--provenance-dir", str(scratch / "runs")]) == 0

    (record,) = _records(scratch / "runs")
    assert record["details"]["action"] == "locate"
    assert record["outputs"] == [str(csv_path.resolve())]
    header = [line for line in csv_path.read_text().splitlines() if line.startswith("#")]
    assert f"# command: {record['details']['command']}" in header
    assert f"# provenance record: run {record['run_id']}, " in "\n".join(header)
    assert any(line.startswith("# written: 20") for line in header)
