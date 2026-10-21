"""Locate by condition (`--locate` with no box, `--where`): the options checked
without a file, the pass on a synthetic file with outliers planted at known
cells, and the run it writes to its session."""

import pytest

from visplot.locate_where import check_locate_request, locate_session_name
from visplot.request import PlotRequest
from visplot.run import RequestError, check_request


def _request(*argv):
    return PlotRequest.from_argv(["obs.fits", *argv])


def test_a_locate_by_conditions_is_a_run_of_its_own():
    checked = check_request(_request("--locate", "--where", "amp(V) > 5", "--where", "el > 20 deg"))
    assert [c.text for c in checked.conditions] == ["amp(V) > 5", "el > 20 deg"] and checked.plot_names == []
    assert check_request(_request("--locate", "--where", "amp(V) > 5", "--listobs")).conditions  # with a listing
    assert " --locate --where 'amp(V) > 5' " in _request("--locate", "--where", "amp(V) > 5").command_line()


@pytest.mark.parametrize("argv, message", [
    (("--where", "amp(V) > 5"), r"--where gives the conditions of --locate: add --locate, with no value"),
    (("--plots", "amp-vs-time", "--locate", "0:1,0:5", "--locate-csv", "x.csv", "--where", "amp(V) > 5"),
     r"--where goes with --locate given no box"),
    (("--locate",), r"give at least one --where"),
    (("--locate", "--where", "el > 20 deg"), r"a Locate run needs a condition on a data value"),
    (("--locate", "--where", "amp(V) > 5", "--stokes", "RR"), r"--stokes: the --where conditions name the products"),
    (("--locate", "--where", "amp(V) > 5", "--plots", "amp-vs-time"), r"--plots: give the plots in a separate run"),
    (("--locate", "--where", "amp(V) > 5", "--locate-csv", "x.csv"), r"--locate-csv: that is the box's CSV"),
    (("--locate", "--where", "amp(V) > 5", "--session", "my run"), r"--session 'my run': a name of letters"),
    (("--listobs", "--session", "s1"), r"--session goes with --locate \(no box\) and --where"),
    (("--listobs", "--flagged", "only"), r"--flagged goes with --locate"),
    (("--locate", "--where", "amp(V) > 5 Jy deg"), r"is not a number with a unit"),
])
def test_what_does_not_make_a_locate_by_conditions_is_refused_before_the_file_is_opened(argv, message):
    with pytest.raises(RequestError, match=message):
        check_request(_request(*argv))


def test_the_session_defaults_to_the_files_name_without_its_extension():
    assert locate_session_name(_request("--locate", "--where", "amp(V) > 5")) == ("obs", False)
    assert locate_session_name(_request("--locate", "--where", "amp(V) > 5", "--session", "cal-rfi.1")) == (
        "cal-rfi.1", True)
    odd = PlotRequest("/data/my obs+2.FITS", None, locate="", where=("amp(V) > 5",))
    assert locate_session_name(odd) == ("my_obs_2", False)
    assert check_locate_request(PlotRequest("obs.fits", "amp-vs-time")) == ()


# --- the pass, on the synthetic file (tests/test_cli_visplot_output.py): 40 rows, RR and LL, 4 channels of
# 400-403 MHz, the real part of both products row + 1 (so V = RR - LL is small), RR of row 0 channel 0 flagged.
# Planted: row 5 channel 2, RR + 100 (|V| about 100); row 12 every channel, RR + 50; row 0 channel 0 (flagged),
# RR + 100. Rows are 10 s apart, in threes, from 18:00:00 recorded on day 0.

import json

import numpy as np

from conftest import TEST_TMP_ROOT, make_scratch_dir
from data_io.found_file import read_log, read_run
from test_cli_visplot_output import _make_synthetic_file
from visplot.run import Stopped, locate_by_conditions, open_file


def _plant(image_data):
    image_data[5, 0, 2, 0, 0] += 100.0
    image_data[12, 0, :, 0, 0] += 50.0
    image_data[0, 0, 0, 0, 0] += 100.0


def _locate(name, *argv, edit=_plant, **kwargs):
    scratch = make_scratch_dir(f"locate_where_{name}")
    path = _make_synthetic_file(scratch / "obs.fits", edit=edit)
    request = PlotRequest.from_argv([str(path), "--locate", *argv, "--provenance-dir", str(scratch / "runs")])
    reports = []
    run_path = locate_by_conditions(request, open_file(path), report=lambda level, text: reports.append(text), **kwargs)
    return read_run(run_path), reports, run_path


def test_a_v_outlier_is_recorded_at_both_products_it_is_made_from():
    run, reports, path = _locate("v", "--where", "amp(V) > 10")
    cells, rows = run.cells, run.rows
    # row 5 channel 2: one visibility of each of RR and LL; row 0 channel 0 skipped, RR flagged by the file
    assert list(zip(cells["row"], cells["channel"], cells["product"])) == [(5, 2, -1), (5, 2, -2)]
    assert cells["value"][0].real == pytest.approx(106.0) and cells["value"][1].real == pytest.approx(6.0)
    assert cells["weight"].tolist() == [1.0, 1.0] and cells["flagged"].tolist() == [0, 0]
    assert cells["cond0"][0] == cells["cond0"][1] == pytest.approx(abs(cells["value"][0] - cells["value"][1]))
    # row 12: every channel found, one whole-row record per product
    assert list(zip(rows["row"], rows["product"], rows["n_cells"])) == [(12, -1, 4), (12, -2, 4)]
    assert rows["cond0_min"][0] == pytest.approx(50.0, abs=1.0) and rows["cond0_max"][0] >= rows["cond0_min"][0]
    assert path.name == "run-001.npz" and path.parent.name == "obs"  # the default session: the file's name
    summary = run.header["summary"]
    assert (summary["cells_found"], summary["rows_whole"], summary["records"]) == (5, 1, {"cells": 2, "rows": 2})
    assert run.header["products"] == {"RR": -1, "LL": -2} and run.header["columns"] == {"cond0": "amp(V) > 10"}
    assert run.searched_rows.sum() == 40 and "--locate --where 'amp(V) > 10'" in summary["command"]
    assert any("Locate session obs, new" in r and "the file's default" in r for r in reports)
    assert any("found 5 visibilities (row, channel) meeting every condition, 1 rows whole" in r for r in reports)


def test_flagged_include_tests_the_files_flagged_visibilities_too_recording_them_flagged():
    run, _, _ = _locate("flagged_include", "--where", "amp(V) > 10", "--flagged", "include")
    cells = run.cells
    assert list(zip(cells["row"], cells["channel"], cells["product"], cells["flagged"])) == [
        (0, 0, -1, 1), (0, 0, -2, 0), (5, 2, -1, 0), (5, 2, -2, 0)]


def test_flagged_only_tests_only_the_files_already_flagged_visibilities():
    # row 0 channel 0's RR is the synthetic file's one pre-existing flag (LL there is not), also planted to
    # match; row 5 channel 2 matches too but neither of its products is flagged, so --flagged only (a cell is
    # a candidate if either product V is made from is flagged) must leave it out.
    run, _, _ = _locate("flagged_only", "--where", "amp(V) > 10", "--flagged", "only")
    cells = run.cells
    assert list(zip(cells["row"], cells["channel"], cells["product"], cells["flagged"])) == [
        (0, 0, -1, 1), (0, 0, -2, 0)]


def test_a_row_whole_except_for_an_already_flagged_channel_is_not_compacted():
    # T26 2b-2, the user's fix (2026-10-21): a row is found whole only when every candidate channel (here,
    # every unflagged one) is found -- never "found, or skipped as already flagged". Row 20: channel 1's RR
    # already flagged by the file; channels 0, 2, 3 all bumped to match. Under the old rule this row would
    # have compacted to one ambiguous "whole" record (n_cells=3, without naming which three); now it must fall
    # through to explicit cells naming exactly 0, 2 and 3 -- never 1, and never one row record.
    def edit(image_data):
        image_data[20, 0, 1, 0, 2] = -1.0  # channel 1's RR: flagged
        image_data[20, 0, [0, 2, 3], 0, 0] += 100.0  # channels 0, 2, 3: bumped to match amp(V) > 10

    run, _, _ = _locate("mixed_whole", "--where", "amp(V) > 10", edit=edit)
    assert run.rows["row"].size == 0  # not compacted at all
    found = sorted(zip(run.cells["row"].tolist(), run.cells["channel"].tolist(), run.cells["product"].tolist()))
    assert found == sorted([(20, 0, -1), (20, 0, -2), (20, 2, -1), (20, 2, -2), (20, 3, -1), (20, 3, -2)])
    assert {c for r, c, p in found} == {0, 2, 3}  # channel 1 never appears: it was out of this run's domain


def test_a_dead_antenna_like_row_with_one_flagged_channel_is_not_whole():
    # row 0: |RR + LL| below 3 at every channel; channel 0's RR already flagged, so excluded from this run's
    # domain. Channels 1, 2, 3 match and are this run's entire domain for the row -- but since channel 0 was
    # never a candidate at all, the row is not "every candidate found" over the full channel axis (T26 2b-2,
    # 2026-10-21): it falls through to explicit cells, naming exactly 1, 2 and 3, never channel 0.
    run, _, _ = _locate("dead", "--where", "amp(I) < 3", edit=None)
    assert run.rows["row"].size == 0
    found = sorted(zip(run.cells["row"].tolist(), run.cells["channel"].tolist(), run.cells["product"].tolist()))
    assert found == sorted([(0, c, p) for c in (1, 2, 3) for p in (-1, -2)])


def test_coordinate_conditions_narrow_where_it_looks():
    run, _, _ = _locate("freq", "--where", "amp(V) > 10", "--where", "401 MHz <= freq <= 403 MHz")
    assert run.header["searched"]["channels"] == [1, 2, 3]
    assert list(zip(run.cells["row"], run.cells["channel"])) == [(5, 2), (5, 2)]
    assert list(zip(run.rows["row"], run.rows["n_cells"])) == [(12, 3), (12, 3)]  # whole over the channels searched
    run, _, _ = _locate("channel", "--where", "amp(V) > 10", "--where", "channel >= 3")
    assert list(zip(run.cells["row"], run.cells["channel"])) == [(12, 3), (12, 3)] and run.rows["row"].size == 0
    run, _, _ = _locate("time", "--where", "amp(V) > 10", "--where", "time < 0/18:00:25 recorded")  # rows 0-8
    assert run.header["searched"]["rows"] == 9 and set(run.cells["row"]) == {5} and run.rows["row"].size == 0


def test_runs_go_to_their_session_in_turn():
    scratch = make_scratch_dir("locate_where_session")
    path = _make_synthetic_file(scratch / "obs.fits", edit=_plant)
    opened = open_file(path)
    for where in ("amp(V) > 10", "amp(RR) > 100"):
        locate_by_conditions(PlotRequest.from_argv([str(path), "--locate", "--where", where, "--session", "rfi",
                                                    "--provenance-dir", str(scratch / "runs")]), opened)
    log = read_log(scratch / "runs" / "found" / "rfi")
    assert [(e["entry"], e.get("run"), e.get("conditions")) for e in log] == [
        ("session", None, None), ("run", 1, ["amp(V) > 10"]), ("run", 2, ["amp(RR) > 100"])]
    assert log[0]["file"]["name"] == "obs.fits" and log[0]["file"]["gcount"] == 40


def test_the_result_is_the_same_for_any_chunk_size_and_thread_count(monkeypatch):
    first, _, _ = _locate("chunks_a", "--where", "amp(V) > 10", "--threads", "1")
    import visplot.run as run_module
    monkeypatch.setattr(run_module, "stream_chunk_bytes", lambda: 3 * 100)  # a few rows per chunk
    second, _, _ = _locate("chunks_b", "--where", "amp(V) > 10", "--threads", "3")
    for table in ("cells", "rows"):
        for name, values in getattr(first, table).items():
            assert np.array_equal(values, getattr(second, table)[name]), (table, name)


def test_a_stopped_run_keeps_nothing():
    with pytest.raises(Stopped, match="stopped while locating; nothing kept"):
        _locate("stopped", "--where", "amp(V) > 10", progress=lambda p: (lambda rows_done: False))
    session = TEST_TMP_ROOT / "locate_where_stopped" / "runs" / "found" / "obs"
    assert sorted(p.name for p in session.iterdir()) == [".lock", "session.jsonl"]


@pytest.mark.parametrize("argv, message", [
    (("--where", "amp(Q) > 1"), r"Stokes Q of circular feeds needs RL, LR"),
    (("--where", "amp(V) > 10", "--uvdist-range", "1000:2000 km"), r"no rows selected"),
])
def test_what_the_file_cannot_give_is_refused(argv, message):
    with pytest.raises(RequestError, match=message):
        _locate("refused", *argv)


def test_a_session_of_another_file_is_refused():
    scratch = make_scratch_dir("locate_where_other")
    first = _make_synthetic_file(scratch / "a" / "obs.fits") if (scratch / "a").mkdir() is None else None
    other = _make_synthetic_file(scratch / "b" / "obs.fits", n_rows=30) if (scratch / "b").mkdir() is None else None
    argv = ["--locate", "--where", "amp(V) > 10", "--provenance-dir", str(scratch / "runs")]
    locate_by_conditions(PlotRequest.from_argv([str(first), *argv]), open_file(first))
    with pytest.raises(RequestError, match=r"session 'obs' .* belongs to obs\.fits .* 40 rows.* give --session"):
        locate_by_conditions(PlotRequest.from_argv([str(other), *argv]), open_file(other))


def test_the_command_line_runs_a_locate_and_its_record_names_the_run(capsys):
    from cli.run_visplot import main

    scratch = make_scratch_dir("locate_where_cli")
    path = _make_synthetic_file(scratch / "obs.fits", edit=_plant)
    assert main(["visplot", str(path), "--locate", "--where", "amp(V) > 10", "--session", "cal-rfi",
                 "--provenance-dir", str(scratch / "runs")]) == 0
    printed = capsys.readouterr().out
    run_file = scratch / "runs" / "found" / "cal-rfi" / "run-001.npz"
    assert "Locate session cal-rfi, new" in printed and "locating: 40 / 40 rows (100%)" in printed
    assert "run 1 of session cal-rfi: 2 records of single visibilities, 2 of whole rows" in printed
    [record] = (scratch / "runs" / "provenance" / "visplot").glob("*.json")
    manifest = json.loads(record.read_text())
    assert manifest["details"]["action"] == "locate" and manifest["outcome"]["status"] == "success"
    assert str(run_file) in manifest["outputs"]
    assert read_run(run_file).header["summary"]["record"].startswith(f"run {manifest['run_id']}")


# --- listing a session (T26 step 2b-2): --session NAME alone, the union of its active runs ---

from visplot.locate_where import lists_session, session_listing_text
from visplot.run import list_locate_session


def test_lists_session_is_session_alone():
    assert lists_session(_request("--session", "s1")) and not lists_session(_request())
    assert not lists_session(_request("--session", "s1", "--locate", "--where", "amp(V) > 5"))
    assert not lists_session(_request("--session", "s1", "--listobs"))
    assert not lists_session(_request("--session", "s1", "--plots", "amp-vs-time"))
    assert not lists_session(_request("--session", "s1", "--flagged", "only"))


def test_session_alone_is_a_request_of_its_own():
    checked = check_request(_request("--session", "s1"))
    assert checked.lists_session and checked.conditions == () and checked.plot_names == []
    assert not check_request(_request("--locate", "--where", "amp(V) > 5")).lists_session


def test_a_session_with_no_runs_yet_says_so():
    scratch = make_scratch_dir("locate_where_list_empty")
    path = _make_synthetic_file(scratch / "obs.fits")
    text = list_locate_session(PlotRequest.from_argv([str(path), "--session", "s1",
                                                      "--provenance-dir", str(scratch / "runs")]), open_file(path))
    assert "Locate session s1: obs.fits, 40 rows" in text and "no runs yet" in text


def test_listing_shows_every_run_its_status_and_the_active_found_set():
    scratch = make_scratch_dir("locate_where_list")
    path = _make_synthetic_file(scratch / "obs.fits", edit=_plant)
    argv = ["--session", "s1", "--provenance-dir", str(scratch / "runs")]
    opened = open_file(path)
    locate_by_conditions(PlotRequest.from_argv([str(path), "--locate", "--where", "amp(V) > 10", *argv]), opened)
    locate_by_conditions(PlotRequest.from_argv([str(path), "--locate", "--where", "amp(RR) > 1000", *argv]), opened)
    text = list_locate_session(PlotRequest.from_argv([str(path), *argv]), opened)
    assert "run" in text and "status" in text and "conditions" in text  # the header row
    lines = [line.split() for line in text.splitlines() if line.strip().startswith(("1", "2"))]
    assert lines[0][:2] == ["1", "active"] and lines[1][:2] == ["2", "active"]
    assert "found set: the union of runs 1, 2" in text

    session_dir = scratch / "runs" / "found" / "s1"
    with open(session_dir / "session.jsonl", "a") as f:
        f.write(json.dumps({"entry": "operation", "op": "undo", "target": 1}) + "\n")
    text = list_locate_session(PlotRequest.from_argv([str(path), *argv]), opened)
    lines = [line.split() for line in text.splitlines() if line.strip().startswith(("1", "2"))]
    assert lines[0][:2] == ["1", "aside"] and lines[1][:2] == ["2", "active"]
    assert "found set: the union of run 2" in text


def test_the_command_line_lists_a_session(capsys):
    from cli.run_visplot import main

    scratch = make_scratch_dir("locate_where_list_cli")
    path = _make_synthetic_file(scratch / "obs.fits", edit=_plant)
    argv = [str(path), "--locate", "--where", "amp(V) > 10", "--session", "s1", "--provenance-dir",
           str(scratch / "runs")]
    assert main(["visplot", *argv]) == 0
    capsys.readouterr()
    assert main(["visplot", str(path), "--session", "s1", "--provenance-dir", str(scratch / "runs")]) == 0
    printed = capsys.readouterr().out
    assert "Locate session s1: obs.fits, 40 rows" in printed and "amp(V) > 10" in printed
    assert "provenance record: run " in printed  # a listing is still a recorded run
    assert not (scratch / "runs" / "found" / "s1" / "run-002.npz").exists()  # nothing written by listing


def test_a_session_of_another_file_cannot_be_listed():
    scratch = make_scratch_dir("locate_where_list_other")
    first = _make_synthetic_file(scratch / "a" / "obs.fits") if (scratch / "a").mkdir() is None else None
    other = _make_synthetic_file(scratch / "b" / "obs.fits", n_rows=30) if (scratch / "b").mkdir() is None else None
    argv = ["--session", "s1", "--provenance-dir", str(scratch / "runs")]
    locate_by_conditions(PlotRequest.from_argv([str(first), "--locate", "--where", "amp(V) > 10", *argv]),
                         open_file(first))
    with pytest.raises(RequestError, match=r"session 's1' .* belongs to obs\.fits"):
        list_locate_session(PlotRequest.from_argv([str(other), *argv]), open_file(other))
