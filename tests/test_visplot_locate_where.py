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
    (("--undo", "2", "--restore", "3"), r"give one of .*; got --undo, --restore"),
    (("--undo", "2", "--locate", "--where", "amp(V) > 5"), r"--locate with no box runs by its --where conditions; "
                                                           r"--undo 2: give it in a separate run"),
    (("--undo", "2", "--plots", "amp-vs-time"), r"--undo 2 runs on a session by itself; --plots:"),
    (("--restore", "2", "--listobs"), r"--restore 2 runs on a session by itself; --listobs:"),
    (("--undo", "2", "--flagged", "only"), r"--flagged goes with --locate \(no box\) and --where; --undo takes no"),
    (("--undo", "2", "--session", "no good"), r"--session 'no good': a name of letters"),
    (("--undo", "2", "--replace", "3"), r"give one of .*; got --undo, --replace"),
    (("--restore", "2", "--replace", "3"), r"give one of .*; got --restore, --replace"),
    (("--replace", "2",), r"--replace goes with --locate \(no box\) and --where, or --session alone to list"),
    (("--replace", "2", "--session", "s1"), r"--session goes with --locate \(no box\) and --where, or --session "
                                            r"alone to list"),  # --session is given too, and named first
    (("--replace", "2", "--locate", "0:1,0:5", "--locate-csv", "x.csv"),
     r"--replace goes with --locate \(no box\) and --where, or --session alone to list"),
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
    found = sorted(zip(run.cells["row"], run.cells["channel"], run.cells["product"].tolist()))
    assert found == sorted([(20, 0, -1), (20, 0, -2), (20, 2, -1), (20, 2, -2), (20, 3, -1), (20, 3, -2)])
    assert {c for r, c, p in found} == {0, 2, 3}  # channel 1 never appears: it was out of this run's domain


def test_a_dead_antenna_like_row_with_one_flagged_channel_is_not_whole():
    # row 0: |RR + LL| below 3 at every channel; channel 0's RR already flagged, so excluded from this run's
    # domain. Channels 1, 2, 3 match and are this run's entire domain for the row -- but since channel 0 was
    # never a candidate at all, the row is not "every candidate found" over the full channel axis (T26 2b-2,
    # 2026-10-21): it falls through to explicit cells, naming exactly 1, 2 and 3, never channel 0.
    run, _, _ = _locate("dead", "--where", "amp(I) < 3", edit=None)
    assert run.rows["row"].size == 0
    found = sorted(zip(run.cells["row"], run.cells["channel"], run.cells["product"].tolist()))
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


# --- the run operations (T26 2b-2b): --undo, --restore ---

from visplot.locate_where import run_operation
from visplot.run import perform_run_operation


def test_run_operation_and_lists_session_agree_on_what_each_is():
    assert run_operation(_request("--undo", "2")) == "undo"
    assert run_operation(_request("--restore", "2")) == "restore"
    assert run_operation(_request()) is None
    assert not lists_session(_request("--session", "s1", "--undo", "2"))
    assert not lists_session(_request("--session", "s1", "--restore", "2"))
    assert lists_session(_request("--session", "s1"))


def test_undo_then_restore_on_a_real_session(capsys):
    from cli.run_visplot import main

    scratch = make_scratch_dir("locate_where_undo")
    path = _make_synthetic_file(scratch / "obs.fits", edit=_plant)
    argv = [str(path), "--provenance-dir", str(scratch / "runs"), "--session", "s1"]
    assert main(["visplot", *argv, "--locate", "--where", "amp(V) > 10"]) == 0
    capsys.readouterr()

    assert main(["visplot", *argv, "--undo", "1"]) == 0
    printed = capsys.readouterr().out
    assert "provenance record: run " in printed  # an operation is still a recorded run
    lines = [line.split() for line in printed.splitlines() if line.strip().startswith("1")]
    assert lines[0][:2] == ["1", "aside"]
    assert "found set: empty, no run active" in printed
    session_dir = scratch / "runs" / "found" / "s1"
    assert json.loads(session_dir.joinpath("session.jsonl").read_text().splitlines()[-1]) == {
        "entry": "operation", "op": "undo", "target": 1,
        "written_utc": json.loads(session_dir.joinpath("session.jsonl").read_text().splitlines()[-1])["written_utc"]}

    assert main(["visplot", *argv, "--restore", "1"]) == 0
    printed = capsys.readouterr().out
    lines = [line.split() for line in printed.splitlines() if line.strip().startswith("1")]
    assert lines[0][:2] == ["1", "active"]


def test_perform_run_operation_refuses_an_unknown_run():
    scratch = make_scratch_dir("locate_where_undo_refuse")
    path = _make_synthetic_file(scratch / "obs.fits", edit=_plant)
    request = PlotRequest.from_argv([str(path), "--undo", "9", "--provenance-dir", str(scratch / "runs"),
                                     "--session", "s1"])
    with pytest.raises(RequestError, match=r"run 9 is not a run of session 's1'"):
        perform_run_operation(request, open_file(path))


def test_a_run_operation_on_a_session_of_another_file_is_refused():
    scratch = make_scratch_dir("locate_where_undo_other")
    first = _make_synthetic_file(scratch / "a" / "obs.fits") if (scratch / "a").mkdir() is None else None
    other = _make_synthetic_file(scratch / "b" / "obs.fits", n_rows=30) if (scratch / "b").mkdir() is None else None
    argv = ["--session", "s1", "--provenance-dir", str(scratch / "runs")]
    locate_by_conditions(PlotRequest.from_argv([str(first), "--locate", "--where", "amp(V) > 10", *argv]),
                         open_file(first))
    request = PlotRequest.from_argv([str(other), "--undo", "1", *argv])
    with pytest.raises(RequestError, match=r"session 's1' .* belongs to obs\.fits"):
        perform_run_operation(request, open_file(other))


# --- the run operations (T26 2b-2b, slice 2): --replace ---

def test_replace_rides_with_locate_and_is_excluded_from_the_run_operations_alone():
    assert run_operation(_request("--replace", "2")) is None  # it is not a "session alone" operation
    assert not lists_session(_request("--session", "s1", "--replace", "2"))


def test_replace_writes_a_new_run_and_sets_the_old_one_aside(capsys):
    from cli.run_visplot import main

    scratch = make_scratch_dir("locate_where_replace")
    path = _make_synthetic_file(scratch / "obs.fits", edit=_plant)
    argv = [str(path), "--provenance-dir", str(scratch / "runs"), "--session", "s1"]
    assert main(["visplot", *argv, "--locate", "--where", "amp(V) > 10"]) == 0  # run 1
    capsys.readouterr()

    assert main(["visplot", *argv, "--locate", "--where", "amp(V) > 5", "--replace", "1"]) == 0  # run 2
    printed = capsys.readouterr().out
    assert "run 2 of session s1" in printed and "replacing run 1 (now aside)" in printed
    session_dir = scratch / "runs" / "found" / "s1"
    lines = [json.loads(line) for line in session_dir.joinpath("session.jsonl").read_text().splitlines()]
    operation = lines[-1]
    assert operation["entry"] == "operation" and operation["op"] == "undo" and operation["target"] == 1
    assert operation["reason"] == "replaced by run 2"
    run = read_run(session_dir / "run-002.npz")
    assert run.header["summary"]["replaces"] == 1

    listed = main(["visplot", str(path), "--session", "s1", "--provenance-dir", str(scratch / "runs")])
    assert listed == 0
    printed = capsys.readouterr().out
    assert "found set: the union of run 2" in printed


def test_replace_refuses_an_unknown_or_already_aside_run():
    scratch = make_scratch_dir("locate_where_replace_refuse")
    path = _make_synthetic_file(scratch / "obs.fits", edit=_plant)
    argv = [str(path), "--provenance-dir", str(scratch / "runs"), "--session", "s1"]
    locate_by_conditions(PlotRequest.from_argv([*argv, "--locate", "--where", "amp(V) > 10"]), open_file(path))
    with pytest.raises(RequestError, match=r"run 9 is not a run of session 's1'"):
        locate_by_conditions(PlotRequest.from_argv([*argv, "--locate", "--where", "amp(V) > 5", "--replace", "9"]),
                             open_file(path))
    locate_by_conditions(PlotRequest.from_argv([*argv, "--locate", "--where", "amp(V) > 5", "--replace", "1"]),
                         open_file(path))  # run 1 now aside
    with pytest.raises(RequestError, match=r"run 1 of session 's1' is already aside"):
        locate_by_conditions(PlotRequest.from_argv([*argv, "--locate", "--where", "amp(V) > 5", "--replace", "1"]),
                             open_file(path))


# --- the run operations (T26 2b-2b, slice 3): --intersect, --difference ---

from data_io.found_file import FoundRun
from visplot.locate_where import WHERE_FIELDS, combine_found, resolve_run_pair_arg


def _columns(records, names):
    """`records` (each a tuple matching `names`) as a RunWriter/FoundRun
    table: every location field besides `names` 0."""
    cols = {name: [] for name in (*names, *WHERE_FIELDS)}
    for record in records:
        for name, value in zip(names, record):
            cols[name].append(value)
        for field in WHERE_FIELDS:
            cols[field].append(0)
    return {name: np.array(values) for name, values in cols.items()}


def _found_run(channels, products, cells=(), rows=()):
    """A minimal `FoundRun` for `combine_found` tests: `channels`/`products`
    its own declared domain; `cells` an iterable of (row, product, channel),
    `rows` of (row, product)."""
    header = {"searched": {"channels": list(channels)}, "products": {str(p): p for p in products}}
    return FoundRun(header, np.array([], dtype=bool), _columns(cells, ("row", "product", "channel")),
                    _columns(rows, ("row", "product")))


def test_combine_found_intersects_two_explicit_cells_by_channel():
    a = _found_run([1, 2, 3], [5], cells=[(1, 5, 1), (1, 5, 2)])
    b = _found_run([1, 2, 3], [5], cells=[(1, 5, 2), (1, 5, 3)])
    cells, rows, channels, products = combine_found(a, b, "intersect")
    assert cells["row"] == [1] and cells["channel"] == [2] and not len(rows["row"])
    assert channels == (1, 2, 3) and products == (5,)


def test_combine_found_intersects_a_whole_row_with_an_explicit_one():
    a = _found_run([1, 2, 3], [5], rows=[(1, 5)])  # whole: every channel of its own range found
    b = _found_run([1, 2, 3], [5], cells=[(1, 5, 2)])
    cells, rows, channels, products = combine_found(a, b, "intersect")
    assert cells["row"] == [1] and cells["channel"] == [2] and not len(rows["row"])
    assert channels == (1, 2, 3)


def test_combine_found_intersects_two_whole_rows_into_a_whole_result():
    a = _found_run([1, 2, 3], [5], rows=[(1, 5)])
    b = _found_run([1, 2, 3], [5], rows=[(1, 5)])
    cells, rows, channels, products = combine_found(a, b, "intersect")
    assert not len(cells["row"]) and rows["row"] == [1] and rows["n_cells"] == [3]


def test_combine_found_intersect_narrows_to_the_two_runs_shared_channel_range():
    a = _found_run([1, 2, 3, 4], [5], rows=[(1, 5)])
    b = _found_run([3, 4, 5, 6], [5], rows=[(1, 5)])
    cells, rows, channels, products = combine_found(a, b, "intersect")
    assert channels == (3, 4) and rows["n_cells"] == [2] and not len(cells["row"])


def test_combine_found_intersect_of_different_products_is_empty_with_no_special_casing():
    a = _found_run([1, 2, 3], [5], cells=[(1, 5, 1)])
    b = _found_run([1, 2, 3], [6], cells=[(1, 6, 1)])
    cells, rows, channels, products = combine_found(a, b, "intersect")
    assert not len(cells["row"]) and not len(rows["row"]) and products == ()


def test_combine_found_difference_keeps_what_the_other_run_never_searched():
    a = _found_run([1, 2, 3], [5], cells=[(2, 5, 4)])  # row 2: outside b's own domain
    b = _found_run([1, 2, 3], [5], cells=[(1, 5, 1)])
    cells, rows, channels, products = combine_found(a, b, "difference")
    assert cells["row"] == [2] and cells["channel"] == [4]
    assert channels == (1, 2, 3)  # difference's own domain is a's, unconditionally


def test_combine_found_difference_removes_what_the_other_run_also_found():
    a = _found_run([1, 2, 3], [5], cells=[(3, 5, 8)])
    b = _found_run([1, 2, 3], [5], cells=[(3, 5, 8)])
    cells, rows, channels, products = combine_found(a, b, "difference")
    assert not len(cells["row"]) and not len(rows["row"])


def test_combine_found_difference_of_a_whole_row_against_a_partial_one():
    a = _found_run([1, 2, 3], [5], rows=[(1, 5)])  # whole: found 1, 2 and 3
    b = _found_run([1, 2, 3], [5], cells=[(1, 5, 2)])  # found only 2
    cells, rows, channels, products = combine_found(a, b, "difference")
    assert not len(rows["row"]) and sorted(cells["channel"]) == [1, 3]


def test_resolve_run_pair_arg_parses_and_refuses():
    assert resolve_run_pair_arg(None, "--intersect") is None
    assert resolve_run_pair_arg("2,3", "--intersect") == (2, 3)
    for spec, message in (("2", r"--intersect takes 'A,B'"), ("2,3,4", r"--intersect takes 'A,B'"),
                          ("x,y", r"--intersect takes 'A,B'"), ("2,2", r"--intersect 2,2: give two different runs")):
        with pytest.raises(ValueError, match=message):
            resolve_run_pair_arg(spec, "--intersect")


from data_io.found_file import FileIdentity, LocateSession
from visplot.locate_where import combined_fields


def _write_located_run(session, gcount, channels, products, cells=(), rows=(), conditions=("test",)):
    """A run of `session`, written directly with known content (bypassing
    the FITS pass): for --intersect/--difference end-to-end tests. `cells`
    an iterable of (row, product, channel), `rows` of (row, product,
    n_cells)."""
    writer = session.new_run(combined_fields(gcount, max(channels, default=0) + 1, 4, 4))
    writer.append("cells", _columns(cells, ("row", "product", "channel")))
    writer.append("rows", _columns(rows, ("row", "product", "n_cells")))
    summary = {"written_utc": "2026-01-01T00:00:00+00:00", "conditions": list(conditions),
              "cells_found": len(cells) + len(rows), "rows_whole": len(rows), "records": dict(writer.counts)}
    header = {"summary": summary, "searched": {"channels": list(channels)},
             "products": {str(p): p for p in products}}
    return writer.commit(header, np.zeros(gcount, dtype=bool))


def _combine_session(name):
    scratch = make_scratch_dir(f"locate_where_{name}")
    path = _make_synthetic_file(scratch / "obs.fits")
    opened = open_file(path)
    gcount = int(opened.index.gcount)
    session = LocateSession.open(scratch / "runs" / "found", "s1", FileIdentity.of(str(path), gcount))
    return scratch, path, session, gcount


def test_combine_runs_intersect_end_to_end(capsys):
    from cli.run_visplot import main

    scratch, path, session, gcount = _combine_session("intersect")
    _write_located_run(session, gcount, [0, 1, 2], [5], cells=[(1, 5, 0), (1, 5, 1)], conditions=["run a"])  # 1
    _write_located_run(session, gcount, [0, 1, 2], [5], cells=[(1, 5, 1), (1, 5, 2)], conditions=["run b"])  # 2

    argv = [str(path), "--provenance-dir", str(scratch / "runs"), "--session", "s1"]
    assert main(["visplot", *argv, "--intersect", "1,2"]) == 0
    printed = capsys.readouterr().out
    assert "intersect of runs 1 and 2" in printed and "runs 1 and 2 set aside" in printed
    run = read_run(session.run_path(3))
    assert run.cells["row"].tolist() == [1] and run.cells["channel"].tolist() == [1]
    summary = run.header["summary"]
    assert (summary["op"], summary["a"], summary["b"]) == ("intersect", 1, 2)
    entries = read_log(session.directory)
    assert entries[-1] == {"entry": "operation", "op": "intersect", "a": 1, "b": 2, "run": 3,
                           "written_utc": entries[-1]["written_utc"]}

    listed = main(["visplot", str(path), "--session", "s1", "--provenance-dir", str(scratch / "runs")])
    assert listed == 0
    printed = capsys.readouterr().out
    assert "found set: the union of run 3" in printed


def test_combine_runs_difference_end_to_end(capsys):
    from cli.run_visplot import main

    scratch, path, session, gcount = _combine_session("difference")
    _write_located_run(session, gcount, [0, 1, 2], [5], cells=[(1, 5, 0), (1, 5, 1)])  # 1
    _write_located_run(session, gcount, [0, 1, 2], [5], cells=[(1, 5, 1)])  # 2

    argv = [str(path), "--provenance-dir", str(scratch / "runs"), "--session", "s1"]
    assert main(["visplot", *argv, "--difference", "1,2"]) == 0
    printed = capsys.readouterr().out
    assert "difference of runs 1 and 2" in printed
    run = read_run(session.run_path(3))
    assert run.cells["row"].tolist() == [1] and run.cells["channel"].tolist() == [0]


def test_combine_runs_refuses_an_unknown_already_aside_or_repeated_run(capsys):
    from cli.run_visplot import main

    scratch, path, session, gcount = _combine_session("refuse")
    _write_located_run(session, gcount, [0, 1, 2], [5], cells=[(1, 5, 0)])  # 1
    _write_located_run(session, gcount, [0, 1, 2], [5], cells=[(1, 5, 1)])  # 2
    argv = [str(path), "--provenance-dir", str(scratch / "runs"), "--session", "s1"]

    with pytest.raises(SystemExit):
        main(["visplot", *argv, "--intersect", "1,9"])
    assert "run 9 is not a run of session 's1'" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["visplot", *argv, "--intersect", "1,1"])
    assert "give two different runs" in capsys.readouterr().err

    assert main(["visplot", *argv, "--intersect", "1,2"]) == 0  # run 3; 1 and 2 now aside
    capsys.readouterr()
    with pytest.raises(SystemExit):
        main(["visplot", *argv, "--difference", "1,3"])
    assert "run 1 of session 's1' is already aside" in capsys.readouterr().err
