"""Locate sessions on disk: a run written chunk by chunk and read back, run
numbers, a session tied to one file, and what a stopped or killed run leaves."""

import json
import shutil
import subprocess
import sys
from collections import namedtuple

import numpy as np
import pytest

from conftest import make_scratch_dir
from data_io import found_file
from data_io.found_file import FileIdentity, LocateSession, NoSpace, SessionError, log_combination, read_active, \
    read_log, read_run, restore_run, run_status, undo_run

FIELDS = {"cells": {"row": "<u4", "channel": "<u2", "value": "<c8"}, "rows": {"row": "<u4", "n_cells": "<u4"}}


def _identity(name="obs.fits", size=1000, gcount=10):
    return FileIdentity(name, size, gcount, f"/data/{name}")


def _write_run(session, n_chunks=2, summary=None):
    writer = session.new_run(FIELDS)
    for chunk in range(n_chunks):
        writer.append("cells", {"row": np.array([chunk, chunk]), "channel": np.array([1, 2]),
                                "value": np.array([1 + 2j, 3 - 1j])})
    writer.append("rows", {"row": np.array([9]), "n_cells": np.array([4])})
    searched = np.zeros(10, dtype=bool)
    searched[:5] = True
    return writer.commit({"summary": summary or {"conditions": ["amp(V) > 5"]}}, searched)


def test_a_run_is_written_chunk_by_chunk_and_read_back_whole():
    found = make_scratch_dir("found_run")
    session = LocateSession.open(found, "s1", _identity())
    assert session.created and session.runs() == []
    number, path = _write_run(session)
    assert (number, path.name) == (1, "run-001.npz") and not list(found.glob("s1/.partial-*"))
    run = read_run(path)
    assert run.cells["row"].tolist() == [0, 0, 1, 1] and run.cells["channel"].dtype == np.dtype("<u2")
    assert run.cells["value"].tolist() == [1 + 2j, 3 - 1j] * 2 and run.rows["n_cells"].tolist() == [4]
    assert run.searched_rows.tolist() == [True] * 5 + [False] * 5
    assert run.header["counts"] == {"cells": 4, "rows": 1} and run.header["session"] == "s1"
    assert session.runs() == [{"entry": "run", "run": 1, "file": "run-001.npz", "conditions": ["amp(V) > 5"]}]


def test_runs_are_numbered_in_turn_and_a_session_reopened_keeps_them():
    found = make_scratch_dir("found_numbers")
    _write_run(LocateSession.open(found, "s1", _identity()))
    reopened = LocateSession.open(found, "s1", FileIdentity("obs.fits", 1000, 10, "/moved/obs.fits"))  # data moved
    assert not reopened.created and _write_run(reopened)[0] == 2
    assert [entry["run"] for entry in reopened.runs()] == [1, 2]


def test_a_session_belongs_to_one_file():
    found = make_scratch_dir("found_other_file")
    LocateSession.open(found, "s1", _identity())
    with pytest.raises(SessionError, match=r"session 's1' .* belongs to obs\.fits .* this file is other\.fits"):
        LocateSession.open(found, "s1", _identity("other.fits"))


def test_a_stopped_run_leaves_the_session_as_it_was():
    found = make_scratch_dir("found_abort")
    session = LocateSession.open(found, "s1", _identity())
    writer = session.new_run(FIELDS)
    writer.append("cells", {"row": np.array([0]), "channel": np.array([0]), "value": np.array([1j])})
    writer.abort()
    assert sorted(p.name for p in (found / "s1").iterdir()) == [".lock", "session.jsonl"]


def test_a_killed_runs_partial_files_are_removed_at_the_next_opening():
    found = make_scratch_dir("found_killed")
    LocateSession.open(found, "s1", _identity())
    dead = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True, text=True)
    stale = found / "s1" / f".partial-{found_file._host()}-{int(dead.stdout)}-1"
    stale.mkdir()
    (stale / "cells_row.raw").write_bytes(b"\0" * 8)
    live = found / "s1" / f".partial-{found_file._host()}-{found_file.os.getpid()}-1"  # a run of ours, still going
    live.mkdir()
    session = LocateSession.open(found, "s1", _identity())
    assert session.removed_partials == [stale] and live.exists()


def test_a_whole_run_file_without_its_log_line_is_logged_late_saying_so():
    found = make_scratch_dir("found_unlogged")
    session = LocateSession.open(found, "s1", _identity())
    _, path = _write_run(session, summary={"conditions": ["amp(I) < 0.005"]})
    log = found / "s1" / "session.jsonl"
    written = json.loads(log.read_text().splitlines()[1])
    log.write_text(log.read_text().splitlines()[0] + "\n")  # killed between the rename and the log line
    assert found_file.check_run_file(path) is None
    reopened = LocateSession.open(found, "s1", _identity())
    late = reopened.runs()[0]
    assert reopened.logged_late == [1] and "its process stopped between the two" in late["logged_late"]
    assert {k: v for k, v in late.items() if not k.startswith("logged_late")} == written  # the line it would have had


def test_a_run_file_not_whole_is_set_aside_and_its_number_not_reused():
    found = make_scratch_dir("found_not_whole")
    session = LocateSession.open(found, "s1", _identity())
    _, path = _write_run(session)
    log = found / "s1" / "session.jsonl"
    log.write_text(log.read_text().splitlines()[0] + "\n")
    path.write_bytes(path.read_bytes()[:-200])  # cut short (a disk that lost its end)
    reopened = LocateSession.open(found, "s1", _identity())
    [(aside, problem)] = reopened.set_aside
    assert aside.name == "unlogged-run-001.npz" and problem and reopened.runs() == []
    assert _write_run(reopened)[0] == 2


def test_a_log_line_cut_short_at_the_end_is_left_out():
    found = make_scratch_dir("found_torn")
    session = LocateSession.open(found, "s1", _identity())
    _write_run(session)
    with open(found / "s1" / "session.jsonl", "a") as f:
        f.write('{"entry": "run", "ru')
    assert [entry["entry"] for entry in read_log(found / "s1")] == ["session", "run"]


def test_a_run_stops_before_it_would_fill_its_disk(monkeypatch):
    found = make_scratch_dir("found_space")
    writer = LocateSession.open(found, "s1", _identity()).new_run(FIELDS)
    writer.append("cells", {"row": np.arange(1000), "channel": np.zeros(1000), "value": np.zeros(1000)})
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(found_file.shutil, "disk_usage", lambda path: usage(100e9, 0, 2e9))
    writer.check_space(0.5)  # 14 kB written, about 35 kB in all: room
    monkeypatch.setattr(found_file.shutil, "disk_usage", lambda path: usage(100e9, 0, 1.0e9 + 20e3))
    with pytest.raises(NoSpace, match=r"of which 1\.0 GB is kept free; nothing kept"):
        writer.check_space(0.5)
    writer.abort()
    shutil.rmtree(found)


# --- run_status, active_runs, read_active (T26 step 2b-2: a session as the union of its active runs) ---

def test_run_status_folds_operation_entries_over_run_entries_in_order():
    entries = [
        {"entry": "session"},
        {"entry": "run", "run": 1},
        {"entry": "run", "run": 2},
        {"entry": "operation", "op": "undo", "target": 1},
        {"entry": "run", "run": 3},
        {"entry": "operation", "op": "intersect", "a": 2, "b": 3, "run": 4},
        {"entry": "run", "run": 4},
        {"entry": "operation", "op": "restore", "target": 1},
    ]
    assert run_status(entries) == {1: "active", 2: "aside", 3: "aside", 4: "active"}


def test_run_status_leaves_out_a_run_never_logged():
    assert run_status([{"entry": "session"}, {"entry": "operation", "op": "undo", "target": 5}]) == {5: "aside"}


def test_a_sessions_active_runs_are_those_never_set_aside():
    found = make_scratch_dir("found_active")
    session = LocateSession.open(found, "s1", _identity())
    _write_run(session)  # run 1
    _write_run(session)  # run 2
    assert session.status() == {1: "active", 2: "active"} and session.active_runs() == [1, 2]
    with open(session.directory / "session.jsonl", "a") as f:
        f.write(json.dumps({"entry": "operation", "op": "undo", "target": 1}) + "\n")
    reopened = LocateSession.open(found, "s1", _identity())
    assert reopened.status() == {1: "aside", 2: "active"} and reopened.active_runs() == [2]


def test_read_active_reads_every_active_runs_file_and_none_set_aside():
    found = make_scratch_dir("found_read_active")
    session = LocateSession.open(found, "s1", _identity())
    _write_run(session, summary={"conditions": ["amp(V) > 5"]})  # run 1
    _write_run(session, summary={"conditions": ["amp(RR) > 100"]})  # run 2
    with open(session.directory / "session.jsonl", "a") as f:
        f.write(json.dumps({"entry": "operation", "op": "undo", "target": 1}) + "\n")
    reopened = LocateSession.open(found, "s1", _identity())
    active = read_active(reopened)
    assert [run.header["summary"]["conditions"] for run in active] == [["amp(RR) > 100"]]


def test_a_sessions_run_path_names_a_run_whatever_its_status():
    found = make_scratch_dir("found_run_path")
    session = LocateSession.open(found, "s1", _identity())
    _, path = _write_run(session)
    assert session.run_path(1) == path


# --- undo_run, restore_run (T26 2b-2b: the run operations) ---

def test_undo_run_sets_a_run_aside_and_logs_the_entry():
    found = make_scratch_dir("found_undo")
    session = LocateSession.open(found, "s1", _identity())
    _write_run(session)  # run 1
    undo_run(session, 1)
    assert session.status() == {1: "aside"} and session.active_runs() == []
    entries = read_log(session.directory)
    assert entries[-1] == {"entry": "operation", "op": "undo", "target": 1, "written_utc": entries[-1]["written_utc"]}


def test_undo_run_keeps_its_reason():
    found = make_scratch_dir("found_undo_reason")
    session = LocateSession.open(found, "s1", _identity())
    _write_run(session)
    undo_run(session, 1, reason="replaced by run 2")
    assert read_log(session.directory)[-1]["reason"] == "replaced by run 2"


def test_undo_run_refuses_an_unknown_or_already_aside_run():
    found = make_scratch_dir("found_undo_refuse")
    session = LocateSession.open(found, "s1", _identity())
    _write_run(session)
    with pytest.raises(ValueError, match=r"run 9 is not a run of session 's1'"):
        undo_run(session, 9)
    undo_run(session, 1)
    with pytest.raises(ValueError, match=r"run 1 of session 's1' is already aside"):
        undo_run(session, 1)


def test_restore_run_makes_a_run_active_again_and_logs_the_entry():
    found = make_scratch_dir("found_restore")
    session = LocateSession.open(found, "s1", _identity())
    _write_run(session)
    undo_run(session, 1)
    restore_run(session, 1)
    assert session.status() == {1: "active"} and session.active_runs() == [1]
    assert read_log(session.directory)[-1] == {"entry": "operation", "op": "restore", "target": 1,
                                                "written_utc": read_log(session.directory)[-1]["written_utc"]}


def test_restore_run_refuses_an_unknown_or_already_active_run():
    found = make_scratch_dir("found_restore_refuse")
    session = LocateSession.open(found, "s1", _identity())
    _write_run(session)
    with pytest.raises(ValueError, match=r"run 9 is not a run of session 's1'"):
        restore_run(session, 9)
    with pytest.raises(ValueError, match=r"run 1 of session 's1' is already active"):
        restore_run(session, 1)


# --- log_combination (T26 2b-2b, slice 3: the run operations, intersect and difference) ---

def test_log_combination_sets_both_runs_aside_and_logs_the_entry():
    found = make_scratch_dir("found_combine")
    session = LocateSession.open(found, "s1", _identity())
    _write_run(session)  # run 1
    _write_run(session)  # run 2
    log_combination(session, "intersect", 1, 2, 3)
    assert session.status() == {1: "aside", 2: "aside"} and session.active_runs() == []
    entries = read_log(session.directory)
    assert entries[-1] == {"entry": "operation", "op": "intersect", "a": 1, "b": 2, "run": 3,
                           "written_utc": entries[-1]["written_utc"]}


def test_log_combination_refuses_the_same_run_twice():
    found = make_scratch_dir("found_combine_same")
    session = LocateSession.open(found, "s1", _identity())
    _write_run(session)
    with pytest.raises(ValueError, match=r"intersect needs two different runs, got 1 and 1 both"):
        log_combination(session, "intersect", 1, 1, 2)


def test_log_combination_refuses_an_unknown_or_already_aside_run():
    found = make_scratch_dir("found_combine_refuse")
    session = LocateSession.open(found, "s1", _identity())
    _write_run(session)  # run 1
    _write_run(session)  # run 2
    with pytest.raises(ValueError, match=r"run 9 is not a run of session 's1'"):
        log_combination(session, "difference", 1, 9, 3)
    log_combination(session, "difference", 1, 2, 3)  # 1 and 2 now aside
    _write_run(session)  # run 4, active
    with pytest.raises(ValueError, match=r"run 1 of session 's1' is already aside"):
        log_combination(session, "difference", 1, 4, 5)
