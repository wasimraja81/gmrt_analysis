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
from data_io.found_file import FileIdentity, LocateSession, NoSpace, SessionError, read_log, read_run

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
