import json

from provenance.run_index import append_to_run_index, run_index_path

from conftest import make_scratch_dir


def test_append_to_run_index_writes_one_json_line():
    scratch = make_scratch_dir("run_index_single")
    work_dir = scratch / "work"

    append_to_run_index(work_dir, {"run_id": "r1", "stage": "unit_test_stage"})

    lines = run_index_path(work_dir).read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == {"run_id": "r1", "stage": "unit_test_stage"}


def test_append_to_run_index_appends_without_rewriting_prior_lines():
    scratch = make_scratch_dir("run_index_multi")
    work_dir = scratch / "work"

    append_to_run_index(work_dir, {"run_id": "r1", "stage": "primary_calibration"})
    append_to_run_index(work_dir, {"run_id": "r2", "stage": "secondary_calibration"})

    lines = run_index_path(work_dir).read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["run_id"] == "r1"
    assert json.loads(lines[1])["run_id"] == "r2"
