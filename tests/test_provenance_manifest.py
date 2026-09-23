import json
import subprocess
from pathlib import Path

import pytest

from data_io.raw_data_access import RawDataProtectionError
from provenance.manifest import RunManifest

from conftest import make_scratch_dir


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _init_scratch_repo(repo: Path) -> None:
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "commit.gpgsign", "false")
    (repo / "tracked.txt").write_text("v1\n")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-q", "-m", "initial")


def _make_input_file(directory: Path, name: str, content: str) -> Path:
    path = directory / name
    path.write_text(content)
    return path


def test_successful_run_writes_manifest_with_expected_fields():
    scratch = make_scratch_dir("manifest_success")
    repo = scratch / "repo"
    repo.mkdir()
    _init_scratch_repo(repo)
    work_dir = scratch / "work"
    work_dir.mkdir()
    input_file = _make_input_file(scratch, "raw_input.fits", "fake fits bytes")

    with RunManifest(
        stage="unit_test_stage",
        work_dir=work_dir,
        parameters={"chan_start": 1731, "chan_end": 1901},
        inputs=[input_file],
        repo_root=repo,
    ) as manifest:
        output_file = work_dir / "output.npz"
        output_file.write_text("fake output")
        manifest.add_output(output_file)

    record = json.loads(manifest.manifest_path.read_text())

    assert record["stage"] == "unit_test_stage"
    assert record["parameters"] == {"chan_start": 1731, "chan_end": 1901}
    assert record["outputs"] == [str(output_file)]
    assert record["outcome"] == {
        "status": "success",
        "error_type": None,
        "error_message": None,
        "traceback": None,
    }
    assert record["git"]["commit"] == _git(repo, "rev-parse", "HEAD")
    assert record["git"]["dirty"] is False
    assert "diff_file" not in record["git"]

    [input_record] = record["inputs"]
    assert input_record["path"] == str(input_file.resolve())
    assert input_record["size_bytes"] == input_file.stat().st_size


def test_failed_run_records_exception_and_still_raises():
    scratch = make_scratch_dir("manifest_failure")
    repo = scratch / "repo"
    repo.mkdir()
    _init_scratch_repo(repo)
    work_dir = scratch / "work"
    work_dir.mkdir()
    input_file = _make_input_file(scratch, "raw_input.fits", "fake fits bytes")

    with pytest.raises(ValueError, match="deliberate failure"):
        with RunManifest(
            stage="unit_test_stage",
            work_dir=work_dir,
            parameters={},
            inputs=[input_file],
            repo_root=repo,
        ) as manifest:
            raise ValueError("deliberate failure")

    record = json.loads(manifest.manifest_path.read_text())
    assert record["outcome"]["status"] == "failed"
    assert record["outcome"]["error_type"] == "ValueError"
    assert record["outcome"]["error_message"] == "deliberate failure"
    assert "ValueError: deliberate failure" in record["outcome"]["traceback"]


def test_dirty_repo_writes_diff_sidecar_and_records_it_in_manifest():
    scratch = make_scratch_dir("manifest_dirty")
    repo = scratch / "repo"
    repo.mkdir()
    _init_scratch_repo(repo)
    # Uncommitted change -- this is what should make the repo "dirty".
    (repo / "tracked.txt").write_text("v2 uncommitted change\n")
    work_dir = scratch / "work"
    work_dir.mkdir()
    input_file = _make_input_file(scratch, "raw_input.fits", "fake fits bytes")

    with RunManifest(
        stage="unit_test_stage",
        work_dir=work_dir,
        parameters={},
        inputs=[input_file],
        repo_root=repo,
    ) as manifest:
        pass

    record = json.loads(manifest.manifest_path.read_text())
    assert record["git"]["dirty"] is True
    diff_file = manifest.manifest_path.parent / record["git"]["diff_file"]
    assert diff_file.exists()
    assert "v2 uncommitted change" in diff_file.read_text()


def test_input_fingerprint_uses_size_and_mtime_not_a_checksum():
    # Explicit regression test for the documented design decision: hashing the raw
    # GWB FITS file (hundreds of GB) on every run is not practical, so identity is
    # (path, size, mtime), not a cryptographic checksum. This test exists so that
    # decision can't silently drift back into an expensive checksum later.
    scratch = make_scratch_dir("manifest_fingerprint")
    repo = scratch / "repo"
    repo.mkdir()
    _init_scratch_repo(repo)
    work_dir = scratch / "work"
    work_dir.mkdir()
    input_file = _make_input_file(scratch, "raw_input.fits", "fake fits bytes")

    with RunManifest(
        stage="unit_test_stage",
        work_dir=work_dir,
        parameters={},
        inputs=[input_file],
        repo_root=repo,
    ) as manifest:
        pass

    record = json.loads(manifest.manifest_path.read_text())
    [input_record] = record["inputs"]
    assert set(input_record.keys()) == {"path", "size_bytes", "mtime_utc"}


def test_add_output_refuses_to_register_an_output_matching_a_raw_input():
    # This is the actual enforcement behind the user guide's claim that the pipeline
    # "refuses to run" rather than let a stage's output land on a raw input file --
    # it must be automatic here, in the one place every stage registers its outputs,
    # not something each stage author has to remember to check separately.
    scratch = make_scratch_dir("manifest_raw_data_guard")
    repo = scratch / "repo"
    repo.mkdir()
    _init_scratch_repo(repo)
    work_dir = scratch / "work"
    work_dir.mkdir()
    input_file = _make_input_file(scratch, "raw_input.fits", "fake fits bytes")

    with pytest.raises(RawDataProtectionError):
        with RunManifest(
            stage="unit_test_stage",
            work_dir=work_dir,
            parameters={},
            inputs=[input_file],
            repo_root=repo,
        ) as manifest:
            manifest.add_output(input_file)  # must be rejected, not recorded

    record = json.loads(manifest.manifest_path.read_text())
    assert record["outcome"]["status"] == "failed"
    assert record["outcome"]["error_type"] == "RawDataProtectionError"
    assert record["outputs"] == []
