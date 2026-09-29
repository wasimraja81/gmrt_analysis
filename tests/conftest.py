import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# Repo-local scratch space for tests -- never system /tmp (see feedback_no_tmp_at_all
# in project memory: this project never writes outside the repo/data tree).
TEST_TMP_ROOT = REPO_ROOT / "tmp" / "pytest"


def make_scratch_dir(name: str) -> Path:
    """Create (and truncate if present) a repo-local scratch directory for one test."""
    scratch = TEST_TMP_ROOT / name
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    return scratch


@pytest.fixture(scope="session")
def _visplot_records_dir() -> Path:
    return make_scratch_dir("visplot_runs")


@pytest.fixture(autouse=True)
def visplot_records_dir(_visplot_records_dir, monkeypatch) -> Path:
    """visplot's provenance records (default ./visplot_runs, in the directory
    it starts in) go to scratch in every test that does not name a folder."""
    monkeypatch.setattr("visplot.request.DEFAULT_PROVENANCE_DIR", str(_visplot_records_dir))
    return _visplot_records_dir
