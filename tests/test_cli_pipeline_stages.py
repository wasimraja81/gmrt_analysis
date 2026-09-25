import subprocess
from pathlib import Path

import numpy as np
import yaml
from astropy.io import fits

from cli.pipeline_stages import STAGE_ORDER, load_pipeline_config, run_build_index_stage, run_pipeline
from data_io.row_index import default_row_index_path, load_row_index
from provenance.run_index import run_index_path

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


def _make_synthetic_gwb_file(path):
    n = 3
    source_id = np.ones(n, dtype=">f4")
    baseline = np.array([1 * 256 + 2, 1 * 256 + 3, 2 * 256 + 3], dtype=">f4")
    image_data = np.random.default_rng(0).random((n, 1, 2, 1, 3)).astype(">f4")
    parnames = ["UU---SIN", "VV---SIN", "WW---SIN", "BASELINE", "DATE", "DATE", "SOURCE", "FREQSEL"]
    pardata = [
        np.zeros(n, dtype=">f4"), np.zeros(n, dtype=">f4"), np.zeros(n, dtype=">f4"),
        baseline, np.full(n, 2460100.0, dtype=">f8"), np.zeros(n, dtype=">f8"),
        source_id, np.ones(n, dtype=">f4"),
    ]
    gdata = fits.GroupData(image_data, parnames=parnames, pardata=pardata, bitpix=-32)
    hdu = fits.GroupsHDU(gdata)
    hdu.header["TELESCOP"] = "GMRT"
    hdu.header["CTYPE2"] = "COMPLEX"
    hdu.header["CTYPE3"] = "STOKES"
    hdu.header["CRVAL3"] = -1.0
    hdu.header["CDELT3"] = -1.0
    hdu.header["CTYPE4"] = "FREQ"
    hdu.header["CRVAL4"] = 100e6
    hdu.header["CDELT4"] = 1e6
    hdu.header["CRPIX4"] = 1.0
    hdu.header["CTYPE5"] = "IF"
    su_columns = fits.ColDefs([
        fits.Column(name="ID. NO.", format="J", array=np.array([1], dtype=np.int32)),
        fits.Column(name="SOURCE", format="16A", array=np.array(["3C48"])),
    ])
    su_hdu = fits.BinTableHDU.from_columns(su_columns, name="AIPS SU")
    an_columns = fits.ColDefs([
        fits.Column(name="NOSTA", format="J", array=np.array([1, 2, 3], dtype=np.int32)),
        fits.Column(name="ANNAME", format="8A", array=np.array(["C00:01", "C01:02", "C02:03"])),
        fits.Column(name="STABXYZ", format="3D", array=np.zeros((3, 3))),
    ])
    an_hdu = fits.BinTableHDU.from_columns(an_columns, name="AIPS AN")
    an_hdu.header["ARRAYX"] = 1657004.629
    an_hdu.header["ARRAYY"] = 5797894.3801
    an_hdu.header["ARRAYZ"] = 2073303.1705
    fits.HDUList([hdu, su_hdu, an_hdu]).writeto(path)
    return path


def test_stage_order_contains_build_index():
    names = [name for name, _fn in STAGE_ORDER]
    assert "build_index" in names


def test_run_build_index_stage_writes_manifest_log_and_index():
    scratch = make_scratch_dir("cli_pipeline_build_index")
    repo = scratch / "repo"
    repo.mkdir()
    _init_scratch_repo(repo)
    fits_path = repo / "synthetic.fits"
    _make_synthetic_gwb_file(fits_path)
    work_dir = scratch / "work"

    config = {"fits_path": str(fits_path), "work_dir": str(work_dir)}
    idx_path = run_build_index_stage(config)

    assert idx_path == default_row_index_path(fits_path)
    assert idx_path.exists()

    loaded = load_row_index(idx_path)
    assert loaded.gcount == 3

    manifest_dir = work_dir / "provenance" / "build_index"
    manifest_files = list(manifest_dir.glob("*.json"))
    assert len(manifest_files) == 1

    log_dir = work_dir / "logs" / "build_index"
    assert list(log_dir.glob("*.log"))

    assert run_index_path(work_dir).exists()


def test_run_build_index_stage_is_idempotent_by_default():
    scratch = make_scratch_dir("cli_pipeline_idempotent")
    repo = scratch / "repo"
    repo.mkdir()
    _init_scratch_repo(repo)
    fits_path = repo / "synthetic.fits"
    _make_synthetic_gwb_file(fits_path)
    work_dir = scratch / "work"
    config = {"fits_path": str(fits_path), "work_dir": str(work_dir)}

    idx_path = run_build_index_stage(config)
    first_write_time = idx_path.stat().st_mtime_ns

    idx_path_second = run_build_index_stage(config)

    assert idx_path_second == idx_path
    assert idx_path.stat().st_mtime_ns == first_write_time  # not rewritten

    manifest_dir = work_dir / "provenance" / "build_index"
    assert len(list(manifest_dir.glob("*.json"))) == 2  # both runs still recorded


def test_run_build_index_stage_force_rebuild_rewrites_the_index():
    scratch = make_scratch_dir("cli_pipeline_force_rebuild")
    repo = scratch / "repo"
    repo.mkdir()
    _init_scratch_repo(repo)
    fits_path = repo / "synthetic.fits"
    _make_synthetic_gwb_file(fits_path)
    work_dir = scratch / "work"
    config = {"fits_path": str(fits_path), "work_dir": str(work_dir)}

    idx_path = run_build_index_stage(config)
    first_write_time = idx_path.stat().st_mtime_ns

    config["build_index"] = {"force_rebuild": True}
    run_build_index_stage(config)

    assert idx_path.stat().st_mtime_ns != first_write_time  # genuinely rewritten


def test_run_pipeline_skips_a_disabled_stage():
    scratch = make_scratch_dir("cli_pipeline_skip")
    repo = scratch / "repo"
    repo.mkdir()
    _init_scratch_repo(repo)
    fits_path = repo / "synthetic.fits"
    _make_synthetic_gwb_file(fits_path)
    work_dir = scratch / "work"

    config = {
        "fits_path": str(fits_path),
        "work_dir": str(work_dir),
        "stages": {"build_index": False},
    }
    run_pipeline(config)

    assert not default_row_index_path(fits_path).exists()
    assert not work_dir.exists()


def test_run_pipeline_runs_an_enabled_stage():
    scratch = make_scratch_dir("cli_pipeline_run")
    repo = scratch / "repo"
    repo.mkdir()
    _init_scratch_repo(repo)
    fits_path = repo / "synthetic.fits"
    _make_synthetic_gwb_file(fits_path)
    work_dir = scratch / "work"

    config = {
        "fits_path": str(fits_path),
        "work_dir": str(work_dir),
        "stages": {"build_index": True},
    }
    run_pipeline(config)

    assert default_row_index_path(fits_path).exists()


def test_bin_run_gwb_pipeline_sh_end_to_end():
    # Invokes the actual bin/ entry point, not the Python functions directly --
    # catches sys.path/import wiring issues the in-process tests can't see.
    scratch = make_scratch_dir("cli_pipeline_bin_e2e")
    repo = scratch / "repo"
    repo.mkdir()
    _init_scratch_repo(repo)
    fits_path = repo / "synthetic.fits"
    _make_synthetic_gwb_file(fits_path)
    work_dir = scratch / "work"

    repo_root = Path(__file__).resolve().parents[1]
    config_path = scratch / "obs.yaml"
    config_path.write_text(yaml.dump({
        "fits_path": str(fits_path),
        "work_dir": str(work_dir),
        "stages": {"build_index": True},
    }))

    result = subprocess.run(
        [str(repo_root / "bin" / "run_gwb_pipeline.sh"), str(config_path)],
        capture_output=True, text=True,
    )

    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert default_row_index_path(fits_path).exists()


def test_load_pipeline_config_parses_yaml():
    scratch = make_scratch_dir("cli_pipeline_config")
    config_path = scratch / "test_obs.yaml"
    config_path.write_text(yaml.dump({
        "fits_path": "/some/path.FITS",
        "work_dir": "/some/work",
        "stages": {"build_index": True},
    }))

    config = load_pipeline_config(config_path)

    assert config["fits_path"] == "/some/path.FITS"
    assert config["stages"]["build_index"] is True
