import os

import numpy as np
import pytest
from astropy.io import fits

from data_io.raw_data_access import (
    RawDataProtectionError,
    guard_output_path,
    open_fits_readonly,
    open_fits_tables,
    open_raw_memmap,
)

from conftest import make_scratch_dir

REAL_GWB_FITS = "/data1/gmrt/40_014_25JUL2021/40_014_25jul2021_2.6s_gwb.FITS"


def _make_synthetic_fits(directory) -> "os.PathLike":
    path = directory / "synthetic.fits"
    hdu = fits.PrimaryHDU(data=np.arange(12, dtype=np.float32).reshape(3, 4))
    hdu.writeto(path)
    return path


def test_open_fits_readonly_can_read_data():
    scratch = make_scratch_dir("data_io_fits_read")
    path = _make_synthetic_fits(scratch)

    with open_fits_readonly(path) as hdul:
        assert hdul.fileinfo(0)["filemode"] == "readonly"
        data = hdul[0].data
        assert data.shape == (3, 4)
        assert data[0, 0] == 0
        assert data[2, 3] == 11


def test_open_fits_readonly_exposes_no_write_mode_parameter():
    # There is deliberately no way to ask this function for a writable handle.
    import inspect

    from data_io import raw_data_access

    params = inspect.signature(raw_data_access.open_fits_readonly).parameters
    assert "mode" not in params


def test_open_fits_tables_reads_a_table_read_only_without_a_memory_map():
    import inspect

    from data_io import raw_data_access

    scratch = make_scratch_dir("data_io_fits_tables")
    path = scratch / "tables.fits"
    table = fits.BinTableHDU.from_columns([fits.Column(name="NOSTA", format="J", array=np.array([1, 2]))],
                                          name="AIPS AN")
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(path)
    with open_fits_tables(path) as hdul:
        assert hdul.fileinfo(1)["filemode"] == "readonly"
        assert list(hdul["AIPS AN"].data["NOSTA"]) == [1, 2]
    assert "mode" not in inspect.signature(raw_data_access.open_fits_tables).parameters


def test_open_raw_memmap_reads_expected_values_and_rejects_writes():
    scratch = make_scratch_dir("data_io_memmap")
    path = scratch / "raw_block.bin"
    array = np.arange(20, dtype=np.float32)
    array.tofile(path)

    mapped = open_raw_memmap(path, dtype=np.float32, shape=(20,))
    assert mapped[5] == 5.0

    with pytest.raises(ValueError):
        mapped[0] = 99.0


def test_guard_output_path_allows_a_genuinely_different_path():
    scratch = make_scratch_dir("data_io_guard_ok")
    raw_input = scratch / "raw.fits"
    raw_input.write_text("raw")
    output = scratch / "derived" / "bandpass.npz"

    guard_output_path(output, raw_input)  # must not raise


def test_guard_output_path_blocks_the_exact_same_path():
    scratch = make_scratch_dir("data_io_guard_same")
    raw_input = scratch / "raw.fits"
    raw_input.write_text("raw")

    with pytest.raises(RawDataProtectionError):
        guard_output_path(raw_input, raw_input)


def test_guard_output_path_blocks_a_relative_path_that_resolves_to_the_same_file():
    scratch = make_scratch_dir("data_io_guard_relative")
    (scratch / "sub").mkdir()
    raw_input = scratch / "raw.fits"
    raw_input.write_text("raw")
    disguised_output = scratch / "sub" / ".." / "raw.fits"

    with pytest.raises(RawDataProtectionError):
        guard_output_path(disguised_output, raw_input)


def test_guard_output_path_blocks_a_symlink_to_the_raw_input():
    scratch = make_scratch_dir("data_io_guard_symlink")
    raw_input = scratch / "raw.fits"
    raw_input.write_text("raw")
    symlinked_output = scratch / "looks_like_a_new_file.fits"
    symlinked_output.symlink_to(raw_input)

    with pytest.raises(RawDataProtectionError):
        guard_output_path(symlinked_output, raw_input)


def test_guard_output_path_checks_every_raw_input_in_a_list():
    scratch = make_scratch_dir("data_io_guard_multi")
    raw_a = scratch / "raw_a.fits"
    raw_b = scratch / "raw_b.fits"
    raw_a.write_text("a")
    raw_b.write_text("b")

    with pytest.raises(RawDataProtectionError):
        guard_output_path(raw_b, [raw_a, raw_b])


@pytest.mark.skipif(not os.path.exists(REAL_GWB_FITS), reason="real GWB raw data file not present on this host")
def test_open_fits_readonly_against_the_real_gwb_file():
    with open_fits_readonly(REAL_GWB_FITS) as hdul:
        assert hdul.fileinfo(0)["filemode"] == "readonly"
        assert hdul[0].header is not None


@pytest.mark.skipif(not os.path.exists(REAL_GWB_FITS), reason="archival GWB raw data file not present on this host")
def test_opening_the_archival_gwb_file_reads_its_tables_without_astropys_memory_map_warning():
    """T53: astropy's readonly memory map of the 389 GB file was refused and fell back, warning."""
    import warnings

    from astropy.utils.exceptions import AstropyUserWarning

    from visplot.run import open_file

    with warnings.catch_warnings():
        warnings.simplefilter("error", AstropyUserWarning)
        opened = open_file(REAL_GWB_FITS)
    assert len(opened.antennas) == 30 and len(opened.frequency_setups) == 1
