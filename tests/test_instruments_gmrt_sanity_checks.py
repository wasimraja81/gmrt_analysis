import numpy as np
import pytest
from astropy.io import fits

from data_io.row_index import RowIndex
from instruments.gmrt.antenna_table import Antenna
from instruments.gmrt.sanity_checks import (
    GmrtValidationError,
    check_antenna_count_is_plausible,
    check_row_count_consistency,
    check_telescope_is_gmrt,
)

from conftest import make_scratch_dir


def _make_minimal_fits(path, telescop):
    n = 1
    image_data = np.zeros((n, 1, 1, 1, 3), dtype=">f4")
    parnames = ["UU---SIN", "BASELINE"]
    pardata = [np.zeros(n, dtype=">f4"), np.array([1 * 256 + 2], dtype=">f4")]
    gdata = fits.GroupData(image_data, parnames=parnames, pardata=pardata, bitpix=-32)
    hdu = fits.GroupsHDU(gdata)
    if telescop is not None:
        hdu.header["TELESCOP"] = telescop
    hdu.writeto(path)
    return path


def test_check_telescope_is_gmrt_passes_for_gmrt():
    scratch = make_scratch_dir("gmrt_sanity")
    path = scratch / "gmrt.fits"
    _make_minimal_fits(path, "GMRT")
    check_telescope_is_gmrt(path)  # must not raise


def test_check_telescope_is_gmrt_raises_for_a_different_telescope():
    scratch = make_scratch_dir("gmrt_sanity")
    path = scratch / "vla.fits"
    _make_minimal_fits(path, "VLA")
    with pytest.raises(GmrtValidationError, match="VLA"):
        check_telescope_is_gmrt(path)


def test_check_telescope_is_gmrt_raises_when_missing():
    scratch = make_scratch_dir("gmrt_sanity")
    path = scratch / "no_telescop.fits"
    _make_minimal_fits(path, None)
    with pytest.raises(GmrtValidationError):
        check_telescope_is_gmrt(path)


def test_check_antenna_count_is_plausible_rejects_empty():
    with pytest.raises(GmrtValidationError, match="zero"):
        check_antenna_count_is_plausible([])


def test_check_antenna_count_is_plausible_rejects_more_than_the_encoding_can_represent():
    too_many = [Antenna(station_number=i, name=f"X:{i}") for i in range(2048)]
    with pytest.raises(GmrtValidationError):
        check_antenna_count_is_plausible(too_many)


def test_check_antenna_count_is_plausible_accepts_a_normal_gmrt_count():
    thirty = [Antenna(station_number=i, name=f"X:{i}") for i in range(1, 31)]
    check_antenna_count_is_plausible(thirty)  # must not raise


def _make_index(source_id, jd, ant1, ant2, integration_boundaries):
    return RowIndex(
        path="synthetic", gcount=len(source_id), pcount=8,
        source_id=np.asarray(source_id), jd=np.asarray(jd),
        ant1=np.asarray(ant1, dtype=np.int16), ant2=np.asarray(ant2, dtype=np.int16),
        uu_sec=np.zeros(len(source_id), dtype=np.float32), vv_sec=np.zeros(len(source_id), dtype=np.float32),
        source_ranges={}, integration_boundaries=np.asarray(integration_boundaries),
        id_to_name={}, chan_freqs_hz=np.array([]), stokes_labels=[],
        data_axis_lengths=[3, 1, 1, 1], data_axis_types=["COMPLEX", "STOKES", "FREQ", "IF"],
        data_offset=0, build_time_sec=0.0,
    )


def test_check_row_count_consistency_matches_for_a_uniform_cross_only_file():
    # 3 antennas, cross-only, 2 integrations of 3 rows each -- C(3,2) = 3.
    index = _make_index(
        source_id=[1, 1, 1, 1, 1, 1],
        jd=[100.0, 100.0, 100.0, 100.1, 100.1, 100.1],
        ant1=[1, 1, 2, 1, 1, 2], ant2=[2, 3, 3, 2, 3, 3],
        integration_boundaries=[0, 3, 6],
    )
    antennas = [Antenna(1, "A:1"), Antenna(2, "A:2"), Antenna(3, "A:3")]

    report = check_row_count_consistency(index, antennas)

    assert report.n_integrations == 2
    assert report.uniform is True
    assert report.common_row_count == 3
    assert report.autocorrelations_present is False
    assert report.expected_row_count == 3
    assert report.matches_expected is True


def test_check_row_count_consistency_accounts_for_autocorrelations():
    # 2 antennas, with autos: C(2,2)=1 cross + 2 autos = 3 rows/integration.
    index = _make_index(
        source_id=[1, 1, 1],
        jd=[100.0, 100.0, 100.0],
        ant1=[1, 1, 2], ant2=[1, 2, 2],  # two autos (1,1) and (2,2), one cross (1,2)
        integration_boundaries=[0, 3],
    )
    antennas = [Antenna(1, "A:1"), Antenna(2, "A:2")]

    report = check_row_count_consistency(index, antennas)

    assert report.autocorrelations_present is True
    assert report.expected_row_count == 3  # 1 cross + 2 autos
    assert report.matches_expected is True


def test_check_row_count_consistency_reports_non_uniform_without_raising():
    # Integration 1 has 3 rows, integration 2 has only 2 -- allowed (standing
    # rule 9), reported as non-uniform rather than treated as an error.
    index = _make_index(
        source_id=[1, 1, 1, 1, 1],
        jd=[100.0, 100.0, 100.0, 100.1, 100.1],
        ant1=[1, 1, 2, 1, 1], ant2=[2, 3, 3, 2, 3],
        integration_boundaries=[0, 3, 5],
    )
    antennas = [Antenna(1, "A:1"), Antenna(2, "A:2"), Antenna(3, "A:3")]

    report = check_row_count_consistency(index, antennas)

    assert report.uniform is False
    assert report.common_row_count is None
    assert report.matches_expected is False


def test_check_row_count_consistency_detects_a_mismatch_when_uniform():
    # Uniform row count, but wrong for the given (wrong) active-antenna list --
    # simulates a bad antenna resolution being caught.
    index = _make_index(
        source_id=[1, 1, 1],
        jd=[100.0, 100.0, 100.0],
        ant1=[1, 1, 2], ant2=[2, 3, 3],
        integration_boundaries=[0, 3],
    )
    antennas = [Antenna(1, "A:1"), Antenna(2, "A:2"), Antenna(3, "A:3"), Antenna(4, "A:4")]  # wrong: 4, not 3

    report = check_row_count_consistency(index, antennas)

    assert report.uniform is True
    assert report.common_row_count == 3
    assert report.expected_row_count == 6  # C(4,2)
    assert report.matches_expected is False
