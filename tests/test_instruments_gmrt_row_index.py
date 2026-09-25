import numpy as np
import pytest
from astropy.io import fits

from instruments.gmrt.row_index import GmrtRowCountMismatch, build_gmrt_row_index

from conftest import make_scratch_dir


def _make_synthetic_gwb_file(path, extra_antenna=None):
    """4 antennas in the AN table (stations 1-4), but the raw data only ever
    uses stations 1, 2, 3 -- station 4 (C03:04) has zero rows, the same
    situation found in the real GWB file for C03:04/C10:10. Exactly 3 rows
    (one integration): the cross-only baseline count for the 3 antennas that
    actually have data, so row-count consistency holds once station 4 is
    correctly derived as dead-this-observation.

    `extra_antenna`, if given, is an extra (station_number, name) appended
    to the AN table with zero data -- for testing the structural-DUD default
    against a name like "C07:99" that isn't part of the normal 4.
    """
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

    nostas = [1, 2, 3, 4]
    annames = ["C00:01", "C01:02", "C02:03", "C03:04"]
    if extra_antenna is not None:
        nostas.append(extra_antenna[0])
        annames.append(extra_antenna[1])
    an_columns = fits.ColDefs([
        fits.Column(name="NOSTA", format="J", array=np.array(nostas, dtype=np.int32)),
        fits.Column(name="ANNAME", format="8A", array=np.array(annames)),
    ])
    an_hdu = fits.BinTableHDU.from_columns(an_columns, name="AIPS AN")

    fits.HDUList([hdu, su_hdu, an_hdu]).writeto(path)
    return path


def test_build_gmrt_row_index_derives_dead_this_observation_automatically():
    scratch = make_scratch_dir("gmrt_row_index")
    path = scratch / "synthetic.fits"
    _make_synthetic_gwb_file(path)

    index, resolution = build_gmrt_row_index(path)

    assert {a.name for a in resolution.active_antennas} == {"C00:01", "C01:02", "C02:03"}
    assert {a.name for a in resolution.dead_this_observation_antennas} == {"C03:04"}
    assert resolution.structural_dud_antennas == []  # C07/S05 default -- neither matches this AN table
    assert index.gcount == 3  # the plain, unmodified generic index -- every row still present

    report = resolution.row_count_consistency
    assert report.uniform is True
    assert report.common_row_count == 3
    assert report.n_active_antennas == 3
    assert report.autocorrelations_present is False
    assert report.expected_row_count == 3  # C(3,2), no autos
    assert report.matches_expected is True


def test_build_gmrt_row_index_structural_dud_default_applies_automatically():
    # An AN-table entry named like the hardcoded structural DUD (C07) is
    # excluded regardless of whether it has data -- station 4's absence is
    # separately, automatically derived as dead-this-observation.
    scratch = make_scratch_dir("gmrt_row_index")
    path = scratch / "synthetic.fits"
    _make_synthetic_gwb_file(path, extra_antenna=(5, "C07:05"))

    index, resolution = build_gmrt_row_index(path)

    assert {a.name for a in resolution.structural_dud_antennas} == {"C07:05"}
    assert {a.name for a in resolution.dead_this_observation_antennas} == {"C03:04"}
    assert {a.name for a in resolution.active_antennas} == {"C00:01", "C01:02", "C02:03"}


def test_build_gmrt_row_index_reports_unmatched_structural_dud_names():
    scratch = make_scratch_dir("gmrt_row_index")
    path = scratch / "synthetic.fits"
    _make_synthetic_gwb_file(path)

    _, resolution = build_gmrt_row_index(path)

    assert resolution.unmatched_structural_dud_names == ["C07", "S05"]  # neither exists in this AN table


def _make_synthetic_file_with_a_missing_baseline(path):
    """3 antennas (stations 1-3), all with data, but only 2 rows in the one
    integration where 3 are expected (C(3,2)=3) -- a row-count-only
    mismatch, unrelated to antenna presence.
    """
    n = 2
    source_id = np.ones(n, dtype=">f4")
    baseline = np.array([1 * 256 + 2, 1 * 256 + 3], dtype=">f4")  # (2,3) missing
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
    ])
    an_hdu = fits.BinTableHDU.from_columns(an_columns, name="AIPS AN")
    fits.HDUList([hdu, su_hdu, an_hdu]).writeto(path)
    return path


def test_build_gmrt_row_index_raises_on_a_row_count_mismatch():
    scratch = make_scratch_dir("gmrt_row_index_count_mismatch")
    path = scratch / "synthetic.fits"
    _make_synthetic_file_with_a_missing_baseline(path)

    # All 3 antennas have some data (none dead-this-observation), but the row
    # count is still wrong for a dense 3-antenna cross-only set.
    with pytest.raises(GmrtRowCountMismatch, match="3 active antennas"):
        build_gmrt_row_index(path)


def test_build_gmrt_row_index_row_count_mismatch_strict_false_does_not_raise():
    scratch = make_scratch_dir("gmrt_row_index_count_mismatch")
    path = scratch / "synthetic.fits"
    _make_synthetic_file_with_a_missing_baseline(path)

    index, resolution = build_gmrt_row_index(path, strict=False)

    assert resolution.row_count_consistency.matches_expected is False
    assert resolution.row_count_consistency.common_row_count == 2
    assert resolution.row_count_consistency.expected_row_count == 3
