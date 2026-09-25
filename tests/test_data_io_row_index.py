from unittest.mock import patch

import numpy as np
import pytest
from astropy.io import fits

from data_io.row_index import build_row_index, default_row_index_path, load_row_index, save_row_index

from conftest import make_scratch_dir


def _make_synthetic_uvfits_with_sources(path):
    """A small random-groups UVFITS file shaped like the real GWB/GSB files:
    same PTYPE order (including a doubled DATE column), an AIPS SU table,
    and two sources that each appear in two separate, non-contiguous blocks
    of rows -- exactly like source id 4 in the real GWB index, which has 9
    separate blocks rather than one.
    """
    n = 10
    source_id = np.array([1, 1, 1, 2, 2, 1, 1, 2, 2, 2], dtype=">f4")
    # ant pairs, including three autocorrelations (rows 2, 4, 6, 9) that must survive.
    baseline = np.array(
        [
            1 * 256 + 2, 1 * 256 + 3, 2 * 256 + 2, 1 * 256 + 2, 3 * 256 + 3,
            2 * 256 + 3, 1 * 256 + 1, 2 * 256 + 3, 1 * 256 + 2, 3 * 256 + 3,
        ],
        dtype=">f4",
    )
    uu = np.arange(n, dtype=">f4") * 10.0
    vv = np.arange(n, dtype=">f4") * 20.0
    ww = np.zeros(n, dtype=">f4")
    date1 = np.full(n, 2460100.0, dtype=">f8")
    date2 = np.arange(n, dtype=">f8") * 0.001
    freqsel = np.ones(n, dtype=">f4")

    # naxis2=3 (complex/weight), naxis3=2 (stokes), naxis4=4 (chan), naxis5=1 (IF)
    image_data = np.random.default_rng(0).random((n, 1, 4, 2, 3)).astype(">f4")
    parnames = ["UU---SIN", "VV---SIN", "WW---SIN", "BASELINE", "DATE", "DATE", "SOURCE", "FREQSEL"]
    pardata = [uu, vv, ww, baseline, date1, date2, source_id, freqsel]
    gdata = fits.GroupData(image_data, parnames=parnames, pardata=pardata, bitpix=-32)
    hdu = fits.GroupsHDU(gdata)
    # CTYPEs matching the real GWB/GSB header exactly (confirmed 2026-09-24) --
    # astropy does not set these automatically for a GroupsHDU, so a fixture
    # that omits them isn't actually representative of the real file's shape.
    hdu.header["CTYPE2"] = "COMPLEX"
    hdu.header["CTYPE3"] = "STOKES"
    hdu.header["CRVAL3"] = -1.0  # stokes: RR, LL
    hdu.header["CDELT3"] = -1.0
    hdu.header["CTYPE4"] = "FREQ"
    hdu.header["CRVAL4"] = 100e6  # channel 1 at 100 MHz
    hdu.header["CDELT4"] = 1e6
    hdu.header["CRPIX4"] = 1.0
    hdu.header["CTYPE5"] = "IF"

    su_columns = fits.ColDefs([
        fits.Column(name="ID. NO.", format="J", array=np.array([1, 2], dtype=np.int32)),
        fits.Column(name="SOURCE", format="16A", array=np.array(["3C48", "3C468.1"])),
    ])
    su_hdu = fits.BinTableHDU.from_columns(su_columns, name="AIPS SU")

    fits.HDUList([hdu, su_hdu]).writeto(path)
    return path


def test_build_row_index_against_a_synthetic_file_shaped_like_the_real_data():
    scratch = make_scratch_dir("row_index")
    path = scratch / "synthetic.fits"
    _make_synthetic_uvfits_with_sources(path)

    index = build_row_index(path)

    assert index.gcount == 10
    assert index.pcount == 8

    np.testing.assert_array_equal(index.ant1, [1, 1, 2, 1, 3, 2, 1, 2, 1, 3])
    np.testing.assert_array_equal(index.ant2, [2, 3, 2, 2, 3, 3, 1, 3, 2, 3])
    # Rows 2, 4, 6, 9 are autocorrelations -- must survive, not be dropped.
    assert list(index.ant1 == index.ant2) == [False, False, True, False, True, False, True, False, False, True]

    assert index.source_ranges == {1: [(0, 3), (5, 7)], 2: [(3, 5), (7, 10)]}
    assert index.id_to_name == {1: "3C48", 2: "3C468.1"}

    np.testing.assert_allclose(index.uu_sec, np.arange(10) * 10.0, atol=1e-2)
    np.testing.assert_allclose(index.jd, 2460100.0 + np.arange(10) * 0.001, atol=1e-6)

    np.testing.assert_allclose(index.chan_freqs_hz, [100e6, 101e6, 102e6, 103e6])
    assert index.stokes_labels == ["RR", "LL"]

    assert index.build_time_sec >= 0.0
    assert index.data_axis_lengths == [3, 2, 4, 1]
    assert index.data_axis_types == ["COMPLEX", "STOKES", "FREQ", "IF"]
    assert index.data_offset > 0


def test_build_row_index_finds_stokes_and_freq_axes_by_ctype_not_fixed_position():
    # Same as the real GWB header: NAXIS=7, with RA and DEC axes (length 1)
    # *after* IF -- CTYPE-driven lookup must still find STOKES/FREQ correctly
    # rather than assuming they sit at NAXIS3/NAXIS4 because nothing else exists.
    scratch = make_scratch_dir("row_index_extra_axes")
    path = scratch / "synthetic.fits"
    n = 4
    image_data = np.random.default_rng(0).random((n, 1, 1, 1, 4, 2, 3)).astype(">f4")
    parnames = ["UU---SIN", "VV---SIN", "WW---SIN", "BASELINE", "DATE", "DATE", "SOURCE", "FREQSEL"]
    pardata = [
        np.zeros(n, dtype=">f4"), np.zeros(n, dtype=">f4"), np.zeros(n, dtype=">f4"),
        np.array([1 * 256 + 2] * n, dtype=">f4"), np.full(n, 2460100.0, dtype=">f8"),
        np.zeros(n, dtype=">f8"), np.ones(n, dtype=">f4"), np.ones(n, dtype=">f4"),
    ]
    gdata = fits.GroupData(image_data, parnames=parnames, pardata=pardata, bitpix=-32)
    hdu = fits.GroupsHDU(gdata)
    assert hdu.header["NAXIS"] == 7, "fixture must actually have 7 axes to test this"
    hdu.header["CTYPE2"] = "COMPLEX"
    hdu.header["CTYPE3"] = "STOKES"
    hdu.header["CRVAL3"] = -1.0
    hdu.header["CDELT3"] = -1.0
    hdu.header["CTYPE4"] = "FREQ"
    hdu.header["CRVAL4"] = 100e6
    hdu.header["CDELT4"] = 1e6
    hdu.header["CRPIX4"] = 1.0
    hdu.header["CTYPE5"] = "IF"
    hdu.header["CTYPE6"] = "RA"
    hdu.header["CTYPE7"] = "DEC"
    su_columns = fits.ColDefs([
        fits.Column(name="ID. NO.", format="J", array=np.array([1], dtype=np.int32)),
        fits.Column(name="SOURCE", format="16A", array=np.array(["3C48"])),
    ])
    su_hdu = fits.BinTableHDU.from_columns(su_columns, name="AIPS SU")
    fits.HDUList([hdu, su_hdu]).writeto(path)

    index = build_row_index(path)

    assert index.data_axis_types == ["COMPLEX", "STOKES", "FREQ", "IF", "RA", "DEC"]
    assert index.data_axis_lengths == [3, 2, 4, 1, 1, 1]
    assert index.stokes_labels == ["RR", "LL"]
    np.testing.assert_allclose(index.chan_freqs_hz, [100e6, 101e6, 102e6, 103e6])


def test_build_row_index_source_ranges_cover_every_row_exactly_once():
    scratch = make_scratch_dir("row_index")
    path = scratch / "synthetic.fits"
    _make_synthetic_uvfits_with_sources(path)

    index = build_row_index(path)

    covered = np.zeros(index.gcount, dtype=bool)
    for runs in index.source_ranges.values():
        for start, stop in runs:
            assert not covered[start:stop].any(), "a row was assigned to more than one source range"
            covered[start:stop] = True
    assert covered.all(), "a row was not assigned to any source range"


def test_build_row_index_integration_boundaries_group_rows_sharing_source_and_time():
    # 3 integrations: rows [0,3) share (source=1, jd=100.0), rows [3,5) share
    # (source=1, jd=100.1) -- same source, new time -- and rows [5,8) share
    # (source=2, jd=100.1) -- same time, new source. Each must be its own
    # integration: a new one starts whenever SOURCE *or* DATE changes.
    scratch = make_scratch_dir("row_index_integrations")
    path = scratch / "synthetic.fits"
    n = 8
    source_id = np.array([1, 1, 1, 1, 1, 2, 2, 2], dtype=">f4")
    date2 = np.array([0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=">f8")
    baseline = np.array([1 * 256 + 2] * n, dtype=">f4")

    _write_variable_integration_fixture(path, source_id, date2, baseline)

    index = build_row_index(path)

    np.testing.assert_array_equal(index.integration_boundaries, [0, 3, 5, 8])


def _write_variable_integration_fixture(path, source_id, date2, baseline):
    from astropy.io import fits as _fits
    n = len(source_id)
    image_data = np.random.default_rng(0).random((n, 1, 4, 2, 3)).astype(">f4")
    parnames = ["UU---SIN", "VV---SIN", "WW---SIN", "BASELINE", "DATE", "DATE", "SOURCE", "FREQSEL"]
    pardata = [
        np.zeros(n, dtype=">f4"), np.zeros(n, dtype=">f4"), np.zeros(n, dtype=">f4"),
        baseline, np.full(n, 2460100.0, dtype=">f8"), date2, source_id, np.ones(n, dtype=">f4"),
    ]
    gdata = _fits.GroupData(image_data, parnames=parnames, pardata=pardata, bitpix=-32)
    hdu = _fits.GroupsHDU(gdata)
    hdu.header["CTYPE2"] = "COMPLEX"
    hdu.header["CTYPE3"] = "STOKES"
    hdu.header["CRVAL3"] = -1.0
    hdu.header["CDELT3"] = -1.0
    hdu.header["CTYPE4"] = "FREQ"
    hdu.header["CRVAL4"] = 100e6
    hdu.header["CDELT4"] = 1e6
    hdu.header["CRPIX4"] = 1.0
    hdu.header["CTYPE5"] = "IF"
    su_columns = _fits.ColDefs([
        _fits.Column(name="ID. NO.", format="J", array=np.array([1, 2], dtype=np.int32)),
        _fits.Column(name="SOURCE", format="16A", array=np.array(["3C48", "3C468.1"])),
    ])
    su_hdu = _fits.BinTableHDU.from_columns(su_columns, name="AIPS SU")
    _fits.HDUList([hdu, su_hdu]).writeto(path)
    return path


def test_default_row_index_path_is_adjacent_with_idx_npz_suffix():
    fits_path = "/data1/gmrt/some_obs/40_014_25jul2021_2.6s_gwb.FITS"
    assert str(default_row_index_path(fits_path)) == (
        "/data1/gmrt/some_obs/40_014_25jul2021_2.6s_gwb.FITS.idx.npz"
    )


def test_save_and_load_row_index_round_trips_every_field():
    scratch = make_scratch_dir("row_index_persistence")
    fits_path = scratch / "synthetic.fits"
    _make_synthetic_uvfits_with_sources(fits_path)
    index = build_row_index(fits_path)

    idx_path = default_row_index_path(fits_path)
    saved_path = save_row_index(index, idx_path)
    assert saved_path == idx_path
    assert idx_path.exists()

    loaded = load_row_index(idx_path)

    assert loaded.path == index.path
    assert loaded.gcount == index.gcount
    assert loaded.pcount == index.pcount
    np.testing.assert_array_equal(loaded.source_id, index.source_id)
    np.testing.assert_array_equal(loaded.jd, index.jd)
    np.testing.assert_array_equal(loaded.ant1, index.ant1)
    np.testing.assert_array_equal(loaded.ant2, index.ant2)
    np.testing.assert_array_equal(loaded.uu_sec, index.uu_sec)
    np.testing.assert_array_equal(loaded.vv_sec, index.vv_sec)
    assert loaded.ant1.dtype == index.ant1.dtype
    assert loaded.source_ranges == index.source_ranges
    np.testing.assert_array_equal(loaded.integration_boundaries, index.integration_boundaries)
    assert loaded.id_to_name == index.id_to_name
    np.testing.assert_array_equal(loaded.chan_freqs_hz, index.chan_freqs_hz)
    assert loaded.stokes_labels == index.stokes_labels
    assert loaded.data_axis_lengths == index.data_axis_lengths
    assert loaded.data_axis_types == index.data_axis_types
    assert loaded.data_offset == index.data_offset
    assert loaded.build_time_sec == index.build_time_sec

    leftover_temp_files = [p for p in idx_path.parent.iterdir() if ".tmp" in p.name]
    assert leftover_temp_files == []


def test_save_row_index_does_not_corrupt_an_existing_index_if_the_write_fails():
    scratch = make_scratch_dir("row_index_persistence_failure")
    fits_path = scratch / "synthetic.fits"
    _make_synthetic_uvfits_with_sources(fits_path)
    index = build_row_index(fits_path)
    idx_path = default_row_index_path(fits_path)

    save_row_index(index, idx_path)
    original_bytes = idx_path.read_bytes()

    with patch.object(np, "savez", side_effect=RuntimeError("simulated crash mid-write")):
        with pytest.raises(RuntimeError):
            save_row_index(index, idx_path)

    # The previously-saved, complete index must be untouched -- not a
    # truncated or partial file from the failed second write.
    assert idx_path.read_bytes() == original_bytes
    leftover_temp_files = [p for p in idx_path.parent.iterdir() if ".tmp" in p.name]
    assert leftover_temp_files == []
