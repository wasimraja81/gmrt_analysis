import numpy as np
import pytest
from astropy.io import fits

from data_io.row_index import build_row_index
from data_io.visibility_data import read_visibility_data

from conftest import make_scratch_dir


def _make_synthetic_uvfits_with_known_visibilities(path, n_rows=4, n_chan=3, n_stokes=2, n_if=1):
    """A random-groups file with a formula-based, exactly-known data value per
    (row, if, channel, stokes): real=row*1000+iff*100+chan*10+stokes,
    imag=real+0.5, weight=1.0+row*0.01 -- lets a test check read_visibility_data
    recovers exact values, not just plausible-looking ones.
    """
    image_data = np.zeros((n_rows, n_if, n_chan, n_stokes, 3), dtype=">f4")
    for r in range(n_rows):
        for f in range(n_if):
            for c in range(n_chan):
                for s in range(n_stokes):
                    real = r * 1000 + f * 100 + c * 10 + s
                    image_data[r, f, c, s, 0] = real
                    image_data[r, f, c, s, 1] = real + 0.5
                    image_data[r, f, c, s, 2] = 1.0 + r * 0.01

    baseline = np.array([1 * 256 + 2, 1 * 256 + 3, 2 * 256 + 3, 1 * 256 + 1], dtype=">f4")[:n_rows]
    parnames = ["UU---SIN", "VV---SIN", "WW---SIN", "BASELINE", "DATE", "DATE", "SOURCE", "FREQSEL"]
    pardata = [
        np.arange(n_rows, dtype=">f4") * 10.0, np.zeros(n_rows, dtype=">f4"), np.zeros(n_rows, dtype=">f4"),
        baseline, np.full(n_rows, 2460100.0, dtype=">f8"), np.zeros(n_rows, dtype=">f8"),
        np.ones(n_rows, dtype=">f4"), np.ones(n_rows, dtype=">f4"),
    ]
    gdata = fits.GroupData(image_data, parnames=parnames, pardata=pardata, bitpix=-32)
    hdu = fits.GroupsHDU(gdata)
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
    fits.HDUList([hdu, su_hdu]).writeto(path)
    return path


def _expected_value(r, f, c, s):
    real = r * 1000 + f * 100 + c * 10 + s
    return complex(real, real + 0.5), 1.0 + r * 0.01


def test_read_visibility_data_recovers_known_values_for_every_row_chan_stokes():
    scratch = make_scratch_dir("visibility_data")
    path = scratch / "synthetic.fits"
    _make_synthetic_uvfits_with_known_visibilities(path, n_rows=4, n_chan=3, n_stokes=2)
    index = build_row_index(path)

    block = read_visibility_data(path, index, row_indices=np.arange(4))

    assert block.axis_types == ["STOKES", "FREQ", "IF"]
    assert block.data.shape == (4, 2, 3, 1)
    assert block.weight.shape == (4, 2, 3, 1)
    for r in range(4):
        for c in range(3):
            for s in range(2):
                expected_data, expected_weight = _expected_value(r, 0, c, s)
                assert block.data[r, s, c, 0] == pytest.approx(expected_data)
                assert block.weight[r, s, c, 0] == pytest.approx(expected_weight)

    np.testing.assert_array_equal(block.row_indices, [0, 1, 2, 3])
    np.testing.assert_array_equal(block.ant1, index.ant1)
    np.testing.assert_array_equal(block.ant2, index.ant2)


def test_read_visibility_data_channel_and_stokes_selection():
    scratch = make_scratch_dir("visibility_data")
    path = scratch / "synthetic.fits"
    _make_synthetic_uvfits_with_known_visibilities(path, n_rows=4, n_chan=3, n_stokes=2)
    index = build_row_index(path)

    # select channels 0 and 2 (skip 1), stokes index 1 only
    block = read_visibility_data(
        path, index, row_indices=np.array([1, 3]),
        axis_selection={"FREQ": np.array([0, 2]), "STOKES": np.array([1])},
    )

    assert block.data.shape == (2, 1, 2, 1)
    expected_00, _ = _expected_value(1, 0, 0, 1)
    expected_01, _ = _expected_value(1, 0, 2, 1)
    expected_10, _ = _expected_value(3, 0, 0, 1)
    expected_11, _ = _expected_value(3, 0, 2, 1)
    assert block.data[0, 0, 0, 0] == pytest.approx(expected_00)
    assert block.data[0, 0, 1, 0] == pytest.approx(expected_01)
    assert block.data[1, 0, 0, 0] == pytest.approx(expected_10)
    assert block.data[1, 0, 1, 0] == pytest.approx(expected_11)
    np.testing.assert_allclose(block.chan_freqs_hz, [100e6, 102e6])
    assert block.stokes_labels == ["LL"]


def test_read_visibility_data_with_non_contiguous_row_indices():
    scratch = make_scratch_dir("visibility_data")
    path = scratch / "synthetic.fits"
    _make_synthetic_uvfits_with_known_visibilities(path, n_rows=4, n_chan=3, n_stokes=2)
    index = build_row_index(path)

    # rows 0 and 3 -- two separate contiguous runs, exercises the multi-run path.
    block = read_visibility_data(path, index, row_indices=np.array([3, 0]))

    np.testing.assert_array_equal(block.row_indices, [0, 3])  # sorted, not input order
    expected_0, _ = _expected_value(0, 0, 1, 0)
    expected_3, _ = _expected_value(3, 0, 1, 0)
    assert block.data[0, 0, 1, 0] == pytest.approx(expected_0)
    assert block.data[1, 0, 1, 0] == pytest.approx(expected_3)


def test_read_visibility_data_raises_when_estimated_size_exceeds_budget():
    scratch = make_scratch_dir("visibility_data")
    path = scratch / "synthetic.fits"
    _make_synthetic_uvfits_with_known_visibilities(path, n_rows=4, n_chan=3, n_stokes=2)
    index = build_row_index(path)

    with pytest.raises(MemoryError):
        read_visibility_data(path, index, row_indices=np.arange(4), max_bytes=10)  # absurdly small


def test_read_visibility_data_rejects_an_unknown_axis_selection_name():
    scratch = make_scratch_dir("visibility_data")
    path = scratch / "synthetic.fits"
    _make_synthetic_uvfits_with_known_visibilities(path, n_rows=2, n_chan=2, n_stokes=1)
    index = build_row_index(path)

    with pytest.raises(ValueError, match="RA"):
        read_visibility_data(path, index, row_indices=np.arange(2), axis_selection={"RA": [0]})


def test_read_visibility_data_handles_a_genuinely_non_trivial_extra_axis():
    # 2 IFs, both with real, distinct data -- the case that used to raise
    # NotImplementedError. Default (no axis_selection) must read both IFs
    # correctly; explicit selection must pick out just one.
    scratch = make_scratch_dir("visibility_data_multi_if")
    path = scratch / "synthetic.fits"
    _make_synthetic_uvfits_with_known_visibilities(path, n_rows=2, n_chan=2, n_stokes=1, n_if=2)
    index = build_row_index(path)

    block_all = read_visibility_data(path, index, row_indices=np.arange(2))
    assert block_all.axis_types == ["STOKES", "FREQ", "IF"]
    assert block_all.data.shape == (2, 1, 2, 2)
    for r in range(2):
        for f in range(2):
            for c in range(2):
                expected, _ = _expected_value(r, f, c, 0)
                assert block_all.data[r, 0, c, f] == pytest.approx(expected)

    block_if1 = read_visibility_data(path, index, row_indices=np.arange(2), axis_selection={"IF": np.array([1])})
    assert block_if1.data.shape == (2, 1, 2, 1)
    expected, _ = _expected_value(0, 1, 0, 0)
    assert block_if1.data[0, 0, 0, 0] == pytest.approx(expected)
    np.testing.assert_array_equal(block_if1.axis_indices["IF"], [1])


def _make_synthetic_uvfits_with_a_multi_pointing_ra_axis(path, n_rows=2, n_chan=2, n_stokes=1, n_ra=2):
    """RA (CTYPE6) with length > 1 -- a genuine UVFITS multi-pointing axis, not
    the IF axis the other tests use. Proves the generic selection mechanism
    isn't secretly IF-specific: known value formula never mentions IF at all.
    real = row*1000 + ra*200 + chan*10 + stokes.
    """
    n_if = 1
    # numpy shape (after row axis) is in descending axis-number order:
    # RA(6), IF(5), FREQ(4), STOKES(3), COMPLEX(2).
    image_data = np.zeros((n_rows, n_ra, n_if, n_chan, n_stokes, 3), dtype=">f4")
    for r in range(n_rows):
        for ra in range(n_ra):
            for c in range(n_chan):
                for s in range(n_stokes):
                    real = r * 1000 + ra * 200 + c * 10 + s
                    image_data[r, ra, 0, c, s, 0] = real
                    image_data[r, ra, 0, c, s, 1] = real + 0.5
                    image_data[r, ra, 0, c, s, 2] = 1.0 + r * 0.01

    baseline = np.array([1 * 256 + 2, 1 * 256 + 3], dtype=">f4")[:n_rows]
    parnames = ["UU---SIN", "VV---SIN", "WW---SIN", "BASELINE", "DATE", "DATE", "SOURCE", "FREQSEL"]
    pardata = [
        np.zeros(n_rows, dtype=">f4"), np.zeros(n_rows, dtype=">f4"), np.zeros(n_rows, dtype=">f4"),
        baseline, np.full(n_rows, 2460100.0, dtype=">f8"), np.zeros(n_rows, dtype=">f8"),
        np.ones(n_rows, dtype=">f4"), np.ones(n_rows, dtype=">f4"),
    ]
    gdata = fits.GroupData(image_data, parnames=parnames, pardata=pardata, bitpix=-32)
    hdu = fits.GroupsHDU(gdata)
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
    su_columns = fits.ColDefs([
        fits.Column(name="ID. NO.", format="J", array=np.array([1], dtype=np.int32)),
        fits.Column(name="SOURCE", format="16A", array=np.array(["3C48"])),
    ])
    su_hdu = fits.BinTableHDU.from_columns(su_columns, name="AIPS SU")
    fits.HDUList([hdu, su_hdu]).writeto(path)
    return path


def test_read_visibility_data_handles_a_non_trivial_ra_axis_not_just_if():
    # Proves genericity isn't secretly tied to the IF axis specifically --
    # RA is a different axis, in a different position, never named anywhere
    # in read_visibility_data's own logic.
    scratch = make_scratch_dir("visibility_data_multi_ra")
    path = scratch / "synthetic.fits"
    _make_synthetic_uvfits_with_a_multi_pointing_ra_axis(path, n_rows=2, n_chan=2, n_stokes=1, n_ra=2)
    index = build_row_index(path)
    assert index.data_axis_types == ["COMPLEX", "STOKES", "FREQ", "IF", "RA"]

    block_all = read_visibility_data(path, index, row_indices=np.arange(2))
    assert block_all.axis_types == ["STOKES", "FREQ", "IF", "RA"]
    assert block_all.data.shape == (2, 1, 2, 1, 2)
    for r in range(2):
        for ra in range(2):
            for c in range(2):
                real = r * 1000 + ra * 200 + c * 10 + 0
                expected = complex(real, real + 0.5)
                assert block_all.data[r, 0, c, 0, ra] == pytest.approx(expected)

    block_ra0 = read_visibility_data(path, index, row_indices=np.arange(2), axis_selection={"RA": np.array([0])})
    assert block_ra0.data.shape == (2, 1, 2, 1, 1)
    real = 1 * 1000 + 0 * 200 + 1 * 10 + 0
    assert block_ra0.data[1, 0, 1, 0, 0] == pytest.approx(complex(real, real + 0.5))
    np.testing.assert_array_equal(block_ra0.axis_indices["RA"], [0])
