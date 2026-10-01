import numpy as np
import pytest
from astropy.io import fits

from data_io.row_index import build_row_index
from data_io.visibility_data import iter_visibility_chunks

from conftest import make_scratch_dir


def _make_synthetic_uvfits_with_known_visibilities(path, n_rows=4, n_chan=3, n_stokes=2, n_if=1):
    """A random-groups file with a formula-based, exactly-known data value per
    (row, if, channel, stokes): real=row*1000+iff*100+chan*10+stokes,
    imag=real+0.5, weight=1.0+row*0.01 -- lets a test check iter_visibility_chunks
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


def _read_one_chunk(path, index, row_indices, axis_selection=None):
    """These files are a few rows, so the default chunk size covers them in one chunk."""
    chunks = list(iter_visibility_chunks(path, index, row_indices, axis_selection))
    assert len(chunks) == 1
    return chunks[0]


def _row_bytes(index):
    return (index.pcount + int(np.prod(index.data_axis_lengths))) * 4


def test_iter_visibility_chunks_concatenate_to_the_one_chunk_read():
    scratch = make_scratch_dir("visibility_data")
    path = scratch / "synthetic.fits"
    _make_synthetic_uvfits_with_known_visibilities(path, n_rows=4, n_chan=3, n_stokes=2)
    index = build_row_index(path)
    selection = {"FREQ": np.array([0, 2]), "STOKES": np.array([1])}

    whole = _read_one_chunk(path, index, np.arange(4), axis_selection=selection)
    chunks = list(iter_visibility_chunks(path, index, np.arange(4), selection, max_chunk_bytes=_row_bytes(index)))

    assert len(chunks) == 4  # one row per chunk
    np.testing.assert_array_equal(np.concatenate([c.data for c in chunks]), whole.data)
    np.testing.assert_array_equal(np.concatenate([c.weight for c in chunks]), whole.weight)
    np.testing.assert_array_equal(np.concatenate([c.row_indices for c in chunks]), [0, 1, 2, 3])
    np.testing.assert_array_equal(np.concatenate([c.jd for c in chunks]), whole.jd)


def test_iter_visibility_chunks_groups_separate_runs_into_one_chunk():
    scratch = make_scratch_dir("visibility_data")
    path = scratch / "synthetic.fits"
    _make_synthetic_uvfits_with_known_visibilities(path, n_rows=4, n_chan=3, n_stokes=2)
    index = build_row_index(path)

    # rows 0 and 2 are two runs; a two-row chunk budget holds both.
    chunks = list(iter_visibility_chunks(path, index, np.array([2, 0]), max_chunk_bytes=2 * _row_bytes(index)))

    assert len(chunks) == 1
    np.testing.assert_array_equal(chunks[0].row_indices, [0, 2])
    expected_0, _ = _expected_value(0, 0, 1, 0)
    expected_2, _ = _expected_value(2, 0, 1, 0)
    assert chunks[0].data[0, 0, 1, 0] == pytest.approx(expected_0)
    assert chunks[0].data[1, 0, 1, 0] == pytest.approx(expected_2)


def test_iter_visibility_chunks_never_exceeds_its_row_limit():
    scratch = make_scratch_dir("visibility_data")
    path = scratch / "synthetic.fits"
    _make_synthetic_uvfits_with_known_visibilities(path, n_rows=4, n_chan=3, n_stokes=2)
    index = build_row_index(path)

    chunks = list(iter_visibility_chunks(path, index, np.array([0, 1, 3]), max_chunk_bytes=2 * _row_bytes(index)))

    assert [len(c.row_indices) for c in chunks] == [2, 1]


def test_iter_visibility_chunks_recovers_known_values_for_every_row_chan_stokes():
    scratch = make_scratch_dir("visibility_data")
    path = scratch / "synthetic.fits"
    _make_synthetic_uvfits_with_known_visibilities(path, n_rows=4, n_chan=3, n_stokes=2)
    index = build_row_index(path)

    block = _read_one_chunk(path, index, row_indices=np.arange(4))

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


def test_iter_visibility_chunks_channel_and_stokes_selection():
    scratch = make_scratch_dir("visibility_data")
    path = scratch / "synthetic.fits"
    _make_synthetic_uvfits_with_known_visibilities(path, n_rows=4, n_chan=3, n_stokes=2)
    index = build_row_index(path)

    # select channels 0 and 2 (skip 1), stokes index 1 only
    block = _read_one_chunk(
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


def test_iter_visibility_chunks_with_non_contiguous_row_indices():
    scratch = make_scratch_dir("visibility_data")
    path = scratch / "synthetic.fits"
    _make_synthetic_uvfits_with_known_visibilities(path, n_rows=4, n_chan=3, n_stokes=2)
    index = build_row_index(path)

    # rows 0 and 3 -- two separate contiguous runs, exercises the multi-run path.
    block = _read_one_chunk(path, index, row_indices=np.array([3, 0]))

    np.testing.assert_array_equal(block.row_indices, [0, 3])  # sorted, not input order
    expected_0, _ = _expected_value(0, 0, 1, 0)
    expected_3, _ = _expected_value(3, 0, 1, 0)
    assert block.data[0, 0, 1, 0] == pytest.approx(expected_0)
    assert block.data[1, 0, 1, 0] == pytest.approx(expected_3)


def test_iter_visibility_chunks_rejects_an_unknown_axis_selection_name():
    scratch = make_scratch_dir("visibility_data")
    path = scratch / "synthetic.fits"
    _make_synthetic_uvfits_with_known_visibilities(path, n_rows=2, n_chan=2, n_stokes=1)
    index = build_row_index(path)

    with pytest.raises(ValueError, match="RA"):
        _read_one_chunk(path, index, row_indices=np.arange(2), axis_selection={"RA": [0]})


def test_iter_visibility_chunks_handles_a_genuinely_non_trivial_extra_axis():
    # 2 IFs, both with real, distinct data -- the case that used to raise
    # NotImplementedError. Default (no axis_selection) must read both IFs
    # correctly; explicit selection must pick out just one.
    scratch = make_scratch_dir("visibility_data_multi_if")
    path = scratch / "synthetic.fits"
    _make_synthetic_uvfits_with_known_visibilities(path, n_rows=2, n_chan=2, n_stokes=1, n_if=2)
    index = build_row_index(path)

    block_all = _read_one_chunk(path, index, row_indices=np.arange(2))
    assert block_all.axis_types == ["STOKES", "FREQ", "IF"]
    assert block_all.data.shape == (2, 1, 2, 2)
    for r in range(2):
        for f in range(2):
            for c in range(2):
                expected, _ = _expected_value(r, f, c, 0)
                assert block_all.data[r, 0, c, f] == pytest.approx(expected)

    block_if1 = _read_one_chunk(path, index, row_indices=np.arange(2), axis_selection={"IF": np.array([1])})
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


def test_iter_visibility_chunks_handles_a_non_trivial_ra_axis_not_just_if():
    # Proves genericity isn't secretly tied to the IF axis specifically --
    # RA is a different axis, in a different position, never named anywhere
    # in iter_visibility_chunks's own logic.
    scratch = make_scratch_dir("visibility_data_multi_ra")
    path = scratch / "synthetic.fits"
    _make_synthetic_uvfits_with_a_multi_pointing_ra_axis(path, n_rows=2, n_chan=2, n_stokes=1, n_ra=2)
    index = build_row_index(path)
    assert index.data_axis_types == ["COMPLEX", "STOKES", "FREQ", "IF", "RA"]

    block_all = _read_one_chunk(path, index, row_indices=np.arange(2))
    assert block_all.axis_types == ["STOKES", "FREQ", "IF", "RA"]
    assert block_all.data.shape == (2, 1, 2, 1, 2)
    for r in range(2):
        for ra in range(2):
            for c in range(2):
                real = r * 1000 + ra * 200 + c * 10 + 0
                expected = complex(real, real + 0.5)
                assert block_all.data[r, 0, c, 0, ra] == pytest.approx(expected)

    block_ra0 = _read_one_chunk(path, index, row_indices=np.arange(2), axis_selection={"RA": np.array([0])})
    assert block_ra0.data.shape == (2, 1, 2, 1, 1)
    real = 1 * 1000 + 0 * 200 + 1 * 10 + 0
    assert block_ra0.data[1, 0, 1, 0, 0] == pytest.approx(complex(real, real + 0.5))
    np.testing.assert_array_equal(block_ra0.axis_indices["RA"], [0])


def test_iter_visibility_chunks_metadata_only_reads_no_data():
    scratch = make_scratch_dir("visibility_data")
    path = scratch / "synthetic.fits"
    _make_synthetic_uvfits_with_known_visibilities(path, n_rows=4, n_chan=3, n_stokes=2)
    index = build_row_index(path)

    with_data = _read_one_chunk(path, index, np.arange(4))
    [meta] = list(iter_visibility_chunks(path, index, np.arange(4), read_data=False))

    assert meta.data is None and meta.weight is None
    np.testing.assert_array_equal(meta.jd, with_data.jd)
    np.testing.assert_array_equal(meta.source_id, index.source_id)
    assert meta.axis_types == with_data.axis_types


def test_spread_order_spreads_every_prefix_over_the_range():
    from data_io.visibility_data import spread_order

    assert spread_order(8).tolist() == [0, 4, 2, 6, 1, 5, 3, 7]
    assert spread_order(5).tolist() == [0, 4, 2, 1, 3]
    assert sorted(spread_order(13).tolist()) == list(range(13)) and spread_order(1).tolist() == [0]


def test_spread_chunks_read_every_row_once_ascending_and_cover_the_time_range_first(monkeypatch):
    import data_io.visibility_data as visibility_data
    from test_cli_visplot_output import _make_synthetic_file

    path = _make_synthetic_file(make_scratch_dir("visibility_spread") / "obs.fits")  # an integration every 3 rows
    index = build_row_index(path)
    row_bytes = _row_bytes(index)
    monkeypatch.setattr(visibility_data, "SPREAD_UNIT_BYTES", 3 * row_bytes)  # one integration per unit
    rows = np.arange(40)
    in_order = list(iter_visibility_chunks(path, index, rows, max_chunk_bytes=6 * row_bytes))
    spread = list(iter_visibility_chunks(path, index, rows, max_chunk_bytes=6 * row_bytes, spread=True))
    read = np.concatenate([c.row_indices for c in spread])
    assert sorted(read.tolist()) == list(range(40))  # every row once
    assert all(np.all(np.diff(c.row_indices) > 0) for c in spread)  # ascending within a chunk
    assert in_order[0].row_indices.max() == 5 and spread[0].row_indices.max() >= 21  # the first chunk spans the range
    by_row = {int(r): d for c in in_order for r, d in zip(c.row_indices, c.data)}
    assert all(np.array_equal(d, by_row[int(r)]) for c in spread for r, d in zip(c.row_indices, c.data))
