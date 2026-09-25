import os

import numpy as np
import pytest
from astropy.io import fits

from data_io.uvfits_group_params import (
    decode_baseline,
    read_all_param_columns,
    read_group_param_columns,
    read_group_params_layout,
    resolve_param_column,
)

from conftest import make_scratch_dir

REAL_GWB_FITS = "/data1/gmrt/40_014_25JUL2021/40_014_25jul2021_2.6s_gwb.FITS"


def _make_synthetic_groups_fits(path, uu, vv, ww, baseline, date1, date2, source):
    n_groups = len(baseline)
    # Small data cube per group: (complex/weight=3, pol=2, chan=4) -- deliberately
    # much larger than the parameter block, the same shape as the real file.
    image_data = np.random.default_rng(0).random((n_groups, 1, 1, 4, 2, 3)).astype(">f4")
    parnames = ["UU---SIN", "VV---SIN", "WW---SIN", "BASELINE", "DATE", "DATE", "SOURCE"]
    pardata = [
        np.asarray(uu, dtype=">f4"),
        np.asarray(vv, dtype=">f4"),
        np.asarray(ww, dtype=">f4"),
        np.asarray(baseline, dtype=">f4"),
        np.asarray(date1, dtype=">f8"),
        np.asarray(date2, dtype=">f8"),
        np.asarray(source, dtype=">f4"),
    ]
    gdata = fits.GroupData(image_data, parnames=parnames, pardata=pardata, bitpix=-32)
    hdu = fits.GroupsHDU(gdata)
    hdu.writeto(path)
    return path


def test_decode_baseline_standard_encoding_including_an_autocorrelation():
    # ant1=5, ant2=12 -> 5*256+12 ; ant1=7, ant2=7 (autocorrelation) -> 7*256+7
    baseline = np.array([5 * 256 + 12, 7 * 256 + 7], dtype=np.float64)
    ant1, ant2 = decode_baseline(baseline)
    np.testing.assert_array_equal(ant1, [5, 7])
    np.testing.assert_array_equal(ant2, [12, 7])


def test_decode_baseline_extended_encoding_for_large_antenna_numbers():
    # ant1=300, ant2=17 in the extended scheme: 65536 + 300*2048 + 17
    value = 65536 + 300 * 2048 + 17
    ant1, ant2 = decode_baseline(np.array([value], dtype=np.float64))
    assert ant1[0] == 300
    assert ant2[0] == 17


def test_read_group_params_layout_matches_a_synthetic_file():
    scratch = make_scratch_dir("uvfits_layout")
    path = scratch / "synthetic.fits"
    n = 5
    _make_synthetic_groups_fits(
        path,
        uu=np.zeros(n), vv=np.zeros(n), ww=np.zeros(n),
        baseline=np.arange(n) + 1.0,
        date1=np.full(n, 2460000.0), date2=np.zeros(n),
        source=np.ones(n),
    )

    layout = read_group_params_layout(path)

    assert layout.gcount == n
    assert layout.pcount == 7
    assert layout.data_floats_per_group == 1 * 1 * 4 * 2 * 3
    assert layout.param_names[3] == "BASELINE"


def test_resolve_param_column_is_tolerant_of_uvfits_punctuation():
    scratch = make_scratch_dir("uvfits_resolve")
    path = scratch / "synthetic.fits"
    _make_synthetic_groups_fits(
        path,
        uu=np.zeros(3), vv=np.zeros(3), ww=np.zeros(3),
        baseline=np.array([1.0, 2.0, 3.0]),
        date1=np.zeros(3), date2=np.zeros(3), source=np.zeros(3),
    )
    layout = read_group_params_layout(path)

    assert resolve_param_column(layout, "UU") == 0
    assert resolve_param_column(layout, "BASELINE") == 3
    assert resolve_param_column(layout, "SOURCE") == 6

    with pytest.raises(ValueError):
        resolve_param_column(layout, "NOT_A_REAL_PARAMETER")


def test_read_group_param_columns_recovers_known_values_including_an_autocorrelation():
    scratch = make_scratch_dir("uvfits_read_columns")
    path = scratch / "synthetic.fits"
    n = 6
    known_baseline = np.array(
        [1 * 256 + 2, 3 * 256 + 4, 7 * 256 + 7, 1 * 256 + 3, 2 * 256 + 4, 7 * 256 + 7],
        dtype=np.float64,
    )
    known_source = np.array([1, 1, 1, 2, 2, 2], dtype=np.float64)
    known_uu = np.array([10.0, 20.0, 0.0, 30.0, 40.0, 0.0], dtype=np.float64)

    _make_synthetic_groups_fits(
        path,
        uu=known_uu, vv=np.zeros(n), ww=np.zeros(n),
        baseline=known_baseline,
        date1=np.full(n, 2460100.5), date2=np.zeros(n),
        source=known_source,
    )
    layout = read_group_params_layout(path)

    columns = read_group_param_columns(path, layout, ["BASELINE", "SOURCE", "UU"])

    np.testing.assert_allclose(columns["BASELINE"], known_baseline)
    np.testing.assert_allclose(columns["SOURCE"], known_source)
    np.testing.assert_allclose(columns["UU"], known_uu, atol=1e-3)

    ant1, ant2 = decode_baseline(columns["BASELINE"])
    # Row 2 and row 5 are autocorrelations (ant1 == ant2) -- must survive, not be dropped.
    assert (ant1 == ant2).sum() == 2
    np.testing.assert_array_equal(ant1 == ant2, [False, False, True, False, False, True])


def test_decode_baseline_returns_int16_not_a_gmrt_sized_assumption():
    # int16 (max 32767) comfortably covers the AIPS extended encoding's own
    # ceiling of 2047 -- correct for any telescope using this convention,
    # not narrowed to GMRT's 30 antennas specifically.
    ant1, ant2 = decode_baseline(np.array([1 * 256 + 2], dtype=np.float64))
    assert ant1.dtype == np.int16
    assert ant2.dtype == np.int16


def test_read_all_param_columns_with_a_small_chunk_budget_matches_reading_it_whole():
    # max_chunk_bytes forced small enough to require several chunks, on a file
    # with a known autocorrelation -- chunk boundaries must not corrupt or
    # drop any row, including the one that doesn't divide evenly into chunks.
    scratch = make_scratch_dir("uvfits_chunked_read")
    path = scratch / "synthetic.fits"
    n = 7
    known_baseline = np.array(
        [1 * 256 + 2, 3 * 256 + 4, 7 * 256 + 7, 1 * 256 + 3, 2 * 256 + 4, 7 * 256 + 7, 5 * 256 + 6],
        dtype=np.float64,
    )
    _make_synthetic_groups_fits(
        path,
        uu=np.arange(n, dtype=np.float64), vv=np.zeros(n), ww=np.zeros(n),
        baseline=known_baseline,
        date1=np.full(n, 2460100.5), date2=np.zeros(n),
        source=np.ones(n),
    )
    layout = read_group_params_layout(path)

    whole = read_all_param_columns(path, layout)  # default budget: one chunk, this file is tiny
    row_bytes = (layout.pcount + layout.data_floats_per_group) * 4
    chunked = read_all_param_columns(path, layout, max_chunk_bytes=row_bytes * 2)  # forces ~4 chunks for n=7

    np.testing.assert_array_equal(whole, chunked)
    bl_col = resolve_param_column(layout, "BASELINE")
    ant1, ant2 = decode_baseline(chunked[:, bl_col])
    np.testing.assert_array_equal(ant1 == ant2, [False, False, True, False, False, True, False])


_RUN_REAL_GWB_SCAN = os.environ.get("RUN_REAL_GWB_SCAN") == "1"


@pytest.mark.skipif(
    not (_RUN_REAL_GWB_SCAN and os.path.exists(REAL_GWB_FITS)),
    reason="reads the full 389GB raw file (minutes, real disk I/O) -- opt in with RUN_REAL_GWB_SCAN=1, never runs by default",
)
def test_read_group_params_against_the_real_gwb_file_is_fast_and_correct():
    import time

    layout = read_group_params_layout(REAL_GWB_FITS)
    assert layout.gcount == 3959928
    assert layout.pcount == 8

    started = time.monotonic()
    columns = read_group_param_columns(REAL_GWB_FITS, layout, ["BASELINE", "SOURCE"])
    elapsed = time.monotonic() - started

    assert columns["BASELINE"].shape == (layout.gcount,)
    print(f"read_group_param_columns on the real GWB file took {elapsed:.1f}s")

    ant1, ant2 = decode_baseline(columns["BASELINE"])
    assert not np.any(ant1 == ant2)  # matches the cached index's own finding
