import numpy as np
import pytest

from data_io.row_index import RowIndex
from data_io.row_selection import select_rows


def _make_index():
    # 10 rows, 2 sources, 4 integrations (boundaries at 0,3,5,7,10).
    # ant1==ant2 (autocorrelations) at rows 2, 4, 6, 9.
    source_id = np.array([1, 1, 1, 2, 2, 1, 1, 2, 2, 2], dtype=np.int32)
    jd = np.array([100.0, 100.0, 100.0, 100.1, 100.1, 100.2, 100.2, 100.3, 100.3, 100.3])
    ant1 = np.array([1, 1, 2, 1, 3, 2, 1, 2, 1, 3], dtype=np.int16)
    ant2 = np.array([2, 3, 2, 2, 3, 3, 1, 3, 2, 3], dtype=np.int16)
    uu_sec = np.array([0, 1e-6, 0, 2e-6, 0, 3e-6, 0, 4e-6, 0, 5e-6], dtype=np.float32)
    vv_sec = np.zeros(10, dtype=np.float32)

    return RowIndex(
        path="synthetic", gcount=10, pcount=8,
        source_id=source_id, jd=jd, ant1=ant1, ant2=ant2, uu_sec=uu_sec, vv_sec=vv_sec,
        source_ranges={1: [(0, 3), (5, 7)], 2: [(3, 5), (7, 10)]},
        integration_boundaries=np.array([0, 3, 5, 7, 10]),
        id_to_name={1: "3C48", 2: "3C286"},
        chan_freqs_hz=np.array([100e6, 101e6]),
        stokes_labels=["RR", "LL"],
        data_axis_lengths=[3, 2, 2, 1], data_axis_types=["COMPLEX", "STOKES", "FREQ", "IF"],
        data_offset=8640, build_time_sec=0.1,
    )


def test_select_rows_with_no_criteria_defaults_to_cross_only():
    index = _make_index()
    sel = select_rows(index)
    np.testing.assert_array_equal(sel.row_indices, [0, 1, 3, 5, 7, 8])
    assert sel.n_rows == 6


def test_select_rows_filters_by_source_name():
    index = _make_index()
    sel = select_rows(index, sources="3C48")
    np.testing.assert_array_equal(sel.row_indices, [0, 1, 5])
    assert sel.sources == {1: "3C48"}


def test_select_rows_filters_by_source_id():
    index = _make_index()
    sel = select_rows(index, sources=1)
    np.testing.assert_array_equal(sel.row_indices, [0, 1, 5])


def test_select_rows_filters_by_a_list_of_sources_mixing_name_and_id():
    index = _make_index()
    sel = select_rows(index, sources=["3C48", 2], correlation_type="both")
    np.testing.assert_array_equal(sel.row_indices, np.arange(10))
    assert sel.sources == {1: "3C48", 2: "3C286"}


def test_select_rows_unknown_source_name_raises_with_available_list():
    index = _make_index()
    with pytest.raises(ValueError, match="3C48"):
        select_rows(index, sources="not-a-real-source")


def test_select_rows_correlation_type_auto_and_both():
    index = _make_index()
    auto = select_rows(index, correlation_type="auto")
    np.testing.assert_array_equal(auto.row_indices, [2, 4, 6, 9])

    both = select_rows(index, correlation_type="both")
    np.testing.assert_array_equal(both.row_indices, np.arange(10))

    with pytest.raises(ValueError):
        select_rows(index, correlation_type="not-a-real-type")


def test_select_rows_by_integration_range():
    index = _make_index()
    sel = select_rows(index, correlation_type="both", integration_range=(1, 2))
    np.testing.assert_array_equal(sel.row_indices, [3, 4, 5, 6])

    with pytest.raises(ValueError):
        select_rows(index, integration_range=(0, 99))


def test_select_rows_by_jd_range():
    index = _make_index()
    sel = select_rows(index, correlation_type="both", jd_range=(100.1, 100.2))
    np.testing.assert_array_equal(sel.row_indices, [3, 4, 5, 6])


def test_select_rows_by_antennas_include_list():
    index = _make_index()
    sel = select_rows(index, correlation_type="both", antennas=[1, 2])
    np.testing.assert_array_equal(sel.row_indices, [0, 2, 3, 6, 8])


def test_select_rows_by_exclude_antennas():
    index = _make_index()
    sel = select_rows(index, correlation_type="both", exclude_antennas=[3])
    np.testing.assert_array_equal(sel.row_indices, [0, 2, 3, 6, 8])


def test_select_rows_by_baselines_is_order_insensitive():
    index = _make_index()
    # (2, 1) reversed -- must still match rows stored as ant1=1, ant2=2.
    sel = select_rows(index, correlation_type="both", baselines=[(2, 1)])
    np.testing.assert_array_equal(sel.row_indices, [0, 3, 8])


def test_select_rows_by_exclude_baselines_is_order_insensitive():
    index = _make_index()
    sel = select_rows(index, correlation_type="both", exclude_baselines=[(2, 1)])
    np.testing.assert_array_equal(sel.row_indices, [1, 2, 4, 5, 6, 7, 9])


def test_select_rows_by_uvdist_range_m():
    index = _make_index()
    # uu_sec=2e-6 -> ~599.6m ; uu_sec=3e-6 -> ~899.4m
    sel = select_rows(index, correlation_type="both", uvdist_range_m=(500.0, 900.0))
    np.testing.assert_array_equal(sel.row_indices, [3, 5])


def test_select_rows_every_nth_is_explicit_and_deterministic():
    index = _make_index()
    sel = select_rows(index, correlation_type="both", every_nth=3)
    np.testing.assert_array_equal(sel.row_indices, [0, 3, 6, 9])


def test_select_rows_random_subset_requires_a_seed():
    index = _make_index()
    with pytest.raises(ValueError):
        select_rows(index, random_subset_n=3)


def test_select_rows_random_subset_is_reproducible_with_the_same_seed():
    index = _make_index()
    a = select_rows(index, correlation_type="both", random_subset_n=4, random_seed=42)
    b = select_rows(index, correlation_type="both", random_subset_n=4, random_seed=42)
    np.testing.assert_array_equal(a.row_indices, b.row_indices)
    assert a.n_rows == 4


def test_select_rows_rejects_both_every_nth_and_random_subset():
    index = _make_index()
    with pytest.raises(ValueError):
        select_rows(index, every_nth=2, random_subset_n=3, random_seed=1)


def test_select_rows_random_subset_larger_than_available_returns_everything():
    index = _make_index()
    sel = select_rows(index, random_subset_n=1000, random_seed=1)  # only 6 cross rows exist
    np.testing.assert_array_equal(sel.row_indices, [0, 1, 3, 5, 7, 8])
