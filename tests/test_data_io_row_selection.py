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


# --- Geometry filters (ha/az/el/parallactic angle): source 1 (RA=180, dec=30)
# and source 2 (RA=60, dec=-10), both at the same JD from GMRT's location,
# give distinct HA/Az/El/PA (computed directly with data_io.astrometry, then
# hardcoded here, so these tests catch a wiring regression in select_rows,
# not a regression in the astrometry formulas themselves -- those are
# covered in test_data_io_astrometry.py):
# source 1: HA=+11.33h, az=348.6, el=-39.9 (below horizon), pa=+12.5
# source 2: HA=-4.67h,  az=106.5, el=+15.1,                 pa=-67.0


def _make_geometry_index():
    from astropy.coordinates import EarthLocation
    import astropy.units as u

    from data_io.source_table import Source

    source_id = np.array([1, 2], dtype=np.int32)
    jd = np.array([2460123.5, 2460123.5])
    ant1 = np.array([1, 1], dtype=np.int16)
    ant2 = np.array([2, 2], dtype=np.int16)
    uu_sec = np.zeros(2, dtype=np.float32)
    vv_sec = np.zeros(2, dtype=np.float32)

    index = RowIndex(
        path="synthetic", gcount=2, pcount=8,
        source_id=source_id, jd=jd, ant1=ant1, ant2=ant2, uu_sec=uu_sec, vv_sec=vv_sec,
        source_ranges={1: [(0, 1)], 2: [(1, 2)]},
        integration_boundaries=np.array([0, 2]),
        id_to_name={1: "source-1", 2: "source-2"},
        chan_freqs_hz=np.array([100e6]), stokes_labels=["RR"],
        data_axis_lengths=[3, 1, 1, 1], data_axis_types=["COMPLEX", "STOKES", "FREQ", "IF"],
        data_offset=0, build_time_sec=0.0,
    )

    def _source(sid, name, ra, dec):
        return Source(
            id=sid, name=name, ra_epoch_deg=ra, dec_epoch_deg=dec,
            ra_apparent_deg=ra, dec_apparent_deg=dec, epoch_year=2000.0,
            calcode="", flux_i_jy=(), flux_q_jy=(), flux_u_jy=(), flux_v_jy=(),
        )

    source_table = {
        1: _source(1, "source-1", 180.0, 30.0),
        2: _source(2, "source-2", 60.0, -10.0),
    }
    array_location = EarthLocation.from_geodetic(lon=74.050 * u.deg, lat=19.093 * u.deg, height=640 * u.m)
    return index, source_table, array_location


def test_select_rows_geometry_filters_require_source_table_and_array_location():
    index, _, _ = _make_geometry_index()
    with pytest.raises(ValueError, match="source_table"):
        select_rows(index, correlation_type="both", ha_range_hours=(0.0, 1.0))


def test_select_rows_geometry_filter_raises_if_source_table_missing_a_present_source():
    index, source_table, array_location = _make_geometry_index()
    del source_table[2]
    with pytest.raises(ValueError, match=r"\[2\]"):
        select_rows(
            index, correlation_type="both", ha_range_hours=(0.0, 1.0),
            source_table=source_table, array_location=array_location,
        )


def test_select_rows_by_ha_range_hours():
    index, source_table, array_location = _make_geometry_index()
    sel = select_rows(
        index, correlation_type="both", ha_range_hours=(10.0, 12.0),
        source_table=source_table, array_location=array_location,
    )
    np.testing.assert_array_equal(sel.row_indices, [0])  # source 1 only


def test_select_rows_by_el_range_deg():
    index, source_table, array_location = _make_geometry_index()
    sel = select_rows(
        index, correlation_type="both", el_range_deg=(0.0, 90.0),
        source_table=source_table, array_location=array_location,
    )
    np.testing.assert_array_equal(sel.row_indices, [1])  # source 2 only -- source 1 is below the horizon


def test_select_rows_by_az_range_deg():
    index, source_table, array_location = _make_geometry_index()
    sel = select_rows(
        index, correlation_type="both", az_range_deg=(340.0, 360.0),
        source_table=source_table, array_location=array_location,
    )
    np.testing.assert_array_equal(sel.row_indices, [0])  # source 1 only


def test_select_rows_by_parallactic_angle_range_deg():
    index, source_table, array_location = _make_geometry_index()
    sel = select_rows(
        index, correlation_type="both", parallactic_angle_range_deg=(-80.0, -50.0),
        source_table=source_table, array_location=array_location,
    )
    np.testing.assert_array_equal(sel.row_indices, [1])  # source 2 only


def test_cyclic_range_mask_wraps_when_the_low_bound_exceeds_the_high_bound():
    from data_io.row_selection import _cyclic_range_mask

    values = np.array([-11.9, 0.0, 11.9])
    mask = _cyclic_range_mask(values, (10.0, -10.0))
    np.testing.assert_array_equal(mask, [True, False, True])
