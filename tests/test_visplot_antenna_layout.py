import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pytest
from astropy.coordinates import EarthLocation
import astropy.units as u

from data_io.antenna_table import Antenna
from visplot.antenna_layout import (
    _choose_inset_corner,
    _detect_compact_core_mask,
    _ecef_offsets_to_enu_m,
    _largest_relative_gap_threshold,
    _nearest_neighbor_distances_m,
    _place_labels_without_overlap,
    antenna_layout,
)


def test_ecef_offsets_to_enu_m_at_the_equator_prime_meridian():
    # At lat=0, lon=0: ECEF +Z is north, +Y is east (ECEF +X is up, not
    # exercised here) -- a fixed, checkable case for the rotation formula.
    east_m, north_m = _ecef_offsets_to_enu_m(dx_m=0.0, dy_m=500.0, dz_m=1000.0, lat_rad=0.0, lon_rad=0.0)
    assert east_m == pytest.approx(500.0)
    assert north_m == pytest.approx(1000.0)


def test_antenna_layout_places_antennas_at_their_local_enu_offsets():
    location = EarthLocation.from_geodetic(lon=0 * u.deg, lat=0 * u.deg, height=0 * u.m)
    ref_x_m, ref_y_m, ref_z_m = (v.to_value("m") for v in location.to_geocentric())

    antennas = [
        Antenna(station_number=1, name="A:01", x_m=ref_x_m, y_m=ref_y_m, z_m=ref_z_m),
        Antenna(station_number=2, name="A:02", x_m=ref_x_m, y_m=ref_y_m + 500.0, z_m=ref_z_m + 1000.0),
    ]

    fig = antenna_layout(antennas, location)
    ax = fig.axes[0]
    offsets = ax.collections[0].get_offsets()
    np.testing.assert_allclose(offsets, [[0.0, 0.0], [500.0, 1000.0]], atol=1e-6)
    assert {t.get_text() for t in ax.texts} == {"A:01", "A:02"}
    plt.close(fig)


def test_antenna_layout_title_defaults_to_a_plain_title_without_provenance():
    location = EarthLocation.from_geodetic(lon=0 * u.deg, lat=0 * u.deg, height=0 * u.m)
    ref_x_m, ref_y_m, ref_z_m = (v.to_value("m") for v in location.to_geocentric())
    antennas = [Antenna(station_number=1, name="A:01", x_m=ref_x_m, y_m=ref_y_m, z_m=ref_z_m)]

    fig = antenna_layout(antennas, location)
    assert fig.axes[0].get_title() == "Antenna layout"
    plt.close(fig)


def test_antenna_layout_title_includes_telescope_and_source_file():
    location = EarthLocation.from_geodetic(lon=0 * u.deg, lat=0 * u.deg, height=0 * u.m)
    ref_x_m, ref_y_m, ref_z_m = (v.to_value("m") for v in location.to_geocentric())
    antennas = [Antenna(station_number=1, name="A:01", x_m=ref_x_m, y_m=ref_y_m, z_m=ref_z_m)]

    fig = antenna_layout(
        antennas, location, telescope="GMRT", source_path="/data1/gmrt/some_obs/40_014_25jul2021_2.6s_gwb.FITS",
    )
    assert fig.axes[0].get_title() == "Antenna layout: GMRT\n(file: 40_014_25jul2021_2.6s_gwb.FITS)"
    plt.close(fig)


def test_nearest_neighbor_distances_m_finds_the_closest_other_point():
    east_m = np.array([0.0, 1.0, 10.0])
    north_m = np.array([0.0, 0.0, 0.0])
    np.testing.assert_allclose(_nearest_neighbor_distances_m(east_m, north_m), [1.0, 1.0, 9.0])


def test_largest_relative_gap_threshold_finds_a_genuine_gap():
    # 4 values around 10-13, 3 values around 500-510 -- a clear split
    # regardless of the values' own absolute scale.
    values = np.array([10.0, 12.0, 11.0, 13.0, 500.0, 510.0, 505.0])
    assert _largest_relative_gap_threshold(values) == pytest.approx((13.0 + 500.0) / 2.0)


def test_largest_relative_gap_threshold_returns_none_for_uniform_log_spacing():
    # A constant 1.2x ratio between consecutive values -- normal log-spaced
    # placement, not a genuine split.
    values = 10.0 * 1.2 ** np.arange(10)
    assert _largest_relative_gap_threshold(values) is None


def test_largest_relative_gap_threshold_returns_none_when_too_few_values_precede_the_gap():
    # Only 2 values before the huge jump -- below the default min_count=3.
    values = np.array([10.0, 11.0, 5000.0, 5010.0])
    assert _largest_relative_gap_threshold(values) is None


def _make_core_plus_isolated_arms_antennas():
    # 6 antennas packed within a few metres of each other (a compact core --
    # each one's nearest neighbour is another core antenna, a few metres
    # away) plus 3 antennas each individually isolated ~500m out (each
    # one's nearest neighbour, core or another arm antenna, is still
    # ~500m away) -- mirrors GMRT's own structure of a dense central
    # square plus individually spaced-out arm antennas.
    location = EarthLocation.from_geodetic(lon=0 * u.deg, lat=0 * u.deg, height=0 * u.m)
    ref_x_m, ref_y_m, ref_z_m = (v.to_value("m") for v in location.to_geocentric())
    core_offsets = [(0.0, 0.0), (5.0, 0.0), (0.0, 5.0), (-5.0, 0.0), (0.0, -5.0), (3.0, 3.0)]
    arm_offsets = [(500.0, 0.0), (0.0, 500.0), (-500.0, 0.0)]
    antennas = [
        Antenna(station_number=i, name=f"C{i}", x_m=ref_x_m, y_m=ref_y_m + e, z_m=ref_z_m + n)
        for i, (e, n) in enumerate(core_offsets)
    ] + [
        Antenna(station_number=100 + i, name=f"A{i}", x_m=ref_x_m, y_m=ref_y_m + e, z_m=ref_z_m + n)
        for i, (e, n) in enumerate(arm_offsets)
    ]
    return antennas, location


def test_detect_compact_core_mask_separates_a_dense_core_from_isolated_arms():
    antennas, location = _make_core_plus_isolated_arms_antennas()
    ref_x_m, ref_y_m, ref_z_m = (v.to_value("m") for v in location.to_geocentric())
    east_m = np.array([a.y_m - ref_y_m for a in antennas])
    north_m = np.array([a.z_m - ref_z_m for a in antennas])
    mask = _detect_compact_core_mask(east_m, north_m)
    assert mask.tolist() == [True] * 6 + [False] * 3


def test_antenna_layout_adds_an_inset_when_a_compact_core_is_detected():
    antennas, location = _make_core_plus_isolated_arms_antennas()
    fig = antenna_layout(antennas, location)
    ax = fig.axes[0]
    assert len(ax.child_axes) == 1
    axins = ax.child_axes[0]
    xlo, xhi = axins.get_xlim()
    ylo, yhi = axins.get_ylim()
    assert xhi - xlo < 50.0  # zoomed to the core's own few-metre scale, not the ~500m full array
    assert yhi - ylo < 50.0
    assert {t.get_text() for t in axins.texts} == {"C0", "C1", "C2", "C3", "C4", "C5"}
    plt.close(fig)


def test_antenna_layout_has_no_inset_without_a_compact_core():
    # 8 antennas evenly spaced on a ring -- every antenna's nearest
    # neighbour is the same distance away, so there is no density split.
    location = EarthLocation.from_geodetic(lon=0 * u.deg, lat=0 * u.deg, height=0 * u.m)
    ref_x_m, ref_y_m, ref_z_m = (v.to_value("m") for v in location.to_geocentric())
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    antennas = [
        Antenna(
            station_number=i, name=f"A:{i}",
            x_m=ref_x_m, y_m=ref_y_m + 100.0 * np.cos(a), z_m=ref_z_m + 100.0 * np.sin(a),
        )
        for i, a in enumerate(angles)
    ]
    fig = antenna_layout(antennas, location)
    ax = fig.axes[0]
    assert len(ax.child_axes) == 0
    plt.close(fig)


def test_choose_inset_corner_picks_the_corner_with_fewest_outside_antennas():
    fig, ax = plt.subplots()
    ax.set_xlim(0.0, 100.0)
    ax.set_ylim(0.0, 100.0)
    # Points crowd the lower-left, lower-right, and upper-left corners;
    # upper-right is empty and should be chosen.
    outside_e = np.array([5.0, 95.0, 5.0])
    outside_n = np.array([5.0, 5.0, 95.0])
    corner = _choose_inset_corner(ax, outside_e, outside_n)
    assert corner == (0.52, 0.52)
    plt.close(fig)


def test_place_labels_without_overlap_keeps_close_labels_from_colliding():
    fig, ax = plt.subplots()
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Two points 0.05 data-units apart -- close enough that a naive fixed
    # offset would put both labels in the same spot.
    _place_labels_without_overlap(ax, renderer, ["AAAAA", "BBBBB"], [0.0, 0.05], [0.0, 0.0], fontsize=10)

    bboxes = [t.get_window_extent(renderer=renderer) for t in ax.texts]
    assert not bboxes[0].overlaps(bboxes[1])
    plt.close(fig)


def test_place_labels_without_overlap_routes_around_avoid_bboxes():
    fig, ax = plt.subplots()
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # An obstruction bbox covering the entire axes -- every candidate offset
    # for the one label collides with it, so the fallback (still avoiding
    # overlap with other *labels*, of which there are none here) is used
    # rather than raising or silently ignoring the obstruction.
    full_axes_bbox = ax.get_window_extent(renderer=renderer)
    _place_labels_without_overlap(ax, renderer, ["ONLY"], [0.0], [0.0], fontsize=10, avoid_bboxes=[full_axes_bbox])

    assert len(ax.texts) == 1  # exactly one label kept, no leftover fallback duplicates
    plt.close(fig)
