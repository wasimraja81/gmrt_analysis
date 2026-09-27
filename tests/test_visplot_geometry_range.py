import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np

from visplot.geometry_range import _hours_from_start, az_el_range, hour_angle_range, parallactic_angle_range


def test_hours_from_start_starts_at_zero():
    jd = np.array([100.0, 100.5, 101.0])
    np.testing.assert_allclose(_hours_from_start(jd), [0.0, 12.0, 24.0])


def _synthetic_two_source_data():
    jd = np.array([100.0, 100.1, 100.2, 100.3])
    labels = np.array(["3C286", "3C286", "3C48", "3C48"])
    return jd, labels


def test_hour_angle_range_plots_one_series_per_source_with_legend():
    jd, labels = _synthetic_two_source_data()
    ha_hours = np.array([-2.0, -1.0, 3.0, 4.0])

    fig = hour_angle_range(jd, ha_hours, labels)
    ax = fig.axes[0]

    assert ax.get_ylim() == (-12.0, 12.0)
    assert len(ax.collections) == 2  # one scatter series per source
    legend_labels = {t.get_text() for t in ax.get_legend().get_texts()}
    assert legend_labels == {"3C286", "3C48"}
    plt.close(fig)


def test_hour_angle_range_has_no_legend_for_a_single_source():
    jd = np.array([100.0, 100.1])
    labels = np.array(["3C286", "3C286"])
    ha_hours = np.array([-1.0, 0.0])

    fig = hour_angle_range(jd, ha_hours, labels)
    ax = fig.axes[0]

    assert ax.get_legend() is None
    assert "3C286" in ax.get_title()
    plt.close(fig)


def test_hour_angle_range_title_includes_all_sources_and_provenance():
    jd, labels = _synthetic_two_source_data()
    ha_hours = np.array([-2.0, -1.0, 3.0, 4.0])

    fig = hour_angle_range(jd, ha_hours, labels, telescope="GMRT", source_path="/data/obs.fits")
    ax = fig.axes[0]

    title = ax.get_title()
    assert "3C286" in title and "3C48" in title
    assert "GMRT" in title
    assert "obs.fits" in title
    plt.close(fig)


def test_parallactic_angle_range_sets_the_full_pa_range():
    jd, labels = _synthetic_two_source_data()
    pa_deg = np.array([-170.0, -90.0, 0.0, 170.0])

    fig = parallactic_angle_range(jd, pa_deg, labels)
    ax = fig.axes[0]

    assert ax.get_ylim() == (-180.0, 180.0)
    plt.close(fig)


def test_az_el_range_has_two_panels_with_correct_ylims_and_one_legend():
    jd, labels = _synthetic_two_source_data()
    az_deg = np.array([10.0, 20.0, 300.0, 310.0])
    el_deg = np.array([30.0, 40.0, 50.0, 60.0])

    fig = az_el_range(jd, az_deg, el_deg, labels)
    ax_el, ax_az = fig.axes

    assert ax_el.get_ylim() == (-90.0, 90.0)
    assert ax_az.get_ylim() == (0.0, 360.0)
    assert ax_el.get_legend() is not None
    assert ax_az.get_legend() is None  # avoid a duplicate legend on the second panel
    plt.close(fig)


def test_az_el_range_single_source_title_names_the_source():
    jd = np.array([100.0, 100.1])
    labels = np.array(["3C286", "3C286"])
    az_deg = np.array([10.0, 20.0])
    el_deg = np.array([30.0, 40.0])

    fig = az_el_range(jd, az_deg, el_deg, labels)

    assert "3C286" in fig._suptitle.get_text()
    plt.close(fig)
