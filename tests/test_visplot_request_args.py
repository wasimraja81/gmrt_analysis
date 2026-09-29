import pytest

from visplot.request_args import (
    parse_plot_names,
    parse_quantity_pair,
    resolve_antennas_arg,
    resolve_channels_arg,
    resolve_deg_range_arg,
    resolve_dpi_arg,
    resolve_figure_size_arg,
    resolve_ha_range_arg,
    resolve_klambda_range_arg,
    resolve_stokes_axis_selection,
    resolve_time_range_arg,
    resolve_uvdist_range_arg,
    resolve_plain_range_arg,
    validate_colorize_by,
)
from data_io.antenna_table import Antenna

ANTENNAS = [
    Antenna(station_number=1, name="C00:01", x_m=0.0, y_m=0.0, z_m=0.0),
    Antenna(station_number=25, name="W01:25", x_m=0.0, y_m=0.0, z_m=0.0),
]


def test_parse_plot_names_splits_on_comma():
    assert parse_plot_names("antenna-layout, amp-vs-time_h") == ["antenna-layout", "amp-vs-time_h"]


def test_parse_plot_names_rejects_empty():
    with pytest.raises(ValueError, match="requires at least one"):
        parse_plot_names("")


def test_parse_quantity_pair_splits_on_the_literal_separator():
    assert parse_quantity_pair("amp-vs-freq_mhz") == ("amp", "freq_mhz")
    assert parse_quantity_pair("u_klambda-vs-v_klambda") == ("u_klambda", "v_klambda")


def test_parse_quantity_pair_rejects_a_name_with_no_separator():
    with pytest.raises(ValueError, match="not a recognized plot name"):
        parse_quantity_pair("garbage")


def test_resolve_antennas_arg_returns_none_when_not_given():
    assert resolve_antennas_arg(None, ANTENNAS) is None


def test_resolve_antennas_arg_resolves_a_gmrt_prefix():
    assert resolve_antennas_arg("C00", ANTENNAS) == [1]


def test_resolve_channels_arg_returns_none_when_not_given():
    assert resolve_channels_arg(None, [100e6, 101e6]) is None


def test_resolve_channels_arg_resolves_a_plain_index_range():
    assert resolve_channels_arg("0:1", [100e6, 101e6, 102e6]) == [0, 1]


def test_resolve_time_range_arg_returns_none_when_not_given():
    assert resolve_time_range_arg(None, 2460123.5) is None


def test_resolve_time_range_arg_resolves_relative_hours():
    lo, hi = resolve_time_range_arg("0:1", 2460123.5)
    assert lo == pytest.approx(2460123.5)
    assert hi == pytest.approx(2460123.5 + 1.0 / 24.0)


def test_resolve_uvdist_range_arg_defaults_to_metres():
    assert resolve_uvdist_range_arg("0:5000") == (0.0, 5000.0)


def test_resolve_uvdist_range_arg_converts_km():
    assert resolve_uvdist_range_arg("0:5km") == (0.0, 5000.0)


def test_resolve_klambda_range_arg_is_a_plain_number_range():
    assert resolve_klambda_range_arg("0:5") == (0.0, 5.0)


def test_resolve_ha_range_arg_defaults_to_hours():
    assert resolve_ha_range_arg("-6:6") == (-6.0, 6.0)


def test_resolve_ha_range_arg_converts_degrees():
    lo, hi = resolve_ha_range_arg("-90:90deg")
    assert lo == pytest.approx(-6.0)
    assert hi == pytest.approx(6.0)


def test_resolve_deg_range_arg_defaults_to_degrees():
    assert resolve_deg_range_arg("0:90") == (0.0, 90.0)


def test_resolve_stokes_axis_selection_returns_none_when_not_given():
    assert resolve_stokes_axis_selection(None, ["RR", "LL"]) is None


def test_resolve_stokes_axis_selection_matches_case_insensitively():
    assert resolve_stokes_axis_selection("rr", ["RR", "LL"]) == [0]


def test_resolve_stokes_axis_selection_raises_without_a_stokes_axis():
    with pytest.raises(ValueError, match="no STOKES axis"):
        resolve_stokes_axis_selection("RR", None)


def test_resolve_stokes_axis_selection_raises_when_nothing_matches():
    with pytest.raises(ValueError, match="matched none"):
        resolve_stokes_axis_selection("XX", ["RR", "LL"])


def test_parse_plot_names_rejects_a_bare_quantity_and_suggests_pairs():
    with pytest.raises(ValueError, match=r"is a quantity; a plot needs two.*amp-vs-uvdist_klambda"):
        parse_plot_names("uvdist_klambda")


def test_parse_plot_names_rejects_an_unknown_quantity_in_a_pair():
    with pytest.raises(ValueError, match=r"unknown quantity 'ampl' in plot 'ampl-vs-freq_mhz'"):
        parse_plot_names("ampl-vs-freq_mhz")


def test_parse_plot_names_rejects_an_unrecognized_name():
    with pytest.raises(ValueError, match="unrecognized plot name 'uv-coverage'"):
        parse_plot_names("antenna-layout,uv-coverage")


def test_validate_colorize_by_accepts_only_categories():
    validate_colorize_by("stokes")
    validate_colorize_by("source")
    with pytest.raises(ValueError, match="takes a category"):
        validate_colorize_by("amp")


def test_parse_plot_names_accepts_presets_and_geometry_quantities():
    assert parse_plot_names("ha-range,el_deg-vs-time_h") == ["ha-range", "el_deg-vs-time_h"]


def test_resolve_plain_range_arg():
    assert resolve_plain_range_arg(None) is None
    assert resolve_plain_range_arg("0:250") == (0.0, 250.0)


def test_parse_plot_names_accepts_quantity_names_and_earlier_names_with_units():
    assert parse_plot_names("amp-vs-uvdist,v-vs-u,amp-vs-uvdist_klambda,phase_deg-vs-time_h") == [
        "amp-vs-uvdist", "v-vs-u", "amp-vs-uvdist_klambda", "phase_deg-vs-time_h"]


def test_dpi_and_figure_size_are_checked():
    assert resolve_dpi_arg(150) == 150
    for dpi in (49, 2401):
        with pytest.raises(ValueError, match="--dpi needs 50 to 2400"):
            resolve_dpi_arg(dpi)
    assert resolve_figure_size_arg("8,6") == (8.0, 6.0)
    assert resolve_figure_size_arg("10.93,7.42") == (10.93, 7.42)
    for spec in ("8", "8,6,1", "8x6", "0,6", "8,-1", "8,nan", "101,6"):
        with pytest.raises(ValueError, match="--figure-size takes 'W,H' in inches"):
            resolve_figure_size_arg(spec)
