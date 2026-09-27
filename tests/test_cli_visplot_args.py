import pytest

from cli.visplot_args import (
    is_generic_quantity_plot,
    parse_plot_names,
    parse_quantity_pair,
    resolve_antennas_arg,
    resolve_channels_arg,
    resolve_deg_range_arg,
    resolve_ha_range_arg,
    resolve_klambda_range_arg,
    resolve_stokes_axis_selection,
    resolve_time_range_arg,
    resolve_uvdist_range_arg,
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


def test_is_generic_quantity_plot_distinguishes_named_from_generic():
    assert is_generic_quantity_plot("antenna-layout") is False
    assert is_generic_quantity_plot("amp-vs-time_h") is True
    assert is_generic_quantity_plot("not-a-plot-at-all") is False


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
