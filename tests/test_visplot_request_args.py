import pytest

from visplot.request_args import (
    parse_plot_names,
    parse_quantity_pair,
    resolve_antennas_arg,
    resolve_channels_arg,
    resolve_dpi_arg,
    resolve_figure_size_arg,
    resolve_klambda_range_arg,
    resolve_stokes_products,
    resolve_time_range_arg,
    resolve_uvdist_range_arg,
    resolve_plain_range_arg,
    resolve_range_arg,
    validate_colorize_by,
)
from data_io.antenna_table import Antenna

ANTENNAS = [
    Antenna(station_number=1, name="C00:01", x_m=0.0, y_m=0.0, z_m=0.0),
    Antenna(station_number=25, name="W01:25", x_m=0.0, y_m=0.0, z_m=0.0),
]


def test_parse_plot_names_splits_on_comma():
    assert parse_plot_names("antenna-layout, amp-vs-time") == ["antenna-layout", "amp-vs-time"]


def test_parse_plot_names_rejects_empty():
    with pytest.raises(ValueError, match="requires at least one"):
        parse_plot_names("")


def test_parse_quantity_pair_splits_on_the_literal_separator():
    assert parse_quantity_pair("amp-vs-freq") == ("amp", "freq")
    assert parse_quantity_pair("uu-vs-vv") == ("uu", "vv")


def test_names_that_carried_a_unit_and_the_single_letter_baseline_names_are_unknown():
    # removed 2026-10-04 (pre-release, no aliases): a unit is given with --x-unit / --y-unit
    for name in ("amp-vs-freq_mhz", "amp-vs-time_h", "vv-vs-u"):
        with pytest.raises(ValueError, match="unknown quantity"):
            parse_plot_names(name)


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


def test_a_selection_range_takes_its_unit_explicitly_and_converts_to_select_rows_units():
    assert resolve_range_arg("--ha-range", "-6:6 h") == (-6.0, 6.0)
    assert resolve_range_arg("--ha-range", "-90:90deg") == pytest.approx((-6.0, 6.0))  # a space is optional
    assert resolve_range_arg("--el-range", "0:0.5 rad") == pytest.approx((0.0, 28.6478897565))
    assert resolve_range_arg("--az-range", "350:10 deg") == (350.0, 10.0)  # wraps through north
    assert resolve_range_arg("--el-range", None) is None


@pytest.mark.parametrize("option, spec, message", [
    ("--ha-range", "-6:6", r"give the unit, one of: h, deg, rad"),
    ("--el-range", "0:30", r"give the unit, one of: deg, rad"),
    ("--el-range", "0:30 h", r"unit 'h' does not apply; one of: deg, rad"),
    ("--el-range", "0:30 DEG", r"unit 'DEG' does not apply"),  # units are case-sensitive
    ("--el-range", "0:95 deg", r"el lies in -90\.\.90 deg; 95 is outside it"),
    ("--ha-range", "-13:0 h", r"ha lies in -12\.\.12 h"),
    ("--az-range", "0:400 deg", r"az lies in 0\.\.360 deg"),
    ("--el-range", "30:10 deg", r"the lower end 30 is above the upper end 10"),  # elevation does not wrap
    ("--el-range", "0-30 deg", r"write 'LO:HI UNIT'"),
])
def test_a_selection_range_without_its_unit_or_outside_its_domain_is_refused(option, spec, message):
    with pytest.raises(ValueError, match=message):
        resolve_range_arg(option, spec)


def test_resolve_stokes_products_returns_none_when_not_given():
    assert resolve_stokes_products(None, ["RR", "LL"]) is None


def test_resolve_stokes_products_matches_case_insensitively_in_the_files_order():
    plan = resolve_stokes_products("ll,rr", ["RR", "LL"])
    assert plan.labels == ("RR", "LL") and plan.read == (0, 1) and not plan.derived


def test_resolve_stokes_products_raises_without_a_stokes_axis():
    with pytest.raises(ValueError, match="no STOKES axis"):
        resolve_stokes_products("RR", None)


def test_resolve_stokes_products_refuses_any_product_it_cannot_give():
    # the resolver it replaced kept RR here and dropped XX without a word
    with pytest.raises(ValueError, match=r"--stokes 'RR,XX': Stokes XX: not among this file's products"):
        resolve_stokes_products("RR,XX", ["RR", "LL"])
    with pytest.raises(ValueError, match=r"Stokes U of circular feeds needs RL, LR"):
        resolve_stokes_products("U", ["RR", "LL"])


def test_resolve_stokes_products_derives_i_q_u_v():
    plan = resolve_stokes_products("RR,V", ["RR", "LL"])
    assert plan.labels == ("RR", "V") and plan.read == (0, 1) and plan.derived


def test_parse_plot_names_rejects_a_bare_quantity_and_suggests_pairs():
    with pytest.raises(ValueError, match=r"is a quantity; a plot needs two.*amp-vs-uvdist"):
        parse_plot_names("uvdist")


def test_parse_plot_names_rejects_an_unknown_quantity_in_a_pair():
    with pytest.raises(ValueError, match=r"unknown quantity 'ampl' in plot 'ampl-vs-freq'"):
        parse_plot_names("ampl-vs-freq")


def test_parse_plot_names_rejects_an_unrecognized_name():
    with pytest.raises(ValueError, match="unrecognized plot name 'uv-coverage'"):
        parse_plot_names("antenna-layout,uv-coverage")


def test_validate_colorize_by_accepts_only_categories():
    validate_colorize_by("stokes")
    validate_colorize_by("source")
    with pytest.raises(ValueError, match="takes a category"):
        validate_colorize_by("amp")


def test_parse_plot_names_accepts_presets_and_geometry_quantities():
    assert parse_plot_names("ha-range,el-vs-time") == ["ha-range", "el-vs-time"]


def test_resolve_plain_range_arg():
    assert resolve_plain_range_arg(None) is None
    assert resolve_plain_range_arg("0:250") == (0.0, 250.0)


def test_parse_plot_names_accepts_quantity_names_and_earlier_names_with_units():
    assert parse_plot_names("amp-vs-uvdist,vv-vs-uu,amp-vs-uvdist,phase-vs-time") == [
        "amp-vs-uvdist", "vv-vs-uu", "amp-vs-uvdist", "phase-vs-time"]


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
