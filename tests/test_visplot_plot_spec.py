import math

import pytest

from visplot.plot_spec import PlotSpec, expand_plot_name
from visplot.quantities import QuantityContext

CTX = QuantityContext(time_reference_jd=2459421.2, bunit="JY")


def test_an_alias_becomes_its_quantity_and_unit():
    assert PlotSpec(y="amp", x="uvdist_klambda") == PlotSpec(y="amp", x="uvdist", x_unit="klambda")
    assert PlotSpec(y="v_klambda", x="u_klambda").y_unit == "klambda"
    with pytest.raises(ValueError, match="'u_sec' is no longer available"):
        PlotSpec(y="amp", x="u_sec")


def test_a_unit_given_by_its_label_is_stored_by_name():
    assert PlotSpec(y="v", x="u", x_unit="kλ", y_unit="Mλ") == PlotSpec(y="v", x="u", x_unit="klambda", y_unit="Mlambda")


def test_an_alias_with_a_different_unit_is_rejected():
    with pytest.raises(ValueError, match="'uvdist_klambda' is 'uvdist' in klambda"):
        PlotSpec(y="amp", x="uvdist_klambda", x_unit="m")
    assert PlotSpec(y="amp", x="uvdist_klambda", x_unit="kλ").x_unit == "klambda"  # the same unit is fine


def test_a_unit_that_does_not_apply_is_rejected_with_the_choices():
    with pytest.raises(ValueError, match="choose one of: recorded, UTC, local, LST, h, min, s"):
        PlotSpec(y="amp", x="time", x_unit="klambda")


def test_clock_time_needs_a_linear_axis_and_no_mirroring():
    with pytest.raises(ValueError, match="needs a linear x scale"):
        PlotSpec(y="amp", x="time", x_unit="UTC", x_scale="log")
    with pytest.raises(ValueError, match="has no zero"):
        PlotSpec(y="amp", x="time", x_unit="LST", mirror=True)


def test_visibility_units_are_checked_against_the_files_bunit():
    plot = PlotSpec(y="amp", x="time", y_unit="mJy")  # accepted until the file's BUNIT is known
    plot.check_units(CTX)
    with pytest.raises(ValueError, match="not a flux density"):
        plot.check_units(QuantityContext(time_reference_jd=2459421.2, bunit="UNCALIB"))


def test_with_units_names_every_default():
    plot = PlotSpec(y="amp", x="uvdist").with_units(CTX)
    assert (plot.y_unit, plot.x_unit) == ("JY", "klambda")


def test_equal_aspect_needs_the_same_unit():
    assert PlotSpec(y="v", x="u").equal_aspect
    assert PlotSpec(y="v", x="u", x_unit="klambda").equal_aspect  # the default named
    assert PlotSpec(y="v", x="u", x_unit="m", y_unit="m").equal_aspect
    assert not PlotSpec(y="v", x="u", x_unit="m").equal_aspect  # m against kλ


def test_a_preset_takes_the_units_with_its_range_and_lines_converted():
    style = PlotSpec(y="", x="", y_unit="deg", x_unit="UTC")
    (ha,) = expand_plot_name("ha-range", style)
    assert (ha.y_unit, ha.x_unit) == ("deg", "UTC")
    assert ha.y_range == pytest.approx((-180.0, 180.0))
    assert ha.reference_lines == (("y", 0.0),)
    el, az = expand_plot_name("az-el-range", PlotSpec(y="", x="", y_unit="rad"))
    assert el.y_range == pytest.approx((0.0, math.pi / 2))  # from the horizon (T47)
    assert az.y_range == pytest.approx((0.0, 2 * math.pi))
    assert (el.name, az.name) == ("az-el-range_el", "az-el-range_az")


def test_a_preset_rejects_a_unit_its_quantity_does_not_have():
    with pytest.raises(ValueError, match="does not apply to 'ha'"):
        expand_plot_name("ha-range", PlotSpec(y="", x="", y_unit="klambda"))


def test_a_stack_is_two_plots_on_one_x_axis_the_lower_with_its_own_y_axis():
    from visplot.request import PlotRequest
    from visplot.run import RequestError, check_request

    checked = check_request(PlotRequest("obs.fits", "amp-vs-time+phase-vs-time", y2_unit="rad", y2_range="-3:3"))
    (stack,) = checked.plots_by_name.values()
    assert [(p.y, p.x) for p in stack] == [("amp", "time"), ("phase", "time")]
    assert (stack[1].y_unit, stack[1].y_range) == ("rad", (-3.0, 3.0)) and stack[0].y_range is None
    (el, az) = check_request(PlotRequest("obs.fits", "az-el-range", y2_unit="rad")).plots_by_name["az-el-range"]
    assert (el.y, el.y_unit, az.y, az.y_unit) == ("el", "deg", "az", "rad")  # the preset's lower plot takes --y2-*
    for plots, options, message in (
        ("amp-vs-time+phase-vs-time+real-vs-time", {}, "a stack holds two plots"),
        ("amp-vs-time+phase-vs-freq", {}, "stacked plots share their x axis"),
        ("antenna-layout+amp-vs-time", {}, "only streamed plots stack"),
        ("az-el-range+ha-range", {}, "a stack holds two plots"),
        ("amp-vs-time", {"y2_unit": "rad"}, "--y2-unit set the lower plot of a stack"),
    ):
        with pytest.raises(RequestError, match=message):
            check_request(PlotRequest("obs.fits", plots, **options))
