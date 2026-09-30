import argparse
import os

import pytest

from visplot.request import PATH_OPTIONS, PlotRequest, build_arg_parser, request_actions, request_option_names


def _every_kind_of_value():
    return PlotRequest(
        "tmp/obs.fits", "amp-vs-uvdist,ha-range", sources="3C286,3C48", correlation_type="both",
        antennas="1:5,W01", baselines_with="C01", exclude_antennas="C00",
        time_range="2021-07-25T17:00:00/2021-07-25T18:00:00",
        uvdist_range="0:5km", u_range_klambda="-5:5", ha_range="-2:2", every_nth=3, channels="400:450MHz",
        stokes="RR,LL", point_size=0.25, color="k", colorize_by="stokes", show_flagged=True, mirror=True,
        x_unit="Mlambda", y_unit="UNCALIB", time_zone="Asia/Kolkata", x_range="0:40", y_range_mode="percentile",
        range_percentiles="1:99", y_scale="asinh", aspect="free", scale_linear_width=0.1, output_dir="tmp/out",
        output_prefix="run1", no_highres_pdf=True, dpi=300, figure_size="10.5,7.25", cache_dir="tmp/cache", threads=3,
        panel_font="dejavu-sans", plot_theme="dark",
    )


def test_a_request_read_back_from_its_command_line_is_the_same_request():
    request = _every_kind_of_value()
    assert PlotRequest.from_argv(request.to_argv()) == request
    default = PlotRequest("tmp/obs.fits", "v-vs-u")
    assert PlotRequest.from_argv(default.to_argv()) == default


def test_the_command_line_states_every_option_with_a_value_defaults_included():
    argv = PlotRequest("tmp/obs.fits", "v-vs-u").to_argv()
    parser = build_arg_parser()
    for action in request_actions(parser):
        has_value = action.default is not None and not isinstance(action, argparse._StoreTrueAction)
        if action.option_strings and has_value:
            assert action.option_strings[-1] in argv, action.dest
    assert argv[argv.index("--threads") + 1] == str(parser.get_default("threads"))


def test_paths_are_held_absolute():
    request = _every_kind_of_value()
    for name in PATH_OPTIONS:
        value = getattr(request, name)
        assert value is None or os.path.isabs(value), name


def test_the_request_has_exactly_the_parsers_options():
    parser = build_arg_parser()
    dests = {a.dest for a in parser._actions} - {"help", "clear_cache"}
    assert set(request_option_names()) == dests
    assert set(PlotRequest("x.fits", "v-vs-u").as_dict()) == dests
    with pytest.raises(TypeError, match="not visplot options"):
        PlotRequest("x.fits", "v-vs-u", colour="red")


def test_a_request_is_immutable_and_replace_makes_a_changed_copy():
    request = PlotRequest("x.fits", "v-vs-u")
    with pytest.raises(AttributeError):
        request.mirror = True
    mirrored = request.replace(mirror=True)
    assert mirrored.mirror and not request.mirror
    assert "--mirror" in mirrored.command_line() and "--mirror" not in request.command_line()


def test_an_option_with_a_default_cannot_be_left_without_a_value():
    with pytest.raises(ValueError, match="--range-percentiles needs a value"):
        PlotRequest("x.fits", "v-vs-u", range_percentiles=None)
