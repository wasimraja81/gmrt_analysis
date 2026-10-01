"""The panel under a plot (T37): the color key and what the plot shows."""

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest
from matplotlib.colors import to_rgba

from conftest import make_scratch_dir
from test_cli_visplot_output import _make_synthetic_file
from visplot.plot_panel import FLAGGED_COLOR, category_palette, selection_filters
from visplot.plot_spec import PlotSpec
from visplot.request import PlotRequest
from visplot.run import prepare
from visplot.stream import FLAGGED_LAYER
from visplot.xy_figure import cross_offsets, draw_markers, marker_offsets


def _reds_and_pinks():
    tab10, tab20 = matplotlib.colormaps["tab10"].colors, matplotlib.colormaps["tab20"].colors
    return {tab10[3], tab10[6], tab20[6], tab20[7], tab20[12], tab20[13]}


def test_category_palettes_leave_out_red_and_pink_which_mark_flags():
    assert len(category_palette(8)) == 8 and len(category_palette(9)) == 16
    for n in (2, 8, 9, 16):
        assert not set(category_palette(n)) & _reds_and_pinks()


def test_the_selection_filters_read_as_text():
    request = PlotRequest("x.fits", "amp-vs-time", uvdist_range="0:5km", ha_range="-2:2", every_nth=40,
                          random_subset_n=1000, random_seed=3)
    assert selection_filters(request) == "uv distance 0:5km, hour angle -2:2, every 40th row, 1,000 random rows, seed 3"
    assert [selection_filters(PlotRequest("x.fits", "v-vs-u", every_nth=n)) for n in (1, 2, 3, 11, 21, 112)] == [
        "", "every 2nd row", "every 3rd row", "every 11th row", "every 21st row", "every 112th row"]


def _run(name, **options):
    path = _make_synthetic_file(make_scratch_dir(name) / "obs.fits")
    run = prepare(PlotRequest(str(path), options.pop("plots", "amp-vs-freq"), **options))
    (plot,) = run.xy_plots
    return run, run.xy_figures[plot]


def test_the_panel_states_what_the_selection_holds():
    run, figure = _run("panel_facts", stokes="LL", channels="0:1", sources="3C286")
    facts, index = figure.panel.facts, run.file.index
    rows = run.selection.row_indices
    cross = {(int(a), int(b)) for a, b in zip(index.ant1[rows], index.ant2[rows]) if a != b}
    assert facts.stokes == ("LL",)
    assert facts.n_channels == 2 and facts.n_channels_in_file == len(index.chan_freqs_hz)
    assert facts.freq_mhz == pytest.approx(tuple(np.asarray(index.chan_freqs_hz[:2]) / 1e6))
    assert facts.n_baselines == len(cross)
    assert [name for _, name in facts.sources] == ["3C286"]
    assert facts.time_utc[0] <= facts.time_utc[1]
    assert facts.filters == ""


def test_the_key_is_fixed_before_drawing_and_matches_the_images_colors():
    run, figure = _run("panel_key", colorize_by="stokes")
    entries = figure.panel.key_entries()
    assert [label for label, _ in entries] == ["RR", "LL"]
    colors = figure.panel.colors()
    assert [colors[code + 1] for code in (0, 1)] == [rgba for _, rgba in entries]
    assert colors[FLAGGED_LAYER] == to_rgba(FLAGGED_COLOR)


def test_the_panel_keeps_its_height_when_the_figure_grows():
    _, figure = _run("panel_resize")
    below_axes_in = []
    for height_in in (7.0, 11.0):
        figure.fig.set_size_inches(8.0, height_in)
        figure.relayout()
        below_axes_in.append(figure.ax.get_position().y0 * height_in)
    assert below_axes_in[0] == pytest.approx(below_axes_in[1])


def test_the_run_names_its_record_on_every_plot():
    run, figure = _run("panel_record", plots="amp-vs-freq")
    run.set_record("20260929T000000Z_abcdef12")
    assert figure.record_id == "20260929T000000Z_abcdef12"


def test_flagged_samples_are_crosses_and_the_key_says_when_none_are_drawn():
    layers = np.zeros((9, 9), dtype=np.int16)
    layers[4, 4] = FLAGGED_LAYER
    layers[1, 1] = 1
    out = draw_markers(layers, marker_offsets(0.5, square=True), cross_offsets(0.5))
    flagged = {(int(y), int(x)) for y, x in zip(*np.nonzero(out == FLAGGED_LAYER))}
    # one-pixel markers: the smallest cross, 3 x 3 px
    assert flagged == {(4 + d, 4 + d) for d in range(-1, 2)} | {(4 + d, 4 - d) for d in range(-1, 2)}
    assert out[1, 1] == 1 and (out == 1).sum() == 1  # the one-pixel sample stays a pixel
    # larger markers: the cross reaches as far as the marker does
    radius = 2.08
    assert max(abs(dy) for dy, _ in cross_offsets(radius)) == max(abs(dy) for dy, _ in marker_offsets(radius, False))

    _, figure = _run("panel_flagged_key", show_flagged=True)
    figure.panel.set_flagged_drawn(False)
    assert figure.panel.flagged.get_text() == "flagged: none"


def test_a_long_key_wraps_between_entries_within_its_column():
    from matplotlib.offsetbox import DrawingArea

    from visplot.plot_panel import _swatch, _text_width_pt

    _, figure = _run("panel_wrap", colorize_by="source")
    names = ["3C286", "3C48", "MOON0520", "MOON0625", "DA240", "B1929+10", "Cas-A"]
    items = [x for name in names for x in (_swatch("tab:blue"), f"{name}   ")]
    width_pt = 260.0
    lines = figure.panel._wrap("Sources", items, width_pt)
    assert len(lines) > 1 and lines[0][0] == "Sources" and all(label == "" for label, _ in lines[1:])
    shown = []
    for _, line in lines:
        assert isinstance(line[0], DrawingArea) and not isinstance(line[-1], DrawingArea)
        texts = [i for i in line if isinstance(i, str)]
        shown += texts
        room = width_pt - figure.panel.label_width_pt - 8
        assert sum(_text_width_pt(t) + 4 for t in texts) + (9 + 4) * len(texts) <= room
    assert [t.strip() for t in shown] == names


def test_the_panel_knows_which_axes_a_plot_varies_along_as_the_stream_shapes_it():
    """`varies_along` (for the panel's words, before drawing) agrees with the
    arrays `ChunkValues` builds, for every quantity in every unit."""
    from test_visplot_stream import CTX as STREAM_CTX, _block

    from visplot.plot_panel import varies_along
    from visplot.quantities import QUANTITIES, units_of
    from visplot.stream import ChunkValues

    ctx = STREAM_CTX
    values = ChunkValues(_block([0, 1, 2]), ctx)
    axes = {"STOKES": 1, "FREQ": 2}  # the stream test block's axis order
    checked = 0
    for name, quantity in QUANTITIES.items():
        if name in ("ha", "az", "el", "pa"):  # need the source table; per row like time
            continue
        for unit in (units_of(name, ctx) if not quantity.categorical else [None]):
            unit_name = unit.name if unit is not None else None
            plot = PlotSpec(y=name, x="time_h", y_unit=unit_name) if unit_name else PlotSpec(y=name, x="time_h")
            try:
                shape = values.axis(plot, "y").shape
            except ValueError:  # e.g. local time: this context has no time zone
                continue
            for axis_type, k in axes.items():
                assert varies_along(plot, axis_type, ctx) == (shape[k] > 1), (name, unit_name, axis_type)
                checked += 1
    assert checked > 20


def test_the_panel_says_what_a_row_stride_kept():
    _, figure = _run("panel_stride", every_nth=3)  # 3 baselines per integration: every 3rd row keeps one
    assert figure.panel.facts.stride_pairs == (1, 3)
    _, figure = _run("panel_integration_stride", every_nth_integration=2)
    assert figure.panel.facts.stride_pairs is None and figure.panel.facts.n_baselines == 3
    assert figure.panel.facts.filters == "every 2nd integration"


def test_the_flags_line_states_the_flag_rule():
    _, any_figure = _run("panel_flag_rule_any", plots="v-vs-u")
    assert any_figure.panel.flag_rule() == "a point is flagged if any of RR, LL is"
    _, all_figure = _run("panel_flag_rule_all", plots="v-vs-u", combine_flags="all")
    assert all_figure.panel.flag_rule() == "a point is flagged if all of RR, LL are"
    _, per_stokes = _run("panel_flag_rule_amp", combine_flags="all")  # amp varies with Stokes: no line
    assert per_stokes.panel.flag_rule() == ""


def test_the_panel_names_the_sources_unless_the_key_does():
    _, figure = _run("panel_sources")  # the synthetic file's two sources
    assert figure.ax.get_title().startswith("Amplitude vs Frequency: multiple sources")
    assert ("Sources", ["2 sources: 3C286, 3C48"]) in figure.panel._lines((), 8.0)[1]
    _, one = _run("panel_one_source", sources="3C286")
    assert one.ax.get_title().startswith("Amplitude vs Frequency: 3C286")
    assert ("Sources", ["3C286"]) in one.panel._lines((), 8.0)[1]
    _, keyed = _run("panel_sources_keyed", colorize_by="source")  # the key lists them: no second line
    assert all(label != "Sources" for label, _ in keyed.panel._lines((), 8.0)[1])
