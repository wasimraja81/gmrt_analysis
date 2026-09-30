import numpy as np
import pytest

from data_io.visibility_data import VisibilityBlock
from visplot.plot_spec import PlotSpec
from visplot.quantities import QuantityContext
from visplot.stream import FLAGGED_LAYER, ChunkValues, GridReducer, RangeReducer, run_stream

CTX = QuantityContext(time_reference_jd=2459421.0, stokes_labels=("RR", "LL"))


def _block(rows, weight=None, n_chan=2):
    """rows: list of row indices; jd = JD0 + row/24 (so time_h == row); u = row km; 2 Stokes x n_chan."""
    rows = np.asarray(rows)
    data = (rows[:, None, None] + 1.0) * np.ones((1, 2, n_chan)) + 0j
    return VisibilityBlock(
        row_indices=rows,
        data=data,
        weight=np.ones_like(data.real) if weight is None else weight,
        axis_types=["STOKES", "FREQ"],
        axis_indices={"STOKES": np.array([0, 1]), "FREQ": np.arange(n_chan)},
        ant1=np.ones(len(rows)), ant2=np.full(len(rows), 2), source_id=np.ones(len(rows), dtype=int),
        jd=2459421.0 + rows / 24.0,
        uu_sec=rows * 1e3 / 299_792_458.0, vv_sec=rows * 0.0, ww_sec=rows * 0.0,  # u = row km
        chan_freqs_hz=np.linspace(1e9, 1.1e9, n_chan),
        stokes_labels=["RR", "LL"],
    )


def test_samples_broadcast_to_the_shape_the_pair_needs():
    values = ChunkValues(_block([0, 1, 2]), CTX)
    x, y, code, flagged = values.samples(PlotSpec(y="u", y_unit="km", x="time_h", apply_flags=False))
    assert x.size == 3  # one per row: neither quantity varies over Stokes or channel
    x, y, code, flagged = values.samples(PlotSpec(y="amp", x="time_h"))
    assert x.size == 3 * 2 * 2


def test_samples_drop_flagged_unless_shown():
    weight = np.ones((2, 2, 2))
    weight[0, 0, 0] = -1.0
    values = ChunkValues(_block([0, 1], weight=weight), CTX)
    x, _, _, flagged = values.samples(PlotSpec(y="amp", x="time_h"))
    assert x.size == 7 and flagged is None  # None: no flagged sample is drawn
    x, _, _, flagged = values.samples(PlotSpec(y="amp", x="time_h", show_flagged=True))
    assert x.size == 8 and flagged.sum() == 1


def test_samples_mirror_appends_the_negated_points():
    values = ChunkValues(_block([1, 2]), CTX)
    x, y, _, _ = values.samples(PlotSpec(y="u", y_unit="km", x="time_h", apply_flags=False, mirror=True))
    np.testing.assert_allclose(sorted(x), [-2, -1, 1, 2])


def test_run_stream_feeds_every_chunk_and_can_stop_early():
    plot = PlotSpec(y="amp", x="time_h")
    seen = []

    class Recorder:
        def compute(self, values):
            return values.block.row_indices.tolist()

        def apply(self, rows):
            seen.append(rows)

    assert run_stream([_block([0]), _block([1]), _block([2])], CTX, [Recorder()]) is True
    assert seen == [[0], [1], [2]]
    seen.clear()
    assert run_stream([_block([0]), _block([1]), _block([2])], CTX, [Recorder()], on_chunk=lambda rows: rows < 2) is False
    assert seen == [[0], [1]]


def test_range_reducer_spans_all_chunks():
    plot = PlotSpec(y="amp", x="time_h")
    rx, ry = RangeReducer(plot, "x"), RangeReducer(plot, "y")
    run_stream([_block([0, 1]), _block([5])], CTX, [rx, ry])
    assert (rx.lo, rx.hi) == pytest.approx((0.0, 5.0))  # JD arithmetic carries ~1e-8 h of rounding
    assert (ry.lo, ry.hi) == (1.0, 6.0)
    lo, hi = rx.extent(margin=0.1)
    assert lo == pytest.approx(-0.5) and hi == pytest.approx(5.5)


def test_range_reducer_category_axis_gets_half_a_slot_each_side():
    plot = PlotSpec(y="amp", x="stokes")
    rx = RangeReducer(plot, "x")
    run_stream([_block([0])], CTX, [rx])
    assert rx.extent() == (-0.5, 1.5)


def test_grid_reducer_bins_samples_into_pixels():
    plot = PlotSpec(y="u", y_unit="km", x="time_h", apply_flags=False)
    # samples at pixel centres, clear of JD rounding at the pixel edges
    grid = GridReducer(plot, (-0.5, 3.5), (-0.5, 3.5), height=4, width=4)
    run_stream([_block([0, 1]), _block([3])], CTX, [grid])
    occupied = np.argwhere(grid.layers_2d() > 0).tolist()
    assert occupied == [[0, 0], [1, 1], [3, 3]]  # (iy, ix): time 0,1,3 with u 0,1,3 km
    assert grid.n_samples == 3


def test_grid_reducer_is_unchanged_by_adding_the_same_samples_again():
    plot = PlotSpec(y="amp", x="time_h")
    grid = GridReducer(plot, (0.0, 4.0), (0.0, 5.0), height=8, width=8)
    run_stream([_block([0, 1, 2])], CTX, [grid])
    first = grid.layers.copy()
    run_stream([_block([0, 1, 2])], CTX, [grid])
    np.testing.assert_array_equal(grid.layers, first)


def test_grid_reducer_paints_higher_codes_and_flagged_on_top():
    weight = np.ones((1, 2, 1))
    plot = PlotSpec(y="u", y_unit="km", x="time_h", colorize_by="stokes", show_flagged=True)
    grid = GridReducer(plot, (0.0, 1.0), (0.0, 1.0), height=1, width=1)
    run_stream([_block([0], weight=weight, n_chan=1)], CTX, [grid])
    assert grid.layers_2d()[0, 0] == 2  # LL (code 1) painted over RR (code 0)
    assert grid.seen_codes == {0, 1}
    weight[0, 0, 0] = -1.0
    run_stream([_block([0], weight=weight, n_chan=1)], CTX, [grid])
    assert grid.layers_2d()[0, 0] == FLAGGED_LAYER


def test_grid_reducer_ignores_samples_outside_its_extent():
    plot = PlotSpec(y="u", y_unit="km", x="time_h", apply_flags=False)
    grid = GridReducer(plot, (10.0, 20.0), (0.0, 1.0), height=2, width=2)
    run_stream([_block([0, 1])], CTX, [grid])
    assert grid.n_samples == 0 and not grid.layers.any()


def test_range_reducer_evaluates_only_its_own_axis():
    meta_only = VisibilityBlock(**{**_block([0, 3]).__dict__, "data": None, "weight": None})
    plot = PlotSpec(y="amp", x="time_h", y_range=(0.0, 1.0))
    rx = RangeReducer(plot, "x")
    run_stream([meta_only], CTX, [rx])  # amp is never evaluated, so no data is needed
    assert (rx.lo, rx.hi) == pytest.approx((0.0, 3.0))


def test_range_reducer_mirror_covers_the_negated_values():
    plot = PlotSpec(y="u", y_unit="km", x="time_h", apply_flags=False, mirror=True)
    rx = RangeReducer(plot, "x")
    run_stream([_block([1, 2])], CTX, [rx])
    assert (rx.lo, rx.hi) == pytest.approx((-2.0, 2.0))


def test_threaded_stream_gives_the_same_grid_as_one_thread():
    weight = np.ones((40, 2, 3))
    weight[::7, 1, :] = -1.0
    chunks = [_block(range(0, 40), weight=weight, n_chan=3), _block(range(40, 55), n_chan=3)]
    plot = PlotSpec(y="amp", x="time_h", colorize_by="stokes", show_flagged=True, mirror=True)

    import visplot.stream as stream

    grids = []
    for threads in (1, 4):
        grid = GridReducer(plot, (-60.0, 60.0), (-60.0, 60.0), height=37, width=41)
        min_rows = stream.MIN_ROWS_PER_THREAD
        stream.MIN_ROWS_PER_THREAD = 1  # split these small chunks, to exercise the threads
        try:
            run_stream(chunks, CTX, [grid], threads=threads)
        finally:
            stream.MIN_ROWS_PER_THREAD = min_rows
        grids.append(grid)
    np.testing.assert_array_equal(grids[0].layers, grids[1].layers)
    assert grids[0].n_samples == grids[1].n_samples
    assert grids[0].seen_codes == grids[1].seen_codes


def test_prefetched_stream_raises_reader_errors():
    def failing_chunks():
        yield _block([0])
        raise OSError("disk went away")

    grid = GridReducer(PlotSpec(y="amp", x="time_h"), (0.0, 1.0), (0.0, 1.0), height=2, width=2)
    with pytest.raises(OSError, match="disk went away"):
        run_stream(failing_chunks(), CTX, [grid])


def test_prefetched_stream_stops_its_reader_when_stopped_early():
    import threading

    produced = []

    def chunks():
        for i in range(100):
            produced.append(i)
            yield _block([i])

    grid = GridReducer(PlotSpec(y="amp", x="time_h"), (0.0, 200.0), (0.0, 200.0), height=4, width=4)
    assert run_stream(chunks(), CTX, [grid], on_chunk=lambda rows: rows < 3) is False
    assert len(produced) < 10  # the reader stayed at most a chunk or two ahead
    assert not any(t.name == "visplot-reader" for t in threading.enumerate())


def test_range_reducer_percentile_mode_narrows_to_the_bulk():
    # amp grows with row: rows 0..999 give amp 1..1000; the 1-99 percentiles leave out the lowest and highest 1%
    plot = PlotSpec(y="amp", x="time_h", y_range_mode="percentile", range_percentiles=(1.0, 99.0))
    ry = RangeReducer(plot, "y")
    run_stream([_block(range(0, 1000))], CTX, [ry])
    lo, hi = ry.extent(margin=0.0)
    assert (ry.lo, ry.hi) == (1.0, 1000.0)
    assert 10.0 <= lo <= 11.1 and 990.0 <= hi <= 991.1


def test_range_reducer_log_scale_ignores_non_positive_values():
    plot = PlotSpec(y="u", y_unit="km", x="time_h", apply_flags=False, x_scale="log")
    rx = RangeReducer(plot, "x")
    run_stream([_block([0, 1, 10])], CTX, [rx])  # time_h 0 cannot be shown on a log axis
    assert (rx.lo, rx.hi) == pytest.approx((1.0, 10.0))
    lo, hi = rx.extent(margin=0.1)  # margin in decades: 10% of one decade each side
    assert lo == pytest.approx(10 ** -0.1) and hi == pytest.approx(10 ** 1.1)


def test_grid_reducer_bins_evenly_in_the_log_coordinate_and_counts_what_it_leaves_out():
    plot = PlotSpec(y="u", y_unit="km", x="time_h", apply_flags=False, x_scale="log")
    grid = GridReducer(plot, (1.0, 1000.0), (-1.0, 2000.0), height=1, width=3)
    run_stream([_block([0, 2, 20, 200, 5000])], CTX, [grid])  # time 0 is not positive; 5000 is beyond the range
    columns = np.flatnonzero(grid.layers_2d()[0])
    assert columns.tolist() == [0, 1, 2]  # one decade per pixel: 2, 20, 200
    assert grid.n_samples == 3 and grid.n_outside == 2


def test_locate_finds_the_samples_in_a_box_with_where_each_comes_from():
    from visplot.stream import LocateReducer

    # amp = row + 1 for every Stokes and channel; box around rows 2-3 (amp 3-4), channel 1 only (1.05 GHz)
    plot = PlotSpec(y="amp", x="freq_mhz")
    locate = LocateReducer(plot, (1040.0, 1060.0), (2.5, 4.5))
    run_stream([_block(range(0, 6), n_chan=3)], CTX, [locate])
    assert locate.n_found == 2 * 2  # rows 2, 3 x Stokes RR, LL
    got = sorted((r["row"], r["channel"], r["stokes"]) for r in locate.records)
    assert got == [(2, 1, "LL"), (2, 1, "RR"), (3, 1, "LL"), (3, 1, "RR")]
    assert all(r["freq_hz"] == 1.05e9 and r["weight"] == 1.0 for r in locate.records)
    assert locate.by_baseline == {(1, 2): 4}


def test_locate_on_a_per_row_plot_reports_channel_and_stokes_as_all():
    from visplot.stream import LocateReducer

    plot = PlotSpec(y="u", y_unit="km", x="time_h", apply_flags=False)
    locate = LocateReducer(plot, (0.5, 2.5), (0.0, 3.0))
    run_stream([_block(range(0, 5))], CTX, [locate])
    assert [r["row"] for r in locate.records] == [1, 2]
    assert all(r["channel"] is None and r["stokes"] is None for r in locate.records)


def test_locate_keeps_at_most_limit_records_but_counts_all():
    from visplot.stream import LocateReducer

    locate = LocateReducer(PlotSpec(y="amp", x="time_h"), (-1.0, 100.0), (0.0, 100.0), limit=5)
    run_stream([_block(range(0, 10)), _block(range(10, 20))], CTX, [locate])
    assert locate.n_found == 20 * 2 * 2 and len(locate.records) == 5


def test_locate_marks_mirrored_samples():
    from visplot.stream import LocateReducer

    plot = PlotSpec(y="u", y_unit="km", x="time_h", apply_flags=False, mirror=True)
    locate = LocateReducer(plot, (-2.5, -1.5), (-3.0, 0.0))  # the mirror of row 2 (time 2 h, u 2 km)
    run_stream([_block(range(0, 5))], CTX, [locate])
    assert [(r["row"], r["mirrored"]) for r in locate.records] == [(2, True)]


def test_grid_snapshot_is_an_independent_copy():
    grid = GridReducer(PlotSpec(y="amp", x="time_h"), (0.0, 10.0), (0.0, 10.0), height=4, width=4)
    run_stream([_block([1])], CTX, [grid])
    snap = grid.snapshot()
    run_stream([_block([8])], CTX, [grid])
    assert snap.n_samples < grid.n_samples
    assert (snap.layers > 0).sum() < (grid.layers > 0).sum()


# The flag rule (the user's, 2026-09-29): flags have the data's shape, one per visibility's
# time, baseline, channel and Stokes; a point whose quantities do not vary with Stokes combines
# its visibility's selected Stokes and is flagged if any of them is; one Stokes selected, its own
# flag decides; channels are never combined.

def _flag_block():
    weight = np.ones((2, 2, 2))  # rows, Stokes (RR, LL), channels
    weight[0, 0, 0] = -1.0  # row 0: RR flagged in channel 0 (LL is not)
    weight[1, 1, :] = -1.0  # row 1: LL flagged in both channels
    return _block([0, 1], weight=weight)


def test_a_point_combining_stokes_is_flagged_if_any_stokes_is():
    from dataclasses import replace

    values = ChunkValues(_flag_block(), CTX)
    uv = PlotSpec(y="v", x="u")  # in kλ: one point per row and channel, the Stokes combined
    x, _, _, _ = values.samples(uv)
    assert x.size == 1  # of 2 rows x 2 channels only row 0, channel 1 has every Stokes unflagged
    x, _, _, flagged = values.samples(replace(uv, show_flagged=True))
    assert x.size == 4 and flagged.sum() == 3  # one sample per point, its Stokes combined


def test_by_the_rule_all_a_point_combining_stokes_is_flagged_only_if_every_stokes_is():
    from dataclasses import replace

    from visplot.stream import combine_flags

    block = _flag_block()
    weight = block.weight.copy()
    weight[1, 0, 0] = -1.0  # row 1, channel 0: RR flagged too, so both Stokes are
    values = ChunkValues(replace(block, weight=weight), CTX)
    uv = PlotSpec(y="v", x="u", combine_flags="all")
    x, _, _, flagged = values.samples(replace(uv, show_flagged=True))
    assert x.size == 4 and flagged.sum() == 1  # only row 1, channel 0 has every Stokes flagged
    assert values.samples(uv)[0].size == 3
    assert values.samples(PlotSpec(y="v", x="u"))[0].size == 1  # the rule any: row 0, channel 1 alone
    per_stokes = PlotSpec(y="amp", x="u", combine_flags="all")  # varies with Stokes: each visibility's own flag
    assert values.samples(per_stokes)[0].size == values.samples(PlotSpec(y="amp", x="u"))[0].size == 4
    with pytest.raises(ValueError, match="'any' or 'all'"):
        combine_flags(weight > 0, (2, 1, 2), ["STOKES", "FREQ"], "most")


def test_one_stokes_selected_its_own_flags_decide():
    from dataclasses import replace

    block = _flag_block()
    ll_only = replace(block, data=block.data[:, 1:], weight=block.weight[:, 1:],
                      axis_indices={"STOKES": np.array([1]), "FREQ": np.arange(2)}, stokes_labels=["LL"])
    x, _, _, _ = ChunkValues(ll_only, QuantityContext(time_reference_jd=2459421.0, stokes_labels=("LL",))).samples(
        PlotSpec(y="v", x="u"))
    assert x.size == 2  # row 0's LL is unflagged in both channels; row 1's LL is flagged


def test_a_per_row_quantity_is_one_point_per_channels_visibility():
    from dataclasses import replace

    values = ChunkValues(_flag_block(), CTX)
    per_row = PlotSpec(y="u", y_unit="km", x="time_h")  # the same u for every channel of a row
    x, _, _, _ = values.samples(per_row)
    np.testing.assert_allclose(x, [0.0])  # only row 0, channel 1 has every Stokes unflagged
    x, _, _, flagged = values.samples(replace(per_row, show_flagged=True))
    assert x.size == 4 and flagged.sum() == 3  # 2 rows x 2 channels, each channel's own flags


def test_locate_finds_the_points_the_plot_draws_with_their_flags():
    from visplot.stream import LocateReducer

    values = ChunkValues(_flag_block(), CTX)
    plot = PlotSpec(y="v", x="u", show_flagged=True)
    n_found, pieces, _ = LocateReducer(plot, (-1e9, 1e9), (-1e9, 1e9)).compute(values)
    assert n_found == values.samples(plot)[0].size == 4
    (columns,) = pieces
    assert list(columns["stokes"]) == ["all"] * 4 and sorted(columns["flagged"]) == ["no", "yes", "yes", "yes"]
    assert np.isnan(columns["weight"]).all()  # a point combining Stokes has no one weight
