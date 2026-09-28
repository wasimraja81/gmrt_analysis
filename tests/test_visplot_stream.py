import numpy as np
import pytest

from data_io.visibility_data import VisibilityBlock
from visplot.plot_spec import PlotSpec
from visplot.quantities import QuantityContext
from visplot.stream import FLAGGED_LAYER, ChunkValues, GridReducer, RangeReducer, run_stream

CTX = QuantityContext(time_reference_jd=2459421.0, stokes_labels=("RR", "LL"))


def _block(rows, weight=None, n_chan=2):
    """rows: list of row indices; jd = JD0 + row/24 (so time_h == row); 2 Stokes x n_chan."""
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
        uu_sec=rows * 1e-6, vv_sec=rows * 0.0, ww_sec=rows * 0.0,
        chan_freqs_hz=np.linspace(1e9, 1.1e9, n_chan),
        stokes_labels=["RR", "LL"],
    )


def test_samples_broadcast_to_the_shape_the_pair_needs():
    values = ChunkValues(_block([0, 1, 2]), CTX)
    x, y, code, flagged = values.samples(PlotSpec(y="u_sec", x="time_h", apply_flags=False))
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
    x, y, _, _ = values.samples(PlotSpec(y="u_sec", x="time_h", apply_flags=False, mirror=True))
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
    plot = PlotSpec(y="u_sec", x="time_h", apply_flags=False)
    # samples at pixel centres, clear of JD rounding at the pixel edges
    grid = GridReducer(plot, (-0.5, 3.5), (-0.5e-6, 3.5e-6), height=4, width=4)
    run_stream([_block([0, 1]), _block([3])], CTX, [grid])
    occupied = np.argwhere(grid.layers_2d() > 0).tolist()
    assert occupied == [[0, 0], [1, 1], [3, 3]]  # (iy, ix): time 0,1,3 with u 0,1,3 microseconds
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
    plot = PlotSpec(y="u_sec", x="time_h", colorize_by="stokes", show_flagged=True)
    grid = GridReducer(plot, (0.0, 1.0), (0.0, 1.0), height=1, width=1)
    run_stream([_block([0], weight=weight, n_chan=1)], CTX, [grid])
    assert grid.layers_2d()[0, 0] == 2  # LL (code 1) painted over RR (code 0)
    assert grid.seen_codes == {0, 1}
    weight[0, 0, 0] = -1.0
    run_stream([_block([0], weight=weight, n_chan=1)], CTX, [grid])
    assert grid.layers_2d()[0, 0] == FLAGGED_LAYER


def test_grid_reducer_ignores_samples_outside_its_extent():
    plot = PlotSpec(y="u_sec", x="time_h", apply_flags=False)
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
    plot = PlotSpec(y="u_sec", x="time_h", apply_flags=False, mirror=True)
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
