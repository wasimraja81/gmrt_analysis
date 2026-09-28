"""run_interactive driven headlessly (Agg): no window appears, but the loop,
refresh, and zoom re-stream logic run as they would with a display."""

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np

from data_io.visibility_data import VisibilityBlock
from visplot.plot_spec import PlotSpec
from visplot.quantities import QuantityContext
from visplot.stream import run_stream
from visplot.xy_figure import XYFigure
from visplot.xy_interactive import run_interactive

CTX = QuantityContext(time_reference_jd=2459421.0)


def _block(rows):
    rows = np.asarray(rows)
    return VisibilityBlock(
        row_indices=rows, data=None, weight=None,
        axis_types=["STOKES", "FREQ"], axis_indices={"STOKES": np.array([0]), "FREQ": np.array([0])},
        ant1=np.ones(len(rows)), ant2=np.full(len(rows), 2), source_id=np.ones(len(rows), dtype=int),
        jd=2459421.0 + rows / 24.0, uu_sec=rows * 1e-6, vv_sec=rows * 0.0, ww_sec=rows * 0.0,
        chan_freqs_hz=np.array([1e9]), stokes_labels=["RR"],
    )


class FakeSource:
    """Streams 10 rows in two chunks; records each pass's grid extents."""

    n_rows = 10

    def __init__(self):
        self.passes = []

    def stream(self, reducers, read_data, on_chunk=None):
        self.passes.append([(r.x_extent, r.y_extent) for r in reducers])
        return run_stream([_block(range(0, 5)), _block(range(5, 10))], CTX, reducers, on_chunk)


def _run(stop_after_passes, zoom_to=None):
    plot = PlotSpec(y="u_sec", x="time_h", apply_flags=False)
    figure = XYFigure(plot, CTX)
    source = FakeSource()
    extents = {plot: ((-0.5, 9.5), (-0.5e-6, 9.5e-6))}
    zoomed = []

    def keep_running():
        if zoom_to is not None and len(source.passes) == 1 and not zoomed:
            figure.ax.set_xlim(zoom_to[0])
            figure.ax.set_ylim(zoom_to[1])
            zoomed.append(True)
        still_streaming = figure.status.get_text().endswith("elapsed")  # a pass progress line
        return len(source.passes) < stop_after_passes or still_streaming

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Agg cannot show windows
        run_interactive(source, {plot: figure}, extents, refresh_s=0.0, settle_s=0.0, keep_running=keep_running)
    return figure, source


def test_window_fills_and_reports_samples():
    figure, source = _run(stop_after_passes=1)
    assert len(source.passes) == 1
    assert figure.status.get_text() == "10 samples from 10 rows"
    assert figure.image is not None
    plt.close("all")


def test_zoom_restreams_over_the_new_limits():
    zoom = ((2.0, 4.0), (1e-6, 3e-6))
    figure, source = _run(stop_after_passes=2, zoom_to=zoom)
    assert len(source.passes) == 2
    assert source.passes[1] == [zoom]
    assert figure.image.get_extent() == [2.0, 4.0, 1e-6, 3e-6]
    plt.close("all")
