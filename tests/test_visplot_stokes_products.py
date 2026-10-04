"""Derived Stokes products through a selection: `--stokes I,V` reads RR and
LL and shows I = RR + LL, V = RR - LL (the IAU definitions), chunk by chunk;
counted, paged and refused as the file's own products are."""

import numpy as np
import pytest
from astropy.io import fits

from cli.run_visplot import main
from conftest import make_scratch_dir
from test_cli_visplot_output import _make_synthetic_file
from visplot.iterations import iteration_source, list_iterations
from visplot.request import PlotRequest
from visplot.run import RequestError, count_selection, open_file, prepare


class _Collect:
    """A reducer keeping every block it is fed."""

    def __init__(self):
        self.blocks = []

    def compute(self, values):
        return values.block

    def apply(self, block):
        self.blocks.append(block)


def _rows_chans_stokes(block, array):
    """`array` (a block's data or weight) as [rows, channels, products]."""
    moved = np.moveaxis(array, [1 + block.axis_types.index("FREQ"), 1 + block.axis_types.index("STOKES")], [1, 2])
    return moved.reshape(moved.shape[0], moved.shape[1], moved.shape[2])


def _streamed(source):
    collect = _Collect()
    assert source.stream([collect], read_data=True)
    data = np.concatenate([_rows_chans_stokes(b, b.data) for b in collect.blocks])
    weight = np.concatenate([_rows_chans_stokes(b, b.weight) for b in collect.blocks])
    return collect.blocks[0].stokes_labels, data, weight


def _raw(path):
    """RR and LL as the file stores them, [rows, channels] each (axes: rows, IF, FREQ, STOKES, COMPLEX)."""
    with fits.open(path) as hdul:
        raw = np.asarray(hdul[0].data.data, dtype=np.float64)[:, 0]
    values = raw[..., 0] + 1j * raw[..., 1]
    return values[..., 0], values[..., 1]


def test_stokes_i_and_v_are_made_from_rr_and_ll_as_the_rows_are_read():
    path = _make_synthetic_file(make_scratch_dir("stokes_iv") / "obs.fits")
    run = prepare(PlotRequest(str(path), "amp-vs-freq", stokes="I,V"), open_file(path))
    labels, data, weight = _streamed(run.source)
    rr, ll = _raw(path)
    assert labels == ["I", "V"] and data.shape == (40, 4, 2)
    np.testing.assert_allclose(data[..., 0], rr + ll, rtol=1e-6)  # I = RR + LL
    np.testing.assert_allclose(data[..., 1], rr - ll, rtol=1e-6, atol=1e-6)  # V = RR - LL
    flagged = weight <= 0  # the file flags RR of row 0, channel 0: I and V there, nowhere else
    assert flagged[0, 0].all() and flagged.sum() == 2


def test_derived_products_are_counted_paged_and_refused_like_the_files_own():
    path = _make_synthetic_file(make_scratch_dir("stokes_pages") / "obs.fits")
    opened = open_file(path)
    assert count_selection(PlotRequest(str(path), "amp-vs-freq", stokes="I,V"), opened).samples_per_row == 8
    assert count_selection(PlotRequest(str(path), "amp-vs-freq", stokes="V"), opened).samples_per_row == 4
    run = prepare(PlotRequest(str(path), "amp-vs-freq", stokes="RR,V", one_plot_per="stokes"), opened)
    assert [it.value for it in run.iterations] == ["RR", "V"]
    rr, ll = _raw(path)
    for iteration, expected in zip(run.iterations, (rr, rr - ll)):
        labels, data, _ = _streamed(iteration_source(run.source, iteration))
        assert labels == [iteration.value]
        np.testing.assert_allclose(data[..., 0], expected, rtol=1e-6, atol=1e-6)
    with pytest.raises(RequestError, match=r"--stokes 'RR,XX': Stokes XX: not among this file's products"):
        prepare(PlotRequest(str(path), "amp-vs-freq", stokes="RR,XX"), opened)
    with pytest.raises(RequestError, match=r"Stokes Q of circular feeds needs RL, LR"):
        prepare(PlotRequest(str(path), "amp-vs-freq", stokes="Q"), opened)


def test_a_plot_of_stokes_v_is_saved_from_the_command_line():
    scratch = make_scratch_dir("stokes_cli")
    path = _make_synthetic_file(scratch / "obs.fits")
    out = scratch / "out"
    assert main(["visplot", str(path), "--plots", "amp-vs-freq", "--stokes", "V", "--output-dir", str(out),
                 "--provenance-dir", str(scratch / "runs")]) == 0
    assert list(out.glob("*.png"))
