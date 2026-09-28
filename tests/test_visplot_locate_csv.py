import numpy as np

from conftest import make_scratch_dir
from data_io.visibility_data import VisibilityBlock
from visplot.locate_csv import COLUMNS, LocateCsvWriter
from visplot.plot_spec import PlotSpec
from visplot.quantities import QuantityContext
from visplot.stream import LocateReducer, run_stream

CTX = QuantityContext(time_reference_jd=2459421.0, stokes_labels=("RR", "LL"),
                      antenna_names={1: "C00:01", 2: "C01:02"}, source_names={1: "3C286"})


def _block(rows):
    rows = np.asarray(rows)
    data = (rows[:, None, None] + 1.0) * np.ones((1, 2, 2)) + 0j
    return VisibilityBlock(
        row_indices=rows, data=data, weight=np.ones(data.shape), axis_types=["STOKES", "FREQ"],
        axis_indices={"STOKES": np.array([0, 1]), "FREQ": np.array([10, 11])},
        ant1=np.ones(len(rows), dtype=int), ant2=np.full(len(rows), 2), source_id=np.ones(len(rows), dtype=int),
        jd=2459421.0 + rows / 24.0, uu_sec=rows * 1e-6, vv_sec=rows * 0.0, ww_sec=rows * 0.0,
        chan_freqs_hz=np.array([1.0e9, 1.1e9]), stokes_labels=["RR", "LL"],
    )


def test_writer_streams_every_located_sample_with_readable_and_exact_columns():
    scratch = make_scratch_dir("locate_csv_all")
    path = scratch / "located.csv"
    locate = LocateReducer(PlotSpec(y="amp", x="freq_mhz"), (0.0, 2000.0), (2.5, 4.5), limit=1)
    writer = LocateCsvWriter(path, locate, CTX)
    locate.sink = writer
    run_stream([_block([0, 1, 2]), _block([3, 4])], CTX, [locate])
    assert writer.close(locate, completed=True) == path

    text = path.read_text().splitlines()
    rows = [line.split(",") for line in text if not line.startswith("#")]
    assert tuple(rows[0]) == COLUMNS
    body = rows[1:]
    assert len(body) == locate.n_found == 2 * 2 * 2  # rows 2, 3 x 2 Stokes x 2 channels
    first = dict(zip(COLUMNS, body[0]))
    assert first["baseline"] == "C00:01-C01:02" and first["ant1"] == "1" and first["ant2"] == "2"
    assert first["channel"] in {"10", "11"} and first["stokes"] in {"RR", "LL"} and first["source"] == "3C286"
    assert first["time_utc"].startswith("2021-07-25")
    assert "# total: 8 samples in the box, all listed above" in text
    assert "# baseline C00:01-C01:02: 8" in text


def test_an_interrupted_pass_leaves_no_file():
    scratch = make_scratch_dir("locate_csv_interrupted")
    path = scratch / "located.csv"
    locate = LocateReducer(PlotSpec(y="amp", x="freq_mhz"), (0.0, 2000.0), (0.0, 100.0))
    writer = LocateCsvWriter(path, locate, CTX)
    locate.sink = writer
    run_stream([_block([0, 1])], CTX, [locate])
    assert writer.close(locate, completed=False) is None
    assert list(scratch.iterdir()) == []
