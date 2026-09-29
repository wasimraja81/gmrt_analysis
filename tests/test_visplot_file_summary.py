"""The GUI's file summary (`visplot.file_summary`)."""

from conftest import make_scratch_dir
from test_cli_visplot_output import _make_synthetic_file
from visplot.file_summary import summarize
from visplot.run import open_file


def test_the_summary_counts_the_files_rows_antennas_baselines_and_sources():
    path = _make_synthetic_file(make_scratch_dir("file_summary") / "obs.fits")
    opened = open_file(path)
    index = opened.index
    summary = summarize(opened)

    cross = {(int(a), int(b)) for a, b in zip(index.ant1, index.ant2) if a != b}
    assert summary.n_baselines == len(cross)
    assert summary.n_antennas == len(set(index.ant1.tolist()) | set(index.ant2.tolist()))
    assert summary.n_rows == index.gcount
    assert sum(rows for _, _, rows in summary.sources) == index.gcount
    assert summary.n_channels == len(index.chan_freqs_hz)
