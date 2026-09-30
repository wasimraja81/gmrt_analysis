"""--antennas, --baselines-with and an empty selection (T44), on the
synthetic file: antennas C00:01, C01:02, C02:03; 40 rows cycling through
baselines 1-2, 1-3, 2-3; no autocorrelation rows."""

import pytest

from conftest import make_scratch_dir
from test_cli_visplot_output import _make_synthetic_file
from visplot.request import PlotRequest
from visplot.run import RequestError, check_request, count_selection, open_file, prepare


def _file(name):
    return str(_make_synthetic_file(make_scratch_dir(name) / "obs.fits"))


def test_antennas_keep_every_baseline_with_one_of_them_and_baselines_with_narrows_them():
    path = _file("antenna_options")
    opened = open_file(path)

    def rows(**options):
        return count_selection(PlotRequest(path, "amp-vs-freq", **options), opened).rows

    assert rows(antennas="C00") == 27  # 1-2 and 1-3
    assert rows(antennas="C00", baselines_with="C02") == 13  # 1-3
    assert rows(antennas="C00,C01", baselines_with="C00,C01") == 14  # the baselines among them: 1-2


def test_an_empty_selection_stops_a_plot_and_says_why():
    path = _file("antenna_options_empty")
    opened = open_file(path)
    request = PlotRequest(path, "amp-vs-freq", antennas="C00", baselines_with="C00")
    message = "no rows selected: --antennas C00 --baselines-with C00 leaves none of the 40 rows selected before it"
    with pytest.raises(RequestError, match=message):
        prepare(request, opened)
    with pytest.raises(RequestError, match=message):
        count_selection(request, opened)
    with pytest.raises(RequestError, match=r"this file has no autocorrelation rows \(--correlation-type auto\)"):
        prepare(PlotRequest(path, "amp-vs-freq", sources="3C286", correlation_type="auto"), opened)
    table = prepare(PlotRequest(path, "antenna-layout", correlation_type="auto"), opened)  # reads no rows
    assert [name for name, _ in table.figures()] == ["antenna-layout"]


def test_baselines_with_needs_antennas_and_cross_correlations():
    with pytest.raises(RequestError, match="--baselines-with needs --antennas"):
        check_request(PlotRequest("x.fits", "amp-vs-freq", baselines_with="C01"))
    with pytest.raises(RequestError, match="--antennas alone chooses whose autocorrelations"):
        check_request(PlotRequest("x.fits", "amp-vs-freq", antennas="C00", baselines_with="C01",
                                  correlation_type="auto"))
    check_request(PlotRequest("x.fits", "amp-vs-freq", antennas="C00", baselines_with="C01", correlation_type="both"))
