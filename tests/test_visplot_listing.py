"""listObs (T20): the listing of what a file holds -- its sections, scans
derived by AIPS INDXR's rules, the flags section's pass, --listobs on the
command line."""

from types import SimpleNamespace

import numpy as np
import pytest

from cli.run_visplot import main
from conftest import make_scratch_dir
from data_io.observation_summary import Integrations, derive_scans
from test_cli_visplot_output import _make_synthetic_file
from visplot.flag_summary import FlagCounter
from visplot.listing import ListingOptions, list_observation
from visplot.request import PlotRequest
from visplot.run import RequestError, Stopped, check_request, count_selection, listing_text, open_file


def test_scans_start_at_a_source_change_a_gap_or_the_longest_scan():
    seconds = np.array([0, 10, 20, 30, 100, 110, 120, 130, 140, 150], dtype=float)  # a 70 s gap after 30 s
    integrations = Integrations(2459421.0 + seconds / 86400.0, np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2]),
                                np.full(10, 3))
    scans = derive_scans(integrations, gap_integrations=3.0, longest_s=3600.0)
    assert [(s.source_id, s.n_integrations, s.n_rows) for s in scans] == [(1, 4, 12), (1, 2, 6), (2, 4, 12)]
    assert scans[0].length_s == pytest.approx(40.0, abs=1e-3) and [s.number for s in scans] == [1, 2, 3]  # JD: ~40 us
    cut = derive_scans(integrations, gap_integrations=3.0, longest_s=25.0)  # at most 2 integrations of 10 s
    assert [s.n_integrations for s in cut] == [2, 2, 2, 2, 2]
    assert derive_scans(Integrations(np.array([]), np.array([]), np.array([]))) == []


def test_a_listing_holds_its_sections_from_the_tables_and_the_rows():
    path = _make_synthetic_file(make_scratch_dir("listing_sections") / "obs.fits")
    text = list_observation(open_file(path))
    for heading in ("Observation", "Scans (2)", "Sources (2 in the SU table)", "Spectral setup", "Antennas (3"):
        assert heading in text
    assert "Selection       the whole file" in text and "Rows            40 (" in text
    assert "Correlations    3 baselines, 40 cross-correlation rows; no autocorrelations" in text
    assert "No FQ table: the header's FREQ axis alone" in text and "Stokes: RR, LL" in text
    lines = text.splitlines()
    scan_rows = [line.split() for line in lines[lines.index(next(l for l in lines if l.startswith("Scans"))) + 2:][:2]]
    assert [(r[1], r[5], r[6]) for r in scan_rows] == [("3C286", "7", "20"), ("3C48", "8", "20")]
    antennas = [line.split() for line in lines if line.split()[2:3] in (["C00:01"], ["C01:02"], ["C02:03"])]
    assert [(a[0], a[1], a[-1]) for a in antennas] == [("1", "1", "27"), ("2", "2", "27"), ("3", "3", "26")]
    assert "in no row of the file" not in text  # serial, station, and the rows each antenna holds
    only_scans = list_observation(open_file(path), ListingOptions(sections=("scans",)))
    assert only_scans.startswith("Scans (2)") and "Antennas" not in only_scans


def test_listobs_prints_saves_and_lists_a_selection(capsys):
    scratch = make_scratch_dir("listing_cli")
    path = _make_synthetic_file(scratch / "obs.fits")
    assert main(["visplot", str(path), "--listobs", "--provenance-dir", str(scratch / "runs")]) == 0
    printed = capsys.readouterr().out
    assert "Scans (2)" in printed and "Antennas (3" in printed and "Flags" not in printed  # flags: asked for
    out = scratch / "out"
    assert main(["visplot", str(path), "--listobs", "observation,scans", "--sources", "3C48", "--output-dir", str(out),
                 "--provenance-dir", str(scratch / "runs")]) == 0
    saved = (out / "visplot_listobs.txt").read_text()
    assert saved.startswith("# visplot listing of obs.fits\n# command: bin/visplot.sh")
    assert "--listobs observation,scans" in saved and "# record: run " in saved
    assert "Rows            20 of 40" in saved and "Selection       sources 3C48, cross- and autocorrelations" in saved
    assert "Scans (1)" in saved and "Sources (" not in saved


def test_listing_options_are_checked():
    check_request(PlotRequest("obs.fits", None, listobs="scans"))  # a listing alone
    for options, message in ((dict(), "give --plots, --listobs, or both"),
                             (dict(listobs="scans,bogus"), "--listobs takes sections among"),
                             (dict(listobs="scans", scan_gap=0.0), "--scan-gap and --scan-longest take a positive")):
        with pytest.raises(RequestError, match=message):
            check_request(PlotRequest("obs.fits", None, **options))


def test_the_flags_section_counts_the_flagged_visibilities_by_stokes_source_scan_and_antenna(capsys):
    scratch = make_scratch_dir("listing_flags")
    path = _make_synthetic_file(scratch / "obs.fits")  # row 0 (baseline 1-2, 3C286, scan 1): RR, channel 1 flagged
    opened = open_file(path)
    reports = []
    text = listing_text(PlotRequest(str(path), None, listobs="flags"), opened,
                        report=lambda level, message: reports.append(message))
    assert reports[-1] == "flags: one pass over 40 rows, 0.0 GB of visibilities to read"
    assert text.startswith("Flags: 0.31% of 320 visibilities flagged")
    rows = {tuple(line.split()[:-3]): line.split()[-3:] for line in text.splitlines()[1:] if line.strip()}
    assert rows[("RR",)] == ["160", "1", "0.62"] and rows[("LL",)] == ["160", "0", "0.00"]
    assert rows[("1", "3C286")] == ["160", "1", "0.62"] and rows[("2", "3C48")] == ["160", "0", "0.00"]
    assert rows[("1", "3C286", "0/18:00:00")] == ["160", "1", "0.62"]
    assert rows[("1", "1", "C00:01")] == ["216", "1", "0.46"] and rows[("3", "3", "C02:03")] == ["208", "0", "0.00"]
    only_ll = listing_text(PlotRequest(str(path), None, listobs="flags", stokes="LL"), opened)
    assert only_ll.startswith("Flags: 0.00% of 160 visibilities") and "RR" not in only_ll
    with pytest.raises(Stopped, match="stopped while counting the flags"):
        listing_text(PlotRequest(str(path), None, listobs="scans,flags"), opened,
                     progress=lambda pass_progress: lambda rows_done: False)
    assert main(["visplot", str(path), "--listobs", "flags", "--provenance-dir", str(scratch / "runs")]) == 0
    printed = capsys.readouterr().out
    assert "counting flags: 40 / 40 rows (100%)" in printed and "Flags: 0.31% of 320" in printed


def test_flag_counts_take_an_autocorrelation_once_and_a_file_without_stokes():
    weight = np.ones((2, 1, 3), dtype=np.float32)  # 2 rows, IF, 3 channels: no STOKES axis
    weight[1, 0, :2] = 0.0
    block = SimpleNamespace(weight=weight, axis_types=["IF", "FREQ"], stokes_labels=None, row_indices=np.array([0, 2]),
                            ant1=np.array([1, 4]), ant2=np.array([2, 4]), source_id=np.array([5, 5]))
    counter = FlagCounter(np.array([0, 2, 4]))
    counter.apply(counter.compute(SimpleNamespace(block=block)))
    counts = counter.counts
    assert (counts.flagged, counts.total) == (2, 6) and counts.by_stokes == {"all": [2, 6]}
    assert counts.by_antenna == {1: [0, 3], 2: [0, 3], 4: [2, 3]} and counts.by_source == {5: [2, 6]}
    assert counts.integration_flagged.tolist() == [0, 2] and counts.integration_total.tolist() == [3, 3]


def test_a_listing_counts_the_files_autocorrelations_selected_or_not():
    scratch = make_scratch_dir("listing_autos")
    path = _make_synthetic_file(scratch / "obs.fits", baselines=(1 * 256 + 1, 1 * 256 + 2, 2 * 256 + 3))  # 1-1: auto
    opened = open_file(path)
    reports = []
    whole = listing_text(PlotRequest(str(path), None, listobs="observation"), opened,
                         report=lambda level, message: reports.append(message))
    assert reports == ["listing cross- and autocorrelations (--correlation-type not given: a listing's default)"]
    assert "Selection       the whole file" in whole
    assert "Correlations    2 baselines, 26 cross-correlation rows; autocorrelations of 1 antenna, 14 rows\n" in whole
    cross = listing_text(PlotRequest(str(path), None, listobs="observation", correlation_type="cross"), opened)
    assert "Rows            26 of 40" in cross and "Selection       cross-correlations" in cross
    assert "autocorrelations of 1 antenna, 14 rows (none of them selected)" in cross


def test_correlation_type_not_given_resolves_by_what_runs():
    """One request without --correlation-type, as the command line and the
    GUI both write it: a plot takes the cross-correlations, a listing every
    row, and a rerun of the same command line the same."""
    scratch = make_scratch_dir("correlation_default")
    path = _make_synthetic_file(scratch / "obs.fits", baselines=(1 * 256 + 1, 1 * 256 + 2, 2 * 256 + 3))
    opened = open_file(path)
    request = PlotRequest(str(path), "amp-vs-freq", listobs="observation")
    assert "--correlation-type" not in request.command_line()
    assert count_selection(request, opened).rows == 26  # the plot: cross-correlations
    assert "Rows            40 (" in listing_text(request, opened)  # the listing: every row
    rerun = PlotRequest.from_argv(request.to_argv())
    assert rerun == request and count_selection(rerun, opened).rows == 26
    autos_only = _make_synthetic_file(scratch / "autos.fits", baselines=(1 * 256 + 1, 2 * 256 + 2, 3 * 256 + 3))
    with pytest.raises(RequestError, match=r"no cross-correlation rows \(--correlation-type cross, not given: a "
                                           r"plot's default\)"):
        count_selection(PlotRequest(str(autos_only), "amp-vs-freq"), open_file(autos_only))


def test_antennas_without_rows_are_named_and_the_rest_counted():
    """An antenna of the AN table that no row holds (as the archival files' C03:04
    and C10:10): the Antennas and Flags tables say so, their Serial column
    counting the antennas each lists."""
    scratch = make_scratch_dir("listing_antennas_without_rows")
    path = _make_synthetic_file(scratch / "obs.fits", baselines=(1 * 256 + 3,))  # C01:02 (station 2) in no row
    opened = open_file(path)
    text = listing_text(PlotRequest(str(path), None, listobs="antennas,flags"), opened)
    lines = text.splitlines()
    antennas = [line.split() for line in lines if line.split()[2:3] in (["C00:01"], ["C01:02"], ["C02:03"])]
    listed = [(a[0], a[1], a[-1]) for a in antennas]
    assert listed[:3] == [("1", "1", "40"), ("2", "2", "0"), ("3", "3", "40")]  # the Antennas table: all three
    assert [(a[0], a[1], a[2]) for a in antennas[3:]] == [("1", "1", "C00:01"), ("2", "3", "C02:03")]  # Flags: two
    assert text.count("In the AN table, in no row of the file (nothing recorded for them): C01:02") == 2
    three = _make_synthetic_file(scratch / "three.fits")  # baselines 1-2, 1-3, 2-3
    unselected = listing_text(PlotRequest(str(three), None, listobs="antennas", exclude_antennas="C02"), open_file(three))
    assert "In no selected row: C02:03" in unselected and "in no row of the file" not in unselected
