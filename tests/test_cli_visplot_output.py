import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pytest
from astropy.io import fits

from conftest import make_scratch_dir
from cli.run_visplot import build_arg_parser, main
from visplot.request_args import NAMED_PLOTS, QUANTITY_NAMES
from data_io.row_index import build_row_index, default_row_index_path, save_row_index

JD0 = 2459421.2  # 2021-07-25


def _make_synthetic_file(path, n_rows=40):
    """n_rows rows over 3 baselines, 2 sources (first and second half), 2 Stokes x 4 channels;
    amplitude grows with row; row 0's first sample is flagged."""
    rng = np.random.default_rng(0)
    image_data = np.zeros((n_rows, 1, 4, 2, 3), dtype=">f4")
    image_data[..., 0] = (np.arange(n_rows)[:, None, None, None] + 1.0) * np.ones((1, 1, 4, 2))
    image_data[..., 1] = rng.random((n_rows, 1, 4, 2))
    image_data[..., 2] = 1.0
    image_data[0, 0, 0, 0, 2] = -1.0
    baseline = np.tile(np.array([1 * 256 + 2, 1 * 256 + 3, 2 * 256 + 3], dtype=">f4"), n_rows)[:n_rows]
    source = np.where(np.arange(n_rows) < n_rows // 2, 1, 2).astype(">f4")
    parnames = ["UU---SIN", "VV---SIN", "WW---SIN", "BASELINE", "DATE", "DATE", "SOURCE", "FREQSEL"]
    pardata = [
        (np.arange(n_rows) * 1e-7).astype(">f4"), (np.arange(n_rows) * 2e-7).astype(">f4"), np.zeros(n_rows, dtype=">f4"),
        baseline, np.full(n_rows, JD0, dtype=">f8"), (np.arange(n_rows) // 3 * 10 / 86400).astype(">f8"),
        source, np.ones(n_rows, dtype=">f4"),
    ]
    hdu = fits.GroupsHDU(fits.GroupData(image_data, parnames=parnames, pardata=pardata, bitpix=-32))
    hdu.header["TELESCOP"] = "GMRT"
    hdu.header["BUNIT"] = "UNCALIB"
    hdu.header["CTYPE2"] = "COMPLEX"
    hdu.header["CTYPE3"] = "STOKES"
    hdu.header["CRVAL3"] = -1.0
    hdu.header["CDELT3"] = -1.0
    hdu.header["CTYPE4"] = "FREQ"
    hdu.header["CRVAL4"] = 400e6
    hdu.header["CDELT4"] = 1e6
    hdu.header["CRPIX4"] = 1.0
    hdu.header["CTYPE5"] = "IF"
    su_hdu = fits.BinTableHDU.from_columns(fits.ColDefs([
        fits.Column(name="ID. NO.", format="J", array=np.array([1, 2], dtype=np.int32)),
        fits.Column(name="SOURCE", format="16A", array=np.array(["3C286", "3C48"])),
        fits.Column(name="RAAPP", format="D", array=np.array([202.8, 24.4])),
        fits.Column(name="DECAPP", format="D", array=np.array([30.5, 33.2])),
    ]), name="AIPS SU")
    an_hdu = fits.BinTableHDU.from_columns(fits.ColDefs([
        fits.Column(name="NOSTA", format="J", array=np.array([1, 2, 3], dtype=np.int32)),
        fits.Column(name="ANNAME", format="8A", array=np.array(["C00:01", "C01:02", "C02:03"])),
        fits.Column(name="STABXYZ", format="3D", array=np.zeros((3, 3))),
    ]), name="AIPS AN")
    an_hdu.header["ARRAYX"] = 1657004.629
    an_hdu.header["ARRAYY"] = 5797894.3801
    an_hdu.header["ARRAYZ"] = 2073303.1705
    fits.HDUList([hdu, su_hdu, an_hdu]).writeto(path)
    save_row_index(build_row_index(path), default_row_index_path(path))
    return path


def test_every_argument_has_help_text():
    parser = build_arg_parser()
    assert [a.dest for a in parser._actions if not a.help] == []


def test_help_plot_names_section_lists_every_plot_and_quantity():
    section = build_arg_parser().epilog.split("how plots are drawn:")[0]
    for name in sorted(NAMED_PLOTS | QUANTITY_NAMES):
        assert re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", section), name


def test_highres_pdf_is_written_by_default_and_can_be_suppressed():
    parser = build_arg_parser()
    assert parser.parse_args(["x.fits", "--plots", "antenna-layout"]).no_highres_pdf is False
    assert parser.parse_args(["x.fits", "--plots", "antenna-layout", "--no-highres-pdf"]).no_highres_pdf is True


def test_point_size_defaults_to_automatic():
    assert build_arg_parser().parse_args(["x.fits", "--plots", "amp-vs-freq_mhz"]).point_size is None


def test_colorize_by_accepts_only_categories():
    parser = build_arg_parser()
    assert parser.parse_args(["x.fits", "--plots", "amp-vs-freq_mhz", "--colorize-by", "stokes"]).colorize_by == "stokes"
    with pytest.raises(SystemExit):
        parser.parse_args(["x.fits", "--plots", "amp-vs-freq_mhz", "--colorize-by", "amp"])


def test_saved_run_writes_every_output():
    scratch = make_scratch_dir("cli_visplot_saved_run")
    path = _make_synthetic_file(scratch / "obs.fits")
    out = scratch / "out"

    code = main(["visplot", str(path), "--plots", "antenna-layout,amp-vs-freq_mhz,az-el-range",
                 "--colorize-by", "stokes", "--output-dir", str(out), "--output-prefix", "t"])

    assert code == 0
    assert sorted(p.name for p in out.iterdir()) == sorted([
        "t_antenna-layout.png", "t_amp-vs-freq_mhz.png", "t_az-el-range_el.png", "t_az-el-range_az.png",
        "t_lowres.pdf", "t_highres.pdf",
    ])
    plt.close("all")


def test_saved_generic_plot_marks_every_unflagged_sample(monkeypatch):
    scratch = make_scratch_dir("cli_visplot_samples")
    path = _make_synthetic_file(scratch / "obs.fits")
    shown = {}

    import visplot.run as cli

    real_show = cli.XYFigure.show

    def spy(self, grid, display_dpi, downsample=1):
        shown[self.plot.name] = grid
        return real_show(self, grid, display_dpi, downsample)

    monkeypatch.setattr(cli.XYFigure, "show", spy)
    main(["visplot", str(path), "--plots", "amp-vs-freq_mhz", "--output-dir", str(scratch / "out"),
          "--no-highres-pdf"])

    assert shown["amp-vs-freq_mhz"].n_samples == 40 * 4 * 2 - 1  # every sample but the one flagged
    plt.close("all")


def test_geometry_preset_reads_no_visibility_data(monkeypatch):
    scratch = make_scratch_dir("cli_visplot_geometry")
    path = _make_synthetic_file(scratch / "obs.fits")

    import data_io.visibility_data as vd

    def fail(*args, **kwargs):
        raise AssertionError("a geometry preset read visibility data")

    monkeypatch.setattr(vd, "_read_run_into", fail)
    assert main(["visplot", str(path), "--plots", "ha-range", "--output-dir", str(scratch / "out"),
                 "--no-highres-pdf"]) == 0
    plt.close("all")


def test_geometry_run_reports_where_ut1_comes_from(capsys):
    scratch = make_scratch_dir("cli_visplot_ut1_source")
    path = _make_synthetic_file(scratch / "obs.fits")

    main(["visplot", str(path), "--plots", "ha-range", "--output-dir", str(scratch / "out"), "--no-highres-pdf"])

    assert "UT1 - UTC (for hour angle, azimuth, elevation, parallactic angle, LST): IERS-B (bundled with astropy)" \
        in capsys.readouterr().out
    plt.close("all")


def test_ut1_fallback_is_warned_in_the_terminal_and_on_the_plot(monkeypatch, capsys):
    import visplot.run as cli
    import data_io.astrometry as astrometry

    class NoTables(astrometry.Ut1Provider):
        def ut1_minus_utc_s(self, jd):
            self.fallback_used = True
            self.sources_used.add("none: UT1 = UTC assumed")
            return np.zeros(np.shape(np.atleast_1d(jd)))

    provider = NoTables()
    monkeypatch.setattr(astrometry, "DEFAULT_UT1", provider)
    monkeypatch.setattr(cli, "DEFAULT_UT1", provider)
    notes = []
    original_set_note = cli.XYFigure.set_note
    monkeypatch.setattr(cli.XYFigure, "set_note", lambda self, text: (notes.append(text), original_set_note(self, text)))

    scratch = make_scratch_dir("cli_visplot_ut1_fallback")
    path = _make_synthetic_file(scratch / "obs.fits")
    main(["visplot", str(path), "--plots", "ha-range,amp-vs-freq_mhz", "--output-dir", str(scratch / "out"),
          "--no-highres-pdf"])

    assert "WARNING: UT1 - UTC unavailable" in capsys.readouterr().err
    assert notes == ["UT1 = UTC assumed: hour angle and LST may be off by up to 0.9 s of time"]  # the HA plot only
    plt.close("all")


def test_scale_on_a_category_axis_is_rejected(capsys):
    scratch = make_scratch_dir("cli_visplot_scale_category")
    path = _make_synthetic_file(scratch / "obs.fits")
    with pytest.raises(SystemExit):
        main(["visplot", str(path), "--plots", "amp-vs-stokes", "--x-scale", "log"])
    assert "applies to numeric axes" in capsys.readouterr().err


def test_log_scale_with_a_non_positive_range_is_rejected(capsys):
    with pytest.raises(SystemExit):
        main(["visplot", "x.fits", "--plots", "amp-vs-freq_mhz", "--y-scale", "log", "--y-range", "0:100"])
    assert "positive values only" in capsys.readouterr().err


def test_log_scale_and_percentile_range_run_end_to_end(monkeypatch):
    scratch = make_scratch_dir("cli_visplot_log_percentile")
    path = _make_synthetic_file(scratch / "obs.fits")
    statuses = []

    import visplot.run as cli

    original_set_status = cli.XYFigure.set_status
    monkeypatch.setattr(cli.XYFigure, "set_status",
                        lambda self, text: (statuses.append(text), original_set_status(self, text)))
    assert main(["visplot", str(path), "--plots", "amp-vs-freq_mhz", "--y-scale", "log",
                 "--y-range-mode", "percentile", "--range-percentiles", "10:90",
                 "--output-dir", str(scratch / "out"), "--no-highres-pdf"]) == 0
    assert any("outside the axis ranges, left out" in s for s in statuses)
    plt.close("all")


def test_cache_dir_skips_the_range_pass_on_a_repeat_run_and_clear_cache_removes_it(capsys):
    scratch = make_scratch_dir("cli_visplot_cache")
    path = _make_synthetic_file(scratch / "obs.fits")
    cache_dir = scratch / "cache"
    run = ["visplot", str(path), "--plots", "amp-vs-freq_mhz", "--output-dir", str(scratch / "out"),
           "--no-highres-pdf", "--cache-dir", str(cache_dir)]

    main(run)
    first = capsys.readouterr().out
    assert "pass 1 of 2: find the data range" in first
    assert "remove them with: bin/visplot.sh --clear-cache" in first
    main(run)
    second = capsys.readouterr().out
    assert "data ranges taken from the cache" in second and "pass 1 of 1: draw the plots" in second

    assert main(["visplot", "--clear-cache", str(cache_dir)]) == 0
    assert "removed 2 visplot cache file(s)" in capsys.readouterr().out
    assert list(cache_dir.iterdir()) == []
    plt.close("all")


def test_locate_from_the_command_line_writes_every_sample(capsys):
    scratch = make_scratch_dir("cli_visplot_locate")
    path = _make_synthetic_file(scratch / "obs.fits")
    csv_path = scratch / "located.csv"
    assert main(["visplot", str(path), "--plots", "amp-vs-freq_mhz", "--locate", "400.5:401.5,4.5:6.5",
                 "--locate-csv", str(csv_path)]) == 0
    assert "located 4 samples on 2 baselines" in capsys.readouterr().out
    rows = [line for line in csv_path.read_text().splitlines() if not line.startswith("#")]
    assert len(rows) == 1 + 4


def test_locate_needs_one_plot_and_a_csv(capsys):
    scratch = make_scratch_dir("cli_visplot_locate_errors")
    path = _make_synthetic_file(scratch / "obs.fits")
    with pytest.raises(SystemExit):
        main(["visplot", str(path), "--plots", "amp-vs-freq_mhz", "--locate", "0:1,0:1"])
    assert "--locate needs --locate-csv" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["visplot", str(path), "--plots", "amp-vs-freq_mhz,amp-vs-time_h", "--locate", "0:1,0:1",
              "--locate-csv", str(scratch / "x.csv")])
    assert "exactly one streamed plot" in capsys.readouterr().err


class _RecordedFigures:
    """Collects the XYFigures a run makes, to inspect their axes."""

    def __init__(self, monkeypatch):
        import visplot.run as cli

        self.figures = []
        recorded = self.figures

        class Recording(cli.XYFigure):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                recorded.append(self)

        monkeypatch.setattr(cli, "XYFigure", Recording)


def test_units_apply_to_every_plots_axis_presets_included(monkeypatch, capsys):
    recorded = _RecordedFigures(monkeypatch)
    scratch = make_scratch_dir("cli_visplot_units")
    path = _make_synthetic_file(scratch / "obs.fits")
    assert main(["visplot", str(path), "--plots", "phase-vs-time,ha-range", "--x-unit", "local", "--y-unit", "deg",
                 "--output-dir", str(scratch / "out"), "--no-highres-pdf"]) == 0
    phase, ha = recorded.figures
    assert phase.ax.get_xlabel() == ha.ax.get_xlabel() == "Time (IST, UTC+05:30; day 0 = 2021-07-25)"
    assert (phase.ax.get_ylabel(), ha.ax.get_ylabel()) == ("Phase (deg)", "Hour angle (deg)")
    assert ha.ax.get_ylim() == pytest.approx((-180.0, 180.0))  # the preset's +-12 h, converted
    assert "local time: Asia/Kolkata (IST, UTC+05:30)" in capsys.readouterr().out
    plt.close("all")


@pytest.mark.parametrize("args, message", [
    (["--plots", "amp-vs-uvdist_klambda", "--x-unit", "m"],
     "plot 'amp-vs-uvdist_klambda': 'uvdist_klambda' is 'uvdist' in klambda"),
    (["--plots", "amp-vs-time", "--x-unit", "klambda"], "unit 'klambda' does not apply to 'time'"),
    (["--plots", "amp-vs-time", "--y-unit", "mJy"], "BUNIT ('UNCALIB'), which is not a flux density"),
    (["--plots", "amp-vs-time", "--x-unit", "UTC", "--x-scale", "log"], "needs a linear x scale"),
])
def test_a_unit_that_does_not_apply_is_rejected_before_reading(args, message, capsys):
    scratch = make_scratch_dir("cli_visplot_unit_errors")
    path = _make_synthetic_file(scratch / "obs.fits")
    with pytest.raises(SystemExit):
        main(["visplot", str(path), *args, "--output-dir", str(scratch / "out")])
    err = capsys.readouterr().err
    assert message in err
    assert not (scratch / "out").exists()


def test_local_time_needs_a_known_time_zone(capsys):
    scratch = make_scratch_dir("cli_visplot_time_zone")
    path = _make_synthetic_file(scratch / "obs.fits")
    fits.setval(path, "TELESCOP", value="OTHER")  # a synthetic scratch file
    base = ["visplot", str(path), "--plots", "amp-vs-time", "--x-unit", "local", "--output-dir", str(scratch / "out"),
            "--no-highres-pdf"]
    with pytest.raises(SystemExit):
        main(base)
    assert "TELESCOP 'OTHER' is not among the known ones (GMRT: Asia/Kolkata): give --time-zone" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(base + ["--time-zone", "Mars/Olympus"])
    assert "unknown time zone 'Mars/Olympus'" in capsys.readouterr().err
    assert main(base + ["--time-zone", "Asia/Kolkata"]) == 0
    plt.close("all")


def test_recorded_time_is_the_default_with_the_files_time_system_and_reference_date(monkeypatch):
    recorded = _RecordedFigures(monkeypatch)
    scratch = make_scratch_dir("cli_visplot_recorded_time")
    path = _make_synthetic_file(scratch / "obs.fits")
    fits.setval(path, "RDATE", value="2021-07-24", extname="AIPS AN")  # a synthetic scratch file
    fits.setval(path, "TIMSYS", value="IAT", extname="AIPS AN")
    fits.setval(path, "IATUTC", value=35.0, extname="AIPS AN")
    assert main(["visplot", str(path), "--plots", "amp-vs-time", "--output-dir", str(scratch / "out"),
                 "--no-highres-pdf"]) == 0
    (figure,) = recorded.figures
    assert figure.ax.get_xlabel() == "Time (recorded, IAT; day 0 = 2021-07-24)"
    assert [t.get_text() for t in figure.ax.get_xticklabels()][0].startswith("01:")  # the day after RDATE
    plt.close("all")


def test_a_utc_plot_reports_the_time_system_and_the_timestamp_check(capsys):
    scratch = make_scratch_dir("cli_visplot_time_system")
    path = _make_synthetic_file(scratch / "obs.fits")
    fits.setval(path, "TIMSYS", value="IAT", extname="AIPS AN")
    fits.setval(path, "IATUTC", value=35.0, extname="AIPS AN")
    assert main(["visplot", str(path), "--plots", "ha-range", "--output-dir", str(scratch / "out"),
                 "--no-highres-pdf"]) == 0
    out, err = capsys.readouterr()
    assert "time system: TIMSYS IAT, IATUTC 35.0, DATUTC None: UTC = recorded - 35 s (IATUTC)" in out
    # the synthetic file's u, v, w come from no real geometry: the check says it cannot measure
    assert "WARNING: timestamps: recorded - UTC could not be measured from u, v, w" in err
    plt.close("all")


def test_iat_without_iatutc_is_rejected(capsys):
    scratch = make_scratch_dir("cli_visplot_iat_without_iatutc")
    path = _make_synthetic_file(scratch / "obs.fits")
    fits.setval(path, "TIMSYS", value="IAT", extname="AIPS AN")
    with pytest.raises(SystemExit):
        main(["visplot", str(path), "--plots", "ha-range", "--output-dir", str(scratch / "out")])
    assert "TIMSYS is 'IAT' but the antenna table has no IATUTC" in capsys.readouterr().err


def test_the_launcher_runs_as_a_script():
    """bin/visplot.sh runs src/cli/run_visplot.py as a script, which puts
    src/cli on the import path: no module there may share a package's name
    (a cli/visplot.py shadowed the visplot package once __init__.py went)."""
    import subprocess
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run([str(repo / "bin" / "visplot.sh"), "--help"], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert "usage: bin/visplot.sh" in result.stdout
