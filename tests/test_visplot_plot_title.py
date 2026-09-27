from visplot.plot_title import build_plot_title


def test_build_plot_title_with_only_base_title():
    assert build_plot_title("Hour angle range") == "Hour angle range"


def test_build_plot_title_appends_a_single_source():
    assert build_plot_title("Hour angle range", ["3C286"]) == "Hour angle range: 3C286"


def test_build_plot_title_appends_multiple_sources_in_given_order():
    assert build_plot_title("Hour angle range", ["3C48", "3C286"]) == "Hour angle range: 3C48, 3C286"


def test_build_plot_title_ignores_an_empty_sources_list():
    assert build_plot_title("Hour angle range", []) == "Hour angle range"


def test_build_plot_title_adds_a_provenance_line_with_telescope_only():
    title = build_plot_title("Antenna layout", telescope="GMRT")
    assert title == "Antenna layout\n(GMRT)"


def test_build_plot_title_adds_a_provenance_line_with_source_path_only():
    title = build_plot_title("Antenna layout", source_path="/data/obs.fits")
    assert title == "Antenna layout\n(file: obs.fits)"


def test_build_plot_title_combines_sources_telescope_and_path():
    title = build_plot_title("Hour angle range", ["3C286"], telescope="GMRT", source_path="/data/obs.fits")
    assert title == "Hour angle range: 3C286\n(GMRT, file: obs.fits)"


def test_build_plot_title_source_path_accepts_a_full_path_and_keeps_only_the_name():
    title = build_plot_title("Antenna layout", source_path="/very/long/path/to/obs.fits")
    assert "very/long/path" not in title
    assert "obs.fits" in title
