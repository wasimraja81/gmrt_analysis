"""The fonts visplot draws its panels with (T41): installed into the venv from
a pinned download, and loaded from their files."""

import hashlib
import json
import zipfile

import matplotlib

matplotlib.use("Agg")
import pytest

from cli.install_fonts import check, install
from conftest import make_scratch_dir
from test_cli_visplot_output import _make_synthetic_file
import visplot.fonts as fonts
from visplot.fonts import FontNotInstalled, font_file, installed_dir, read_pins
from visplot.request import PlotRequest
from visplot.run import RequestError, check_request, prepare


def _pinned_zip(scratch, sha256=None):
    archive = scratch / "upstream" / "somefont.zip"
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("somefont-regular.otf", b"not a font, a stand-in")
        zf.writestr("somefont-bold.otf", b"another stand-in")
    digest = sha256 or hashlib.sha256(archive.read_bytes()).hexdigest()
    config = scratch / "fonts.txt"
    config.write_text(f"# name version sha256 url\nsomefont 1.0 {digest} {archive.as_uri()}\n")
    return config


def test_the_installer_unpacks_a_pinned_download_into_the_venv_once():
    scratch = make_scratch_dir("fonts_install")
    config = _pinned_zip(scratch)
    prefix, downloads = scratch / "venv", scratch / "downloads"
    (line,) = install(config, prefix, downloads)
    target = installed_dir("somefont", prefix)
    assert "installed" in line and sorted(p.name for p in target.iterdir()) == [
        "SOURCE.txt", "somefont-bold.otf", "somefont-regular.otf"]
    assert (target / "SOURCE.txt").read_text() == read_pins(config)[0].stamp()
    assert not downloads.exists()  # the download is deleted once installed
    (line,) = install(config, prefix, downloads)
    assert "already installed" in line
    assert check(config, prefix) == (True, ["somefont 1.0: installed"])
    ok, (line,) = check(config, scratch / "another-venv")
    assert not ok and "run bin/build_venv.sh" in line


def test_a_download_that_does_not_match_its_pin_is_refused():
    scratch = make_scratch_dir("fonts_bad_checksum")
    config = _pinned_zip(scratch, sha256="0" * 64)
    with pytest.raises(SystemExit, match="pins 0000"):
        install(config, scratch / "venv", scratch / "downloads")
    assert not installed_dir("somefont", scratch / "venv").exists()
    assert not any((scratch / "downloads").iterdir())


def test_the_repositorys_pin_is_the_installed_font():
    (pin,) = read_pins()
    assert pin.name == "tex-gyre-heros"
    assert (installed_dir(pin.name) / "SOURCE.txt").read_text() == pin.stamp()


def test_panel_fonts_are_files_in_the_venv_or_matplotlib():
    assert font_file("tex-gyre-heros") == installed_dir("tex-gyre-heros") / "texgyreheros-regular.otf"
    assert font_file("dejavu-sans").name == "DejaVuSans.ttf" and font_file("dejavu-sans").is_file()


def test_a_font_not_installed_stops_the_run_and_says_how_to_install_it(monkeypatch):
    scratch = make_scratch_dir("fonts_missing")
    monkeypatch.setattr(fonts, "installed_dir", lambda name, prefix=None: scratch / name)
    with pytest.raises(FontNotInstalled, match="run bin/build_venv.sh"):
        font_file("tex-gyre-heros")
    with pytest.raises(RequestError, match=r"run bin/build_venv.sh \(it installs config/fonts.txt\), or choose --panel-font dejavu-sans"):
        check_request(PlotRequest("x.fits", "amp-vs-freq"))
    check_request(PlotRequest("x.fits", "amp-vs-freq", panel_font="dejavu-sans"))


def test_the_panel_draws_with_the_chosen_fonts_file_and_the_record_names_it():
    from visplot.records import PlotRecord

    scratch = make_scratch_dir("fonts_panel")
    path = _make_synthetic_file(scratch / "obs.fits")
    for name in ("tex-gyre-heros", "dejavu-sans"):
        request = PlotRequest(str(path), "amp-vs-freq", panel_font=name, provenance_dir=str(scratch / "runs"))
        run = prepare(request)
        (figure,) = run.xy_figures.values()
        assert figure.panel.value_props["fontproperties"].get_file() == str(font_file(name))
        record = PlotRecord(request, "plot")
        record.finish()
        details = json.loads(record.path.read_text())["details"]["panel_font"]
        assert details == {"name": name, "file": str(font_file(name)),
                           "sha256": hashlib.sha256(font_file(name).read_bytes()).hexdigest()}
