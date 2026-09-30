import os

import pytest

from data_io.antenna_table import Antenna
from data_io.row_index import default_row_index_path
from instruments.structural_duds import structural_dud_names, without_structural_duds

REAL_GSB_FITS = "/data1/gmrt/40_014_25JUL2021/40_014_25jul2021_gsb.FITS"


def _antennas(*names):
    return [Antenna(station_number=n, name=name, x_m=0.0, y_m=0.0, z_m=0.0) for n, name in enumerate(names, 1)]


def test_the_structural_duds_are_gmrts_by_telescop():
    assert structural_dud_names("GMRT") == ["C07", "S05"]
    assert structural_dud_names(" gmrt ") == ["C07", "S05"]
    assert structural_dud_names("VLA") == [] and structural_dud_names(None) == []


def test_a_gmrt_table_loses_its_dud_entries_and_another_telescopes_keeps_every_entry():
    table = _antennas("C00:01", "W06:02", "C07:03", "S05:04")
    gmrt = without_structural_duds(table, "GMRT")
    assert [a.name for a in gmrt.active_antennas] == ["C00:01", "W06:02"]
    assert [a.name for a in gmrt.dud_antennas] == ["C07:03", "S05:04"]
    assert without_structural_duds(table, "OTHER").active_antennas == table
    gwb_like = without_structural_duds(_antennas("C00:01", "W06:02"), "GMRT")  # GWB's table has no DUD entries
    assert len(gwb_like.active_antennas) == 2 and gwb_like.unmatched_dud_names == ["C07", "S05"]


@pytest.mark.skipif(not (os.path.exists(REAL_GSB_FITS) and default_row_index_path(REAL_GSB_FITS).exists()),
                    reason="archival GSB raw data file or its row index not present on this host")
def test_visplot_opens_the_gsb_file_with_its_30_antennas():
    from visplot.antenna_layout import antenna_layout
    from visplot.request_args import resolve_antennas_arg
    from visplot.run import open_file

    opened = open_file(REAL_GSB_FITS)
    assert len(opened.antennas) == 30
    assert [a.name for a in opened.dud_antennas] == ["C07:31", "S05:32"]
    with pytest.raises(ValueError):
        resolve_antennas_arg("C07", opened.antennas)
    assert resolve_antennas_arg("W06", opened.antennas) == [30]

    fig = antenna_layout(opened.antennas, opened.array_location, telescope=opened.telescope,
                         left_out=[a.name for a in opened.dud_antennas])
    assert len(fig.axes[0].collections[0].get_offsets()) == 30
    assert any(t.get_text().startswith("Left out: C07:31, S05:32") for t in fig.texts)
