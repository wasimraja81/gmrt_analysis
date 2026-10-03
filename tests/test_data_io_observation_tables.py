"""The header keywords and tables a listing reads (T20): the primary
header's description, the FQ table's frequency setups, the AN table's mounts
and feeds, and the rest of the SU table -- each optional, as other files may
lack them."""

import os

import numpy as np
import pytest
from astropy.io import fits

from conftest import make_scratch_dir
from data_io.antenna_table import MOUNT_TYPES, read_antenna_feeds
from data_io.frequency_table import read_frequency_setups
from data_io.observation_header import read_observation_header
from data_io.source_table import read_source_table, read_source_table_frame
from test_cli_visplot_output import _make_synthetic_file

REAL_GWB_FITS = "/data1/gmrt/40_014_25JUL2021/40_014_25jul2021_2.6s_gwb.FITS"


def test_a_file_without_the_tables_or_keywords_gives_what_it_has():
    path = _make_synthetic_file(make_scratch_dir("observation_tables_bare") / "obs.fits")
    header = read_observation_header(path)
    assert (header.telescope, header.bunit, header.reference_freq_hz) == ("GMRT", "UNCALIB", 400e6)
    assert (header.observer, header.object_name, header.date_obs, header.epoch) == (None, None, None, None)
    assert read_frequency_setups(path) == []  # no FQ table
    feeds = read_antenna_feeds(path)
    assert sorted(feeds) == [1, 2, 3] and feeds[1].mount_type is None and feeds[1].pol_type_a == ""
    assert read_source_table_frame(path).n_if is None
    assert read_source_table(path)[1].bandwidth_hz is None


def test_several_frequency_setups_are_read_one_per_row():
    path = make_scratch_dir("observation_tables_fq") / "fq.fits"
    fq = fits.BinTableHDU.from_columns(fits.ColDefs([
        fits.Column(name="FRQSEL", format="J", array=np.array([1, 2], dtype=np.int32)),
        fits.Column(name="IF FREQ", format="2D", unit="Hz", array=np.array([[0.0, 16e6], [0.0, 32e6]])),
        fits.Column(name="CH WIDTH", format="2E", unit="Hz", array=np.array([[1e5, 1e5], [-2e5, -2e5]])),
        fits.Column(name="TOTAL BANDWIDTH", format="2E", unit="Hz", array=np.array([[16e6, 16e6], [32e6, 32e6]])),
        fits.Column(name="SIDEBAND", format="2J", array=np.array([[1, 1], [-1, -1]], dtype=np.int32)),
    ]), name="AIPS FQ")
    fits.HDUList([fits.PrimaryHDU(), fq]).writeto(path, overwrite=True)
    first, second = read_frequency_setups(path)
    assert (first.id, first.if_offset_hz, first.channel_width_hz, first.sideband) == (1, (0.0, 16e6), (1e5, 1e5), (1, 1))
    assert (second.id, second.total_bandwidth_hz, second.sideband) == (2, (32e6, 32e6), (-1, -1))


@pytest.mark.skipif(not os.path.exists(REAL_GWB_FITS), reason="the archival GWB file is not on this host")
def test_the_gwb_files_description_tables_and_feeds():
    header = read_observation_header(REAL_GWB_FITS)
    assert (header.telescope, header.instrument, header.observer, header.object_name, header.date_obs) == (
        "GMRT", "GMRT", "WasimRaja", "MULTI", "2021-07-24")
    (setup,) = read_frequency_setups(REAL_GWB_FITS)  # one setup: the header's FREQ axis describes it
    assert (setup.if_offset_hz, setup.channel_width_hz, setup.total_bandwidth_hz, setup.sideband) == (
        (0.0,), (-97656.25,), (2e8,), (1,))
    feeds = read_antenna_feeds(REAL_GWB_FITS)
    assert len(feeds) == 30 and {(f.pol_type_a, f.pol_type_b) for f in feeds.values()} == {("R", "L")}
    assert {MOUNT_TYPES[f.mount_type] for f in feeds.values()} == {"alt-azimuth"}
    frame = read_source_table_frame(REAL_GWB_FITS)
    assert (frame.n_if, frame.velocity_type, frame.velocity_definition) == (1, "TOPOCENT", "RADIO")
    source = read_source_table(REAL_GWB_FITS)[1]
    assert (source.qualifier, source.bandwidth_hz, source.freq_offset_hz) == (0, 2e8, (0.0, 0.0))


def test_a_file_of_several_frequency_setups_is_not_plotted_but_its_tables_are():
    from dataclasses import replace

    from data_io.frequency_table import FrequencySetup
    from visplot.request import PlotRequest
    from visplot.run import RequestError, count_selection, open_file, prepare

    path = _make_synthetic_file(make_scratch_dir("observation_tables_two_setups") / "obs.fits")
    setups = [FrequencySetup(1, (0.0,), (1e6,), (4e6,), (1,)), FrequencySetup(2, (8e6,), (1e6,), (4e6,), (1,))]
    opened = replace(open_file(path), frequency_setups=setups)
    for attempt in (lambda: prepare(PlotRequest(str(path), "amp-vs-freq"), opened),
                    lambda: count_selection(PlotRequest(str(path), "amp-vs-freq"), opened)):
        with pytest.raises(RequestError, match="holds 2 frequency setups"):
            attempt()
    prepare(PlotRequest(str(path), "antenna-layout"), opened)  # a table plot: the tables only
