import os

import numpy as np
import pytest
from astropy.io import fits

from data_io.source_table import Source, read_source_table

from conftest import make_scratch_dir

REAL_GWB_FITS = "/data1/gmrt/40_014_25JUL2021/40_014_25jul2021_2.6s_gwb.FITS"


def _make_synthetic_su_table(path):
    su_columns = fits.ColDefs([
        fits.Column(name="ID. NO.", format="J", array=np.array([1, 2], dtype=np.int32)),
        fits.Column(name="SOURCE", format="16A", array=np.array(["3C48", "3C286"])),
        fits.Column(name="CALCODE", format="4A", array=np.array(["", "C"])),
        fits.Column(name="RAEPO", format="D", array=np.array([24.4222, 202.7845])),
        fits.Column(name="DECEPO", format="D", array=np.array([33.1598, 30.5091])),
        fits.Column(name="RAAPP", format="D", array=np.array([24.65, 203.03])),
        fits.Column(name="DECAPP", format="D", array=np.array([33.05, 30.40])),
        fits.Column(name="EPOCH", format="D", array=np.array([2000.0, 2000.0])),
        fits.Column(name="IFLUX", format="2D", array=np.array([[16.5, 16.5], [0.0, 0.0]])),
        fits.Column(name="QFLUX", format="2D", array=np.array([[0.1, 0.1], [0.0, 0.0]])),
        fits.Column(name="UFLUX", format="2D", array=np.array([[0.05, 0.05], [0.0, 0.0]])),
        fits.Column(name="VFLUX", format="2D", array=np.array([[0.0, 0.0], [0.0, 0.0]])),
    ])
    su_hdu = fits.BinTableHDU.from_columns(su_columns, name="AIPS SU")
    fits.HDUList([fits.PrimaryHDU(), su_hdu]).writeto(path)
    return path


def test_read_source_table_recovers_known_fields():
    scratch = make_scratch_dir("data_io_source_table")
    path = scratch / "synthetic.fits"
    _make_synthetic_su_table(path)

    sources = read_source_table(path)

    assert set(sources.keys()) == {1, 2}
    assert sources[1] == Source(
        id=1, name="3C48", ra_epoch_deg=24.4222, dec_epoch_deg=33.1598,
        ra_apparent_deg=24.65, dec_apparent_deg=33.05, epoch_year=2000.0,
        calcode="", flux_i_jy=(16.5, 16.5), flux_q_jy=(0.1, 0.1), flux_u_jy=(0.05, 0.05), flux_v_jy=(0.0, 0.0),
    )
    assert sources[2].calcode == "C"
    assert sources[2].name == "3C286"


def test_read_source_table_returns_empty_dict_when_no_su_table():
    scratch = make_scratch_dir("data_io_source_table")
    path = scratch / "no_su.fits"
    fits.HDUList([fits.PrimaryHDU()]).writeto(path)

    assert read_source_table(path) == {}


@pytest.mark.skipif(not os.path.exists(REAL_GWB_FITS), reason="real GWB raw data file not present on this host")
def test_read_source_table_against_the_real_gwb_file():
    sources = read_source_table(REAL_GWB_FITS)

    assert len(sources) == 13
    threeC286 = sources[1]
    assert threeC286.name == "3C286"
    assert threeC286.ra_epoch_deg == pytest.approx(202.78451255313814)
    assert threeC286.dec_epoch_deg == pytest.approx(30.50913349559711)
    assert threeC286.epoch_year == 2000.0
    # Confirmed directly, 2026-09-25: no calibrator intent was recorded for
    # any source in this observation.
    assert threeC286.calcode == ""
    assert threeC286.flux_i_jy == (0.0, 0.0)
