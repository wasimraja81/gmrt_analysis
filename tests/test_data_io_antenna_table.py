import os

import numpy as np
import pytest
from astropy.io import fits

from data_io.antenna_table import Antenna, read_antenna_table, read_array_reference_position_m

from conftest import make_scratch_dir

REAL_GWB_FITS = "/data1/gmrt/40_014_25JUL2021/40_014_25jul2021_2.6s_gwb.FITS"


def _make_synthetic_an_table(path, array_xyz=(1000.0, 2000.0, 3000.0)):
    an_columns = fits.ColDefs([
        fits.Column(name="NOSTA", format="J", array=np.array([1, 2, 3], dtype=np.int32)),
        fits.Column(name="ANNAME", format="8A", array=np.array(["W:01", "X:02", "Y:03"])),
        fits.Column(name="STABXYZ", format="3D", array=np.array([
            [10.0, 20.0, 30.0],
            [-5.0, 0.0, 15.0],
            [0.0, 0.0, 0.0],
        ])),
    ])
    an_hdu = fits.BinTableHDU.from_columns(an_columns, name="AIPS AN")
    an_hdu.header["ARRAYX"] = array_xyz[0]
    an_hdu.header["ARRAYY"] = array_xyz[1]
    an_hdu.header["ARRAYZ"] = array_xyz[2]
    fits.HDUList([fits.PrimaryHDU(), an_hdu]).writeto(path)
    return path


def test_read_antenna_table_returns_absolute_ecef_positions():
    scratch = make_scratch_dir("data_io_antenna_table")
    path = scratch / "synthetic.fits"
    _make_synthetic_an_table(path, array_xyz=(1000.0, 2000.0, 3000.0))

    antennas = read_antenna_table(path)

    assert antennas == [
        Antenna(station_number=1, name="W:01", x_m=1010.0, y_m=2020.0, z_m=3030.0),
        Antenna(station_number=2, name="X:02", x_m=995.0, y_m=2000.0, z_m=3015.0),
        Antenna(station_number=3, name="Y:03", x_m=1000.0, y_m=2000.0, z_m=3000.0),
    ]


def test_read_antenna_table_does_not_assume_a_naming_convention():
    # Names here don't follow GMRT's "<code>:<station>" pattern at all --
    # the generic reader must not care, only instruments/gmrt/ does.
    scratch = make_scratch_dir("data_io_antenna_table")
    path = scratch / "synthetic.fits"
    _make_synthetic_an_table(path)

    antennas = read_antenna_table(path)

    assert [a.name for a in antennas] == ["W:01", "X:02", "Y:03"]


def test_read_array_reference_position_m():
    scratch = make_scratch_dir("data_io_antenna_table")
    path = scratch / "synthetic.fits"
    _make_synthetic_an_table(path, array_xyz=(1657004.629, 5797894.3801, 2073303.1705))

    assert read_array_reference_position_m(path) == (1657004.629, 5797894.3801, 2073303.1705)


@pytest.mark.skipif(not os.path.exists(REAL_GWB_FITS), reason="real GWB raw data file not present on this host")
def test_antenna_positions_resolve_to_gmrts_known_location_against_the_real_file():
    # The point of storing absolute ECEF, not the raw relative STABXYZ:
    # converting the array reference position straight to geodetic
    # coordinates must land at GMRT's actual, published location (near
    # Khodad, Maharashtra, ~19.1N 74.05E) -- confirmed directly, 2026-09-25.
    from astropy.coordinates import EarthLocation
    import astropy.units as u

    x, y, z = read_array_reference_position_m(REAL_GWB_FITS)
    location = EarthLocation.from_geocentric(x, y, z, unit=u.m)

    assert location.lat.deg == pytest.approx(19.093, abs=0.01)
    assert location.lon.deg == pytest.approx(74.050, abs=0.01)

    antennas = read_antenna_table(REAL_GWB_FITS)
    assert len(antennas) == 30
    # Every antenna's own absolute position must also land near GMRT, not
    # just the array reference point -- antennas are spread over a ~25km
    # array, so a looser tolerance than the reference point itself.
    for antenna in antennas:
        antenna_location = EarthLocation.from_geocentric(antenna.x_m, antenna.y_m, antenna.z_m, unit=u.m)
        assert antenna_location.lat.deg == pytest.approx(19.093, abs=0.2)
        assert antenna_location.lon.deg == pytest.approx(74.050, abs=0.2)
