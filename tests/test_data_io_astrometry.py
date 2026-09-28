import os

import numpy as np
import pytest
from astropy.coordinates import EarthLocation
from astropy.time import Time
from astropy.utils import iers
import astropy.units as u

from data_io.antenna_table import read_array_earth_location
from data_io.astrometry import (
    AstrometryWarning,
    Ut1Provider,
    altaz_deg,
    hour_angle_hours,
    local_sidereal_time_hours,
    parallactic_angle_deg,
)
from data_io.source_table import read_source_table

REAL_GWB_FITS = "/data1/gmrt/40_014_25JUL2021/40_014_25jul2021_2.6s_gwb.FITS"

GMRT_LOCATION = EarthLocation.from_geodetic(lon=74.050 * u.deg, lat=19.093 * u.deg, height=640 * u.m)
JD = 2460123.456


def _lst_hours_directly(jd, location):
    """astropy's own computation at the site longitude, from its bundled tables."""
    with iers.conf.set_temp("auto_download", False):
        return Time(jd, format="jd", scale="utc").sidereal_time("apparent", longitude=location.lon).hour


def test_local_sidereal_time_hours_matches_astropy():
    difference_s = abs(local_sidereal_time_hours(JD, GMRT_LOCATION) - _lst_hours_directly(JD, GMRT_LOCATION)) * 3600
    assert difference_s < 1e-3  # measured 1e-6 s


def test_hour_angle_hours_is_zero_at_transit():
    ra_deg = local_sidereal_time_hours(JD, GMRT_LOCATION) * 15.0
    assert hour_angle_hours(JD, ra_deg, GMRT_LOCATION) == pytest.approx(0.0, abs=1e-9)


def test_hour_angle_hours_wraps_to_plus_minus_12():
    # A source 1 hour of RA east of the transit position is 1 hour before
    # transit -- HA = -1, not +23.
    transit_ra_deg = _lst_hours_directly(JD, GMRT_LOCATION) * 15.0
    ra_deg = (transit_ra_deg + 15.0) % 360.0
    ha = hour_angle_hours(JD, ra_deg, GMRT_LOCATION)
    assert -12.0 <= ha < 12.0
    assert ha == pytest.approx(-1.0)


def test_altaz_deg_matches_known_transit_geometry():
    # A source on the celestial equator (dec=0), south of zenith for a
    # 19.093N observer, transits due south (az=180) at elevation
    # 90 - latitude.
    ra_deg = _lst_hours_directly(JD, GMRT_LOCATION) * 15.0
    az_deg, el_deg = altaz_deg(JD, ra_deg, 0.0, GMRT_LOCATION)
    assert az_deg == pytest.approx(180.0, abs=1e-6)
    assert el_deg == pytest.approx(90.0 - 19.093, abs=1e-6)


def test_parallactic_angle_deg_is_zero_at_transit_south_of_zenith():
    # Zero at transit when the source culminates south of zenith (dec <
    # latitude, as here); a source culminating north of zenith (dec >
    # latitude) is +-180 instead -- the parallactic angle does not pass
    # through zero there, it jumps.
    ra_deg = _lst_hours_directly(JD, GMRT_LOCATION) * 15.0
    pa_deg = parallactic_angle_deg(JD, ra_deg, 0.0, GMRT_LOCATION)
    assert pa_deg == pytest.approx(0.0, abs=1e-6)


@pytest.mark.skipif(not os.path.exists(REAL_GWB_FITS), reason="real GWB raw data file not present on this host")
def test_astrometry_runs_end_to_end_against_the_real_gwb_file():
    location = read_array_earth_location(REAL_GWB_FITS)
    sources = read_source_table(REAL_GWB_FITS)
    threeC286 = sources[1]
    jd = 2459421.5  # 2021-07-25, the observation's own date

    ha = hour_angle_hours(jd, threeC286.ra_apparent_deg, location)
    az_deg, el_deg = altaz_deg(jd, threeC286.ra_apparent_deg, threeC286.dec_apparent_deg, location)
    pa_deg = parallactic_angle_deg(jd, threeC286.ra_apparent_deg, threeC286.dec_apparent_deg, location)

    assert -12.0 <= ha < 12.0
    assert 0.0 <= az_deg < 360.0
    assert -90.0 <= el_deg <= 90.0
    assert -180.0 <= pa_deg <= 180.0


def test_ut1_comes_from_the_bundled_tables_without_network_access(monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("the online IERS tables were consulted for a date the bundled tables cover")

    monkeypatch.setattr(iers.IERS_Auto, "open", no_network)
    provider = Ut1Provider()
    values = provider.ut1_minus_utc_s([2459421.2, 2459421.3])
    expected = iers.IERS_B.open().ut1_utc(Time([2459421.2, 2459421.3], format="jd", scale="utc")).to_value("s")
    np.testing.assert_allclose(values, expected)
    assert provider.sources_used == {"IERS-B (bundled with astropy)"}
    assert not provider.fallback_used


def test_ut1_falls_back_to_utc_with_a_warning_when_no_table_covers_the_date(monkeypatch):
    def offline(*args, **kwargs):
        raise OSError("no network")

    monkeypatch.setattr(iers.IERS_Auto, "open", offline)
    provider = Ut1Provider()
    far_future = 2488069.5  # 2100-01-01, beyond every IERS table
    with pytest.warns(AstrometryWarning, match="UT1 = UTC"):
        values = provider.ut1_minus_utc_s([far_future])
    np.testing.assert_array_equal(values, [0.0])
    assert provider.fallback_used
    assert provider.sources_used == {"none: UT1 = UTC assumed"}


def test_ut1_offline_provider_skips_the_online_tables(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("online tables consulted with allow_online=False")

    monkeypatch.setattr(iers.IERS_Auto, "open", fail)
    provider = Ut1Provider(allow_online=False)
    with pytest.warns(AstrometryWarning):
        provider.ut1_minus_utc_s([2488069.5])
    assert provider.fallback_used
