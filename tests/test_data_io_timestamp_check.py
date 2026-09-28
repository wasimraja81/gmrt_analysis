import os
from types import SimpleNamespace

import numpy as np
import pytest

from data_io.antenna_table import Antenna
from data_io.astrometry import DEFAULT_UT1
from data_io.source_table import Source
from data_io.timestamp_check import RMS_LIMIT_M, check_timestamps, uvw_seconds

REAL_GWB_FITS = "/data1/gmrt/40_014_25JUL2021/40_014_25jul2021_2.6s_gwb.FITS"
GMRT_CENTRE_M = np.array([1657004.6, 5797894.4, 2073303.2])  # ECEF, from the GWB file's ARRAYX/Y/Z


def _source(sid, ra, dec):
    return Source(id=sid, name=f"s{sid}", ra_epoch_deg=ra, dec_epoch_deg=dec, ra_apparent_deg=ra + 0.3,
                  dec_apparent_deg=dec + 0.1, epoch_year=2000.0, calcode="", flux_i_jy=(), flux_q_jy=(),
                  flux_u_jy=(), flux_v_jy=(), )


def _synthetic(offset_s, frame="J2000", n_integrations=8, noise_m=0.0):
    """An index-like object whose u, v, w were computed for timestamps
    `offset_s` after UTC, in `frame`, with baseline = ant2 - ant1."""
    rng = np.random.default_rng(1)
    antennas = [Antenna(k + 1, f"A{k + 1}", *(GMRT_CENTRE_M + rng.uniform(-12e3, 12e3, 3)))
                for k in range(6)]
    positions = {a.station_number: np.array([a.x_m, a.y_m, a.z_m]) for a in antennas}
    sources = {1: _source(1, 202.8, 30.5), 2: _source(2, 350.9, 58.8)}
    pairs = [(a, b) for a in range(1, 7) for b in range(a + 1, 7)]
    rows = {k: [] for k in ("jd", "ant1", "ant2", "source_id", "uu", "vv", "ww")}
    bounds = [0]
    for i in range(n_integrations):
        jd = 2459421.2 + i * 0.03
        sid = 1 + i % 2
        s = sources[sid]
        ra, dec = (s.ra_epoch_deg, s.dec_epoch_deg) if frame == "J2000" else (s.ra_apparent_deg, s.dec_apparent_deg)
        baselines = np.stack([positions[b] - positions[a] for a, b in pairs])
        uvw = uvw_seconds(jd - offset_s / 86400, baselines, np.radians(ra), np.radians(dec), frame, DEFAULT_UT1)[0]
        uvw = uvw + rng.normal(0, noise_m / 299_792_458.0, uvw.shape)
        for (a, b), (u, v, w) in zip(pairs, uvw):
            for key, value in zip(rows, (jd, a, b, sid, u, v, w)):
                rows[key].append(value)
        bounds.append(len(rows["jd"]))
    index = SimpleNamespace(
        jd=np.array(rows["jd"]), ant1=np.array(rows["ant1"]), ant2=np.array(rows["ant2"]),
        source_id=np.array(rows["source_id"]), uu_sec=np.array(rows["uu"], dtype=np.float32),
        vv_sec=np.array(rows["vv"], dtype=np.float32), ww_sec=np.array(rows["ww"], dtype=np.float32),
        integration_boundaries=np.array(bounds),
    )
    return index, antennas, sources


def test_the_offset_the_u_v_w_were_computed_for_is_recovered():
    index, antennas, sources = _synthetic(offset_s=35.0)
    check = check_timestamps(index, antennas, sources, declared_s=35.0)
    assert check.measured_s == pytest.approx(35.0, abs=0.01)
    assert check.frame == "J2000" and check.n_used == 8 and check.rms_m < 0.01
    assert check.agrees
    assert check.summary().endswith("0.000 s apart, within the 0.1 s tolerance")


def test_an_offset_other_than_the_declared_one_disagrees():
    index, antennas, sources = _synthetic(offset_s=34.07)
    check = check_timestamps(index, antennas, sources, declared_s=35.0)
    assert check.measured_s == pytest.approx(34.07, abs=0.01)
    assert check.agrees is False
    assert "0.930 s apart, above the 0.1 s tolerance" in check.summary()


def test_the_apparent_frame_is_recognised():
    index, antennas, sources = _synthetic(offset_s=0.0, frame="apparent")
    check = check_timestamps(index, antennas, sources, declared_s=0.0)
    assert check.frame == "apparent" and check.measured_s == pytest.approx(0.0, abs=0.01)


def test_u_v_w_matching_no_geometry_leave_the_offset_unmeasured():
    index, antennas, sources = _synthetic(offset_s=0.0, noise_m=50 * RMS_LIMIT_M)
    check = check_timestamps(index, antennas, sources, declared_s=0.0)
    assert check.measured_s is None and check.agrees is None
    assert "could not be measured" in check.summary()


@pytest.mark.skipif(not os.path.exists(REAL_GWB_FITS), reason="real GWB raw data file not present on this host")
def test_the_gwb_files_u_v_w_put_its_timestamps_34_s_after_utc():
    from data_io.antenna_table import read_antenna_table, read_time_reference
    from data_io.row_index import default_row_index_path, load_row_index
    from data_io.source_table import read_source_table

    index = load_row_index(default_row_index_path(REAL_GWB_FITS))
    declared = read_time_reference(REAL_GWB_FITS).recorded_minus_utc_s
    check = check_timestamps(index, read_antenna_table(REAL_GWB_FITS), read_source_table(REAL_GWB_FITS), declared)
    assert declared == 35.0
    assert check.measured_s == pytest.approx(34.078, abs=0.02)  # measured 2026-09-28
    assert check.frame == "J2000" and check.rms_m < 0.01 and check.agrees is False
