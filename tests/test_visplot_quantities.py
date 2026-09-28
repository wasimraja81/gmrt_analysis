import numpy as np
import pytest
from astropy.coordinates import EarthLocation
import astropy.units as u

from data_io.astrometry import altaz_deg, hour_angle_hours, parallactic_angle_deg
from data_io.source_table import Source
from data_io.visibility_data import VisibilityBlock
from visplot.quantities import (
    QUANTITIES,
    QuantityContext,
    category_label,
    context_from_source_table,
    quantity_label,
)

GMRT = EarthLocation.from_geodetic(74.05 * u.deg, 19.09 * u.deg, 650 * u.m)
JD0 = 2459421.2  # 2021-07-25, the GWB observation's date


def _block(with_data=True):
    # 2 rows x 2 Stokes x 2 channels.
    data = np.array([
        [[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]],
        [[9 + 10j, 11 + 12j], [13 + 14j, -1 + 0j]],
    ])
    return VisibilityBlock(
        row_indices=np.array([0, 1]),
        data=data if with_data else None,
        weight=np.ones((2, 2, 2)) if with_data else None,
        axis_types=["STOKES", "FREQ"],
        axis_indices={"STOKES": np.array([0, 1]), "FREQ": np.array([0, 1])},
        ant1=np.array([1, 1]), ant2=np.array([2, 2]),
        source_id=np.array([1, 2]),
        jd=np.array([JD0, JD0 + 0.5]),
        uu_sec=np.array([1e-6, 2e-6]), vv_sec=np.array([3e-6, 4e-6]), ww_sec=np.array([5e-6, 6e-6]),
        chan_freqs_hz=np.array([1e9, 1.1e9]),
        stokes_labels=["RR", "LL"],
    )


def _source(sid, name, ra, dec):
    return Source(
        id=sid, name=name, ra_epoch_deg=ra, dec_epoch_deg=dec, ra_apparent_deg=ra, dec_apparent_deg=dec,
        epoch_year=2000.0, calcode="", flux_i_jy=(0.0,), flux_q_jy=(0.0,), flux_u_jy=(0.0,), flux_v_jy=(0.0,),
    )


def _sources():
    return {1: _source(1, "3C286", 202.8, 30.5), 2: _source(2, "3C48", 24.4, 33.2)}


def _ctx():
    return context_from_source_table(JD0, _sources(), GMRT, bunit="UNCALIB", stokes_labels=["RR", "LL"])


def _eval(name, block=None):
    block = block if block is not None else _block()
    return QUANTITIES[name].evaluate(block, _ctx())


def test_data_quantities_have_the_full_sample_shape():
    block = _block()
    np.testing.assert_array_equal(_eval("real", block), block.data.real)
    np.testing.assert_array_equal(_eval("imag", block), block.data.imag)
    np.testing.assert_allclose(_eval("amp", block), np.abs(block.data))
    np.testing.assert_allclose(_eval("phase_deg", block), np.degrees(np.angle(block.data)))


def test_row_quantities_vary_only_along_rows():
    time_h = _eval("time_h")
    assert time_h.shape == (2, 1, 1)
    np.testing.assert_allclose(time_h.ravel(), [0.0, 12.0])  # hours since the context's reference JD
    np.testing.assert_allclose(_eval("uvdist_m").ravel(), np.hypot([1e-6, 2e-6], [3e-6, 4e-6]) * 299_792_458.0)


def test_time_h_uses_the_shared_reference_so_chunks_agree():
    later_chunk = _block()
    later_chunk = VisibilityBlock(**{**later_chunk.__dict__, "jd": np.array([JD0 + 0.5, JD0 + 1.0])})
    np.testing.assert_allclose(_eval("time_h", later_chunk).ravel(), [12.0, 24.0])


def test_klambda_quantities_vary_along_rows_and_channels_only():
    u_kl = _eval("u_klambda")
    assert u_kl.shape == (2, 1, 2)
    np.testing.assert_allclose(u_kl[:, 0, :], np.outer([1e-6, 2e-6], [1e9, 1.1e9]) / 1e3)


def test_freq_mhz_varies_only_along_channels():
    freq = _eval("freq_mhz")
    assert freq.shape == (1, 1, 2)
    np.testing.assert_allclose(freq.ravel(), [1000.0, 1100.0])


def test_pairing_two_quantities_broadcasts_to_the_shape_the_pair_needs():
    x, y = np.broadcast_arrays(_eval("time_h"), _eval("u_sec"))
    assert x.shape == (2, 1, 1)
    x, y = np.broadcast_arrays(_eval("freq_mhz"), _eval("amp"))
    assert x.shape == (2, 2, 2)


def test_geometry_quantities_match_astrometry():
    block = _block(with_data=False)
    ra = np.array([202.8, 24.4])
    dec = np.array([30.5, 33.2])
    np.testing.assert_allclose(_eval("ha_h", block).ravel(), hour_angle_hours(block.jd, ra, GMRT))
    az, el = altaz_deg(block.jd, ra, dec, GMRT)
    np.testing.assert_allclose(_eval("az_deg", block).ravel(), az)
    np.testing.assert_allclose(_eval("el_deg", block).ravel(), el)
    np.testing.assert_allclose(_eval("pa_deg", block).ravel(), parallactic_angle_deg(block.jd, ra, dec, GMRT))


def test_categories_evaluate_to_codes_with_labels():
    ctx = _ctx()
    stokes = _eval("stokes")
    assert stokes.shape == (1, 2, 1)
    assert [category_label("stokes", c, ctx) for c in stokes.ravel()] == ["RR", "LL"]
    source = _eval("source")
    assert [category_label("source", c, ctx) for c in source.ravel()] == ["3C286", "3C48"]


def test_only_data_quantities_need_data():
    assert {q.name for q in QUANTITIES.values() if q.needs_data} == {"real", "imag", "amp", "phase_deg"}


def test_quantity_label_units():
    ctx = _ctx()
    assert quantity_label("freq_mhz", ctx) == "Frequency (MHz)"
    assert quantity_label("amp", ctx) == "Amplitude (UNCALIB)"
    assert quantity_label("amp", QuantityContext(time_reference_jd=0.0)) == "Amplitude"
    assert quantity_label("stokes", ctx) == "Stokes"
    assert quantity_label("u_klambda", ctx) == "U (kλ)"


def test_klambda_without_a_freq_axis_raises():
    block = VisibilityBlock(**{**_block().__dict__, "chan_freqs_hz": None})
    with pytest.raises(ValueError, match="no channel frequencies"):
        _eval("u_klambda", block)
