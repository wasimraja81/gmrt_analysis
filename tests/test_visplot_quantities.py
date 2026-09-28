import numpy as np
import pytest
from astropy.coordinates import EarthLocation
import astropy.units as u

from data_io.astrometry import altaz_deg, hour_angle_hours, local_sidereal_time_hours, parallactic_angle_deg
from data_io.source_table import Source
from data_io.visibility_data import VisibilityBlock
from visplot.quantities import (
    ALIASES,
    QUANTITIES,
    QuantityContext,
    category_label,
    context_from_source_table,
    convert,
    evaluate,
    quantity_label,
    resolve_unit,
    units_of,
    value_description,
)

GMRT = EarthLocation.from_geodetic(74.05 * u.deg, 19.09 * u.deg, 650 * u.m)
JD0 = 2459421.2  # 2021-07-25 16:48 UTC (22:18 IST), the GWB observation's date
C = 299_792_458.0
REF_JD = 2459419.5  # 2021-07-24 0h: the GWB file's RDATE is the day before its first integration


def _block(with_data=True, jd=None):
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
        jd=np.array([JD0, JD0 + 0.5]) if jd is None else np.asarray(jd),
        uu_sec=np.array([1e-6, 2e-6]), vv_sec=np.array([3e-6, 4e-6]), ww_sec=np.array([5e-6, 6e-6]),
        chan_freqs_hz=np.array([1e9, 1.1e9]),
        stokes_labels=["RR", "LL"],
    )


def _row_block(jd):
    """One row per JD, no data, one channel."""
    n = len(jd)
    return VisibilityBlock(
        row_indices=np.arange(n), data=None, weight=None, axis_types=["STOKES", "FREQ"],
        axis_indices={"STOKES": np.array([0]), "FREQ": np.array([0])}, ant1=np.ones(n, int), ant2=np.full(n, 2),
        source_id=np.ones(n, int), jd=np.asarray(jd), uu_sec=np.zeros(n), vv_sec=np.zeros(n), ww_sec=np.zeros(n),
        chan_freqs_hz=np.array([1e9]), stokes_labels=["RR"],
    )


def _source(sid, name, ra, dec):
    return Source(
        id=sid, name=name, ra_epoch_deg=ra, dec_epoch_deg=dec, ra_apparent_deg=ra, dec_apparent_deg=dec,
        epoch_year=2000.0, calcode="", flux_i_jy=(0.0,), flux_q_jy=(0.0,), flux_u_jy=(0.0,), flux_v_jy=(0.0,),
    )


def _sources():
    return {1: _source(1, "3C286", 202.8, 30.5), 2: _source(2, "3C48", 24.4, 33.2)}


def _ctx(bunit="UNCALIB", time_zone="Asia/Kolkata", reference_date_jd=None, time_system=None):
    return context_from_source_table(JD0, _sources(), GMRT, bunit=bunit, stokes_labels=["RR", "LL"],
                                     time_zone=time_zone, reference_date_jd=reference_date_jd,
                                     time_system=time_system)


def _eval(name, block=None, unit=None, ctx=None):
    block = block if block is not None else _block()
    return evaluate(name, block, ctx if ctx is not None else _ctx(), unit)


def test_data_quantities_have_the_full_sample_shape():
    block = _block()
    np.testing.assert_array_equal(_eval("real", block), block.data.real)
    np.testing.assert_array_equal(_eval("imag", block), block.data.imag)
    np.testing.assert_allclose(_eval("amp", block), np.abs(block.data))
    np.testing.assert_allclose(_eval("phase", block), np.degrees(np.angle(block.data)))
    np.testing.assert_allclose(_eval("phase", block, "rad"), np.angle(block.data))


def test_row_quantities_vary_only_along_rows():
    time_h = _eval("time", unit="h")
    assert time_h.shape == (2, 1, 1)
    np.testing.assert_allclose(time_h.ravel(), [0.0, 12.0])  # hours since the context's reference JD
    np.testing.assert_allclose(_eval("uvdist", unit="m").ravel(), np.hypot([1e-6, 2e-6], [3e-6, 4e-6]) * C)


def test_time_uses_the_shared_reference_so_chunks_agree():
    later_chunk = _block(jd=[JD0 + 0.5, JD0 + 1.0])
    np.testing.assert_allclose(_eval("time", later_chunk, "h").ravel(), [12.0, 24.0])


def test_wavelength_units_vary_along_rows_and_channels_delay_units_along_rows():
    u_kl = _eval("u")  # default: klambda
    assert u_kl.shape == (2, 1, 2)
    np.testing.assert_allclose(u_kl[:, 0, :], np.outer([1e-6, 2e-6], [1e9, 1.1e9]) / 1e3)
    assert _eval("u", unit="m").shape == (2, 1, 1)
    np.testing.assert_allclose(_eval("u", unit="m").ravel(), np.array([1e-6, 2e-6]) * C)


def test_units_of_one_base_differ_by_their_factor():
    np.testing.assert_allclose(_eval("w", unit="km"), _eval("w", unit="m") / 1e3)
    np.testing.assert_allclose(_eval("v", unit="Mlambda"), _eval("v", unit="klambda") / 1e3)
    np.testing.assert_allclose(_eval("time", unit="min"), _eval("time", unit="h") * 60)
    np.testing.assert_allclose(_eval("freq", unit="GHz"), _eval("freq", unit="Hz") / 1e9)
    block = _block(with_data=False)
    np.testing.assert_allclose(_eval("ha", block, "deg"), _eval("ha", block) * 15)
    np.testing.assert_allclose(_eval("el", block, "rad"), np.radians(_eval("el", block)))


def test_aliases_are_their_quantity_in_their_unit():
    for alias, (quantity, unit) in ALIASES.items():
        block = _block()
        np.testing.assert_array_equal(_eval(alias, block), _eval(quantity, block, unit))
    np.testing.assert_allclose(_eval("u_klambda")[:, 0, :], np.outer([1e-6, 2e-6], [1e9, 1.1e9]) / 1e3)
    with pytest.raises(ValueError, match="'u_klambda' is 'u' in klambda"):
        _eval("u_klambda", unit="m")


def test_units_are_found_by_name_or_label_and_a_wrong_one_lists_the_choices():
    assert resolve_unit("u", "kλ").name == "klambda"
    assert resolve_unit("u", "Mλ").name == "Mlambda"
    assert resolve_unit("u").name == "klambda"
    with pytest.raises(ValueError, match="choose one of: lambda, klambda, Mlambda, m, km"):
        resolve_unit("u", "deg")
    with pytest.raises(ValueError, match="category and has no unit"):
        resolve_unit("stokes", "h")


def test_flux_units_convert_only_when_bunit_is_a_flux_density():
    jy = _ctx(bunit="JY")  # AIPS's spelling of Jy
    assert [unit.name for unit in units_of("amp", jy)] == ["JY", "Jy", "mJy", "uJy"]
    np.testing.assert_allclose(_eval("amp", unit="mJy", ctx=jy), _eval("amp", ctx=jy) * 1e3)
    np.testing.assert_allclose(_eval("real", unit="µJy", ctx=jy), _eval("real", ctx=jy) * 1e6)
    assert [unit.name for unit in units_of("amp", _ctx())] == ["UNCALIB"]
    with pytest.raises(ValueError, match="BUNIT \\('UNCALIB'\\), which is not a flux density"):
        _eval("amp", unit="mJy")


def test_utc_without_a_reference_date_counts_days_from_the_first_integrations_date():
    np.testing.assert_allclose(_eval("time", unit="UTC").ravel(), [16.8, 28.8])  # 16:48, then 04:48 next day
    assert quantity_label("time", _ctx(), "UTC") == "Time (UTC; day 0 = 2021-07-25)"


def test_recorded_time_is_the_default_counting_days_from_the_files_reference_date():
    ctx = _ctx(reference_date_jd=REF_JD, time_system="IAT")
    np.testing.assert_allclose(_eval("time", ctx=ctx).ravel(), [40.8, 52.8])  # day 1 16:48, day 2 04:48
    assert quantity_label("time", ctx) == "Time (recorded, IAT; day 0 = 2021-07-24)"
    assert value_description("time", ctx) == "time as recorded (TIMSYS IAT), in hours since 2021-07-24 00:00"
    np.testing.assert_allclose(_eval("time", unit="local", ctx=ctx).ravel(), [46.3, 58.3])  # day 1 22:18 IST
    assert quantity_label("time", ctx, "local") == "Time (IST, UTC+05:30; day 0 = 2021-07-24)"


def test_local_time_is_in_the_observatory_time_zone():
    np.testing.assert_allclose(_eval("time", unit="local").ravel(), [22.3, 34.3])  # 22:18 IST, then 10:18
    assert quantity_label("time", _ctx(), "local") == "Time (IST, UTC+05:30; day 0 = 2021-07-25)"
    assert value_description("time", _ctx(), "local") == \
        "time in local time, hours since 2021-07-25 00:00 IST (Asia/Kolkata, UTC+05:30)"
    with pytest.raises(ValueError, match="time zone"):
        _eval("time", unit="local", ctx=_ctx(time_zone=None))
    with pytest.raises(ValueError, match="unknown time zone 'Mars/Olympus'"):
        _eval("time", unit="local", ctx=_ctx(time_zone="Mars/Olympus"))


def test_lst_matches_astrometry_and_continues_past_24h():
    jd = JD0 + np.arange(0.0, 1.3, 0.05)  # over a day: LST passes 0h
    lst = _eval("time", _row_block(jd), "LST").ravel()
    np.testing.assert_allclose(np.mod(lst, 24.0), local_sidereal_time_hours(jd, GMRT), atol=1e-9)
    assert np.all(np.diff(lst) > 0) and lst[-1] > 24.0
    # A later chunk alone gets the same values: the unwrapping counts from the shared reference.
    np.testing.assert_allclose(_eval("time", _row_block(jd[10:]), "LST").ravel(), lst[10:])


def test_freq_varies_only_along_channels():
    freq = _eval("freq")
    assert freq.shape == (1, 1, 2)
    np.testing.assert_allclose(freq.ravel(), [1000.0, 1100.0])


def test_pairing_two_quantities_broadcasts_to_the_shape_the_pair_needs():
    x, y = np.broadcast_arrays(_eval("time"), _eval("u", unit="m"))
    assert x.shape == (2, 1, 1)
    x, y = np.broadcast_arrays(_eval("freq"), _eval("amp"))
    assert x.shape == (2, 2, 2)


def test_geometry_quantities_match_astrometry():
    block = _block(with_data=False)
    ra = np.array([202.8, 24.4])
    dec = np.array([30.5, 33.2])
    np.testing.assert_allclose(_eval("ha", block).ravel(), hour_angle_hours(block.jd, ra, GMRT))
    az, el = altaz_deg(block.jd, ra, dec, GMRT)
    np.testing.assert_allclose(_eval("az", block).ravel(), az)
    np.testing.assert_allclose(_eval("el", block).ravel(), el)
    np.testing.assert_allclose(_eval("pa", block).ravel(), parallactic_angle_deg(block.jd, ra, dec, GMRT))


def test_categories_evaluate_to_codes_with_labels():
    ctx = _ctx()
    stokes = _eval("stokes")
    assert stokes.shape == (1, 2, 1)
    assert [category_label("stokes", c, ctx) for c in stokes.ravel()] == ["RR", "LL"]
    source = _eval("source")
    assert [category_label("source", c, ctx) for c in source.ravel()] == ["3C286", "3C48"]


def test_only_data_quantities_need_data():
    assert {q.name for q in QUANTITIES.values() if q.needs_data} == {"real", "imag", "amp", "phase"}


def test_quantity_label_units():
    ctx = _ctx()
    assert quantity_label("freq", ctx) == "Frequency (MHz)"
    assert quantity_label("freq_mhz", ctx) == "Frequency (MHz)"
    assert quantity_label("amp", ctx) == "Amplitude (UNCALIB)"
    assert quantity_label("amp", QuantityContext(time_reference_jd=JD0)) == "Amplitude"
    assert quantity_label("stokes", ctx) == "Stokes"
    assert quantity_label("u_klambda", ctx) == "U (kλ)"
    assert quantity_label("u", ctx, "km") == "U (km)"
    with pytest.raises(ValueError, match="'u_sec' is no longer available"):
        quantity_label("u_sec", ctx)
    assert quantity_label("time", ctx) == "Time (recorded; day 0 = 2021-07-25)"
    assert quantity_label("time", ctx, "h") == "Time since 2021-07-25 16:48:00 UTC (h)"


def test_every_unit_is_named_in_its_quantitys_help():
    for q in QUANTITIES.values():
        for unit in q.units:
            assert unit.name in q.description, (q.name, unit.name)


def test_convert_between_units_of_one_base_only():
    assert convert(-12.0, "ha", "h", "deg") == pytest.approx(-180.0)
    assert convert(1.0, "u", "km", "m") == pytest.approx(1000.0)
    with pytest.raises(ValueError, match="cannot be converted"):
        convert(1.0, "u", "m", "klambda")


def test_klambda_without_a_freq_axis_raises():
    block = VisibilityBlock(**{**_block().__dict__, "chan_freqs_hz": None})
    with pytest.raises(ValueError, match="no channel frequencies"):
        _eval("u_klambda", block)


def test_utc_based_quantities_subtract_the_recorded_minus_utc_offset():
    ctx = _ctx(reference_date_jd=REF_JD)
    shifted = context_from_source_table(JD0, _sources(), GMRT, bunit="UNCALIB", stokes_labels=["RR", "LL"],
                                        time_zone="Asia/Kolkata", reference_date_jd=REF_JD, recorded_minus_utc_s=35.0)
    block = _block(with_data=False)
    np.testing.assert_allclose(_eval("time", block, "UTC", shifted), _eval("time", block, "UTC", ctx) - 35 / 3600)
    np.testing.assert_allclose(_eval("time", block, "local", shifted), _eval("time", block, "local", ctx) - 35 / 3600)
    np.testing.assert_array_equal(_eval("time", block, "recorded", shifted), _eval("time", block, "recorded", ctx))
    np.testing.assert_array_equal(_eval("time", block, "h", shifted), _eval("time", block, "h", ctx))
    ra = np.array([202.8, 24.4])
    np.testing.assert_allclose(_eval("ha", block, ctx=shifted).ravel(),
                               hour_angle_hours(block.jd - 35 / 86400, ra, GMRT))
    lst = local_sidereal_time_hours(block.jd - 35 / 86400, GMRT)
    np.testing.assert_allclose(np.mod(_eval("time", block, "LST", shifted).ravel(), 24.0), lst, atol=1e-9)
    assert quantity_label("time", shifted, "h") == "Time since 2021-07-25 16:47:25 UTC (h)"
