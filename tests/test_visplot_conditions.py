"""`--where` conditions: the grammar, what is refused and why, and the checks
against a file (its products, BUNIT, channels, band, time span; conditions that
cannot hold together; conditions holding for every cell)."""

import math

import pytest

from conftest import make_scratch_dir
from test_cli_visplot_output import JD0, _make_synthetic_file
from visplot.conditions import (CONDITION_EXAMPLES, Bound, FileFacts, Operand, facts_of, parse_condition,
                                resolve_conditions)
from visplot.run import open_file

# A file like 40_014's: circular feeds RR and LL, BUNIT UNCALIB, 2,048 channels of 300-500 MHz; day 0 is
# 2021-07-24 (JD 2459419.5 at 0h), observed from 2021-07-25 16:00 to 2021-07-26 03:00 as recorded.
DAY0 = 2459419.5
SPAN = (DAY0 + 1 + 16 / 24, DAY0 + 2 + 3 / 24)
FACTS = FileFacts(stokes_labels=("RR", "LL"), bunit="UNCALIB", n_channels=2048, freq_hz=(300e6, 500e6),
                  jd=SPAN, day0_jd=DAY0, recorded_minus_utc_s=35.0)
FULL = FileFacts(**{**FACTS.__dict__, "stokes_labels": ("RR", "LL", "RL", "LR")})


def _resolved(*texts, facts=FACTS):
    return resolve_conditions([parse_condition(text) for text in texts], facts)


def test_every_example_parses_and_resolves_on_a_file_it_fits():
    for text, meaning in CONDITION_EXAMPLES:
        assert meaning and resolve_conditions([parse_condition(text)], FULL).conditions, text


@pytest.mark.parametrize("text, lower, upper", [
    ("amp(V) > 5", Bound(5.0, False), None),
    ("amp(V) >= 5", Bound(5.0, True), None),
    ("amp(I) < 0.005", None, Bound(0.005, False)),
    ("100 < uvdist < 2000 lambda", Bound(100.0, False), Bound(2000.0, False)),
    ("2000 lambda > uvdist >= 100 lambda", Bound(100.0, True), Bound(2000.0, False)),  # a descending range
    ("400 MHz <= freq <= 450 MHz", Bound(400.0, True), Bound(450.0, True)),
])
def test_a_condition_is_an_interval_with_its_ends_open_or_closed(text, lower, upper):
    condition = parse_condition(text)
    assert (condition.lower, condition.upper) == (lower, upper)


def test_names_are_case_insensitive_units_case_sensitive():
    assert parse_condition("AMP(v) > 5").operand == Operand("amp", "V")
    assert parse_condition("EL < 20 deg").operand == Operand("el")
    with pytest.raises(ValueError, match="unit 'mhz' does not apply to 'freq'; choose one of: Hz, kHz, MHz, GHz"):
        parse_condition("freq > 400 mhz")
    assert parse_condition("uvdist < 2 kλ").unit == "klambda"  # a unit's label is its name


@pytest.mark.parametrize("text, message", [
    ("amp > 5", r"amp needs its product, e\.g\. amp\(RR\) or amp\(V\)"),
    ("V > 5", r"a product is written inside a function, e\.g\. amp\(V\); the baseline coordinate is vv"),
    ("v < 2 klambda", r"a product is written inside a function, e\.g\. amp\(V\); the baseline coordinate is vv"),
    ("w < 2 klambda", r"the baseline coordinate is ww"),
    ("ampl(V) > 5", r"'ampl' is not a function; the functions are amp, phase, real, imag"),
    ("amp(ZZ) > 5", r"'ZZ' is not a product"),
    ("elevation < 20 deg", r"'elevation' is neither a coordinate"),
    ("el < 20", r"el needs its unit, one of: deg, rad"),
    ("phase(RL) > 170", r"phase\(RL\) needs its unit"),
    ("100 < uvdist < 2000", r"uvdist needs its unit"),
    ("el < 20 klambda", r"unit 'klambda' does not apply to 'el'"),
    ("channel > 10 MHz", r"channel is a 0-based index and takes no unit"),
    ("amp(V) == 5", r"only <, <=, >, >= compare"),
    ("amp(V) = 5", r"only <, <=, >, >= compare"),
    ("amp(V) 5", r"no comparison"),
    ("1 < 2 < amp(V) < 3", r"at most two comparisons"),
    ("1 < amp(V) > 3", r"a range runs one way"),
    ("5 < amp(V) < 3", r"the range is empty: its lower end 5 is not below its upper end 3"),
    ("5 < amp(V) < 5", r"the range is empty"),
    ("100 m < uvdist < 2 km", r"a range takes one unit, on either end or both; got m and km"),
    ("amp(V) < 5x y", r"is not a number with a unit"),
    ("time > 1/17:42:09", r"time is an instant and its scale"),
    ("time > 17:42 UTC", r"'17:42' is not an instant"),
    ("time > 2 h", r"time is an instant and its scale"),  # elapsed time: its origin depends on the selection
])
def test_what_leaves_a_reading_open_is_refused_saying_how_to_write_it(text, message):
    with pytest.raises(ValueError, match=message):
        parse_condition(text)


@pytest.mark.parametrize("text, message", [
    ("el > 95 deg", r"el lies in -90\.\.90 deg; 95 deg is outside it"),
    ("el > 2 rad", r"el lies in -1\.5708\.\.1\.5708 rad; 2 rad is outside it"),
    ("az < 400 deg", r"az lies in 0\.\.360 deg"),
    ("ha > 13 h", r"ha lies in -12\.\.12 h"),
    ("phase(RL) > 200 deg", r"phase\(RL\) lies in -180\.\.180 deg"),
    ("amp(V) < -1", r"amp\(V\) lies in at least 0"),
    ("uvdist > -1 m", r"uvdist lies in at least 0 m"),
    ("uvdist > -1 klambda", r"uvdist lies in at least 0 klambda"),  # zero is zero in wavelengths too
])
def test_unphysical_values_are_refused(text, message):
    with pytest.raises(ValueError, match=message):
        parse_condition(text)


def test_against_a_file_its_products_bunit_channels_band_and_time_span():
    with pytest.raises(ValueError, match=r"Stokes Q of circular feeds needs RL, LR"):
        _resolved("amp(Q) > 5")
    with pytest.raises(ValueError, match=r"BUNIT \('UNCALIB'\), which is not a flux density"):
        _resolved("amp(V) > 5 Jy")
    assert _resolved("amp(V) > 5 UNCALIB").conditions[0].lower == Bound(5.0, False)
    with pytest.raises(ValueError, match=r"channel in this file lies in 0\.\.2047"):
        _resolved("channel > 3000")
    with pytest.raises(ValueError, match=r"freq in this file lies in 300\.000000\.\.500\.000000 MHz"):
        _resolved("freq > 600 MHz")
    with pytest.raises(ValueError, match=r"time in this file lies in"):
        _resolved("time > 5/00:00:00 UTC")
    jansky = FileFacts(**{**FACTS.__dict__, "bunit": "JY"})
    assert _resolved("amp(V) > 500 mJy", facts=jansky).conditions[0].lower.value == pytest.approx(0.5)  # in BUNIT


def test_an_instant_is_the_rows_recorded_time_utc_offset_by_the_files_recorded_minus_utc():
    recorded = _resolved("time >= 1/20:00:00 recorded").conditions[0].lower.value
    utc = _resolved("time >= 1/20:00:00 UTC").conditions[0].lower.value
    assert recorded == pytest.approx(DAY0 + 1 + 20 / 24)  # day 1 from day 0's 0h, plus 20 h
    assert (utc - recorded) * 86400 == pytest.approx(35.0)  # the same clock reading in UTC is 35 s later, recorded
    iso = _resolved("time >= 2021-07-25T20:00:00 recorded").conditions[0].lower.value
    assert iso == pytest.approx(recorded, abs=1e-9)


def test_conditions_that_cannot_hold_together_are_refused_before_any_data_is_read():
    with pytest.raises(ValueError, match=r"--where 'amp\(V\) > 5' and --where 'amp\(V\) < 3' cannot both hold"):
        _resolved("amp(V) > 5", "amp(V) < 3")
    with pytest.raises(ValueError, match="cannot both hold"):
        _resolved("amp(V) > 5", "amp(V) <= 5")  # the open and closed ends kept
    with pytest.raises(ValueError, match="cannot both hold"):
        _resolved("el > 0.5 rad", "el < 20 deg")  # 28.6 deg against 20 deg, in one unit
    with pytest.raises(ValueError, match="cannot both hold"):
        _resolved("phase(RL) > 170 deg", "phase(RL) < -170 deg", facts=FULL)  # a band across the wrap: two runs
    assert _resolved("amp(V) >= 5", "amp(V) <= 5").conditions  # exactly 5: possible
    assert _resolved("amp(V) > 5", "amp(I) < 3").conditions  # different operands: never judged impossible
    assert _resolved("amp(RR) > 5", "amp(LL) < 3").conditions  # one function, different products
    assert _resolved("uvdist > 3 km", "uvdist < 2 klambda").conditions  # metres and wavelengths: no fixed factor


def test_a_condition_holding_for_every_cell_is_warned_of_and_kept():
    for text in ("amp(V) >= 0", "el >= -90 deg", "channel >= 0", "uvdist >= 0 klambda"):
        resolved = _resolved(text)
        assert resolved.warnings == (f"--where {text!r} holds for every cell",) and resolved.conditions
    assert _resolved("amp(V) > 0", "el > -90 deg").warnings == ()  # an open end at the edge excludes it


def test_the_facts_of_an_opened_file():
    path = _make_synthetic_file(make_scratch_dir("conditions_facts") / "obs.fits")
    facts = facts_of(open_file(path))
    assert facts.stokes_labels == ("RR", "LL") and facts.bunit == "UNCALIB" and facts.n_channels == 4
    assert facts.freq_hz == (400e6, 403e6) and facts.jd[0] == pytest.approx(JD0)
    assert math.isfinite(facts.day0_jd) and facts.day0_jd <= facts.jd[0] < facts.day0_jd + 1


def test_one_day_0_per_file_for_plots_listings_and_conditions():
    """Without RDATE or DATE-OBS, day 0 is the date of the file's first integration, never a selection's."""
    from visplot.quantities import day_origin_jd, utc_day_start_jd
    from visplot.request import PlotRequest
    from visplot.run import prepare

    path = _make_synthetic_file(make_scratch_dir("conditions_day0") / "obs.fits")
    opened = open_file(path)
    assert opened.reference_date_jd is None  # the synthetic file has neither RDATE nor DATE-OBS
    assert opened.day0_jd == utc_day_start_jd(float(opened.index.jd.min()))
    later = prepare(PlotRequest(str(path), "amp-vs-time", sources="3C48"), opened)  # starts after the file does
    assert later.ctx.time_reference_jd > float(opened.index.jd.min())
    assert day_origin_jd(later.ctx) == opened.day0_jd == facts_of(opened).day0_jd

