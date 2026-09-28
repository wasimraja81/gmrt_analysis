import pytest

from visplot.time_range import resolve_time_range_jd

REFERENCE_JD = 2460123.5  # an arbitrary observation start


def test_resolve_time_range_jd_bare_numbers_are_relative_hours():
    lo, hi = resolve_time_range_jd("0:2", REFERENCE_JD)
    assert lo == pytest.approx(REFERENCE_JD)
    assert hi == pytest.approx(REFERENCE_JD + 2.0 / 24.0)


def test_resolve_time_range_jd_explicit_hour_suffix_matches_bare_numbers():
    assert resolve_time_range_jd("0h:2h", REFERENCE_JD) == resolve_time_range_jd("0:2", REFERENCE_JD)


def test_resolve_time_range_jd_jd_suffix_is_absolute():
    lo, hi = resolve_time_range_jd("2460123.5:2460123.6jd", REFERENCE_JD)
    assert lo == pytest.approx(2460123.5)
    assert hi == pytest.approx(2460123.6)


def test_resolve_time_range_jd_jd_suffix_only_needs_to_appear_once():
    lo, hi = resolve_time_range_jd("2460123.5jd:2460123.6", REFERENCE_JD)
    assert lo == pytest.approx(2460123.5)
    assert hi == pytest.approx(2460123.6)


def test_resolve_time_range_jd_slash_form_parses_iso_timestamps():
    lo, hi = resolve_time_range_jd("2021-07-25T10:00:00/2021-07-25T12:00:00", REFERENCE_JD)
    assert hi > lo
    assert (hi - lo) * 24.0 == pytest.approx(2.0)


def test_resolve_time_range_jd_slash_form_accepts_a_bare_date():
    lo, hi = resolve_time_range_jd("2021-07-25/2021-07-26", REFERENCE_JD)
    assert (hi - lo) == pytest.approx(1.0)


def test_resolve_time_range_jd_rejects_more_than_two_colon_terms():
    # A malformed attempt at an ISO range without the required slash --
    # must raise clearly, not silently misparse.
    with pytest.raises(ValueError, match="expected a single lo:hi"):
        resolve_time_range_jd("2021-07-25T10:00:00:2021-07-25T12:00:00", REFERENCE_JD)


def test_resolve_time_range_jd_rejects_conflicting_units_on_the_two_terms():
    with pytest.raises(ValueError, match="mixed time units"):
        resolve_time_range_jd("0h:2460123.6jd", REFERENCE_JD)


def test_resolve_time_range_jd_rejects_an_unrecognized_unit():
    with pytest.raises(ValueError, match="unrecognized time unit"):
        resolve_time_range_jd("0:2min", REFERENCE_JD)


def test_resolve_time_range_jd_rejects_a_bound_that_looks_like_an_unrecognized_unit():
    # "start" is all letters, so it's caught as an unrecognized unit before
    # numeric parsing is even attempted.
    with pytest.raises(ValueError, match="unrecognized time unit"):
        resolve_time_range_jd("start:2", REFERENCE_JD)


def test_resolve_time_range_jd_rejects_a_non_numeric_bound():
    # No trailing letters at all here (so no unit is even detected), but
    # the numeric part itself still isn't a valid float.
    with pytest.raises(ValueError, match="not a valid time bound"):
        resolve_time_range_jd("1.2.3:2", REFERENCE_JD)


def test_absolute_bounds_are_utc_and_become_recorded_time():
    ref = 2459421.2
    iso = resolve_time_range_jd("2021-07-25T17:00:00/2021-07-25T18:00:00", ref, recorded_minus_utc_s=35.0)
    plain = resolve_time_range_jd("2021-07-25T17:00:00/2021-07-25T18:00:00", ref)
    assert iso == pytest.approx((plain[0] + 35 / 86400, plain[1] + 35 / 86400), abs=1e-9)
    assert resolve_time_range_jd("2459421.3:2459421.4jd", ref, 35.0) == pytest.approx(
        (2459421.3 + 35 / 86400, 2459421.4 + 35 / 86400), abs=1e-9)
    assert resolve_time_range_jd("0:1", ref, 35.0) == resolve_time_range_jd("0:1", ref)  # relative: unchanged
