import astropy.units as u
import pytest

from visplot.range_spec import (
    expand_int_ranges,
    parse_quantity_range,
    parse_range_spec,
    parse_single_quantity_range,
    parse_single_range,
)


def test_parse_range_spec_handles_bare_values_and_ranges():
    assert parse_range_spec("1:5,10,12:14") == [(1.0, 5.0), (10.0, 10.0), (12.0, 14.0)]


def test_parse_range_spec_preserves_a_reversed_term_for_wrapping_callers():
    # select_rows's HA/Az/parallactic-angle filters treat lo > hi as
    # wrapping through the cyclic boundary -- this must not be "fixed" here.
    assert parse_range_spec("22:2") == [(22.0, 2.0)]


def test_parse_range_spec_strips_whitespace_around_terms():
    assert parse_range_spec(" 1:5 , 10 ") == [(1.0, 5.0), (10.0, 10.0)]


def test_parse_range_spec_rejects_an_empty_spec():
    with pytest.raises(ValueError, match="empty range spec"):
        parse_range_spec("")


def test_expand_int_ranges_covers_every_integer_in_each_range():
    assert expand_int_ranges("1:5,10,12:14") == [1, 2, 3, 4, 5, 10, 12, 13, 14]


def test_expand_int_ranges_deduplicates_overlapping_terms():
    assert expand_int_ranges("1:5,3:7") == [1, 2, 3, 4, 5, 6, 7]


def test_expand_int_ranges_treats_a_reversed_term_the_same_as_forward():
    # No wrapping concept for a discrete id list -- "5:1" is just [1..5].
    assert expand_int_ranges("5:1") == [1, 2, 3, 4, 5]


def test_parse_single_range_returns_one_tuple():
    assert parse_single_range("-6:6") == (-6.0, 6.0)


def test_parse_single_range_rejects_more_than_one_term():
    with pytest.raises(ValueError, match="expected a single lo:hi range"):
        parse_single_range("1:5,10:14")


def test_parse_quantity_range_with_no_unit_suffix_assumes_the_native_unit():
    assert parse_quantity_range("300:310", u.MHz) == [(300.0, 310.0)]


def test_parse_quantity_range_converts_an_explicit_unit_to_the_native_unit():
    lo, hi = parse_single_quantity_range("300:310MHz", u.Hz)
    assert lo == pytest.approx(300e6)
    assert hi == pytest.approx(310e6)


def test_parse_quantity_range_handles_hourangle_conversion_for_ha():
    # HA's native unit is hours; a user typing degrees should get exactly
    # the same 15deg-per-hour conversion astropy's own hourangle uses.
    lo, hi = parse_single_quantity_range("-90:90deg", u.hourangle)
    assert lo == pytest.approx(-6.0)
    assert hi == pytest.approx(6.0)


def test_parse_quantity_range_converts_km_to_the_native_metre_unit():
    lo, hi = parse_single_quantity_range("0:5km", u.m)
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(5000.0)


def test_parse_quantity_range_is_case_sensitive_about_units():
    # "mhz" is not a recognized unit (unlike "MHz" or "mHz") -- astropy
    # itself raises, and this must not silently case-fold to "fix" it.
    with pytest.raises(ValueError):
        parse_single_quantity_range("300:310mhz", u.Hz)


def test_parse_quantity_range_handles_multiple_terms_with_one_shared_unit():
    assert parse_quantity_range("1:5,10km", u.m) == [(1000.0, 5000.0), (10000.0, 10000.0)]


def test_parse_quantity_range_applies_a_unit_found_on_any_one_term_to_all_terms():
    # The unit appears on the first term here, not the last -- must still
    # apply to the second, unitless term.
    assert parse_quantity_range("1km,2:3", u.m) == [(1000.0, 1000.0), (2000.0, 3000.0)]


def test_parse_quantity_range_accepts_the_same_unit_repeated_on_every_term():
    assert parse_quantity_range("100:100.5MHz,103.5:104MHz", u.Hz) == [
        (100e6, 100.5e6), (103.5e6, 104e6),
    ]


def test_parse_quantity_range_rejects_conflicting_units_across_terms():
    with pytest.raises(ValueError, match="mixed units"):
        parse_quantity_range("1km,2000m", u.m)
