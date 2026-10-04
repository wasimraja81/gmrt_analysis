"""Instants and --time-range 'LO .. HI SCALE': the two forms of an instant, the
scale always written, both ends included, and what is refused."""

import pytest

from visplot.instants import TimeRange, check_instant, parse_time_range, resolve_time_range_jd, split_time_range_scale

DAY0 = 2459419.5  # 2021-07-24 at 0h, a file's day 0
OFFSET_S = 35.0  # recorded - UTC: TIMSYS IAT, IATUTC 35


def test_a_day_count_and_an_iso_date_give_the_same_recorded_jd():
    day_count = resolve_time_range_jd("1/17:00:00 .. 1/18:00:00 recorded", DAY0, OFFSET_S)
    assert day_count == pytest.approx((DAY0 + 1 + 17 / 24, DAY0 + 1 + 18 / 24))
    iso = resolve_time_range_jd("2021-07-25T17:00:00 .. 2021-07-25T18:00:00 recorded", DAY0, OFFSET_S)
    assert iso == pytest.approx(day_count, abs=1e-9)


def test_utc_is_the_recorded_time_less_the_files_recorded_minus_utc():
    recorded = resolve_time_range_jd("1/17:00:00 .. 1/18:00:00 recorded", DAY0, OFFSET_S)
    utc = resolve_time_range_jd("1/17:00:00 .. 1/18:00:00 UTC", DAY0, OFFSET_S)
    assert [(u - r) * 86400 for u, r in zip(utc, recorded)] == pytest.approx([OFFSET_S, OFFSET_S])


def test_both_ends_are_included_so_one_instant_is_a_range():
    lo, hi = resolve_time_range_jd("1/17:00:00 .. 1/17:00:00 UTC", DAY0, 0.0)
    assert lo == hi


def test_the_written_form_parses_without_a_file():
    assert parse_time_range("1/17:00:00..2021-07-25T18:00:00 UTC") == TimeRange(
        "1/17:00:00", "2021-07-25T18:00:00", "UTC")  # spaces around '..' optional; the two forms mixed
    assert parse_time_range(" 0/23:59:59.5 .. 1/00:00:00 recorded ") == TimeRange(
        "0/23:59:59.5", "1/00:00:00", "recorded")


def test_an_iso_end_against_a_day_count_is_ordered_against_the_files_day_0():
    parse_time_range("1/18:00:00 .. 2021-07-25T17:00:00 UTC")  # needs day 0 to order
    with pytest.raises(ValueError, match=r"the lower end 1/18:00:00 is after the upper end 2021-07-25T17:00:00"):
        resolve_time_range_jd("1/18:00:00 .. 2021-07-25T17:00:00 UTC", DAY0, 0.0)
    resolve_time_range_jd("1/18:00:00 .. 2021-07-25T17:00:00 UTC", DAY0 - 1, 0.0)  # day 0 a day earlier: in order


@pytest.mark.parametrize("spec, message", [
    ("1/17:00:00 .. 1/18:00:00", r"give the scale, one of: recorded, UTC"),
    ("1/17:00:00 .. 1/18:00:00 utc", r"scale 'utc' does not apply; one of: recorded, UTC"),  # case-sensitive
    ("1/17:00:00 UTC .. 1/18:00:00 UTC", r"the scale is written once, after HI"),
    ("0:0.5", r"write 'LO \.\. HI SCALE'"),  # hours from the first integration: no longer a form
    ("2021-07-25T17:00:00/2021-07-25T18:00:00", r"write 'LO \.\. HI SCALE'"),
    ("1/17:00:00 .. 1/18:00:00 .. 1/19:00:00 UTC", r"write 'LO \.\. HI SCALE'"),
    ("17:00:00 .. 18:00:00 UTC", r"'17:00:00' is not an instant: write d/hh:mm:ss"),
    ("1/24:00:00 .. 2/00:00:00 UTC", r"hours below 24, minutes and seconds below 60"),
    ("1/17:60:00 .. 1/18:00:00 UTC", r"hours below 24"),
    ("1/17:00:00 .. 1/17:00:60 UTC", r"hours below 24"),
    ("2021-02-30T00:00:00 .. 2021-03-01T00:00:00 UTC", r"'2021-02-30T00:00:00' is not a calendar date"),
    ("1/18:00:00 .. 1/17:00:00 UTC", r"the lower end 1/18:00:00 is after the upper end 1/17:00:00"),
    ("2021-07-25T18:00:00 .. 2021-07-25T17:00:00 UTC", r"the lower end .* is after the upper end"),
])
def test_what_leaves_a_reading_open_or_cannot_be_is_refused(spec, message):
    with pytest.raises(ValueError, match=message):
        parse_time_range(spec)


def test_an_instant_is_checked_alike_in_both_forms():
    for text in ("0/00:00:00", "12/23:59:59.999", "2021-07-25T17:42:09", "2021-07-25T17:42:09.25"):
        check_instant(text)
    for text in ("2021-07-25T23:59:60", "1/17:42", "2021-07-25 17:42:09", "-1/17:00:00"):
        with pytest.raises(ValueError):
            check_instant(text)


def test_split_time_range_scale_for_the_gui():
    assert split_time_range_scale("1/17:00:00 .. 1/18:00:00 UTC") == ("1/17:00:00 .. 1/18:00:00", "UTC")
    assert split_time_range_scale("1/17:00:00 .. 1/18:00:00") == ("1/17:00:00 .. 1/18:00:00", None)
    assert split_time_range_scale("1/17:00:00 .. 1/18:00:00 utc") == ("1/17:00:00 .. 1/18:00:00 utc", None)
