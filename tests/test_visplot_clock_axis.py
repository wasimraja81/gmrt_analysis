import datetime

import numpy as np
import pytest

from visplot.clock_axis import TIME_FORMATS, ClockFormatter, ClockLocator, clock_precision, clock_text

DAY0 = datetime.date(2021, 7, 24)


def test_clock_text_is_aips_day_slash_time_by_default():
    assert clock_text(22.3, 0) == "0/22:18:00"
    assert clock_text(40.8, 2) == "1/16:48:00.00"
    assert clock_text(-1.0, 0) == "-1/23:00:00"
    assert clock_text(1 + 1.5 / 3600, 1) == "0/01:00:01.5"
    assert clock_text(23 + 59 / 60 + 59.996 / 3600, 2) == "1/00:00:00.00"  # rounds into the next day


def test_clock_text_in_each_time_format():
    hours = 24 + 16 + 44 / 60  # day 1, 16:44:00
    assert [clock_text(hours, 0, f, DAY0) for f in TIME_FORMATS] == [
        "1/16:44:00", "01:16:44:00", "1/16:44:00", "2021-07-25 16:44:00"]
    assert clock_text(hours, 0, "hh:mm:ss", with_day=False) == "16:44:00"
    assert clock_text(-1.0, 0, "dd:hh:mm:ss") == "-01:23:00:00"
    with pytest.raises(ValueError, match="needs day 0's calendar date"):
        clock_text(hours, 0, "iso")
    with pytest.raises(ValueError, match="no time format"):
        clock_text(hours, 0, "hh:mm")


def test_precision_is_what_the_ticks_need():
    assert clock_precision([1.0, 1.5, 2.0]) == 0
    assert clock_precision([1.0, 1.0 + 0.5 / 3600]) == 1
    assert clock_precision([1.0, 1.0 + 0.25 / 3600]) == 2
    assert clock_precision([]) == 0


@pytest.mark.parametrize("span_h, step_s", [(10.0, 7200), (1.0, 600), (20 / 60, 300), (40.0, 43200), (30 / 3600, 5)])
def test_locator_steps_are_whole_clock_units(span_h, step_s):
    ticks = ClockLocator().tick_values(16.8, 16.8 + span_h)
    steps = np.diff(ticks) * 3600
    assert steps == pytest.approx(np.full(len(steps), step_s))
    assert 2 <= len(ticks) <= 7
    assert np.allclose(np.round(ticks * 3600 / step_s), ticks * 3600 / step_s)  # on whole steps


def test_formatter_labels_all_ticks_alike_and_reads_the_cursor_to_hundredths():
    formatter = ClockFormatter()
    ticks = [22.0, 23.0, 24.0, 25.0]
    assert formatter.format_ticks(ticks) == ["0/22:00:00", "0/23:00:00", "1/00:00:00", "1/01:00:00"]
    assert formatter.format_ticks([1.0, 1.0 + 1.5 / 3600]) == ["0/01:00:00.0", "0/01:00:01.5"]
    assert formatter.format_data_short(22.3 + 1.234 / 3600) == "0/22:18:01.23"


def test_time_of_day_ticks_carry_their_day_at_the_first_tick_and_each_new_day():
    formatter = ClockFormatter("hh:mm:ss")
    assert formatter.format_ticks([22.0, 23.0, 24.0, 25.0, 48.0]) == [
        "0/22:00:00", "23:00:00", "1/00:00:00", "01:00:00", "2/00:00:00"]
    assert formatter.format_data_short(25.5) == "1/01:30:00.00"  # the cursor always says its day
    iso = ClockFormatter("iso", DAY0)
    assert iso.format_ticks([40.0, 48.0]) == ["2021-07-25 16:00:00", "2021-07-26 00:00:00"]


def test_iso_is_refused_for_lst_which_has_no_calendar_date():
    from visplot.request import PlotRequest
    from visplot.run import RequestError, check_request

    with pytest.raises(RequestError, match="LST, a sidereal time, has none"):
        check_request(PlotRequest("x.fits", "amp-vs-time", x_unit="LST", time_format="iso"))
    check_request(PlotRequest("x.fits", "amp-vs-time", x_unit="UTC", time_format="iso"))
    check_request(PlotRequest("x.fits", "amp-vs-time", x_unit="LST", time_format="hh:mm:ss"))
