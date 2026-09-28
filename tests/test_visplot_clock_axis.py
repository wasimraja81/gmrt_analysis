import numpy as np
import pytest

from visplot.clock_axis import ClockFormatter, ClockLocator, clock_precision, clock_text


def test_clock_text_is_days_hours_minutes_seconds():
    assert clock_text(22.3, 0) == "00:22:18:00"
    assert clock_text(40.8, 2) == "01:16:48:00.00"
    assert clock_text(-1.0, 0) == "-01:23:00:00"
    assert clock_text(1 + 1.5 / 3600, 1) == "00:01:00:01.5"
    assert clock_text(23 + 59 / 60 + 59.996 / 3600, 2) == "01:00:00:00.00"  # rounds into the next day


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
    assert formatter.format_ticks(ticks) == ["00:22:00:00", "00:23:00:00", "01:00:00:00", "01:01:00:00"]
    assert formatter.format_ticks([1.0, 1.0 + 1.5 / 3600]) == ["00:01:00:00.0", "00:01:00:01.5"]
    assert formatter.format_data_short(22.3 + 1.234 / 3600) == "00:22:18:01.23"
