import pytest

from data_io.antenna_table import Antenna
from visplot.antenna_selection import resolve_antenna_selection

ANTENNAS = [
    Antenna(station_number=1, name="C00:01", x_m=0.0, y_m=0.0, z_m=0.0),
    Antenna(station_number=2, name="C01:02", x_m=0.0, y_m=0.0, z_m=0.0),
    Antenna(station_number=25, name="W01:25", x_m=0.0, y_m=0.0, z_m=0.0),
    Antenna(station_number=30, name="W06:30", x_m=0.0, y_m=0.0, z_m=0.0),
]


def test_resolve_antenna_selection_by_numeric_id():
    assert resolve_antenna_selection("1,25", ANTENNAS) == [1, 25]


def test_resolve_antenna_selection_by_numeric_range():
    assert resolve_antenna_selection("1:2", ANTENNAS) == [1, 2]


def test_resolve_antenna_selection_by_name():
    assert resolve_antenna_selection("W01:25", ANTENNAS) == [25]


def test_resolve_antenna_selection_by_name_is_case_insensitive():
    assert resolve_antenna_selection("w01:25", ANTENNAS) == [25]


def test_resolve_antenna_selection_mixes_names_and_ids_in_one_spec():
    assert resolve_antenna_selection("1,W01:25,W06:30", ANTENNAS) == [1, 25, 30]


def test_resolve_antenna_selection_name_takes_priority_over_numeric_parsing():
    # "C00:01" would otherwise be misread as the numeric range 0:1 -- the
    # name lookup must be tried first.
    assert resolve_antenna_selection("C00:01", ANTENNAS) == [1]


def test_resolve_antenna_selection_deduplicates_and_sorts():
    assert resolve_antenna_selection("25,1,1", ANTENNAS) == [1, 25]


def test_resolve_antenna_selection_raises_for_an_unrecognized_term():
    with pytest.raises(ValueError, match="neither a known antenna name nor a valid numeric"):
        resolve_antenna_selection("not-a-real-antenna", ANTENNAS)
