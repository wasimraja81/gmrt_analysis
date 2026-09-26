import pytest

from data_io.antenna_table import Antenna
from instruments.gmrt.antenna_selection import resolve_antenna_selection

ANTENNAS = [
    Antenna(station_number=1, name="C00:01", x_m=0.0, y_m=0.0, z_m=0.0),
    Antenna(station_number=2, name="C01:02", x_m=0.0, y_m=0.0, z_m=0.0),
    Antenna(station_number=25, name="W01:25", x_m=0.0, y_m=0.0, z_m=0.0),
    Antenna(station_number=30, name="W06:30", x_m=0.0, y_m=0.0, z_m=0.0),
]


def test_resolve_antenna_selection_by_bare_gmrt_prefix():
    assert resolve_antenna_selection("C00", ANTENNAS) == [1]


def test_resolve_antenna_selection_by_bare_prefix_is_case_insensitive():
    assert resolve_antenna_selection("c00", ANTENNAS) == [1]


def test_resolve_antenna_selection_still_supports_full_names_and_ids():
    assert resolve_antenna_selection("W01:25,2", ANTENNAS) == [2, 25]


def test_resolve_antenna_selection_mixes_prefix_full_name_and_id():
    assert resolve_antenna_selection("C00,W01:25,2", ANTENNAS) == [1, 2, 25]


def test_resolve_antenna_selection_raises_for_a_prefix_matching_more_than_one_antenna():
    ambiguous = ANTENNAS + [Antenna(station_number=99, name="C00:99", x_m=0.0, y_m=0.0, z_m=0.0)]
    with pytest.raises(ValueError, match="matches more than one antenna"):
        resolve_antenna_selection("C00", ambiguous)


def test_resolve_antenna_selection_raises_for_an_unrecognized_term():
    with pytest.raises(ValueError, match="neither a known antenna name nor a valid numeric"):
        resolve_antenna_selection("not-a-real-antenna", ANTENNAS)
