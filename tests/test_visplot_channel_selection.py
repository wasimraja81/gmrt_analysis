import numpy as np
import pytest

from visplot.channel_selection import resolve_channel_selection

# 5 channels, 100-104 MHz, 1 MHz wide each.
CHAN_FREQS_HZ = np.array([100e6, 101e6, 102e6, 103e6, 104e6])


def test_resolve_channel_selection_by_plain_index():
    assert resolve_channel_selection("1:3", CHAN_FREQS_HZ) == [1, 2, 3]


def test_resolve_channel_selection_by_index_union():
    assert resolve_channel_selection("0,2:3", CHAN_FREQS_HZ) == [0, 2, 3]


def test_resolve_channel_selection_by_index_rejects_out_of_range():
    with pytest.raises(ValueError, match="out of range"):
        resolve_channel_selection("3:10", CHAN_FREQS_HZ)


def test_resolve_channel_selection_by_frequency_band():
    # 101-103 MHz covers channels 1, 2, 3 exactly.
    assert resolve_channel_selection("101:103MHz", CHAN_FREQS_HZ) == [1, 2, 3]


def test_resolve_channel_selection_by_frequency_in_hz():
    assert resolve_channel_selection("101e6:103e6Hz", CHAN_FREQS_HZ) == [1, 2, 3]


def test_resolve_channel_selection_by_frequency_excludes_channels_outside_the_band():
    assert resolve_channel_selection("100.5:101.5MHz", CHAN_FREQS_HZ) == [1]


def test_resolve_channel_selection_by_frequency_union_of_two_bands():
    assert resolve_channel_selection("100:100.5MHz,103.5:104MHz", CHAN_FREQS_HZ) == [0, 4]


def test_resolve_channel_selection_by_frequency_does_not_require_ascending_channel_order():
    descending = CHAN_FREQS_HZ[::-1]  # 104, 103, 102, 101, 100 MHz
    assert resolve_channel_selection("101:103MHz", descending) == [1, 2, 3]
