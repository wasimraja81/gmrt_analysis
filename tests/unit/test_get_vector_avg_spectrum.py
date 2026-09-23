"""Unit tests for modules.ugmrt_query.get_vector_avg_spectrum (RC-08).

Reproduces the weighted-average logic by hand on small synthetic
vis_corrected-shaped fixtures and checks edge-case behaviour (all-flagged
channel, zero-weight rows, chan_mask suppression).
"""

import numpy as np
import pytest

from modules import ugmrt_query as q


def test_reproduces_hand_computed_weighted_average(make_synthetic_vis_corrected):
    # 2 rows, 2 channels, 1 pol, known weights/values so the weighted average
    # can be computed by hand.
    vis_complex = np.array(
        [[[1.0 + 0.0j], [2.0 + 0.0j]],
         [[3.0 + 0.0j], [4.0 + 0.0j]]]
    )
    weight = np.array(
        [[[1.0], [1.0]],
         [[3.0], [1.0]]]
    )
    flagged = np.zeros((2, 2, 1), dtype=bool)

    vis = make_synthetic_vis_corrected(
        nrows=2, nchan=2, pols=('RR',),
        vis_complex=vis_complex, weight=weight, flagged=flagged,
    )

    result = q.get_vector_avg_spectrum(vis, 'RR')

    # channel 0: (1*1 + 3*3) / (1+3) = 10/4 = 2.5
    # channel 1: (1*2 + 1*4) / (1+1) = 6/2 = 3.0
    np.testing.assert_allclose(result, [2.5, 3.0])


def test_all_flagged_channel_is_nan_not_zero(make_synthetic_vis_corrected):
    vis_complex = np.array(
        [[[1.0 + 0.0j], [2.0 + 0.0j]],
         [[3.0 + 0.0j], [4.0 + 0.0j]]]
    )
    weight = np.ones((2, 2, 1))
    # Flag every row for channel 0 only.
    flagged = np.array(
        [[[True], [False]],
         [[True], [False]]]
    )

    vis = make_synthetic_vis_corrected(
        nrows=2, nchan=2, pols=('RR',),
        vis_complex=vis_complex, weight=weight, flagged=flagged,
    )

    result = q.get_vector_avg_spectrum(vis, 'RR')

    assert np.isnan(result[0])
    assert not np.isnan(result[1])


def test_zero_weight_rows_excluded(make_synthetic_vis_corrected):
    vis_complex = np.array(
        [[[1.0 + 0.0j]],
         [[100.0 + 0.0j]]]
    )
    # Second row has zero weight -> must not contribute, even though its
    # value would otherwise dominate the average.
    weight = np.array([[[1.0]], [[0.0]]])
    flagged = np.zeros((2, 1, 1), dtype=bool)

    vis = make_synthetic_vis_corrected(
        nrows=2, nchan=1, pols=('RR',),
        vis_complex=vis_complex, weight=weight, flagged=flagged,
    )

    result = q.get_vector_avg_spectrum(vis, 'RR')

    np.testing.assert_allclose(result, [1.0])


def test_chan_mask_suppresses_edge_channels(make_synthetic_vis_corrected):
    vis = make_synthetic_vis_corrected(nrows=3, nchan=4, pols=('RR', 'LL'))

    chan_mask = np.array([False, True, True, False])
    result = q.get_vector_avg_spectrum(vis, 'RR', chan_mask=chan_mask)

    assert np.isnan(result[0])
    assert np.isnan(result[3])
    assert not np.isnan(result[1])
    assert not np.isnan(result[2])


def test_falls_back_to_uncorrected_keys_when_corrected_absent(make_synthetic_vis_corrected):
    vis = make_synthetic_vis_corrected(nrows=2, nchan=2, pols=('RR',))
    # Simulate a raw (uncorrected) vis dict: rename keys to the
    # non-'_corrected' variants that load_vis_for_source() produces.
    raw_vis = {
        'stokes_labels': vis['stokes_labels'],
        'vis_complex': vis['vis_complex_corrected'],
        'weight': vis['weight'],
        'flagged': vis['flagged_corrected'],
    }

    result_raw = q.get_vector_avg_spectrum(raw_vis, 'RR')
    result_corrected = q.get_vector_avg_spectrum(vis, 'RR')

    np.testing.assert_allclose(result_raw, result_corrected)


def test_unknown_pol_label_raises(make_synthetic_vis_corrected):
    vis = make_synthetic_vis_corrected(nrows=2, nchan=2, pols=('RR', 'LL'))
    with pytest.raises(ValueError):
        q.get_vector_avg_spectrum(vis, 'RL')
