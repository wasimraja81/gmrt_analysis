import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pytest

from visplot.uv_coverage import uv_coverage


def test_uv_coverage_converts_seconds_and_frequency_to_kilolambda():
    # 1 microsecond of delay at 1 GHz is exactly 1000 wavelengths = 1 klambda.
    uu_sec = np.array([1e-6])
    vv_sec = np.array([0.0])
    freq_hz = np.array([1e9])

    fig = uv_coverage(uu_sec, vv_sec, freq_hz)
    ax = fig.axes[0]
    offsets = ax.collections[0].get_offsets()  # the (+u, +v) scatter, added first
    np.testing.assert_allclose(offsets, [[1.0, 0.0]])
    plt.close(fig)


def test_uv_coverage_includes_the_conjugate_point():
    uu_sec = np.array([1e-6])
    vv_sec = np.array([2e-6])
    freq_hz = np.array([1e9])

    fig = uv_coverage(uu_sec, vv_sec, freq_hz)
    ax = fig.axes[0]
    positive = ax.collections[0].get_offsets()
    conjugate = ax.collections[1].get_offsets()
    np.testing.assert_allclose(positive, [[1.0, 2.0]])
    np.testing.assert_allclose(conjugate, [[-1.0, -2.0]])
    plt.close(fig)


def test_uv_coverage_broadcasts_over_multiple_channels():
    # 2 rows x 3 channels -> 6 points per (u, v) scatter, not 2.
    uu_sec = np.array([1e-6, 2e-6])
    vv_sec = np.array([0.0, 0.0])
    freq_hz = np.array([1e9, 1.1e9, 1.2e9])

    fig = uv_coverage(uu_sec, vv_sec, freq_hz)
    ax = fig.axes[0]
    assert len(ax.collections[0].get_offsets()) == 6
    plt.close(fig)


def test_uv_coverage_accepts_a_scalar_reference_frequency():
    fig = uv_coverage(np.array([1e-6]), np.array([0.0]), 1e9)
    ax = fig.axes[0]
    np.testing.assert_allclose(ax.collections[0].get_offsets(), [[1.0, 0.0]])
    plt.close(fig)
