import numpy as np
import pytest

from data_io.visibility_data import VisibilityBlock
from visplot.derived_quantities import compute_quantity


def _make_block(chan_freqs_hz=np.array([1e9, 1.1e9])):
    # 2 rows x 2 Stokes x 2 freq channels.
    data = np.array([
        [[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]],
        [[9 + 10j, 11 + 12j], [13 + 14j, -1 + 0j]],
    ])
    weight = np.ones((2, 2, 2))
    return VisibilityBlock(
        row_indices=np.array([0, 1]),
        data=data, weight=weight,
        axis_types=["STOKES", "FREQ"],
        axis_indices={"STOKES": np.array([0, 1]), "FREQ": np.array([0, 1])},
        ant1=np.array([1, 1]), ant2=np.array([2, 2]),
        jd=np.array([100.0, 100.5]),
        uu_sec=np.array([1e-6, 2e-6]), vv_sec=np.array([3e-6, 4e-6]), ww_sec=np.array([5e-6, 6e-6]),
        chan_freqs_hz=chan_freqs_hz,
        stokes_labels=["RR", "LL"],
    )


def test_compute_quantity_real_imag_amp_phase_read_directly_from_data():
    block = _make_block()
    np.testing.assert_array_equal(compute_quantity("real", block), block.data.real)
    np.testing.assert_array_equal(compute_quantity("imag", block), block.data.imag)
    np.testing.assert_allclose(compute_quantity("amp", block), np.abs(block.data))
    np.testing.assert_allclose(compute_quantity("phase_deg", block), np.degrees(np.angle(block.data)))


def test_compute_quantity_time_h_broadcasts_per_row():
    block = _make_block()
    time_h = compute_quantity("time_h", block)
    assert time_h.shape == block.data.shape
    np.testing.assert_allclose(time_h[0], 0.0)
    np.testing.assert_allclose(time_h[1], 12.0)  # 0.5 days later


def test_compute_quantity_u_sec_broadcasts_per_row():
    block = _make_block()
    u_sec = compute_quantity("u_sec", block)
    assert u_sec.shape == block.data.shape
    np.testing.assert_allclose(u_sec[0], 1e-6)
    np.testing.assert_allclose(u_sec[1], 2e-6)


def test_compute_quantity_uvdist_m_computes_correctly():
    block = _make_block()
    uvdist_m = compute_quantity("uvdist_m", block)
    expected_row0 = np.hypot(1e-6, 3e-6) * 299_792_458.0
    np.testing.assert_allclose(uvdist_m[0], expected_row0)


def test_compute_quantity_freq_mhz_broadcasts_per_channel():
    block = _make_block()
    freq_mhz = compute_quantity("freq_mhz", block)
    assert freq_mhz.shape == block.data.shape
    np.testing.assert_allclose(freq_mhz[:, :, 0], 1000.0)
    np.testing.assert_allclose(freq_mhz[:, :, 1], 1100.0)


def test_compute_quantity_u_klambda_combines_row_and_channel():
    block = _make_block()
    u_klambda = compute_quantity("u_klambda", block)
    # row 0, channel 0: 1e-6 sec * 1e9 Hz / 1e3 = 1.0 klambda
    assert u_klambda[0, 0, 0] == pytest.approx(1.0)
    # row 0, channel 1: 1e-6 sec * 1.1e9 Hz / 1e3 = 1.1 klambda
    assert u_klambda[0, 0, 1] == pytest.approx(1.1)


def test_compute_quantity_stokes_broadcasts_labels():
    block = _make_block()
    stokes = compute_quantity("stokes", block)
    assert stokes.shape == block.data.shape
    assert (stokes[:, 0, :] == "RR").all()
    assert (stokes[:, 1, :] == "LL").all()


def test_compute_quantity_raises_for_an_unknown_name():
    block = _make_block()
    with pytest.raises(ValueError, match="unknown quantity"):
        compute_quantity("not_a_real_quantity", block)


def test_compute_quantity_klambda_raises_without_a_freq_axis():
    block = _make_block(chan_freqs_hz=None)
    with pytest.raises(ValueError, match="uvdist_klambda"):
        compute_quantity("uvdist_klambda", block)
