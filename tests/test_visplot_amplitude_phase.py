import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np

from visplot.amplitude_phase import amplitude_phase


def test_amplitude_phase_computes_amp_and_phase_from_complex_data():
    x = np.array([0.0, 1.0])
    data = np.array([[3.0 + 4.0j], [0.0 + 1.0j]])  # amp 5, phase 53.13deg; amp 1, phase 90deg
    weight = np.ones_like(data, dtype=float)

    fig = amplitude_phase(x, data, weight, xlabel="Time (h)")
    ax_amp, ax_phase = fig.axes

    amp_points = ax_amp.collections[0].get_offsets()
    phase_points = ax_phase.collections[0].get_offsets()
    np.testing.assert_allclose(sorted(amp_points[:, 1]), [1.0, 5.0])
    np.testing.assert_allclose(sorted(phase_points[:, 1]), [53.13010235415598, 90.0])
    plt.close(fig)


def test_amplitude_phase_excludes_flagged_cells():
    # Row 0 is flagged (weight <= 0) and must not appear in either panel.
    x = np.array([0.0, 1.0])
    data = np.array([[100.0 + 0.0j], [1.0 + 0.0j]])
    weight = np.array([[-1.0], [1.0]])

    fig = amplitude_phase(x, data, weight, xlabel="Time (h)")
    ax_amp, _ = fig.axes

    amp_points = ax_amp.collections[0].get_offsets()
    assert len(amp_points) == 1
    np.testing.assert_allclose(amp_points, [[1.0, 1.0]])
    plt.close(fig)


def test_amplitude_phase_broadcasts_x_across_extra_axes():
    # 2 rows x 3 channels of data -- x (one per row) must repeat 3x per row.
    x = np.array([10.0, 20.0])
    data = np.ones((2, 3), dtype=complex)
    weight = np.ones((2, 3))

    fig = amplitude_phase(x, data, weight, xlabel="Time (h)")
    ax_amp, _ = fig.axes
    x_values = sorted(ax_amp.collections[0].get_offsets()[:, 0].tolist())
    assert x_values == [10.0, 10.0, 10.0, 20.0, 20.0, 20.0]
    plt.close(fig)
