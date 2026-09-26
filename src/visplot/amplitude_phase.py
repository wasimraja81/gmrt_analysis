"""Amplitude/phase diagnostic plot, against a caller-supplied per-row x-axis
(time, uvdist, ...) -- generic over what the row axis represents, since
that is a selection choice, not a fact about the data itself.

Values are read as-is from a `VisibilityBlock`, before any calibration --
this is a Phase B ("know your data") diagnostic, not a calibrated
amplitude in Jy. The AIPS weight convention (<=0 means flagged) is
respected: flagged cells are excluded, not plotted as zero.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def amplitude_phase(x: np.ndarray, data: np.ndarray, weight: np.ndarray, xlabel: str) -> Figure:
    """Amplitude and phase, in two panels sharing an x-axis, for every
    unflagged cell of `data` (shape (n_rows, ...), matching `weight`) --
    `x` is one value per row, broadcast across whatever other axes `data`
    carries (channel, Stokes, IF, ...)."""
    x = np.asarray(x, dtype=float)
    data = np.asarray(data)
    weight = np.asarray(weight)

    n_rows = data.shape[0]
    data_flat = data.reshape(n_rows, -1)
    weight_flat = weight.reshape(n_rows, -1)
    x_flat = np.repeat(x, data_flat.shape[1])

    good = ((weight_flat > 0) & np.isfinite(data_flat.real) & np.isfinite(data_flat.imag)).ravel()
    amp = np.abs(data_flat).ravel()
    phase_deg = np.degrees(np.angle(data_flat)).ravel()

    fig, (ax_amp, ax_phase) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax_amp.scatter(x_flat[good], amp[good], s=2, alpha=0.3, c="tab:blue")
    ax_amp.set_ylabel("Amplitude (uncalibrated)")
    ax_amp.grid(True, alpha=0.3)

    ax_phase.scatter(x_flat[good], phase_deg[good], s=2, alpha=0.3, c="tab:blue")
    ax_phase.set_ylabel("Phase (deg)")
    ax_phase.set_ylim(-180, 180)
    ax_phase.set_xlabel(xlabel)
    ax_phase.grid(True, alpha=0.3)

    fig.suptitle("Amplitude / Phase")
    return fig
