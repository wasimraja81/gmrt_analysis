"""UV-coverage plot: sampled (u, v) points in wavelengths.

`uu_sec`/`vv_sec` (baseline projections divided by the speed of light, as
`RowIndex`/`VisibilityBlock` carry them) are frequency-independent; the
actual UV-plane sample position is frequency-dependent (u/v in wavelengths
= (u/v in seconds) * frequency in Hz), so this needs the frequencies of
whatever channels are in the selection, not just the row-level uu/vv.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def uv_coverage(uu_sec: np.ndarray, vv_sec: np.ndarray, freq_hz) -> Figure:
    """Sampled (u, v) points, in kilo-wavelengths, for every (row, channel)
    combination -- both a point and its conjugate (-u, -v), since a
    visibility measures both simultaneously."""
    uu_sec = np.asarray(uu_sec, dtype=float)
    vv_sec = np.asarray(vv_sec, dtype=float)
    freq_hz = np.atleast_1d(np.asarray(freq_hz, dtype=float))

    u_klambda = (uu_sec[:, np.newaxis] * freq_hz[np.newaxis, :] / 1e3).ravel()
    v_klambda = (vv_sec[:, np.newaxis] * freq_hz[np.newaxis, :] / 1e3).ravel()

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(u_klambda, v_klambda, s=2, alpha=0.4, c="tab:blue")
    ax.scatter(-u_klambda, -v_klambda, s=2, alpha=0.4, c="tab:blue")
    ax.set_xlabel("u (kλ)")
    ax.set_ylabel("v (kλ)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("UV coverage")
    ax.grid(True, alpha=0.3)
    return fig
