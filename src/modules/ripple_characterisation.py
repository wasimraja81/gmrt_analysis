"""Pure-numerics core for standing-wave ripple characterisation.

Every function in this module takes and returns NumPy arrays and plain
dicts only — no ``vis`` dicts, no matplotlib, no file I/O. This is what the
pytest suite exercises directly with synthetic ground truth, with zero
UVFITS/CASA dependency. Orchestration (loading real data, applying a
bandpass solution, writing plots/JSON/reports) lives in
``src/characterise_ripple.py``.

See ``ripple_characterisation_tickets.md`` for the ticket-by-ticket design
of each function, and ``ripple_convergence_todos.md`` (Step 1 / Step 1b) for
the underlying physical model.
"""

from __future__ import annotations

import numpy as np


def mad_sigma(values: np.ndarray) -> float:
    """Robust noise-floor estimate via the median absolute deviation.

    Implements the noise floor convention from ``ripple_convergence_todos.md``
    Step 1:

        sigma_hat = 1.4826 * MAD(values)

    where MAD is the median absolute deviation from the median. The 1.4826
    factor converts MAD to a Gaussian-equivalent standard deviation.

    Parameters
    ----------
    values : np.ndarray
        1D array, may contain NaN (NaNs are ignored).

    Returns
    -------
    float
        ``sigma_hat``, or ``nan`` if ``values`` is empty or entirely NaN.
    """
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float('nan')
    median = np.median(finite)
    mad = np.median(np.abs(finite - median))
    return float(1.4826 * mad)
