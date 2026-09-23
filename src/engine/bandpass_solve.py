"""Per-channel antenna-gain solve (StefCal-style alternating least squares).

Solves for complex antenna gains g such that, for every baseline (a, b),
the measured visibility V_ab approx= g_a * conj(g_b) * M_ab, where M is a
point-source model. One call solves one channel; a caller loops over
channels (see docs/dev/GWB_PIPELINE_REFACTOR_PLAN.md T5c for the outer
loop that will do that, plus flagging and convergence across iterations).

An antenna with no baselines in a given call (no data at all -- for any
reason, including GMRT's DUD antennas, see T5b) is left at its untouched
initial gain rather than solved or phase-rotated, so "gain is exactly
1+0j" is itself a signal that an antenna was never constrained by data.
This is a robustness fallback, not DUD-antenna handling itself -- DUD
antennas must be excluded from the antenna list before n_ant or any
baseline count is computed anywhere in the pipeline, since every
independent baseline-count calculation (flagging-percentage denominators,
coverage statistics, etc.) would otherwise be wrong on its own, not just
the solve. See T5b for where that belongs.

Callers must also exclude autocorrelations (ant1 == ant2) before calling
this function -- see the precondition on solve_channel_gains below.

Algorithm: Salvini & Wijnholds (2014)-style alternating update -- each
antenna's gain is re-estimated holding every other antenna's gain fixed
(a per-antenna complex least-squares solve), then relaxed by averaging
with the previous estimate, repeated until the largest per-antenna change
falls below `tol` or `max_iter` is reached. A reference antenna's phase is
fixed to zero afterward, since g_a * conj(g_b) is invariant under a global
phase rotation of every gain -- without fixing one, the solution is only
determined up to that rotation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ChannelGainSolution:
    gains: np.ndarray  # complex, shape (n_ant,)
    converged: bool
    n_iterations: int
    residual_rms: float


def solve_channel_gains(
    vis: np.ndarray,
    model: np.ndarray,
    ant1_idx: np.ndarray,
    ant2_idx: np.ndarray,
    n_ant: int,
    reference_antenna_idx: int,
    max_iter: int = 100,
    tol: float = 1e-7,
) -> ChannelGainSolution:
    """Solve for one channel's antenna gains from baseline visibilities.

    vis, model: complex arrays, shape (n_baselines,). model[k] is the
    point-source model visibility for baseline (ant1_idx[k], ant2_idx[k]),
    related by vis[k] approx= gains[ant1_idx[k]] * conj(gains[ant2_idx[k]]) * model[k].

    Precondition: every baseline must be a cross-correlation (ant1_idx[k] !=
    ant2_idx[k]). An autocorrelation (ant1 == ant2) is a different physical
    quantity -- total power, no phase information -- and does not fit this
    model at all; silently including one would corrupt the solve rather than
    raise an error, so it's checked here instead of left as an assumption
    the caller has to remember. Excluding autocorrelations (and
    resolving which antennas are even active) is the caller's job -- for
    GMRT, see the data adapter in T5b, docs/dev/GWB_PIPELINE_REFACTOR_PLAN.md.
    """
    if np.any(ant1_idx == ant2_idx):
        raise ValueError(
            "solve_channel_gains received at least one autocorrelation (ant1_idx == "
            "ant2_idx); only cross-correlation baselines are valid here -- exclude "
            "autocorrelations before calling this function"
        )

    gains = np.ones(n_ant, dtype=complex)
    converged = False
    iteration = 0

    has_any_data = np.zeros(n_ant, dtype=bool)
    has_any_data[ant1_idx] = True
    has_any_data[ant2_idx] = True

    for iteration in range(1, max_iter + 1):
        numerator = np.zeros(n_ant, dtype=complex)
        denominator = np.zeros(n_ant, dtype=float)

        contrib_a = gains[ant2_idx] * np.conj(model) * vis
        weight_a = np.abs(gains[ant2_idx]) ** 2 * np.abs(model) ** 2
        np.add.at(numerator, ant1_idx, contrib_a)
        np.add.at(denominator, ant1_idx, weight_a)

        contrib_b = gains[ant1_idx] * model * np.conj(vis)
        weight_b = np.abs(gains[ant1_idx]) ** 2 * np.abs(model) ** 2
        np.add.at(numerator, ant2_idx, contrib_b)
        np.add.at(denominator, ant2_idx, weight_b)

        has_data = denominator > 0
        solved = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=has_data)
        new_gains = np.where(has_data, solved, gains)
        relaxed_gains = 0.5 * (gains + new_gains)

        delta = np.max(np.abs(relaxed_gains - gains))
        scale = max(np.max(np.abs(gains)), 1.0)
        gains = relaxed_gains

        if delta < tol * scale:
            converged = True
            break

    gains = _fix_reference_phase(gains, reference_antenna_idx, has_any_data)
    residual_rms = _residual_rms(vis, model, gains, ant1_idx, ant2_idx)

    return ChannelGainSolution(
        gains=gains,
        converged=converged,
        n_iterations=iteration,
        residual_rms=residual_rms,
    )


def _fix_reference_phase(gains: np.ndarray, reference_antenna_idx: int, has_any_data: np.ndarray) -> np.ndarray:
    """Rotate every solved antenna's phase so the reference antenna's phase is zero.

    Antennas with no data at all (has_any_data[i] is False) are left untouched --
    they were never solved, so there is no phase to correct, and rotating them
    anyway would turn their honest "no data" sentinel value into an arbitrary
    one instead.
    """
    ref = gains[reference_antenna_idx]
    if ref == 0:
        return gains
    phase_correction = ref / np.abs(ref)
    corrected = gains / phase_correction
    return np.where(has_any_data, corrected, gains)


def _residual_rms(
    vis: np.ndarray, model: np.ndarray, gains: np.ndarray, ant1_idx: np.ndarray, ant2_idx: np.ndarray
) -> float:
    predicted = gains[ant1_idx] * np.conj(gains[ant2_idx]) * model
    residuals = vis - predicted
    return float(np.sqrt(np.mean(np.abs(residuals) ** 2)))
