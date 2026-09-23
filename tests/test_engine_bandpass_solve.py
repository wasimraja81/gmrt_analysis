import numpy as np
import pytest

from engine.bandpass_solve import solve_channel_gains


def _all_baseline_pairs(n_ant: int):
    ant1, ant2 = [], []
    for a in range(n_ant):
        for b in range(a + 1, n_ant):
            ant1.append(a)
            ant2.append(b)
    return np.array(ant1), np.array(ant2)


def _synthesize_visibilities(true_gains: np.ndarray, model: np.ndarray, ant1_idx: np.ndarray, ant2_idx: np.ndarray):
    return true_gains[ant1_idx] * np.conj(true_gains[ant2_idx]) * model


def test_solver_recovers_known_gains_from_noise_free_data():
    rng = np.random.default_rng(seed=42)
    n_ant = 12
    reference_antenna_idx = 0

    amplitudes = rng.uniform(0.7, 1.3, size=n_ant)
    phases = rng.uniform(-np.pi, np.pi, size=n_ant)
    true_gains = amplitudes * np.exp(1j * phases)
    true_gains *= np.exp(-1j * np.angle(true_gains[reference_antenna_idx]))  # reference phase = 0

    ant1_idx, ant2_idx = _all_baseline_pairs(n_ant)
    model = np.ones(len(ant1_idx), dtype=complex) * 5.0  # a 5 Jy point source

    vis = _synthesize_visibilities(true_gains, model, ant1_idx, ant2_idx)

    result = solve_channel_gains(vis, model, ant1_idx, ant2_idx, n_ant, reference_antenna_idx)

    assert result.converged
    np.testing.assert_allclose(result.gains, true_gains, atol=1e-5)
    assert result.residual_rms < 1e-5


def test_solver_recovers_known_gains_with_small_noise():
    rng = np.random.default_rng(seed=7)
    n_ant = 10
    reference_antenna_idx = 3

    amplitudes = rng.uniform(0.8, 1.2, size=n_ant)
    phases = rng.uniform(-np.pi, np.pi, size=n_ant)
    true_gains = amplitudes * np.exp(1j * phases)
    true_gains *= np.exp(-1j * np.angle(true_gains[reference_antenna_idx]))

    ant1_idx, ant2_idx = _all_baseline_pairs(n_ant)
    model = np.ones(len(ant1_idx), dtype=complex) * 3.0

    vis = _synthesize_visibilities(true_gains, model, ant1_idx, ant2_idx)
    noise_sigma = 0.01
    noise = (rng.normal(0, noise_sigma, len(vis)) + 1j * rng.normal(0, noise_sigma, len(vis)))
    noisy_vis = vis + noise

    result = solve_channel_gains(noisy_vis, model, ant1_idx, ant2_idx, n_ant, reference_antenna_idx)

    assert result.converged
    np.testing.assert_allclose(result.gains, true_gains, atol=0.01)


def test_solver_leaves_an_isolated_antennas_gain_at_its_initial_value():
    rng = np.random.default_rng(seed=1)
    n_ant = 6
    reference_antenna_idx = 0
    isolated_antenna_idx = n_ant - 1  # no baselines will touch this antenna

    amplitudes = rng.uniform(0.9, 1.1, size=n_ant)
    phases = rng.uniform(-np.pi, np.pi, size=n_ant)
    true_gains = amplitudes * np.exp(1j * phases)
    true_gains *= np.exp(-1j * np.angle(true_gains[reference_antenna_idx]))

    ant1_idx, ant2_idx = _all_baseline_pairs(n_ant)
    keep = (ant1_idx != isolated_antenna_idx) & (ant2_idx != isolated_antenna_idx)
    ant1_idx, ant2_idx = ant1_idx[keep], ant2_idx[keep]
    model = np.ones(len(ant1_idx), dtype=complex) * 5.0

    vis = _synthesize_visibilities(true_gains, model, ant1_idx, ant2_idx)

    result = solve_channel_gains(vis, model, ant1_idx, ant2_idx, n_ant, reference_antenna_idx)

    assert result.gains[isolated_antenna_idx] == 1.0 + 0j
    other_antennas = [i for i in range(n_ant) if i != isolated_antenna_idx]
    np.testing.assert_allclose(result.gains[other_antennas], true_gains[other_antennas], atol=1e-5)


def test_solver_reports_not_converged_when_max_iter_is_too_low():
    rng = np.random.default_rng(seed=99)
    n_ant = 15
    reference_antenna_idx = 0

    amplitudes = rng.uniform(0.5, 1.5, size=n_ant)
    phases = rng.uniform(-np.pi, np.pi, size=n_ant)
    true_gains = amplitudes * np.exp(1j * phases)
    true_gains *= np.exp(-1j * np.angle(true_gains[reference_antenna_idx]))

    ant1_idx, ant2_idx = _all_baseline_pairs(n_ant)
    model = np.ones(len(ant1_idx), dtype=complex) * 2.0
    vis = _synthesize_visibilities(true_gains, model, ant1_idx, ant2_idx)

    result = solve_channel_gains(vis, model, ant1_idx, ant2_idx, n_ant, reference_antenna_idx, max_iter=1)

    assert result.n_iterations == 1
    assert result.converged is False


def test_solver_rejects_input_containing_an_autocorrelation():
    n_ant = 5
    ant1_idx, ant2_idx = _all_baseline_pairs(n_ant)
    # Sneak in one autocorrelation (ant1 == ant2) alongside cross baselines.
    ant1_idx = np.append(ant1_idx, 2)
    ant2_idx = np.append(ant2_idx, 2)
    vis = np.ones(len(ant1_idx), dtype=complex)
    model = np.ones(len(ant1_idx), dtype=complex)

    with pytest.raises(ValueError, match="autocorrelation"):
        solve_channel_gains(vis, model, ant1_idx, ant2_idx, n_ant, reference_antenna_idx=0)
