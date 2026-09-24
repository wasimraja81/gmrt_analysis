"""Unit tests for characterise_ripple.plot_ripple_characterisation (RC-11).

Uses a real characterise_ripple() run on synthetic data (rather than a
hand-built result dict) so the plot function is exercised against the
actual shape produced by the orchestration function.
"""

import matplotlib
matplotlib.use('Agg')  # headless, no display needed for these tests

import numpy as np
import pytest

from characterise_ripple import characterise_ripple, plot_ripple_characterisation


def _make_result(make_synthetic_vis_corrected, make_synthetic_spectrum, ripple_components=None):
    freqs_hz = np.linspace(300.0, 332.0, 256) * 1e6
    data = make_synthetic_spectrum(
        freqs_hz, alpha=-0.7, beta=0.0,
        ripple_components=ripple_components, noise_sigma_jy=1e-4, seed=0,
    )
    nrows, npol = 5, 2
    vis_complex = np.broadcast_to(
        data['spectrum_jy'][None, :, None], (nrows, freqs_hz.size, npol)
    ).copy().astype(np.complex128)
    vis = make_synthetic_vis_corrected(nrows=nrows, nchan=freqs_hz.size, pols=('RR', 'LL'), vis_complex=vis_complex)
    vis['freqs_hz'] = freqs_hz
    return characterise_ripple(vis, solution=None, source='UNREGISTEREDSRC')


def test_plot_runs_and_writes_nonempty_file_with_components(
    tmp_path, make_synthetic_vis_corrected, make_synthetic_spectrum,
):
    result = _make_result(
        make_synthetic_vis_corrected, make_synthetic_spectrum,
        ripple_components=[{'period_mhz': 8.0, 'amplitude': 0.05}],
    )
    save_path = tmp_path / 'plot.png'

    fig = plot_ripple_characterisation(result, save_path)

    assert save_path.exists()
    assert save_path.stat().st_size > 0
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_runs_on_zero_component_edge_case(
    tmp_path, make_synthetic_vis_corrected, make_synthetic_spectrum,
):
    # No ripple injected -> harmonic search should find nothing significant.
    result = _make_result(make_synthetic_vis_corrected, make_synthetic_spectrum, ripple_components=None)
    save_path = tmp_path / 'plot_zero_components.png'

    fig = plot_ripple_characterisation(result, save_path)

    assert save_path.exists()
    assert save_path.stat().st_size > 0
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_creates_missing_output_directory(
    tmp_path, make_synthetic_vis_corrected, make_synthetic_spectrum,
):
    result = _make_result(make_synthetic_vis_corrected, make_synthetic_spectrum)
    save_path = tmp_path / 'nested' / 'does' / 'not' / 'exist' / 'plot.png'

    fig = plot_ripple_characterisation(result, save_path)

    assert save_path.exists()
    import matplotlib.pyplot as plt
    plt.close(fig)
