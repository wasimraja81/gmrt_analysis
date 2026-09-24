"""Argparse-level unit tests for the RC-13 CLI driver (src/characterise_ripple.py::main).

Full end-to-end behaviour (real UVFITS/CASA-adjacent I/O) is covered by the
shell regression gate (RC-15), not pytest — these tests only exercise
argument parsing/validation, which needs no real data.
"""

import pytest

from characterise_ripple import _build_arg_parser, main


def test_required_args_enforced():
    with pytest.raises(SystemExit):
        _build_arg_parser().parse_args([])


def test_fits_and_source_required_together():
    with pytest.raises(SystemExit):
        _build_arg_parser().parse_args(['--fits', 'x.fits'])  # --source missing


def test_invalid_physical_model_mode_choice_rejected():
    with pytest.raises(SystemExit):
        _build_arg_parser().parse_args([
            '--fits', 'x.fits', '--source', 'SRC', '--physical-model-mode', 'bogus',
        ])


def test_invalid_period_bounds_mode_choice_rejected():
    with pytest.raises(SystemExit):
        _build_arg_parser().parse_args([
            '--fits', 'x.fits', '--source', 'SRC', '--period-bounds-mode', 'bogus',
        ])


def test_defaults_applied_when_optional_args_omitted():
    args = _build_arg_parser().parse_args(['--fits', 'x.fits', '--source', 'SRC'])

    assert args.physical_model_mode == 'fit'
    assert args.period_bounds_mode == 'fourier'
    assert args.n_harmonics == 1
    assert args.peak_snr_threshold == 10.0
    assert args.tau_ripple == 2.0
    assert args.velocity_factor == 1.0
    assert args.bandpass_solution is None
    assert args.chan_range is None


def test_main_errors_on_manual_period_bounds_without_both_values():
    with pytest.raises(SystemExit):
        main(['--fits', 'x.fits', '--source', 'SRC', '--period-bounds-mode', 'manual',
              '--period-min-mhz', '5.0'])
