#!/usr/bin/env python3
"""pipeline_cli.py — unified workflow entrypoint.

This module provides a single Python entrypoint for the existing workflow while
preserving current behavior by delegating to the existing scripts.

Subcommands
-----------
preprocess
    Delegate to preprocess_ugmrt.py (legacy run_preprocess.sh target).

clustering
    Delegate to run_clustering.py (legacy run_clustering.sh target).

secondary
    Delegate to cal_solver.py (secondary calibration solve entrypoint).

solplot
    Delegate to gainPlots.py (gain-table visualization).

bandpass
    Unified Phase-1/Phase-2 orchestration (derive / audit / derive-and-audit).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent

_TOP_HELP = """usage: pipeline_cli.py {preprocess,clustering,secondary,solplot,bandpass,derive,audit,full} [args ...]

Unified pipeline entrypoint.

Commands
    preprocess   Run preprocess_ugmrt.py directly (Phase-1 driver interface)
    clustering   Run run_clustering.py directly (Phase-2 driver interface)
    secondary    Run cal_solver.py directly (secondary solve interface)
    solplot      Run gainPlots.py (primary/secondary gain solution plots)
    bandpass     Unified Phase-1/Phase-2 orchestration (preferred)
    derive       Alias for: bandpass --phase=derive
    audit        Alias for: bandpass --phase=audit
    full         Alias for: bandpass --phase=derive-and-audit

Quick start
    python pipeline_cli.py bandpass --phase=derive-and-audit --no-dry-run --auto --n-iters 100 \\
            --set "SOURCE='3C48'" --set "SOLVE_ELEVATION_MIN_DEG=25.0"
"""


_BANDPASS_HELP = """run_bandpass (via pipeline_cli)
usage: pipeline_cli.py bandpass --phase={derive|audit|derive-and-audit} [args ...]

Phases
    --phase=derive
            Phase-1 iterative flagging + bandpass solve.
    --phase=audit
            Phase-2 clustering audit on saved Phase-1 outputs.
    --phase=derive-and-audit
            Run derive first, then audit.

Shared flags (forwarded to both phases)
    --dry-run / --no-dry-run
    --docal on|off
    --doflag on|off
    --calver VER
    --flagver VER
    --config FILE
    --set KEY=expr
    --log-level LEVEL

Derive-only flags
    --auto
    --n-iters N
    --start-iter N

Audit-only flags
    --refit
    --write-refit / --no-write-refit
    --save-plots

Notes
    - Derive presets are applied by default for parity with legacy run_bandpass.sh.
    - Audit output is tee'd to: /tmp/audit_clustering.log
    - If MPLBACKEND is not set:
            derive uses Agg when --auto is present, else MacOSX;
            audit uses MacOSX.

Compatibility wrappers
    ./run_bandpass.sh ...   -> pipeline_cli.py bandpass ...
    ./run_preprocess.sh ... -> pipeline_cli.py preprocess ...
    ./run_clustering.sh ... -> pipeline_cli.py clustering ...
    ./secondaryCalibration_example.sh -> pipeline_cli.py secondary ...
    ./plotSecondarySolutions_example.sh -> pipeline_cli.py solplot ...
"""


def _run(cmd: list[str]) -> int:
    proc = subprocess.run(cmd)
    return int(proc.returncode)


def _run_with_env(cmd: list[str], env_overrides: dict | None = None) -> int:
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    proc = subprocess.run(cmd, env=env)
    return int(proc.returncode)


def _run_and_tee(cmd: list[str], log_path: Path, env_overrides: dict | None = None) -> int:
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'w', encoding='utf-8') as logf:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            logf.write(line)
        proc.wait()
        return int(proc.returncode)


def _choose_backends(derive_args: list[str]) -> tuple[str, str]:
    if os.environ.get('MPLBACKEND'):
        mb = os.environ['MPLBACKEND']
        return mb, mb
    is_auto = '--auto' in derive_args
    return ('Agg' if is_auto else 'MacOSX', 'MacOSX')


def _run_bandpass(extra: list[str]) -> int:
    if '-h' in extra or '--help' in extra:
        print(_BANDPASS_HELP)
        return 0

    phase = ''
    shared_args: list[str] = []
    derive_args: list[str] = []
    audit_args: list[str] = []
    dry_flag = '--dry-run'

    i = 0
    while i < len(extra):
        a = extra[i]
        if a.startswith('--phase='):
            phase = a.split('=', 1)[1]
            i += 1
            continue
        if a == '--phase' and i + 1 < len(extra):
            phase = extra[i + 1]
            i += 2
            continue
        if a == '--dry-run':
            dry_flag = '--dry-run'
            i += 1
            continue
        if a == '--no-dry-run':
            dry_flag = '--no-dry-run'
            i += 1
            continue

        if a == '--auto':
            derive_args.append(a)
            i += 1
            continue
        if a.startswith('--n-iters=') or a.startswith('--start-iter='):
            derive_args.append(a)
            i += 1
            continue
        if a in ('--n-iters', '--start-iter') and i + 1 < len(extra):
            derive_args.extend([a, extra[i + 1]])
            i += 2
            continue

        if a in ('--refit', '--write-refit', '--no-write-refit', '--save-plots'):
            audit_args.append(a)
            i += 1
            continue

        shared_args.append(a)
        i += 1

    if not phase:
        print('ERROR: --phase is required. Use derive, audit, or derive-and-audit.', file=sys.stderr)
        return 2
    if phase not in ('derive', 'audit', 'derive-and-audit'):
        print(f'ERROR: unknown --phase {phase!r}. Must be derive|audit|derive-and-audit', file=sys.stderr)
        return 2

    derive_backend, audit_backend = _choose_backends(derive_args)
    py = sys.executable
    preprocess_cmd = [py, str(SCRIPT_DIR / 'preprocess_ugmrt.py')]
    clustering_cmd = [py, str(SCRIPT_DIR / 'run_clustering.py')]

    derive_presets = [
        '--step', 'all',
        '--n-iters', '15',
        '--set', "OUTLIER_METRIC='V'",
        '--set', "OUTLIER_METRIC_MERGE_STRATEGY='union'",
        '--set', "ANTENNA_FLAG_THRESHOLD_JY={'V': 5.0}",
        '--set', "BASELINE_FLAG_THRESHOLD_JY={'V': 5.0}",
        '--set', "CONVERGENCE_COMBINE_STRATEGY='all'",
        '--set', "COMPARE_METRICS_FOR_CONVERGENCE=['V', 'Model']",
        '--set', "RUN_ITER0_DIAGNOSTIC=True",
    ]

    def run_derive() -> int:
        cmd = [*preprocess_cmd, *derive_presets, dry_flag, *shared_args, *derive_args]
        return _run_with_env(cmd, {'MPLBACKEND': derive_backend})

    def run_audit() -> int:
        cmd = [*clustering_cmd, dry_flag, *shared_args, *audit_args]
        audit_log = Path('/tmp/audit_clustering.log')
        print(f'Audit log: {audit_log}')
        code = _run_and_tee(cmd, audit_log, {'MPLBACKEND': audit_backend})
        print(f'Audit log saved to: {audit_log}')
        return code

    if phase == 'derive':
        return run_derive()
    if phase == 'audit':
        return run_audit()

    code = run_derive()
    if code != 0:
        return code
    return run_audit()


def main() -> int:
    argv = list(sys.argv[1:])
    if not argv or argv[0] in ('-h', '--help'):
        print(_TOP_HELP)
        return 0

    command, extra = argv[0], argv[1:]
    py = sys.executable

    if command == 'preprocess':
        return _run([py, str(SCRIPT_DIR / 'preprocess_ugmrt.py'), *extra])
    if command == 'clustering':
        return _run([py, str(SCRIPT_DIR / 'run_clustering.py'), *extra])
    if command == 'secondary':
        return _run([py, str(SCRIPT_DIR / 'cal_solver.py'), *extra])
    if command == 'solplot':
        return _run([py, str(SCRIPT_DIR / 'gainPlots.py'), *extra])
    if command == 'bandpass':
        return _run_bandpass(extra)
    if command == 'derive':
        return _run_bandpass(['--phase=derive', *extra])
    if command == 'audit':
        return _run_bandpass(['--phase=audit', *extra])
    if command == 'full':
        return _run_bandpass(['--phase=derive-and-audit', *extra])

    print(f'pipeline_cli.py: unknown command: {command!r}', file=sys.stderr)
    print('Try: pipeline_cli.py --help', file=sys.stderr)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
