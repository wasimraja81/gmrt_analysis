#!/usr/bin/env python3
"""Shared workflow helpers for config loading and CLI overrides."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def load_config_into_globals(config_path: str, target_globals: dict) -> None:
    """Exec config file and inject every public name into target globals."""
    path = Path(config_path).resolve()
    if not path.exists():
        sys.exit(f'ERROR: config file not found: {path}')
    ns: dict = {}
    with open(path) as fh:
        exec(compile(fh.read(), str(path), 'exec'), ns)  # noqa: S102
    for key, val in ns.items():
        if not key.startswith('_'):
            target_globals[key] = val


def apply_overrides_to_globals(overrides: list, target_globals: dict, log=None) -> None:
    """Apply ``--set KEY=expr`` overrides into target globals."""
    if not overrides:
        return
    from pathlib import Path as _Path

    for item in overrides:
        if '=' not in item:
            sys.exit(f'ERROR: --set requires KEY=VALUE format, got: {item!r}')
        key, _, expr = item.partition('=')
        key = key.strip()
        if not key.isidentifier():
            sys.exit(f'ERROR: --set key is not a valid Python identifier: {key!r}')
        try:
            # Evaluate override expressions in a namespace that includes the
            # currently loaded config globals so references like WORK_DIR,
            # DATA_DIR, etc. resolve correctly.
            eval_globals = dict(target_globals)
            eval_globals.update({'Path': _Path, '__builtins__': __builtins__})
            val = eval(expr.strip(), eval_globals, {})  # noqa: S307
        except Exception as exc:
            sys.exit(f'ERROR: could not evaluate --set {key}={expr!r}: {exc}')
        target_globals[key] = val
        if log is not None:
            log.debug('  --set %s = %r', key, val)


def derive_index_cache(index_cache, cal_fits, work_dir) -> Path:
    """Return index-cache path as ``<WORK_DIR>/<cal_fits_stem>.index.npz``."""
    if index_cache is not None:
        return Path(index_cache)
    stem = Path(cal_fits).stem
    base = Path(work_dir) if work_dir is not None else Path(cal_fits).parent
    return base / f'{stem}.index.npz'


def bootstrap_config_from_cli(
    default_config: str,
    target_globals: dict,
    log=None,
):
    """Pre-parse ``--config`` and repeated ``--set``, then apply both.

    Returns the namespace from ``parse_known_args`` with attributes:
    ``config`` and ``set_overrides``.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', default=default_config)
    pre.add_argument('--set', dest='set_overrides', action='append', default=[])
    pre_args, _ = pre.parse_known_args()
    load_config_into_globals(pre_args.config, target_globals)
    apply_overrides_to_globals(pre_args.set_overrides, target_globals, log=log)
    return pre_args


def add_common_config_set_arguments(
    parser: argparse.ArgumentParser,
    *,
    config_default,
    config_help: str,
    set_help: str,
    config_metavar: str = 'FILE',
    set_metavar: str = 'KEY=expr',
) -> None:
    """Add common ``--config`` and ``--set`` arguments to a parser."""
    parser.add_argument(
        '--config',
        default=config_default,
        metavar=config_metavar,
        help=config_help,
    )
    parser.add_argument(
        '--set',
        action='append',
        dest='set_overrides',
        metavar=set_metavar,
        default=[],
        help=set_help,
    )


def add_log_level_argument(
    parser: argparse.ArgumentParser,
    *,
    default=None,
    help_text: str = 'Logging level: DEBUG | INFO | WARNING | ERROR',
    metavar: str = 'LEVEL',
) -> None:
    """Add common ``--log-level`` argument to a parser."""
    parser.add_argument(
        '--log-level',
        default=default,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        metavar=metavar,
        help=help_text,
    )


def resolve_versioned_path(
    *,
    candidates: list,
    version: str,
    default_path=None,
    label: str = 'table',
) -> Path:
    """Resolve a version selector to a concrete file path.

    version semantics:
      - ``latest``: default_path if it exists, else newest candidate by mtime
      - integer ``N``: 1-based index into name-sorted candidates
      - existing path string: use directly
      - otherwise: substring match (case-insensitive) against filename
    """
    cand = [Path(p) for p in candidates if p is not None]
    uniq: list[Path] = []
    seen = set()
    for p in cand:
        rp = p.resolve()
        if rp.exists() and str(rp) not in seen:
            uniq.append(rp)
            seen.add(str(rp))

    sel = (version or 'latest').strip()
    if sel.lower() == 'latest':
        if default_path is not None and Path(default_path).exists():
            return Path(default_path).resolve()
        if not uniq:
            raise ValueError(f'No available {label} candidates to resolve latest.')
        return max(uniq, key=lambda p: p.stat().st_mtime)

    psel = Path(sel)
    if psel.exists():
        return psel.resolve()

    if sel.isdigit():
        idx = int(sel)
        ordered = sorted(uniq, key=lambda p: p.name)
        if idx < 1 or idx > len(ordered):
            raise ValueError(
                f'{label} version {idx} out of range: 1..{len(ordered)} '
                f'(available: {[p.name for p in ordered]})'
            )
        return ordered[idx - 1]

    matched = [p for p in uniq if sel.lower() in p.name.lower()]
    if matched:
        return max(matched, key=lambda p: p.stat().st_mtime)

    raise ValueError(
        f'Could not resolve {label} version {version!r}. '
        f'Available: {[p.name for p in sorted(uniq, key=lambda p: p.name)]}'
    )


def format_table_application_note(
    *,
    docal: str,
    doflag: str,
    calver: str = 'latest',
    flagver: str = 'latest',
    cal_path=None,
    flag_paths=None,
) -> str:
    """Return a compact title suffix describing applied cal/flag tables."""
    _docal = str(docal).lower()
    _doflag = str(doflag).lower()

    if _docal == 'on':
        _cal_name = Path(cal_path).name if cal_path is not None else f'ver={calver}'
    else:
        _cal_name = 'none'

    _fps = [Path(p).name for p in (flag_paths or [])]
    if _doflag == 'on':
        if not _fps:
            _flag_name = 'none'
        elif len(_fps) <= 2:
            _flag_name = ','.join(_fps)
        else:
            _flag_name = f'{_fps[0]},{_fps[1]},+{len(_fps)-2}'
    else:
        _flag_name = 'none'

    return f' [DOCAL={_docal}:{_cal_name} | DOFLAG={_doflag}:{_flag_name}]'
