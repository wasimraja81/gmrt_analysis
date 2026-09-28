"""Entry point for the visPlot exploration tool -- invoked by
`bin/visplot.sh`.

Usage: gmrt/bin/python3 src/cli/run_visplot.py FITS_PATH --plots PLOT[,PLOT...] [options]

With no arguments, or a FITS path alone, it opens the GUI (`visplot.gui`).
Otherwise the arguments become a `visplot.request.PlotRequest`, run by `visplot.run`,
the same code the GUI runs: this module only parses, prints what the run
reports, and chooses between saving (--output-dir), locating (--locate) and
the window (`visplot.qt_inspector`). See `bin/visplot.sh --help`.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

# The backend must be chosen before any submodule imports pyplot: Agg
# (headless, file-only) when only saving is requested, since no display
# may be available at all in that case; otherwise Qt, for the inspection window.
_HEADLESS_OPTIONS = ("--output-dir", "--clear-cache", "--locate")
matplotlib.use("Agg" if any(a.split("=")[0] in _HEADLESS_OPTIONS for a in sys.argv[1:]) else "QtAgg")

from data_io.astrometry import AstrometryWarning  # noqa: E402
from visplot.range_cache import clear_cache  # noqa: E402
from visplot.request import PlotRequest, build_arg_parser  # noqa: E402,F401  (build_arg_parser: for callers)
from visplot.run import MissingIndexError, RequestError, prepare, run_locate, save_outputs  # noqa: E402


def _print_report(level: str, text: str) -> None:
    """A run's messages: information on stdout, warnings on stderr."""
    if level == "warning":
        sys.stdout.flush()
        print(f"WARNING: {text}", file=sys.stderr)
    else:
        print(text, flush=True)


def _terminal_progress(progress):
    """An on_chunk callback printing `progress` (a PassProgress) at every 10%
    of the selection."""
    next_mark = [0.1]

    def on_chunk(rows_done: int) -> bool:
        total = progress.total_rows
        if total and rows_done / total >= next_mark[0]:
            print(progress.text(rows_done), flush=True)
            while rows_done / total >= next_mark[0]:
                next_mark[0] += 0.1
        return True

    return on_chunk


def main(argv: list[str]) -> int:
    # No arguments, or a FITS path alone: the GUI (on the same request and run code).
    if len(argv) == 1 or (len(argv) == 2 and not argv[1].startswith("-")):
        from visplot.gui.main_window import run_gui

        return run_gui(argv[1] if len(argv) == 2 else None)
    parser = build_arg_parser()
    if any(a == "--clear-cache" or a.startswith("--clear-cache=") for a in argv[1:]):
        pre = argparse.ArgumentParser(add_help=False)
        pre.add_argument("--clear-cache", required=True)
        known, _ = pre.parse_known_args(argv[1:])
        removed = clear_cache(known.clear_cache)
        for path in removed:
            print(f"removed {path}")
        print(f"removed {len(removed)} visplot cache file(s) from {known.clear_cache}")
        return 0
    request = PlotRequest.from_namespace(parser.parse_args(argv[1:]))
    warnings.simplefilter("ignore", AstrometryWarning)  # reported once, through the run's report

    try:
        run = prepare(request, report=_print_report)
    except MissingIndexError as err:
        print(err, file=sys.stderr)
        return 1
    except RequestError as err:
        parser.error(str(err))

    if run.locate_box is not None:
        run_locate(run, _print_report, _terminal_progress)
        if not request.output_dir:
            return 0
    if request.output_dir:
        save_outputs(run, _print_report, _terminal_progress)
    else:
        from visplot.qt_inspector import run_inspector

        run_inspector(run.source, run.figures(), run.labels, cache=run.cache, cached=run.cached)
    if run.cache is not None:
        n_files, n_bytes = run.cache.usage()
        size = f"{n_bytes / 1e6:.1f} MB" if n_bytes >= 1e6 else f"{n_bytes / 1e3:.0f} KB"
        print(f"cache: {len(run.cache.written)} file(s) written this run; {n_files} in {run.cache.directory} "
              f"({size}); remove them with: {parser.prog} --clear-cache {run.cache.directory}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
