"""Provenance records for visplot runs (T31), on the pipeline's own
`provenance.manifest.RunManifest`, under a request's --provenance-dir.

- `PlotRecord`: one per command-line run, GUI Plot, and file a plot window
  saves (Locate's "Save all as CSV", Export). Stage "visplot": the request
  (every option, defaults included, paths absolute) as parameters, the
  command line that reproduces it, the GUI session it came from, package
  versions, host, time, git commit and branch (with the uncommitted diff
  beside it), the FITS file and its row index (path, size, modification
  time), the outputs, and how it ended.
- `SessionRecord`: one per GUI session (stage "visplot_session"): its log
  holds every message the window showed, among them each record's run id
  and command.
- `WindowProvenance`: what a plot window records its saved files with, the
  same for a window the command line opened and a GUI tab.

Each record's run id (UTC time and 8 random hex digits) names its JSON
(`provenance/<stage>/<run id>.json`) and its log (`logs/<stage>/<run
id>.log`); `runs_index.jsonl` has a line per record.
"""

from __future__ import annotations

import importlib.metadata
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

from data_io.row_index import default_row_index_path
from provenance.logging_setup import stage_log_path
from provenance.manifest import RunManifest
from visplot.request import PlotRequest
from visplot.run import RequestError, Stopped

PLOT_STAGE = "visplot"
SESSION_STAGE = "visplot_session"
PACKAGES = ("numpy", "scipy", "astropy", "pyerfa", "matplotlib", "PySide6_Essentials")
_NO_CONSOLE = logging.CRITICAL + 10  # records log to their files; the caller shows messages itself


def package_versions() -> dict[str, str]:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


class PlotRecord:
    """The provenance record of one visplot run of `request`. `action` says
    what made it: "save", "locate", "locate, save" or "window" from the
    command line, "plot" from the GUI, "locate" or "export" from a plot
    window. Written at once as started; `finish` completes it."""

    def __init__(self, request: PlotRequest, action: str, session_id: str | None = None,
                 details: dict | None = None):
        fits_path = Path(request.fits_path)
        inputs = [p for p in (fits_path, default_row_index_path(fits_path)) if p.exists()]
        self.request = request
        self.command = request.command_line()
        self.manifest = RunManifest(
            stage=PLOT_STAGE, work_dir=Path(request.provenance_dir), parameters=request.as_dict(), inputs=inputs,
            console_log_level=_NO_CONSOLE,
            details={"action": action, "command": self.command, "session": session_id,
                     "packages": package_versions(), **(details or {})},
        )
        self.manifest.__enter__()
        self.finished = False
        self.log("info", f"{action}: {self.command}")

    @property
    def run_id(self) -> str:
        return self.manifest.run_id

    @property
    def path(self) -> Path:
        return self.manifest.manifest_path

    def describe(self) -> str:
        """The record's run id and file, e.g. for a CSV header."""
        return f"run {self.run_id}, {self.path}"

    def log(self, level: str, text: str) -> None:
        """A message into the record's log (dropped once it is finished)."""
        if not self.finished:
            (self.manifest.logger.warning if level == "warning" else self.manifest.logger.info)(text)

    def reporting(self, report: Callable[[str, str], None]) -> Callable[[str, str], None]:
        """A run's report callback: `report`, and into this record's log."""
        def both(level: str, text: str) -> None:
            report(level, text)
            self.log(level, text)
        return both

    def add_output(self, path) -> None:
        self.manifest.add_output(Path(path).resolve())

    def finish(self, error: str | None = None) -> None:
        """Complete the record: successful, or failed with `error`."""
        if self.finished:
            return
        self.finished = True
        if error is None:
            self.manifest.__exit__(None, None, None)
        else:
            self.manifest.__exit__(RuntimeError, RuntimeError(error), None)


@contextmanager
def recorded_run(request: PlotRequest, action: str, session_id: str | None = None) -> Iterator[PlotRecord]:
    """A `PlotRecord` around a run: complete when the block ends, failed
    with the reason when it raises (the exception goes on). A record that
    cannot be written raises RequestError before anything runs."""
    try:
        record = PlotRecord(request, action, session_id=session_id)
    except Exception as err:
        raise RequestError(f"the provenance record could not be written in {request.provenance_dir} "
                           f"({type(err).__name__}: {err}); nothing was run") from err
    try:
        yield record
    except BaseException as err:  # KeyboardInterrupt included: the record says how the run ended
        record.finish(str(err) if isinstance(err, (RequestError, Stopped)) else f"{type(err).__name__}: {err}")
        raise
    record.finish()


class SessionRecord:
    """The provenance record of one GUI session; its log holds every message
    the window showed."""

    def __init__(self, provenance_dir):
        work_dir = Path(provenance_dir).expanduser().resolve()
        self.manifest = RunManifest(
            stage=SESSION_STAGE, work_dir=work_dir, parameters={"provenance_dir": str(work_dir)}, inputs=[],
            console_log_level=_NO_CONSOLE, details={"packages": package_versions()},
        )
        self.manifest.__enter__()
        self.finished = False

    @property
    def session_id(self) -> str:
        return self.manifest.run_id

    @property
    def log_path(self) -> Path:
        return stage_log_path(self.manifest.work_dir, SESSION_STAGE, self.session_id)

    def log(self, level: str, text: str) -> None:
        """A message into the session's log (dropped once it is closed)."""
        if not self.finished:
            (self.manifest.logger.warning if level == "warning" else self.manifest.logger.info)(text)

    def close(self) -> None:
        if not self.finished:
            self.finished = True
            self.manifest.__exit__(None, None, None)


def action_request(request: PlotRequest, n_streamed: int, kind: str, view=None, box=None, csv_path=None,
                   dpi: int | None = None, figure_size_in=None, equal: bool | None = None) -> tuple[PlotRequest, bool]:
    """The request that reproduces a window action on `request`'s plot, and
    whether it reproduces it in full: a locate adds its `box` ((x0, x1),
    (y0, y1)) and CSV path; an export fixes the axis ranges to the exported
    `view` ((x limits), (y limits)), --dpi and --figure-size to the export's
    (the window's figure, in inches), and --aspect when the window's Equal
    aspect box overrode the request's (`equal`; None: not overridden) -- run
    with --output-dir, it saves the export's image as its PNG. A request with
    several streamed plots (`n_streamed` > 1) cannot be narrowed to one
    plot's action, so it comes back unchanged."""
    if n_streamed != 1:
        return request, False

    def span(lo, hi) -> str:
        lo, hi = sorted((float(lo), float(hi)))
        return f"{lo!r}:{hi!r}"

    if kind == "locate":
        (x0, x1), (y0, y1) = box
        return request.replace(locate=f"{span(x0, x1)},{span(y0, y1)}", locate_csv=str(csv_path)), True
    if kind == "export":
        (x0, x1), (y0, y1) = view
        width, height = (float(v) for v in figure_size_in)
        changes = dict(x_range=span(x0, x1), y_range=span(y0, y1), dpi=int(dpi), figure_size=f"{width!r},{height!r}")
        if equal is not None:
            changes["aspect"] = "equal" if equal else "free"
        return request.replace(**changes), True
    raise ValueError(f"unknown window action {kind!r}")


def _pairs(limits) -> list[list[float]]:
    return [[float(v) for v in pair] for pair in limits]


@dataclass(frozen=True)
class WindowProvenance:
    """What a plot window records the files it saves with: the request its
    plots came from (`n_streamed` of them streamed), the record of the run
    that opened it, and the GUI session, if any."""

    request: PlotRequest
    n_streamed: int
    parent_run_id: str | None = None
    session_id: str | None = None

    def start(self, kind: str, plot_name: str, path, box=None, view=None, dpi: int | None = None,
              figure_size_in=None, equal: bool | None = None) -> PlotRecord:
        """The record of a save about to be written to `path` from the plot
        `plot_name`: "locate" (the samples in `box`) or "export" (`view` at
        `dpi`, the window's figure being `figure_size_in`, `equal` the
        window's aspect override). The caller adds the output and finishes
        it. The details repeat the action's values, for a request that
        cannot be narrowed to it."""
        narrowed, full = action_request(self.request, self.n_streamed, kind, view=view, box=box, csv_path=path,
                                        dpi=dpi, figure_size_in=figure_size_in, equal=equal)
        details = {"from_run": self.parent_run_id, "plot": plot_name, "reproduces_in_full": full}
        if kind == "locate":
            details["box"] = _pairs(box)
        else:
            details.update(view=_pairs(view), dpi=dpi, figure_size_in=[float(v) for v in figure_size_in],
                           equal_aspect_override=equal)
        return PlotRecord(narrowed, kind, session_id=self.session_id, details=details)
