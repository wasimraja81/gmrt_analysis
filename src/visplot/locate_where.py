"""Locate by condition (`--locate` with no box, and `--where`; T26's flagging
design, step 2b): every visibility of the selection meeting all the run's
conditions, written to a run of a Locate session (`data_io.found_file`).

A condition on a data value (amp, phase, real, imag of a product) finds; the
conditions on coordinates (el, az, ha, pa, time, freq, channel, uu, vv, ww,
uvdist) narrow where it looks, so a run needs at least one on a data value.
The products read and recorded are those the data conditions use: `amp(V)`
reads RR and LL and records both at each match. `--flagged` sets which
visibilities the file already flags are candidates: `exclude` (default,
searches only unflagged ones), `only` (searches only the already-flagged
ones -- a diagnosis of what is already flagged and why), or `include` (both).
A row is found whole only when every visibility in that run's own domain --
its searched channels, narrowed to whichever of `--flagged`'s candidates --
is found; this is what makes a whole-row record unambiguous: nothing about
why some other visibility was left out needs to be guessed or stored, since
there is no "left out but counted anyway" case any more.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from data_io.found_file import LocateSession
from data_io.row_index import STOKES_LABELS_BY_CODE
from data_io.stokes_products import ProductPlan, apply_products, plan_products
from visplot.conditions import Condition, ResolvedConditions, parse_condition
from visplot.quantities import evaluate
from visplot.request import LOCATE_BY_CONDITIONS

SESSION_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


def locates_by_conditions(request) -> bool:
    """Whether `request` is a Locate by its --where conditions (`--locate` given no box)."""
    return request.locate == LOCATE_BY_CONDITIONS


# Options naming a run operation on a session (each a run number, or None: not given), option -> request.dest.
# More are added here as they are built (next: "--replace"; then "--intersect", "--difference").
RUN_OPERATIONS = {"--undo": "undo", "--restore": "restore"}


def _given_run_operations(request) -> list[str]:
    """Options among `RUN_OPERATIONS` that `request` gives, in their keys' order."""
    return [option for option, dest in RUN_OPERATIONS.items() if getattr(request, dest) is not None]


def run_operation(request) -> str | None:
    """Which run operation `request` performs on a session ("undo",
    "restore", ...); the first found if more than one was (mistakenly)
    given, None if none was. Never raises -- safe to call before the file is
    checked; `check_locate_request` refuses more than one."""
    given = _given_run_operations(request)
    return RUN_OPERATIONS[given[0]] if given else None


def lists_session(request) -> bool:
    """Whether `request` is `--session NAME` alone: lists that session's runs
    (`data_io.found_file.LocateSession.runs`/`run_status`) instead of running
    anything. --session combined with --locate/--where/--plots/--listobs/a
    --flagged other than its default/a run operation (`run_operation`) is
    always some other action (a Locate run, a run operation, or refused by
    `check_locate_request`)."""
    return (request.locate_session is not None and request.locate is None and not request.plots
            and not request.listobs and request.flagged == "exclude" and run_operation(request) is None)


def check_locate_request(request) -> tuple[Condition, ...]:
    """The --where conditions of a Locate by conditions, parsed and checked
    with the options around them, without the file; () for a request that
    does not locate by conditions, including one that lists a session
    (`lists_session`) or performs a run operation (`run_operation`). Raises
    ValueError saying what to write."""
    if request.locate_session is not None and not SESSION_NAME.match(request.locate_session):
        raise ValueError(f"--session {request.locate_session!r}: a name of letters, digits, '.', '_' and '-'")
    given_operations = _given_run_operations(request)
    if len(given_operations) > 1:
        raise ValueError(f"give one of {', '.join(RUN_OPERATIONS)}; got {', '.join(given_operations)}")
    operation = run_operation(request)
    where = request.where or ()
    if locates_by_conditions(request):
        if operation is not None:
            raise ValueError(f"--locate with no box runs by its --where conditions; {given_operations[0]} "
                             f"{getattr(request, operation)}: give it in a separate run")
        if not where:
            raise ValueError("--locate with no box finds the visibilities meeting --where conditions: give at "
                             "least one --where")
        for option, given, why in (
            ("--plots", request.plots, "give the plots in a separate run"),
            ("--locate-csv", request.locate_csv, "that is the box's CSV; this run is written to its session "
                                                 "(--session)"),
            ("--stokes", request.stokes, "the --where conditions name the products read and recorded (amp(V) "
                                         "reads RR and LL)"),
        ):
            if given:
                raise ValueError(f"--locate with no box runs by its --where conditions; {option}: {why}")
        conditions = tuple(parse_condition(text) for text in where)
        if not any(condition.operand.product is not None for condition in conditions):
            raise ValueError("--where: a Locate run needs a condition on a data value, e.g. 'amp(V) > 5'; "
                             "conditions on coordinates only narrow where it looks; for those alone, the "
                             "selection options (--el-range, --time-range, --channels, ...)")
        return conditions
    if where:
        if request.locate:
            raise ValueError("--where goes with --locate given no box; a box with --where is not available yet")
        raise ValueError("--where gives the conditions of --locate: add --locate, with no value")
    if operation is not None:
        for option, given in (("--plots", request.plots), ("--listobs", request.listobs)):
            if given:
                raise ValueError(f"{given_operations[0]} {getattr(request, operation)} runs on a session by "
                                 f"itself; {option}: give it in a separate run")
        if request.flagged != "exclude":
            raise ValueError(f"--flagged goes with --locate (no box) and --where; {given_operations[0]} takes no "
                             "conditions to narrow")
        return ()
    if not lists_session(request):
        for option, given in (("--session", request.locate_session is not None),
                              ("--flagged", request.flagged != "exclude")):
            if given:
                raise ValueError(f"{option} goes with --locate (no box) and --where, or --session alone to "
                                 "list its runs")
    return ()


def locate_session_name(request) -> tuple[str, bool]:
    """The Locate session of `request` and whether --session gave it; by
    default the FITS file's name without its extension, any character a
    session name does not take made '_'."""
    if request.locate_session is not None:
        return request.locate_session, True
    return re.sub(r"[^A-Za-z0-9._-]", "_", Path(request.fits_path).stem) or "_", False


def _table(headers: list[str], rows: list[list[str]], right: set[int]) -> list[str]:
    """Fixed-width columns: those in `right` aligned right (`visplot.listing`'s own)."""
    widths = [max(len(str(c)) for c in column) for column in zip(headers, *rows)]

    def line(cells):
        return "  ".join(str(c).rjust(w) if i in right else str(c).ljust(w)
                         for i, (c, w) in enumerate(zip(cells, widths))).rstrip()
    return [line(headers)] + [line(r) for r in rows]


def session_listing_text(session: LocateSession) -> str:
    """A session's runs, one line each: its number, status (active or
    aside), conditions, what it found, and when it was written. Logged-late
    and set-aside-unlogged runs are noted first; an empty session says so."""
    lines = [f"Locate session {session.name}: {session.identity.name}, {session.identity.gcount:,} rows"]
    for path in session.removed_partials:
        lines.append(f"  removed {path.name}, left by a run that stopped before its end")
    for number in session.logged_late:
        lines.append(f"  run {number}: logged late (its process stopped between writing the run and logging it)")
    for path, problem in session.set_aside:
        lines.append(f"  {path.name}: not whole, set aside, never logged ({problem})")
    runs = session.runs()
    if not runs:
        lines.append("  no runs yet")
        return "\n".join(lines) + "\n"
    status = session.status()
    rows = [[str(entry["run"]), status.get(entry["run"], "active"), " and ".join(entry["conditions"]),
            f"{entry['cells_found']:,}", f"{entry['rows_whole']:,}", entry["written_utc"]]
           for entry in sorted(runs, key=lambda e: e["run"])]
    headers = ["run", "status", "conditions", "cells found", "rows whole", "written (UTC)"]
    lines += ["  " + line for line in _table(headers, rows, right={0, 3, 4})]
    active = session.active_runs()
    if active:
        lines.append(f"  found set: the union of run{'' if len(active) == 1 else 's'} "
                     f"{', '.join(str(n) for n in active)}")
    else:
        lines.append("  found set: empty, no run active")
    return "\n".join(lines) + "\n"


STOKES_CODES = {label: code for code, label in STOKES_LABELS_BY_CODE.items()}
# Conditions narrowing the channels searched; the other coordinates narrow the rows.
CHANNEL_COORDINATES = ("channel", "freq")


@dataclass(frozen=True)
class LocatePlan:
    """What a Locate by conditions reads and records: its resolved conditions
    on data values (each on product `label_of[i]` of `products`) and on
    coordinates, and the file's products it reads and records (`recorded`,
    in the file's order, with their AIPS Stokes `codes`)."""

    data_conditions: tuple[Condition, ...]
    coordinate_conditions: tuple[Condition, ...]
    products: ProductPlan
    label_of: tuple[int, ...]
    recorded: tuple[str, ...]
    codes: tuple[int, ...]


def plan_locate(resolved: ResolvedConditions, stokes_labels) -> LocatePlan:
    data = tuple(c for c in resolved.conditions if c.operand.product is not None)
    coordinates = tuple(c for c in resolved.conditions if c.operand.product is None)
    products = plan_products(list(dict.fromkeys(c.operand.product for c in data)), list(stokes_labels))
    recorded = tuple(stokes_labels[i] for i in products.read)
    return LocatePlan(data, coordinates, products, tuple(products.labels.index(c.operand.product) for c in data),
                      recorded, tuple(STOKES_CODES[label] for label in recorded))


def check_cell_axes(axis_types, axis_lengths) -> None:
    """A found record names a row, a channel and a product: refused for data
    with another axis longer than one (several IFs)."""
    longer = [f"{t} ({n})" for t, n in zip(axis_types, axis_lengths)
              if t not in ("COMPLEX", "STOKES", "FREQ") and n > 1]
    if longer:
        raise ValueError(f"Locate records a row, a channel and a product per visibility; this file also has "
                         f"{', '.join(longer)}")


def holds(values: np.ndarray, condition: Condition) -> np.ndarray:
    """Where `values` (in the condition's resolved unit) meet `condition`; a
    value that cannot be computed (NaN) never does."""
    keep = np.ones(np.shape(values), dtype=bool)
    lower, upper = condition.lower, condition.upper
    if lower is not None:
        keep &= values >= lower.value if lower.inclusive else values > lower.value
    if upper is not None:
        keep &= values <= upper.value if upper.inclusive else values < upper.value
    return keep


def _overlaps(lo: np.ndarray, hi: np.ndarray, condition: Condition) -> np.ndarray:
    """Where a range [lo, hi] reaches into `condition`'s interval."""
    keep = np.isfinite(lo) & np.isfinite(hi)
    lower, upper = condition.lower, condition.upper
    if lower is not None:
        keep &= hi >= lower.value if lower.inclusive else hi > lower.value
    if upper is not None:
        keep &= lo <= upper.value if upper.inclusive else lo < upper.value
    return keep


def _cells(array, block, stokes_index: int = 0) -> np.ndarray:
    """`array` (dimensioned as the block's data) as (rows, channels), at
    product `stokes_index` where it varies by product."""
    n, n_channels = len(block.row_indices), len(block.axis_indices["FREQ"])
    dim = 1 + block.axis_types.index("STOKES")
    array = np.asarray(array)
    array = np.take(array, stokes_index if array.shape[dim] > 1 else 0, axis=dim)
    shape = [n] + [n_channels if t == "FREQ" else 1 for t in block.axis_types if t != "STOKES"]
    return np.broadcast_to(array, shape).reshape(n, n_channels)


def _by_product(array, block) -> np.ndarray:
    """The block's `array` (data or weight) as (rows, products, channels)."""
    n, n_channels = len(block.row_indices), len(block.axis_indices["FREQ"])
    dim = 1 + block.axis_types.index("STOKES")
    return np.moveaxis(array, dim, 1).reshape(n, array.shape[dim], n_channels)


def coordinate_values(condition: Condition, block, ctx) -> np.ndarray:
    """A coordinate condition's quantity on the block's cells, (rows,
    channels), in the condition's resolved unit (time: the recorded JD)."""
    n, n_channels = len(block.row_indices), len(block.axis_indices["FREQ"])
    name = condition.operand.name
    if name == "time":
        return np.broadcast_to(block.jd[:, None], (n, n_channels))
    if name == "channel":
        return np.broadcast_to(block.axis_indices["FREQ"][None, :], (n, n_channels))
    return _cells(evaluate(name, block, ctx, condition.unit), block)


class NarrowRowsReducer:
    """The rows a Locate can find anything in, by its conditions on row
    coordinates, from row metadata alone: fed the selection's two band-edge
    channels, a row is kept where its range over them reaches into every
    condition (a coordinate in wavelengths varies with frequency, and
    reaches its extremes at the band's edges)."""

    def __init__(self, conditions: tuple[Condition, ...], ctx):
        self.conditions = conditions
        self.ctx = ctx
        self.kept: list[np.ndarray] = []

    def compute(self, values) -> np.ndarray:
        block = values.block
        keep = np.ones(len(block.row_indices), dtype=bool)
        for condition in self.conditions:
            cells = coordinate_values(condition, block, self.ctx)
            with np.errstate(invalid="ignore"):
                keep &= _overlaps(cells.min(axis=1), cells.max(axis=1), condition)
        return block.row_indices[keep]

    def apply(self, result) -> None:
        self.kept.append(result)

    def update(self, values) -> None:
        self.apply(self.compute(values))


def found_fields(plan: LocatePlan, gcount: int, n_channels: int, max_station: int, max_source: int) -> dict:
    """The fields of a run's two tables (`data_io.found_file`), integer widths
    from the file: a record per visibility found ("cells"), and per product
    of each row found whole ("rows")."""
    row = "<u4" if gcount < 2**32 else "<u8"
    channel = "<u2" if n_channels <= 2**16 else "<u4"
    station = "<u2" if max_station < 2**16 else "<u4"
    source = "<u2" if max_source < 2**16 else "<u4"
    where = {"row": row, "product": "<i1", "jd": "<f8", "ant1": station, "ant2": station, "source": source,
             "uu": "<f4", "vv": "<f4", "ww": "<f4"}
    cells = {**where, "channel": channel, "value": "<c8", "weight": "<f4", "flagged": "<u1",
             **{f"cond{i}": "<f4" for i in range(len(plan.data_conditions))}}
    rows = {**where, "n_cells": "<u4",
            **{f"cond{i}_{end}": "<f4" for i in range(len(plan.data_conditions)) for end in ("min", "max")}}
    return {"cells": cells, "rows": rows}


class WhereReducer:
    """A Locate by conditions over the visibilities of each chunk (its
    STOKES axis the plan's `recorded` products): a cell is a candidate under
    `flagged_mode` ("exclude": unflagged only, the default; "only": flagged
    only; "include": both), and found among the candidates where every
    condition holds. A row of more than one searched channel whose every
    candidate cell is found is found whole: one record per product
    (`rows`); every other found cell, one record per product (`cells`).
    Because "whole" means every candidate was found -- never "found, or
    left out for some other reason" -- a whole-row record never hides which
    visibilities it covers: under `flagged_mode`, they are exactly the run's
    searched channels (recorded in its header) restricted to the
    candidates. Records go to `sink(table, columns)` in row order; counts
    kept."""

    def __init__(self, plan: LocatePlan, ctx, flagged_mode: str, sink):
        self.plan = plan
        self.ctx = ctx
        self.flagged_mode = flagged_mode
        self.sink = sink
        self.n_cells_found = 0
        self.n_rows_whole = 0

    def compute(self, values):
        block, plan = values.block, self.plan
        n, n_channels = len(block.row_indices), len(block.axis_indices["FREQ"])
        derived = apply_products(block, plan.products)
        flagged = np.zeros((n, n_channels), dtype=bool)
        for k in set(plan.label_of):
            flagged |= _cells(derived.weight, derived, k) <= 0
        if self.flagged_mode == "only":
            found = flagged.copy()
        elif self.flagged_mode == "include":
            found = np.ones((n, n_channels), dtype=bool)
        else:
            found = ~flagged
        values_of = []
        for condition, k in zip(plan.data_conditions, plan.label_of):
            cells = _cells(evaluate(condition.operand.name, derived, self.ctx, condition.unit), derived, k)
            values_of.append(cells)
            found &= holds(cells, condition)
        for condition in plan.coordinate_conditions:
            found &= holds(coordinate_values(condition, block, self.ctx), condition)
        whole = found.all(axis=1) if n_channels > 1 else np.zeros(n, dtype=bool)
        return self._cells(block, found & ~whole[:, None], values_of), self._rows(block, found, whole, values_of), \
            int(found.sum()), int(whole.sum())

    def _where(self, block, rows: np.ndarray, products: np.ndarray) -> dict:
        return {"row": block.row_indices[rows], "product": np.asarray(self.plan.codes)[products],
                "jd": block.jd[rows], "ant1": block.ant1[rows], "ant2": block.ant2[rows],
                "source": block.source_id[rows], "uu": block.uu_sec[rows], "vv": block.vv_sec[rows],
                "ww": block.ww_sec[rows]}

    def _cells(self, block, found: np.ndarray, values_of: list) -> dict:
        n_products = len(self.plan.recorded)
        rows, channels = np.nonzero(found)  # in row order, then channel
        rows_p, channels_p = np.repeat(rows, n_products), np.repeat(channels, n_products)
        products = np.tile(np.arange(n_products), len(rows))
        weight = _by_product(block.weight, block)[rows_p, products, channels_p]
        return {**self._where(block, rows_p, products), "channel": block.axis_indices["FREQ"][channels_p],
                "value": _by_product(block.data, block)[rows_p, products, channels_p], "weight": weight,
                "flagged": weight <= 0,
                **{f"cond{i}": cells[rows_p, channels_p] for i, cells in enumerate(values_of)}}

    def _rows(self, block, found: np.ndarray, whole: np.ndarray, values_of: list) -> dict:
        n_products = len(self.plan.recorded)
        rows = np.nonzero(whole)[0]
        rows_p, products = np.repeat(rows, n_products), np.tile(np.arange(n_products), len(rows))
        columns = {**self._where(block, rows_p, products), "n_cells": found[rows_p].sum(axis=1)}
        for i, cells in enumerate(values_of):
            masked = np.where(found[rows_p], cells[rows_p], np.nan)
            columns[f"cond{i}_min"], columns[f"cond{i}_max"] = np.nanmin(masked, axis=1), np.nanmax(masked, axis=1)
        return columns

    def apply(self, result) -> None:
        cells, rows, n_found, n_whole = result
        self.sink("cells", cells)
        self.sink("rows", rows)
        self.n_cells_found += n_found
        self.n_rows_whole += n_whole

    def update(self, values) -> None:
        self.apply(self.compute(values))
