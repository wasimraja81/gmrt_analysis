"""A file's integrations and scans, from its row index (T20, listObs), for
any random-groups UVFITS file: no visibility data read.

Neither archival 40_014 file has an NX (scan index) table, so scans are
derived as AIPS INDXR derives one: a new scan where the source changes,
after a gap longer than `gap_integrations` integration times (the
integration time being the median spacing of the timestamps; INDXR's
CPARM(1), whose 0 adapts to the data), and when a scan on one source would
run longer than `longest_s` (INDXR's CPARM(2), 0 meaning 60 min). INDXR's
other rule, a change of frequency setup, is not applied: the row index does
not keep each row's FREQSEL.

Times are as recorded, in the file's time system (TIMSYS), as Julian dates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_GAP_INTEGRATIONS = 3.0
DEFAULT_LONGEST_SCAN_S = 3600.0


@dataclass(frozen=True)
class Integrations:
    """A selection's integrations, in time order: each one's timestamp
    (recorded JD), source and selected rows, and its place among the file's
    integrations (`integration`; None when not given)."""

    jd: np.ndarray
    source_id: np.ndarray
    n_rows: np.ndarray
    integration: np.ndarray | None = None

    @property
    def integration_s(self) -> float | None:
        """The median spacing of the timestamps, in seconds; None for one."""
        stamps = np.unique(self.jd)
        return float(np.median(np.diff(stamps)) * 86400.0) if stamps.size > 1 else None


@dataclass(frozen=True)
class Scan:
    number: int  # from 1, in time order
    source_id: int
    start_jd: float  # the first integration's timestamp (recorded)
    end_jd: float  # the last integration's timestamp
    n_integrations: int
    n_rows: int
    length_s: float  # the integrations times the integration time


def integrations_of(index, row_indices=None) -> Integrations:
    """The integrations of the rows `row_indices` (sorted; None: every row)
    of a `RowIndex`, each with how many of them it holds."""
    bounds = np.asarray(index.integration_boundaries)
    starts = bounds[:-1]
    if row_indices is None:
        present, n_rows = np.arange(len(starts)), np.diff(bounds)
    else:
        which = np.searchsorted(bounds, np.asarray(row_indices), side="right") - 1
        present, n_rows = np.unique(which, return_counts=True)
    first_rows = starts[present]
    return Integrations(np.asarray(index.jd)[first_rows], np.asarray(index.source_id)[first_rows], np.asarray(n_rows),
                        np.asarray(present))


def derive_scans(integrations: Integrations, gap_integrations: float = DEFAULT_GAP_INTEGRATIONS,
                 longest_s: float = DEFAULT_LONGEST_SCAN_S) -> list[Scan]:
    """The scans of `integrations`, by INDXR's rules (the module's note)."""
    jd, source, rows = integrations.jd, integrations.source_id, integrations.n_rows
    if not jd.size:
        return []
    integration_s = integrations.integration_s or 0.0
    step_s = np.diff(jd) * 86400.0
    new = np.concatenate([[True], (source[1:] != source[:-1]) | (step_s > gap_integrations * integration_s)])
    edges = list(np.flatnonzero(new)) + [jd.size]
    scans = []
    for start, stop in zip(edges[:-1], edges[1:]):
        piece = start
        for k in range(start + 1, stop + 1):  # a scan longer than `longest_s` starts a new one
            if k == stop or (jd[k] - jd[piece]) * 86400.0 + integration_s > longest_s:
                scans.append(Scan(len(scans) + 1, int(source[piece]), float(jd[piece]), float(jd[k - 1]),
                                  int(k - piece), int(rows[piece:k].sum()), float((k - piece) * integration_s)))
                piece = k
    return scans


def scan_of_integration(integrations: Integrations, scans: list[Scan], n_integrations: int) -> np.ndarray:
    """For each of the file's `n_integrations` integrations, the number of
    the scan holding it (0: none of `scans`): the scans cover the
    selection's integrations in order."""
    numbers = np.zeros(n_integrations, dtype=np.int64)
    numbers[integrations.integration] = np.repeat([s.number for s in scans], [s.n_integrations for s in scans])
    return numbers


def rows_per_antenna(index, row_indices=None) -> dict[int, int]:
    """How many of the rows hold each antenna (as either end; an
    autocorrelation counted once)."""
    rows = slice(None) if row_indices is None else np.asarray(row_indices)
    ant1, ant2 = np.asarray(index.ant1)[rows], np.asarray(index.ant2)[rows]
    counts = np.bincount(ant1, minlength=1).astype(np.int64)
    others = np.bincount(ant2[ant2 != ant1], minlength=counts.size)
    if others.size > counts.size:
        counts = np.pad(counts, (0, others.size - counts.size))
    counts[:others.size] += others
    return {int(a): int(n) for a, n in enumerate(counts) if n}
