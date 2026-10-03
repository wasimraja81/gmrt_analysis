"""The listing's Flags section (T20): how many of a selection's
visibilities are flagged, by Stokes, source, antenna and integration (the
listing sums integrations into its scans), from one pass over the
visibility data as a plot reads it. A visibility is one (channel, Stokes)
element of a row, flagged when its weight is not positive (the AIPS
convention the plots follow)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FlagCounts:
    """[flagged, all] visibilities, keyed by Stokes label, source id and
    antenna (station number; a baseline's visibilities count for both its
    antennas, an autocorrelation's once); and per integration of the file
    (`integration_flagged`, `integration_total`)."""

    n_integrations: int
    flagged: int = 0
    total: int = 0
    by_stokes: dict[str, list[int]] = field(default_factory=dict)
    by_source: dict[int, list[int]] = field(default_factory=dict)
    by_antenna: dict[int, list[int]] = field(default_factory=dict)
    integration_flagged: np.ndarray = field(init=False)
    integration_total: np.ndarray = field(init=False)

    def __post_init__(self):
        self.integration_flagged = np.zeros(self.n_integrations, dtype=np.int64)
        self.integration_total = np.zeros(self.n_integrations, dtype=np.int64)


class FlagCounter:
    """A stream reducer (`visplot.stream.run_stream`) filling `counts`;
    `boundaries`: the row index's integration boundaries."""

    def __init__(self, boundaries: np.ndarray):
        self.boundaries = np.asarray(boundaries)
        self.counts = FlagCounts(len(self.boundaries) - 1)

    def compute(self, values):
        """One block's flagged visibilities per row and Stokes (on a worker thread)."""
        block = values.block
        if block.weight is None or not len(block.row_indices):
            return None
        flagged = block.weight <= 0
        stokes_axis = block.axis_types.index("STOKES") + 1 if "STOKES" in block.axis_types else None
        others = tuple(i for i in range(1, flagged.ndim) if i != stokes_axis)
        per_stokes = flagged.sum(axis=others, dtype=np.int64)
        if stokes_axis is None:
            per_stokes = per_stokes[:, None]
        per_row_per_stokes = int(np.prod([flagged.shape[i] for i in others]))
        integration = np.searchsorted(self.boundaries, block.row_indices, side="right") - 1
        return (per_stokes, per_row_per_stokes, list(block.stokes_labels or ["all"]), np.asarray(block.ant1),
                np.asarray(block.ant2), np.asarray(block.source_id), integration)

    def apply(self, result) -> None:
        if result is None:
            return
        per_stokes, per_row_per_stokes, labels, ant1, ant2, source, integration = result
        counts = self.counts
        n_rows = per_stokes.shape[0]
        row_flagged = per_stokes.sum(axis=1)
        per_row = per_stokes.shape[1] * per_row_per_stokes
        counts.flagged += int(row_flagged.sum())
        counts.total += n_rows * per_row
        for k, label in enumerate(labels):
            entry = counts.by_stokes.setdefault(label, [0, 0])
            entry[0] += int(per_stokes[:, k].sum())
            entry[1] += n_rows * per_row_per_stokes
        _add(counts.by_source, source, row_flagged, per_row)
        _add(counts.by_antenna, ant1, row_flagged, per_row)
        other_end = ant2 != ant1
        _add(counts.by_antenna, ant2[other_end], row_flagged[other_end], per_row)
        np.add.at(counts.integration_flagged, integration, row_flagged)
        np.add.at(counts.integration_total, integration, per_row)


def _add(table: dict, keys: np.ndarray, row_flagged: np.ndarray, per_row: int) -> None:
    """Each row's flagged visibilities, and its `per_row` visibilities, to its key's entry."""
    if not keys.size:
        return
    unique, inverse = np.unique(keys, return_inverse=True)
    flagged = np.bincount(inverse, weights=row_flagged)
    rows = np.bincount(inverse)
    for key, n_flagged, n_rows in zip(unique.tolist(), flagged.tolist(), rows.tolist()):
        entry = table.setdefault(int(key), [0, 0])
        entry[0] += int(round(n_flagged))
        entry[1] += n_rows * per_row
