"""Pages (T26): one plot per baseline, antenna, source or Stokes product of
the selection, as CASA plotms's `iteraxis` and AIPS VPLOT's one baseline per
plot. A page holds the selection's rows, or its Stokes products, that pass
one more filter; pages exist for the values the selection holds, in antenna,
source and Stokes order."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

import numpy as np

PAGE_KINDS = ("baseline", "antenna", "source", "stokes")


@dataclass(frozen=True)
class Page:
    by: str  # one of PAGE_KINDS
    value: object  # (ant1, ant2), an antenna number, a source id, or a Stokes label
    label: str  # as the title and panel name it, e.g. "C00:01-C01:02"

    @property
    def text(self) -> str:
        """What the page holds, e.g. "baseline C00:01-C01:02", "Stokes RR"."""
        return f"{'Stokes' if self.by == 'stokes' else self.by} {self.label}"

    @property
    def file_label(self) -> str:
        """`label` in a filename: characters other than letters, digits, '.',
        '+' and '-' become '_' (C00:01-C01:02: C00_01-C01_02)."""
        return re.sub(r"[^A-Za-z0-9.+-]", "_", self.label)

    def row_mask(self, ant1, ant2, source_id) -> np.ndarray | None:
        """Which of these rows the page holds; None for a Stokes page, which
        holds every row."""
        if self.by == "baseline":
            a, b = self.value
            return (np.asarray(ant1) == a) & (np.asarray(ant2) == b)
        if self.by == "antenna":
            return (np.asarray(ant1) == self.value) | (np.asarray(ant2) == self.value)
        if self.by == "source":
            return np.asarray(source_id) == self.value
        return None

    def context(self, ctx):
        """The quantity context of the page's samples: a Stokes page's holds
        its one Stokes product, as a selection of it alone would."""
        return replace(ctx, stokes_labels=(self.value,)) if self.by == "stokes" else ctx


def list_pages(by: str, index, row_indices, stokes_labels, ctx) -> list[Page]:
    """The pages of a selection (`row_indices` of `index`, its Stokes
    `stokes_labels`), one per value of `by` it holds, named as `ctx` names
    antennas and sources."""
    rows = np.asarray(row_indices)
    antenna_names = ctx.antenna_names or {}
    source_names = ctx.source_names or {}

    def antenna(number) -> str:
        return str(antenna_names.get(int(number), int(number)))

    if by == "baseline":
        ant1, ant2 = np.asarray(index.ant1)[rows], np.asarray(index.ant2)[rows]
        pairs = np.unique(ant1.astype(np.int64) * 65536 + ant2)  # one integer per pair, as `panel_facts`
        return [Page(by, (int(p // 65536), int(p % 65536)), f"{antenna(p // 65536)}-{antenna(p % 65536)}")
                for p in pairs]
    if by == "antenna":
        numbers = np.union1d(np.asarray(index.ant1)[rows], np.asarray(index.ant2)[rows])
        return [Page(by, int(a), antenna(a)) for a in numbers]
    if by == "source":
        ids = np.unique(np.asarray(index.source_id)[rows])
        return [Page(by, int(s), str(source_names.get(int(s), int(s)))) for s in ids]
    if by == "stokes":
        return [Page(by, label, label) for label in stokes_labels]
    raise ValueError(f"pages are by one of {', '.join(PAGE_KINDS)}; got {by!r}")


def page_source(source, page: Page):
    """`source` (an `XYSource`) narrowed to `page`: its rows on the page, or
    its one Stokes product, with the page's quantity context."""
    if page.by == "stokes":
        pixel = list(source.index.stokes_labels).index(page.value)
        axis_selection = {**(source.axis_selection or {}), "STOKES": np.array([pixel])}
        return replace(source, axis_selection=axis_selection, ctx=page.context(source.ctx))
    rows = source.row_indices
    index = source.index
    mask = page.row_mask(np.asarray(index.ant1)[rows], np.asarray(index.ant2)[rows], np.asarray(index.source_id)[rows])
    return source.subset(rows[mask])
