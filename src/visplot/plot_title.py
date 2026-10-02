"""One consistent title format, shared by every `visplot` function --
which source(s) a plot shows, and (when given) which telescope and file it
came from. A plot with no title context beyond its own bare name is easy
to lose track of once several are open or saved side by side; this exists
so no plot function has to remember to compose that by hand.
"""

from __future__ import annotations

from pathlib import Path


def build_plot_title(
    base_title: str,
    sources: list[str] | None = None,
    telescope: str | None = None,
    source_path: str | Path | None = None,
    part: str | None = None,
) -> str:
    """`base_title`, plus the part of the selection it shows (`part`, e.g.
    "baseline C00:01-C01:02" for one plot per baseline, T26), the source it
    shows (one source) or "multiple sources" (the panel under the plot names
    them; the user, 2026-10-01), plus a second line naming the telescope
    and/or source file (if given)."""
    title = base_title
    shows = [part] if part else []
    if sources:
        shows.append(sources[0] if len(sources) == 1 else "multiple sources")
    if shows:
        title += ": " + ", ".join(shows)

    provenance = []
    if telescope:
        provenance.append(telescope)
    if source_path is not None:
        provenance.append(f"file: {Path(source_path).name}")
    if provenance:
        title += "\n(" + ", ".join(provenance) + ")"
    return title
