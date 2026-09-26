"""GMRT-specific antenna *selection*: resolving a bare "<code>" prefix (e.g.
"C00") to the one antenna it names, on top of the generic name-or-id
resolution in `visplot.antenna_selection`.

GMRT's "<code>:<station number>" naming convention already has one
GMRT-specific interpretation in this codebase (DUD-antenna matching, see
`instruments.gmrt.antenna_table`); this is the same convention applied to
CLI antenna selection instead -- a user picking an antenna by its arm/
station code may reasonably not know or care about the numeric suffix
after the colon.
"""

from __future__ import annotations

from data_io.antenna_table import Antenna
from visplot.antenna_selection import resolve_antenna_selection as _resolve_antenna_selection_generic


def _resolve_gmrt_prefix_to_full_name(term: str, antennas: list[Antenna]) -> str | None:
    """`term`'s one matching antenna's full name, if `term` is a GMRT
    "<code>" prefix (matched the same way `instruments.gmrt.antenna_table`
    matches a DUD name: `name == term` or `name.startswith(f"{term}:")`);
    `None` if it matches no antenna's prefix at all, so the caller can fall
    through to another interpretation (a full name, or a numeric id).
    Raises if it matches more than one antenna -- checked directly against
    this file's own antenna table, never assumed unique."""
    matches = [
        a.name for a in antennas
        if a.name.lower() == term.lower() or a.name.lower().startswith(f"{term.lower()}:")
    ]
    if len(matches) > 1:
        raise ValueError(f"antenna prefix {term!r} matches more than one antenna in this file: {matches}")
    return matches[0] if matches else None


def resolve_antenna_selection(spec: str, antennas: list[Antenna]) -> list[int]:
    """Like `visplot.antenna_selection.resolve_antenna_selection`, but a
    bare GMRT code prefix (e.g. "C00") also resolves to that antenna, not
    only its full name ("C00:01") or numeric station number. A prefix can
    never collide with a full name or a numeric id (a full name always has
    a colon-and-number suffix a bare prefix lacks, and GMRT codes always
    start with a letter), so there is no ordering ambiguity between the
    three interpretations."""
    rewritten_terms = []
    for term in (t.strip() for t in spec.split(",") if t.strip()):
        full_name = _resolve_gmrt_prefix_to_full_name(term, antennas)
        rewritten_terms.append(full_name if full_name is not None else term)
    return _resolve_antenna_selection_generic(",".join(rewritten_terms), antennas)
