"""The panel under a plot (T37): what the plot shows, stated outside the
plot area, as an AIPS TV display does -- the color key (the categories
--colorize-by names, and flagged samples), the Stokes a visibility plot
holds, the channels, baselines, time span and row filters of the selection,
what was drawn, how flags combine where a point stands for several samples,
and the provenance record of the run that made the figure.

`PanelFacts` is what a run selected (`panel_facts`, from `visplot.run`);
`PlotPanel` draws it, one line per anchored box, in the Helvetica family
(CERN ROOT's default text font; its metric clones TeX Gyre Heros and Nimbus
Sans where installed, else DejaVu Sans, matplotlib's own). The panel is laid
out in inches from the figure's bottom edge, so it keeps its size when a
window resizes the figure; `XYFigure` places its axes above it. Red is kept
for flagged samples, drawn as light-coral crosses: the category palettes have
no reds or pinks.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import matplotlib
import numpy as np
from astropy.time import Time
from matplotlib import font_manager
from matplotlib.colors import to_rgba
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea
from matplotlib.patches import Rectangle
from matplotlib.textpath import TextPath
from matplotlib.transforms import blended_transform_factory

from visplot.plot_spec import PlotSpec
from visplot.quantities import QUANTITIES, QuantityContext, category_label, utc_jd
from visplot.stream import FLAGGED_LAYER

FLAGGED_COLOR = "lightcoral"
# tab10 without its red (3) and pink (6); tab20 without its red (6, 7) and pink (12, 13) pairs.
_PALETTE_SMALL = [c for i, c in enumerate(matplotlib.colormaps["tab10"].colors) if i not in (3, 6)]
_PALETTE_LARGE = [c for i, c in enumerate(matplotlib.colormaps["tab20"].colors) if i not in (6, 7, 12, 13)]

# Helvetica's metric clones first; DejaVu Sans ships with matplotlib, so there is always one.
PANEL_FONTS = ("TeX Gyre Heros", "Nimbus Sans", "Liberation Sans", "DejaVu Sans")
FONT_PT = 9.0
FOOTER_PT = 7.0
LINE_IN = 0.21  # one line of the panel
FOOTER_IN = 0.2  # the footer line at the bottom: the caveat note and the record
RULE_GAP_IN = 0.12  # between the rule over the panel and its first line
_LEFT, _MIDDLE, _RIGHT = 0.03, 0.53, 0.97  # figure fractions: the left column, the right column, the edge
_SWATCH_PT = 9.0
_LABELS = ("Stokes", "Sources", "Channels", "Baselines", "Time", "Selection", "Drawn", "Flags")
_ORDINAL = {1: "st", 2: "nd", 3: "rd"}
LABEL_COLOR, VALUE_COLOR = "0.42", "0.08"

# Row filters of a request, as the panel names them.
_FILTERS = (("antennas", "antennas {}"), ("exclude_antennas", "excluding {}"), ("time_range", "time {}"),
            ("uvdist_range", "uv distance {}"), ("u_range_klambda", "u {} kλ"), ("v_range_klambda", "v {} kλ"),
            ("w_range_klambda", "w {} kλ"), ("uvdist_range_klambda", "uv distance {} kλ"),
            ("ha_range", "hour angle {}"), ("az_range", "azimuth {}"), ("el_range", "elevation {}"),
            ("pa_range", "parallactic angle {}"))


@lru_cache(maxsize=1)
def panel_font() -> str:
    """The first of PANEL_FONTS installed."""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    return next(name for name in PANEL_FONTS if name in installed or name == "DejaVu Sans")


def category_palette(n: int) -> list[tuple]:
    """Colors for `n` categories, none of them red or pink: tab10's 8, or
    tab20's 16 for more (repeating beyond 16)."""
    return list(_PALETTE_SMALL if n <= len(_PALETTE_SMALL) else _PALETTE_LARGE)


def category_colors(codes) -> dict[int, tuple]:
    """Category code -> RGBA, by the code's rank among `codes`."""
    codes = sorted(set(codes))
    palette = category_palette(len(codes))
    return {code: to_rgba(palette[rank % len(palette)]) for rank, code in enumerate(codes)}


def varies_along(plot: PlotSpec, axis_type: str, ctx: QuantityContext) -> bool:
    """Whether `plot`'s points differ along `axis_type` ("STOKES" or "FREQ")
    -- some quantity on its axes, or its color, varies along it -- as
    `stream.ChunkValues` shapes them: the visibility quantities along both;
    Stokes as a category along STOKES; frequency, and u, v, w and uv distance
    in wavelengths, along FREQ. Where a plot does not vary along Stokes, each
    point combines its visibility's selected Stokes (`combine_flags`)."""
    names = [(plot.x, plot.unit("x", ctx)), (plot.y, plot.unit("y", ctx))]
    if plot.colorize_by:
        names.append((plot.colorize_by, None))
    for name, unit in names:
        if QUANTITIES[name].needs_data:
            return True
        if axis_type == "STOKES" and name == "stokes":
            return True
        if axis_type == "FREQ" and (name == "freq" or (unit is not None and unit.base == "wavelength")):
            return True
    return False


@dataclass(frozen=True)
class PanelFacts:
    """What a run's selection holds, for the panel."""

    stokes: tuple[str, ...]  # the selected Stokes; a Stokes category code is its position here
    sources: tuple[tuple[int, str], ...]  # (id, name) of the sources in the selection
    n_channels: int
    n_channels_in_file: int
    freq_mhz: tuple[float, float] | None  # lowest and highest selected channel
    n_baselines: int  # cross-correlation baselines in the selection
    n_autocorrelations: int
    n_antennas: int
    time_utc: tuple[str, str] | None  # first and last selected timestamp, UTC
    filters: str  # the row filters given, as text; "" for none
    # (antenna pairs a row stride kept, pairs the selection has without it); None without one
    stride_pairs: tuple[int, int] | None = None


def selection_filters(request) -> str:
    """A request's row filters, e.g. "uv distance 0:5km, every 40th row"."""
    parts = [text.format(getattr(request, dest)) for dest, text in _FILTERS if getattr(request, dest)]
    for n, what in ((request.every_nth, "row"), (request.every_nth_integration, "integration")):
        if n and n > 1:
            suffix = "th" if 10 <= n % 100 <= 20 else _ORDINAL.get(n % 10, "th")
            parts.append(f"every {n}{suffix} {what}")
    if request.random_subset_n:
        seed = f", seed {request.random_seed}" if request.random_seed is not None else ""
        parts.append(f"{request.random_subset_n:,} random rows{seed}")
    return ", ".join(parts)


def panel_facts(request, index, row_indices, channel_indices, stokes_labels, sources: dict,
                ctx: QuantityContext, stride_pairs: tuple[int, int] | None = None) -> PanelFacts:
    """The facts of a run's selection: its rows (`row_indices`), channels
    (`channel_indices`; None: all), Stokes and sources, and what a row
    stride kept (`stride_pairs`, `RowSelection.stride_pairs`)."""
    freqs_hz = np.asarray(index.chan_freqs_hz if index.chan_freqs_hz is not None else [])
    chosen = freqs_hz if channel_indices is None else freqs_hz[np.asarray(channel_indices)]
    rows = np.asarray(row_indices)
    ant1, ant2 = np.asarray(index.ant1)[rows], np.asarray(index.ant2)[rows]
    cross = ant1 != ant2
    time_utc = None
    if rows.size:
        jd = np.asarray(index.jd)[rows]
        time_utc = tuple(Time(float(utc_jd(ctx, v)), format="jd", scale="utc").iso[:19] for v in (jd.min(), jd.max()))
    return PanelFacts(
        stokes=tuple(stokes_labels), sources=tuple(sorted(sources.items())),
        n_channels=len(chosen), n_channels_in_file=len(freqs_hz),
        freq_mhz=(float(chosen.min()) / 1e6, float(chosen.max()) / 1e6) if chosen.size else None,
        # one integer per (ant1, ant2): np.unique over a 2-D array's rows is 30x slower
        n_baselines=len(np.unique(ant1[cross].astype(np.int64) * 65536 + ant2[cross])),
        n_autocorrelations=len(np.unique(ant1[~cross])),
        n_antennas=len(np.union1d(ant1, ant2)), time_utc=time_utc, filters=selection_filters(request),
        stride_pairs=stride_pairs,
    )


def _swatch(color) -> DrawingArea:
    box = DrawingArea(_SWATCH_PT, _SWATCH_PT)
    box.add_artist(Rectangle((0, 0), _SWATCH_PT, _SWATCH_PT, facecolor=color, edgecolor="none"))
    return box


def _cross(color) -> DrawingArea:
    box = DrawingArea(_SWATCH_PT, _SWATCH_PT)
    for ys in ((0, _SWATCH_PT), (_SWATCH_PT, 0)):
        box.add_artist(Line2D((0, _SWATCH_PT), ys, color=color, lw=1.3))
    return box


@lru_cache(maxsize=4096)
def _text_width_pt(text: str, size: float = FONT_PT) -> float:
    """`text`'s width in points in the panel's font, from the font's own
    metrics (no renderer); trailing spaces count as a space's advance each."""
    stripped = text.rstrip()
    prop = FontProperties(family=panel_font(), size=size)
    width = TextPath((0, 0), stripped, prop=prop).get_extents().width if stripped else 0.0
    return width + (len(text) - len(stripped)) * 0.28 * size


def _ellipsize(text: str, width_pt: float) -> str:
    """`text`, cut with "…" to fit `width_pt`."""
    if _text_width_pt(text) <= width_pt:
        return text
    lo, hi = 0, len(text)
    while lo < hi:  # the longest prefix that fits with the ellipsis
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if _text_width_pt(text[:mid] + "…") <= width_pt else (lo, mid - 1)
    return text[:lo] + "…"


class PlotPanel:
    """The panel of one `XYFigure`. `build(width_in)` lays it out for the
    figure's width; `height_in` is how much of the figure's height it takes.
    `status` (what was drawn), `record` and `flagged` are TextAreas kept
    across rebuilds; `set_note` sets the caveat in the footer."""

    def __init__(self, fig, plot: PlotSpec, ctx: QuantityContext, facts: PanelFacts | None = None,
                 sources: list[str] | None = None):
        self.fig, self.plot, self.ctx, self.facts = fig, plot, ctx, facts
        self.source_names = [name for _, name in facts.sources] if facts else list(sources or [])
        font = panel_font()
        self.label_props = dict(family=font, size=FONT_PT, color=LABEL_COLOR)
        self.value_props = dict(family=font, size=FONT_PT, color=VALUE_COLOR)
        footer_props = dict(family=font, size=FOOTER_PT, color=LABEL_COLOR)
        self.status = TextArea("", textprops=self.value_props)
        self.flagged = TextArea("flagged", textprops=self.value_props)  # "flagged: none" when none are drawn
        self.record = TextArea("", textprops=footer_props)
        self.transform = blended_transform_factory(fig.transFigure, fig.dpi_scale_trans)  # x: fraction, y: inches
        self.note = fig.text(_LEFT, 0.06, "", transform=self.transform, ha="left", va="bottom",
                             color="darkred", wrap=True, family=font, size=FOOTER_PT)
        self._record_box = HPacker(children=[TextArea("record", textprops=footer_props), self.record],
                                   align="baseline", pad=0, sep=4)
        self.label_width_pt = max(_text_width_pt(label) for label in _LABELS) + 8
        self.n_lines = 0
        self._artists: list = []
        self._built_for = None

    # ---- the color key -----------------------------------------------------------

    def category_codes(self, seen=()) -> tuple[int, ...]:
        """The codes of the categories --colorize-by colors: those the
        selection holds (known before drawing), and any seen besides."""
        selected = []
        if self.facts is not None:
            if self.plot.colorize_by == "stokes":
                selected = range(len(self.facts.stokes))
            elif self.plot.colorize_by == "source":
                selected = [sid for sid, _ in self.facts.sources]
        return tuple(sorted(set(selected) | set(seen)))

    def colors(self, seen=()) -> dict[int, tuple]:
        """Grid layer -> RGBA: layer code + 1 per category, or 1 for the one
        color, and the flagged layer's."""
        if self.plot.colorize_by:
            colors = {code + 1: rgba for code, rgba in category_colors(self.category_codes(seen)).items()}
        else:
            colors = {1: to_rgba(self.plot.color)}
        colors[FLAGGED_LAYER] = to_rgba(FLAGGED_COLOR)
        return colors

    def key_entries(self, seen=()) -> list[tuple[str, tuple]]:
        """(label, RGBA) of each colored category, in key order."""
        colors = category_colors(self.category_codes(seen))
        return [(category_label(self.plot.colorize_by, code, self.ctx), rgba) for code, rgba in colors.items()]

    # ---- what the panel says -------------------------------------------------------

    def flag_rule(self) -> str:
        """How a point's flag combines its visibility's selected Stokes,
        where the plot does not vary along Stokes ("" otherwise)."""
        facts, plot = self.facts, self.plot
        if facts is None or not plot.apply_flags or len(facts.stokes) < 2:
            return ""
        if varies_along(plot, "STOKES", self.ctx):
            return ""
        return f"a point is flagged if any of {', '.join(facts.stokes)} is"

    def _key_items(self, seen) -> list:
        items = []
        if self.plot.colorize_by:
            for label, rgba in self.key_entries(seen):
                items += [_swatch(rgba), f"{label}   "]
        elif self.plot.show_flagged:
            items += [_swatch(to_rgba(self.plot.color)), "unflagged   "]
        if self.plot.show_flagged:
            items += [_cross(to_rgba(FLAGGED_COLOR)), self.flagged]
        return items

    def _lines(self, seen, width_in: float) -> tuple[list, list, list, list]:
        """(key lines, left column, right column, bottom lines), each a list
        of (label, items)."""
        facts, plot = self.facts, self.plot
        key = []
        items = self._key_items(seen)
        if items:
            label = {"stokes": "Stokes", "source": "Sources"}.get(plot.colorize_by, "")
            key = self._wrap(label, items, (_RIGHT - _LEFT) * width_in * 72.0)
        left, right, bottom = [], [], []
        if facts is not None:
            if plot.needs_data and plot.colorize_by != "stokes":
                left.append(("Stokes", [", ".join(facts.stokes)]))
            if facts.freq_mhz is not None:
                lo, hi = facts.freq_mhz
                count = (f"all {facts.n_channels:,}" if facts.n_channels == facts.n_channels_in_file
                         else f"{facts.n_channels:,} of {facts.n_channels_in_file:,}")
                left.append(("Channels", [f"{count}, {lo:.3f} to {hi:.3f} MHz"]))
            parts = [f"{facts.n_baselines:,} baselines"] if facts.n_baselines else []
            if facts.n_autocorrelations:
                parts.append(f"{facts.n_autocorrelations:,} autocorrelations")
            baselines = f"{' and '.join(parts) or 'none'}, {facts.n_antennas} antennas"
            if facts.stride_pairs is not None and facts.stride_pairs[0] < facts.stride_pairs[1]:
                # a row stride skipped some: said in the warning color
                kept, available = facts.stride_pairs
                what = "antenna pairs" if facts.n_autocorrelations else "baselines"
                baselines = TextArea(f"{kept:,} of {available:,} {what} kept by the row stride, "
                                     f"{facts.n_antennas} antennas", textprops=dict(self.value_props, color="darkred"))
            left.append(("Baselines", [baselines]))
            if facts.time_utc is not None:
                t0, t1 = facts.time_utc
                right.append(("Time", [f"{t0} to {t1[11:] if t1[:10] == t0[:10] else t1} UTC"]))
            right.append(("Selection", [facts.filters or "all rows"]))
            rule = self.flag_rule()
            if rule:
                bottom.append(("Flags", [rule]))
        right.append(("Drawn", [self.status]))
        return key, left, right, bottom

    def _wrap(self, label: str, items, width_pt: float) -> list:
        """(label, items) lines of a key, wrapped between entries (a swatch
        and the text after it stay together)."""
        room = width_pt - self.label_width_pt - 8  # and a margin for the text boxes' own bearings
        units = []
        for item in items:
            if units and isinstance(units[-1][-1], DrawingArea):
                units[-1].append(item)
            else:
                units.append([item])
        lines, current, used = [], [], 0.0
        for unit in units:
            width = sum(_text_width_pt(i) if isinstance(i, str) else
                        (_SWATCH_PT if isinstance(i, DrawingArea) else _text_width_pt(i.get_text())) + 4
                        for i in unit)
            if current and used + width > room:
                lines.append(current)
                current, used = [], 0.0
            current += unit
            used += width
        lines.append(current)
        return [(label if n == 0 else "", line) for n, line in enumerate(lines)]

    # ---- layout -------------------------------------------------------------------

    @property
    def height_in(self) -> float:
        return FOOTER_IN + self.n_lines * LINE_IN + RULE_GAP_IN

    def _line_box(self, label: str, items, width_pt: float) -> HPacker:
        spacer = DrawingArea(max(0.0, self.label_width_pt - _text_width_pt(label)), 1)
        children = [TextArea(label, textprops=self.label_props), spacer]
        room = width_pt - self.label_width_pt
        for item in items:
            children.append(TextArea(_ellipsize(item, room), textprops=self.value_props) if isinstance(item, str)
                            else item)
        return HPacker(children=children, align="center", pad=0, sep=4)

    def _anchor(self, box, x: float, y_in: float, loc: str = "upper left") -> None:
        anchored = AnchoredOffsetbox(loc=loc, child=box, bbox_to_anchor=(x, y_in), bbox_transform=self.transform,
                                     frameon=False, pad=0, borderpad=0)
        self._artists.append(self.fig.add_artist(anchored))

    def build(self, width_in: float, seen=()) -> None:
        """Lay the panel out for a figure `width_in` wide (skipped when
        nothing it depends on changed): the key, two columns of facts, the
        flag rule, and the footer, each line at its own height."""
        codes = self.category_codes(seen)
        if self._built_for == (width_in, codes):
            return
        self._built_for = (width_in, codes)
        for artist in self._artists:
            artist.remove()
        self._artists = []
        key, left, right, bottom = self._lines(seen, width_in)
        rows = max(len(left), len(right))
        self.n_lines = len(key) + rows + len(bottom)
        top_in = FOOTER_IN + self.n_lines * LINE_IN
        rule = Line2D([_LEFT, _RIGHT], [top_in + RULE_GAP_IN * 0.6] * 2, transform=self.transform, lw=0.6,
                      color="0.75")
        self._artists.append(self.fig.add_artist(rule))
        full_pt, left_pt, right_pt = ((_RIGHT - _LEFT) * width_in * 72.0, (_MIDDLE - _LEFT) * width_in * 72.0 - 10,
                                      (_RIGHT - _MIDDLE) * width_in * 72.0)
        line = 0
        for label, items in key:
            self._anchor(self._line_box(label, items, full_pt), _LEFT, top_in - line * LINE_IN)
            line += 1
        for column, x, width_pt in ((left, _LEFT, left_pt), (right, _MIDDLE, right_pt)):
            for n, (label, items) in enumerate(column):
                self._anchor(self._line_box(label, items, width_pt), x, top_in - (line + n) * LINE_IN)
        line += rows
        for label, items in bottom:
            self._anchor(self._line_box(label, items, full_pt), _LEFT, top_in - line * LINE_IN)
            line += 1
        self._anchor(self._record_box, _RIGHT, 0.06, loc="lower right")

    # ---- texts --------------------------------------------------------------------

    def set_note(self, text: str) -> None:
        self.note.set_text(text)

    def set_flagged_drawn(self, drawn: bool) -> None:
        self.flagged.set_text("flagged" if drawn else "flagged: none")
