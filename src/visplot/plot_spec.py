"""What one plot shows: a y quantity against an x quantity, optionally colored
by a category, plus how flags, mirroring and axis ranges are handled. Named
plots are presets that expand to one or more of these."""

from __future__ import annotations

from dataclasses import dataclass, replace

from visplot.quantities import QUANTITIES


@dataclass(frozen=True)
class PlotSpec:
    y: str
    x: str
    colorize_by: str | None = None
    apply_flags: bool = True  # exclude flagged samples (weight <= 0); needs the weights read from disk
    show_flagged: bool = False  # draw flagged samples, as their own layer on top
    mirror: bool = False  # also plot (-x, -y)
    x_range: tuple[float, float] | None = None  # None: from the data
    y_range: tuple[float, float] | None = None
    point_size: float | None = None  # marker area, points^2; None: from the number of samples
    color: str = "tab:blue"
    name: str = ""  # used in output filenames
    reference_lines: tuple[tuple[str, float], ...] = ()  # ("x" or "y", value): dashed lines, e.g. transit

    @property
    def quantities(self) -> list[str]:
        return [q for q in (self.y, self.x, self.colorize_by) if q is not None]

    @property
    def needs_data(self) -> bool:
        """Whether visibility data must be read: for a data quantity, or for
        the weights that flags come from."""
        return self.apply_flags or any(QUANTITIES[q].needs_data for q in self.quantities)

    @property
    def title(self) -> str:
        return f"{QUANTITIES[self.y].display_name} vs {QUANTITIES[self.x].display_name}"


# Observing geometry per selected row, colored by source; flags are
# per-sample, so these plot every selected row and read no visibility data.
PRESETS: dict[str, list[PlotSpec]] = {
    "ha-range": [PlotSpec(y="ha_h", x="time_h", colorize_by="source", apply_flags=False, y_range=(-12.0, 12.0),
                          reference_lines=(("y", 0.0),))],  # transit
    "az-el-range": [
        PlotSpec(y="el_deg", x="time_h", colorize_by="source", apply_flags=False, y_range=(-90.0, 90.0),
                 reference_lines=(("y", 0.0),)),  # horizon
        PlotSpec(y="az_deg", x="time_h", colorize_by="source", apply_flags=False, y_range=(0.0, 360.0)),
    ],
    "parallactic-angle-range": [
        PlotSpec(y="pa_deg", x="time_h", colorize_by="source", apply_flags=False, y_range=(-180.0, 180.0)),
    ],
}


def expand_plot_name(name: str, style: PlotSpec) -> list[PlotSpec]:
    """A preset name, or a generic "Y-vs-X" name, as the plots it stands for.
    `style` supplies everything a generic plot takes from the command line
    (colorize, flags, mirror, ranges, marker size, color); presets keep their
    own quantities, coloring, flag handling and fixed ranges."""
    if name in PRESETS:
        presets = PRESETS[name]
        return [
            replace(p, point_size=style.point_size, name=f"{name}_{p.y}" if len(presets) > 1 else name)
            for p in presets
        ]
    y, x = name.split("-vs-", 1)
    return [replace(style, y=y, x=x, name=name)]
