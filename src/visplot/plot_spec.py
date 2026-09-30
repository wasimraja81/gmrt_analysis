"""What one plot shows: a y quantity against an x quantity, each in a unit,
optionally colored by a category, plus how flags, mirroring and axis ranges
are handled. Named plots are presets that expand to one or more of these."""

from __future__ import annotations

from dataclasses import dataclass, replace

from visplot.axis_scale import AxisScale
from visplot.quantities import QUANTITIES, QuantityContext, Unit, canonical, convert, resolve_unit


@dataclass(frozen=True)
class PlotSpec:
    y: str
    x: str
    colorize_by: str | None = None
    apply_flags: bool = True  # exclude flagged samples (weight <= 0); needs the weights read from disk
    show_flagged: bool = False  # draw flagged samples, as their own layer on top
    # A point combining its visibility's selected Stokes (e.g. u-v) is flagged if "any" of them is, or
    # only if "all" are (`stream.combine_flags`).
    combine_flags: str = "any"
    mirror: bool = False  # also plot (-x, -y)
    x_range: tuple[float, float] | None = None  # None: from the data; in x_unit
    y_range: tuple[float, float] | None = None
    point_size: float | None = None  # marker area, points^2; None: from the number of samples
    color: str = "tab:blue"
    name: str = ""  # used in output filenames
    reference_lines: tuple[tuple[str, float], ...] = ()  # ("x" or "y", value): dashed lines, e.g. transit
    x_scale: str = "linear"  # linear, log, symlog, asinh
    y_scale: str = "linear"
    x_range_mode: str = "minmax"  # minmax, or percentile (range_percentiles) -- when no range is given
    y_range_mode: str = "minmax"
    range_percentiles: tuple[float, float] = (0.1, 99.9)
    scale_linear_width: float = 1.0  # symlog's linthresh, asinh's linear width
    aspect: str = "auto"  # auto (equal for same-kind axes, e.g. u vs v), equal, free
    x_unit: str | None = None  # None: the quantity's default unit
    y_unit: str | None = None

    def __post_init__(self):
        """Alias names become their quantity and unit (u_klambda: u in
        klambda), a unit given by its label becomes its name (kλ: klambda),
        and combinations no plot can show raise ValueError. Units of the
        visibility quantities depend on the file's BUNIT and are checked
        against it by `check_units`."""
        for axis in ("x", "y"):
            name, unit = canonical(getattr(self, axis), getattr(self, f"{axis}_unit"))
            if name in QUANTITIES and not QUANTITIES[name].unit_from_bunit:
                chosen = resolve_unit(name, unit)
                unit = chosen.name if unit is not None else None
                if chosen.clock and not self.axis_scale(axis).is_linear:
                    raise ValueError(f"{name} in {chosen.name} (clock time) needs a linear {axis} scale")
                if chosen.clock and self.mirror:
                    raise ValueError(f"mirroring negates values; {name} in {chosen.name} (clock time) has no zero")
            object.__setattr__(self, axis, name)
            object.__setattr__(self, f"{axis}_unit", unit)

    def unit(self, axis: str, ctx: QuantityContext | None = None) -> Unit:
        """The unit of axis "x" or "y" (the file's BUNIT for a visibility
        quantity comes from `ctx`)."""
        return resolve_unit(self.x if axis == "x" else self.y, self.x_unit if axis == "x" else self.y_unit, ctx)

    def check_units(self, ctx: QuantityContext) -> None:
        """Raise ValueError if a unit does not apply here, e.g. mJy when the
        file's BUNIT is not a flux density."""
        self.unit("x", ctx)
        self.unit("y", ctx)

    def with_units(self, ctx: QuantityContext) -> PlotSpec:
        """This plot with both units named explicitly (defaults resolved), so
        it states what it shows whatever the defaults are later."""
        return replace(self, x_unit=self.unit("x", ctx).name or None, y_unit=self.unit("y", ctx).name or None)

    @property
    def equal_aspect(self) -> bool:
        """Whether one unit is drawn the same length on both axes. `auto`:
        when x and y are the same kind of quantity (their aspect group, e.g.
        u and v) in the same unit, and both axes are linear."""
        if self.aspect != "auto":
            return self.aspect == "equal"
        gx, gy = QUANTITIES[self.x].aspect_group, QUANTITIES[self.y].aspect_group
        linear = self.axis_scale("x").is_linear and self.axis_scale("y").is_linear
        same_unit = _unit_name(self.x, self.x_unit) == _unit_name(self.y, self.y_unit)
        return gx is not None and gx == gy and same_unit and linear

    def axis_scale(self, axis: str) -> AxisScale:
        return AxisScale(self.x_scale if axis == "x" else self.y_scale, self.scale_linear_width)

    def range_mode(self, axis: str) -> str:
        return self.x_range_mode if axis == "x" else self.y_range_mode

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


def _unit_name(name: str, unit: str | None) -> str | None:
    """A unit's name with the default resolved where no file is needed (a
    visibility quantity's default, the file's BUNIT, stays None)."""
    if unit is not None or QUANTITIES[name].unit_from_bunit:
        return unit
    return resolve_unit(name).name


# Observing geometry per selected row, colored by source; flags are
# per-sample, so these plot every selected row and read no visibility data.
PRESETS: dict[str, list[PlotSpec]] = {
    "ha-range": [PlotSpec(y="ha", x="time", colorize_by="source", apply_flags=False, y_range=(-12.0, 12.0),
                          reference_lines=(("y", 0.0),), y_unit="h")],  # transit
    "az-el-range": [
        PlotSpec(y="el", x="time", colorize_by="source", apply_flags=False, y_range=(-90.0, 90.0),
                 reference_lines=(("y", 0.0),), y_unit="deg"),  # horizon
        PlotSpec(y="az", x="time", colorize_by="source", apply_flags=False, y_range=(0.0, 360.0), y_unit="deg"),
    ],
    "parallactic-angle-range": [
        PlotSpec(y="pa", x="time", colorize_by="source", apply_flags=False, y_range=(-180.0, 180.0), y_unit="deg"),
    ],
}


def expand_plot_name(name: str, style: PlotSpec) -> list[PlotSpec]:
    """A preset name, or a generic "Y-vs-X" name, as the plots it stands for.
    `style` supplies everything a generic plot takes from the command line
    (colorize, flags, mirror, ranges, units, marker size, color); presets
    keep their own quantities, coloring, flag handling and fixed ranges, and
    take the style's units (their fixed ranges and reference lines
    converted). Raises ValueError for a unit that does not apply."""
    if name in PRESETS:
        presets = PRESETS[name]
        return [
            _preset_in_units(replace(p, point_size=style.point_size,
                                     name=f"{name}_{p.y}" if len(presets) > 1 else name), style)
            for p in presets
        ]
    y, x = name.split("-vs-", 1)
    return [replace(style, y=y, x=x, name=name)]


def _preset_in_units(plot: PlotSpec, style: PlotSpec) -> PlotSpec:
    changes = {}
    lines = list(plot.reference_lines)
    for axis in ("x", "y"):
        quantity = plot.x if axis == "x" else plot.y
        unit = style.x_unit if axis == "x" else style.y_unit
        if unit is None:
            continue
        old = plot.x_unit if axis == "x" else plot.y_unit
        changes[f"{axis}_unit"] = unit
        fixed = plot.x_range if axis == "x" else plot.y_range
        if fixed is not None:
            changes[f"{axis}_range"] = tuple(convert(v, quantity, old, unit) for v in fixed)
        lines = [(a, convert(v, quantity, old, unit) if a == axis else v) for a, v in lines]
    return replace(plot, reference_lines=tuple(lines), **changes)
