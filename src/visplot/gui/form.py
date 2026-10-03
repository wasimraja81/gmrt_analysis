"""The GUI's form: one control per plot-request option, bound by the
option's own name (its argparse `dest`), so the form and the command line
cannot drift apart.

- `RequestForm.request()` is a `PlotRequest` built from the fields;
  `load(request)` shows a request in them. What a field offers comes from
  the same sources as the command line: choices from the parser, quantities
  and units from the quantity registry, tooltips from the option's `--help`.
- The options a form field does not hold are set by named GUI actions
  (`ACTION_OPTIONS`); a test checks that every request option is one or the
  other, so an option added to the command line without a GUI control fails.
  Options built for the command line ahead of their GUI controls are listed
  in `PENDING_GUI_OPTIONS` with the plan step that adds them; the GUI's
  requests leave them at their defaults until then.
"""

from __future__ import annotations

from typing import Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from PySide6 import QtCore, QtGui, QtWidgets

from instruments.observatory_time_zones import observatory_time_zone
from visplot.fonts import PANEL_FONTS
from visplot.gui.widgets import (CheckedLineEdit, CollapsibleSection, NoteBox, form_layout, grouped_combo, hint, row,
                                 select_data)
from visplot.plot_spec import PRESETS
from visplot.quantities import QUANTITIES, QuantityContext, units_of
from visplot.request import PlotRequest, build_arg_parser
from visplot.request_args import (
    DEFAULT_PAGE_GRID,
    MAX_PAGE_GRID,
    TABLE_PLOTS,
    resolve_antennas_arg,
    resolve_channels_arg,
    resolve_deg_range_arg,
    resolve_ha_range_arg,
    resolve_klambda_range_arg,
    resolve_page_grid_arg,
    resolve_percentiles_arg,
    resolve_plain_range_arg,
    resolve_stokes_axis_selection,
    resolve_time_range_arg,
    resolve_uvdist_range_arg,
)

# Options the GUI sets through an action (a tool or a menu item), and the action that sets each.
ACTION_OPTIONS = {
    "locate": "the Locate tool on a plot (its box)",
    "locate_csv": "the Locate tool's 'Save all as CSV'",
    "output_dir": "File > Save plots as files",
    "output_prefix": "File > Save plots as files",
    "no_highres_pdf": "File > Save plots as files",
    "dpi": "Export on a plot (its dpi), or File > Save plots as files",
    "figure_size": "Export on a plot (the plot's size in the window), or File > Save plots as files",
    "listobs": "the Listing tab's section boxes and List",
    "scan_gap": "the Listing tab's scan gap",
    "scan_longest": "the Listing tab's longest scan",
}

# Options whose GUI controls are still to be built, and the plan step building each (none now).
PENDING_GUI_OPTIONS: dict[str, str] = {}

# Hover text of the Pages section (T26), in everyday words for a first-time reader.
PAGES_HELP = {
    "one_plot_per": "Make a separate plot for each baseline, antenna, source or Stokes product in your selection, "
                    "laid out several to a page.\nAn antenna's plot shows all of that antenna's baselines, so each "
                    "baseline appears in two antennas' plots.",
    "page_grid": "How many plots go on a page: rows x columns.\nAuto: 5 rows by 6 columns, or a smaller grid "
                 "when there are fewer plots.\n1 x 1 puts one plot on each page.",
    "range_from": "How the {axis} axis range of each plot is chosen, when you make one plot per baseline (antenna, "
                  "source, Stokes):\n"
                  "- Each plot: every plot fits its own samples. Each fills its frame, but the scales differ, so "
                  "read each plot's tick labels.\n"
                  "- All plots: one range for every plot on every page, from all the samples. The scales match, so "
                  "plots compare by eye; tick labels show on the outer row and column only.\n"
                  "- Default: all plots when a page holds several plots; each plot with one plot a page.\n"
                  "The Axes section's Range says how a range is found (min to max, or percentiles); a fixed range "
                  "there (lo:hi) is used for every plot instead.",
}
RANGE_FROM_HELP = {
    None: "All plots when a page holds several plots; each plot with one plot a page.",
    "each": "Each plot's {axis} axis fits its own samples: every plot fills its frame, with its own scale.",
    "all": "One {axis} range for every plot on every page, from all the samples: the plots compare by eye.",
}

# Quantity groups in the axis lists, in order; a quantity not listed here goes under "Other",
# so a quantity added to the registry appears without a change here.
QUANTITY_GROUPS = (
    ("Visibility", ("amp", "phase", "real", "imag")),
    ("Time", ("time",)),
    ("Baseline", ("uvdist", "u", "v", "w")),
    ("Frequency", ("freq",)),
    ("Observing geometry", ("ha", "az", "el", "pa")),
    ("Category", ("stokes", "source")),
)
GENERIC = "generic"
PRESET_LABELS = {"ha-range": "Hour angle vs time", "az-el-range": "Elevation and azimuth vs time",
                 "parallactic-angle-range": "Parallactic angle vs time"}
TABLE_LABELS = {"antenna-layout": "Antenna layout", "source-listing": "Source listing"}


def _parser_help() -> dict[str, str]:
    return {a.dest: (a.help or "").replace("%%", "%") for a in build_arg_parser()._actions}


def _parser_choices(dest: str) -> list[str]:
    action = next(a for a in build_arg_parser()._actions if a.dest == dest)
    return list(action.choices)


def _check_with(resolver: Callable) -> Callable[[str], str | None]:
    def check(text: str) -> str | None:
        resolver(text)
        return None
    return check


class Field:
    """One request option in the form: `get()` gives the value the command
    line would parse, `set(value)` shows one; `changed` fires on edits."""

    def __init__(self, dest: str, widget: QtWidgets.QWidget, get: Callable, set: Callable, changed):
        self.dest = dest
        self.widget = widget
        self.get = get
        self.set = set
        self.changed = changed


def text_field(dest: str, edit: QtWidgets.QLineEdit, default=None) -> Field:
    """Text as typed; empty gives `default` (None: the option not given), which
    the field shows as its placeholder."""
    if default is not None:
        edit.setPlaceholderText(str(default))
    return Field(dest, edit, lambda: edit.text().strip() or default,
                 lambda v: edit.setText("" if v is None or v == default else str(v)), edit.textChanged)


def number_field(dest: str, edit: QtWidgets.QLineEdit, kind: type, default=None) -> Field:
    """A number typed as text; empty gives `default` (None: the option not
    given), shown as the placeholder."""
    if default is not None:
        edit.setPlaceholderText(str(default))

    def get():
        text = edit.text().strip()
        return kind(text) if text else default
    return Field(dest, edit, get, lambda v: edit.setText("" if v is None or v == default else str(v)),
                 edit.textChanged)


def combo_field(dest: str, combo: QtWidgets.QComboBox) -> Field:
    return Field(dest, combo, lambda: combo.currentData(), lambda v: select_data(combo, v),
                 combo.currentIndexChanged)


def flag_field(dest: str, box: QtWidgets.QCheckBox) -> Field:
    return Field(dest, box, box.isChecked, lambda v: box.setChecked(bool(v)), box.toggled)


class ChecklistField:
    """Sources or Stokes as check boxes: None when all are checked (the
    option not given), else the checked names joined by commas."""

    def __init__(self, dest: str):
        self.dest = dest
        self.widget = QtWidgets.QListWidget()
        self.widget.setMaximumHeight(130)
        self.changed = self.widget.itemChanged
        self._names: list[str] = []

    def fill(self, items: list[tuple[str, str]]) -> None:
        """(name, text) per item, all checked."""
        self.widget.blockSignals(True)
        self.widget.clear()
        self._names = [name for name, _ in items]
        for name, text in items:
            item = QtWidgets.QListWidgetItem(text)
            item.setData(QtCore.Qt.UserRole, name)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked)
            self.widget.addItem(item)
        self.widget.blockSignals(False)
        self.changed.emit(None)

    def checked(self) -> list[str]:
        return [self.widget.item(i).data(QtCore.Qt.UserRole) for i in range(self.widget.count())
                if self.widget.item(i).checkState() == QtCore.Qt.Checked]

    def set_all(self, on: bool) -> None:
        for i in range(self.widget.count()):
            self.widget.item(i).setCheckState(QtCore.Qt.Checked if on else QtCore.Qt.Unchecked)

    def get(self):
        checked = self.checked()
        return None if not self._names or len(checked) == len(self._names) else ",".join(checked)

    def set(self, value) -> None:
        wanted = None if value is None else {v.strip().upper() for v in str(value).split(",")}
        for i in range(self.widget.count()):
            item = self.widget.item(i)
            on = wanted is None or str(item.data(QtCore.Qt.UserRole)).upper() in wanted
            item.setCheckState(QtCore.Qt.Checked if on else QtCore.Qt.Unchecked)


class AxisControls:
    """One axis: quantity, unit, scale, and its range (from the data, or fixed).
    `optional`: its quantity may be "none" (a stack's lower plot, T26)."""

    def __init__(self, axis: str, form: "RequestForm", optional: bool = False):
        self.axis = axis
        self.form = form
        groups, listed = [], set()
        for heading, names in QUANTITY_GROUPS:
            present = [n for n in names if n in QUANTITIES]
            listed.update(present)
            groups.append((heading, [(f"{QUANTITIES[n].display_name}  ({n})", n) for n in present]))
        other = [n for n in QUANTITIES if n not in listed]
        if other:
            groups.append(("Other", [(f"{QUANTITIES[n].display_name}  ({n})", n) for n in other]))
        self.quantity = grouped_combo(groups)
        if optional:
            none = QtGui.QStandardItem("none")
            none.setData(None, QtCore.Qt.UserRole)
            self.quantity.model().insertRow(0, none)
            self.quantity.setCurrentIndex(0)
        self.unit = QtWidgets.QComboBox()
        self.scale = QtWidgets.QComboBox()
        for name in _parser_choices(f"{axis}_scale"):
            self.scale.addItem(name, name)
        self.range_mode = QtWidgets.QComboBox()
        for name, text in (("minmax", "data min to max"), ("percentile", "data percentiles")):
            if name in _parser_choices(f"{axis}_range_mode"):
                self.range_mode.addItem(text, name)
        self.fixed_range = CheckedLineEdit("lo:hi (overrides)", _check_with(resolve_plain_range_arg))
        self.quantity.currentIndexChanged.connect(self._quantity_changed)
        self.unit.currentIndexChanged.connect(self._unit_changed)

    def quantity_name(self) -> str | None:
        return self.quantity.currentData()

    def _quantity_changed(self) -> None:
        self.fill_units()

    def fill_units(self) -> None:
        name = self.quantity_name()
        previous = self.unit.currentData()
        self.unit.blockSignals(True)
        self.unit.clear()
        if name is not None and not QUANTITIES[name].categorical:
            for unit in units_of(name, self.form.unit_context()):
                text = unit.label or "(file has no BUNIT)"
                if unit.clock:
                    text += "  (clock, dd:hh:mm:ss)"
                self.unit.addItem(text, unit.name or None)
            default = QUANTITIES[name].default_unit
            select_data(self.unit, previous if self.unit.findData(previous) >= 0 else (default or None))
        self.unit.setEnabled(self.unit.count() > 1)
        self.unit.blockSignals(False)
        self._unit_changed()

    def _unit_changed(self) -> None:
        """Clock time and categories take a linear scale only."""
        name = self.quantity_name()
        clock = False
        if name is not None and self.unit.currentData() is not None:
            clock = any(u.clock for u in units_of(name, self.form.unit_context())
                        if u.name == self.unit.currentData())
        linear_only = clock or (name is not None and QUANTITIES[name].categorical)
        if linear_only:
            select_data(self.scale, "linear")
        self.scale.setEnabled(not linear_only)
        self.form.changed.emit()


class RequestForm(QtWidgets.QWidget):
    """The request's fields in sections (Data, Axes, Selection, Display;
    Performance, cache and records). `fields` maps each option name to its
    `Field`."""

    changed = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.help = _parser_help()
        self.fields: dict[str, object] = {}
        self._bunit: str | None = None
        self._antennas = None
        self._chan_freqs_hz = None
        self._stokes_labels = None
        self._first_jd: float | None = None
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.data_section = self._data_section()
        # Selection before Axes: what a plot reads is seen before what it shows (all the data can be slow).
        for section in (self.data_section, self._selection_section(), self._axes_section(), self._pages_section(),
                        self._display_section(), self._performance_section()):
            layout.addWidget(section)
        layout.addStretch(1)
        for f in self.fields.values():
            f.changed.connect(lambda *_: self.changed.emit())
            tip = self.help.get(f.dest)
            if tip and not f.widget.toolTip():
                f.widget.setToolTip(tip)
                if isinstance(f.widget, CheckedLineEdit):
                    f.widget.set_help(tip)
        self.x.fill_units()
        self.y.fill_units()
        self.y2.fill_units()

    # ---- building -----------------------------------------------------------

    def _add(self, f) -> object:
        self.fields[f.dest] = f
        return f.widget

    def _data_section(self) -> CollapsibleSection:
        section = CollapsibleSection("Data")
        form = form_layout(section.body)
        self.path_edit = QtWidgets.QLineEdit()
        self.path_edit.setPlaceholderText("a UVFITS file with its row index")
        self.open_button = QtWidgets.QPushButton("Open…")
        self._add(text_field("fits_path", self.path_edit))
        form.addRow("File", row(self.path_edit, self.open_button, stretches=(1, 0)))
        # Opening or building the file's index: a thin bar, its words, and Stop (a build only).
        self.progress = QtWidgets.QProgressBar(objectName="fileProgress", textVisible=False)
        self.progress.setFixedHeight(6)
        self.stop_build_button = QtWidgets.QToolButton(text="Stop")
        self.stop_build_button.setToolTip("Stop building the index; nothing is saved, and its record says so")
        self.progress_text = hint("")
        self.progress_row = row(self.progress, self.stop_build_button, stretches=(1, 0))
        form.addRow("", self.progress_row)
        form.addRow("", self.progress_text)
        # A file without its row index: what building one takes (a box of fixed height, scrolled
        # when longer, so nothing is clipped beside the button), and the button.
        self.index_note = NoteBox(lines=3)
        self.build_button = QtWidgets.QPushButton("Build index")
        self.build_button.setToolTip("Read the file's row parameters once and save the row index beside it, "
                                     "as the pipeline's build_index stage does")
        self.index_row = row(self.index_note, self.build_button, stretches=(1, 0))
        form.addRow("", self.index_row)
        self.show_progress(None)
        self.show_missing_index(None)
        self.summary = QtWidgets.QLabel("No file open.")
        self.summary.setObjectName("hint")
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        form.addRow(self.summary)
        return section

    def _axes_section(self) -> CollapsibleSection:
        section = CollapsibleSection("Axes")
        form = form_layout(section.body)
        self.kind = QtWidgets.QComboBox()
        self.kind.addItem("Y against X", GENERIC)
        for name in PRESETS:
            self.kind.addItem(PRESET_LABELS.get(name, name), name)
        for name in sorted(TABLE_PLOTS):
            self.kind.addItem(TABLE_LABELS.get(name, name), name)
        self.kind.setToolTip("Y against X: any two quantities. The others are the command line's named plots.")
        form.addRow("Plot", self.kind)
        self.y = AxisControls("y", self)
        self.x = AxisControls("x", self)
        self.swap = QtWidgets.QToolButton(text="⇅")
        self.swap.setToolTip("Swap x and y (quantities, units, scales, ranges)")
        self.swap.clicked.connect(self._swap_axes)
        # A stack's lower plot (T26): Y2 against the same X, under the plot; its fields frozen, and
        # left out of the request, without one.
        self.y2 = AxisControls("y2", self, optional=True)
        self.y2.quantity.setToolTip("Also plot this quantity, below the first, against the same X (a stack: "
                                    "the second plot's y axis takes these fields)")
        defaults = {"y2_unit": None, "y2_range": None, "y2_scale": "linear", "y2_range_mode": "minmax"}
        for axis, controls in (("y", self.y), ("y2", self.y2), ("x", self.x)):
            heading = {"y": "Y axis", "y2": "Also plot below (same X)", "x": "X axis"}[axis]
            label = QtWidgets.QLabel(f"<b>{heading}</b>")
            form.addRow(row(label, self.swap, stretches=(1, 0)) if axis == "y" else label)
            form.addRow("Quantity", controls.quantity)
            fields = [combo_field(f"{axis}_unit", controls.unit), combo_field(f"{axis}_scale", controls.scale),
                      combo_field(f"{axis}_range_mode", controls.range_mode),
                      text_field(f"{axis}_range", controls.fixed_range)]
            if axis == "y2":
                for f in fields:
                    shown = f.get
                    f.get = lambda shown=shown, dest=f.dest: shown() if self._has_lower() else defaults[dest]
            unit, scale, mode, fixed = (self._add(f) for f in fields)
            form.addRow("Unit", unit)
            form.addRow("Scale", scale)
            form.addRow("Range", row(mode, fixed))
        select_data(self.y.quantity, "amp")
        select_data(self.x.quantity, "time")
        self.fields["plots"] = Field("plots", self.kind, self._plots, self._set_plots, self.kind.currentIndexChanged)
        for controls in (self.x, self.y, self.y2):
            controls.quantity.currentIndexChanged.connect(lambda *_: self.changed.emit())
        self.y2.quantity.currentIndexChanged.connect(lambda *_: self._kind_changed())
        self.kind.currentIndexChanged.connect(self._kind_changed)

        self.percentiles = CheckedLineEdit("", _check_with(resolve_percentiles_arg))
        form.addRow("Percentiles", self._add(text_field("range_percentiles", self.percentiles,
                                                        default=build_arg_parser().get_default("range_percentiles"))))
        aspect = QtWidgets.QComboBox()
        for name in _parser_choices("aspect"):
            aspect.addItem(name, name)
        form.addRow("Aspect", self._add(combo_field("aspect", aspect)))
        width = CheckedLineEdit("", lambda t: None if float(t) > 0 else "must be positive")
        form.addRow("Linear width", self._add(number_field("scale_linear_width", width, float,
                                                           default=build_arg_parser().get_default("scale_linear_width"))))
        self.zone = CheckedLineEdit("the observatory's (from TELESCOP)", self._check_zone)
        form.addRow("Time zone", self._add(text_field("time_zone", self.zone)))
        return section

    def _selection_section(self) -> CollapsibleSection:
        section = CollapsibleSection("Selection")
        form = form_layout(section.body)
        self.sources = ChecklistField("sources")
        all_sources = QtWidgets.QPushButton("All")
        no_sources = QtWidgets.QPushButton("None")
        all_sources.clicked.connect(lambda: self.sources.set_all(True))
        no_sources.clicked.connect(lambda: self.sources.set_all(False))
        form.addRow("Sources", self._add(self.sources))
        form.addRow("", row(all_sources, no_sources, QtWidgets.QWidget(), stretches=(0, 0, 1)))
        correlation = QtWidgets.QComboBox()
        for name in _parser_choices("correlation_type"):
            correlation.addItem(name, name)
        form.addRow("Correlations", self._add(combo_field("correlation_type", correlation)))
        self.stokes = ChecklistField("stokes")
        self.stokes.widget.setFlow(QtWidgets.QListView.LeftToRight)
        self.stokes.widget.setMaximumHeight(34)
        form.addRow("Stokes", self._add(self.stokes))
        self.channels = CheckedLineEdit("all; e.g. 10:20 or 400:450MHz", self._check_channels)
        form.addRow("Channels", self._add(text_field("channels", self.channels)))
        self.antennas = CheckedLineEdit("all; e.g. 1:5,W01:25,C00", self._check_antennas)
        form.addRow("Antennas", self._add(text_field("antennas", self.antennas)))
        # Frozen with autocorrelations, whose one antenna Antennas chooses: its text then stays out of the request.
        self.baselines_with = CheckedLineEdit("all; with Antennas, e.g. C01,C02", self._check_antennas)
        baselines_field = text_field("baselines_with", self.baselines_with)
        get_text = baselines_field.get
        baselines_field.get = lambda: get_text() if self.baselines_with.isEnabled() else None
        form.addRow("Baselines with", self._add(baselines_field))
        correlation.currentIndexChanged.connect(
            lambda _: self.baselines_with.setEnabled(correlation.currentData() != "auto"))
        self.exclude = CheckedLineEdit("none", self._check_antennas)
        form.addRow("Exclude", self._add(text_field("exclude_antennas", self.exclude)))
        self.time_range = CheckedLineEdit("all; e.g. 0:0.5 (h from start)", self._check_time_range)
        form.addRow("Time range", self._add(text_field("time_range", self.time_range)))

        more = CollapsibleSection("More filters", expanded=False)
        more_form = form_layout(more.body)
        for dest, label, placeholder, resolver in (
            ("uvdist_range", "UV distance", "lo:hi m, or with km", resolve_uvdist_range_arg),
            ("u_range_klambda", "u (kλ)", "lo:hi", resolve_klambda_range_arg),
            ("v_range_klambda", "v (kλ)", "lo:hi", resolve_klambda_range_arg),
            ("w_range_klambda", "w (kλ)", "lo:hi", resolve_klambda_range_arg),
            ("uvdist_range_klambda", "UV dist. (kλ)", "lo:hi", resolve_klambda_range_arg),
            ("ha_range", "Hour angle", "lo:hi h, or with deg", resolve_ha_range_arg),
            ("az_range", "Azimuth", "lo:hi deg, or with rad", resolve_deg_range_arg),
            ("el_range", "Elevation", "lo:hi deg, or with rad", resolve_deg_range_arg),
            ("pa_range", "Parallactic angle", "lo:hi deg, or with rad", resolve_deg_range_arg),
        ):
            more_form.addRow(label, self._add(text_field(dest, CheckedLineEdit(placeholder, _check_with(resolver)))))
        for dest, label, placeholder in (("every_nth_integration", "Every Nth integration", "all integrations"),
                                         ("every_nth", "Every Nth row", "all rows (can skip baselines)"),
                                         ("random_subset_n", "Random rows", "all rows"),
                                         ("random_seed", "Random seed", "needed for random rows")):
            edit = CheckedLineEdit(placeholder, lambda t: None if int(t) >= 0 else "a whole number, 0 or more")
            more_form.addRow(label, self._add(number_field(dest, edit, int)))
        form.addRow(more)
        return section

    def _pages_section(self) -> CollapsibleSection:
        """One plot per baseline, antenna, source or Stokes (T26), laid out
        on pages; the layout's fields frozen, and left out of the request,
        until a kind is chosen."""
        section = CollapsibleSection("Pages", expanded=False)
        form = form_layout(section.body)
        per = QtWidgets.QComboBox()
        per.addItem("none (one plot)", None)
        for name in _parser_choices("one_plot_per"):
            per.addItem("Stokes product" if name == "stokes" else name, name)
        per.setToolTip(PAGES_HELP["one_plot_per"])
        form.addRow("One plot per", self._add(combo_field("one_plot_per", per)))

        self.auto_grid = QtWidgets.QCheckBox("auto")
        self.auto_grid.setChecked(True)
        self.auto_grid.setToolTip(PAGES_HELP["page_grid"])
        rows, cols = QtWidgets.QSpinBox(), QtWidgets.QSpinBox()
        for box, value in ((rows, DEFAULT_PAGE_GRID[0]), (cols, DEFAULT_PAGE_GRID[1])):
            box.setRange(1, MAX_PAGE_GRID)
            box.setValue(value)
        grid_box = row(self.auto_grid, rows, QtWidgets.QLabel("×"), cols, QtWidgets.QWidget(),
                       stretches=(0, 0, 0, 0, 1))  # rows x columns: the hover says so

        def get_grid():
            if not per.currentData() or self.auto_grid.isChecked():
                return None
            return f"{rows.value()},{cols.value()}"

        def set_grid(value):
            grid = resolve_page_grid_arg(value)
            self.auto_grid.setChecked(grid is None)
            if grid is not None:
                rows.setValue(grid[0])
                cols.setValue(grid[1])
        self.fields["page_grid"] = Field("page_grid", grid_box, get_grid, set_grid, self.auto_grid.toggled)
        for box in (rows, cols):
            box.valueChanged.connect(lambda *_: self.changed.emit())
        for widget in (grid_box, rows, cols):
            widget.setToolTip(PAGES_HELP["page_grid"])
        form.addRow("Plots per page", grid_box)

        range_boxes = []
        for axis in ("x", "y"):
            box = QtWidgets.QComboBox()
            for text, value in (("default", None), ("each plot", "each"), ("all plots", "all")):  # the hover says more
                box.addItem(text, value)
                box.setItemData(box.count() - 1, RANGE_FROM_HELP[value].format(axis=axis), QtCore.Qt.ToolTipRole)
            box.setToolTip(PAGES_HELP["range_from"].format(axis=axis))
            range_field = combo_field(f"{axis}_range_from", box)
            get_range = range_field.get
            range_field.get = lambda g=get_range: g() if per.currentData() else None
            form.addRow(f"{axis.upper()} range from", self._add(range_field))
            range_boxes.append(box)

        def layout_controls(*_):  # the layout's fields: with one plot per something only
            on = per.currentData() is not None
            for widget in (self.auto_grid, *range_boxes):
                widget.setEnabled(on)
            for box in (rows, cols):
                box.setEnabled(on and not self.auto_grid.isChecked())
        per.currentIndexChanged.connect(layout_controls)
        self.auto_grid.toggled.connect(layout_controls)
        layout_controls()
        return section

    def _display_section(self) -> CollapsibleSection:
        section = CollapsibleSection("Display")
        form = form_layout(section.body)
        colorize = QtWidgets.QComboBox()
        colorize.addItem("one color", None)
        for name in _parser_choices("colorize_by"):
            colorize.addItem(name, name)
        form.addRow("Color by", self._add(combo_field("colorize_by", colorize)))
        color = QtWidgets.QComboBox()
        color.setEditable(True)
        color.addItems(["tab:blue", "tab:orange", "tab:green", "tab:red", "k", "0.3"])
        self.fields["color"] = Field("color", color, lambda: color.currentText().strip() or "tab:blue",
                                     lambda v: color.setCurrentText(str(v)), color.currentTextChanged)
        form.addRow("Color", color)
        style = QtWidgets.QComboBox()
        style.addItem("points", "points")
        style.addItem("density (samples per pixel)", "density")
        style.setToolTip("--style: points marks where any sample lands; density colors each pixel by how many land "
                         "there, hues mixed where categories share a pixel")
        form.addRow("Style", self._add(combo_field("style", style)))
        density_scale = QtWidgets.QComboBox()
        density_scale.addItem("log", "log")
        density_scale.addItem("histogram-equalized", "histogram")
        density_scale.setToolTip("--density-scale: log rises with log(count) to the scale's top; histogram-equalized "
                                 "follows each pixel's rank among the occupied pixels")
        form.addRow("Density scale", self._add(combo_field("density_scale", density_scale)))
        density_top = QtWidgets.QDoubleSpinBox()
        density_top.setRange(50.0, 100.0)
        density_top.setDecimals(1)
        density_top.setSuffix(" th percentile")
        density_top.setValue(build_arg_parser().get_default("density_top"))
        density_top.setToolTip("--density-top: the log scale's top among the occupied pixels' counts (100: the "
                               "most crowded pixel)")
        self.fields["density_top"] = Field("density_top", density_top, density_top.value, density_top.setValue,
                                           density_top.valueChanged)
        form.addRow("Scale top", density_top)

        def density_controls(*_):  # density plots only; the top, for the log scale only
            density = style.currentData() == "density"
            density_scale.setEnabled(density)
            density_top.setEnabled(density and density_scale.currentData() == "log")
        style.currentIndexChanged.connect(density_controls)
        density_scale.currentIndexChanged.connect(density_controls)
        density_controls()
        self.auto_size = QtWidgets.QCheckBox("auto")
        self.auto_size.setChecked(True)
        size = QtWidgets.QDoubleSpinBox()
        size.setRange(0.01, 400.0)
        size.setDecimals(3)
        size.setValue(1.0)
        size.setEnabled(False)
        self.auto_size.toggled.connect(lambda on: size.setEnabled(not on))
        box = row(self.auto_size, size, stretches=(0, 1))

        def set_size(v):
            self.auto_size.setChecked(v is None)
            if v is not None:
                size.setValue(float(v))
        self.fields["point_size"] = Field("point_size", box, lambda: None if self.auto_size.isChecked() else size.value(),
                                          set_size, self.auto_size.toggled)
        size.valueChanged.connect(lambda *_: self.changed.emit())
        form.addRow("Marker size", box)
        form.addRow("", self._add(flag_field("show_flagged", QtWidgets.QCheckBox("show flagged samples (light-coral crosses)"))))
        combine = QtWidgets.QComboBox()
        combine.addItem("flagged if any Stokes is", "any")
        combine.addItem("flagged only if all Stokes are", "all")
        combine.setToolTip("A point combining its visibility's selected Stokes (e.g. u-v): --combine-flags; a "
                           "point of one Stokes is that visibility's own flag either way")
        form.addRow("Flag rule", self._add(combo_field("combine_flags", combine)))
        form.addRow("", self._add(flag_field("mirror", QtWidgets.QCheckBox("mirror (-x, -y), e.g. uv coverage"))))
        panel_font = QtWidgets.QComboBox()
        for name in _parser_choices("panel_font"):
            panel_font.addItem(PANEL_FONTS[name][0], name)
        select_data(panel_font, build_arg_parser().get_default("panel_font"))
        form.addRow("Panel font", self._add(combo_field("panel_font", panel_font)))
        time_format = QtWidgets.QComboBox()
        for text, name in (("1/16:44:00  (AIPS, dd/hh:mm:ss)", "dd/hh:mm:ss"),
                           ("01:16:44:00  (dd:hh:mm:ss)", "dd:hh:mm:ss"),
                           ("16:44:00  (time of day)", "hh:mm:ss"), ("2021-07-25 16:44:00  (ISO)", "iso")):
            time_format.addItem(text, name)
        select_data(time_format, build_arg_parser().get_default("time_format"))
        time_format.setToolTip("How clock times are written on the axes (--time-format); days count from the "
                               "file's reference date")
        form.addRow("Time format", self._add(combo_field("time_format", time_format)))
        plot_theme = QtWidgets.QComboBox()
        for name in _parser_choices("plot_theme"):
            plot_theme.addItem(name, name)
        select_data(plot_theme, build_arg_parser().get_default("plot_theme"))
        plot_theme.setToolTip("The plot's colors, on screen and saved (--plot-theme); the View menu sets the "
                              "window's own")
        form.addRow("Plot theme", self._add(combo_field("plot_theme", plot_theme)))
        return section

    def _performance_section(self) -> CollapsibleSection:
        section = CollapsibleSection("Performance, cache and records", expanded=False)
        form = form_layout(section.body)
        threads = QtWidgets.QSpinBox()
        threads.setRange(1, max(1, QtCore.QThread.idealThreadCount()))
        threads.setValue(build_arg_parser().get_default("threads"))
        self.fields["threads"] = Field("threads", threads, threads.value, lambda v: threads.setValue(int(v)),
                                       threads.valueChanged)
        form.addRow("Threads", threads)
        self.cache_edit = QtWidgets.QLineEdit()
        self.cache_edit.setPlaceholderText("no cache (ranges found each time)")
        self.cache_button = QtWidgets.QPushButton("Choose…")
        self._add(text_field("cache_dir", self.cache_edit))
        form.addRow("Range cache", row(self.cache_edit, self.cache_button, stretches=(1, 0)))
        self.provenance_edit = QtWidgets.QLineEdit()
        default = build_arg_parser().get_default("provenance_dir")
        self._add(text_field("provenance_dir", self.provenance_edit, default=default))
        self.provenance_edit.setPlaceholderText(f"./{default}, in the folder visplot started in")
        self.provenance_button = QtWidgets.QPushButton("Choose…")
        form.addRow("Records", row(self.provenance_edit, self.provenance_button, stretches=(1, 0)))
        self.session_hint = hint("")
        form.addRow(self.session_hint)
        return section

    def show_session(self, session_id: str, log_path) -> None:
        """Name the GUI session and its log under the Records field."""
        self.session_hint.setText(f"Each plot, CSV and export is recorded in the Records folder. This session: "
                                  f"{session_id}; every message is kept in {log_path}")

    # ---- the plot choice ----------------------------------------------------

    def _plots(self) -> str:
        kind = self.kind.currentData()
        if kind != GENERIC:
            return kind
        x = self.x.quantity_name()
        lower = self.y2.quantity_name()
        return f"{self.y.quantity_name()}-vs-{x}" + (f"+{lower}-vs-{x}" if lower else "")

    def _set_plots(self, value: str) -> None:
        if value in PRESETS or value in TABLE_PLOTS:
            select_data(self.kind, value)
            return
        first, _, second = value.partition("+")
        y, x = first.split("-vs-", 1)
        select_data(self.kind, GENERIC)
        select_data(self.y.quantity, y)
        select_data(self.x.quantity, x)
        select_data(self.y2.quantity, second.split("-vs-", 1)[0] if second else None)

    def _has_lower(self) -> bool:
        """Whether the plot has a lower plot (a stack, T26): one chosen, or a
        two-plot preset's second (az-el-range's azimuth)."""
        preset = PRESETS.get(self.kind.currentData())
        if preset is not None:
            return len(preset) == 2
        return self.kind.currentData() == GENERIC and self.y2.quantity_name() is not None

    def _kind_changed(self) -> None:
        generic = self.kind.currentData() == GENERIC
        table = self.kind.currentData() in TABLE_PLOTS
        preset = PRESETS.get(self.kind.currentData())
        for controls in (self.x, self.y, self.y2):
            controls.quantity.setEnabled(generic)
            on = not table and (controls is not self.y2 or self._has_lower())
            for w in (controls.unit, controls.scale, controls.range_mode, controls.fixed_range):
                w.setEnabled(on and (w is not controls.unit or controls.unit.count() > 1))
        if preset:  # show the preset's own quantities, for their units
            select_data(self.y.quantity, preset[0].y)
            select_data(self.x.quantity, preset[0].x)
            self.y2.quantity.blockSignals(True)
            select_data(self.y2.quantity, preset[1].y if len(preset) == 2 else None)
            self.y2.quantity.blockSignals(False)
            self.y2.fill_units()
        self.changed.emit()

    def _swap_axes(self) -> None:
        values = {k: self.fields[k].get() for k in ("x_unit", "y_unit", "x_scale", "y_scale", "x_range", "y_range",
                                                     "x_range_mode", "y_range_mode")}
        qx, qy = self.x.quantity_name(), self.y.quantity_name()
        select_data(self.x.quantity, qy)
        select_data(self.y.quantity, qx)
        for axis, other in (("x", "y"), ("y", "x")):
            for key in ("unit", "scale", "range", "range_mode"):
                self.fields[f"{axis}_{key}"].set(values[f"{other}_{key}"])

    # ---- opening and indexing the file ---------------------------------------

    def show_progress(self, text: str | None, fraction: float | None = None, stoppable: bool = False) -> None:
        """The bar under the File field: hidden for `text` None; busy for
        `fraction` None, else that fraction done; Stop shown when `stoppable`."""
        self.progress_row.setVisible(text is not None)
        self.progress_text.setVisible(text is not None)
        if text is None:
            return
        self.progress_text.setText(text)
        if fraction is None:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 1000)
            self.progress.setValue(round(1000 * min(max(fraction, 0.0), 1.0)))
        self.stop_build_button.setVisible(stoppable)
        self.stop_build_button.setEnabled(stoppable)

    def show_missing_index(self, text: str | None) -> None:
        """The note and Build index button of a file without its index (hidden for None)."""
        self.index_row.setVisible(text is not None)
        self.index_note.set_text(text or "")

    # ---- file-dependent checks and lists -------------------------------------

    def unit_context(self) -> QuantityContext:
        return QuantityContext(time_reference_jd=self._first_jd or 0.0, bunit=self._bunit)

    def set_file(self, opened, summary) -> None:
        """Fill what depends on the file: the summary, sources with their row
        counts, Stokes, the visibility units (BUNIT) and the file-based checks."""
        self._bunit = opened.bunit
        self._antennas = opened.antennas
        self._chan_freqs_hz = opened.index.chan_freqs_hz
        self._stokes_labels = opened.index.stokes_labels
        self._first_jd = float(opened.index.jd.min())
        self.summary.setText("<table>" + "".join(f"<tr><td><b>{k}</b>&nbsp;&nbsp;</td><td>{v}</td></tr>"
                                                 for k, v in summary.lines()) + "</table>")
        self.sources.fill([(name, f"{name}  ({rows:,} rows)") for _, name, rows in summary.sources])
        self.stokes.fill([(label, label) for label in summary.stokes_labels])
        zone = observatory_time_zone(opened.telescope)
        self.zone.setPlaceholderText(f"{zone} (from TELESCOP {opened.telescope})" if zone
                                     else f"none known for TELESCOP {opened.telescope!r}: give one for local time")
        self.x.fill_units()
        self.y.fill_units()
        self.y2.fill_units()
        for edit in (self.channels, self.antennas, self.baselines_with, self.exclude, self.time_range):
            edit.set_check(edit._check)

    def _check_channels(self, text: str) -> str | None:
        if self._chan_freqs_hz is not None:
            resolve_channels_arg(text, self._chan_freqs_hz)
        return None

    def _check_antennas(self, text: str) -> str | None:
        if self._antennas is not None:
            resolve_antennas_arg(text, self._antennas)
        return None

    def _check_time_range(self, text: str) -> str | None:
        resolve_time_range_arg(text, self._first_jd or 0.0)
        return None

    @staticmethod
    def _check_zone(text: str) -> str | None:
        try:
            ZoneInfo(text)
        except (ZoneInfoNotFoundError, ValueError):
            return f"unknown time zone {text!r} (an IANA name, e.g. Asia/Kolkata)"
        return None

    # ---- the request ----------------------------------------------------------

    def invalid_fields(self) -> list[str]:
        """Options whose text fails its check (red-framed)."""
        return [dest for dest, f in self.fields.items()
                if isinstance(f.widget, CheckedLineEdit) and not f.widget.valid]

    def request(self, **actions) -> PlotRequest:
        """The form as a request; `actions` sets the ACTION_OPTIONS (e.g. a
        save's output_dir)."""
        values = {dest: f.get() for dest, f in self.fields.items()}
        if values["plots"] in TABLE_PLOTS:  # drawn from the file's tables: no axis units
            values["x_unit"] = values["y_unit"] = None
        values.update(actions)
        fits_path = values.pop("fits_path") or ""
        plots = values.pop("plots")
        return PlotRequest(fits_path, plots, **values)

    def load(self, request: PlotRequest) -> None:
        """Show `request` in the fields (e.g. from the run history)."""
        values = request.as_dict()
        self.fields["plots"].set(values["plots"])
        for axis in ("x", "y"):
            getattr(self, axis).fill_units()
        for dest, f in self.fields.items():
            if dest != "plots":
                f.set(values[dest])
