"""Source listing: a table figure of every source in the file, sorted by
source id for a deterministic, reproducible row order (a dict's own
iteration order is not guaranteed to match the file's own id order).
"""

from __future__ import annotations

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from data_io.source_table import Source
from visplot.plot_theme import DEFAULT_PLOT_THEME, plot_theme

_COLUMNS = ["ID", "Name", "RA (epoch, deg)", "Dec (epoch, deg)", "CALCODE", "Flux I (Jy)"]


def source_listing(source_table: dict[int, Source], theme: str = DEFAULT_PLOT_THEME) -> Figure:
    """A table of every source's id, name, epoch position, CALCODE, and
    first-IF Stokes-I flux, as recorded in the file -- not a source's
    apparent position or a judgement about calibrator suitability. `theme`
    is the --plot-theme (`visplot.plot_theme`)."""
    colors = plot_theme(theme)
    rows = []
    for sid in sorted(source_table):
        source = source_table[sid]
        flux_i = f"{source.flux_i_jy[0]:.3g}" if source.flux_i_jy else "--"
        rows.append([
            str(source.id), source.name,
            f"{source.ra_epoch_deg:.6f}", f"{source.dec_epoch_deg:.6f}",
            source.calcode or "--", flux_i,
        ])

    # A bare Figure with its own Agg canvas (no pyplot): made on any thread, saved with
    # savefig, or embedded in a Qt window.
    fig = Figure(figsize=(9, 0.4 * max(len(rows), 1) + 1), facecolor=colors.figure_face)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    ax.axis("off")
    if rows:
        table = ax.table(cellText=rows, colLabels=_COLUMNS, cellLoc="center", bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.auto_set_column_width(col=list(range(len(_COLUMNS))))
        for cell in table.get_celld().values():
            cell.set_facecolor(colors.table_face)
            cell.set_edgecolor(colors.table_edge)
            cell.get_text().set_color(colors.text)
    else:
        ax.text(0.5, 0.5, "No sources in this file's AIPS SU table", ha="center", va="center", color=colors.text)
    ax.set_title("Source listing", pad=20, color=colors.text)
    return fig
