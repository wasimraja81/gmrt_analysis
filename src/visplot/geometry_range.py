"""Observing-geometry range plots: hour angle, azimuth/elevation, and
parallactic angle over the course of a selection, one series per source.

Each function takes already-computed per-row geometry values (from
`data_io.astrometry`) plus a source label per row -- not a fits path, an
index, or a selection -- consistent with every other function in this
package. A caller assembles those from `select_rows`/`read_source_table`/
the `astrometry` functions and passes the arrays in.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def _scatter_by_source(ax, hours: np.ndarray, values: np.ndarray, source_labels: np.ndarray, ylabel: str, show_legend: bool):
    labels = np.asarray(source_labels)
    for name in sorted(set(labels.tolist())):
        mask = labels == name
        ax.scatter(hours[mask], values[mask], s=8, alpha=0.7, label=str(name))
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if show_legend and len(set(labels.tolist())) > 1:
        ax.legend(fontsize=8, markerscale=2, loc="best")


def _hours_from_start(jd: np.ndarray) -> np.ndarray:
    jd = np.asarray(jd, dtype=float)
    return (jd - jd.min()) * 24.0


def hour_angle_range(jd: np.ndarray, ha_hours: np.ndarray, source_labels) -> Figure:
    """Hour angle over the selection, one series per source. A dashed line
    at HA=0 marks transit."""
    hours = _hours_from_start(jd)
    fig, ax = plt.subplots(figsize=(9, 5))
    _scatter_by_source(ax, hours, np.asarray(ha_hours), source_labels, "Hour angle (h)", show_legend=True)
    ax.axhline(0.0, color="0.5", lw=0.8, ls="--")
    ax.set_ylim(-12, 12)
    ax.set_xlabel("Time from start of selection (h)")
    ax.set_title("Hour angle range")
    return fig


def parallactic_angle_range(jd: np.ndarray, pa_deg: np.ndarray, source_labels) -> Figure:
    """Parallactic angle over the selection, one series per source -- the
    range a source's parallactic angle covers is what determines whether it
    is useful as a polarization-angle calibrator."""
    hours = _hours_from_start(jd)
    fig, ax = plt.subplots(figsize=(9, 5))
    _scatter_by_source(ax, hours, np.asarray(pa_deg), source_labels, "Parallactic angle (deg)", show_legend=True)
    ax.set_ylim(-180, 180)
    ax.set_xlabel("Time from start of selection (h)")
    ax.set_title("Parallactic angle range")
    return fig


def az_el_range(jd: np.ndarray, az_deg: np.ndarray, el_deg: np.ndarray, source_labels) -> Figure:
    """Elevation and azimuth over the selection, one series per source, as
    two panels sharing a time axis -- elevation on top (a dashed line at
    el=0 marks the horizon), azimuth below."""
    hours = _hours_from_start(jd)
    fig, (ax_el, ax_az) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    _scatter_by_source(ax_el, hours, np.asarray(el_deg), source_labels, "Elevation (deg)", show_legend=True)
    ax_el.axhline(0.0, color="0.5", lw=0.8, ls="--")
    ax_el.set_ylim(-90, 90)

    _scatter_by_source(ax_az, hours, np.asarray(az_deg), source_labels, "Azimuth (deg)", show_legend=False)
    ax_az.set_ylim(0, 360)
    ax_az.set_xlabel("Time from start of selection (h)")

    fig.suptitle("Az/El range")
    return fig
