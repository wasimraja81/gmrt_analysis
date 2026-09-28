"""Every quantity a plot can show, in one registry.

A quantity is one `Quantity` entry: its name, display name, unit, whether it
needs visibility data read from disk, whether it is a category, and one
function evaluating it on a `VisibilityBlock` chunk. Adding a quantity is
adding one entry.

Evaluated values keep a size-1 axis wherever the quantity does not vary
(a row quantity is shaped (n_rows, 1, 1, ...); frequency is 1 everywhere but
the FREQ axis), so pairing two quantities with numpy broadcasting yields only
the shape the pair needs: hour angle vs time stays one value per row, while
amplitude vs frequency covers every sample.

Categories (Stokes, source) evaluate to integer codes; `category_label`
turns a code into its label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from data_io.astrometry import altaz_deg, hour_angle_hours, parallactic_angle_deg
from data_io.visibility_data import VisibilityBlock

SPEED_OF_LIGHT_M_PER_S = 299_792_458.0

# Unit marker: the file's own BUNIT header keyword (visibility amplitude units
# depend on whether flux calibration has been applied).
UNIT_FROM_BUNIT = "<BUNIT>"


@dataclass(frozen=True)
class QuantityContext:
    """What evaluating a quantity needs beyond the chunk itself, fixed for a
    whole plot: the time origin (so every chunk shares one), the file's
    BUNIT, the selected Stokes labels, and per-source coordinates and names
    indexed by source id."""

    time_reference_jd: float
    bunit: str | None = None
    stokes_labels: tuple[str, ...] = ()
    source_names: dict[int, str] | None = None
    source_ra_deg: np.ndarray | None = None  # indexed by source id
    source_dec_deg: np.ndarray | None = None
    array_location: object | None = None  # astropy EarthLocation


def context_from_source_table(time_reference_jd, source_table, array_location, bunit=None, stokes_labels=()):
    """A `QuantityContext` with per-source coordinate lookup arrays built once
    from `read_source_table`'s dict (source id -> Source)."""
    max_id = max(source_table) if source_table else 0
    ra = np.full(max_id + 1, np.nan)
    dec = np.full(max_id + 1, np.nan)
    for sid, src in source_table.items():
        ra[sid] = src.ra_apparent_deg
        dec[sid] = src.dec_apparent_deg
    return QuantityContext(
        time_reference_jd=time_reference_jd,
        bunit=bunit,
        stokes_labels=tuple(stokes_labels or ()),
        source_names={sid: src.name for sid, src in source_table.items()},
        source_ra_deg=ra,
        source_dec_deg=dec,
        array_location=array_location,
    )


@dataclass(frozen=True)
class Quantity:
    name: str
    display_name: str
    unit: str | None  # None: no unit; UNIT_FROM_BUNIT: the file's BUNIT
    needs_data: bool
    categorical: bool
    evaluate: Callable[[VisibilityBlock, QuantityContext], np.ndarray]


def _row(values, block: VisibilityBlock) -> np.ndarray:
    values = np.asarray(values)
    return values.reshape((len(values),) + (1,) * len(block.axis_types))


def _along(axis_type: str, values, block: VisibilityBlock) -> np.ndarray:
    if axis_type not in block.axis_types:
        raise ValueError(f"this data has no {axis_type} axis (axes: {block.axis_types})")
    shape = [1] * (1 + len(block.axis_types))
    shape[1 + block.axis_types.index(axis_type)] = len(values)
    return np.asarray(values).reshape(shape)


def _freq_hz(block: VisibilityBlock) -> np.ndarray:
    if block.chan_freqs_hz is None:
        raise ValueError("this data has no channel frequencies (no FREQ axis)")
    return _along("FREQ", block.chan_freqs_hz, block)


def _source_coords(block: VisibilityBlock, ctx: QuantityContext) -> tuple[np.ndarray, np.ndarray]:
    if ctx.source_ra_deg is None or ctx.array_location is None:
        raise ValueError("observing-geometry quantities need the source table and the array location")
    return ctx.source_ra_deg[block.source_id], ctx.source_dec_deg[block.source_id]


def _hour_angle(block, ctx):
    ra, _ = _source_coords(block, ctx)
    return _row(hour_angle_hours(block.jd, ra, ctx.array_location), block)


def _azimuth(block, ctx):
    ra, dec = _source_coords(block, ctx)
    return _row(altaz_deg(block.jd, ra, dec, ctx.array_location)[0], block)


def _elevation(block, ctx):
    ra, dec = _source_coords(block, ctx)
    return _row(altaz_deg(block.jd, ra, dec, ctx.array_location)[1], block)


def _parallactic_angle(block, ctx):
    ra, dec = _source_coords(block, ctx)
    return _row(parallactic_angle_deg(block.jd, ra, dec, ctx.array_location), block)


def _uvdist_sec(block):
    return np.hypot(block.uu_sec.astype(np.float64), block.vv_sec.astype(np.float64))


_ALL = [
    # visibility data: stored at the file's float32; amplitude and phase are
    # computed in float64, so a sample on a pixel boundary bins the same way
    # whatever the storage precision
    Quantity("real", "Real", UNIT_FROM_BUNIT, True, False, lambda b, c: b.data.real),
    Quantity("imag", "Imag", UNIT_FROM_BUNIT, True, False, lambda b, c: b.data.imag),
    Quantity("amp", "Amplitude", UNIT_FROM_BUNIT, True, False, lambda b, c: np.abs(b.data.astype(np.complex128))),
    Quantity("phase_deg", "Phase", "deg", True, False,
             lambda b, c: np.degrees(np.angle(b.data.astype(np.complex128)))),
    # per row
    Quantity("time_h", "Time", "h", False, False,
             lambda b, c: _row((b.jd - c.time_reference_jd) * 24.0, b)),
    Quantity("u_sec", "U", "s", False, False, lambda b, c: _row(b.uu_sec, b)),
    Quantity("v_sec", "V", "s", False, False, lambda b, c: _row(b.vv_sec, b)),
    Quantity("w_sec", "W", "s", False, False, lambda b, c: _row(b.ww_sec, b)),
    Quantity("uvdist_m", "UV distance", "m", False, False,
             lambda b, c: _row(_uvdist_sec(b) * SPEED_OF_LIGHT_M_PER_S, b)),
    Quantity("ha_h", "Hour angle", "h", False, False, _hour_angle),
    Quantity("az_deg", "Azimuth", "deg", False, False, _azimuth),
    Quantity("el_deg", "Elevation", "deg", False, False, _elevation),
    Quantity("pa_deg", "Parallactic angle", "deg", False, False, _parallactic_angle),
    # per row and channel
    Quantity("u_klambda", "U", "kλ", False, False, lambda b, c: _row(b.uu_sec, b) * _freq_hz(b) / 1e3),
    Quantity("v_klambda", "V", "kλ", False, False, lambda b, c: _row(b.vv_sec, b) * _freq_hz(b) / 1e3),
    Quantity("w_klambda", "W", "kλ", False, False, lambda b, c: _row(b.ww_sec, b) * _freq_hz(b) / 1e3),
    Quantity("uvdist_klambda", "UV distance", "kλ", False, False,
             lambda b, c: _row(_uvdist_sec(b), b) * _freq_hz(b) / 1e3),
    # per channel
    Quantity("freq_mhz", "Frequency", "MHz", False, False, lambda b, c: _freq_hz(b) / 1e6),
    # categories
    Quantity("stokes", "Stokes", None, False, True,
             lambda b, c: _along("STOKES", np.arange(len(b.axis_indices.get("STOKES", ()))), b)),
    Quantity("source", "Source", None, False, True, lambda b, c: _row(b.source_id.astype(np.int64), b)),
]

QUANTITIES: dict[str, Quantity] = {q.name: q for q in _ALL}


def quantity_label(name: str, ctx: QuantityContext | None = None) -> str:
    """Display name plus unit, for an axis label, e.g. "Frequency (MHz)",
    "Amplitude (UNCALIB)" (unit from the file's BUNIT; none shown if the
    file has none)."""
    q = QUANTITIES[name]
    unit = (ctx.bunit if ctx is not None else None) if q.unit == UNIT_FROM_BUNIT else q.unit
    return f"{q.display_name} ({unit})" if unit else q.display_name


def category_label(name: str, code: int, ctx: QuantityContext) -> str:
    if name == "stokes":
        return ctx.stokes_labels[code] if code < len(ctx.stokes_labels) else str(code)
    if name == "source":
        return (ctx.source_names or {}).get(int(code), str(code))
    raise ValueError(f"{name!r} is not a category")
