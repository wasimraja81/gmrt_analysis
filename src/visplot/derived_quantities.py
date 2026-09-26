"""Named scientific quantities derived from a `VisibilityBlock`, each
broadcast to the block's own full cell shape (`block.data.shape`) -- so any
two can be paired for a `scatter_xy` plot ("amp vs freq", "real vs u", "u vs
v", "w vs time", and so on) without the caller hand-rolling broadcasting
for every combination.

Two kinds of quantity, by what they vary over:
- Cell-shaped already (real, imag, amp, phase_deg): one value per
  (row, channel, Stokes, ...) cell, straight from `block.data`.
- Row- or channel-only (time_h, u/v/w in seconds or kilo-wavelengths,
  uvdist, freq_mhz, stokes): one value per row or per channel, broadcast
  (not tiled in memory -- `np.broadcast_to` is a view) across every other
  axis.
"""

from __future__ import annotations

import numpy as np

from data_io.visibility_data import VisibilityBlock

SPEED_OF_LIGHT_M_PER_S = 299_792_458.0


def _broadcast_row_quantity(per_row: np.ndarray, block: VisibilityBlock) -> np.ndarray:
    shape = block.data.shape
    reshaped = per_row.reshape((shape[0],) + (1,) * (len(shape) - 1))
    return np.broadcast_to(reshaped, shape)


def _broadcast_channel_quantity(per_channel: np.ndarray, block: VisibilityBlock) -> np.ndarray:
    if "FREQ" not in block.axis_types:
        raise ValueError(f"this block has no FREQ axis (axis_types={block.axis_types})")
    freq_axis = block.axis_types.index("FREQ")
    shape = block.data.shape
    reshape = [1] * len(shape)
    reshape[freq_axis + 1] = shape[freq_axis + 1]  # +1: axis 0 is the row axis, not in axis_types
    return np.broadcast_to(per_channel.reshape(reshape), shape)


def _broadcast_stokes_labels(block: VisibilityBlock) -> np.ndarray:
    if "STOKES" not in block.axis_types or block.stokes_labels is None:
        raise ValueError(f"this block has no STOKES axis (axis_types={block.axis_types})")
    stokes_axis = block.axis_types.index("STOKES")
    shape = block.data.shape
    reshape = [1] * len(shape)
    reshape[stokes_axis + 1] = shape[stokes_axis + 1]
    labels = np.array(block.stokes_labels, dtype=object).reshape(reshape)
    return np.broadcast_to(labels, shape)


_UVW_SEC = {"u_sec": "uu_sec", "v_sec": "vv_sec", "w_sec": "ww_sec"}
_UVW_KLAMBDA = {"u_klambda": "uu_sec", "v_klambda": "vv_sec", "w_klambda": "ww_sec"}


def compute_quantity(name: str, block: VisibilityBlock) -> np.ndarray:
    """A named quantity, broadcast to `block.data`'s full shape. Raises
    `ValueError` for an unknown name, or one that needs an axis this block
    doesn't have (e.g. a kilo-wavelength quantity without a FREQ axis)."""
    if name == "real":
        return block.data.real
    if name == "imag":
        return block.data.imag
    if name == "amp":
        return np.abs(block.data)
    if name == "phase_deg":
        return np.degrees(np.angle(block.data))

    if name == "time_h":
        return _broadcast_row_quantity((block.jd - block.jd.min()) * 24.0, block)
    if name in _UVW_SEC:
        return _broadcast_row_quantity(getattr(block, _UVW_SEC[name]), block)
    if name == "uvdist_m":
        u_sec = _broadcast_row_quantity(block.uu_sec, block)
        v_sec = _broadcast_row_quantity(block.vv_sec, block)
        return np.hypot(u_sec, v_sec) * SPEED_OF_LIGHT_M_PER_S

    if name in _UVW_KLAMBDA or name == "uvdist_klambda":
        if block.chan_freqs_hz is None:
            raise ValueError(f"{name!r} needs channel frequencies, which this block does not have")
        freq_hz = _broadcast_channel_quantity(block.chan_freqs_hz, block)
        if name == "uvdist_klambda":
            u_sec = _broadcast_row_quantity(block.uu_sec, block)
            v_sec = _broadcast_row_quantity(block.vv_sec, block)
            return np.hypot(u_sec, v_sec) * freq_hz / 1e3
        return _broadcast_row_quantity(getattr(block, _UVW_KLAMBDA[name]), block) * freq_hz / 1e3

    if name == "freq_mhz":
        if block.chan_freqs_hz is None:
            raise ValueError("'freq_mhz' needs channel frequencies, which this block does not have")
        return _broadcast_channel_quantity(block.chan_freqs_hz / 1e6, block)

    if name == "stokes":
        return _broadcast_stokes_labels(block)

    raise ValueError(f"unknown quantity {name!r}")
