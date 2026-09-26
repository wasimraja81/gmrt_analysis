"""Resolving a channel range spec into concrete channel indices -- either a
plain index range ("10:20", no unit) or a frequency band with an explicit
unit ("300:310MHz"), resolved against a specific file's own channel
frequencies. Not part of `range_spec` itself, since that module is
quantity-agnostic and has no notion of "a file's channels" to resolve
against.
"""

from __future__ import annotations

import numpy as np
import astropy.units as u

from visplot.range_spec import expand_int_ranges, parse_quantity_range, split_trailing_unit


def resolve_channel_selection(spec: str, chan_freqs_hz: np.ndarray) -> list[int]:
    """Which channel indices `spec` selects.

    No unit suffix ("10:20,30"): plain 0-based channel indices, via
    `expand_int_ranges` -- raises if any index is out of range for
    `chan_freqs_hz`.

    A frequency unit suffix ("300:310MHz"): every channel whose own
    frequency falls within the requested band(s) -- a direct, per-channel
    comparison against `chan_freqs_hz`, not a nearest-endpoint match (a
    channel's frequency need not exactly hit either boundary) and not a
    position-based lookup of any kind, so it carries no dependence on
    channel order, spacing, or monotonicity: membership in the requested
    band is a fact about that channel's own recorded frequency, never
    inferred from where it sits in the array."""
    chan_freqs_hz = np.asarray(chan_freqs_hz)
    terms = [t.strip() for t in spec.split(",") if t.strip()]
    has_unit = any(split_trailing_unit(term)[1] is not None for term in terms)

    if not has_unit:
        indices = expand_int_ranges(spec)
        n = len(chan_freqs_hz)
        out_of_range = [i for i in indices if not (0 <= i < n)]
        if out_of_range:
            raise ValueError(f"channel index out of range for {n} channels: {out_of_range}")
        return indices

    mask = np.zeros(len(chan_freqs_hz), dtype=bool)
    for lo_hz, hi_hz in parse_quantity_range(spec, u.Hz):
        lo_hz, hi_hz = sorted((lo_hz, hi_hz))
        mask |= (chan_freqs_hz >= lo_hz) & (chan_freqs_hz <= hi_hz)
    return sorted(np.where(mask)[0].tolist())
