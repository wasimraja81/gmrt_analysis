"""Axis scales for streamed plots: linear, log, symlog, asinh.

Samples are binned in the scale's own coordinate (log10 of the value for a
log axis, and so on), so pixels are equal-width on screen. The transform used
for binning is matplotlib's own for that scale, so a pixel lines up with the
axis the figure draws with `ax.set_xscale`/`set_yscale`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib import scale as mscale

SCALE_NAMES = ("linear", "log", "symlog", "asinh")


@dataclass(frozen=True)
class AxisScale:
    name: str = "linear"
    linear_width: float = 1.0  # symlog's linthresh, asinh's linear_width

    def __post_init__(self):
        if self.name not in SCALE_NAMES:
            raise ValueError(f"unknown axis scale {self.name!r}; expected one of {SCALE_NAMES}")
        if self.linear_width <= 0:
            raise ValueError(f"linear width must be positive, got {self.linear_width}")

    @property
    def is_linear(self) -> bool:
        return self.name == "linear"

    def mpl_kwargs(self) -> dict:
        """Arguments for `ax.set_xscale`/`set_yscale`."""
        if self.name == "symlog":
            return {"value": "symlog", "linthresh": self.linear_width}
        if self.name == "asinh":
            return {"value": "asinh", "linear_width": self.linear_width}
        return {"value": self.name}

    def _transform(self):
        kwargs = self.mpl_kwargs()
        name = kwargs.pop("value")
        return mscale.scale_factory(name, None, **kwargs).get_transform()

    def forward(self, values: np.ndarray) -> np.ndarray:
        """Values in the scale's coordinate. Only defined where `valid`."""
        if self.is_linear:
            return values
        return self._transform().transform_non_affine(np.asarray(values, dtype=np.float64))

    def inverse(self, values: np.ndarray) -> np.ndarray:
        if self.is_linear:
            return values
        return self._transform().inverted().transform_non_affine(np.asarray(values, dtype=np.float64))

    def valid(self, values: np.ndarray) -> np.ndarray | None:
        """Mask of values the scale can show (log: positive only), or None if all can."""
        if self.name == "log":
            return values > 0
        return None
