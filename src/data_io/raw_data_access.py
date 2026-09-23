"""Safe access to raw GMRT data.

Two jobs: (1) open the raw data read-only, with no code path that could ever
request a writable open, and (2) guard every write of a derived product
against accidentally landing on top of a raw input file. See
docs/user/GWB_USER_GUIDE.md#reading-raw-data-safely for what this means from
the operator's side -- the raw file is never modified, and the pipeline
refuses to run rather than risk it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Union

import numpy as np
from astropy.io import fits

PathLike = Union[str, Path]


class RawDataProtectionError(Exception):
    """Raised when an operation would read raw data unsafely or write over it."""


def open_fits_readonly(path: PathLike) -> fits.HDUList:
    """Open a FITS file for reading only.

    There is no ``mode`` parameter here on purpose -- this function cannot be
    asked to open a file writable, so misuse is structurally prevented rather
    than merely checked for.
    """
    return fits.open(str(path), mode="readonly", memmap=True)


def open_raw_memmap(path: PathLike, dtype, shape: tuple[int, ...], offset: int = 0) -> np.memmap:
    """Memory-map a raw binary data block for reading only.

    Same reasoning as ``open_fits_readonly``: ``mode`` is hardcoded to ``'r'``,
    not exposed as a parameter.
    """
    return np.memmap(str(path), dtype=dtype, mode="r", offset=offset, shape=shape)


def guard_output_path(output_path: PathLike, raw_input_paths: Union[PathLike, Iterable[PathLike]]) -> None:
    """Raise RawDataProtectionError if output_path would overwrite a raw input.

    Call this immediately before writing any derived product, passing every
    raw input path the current stage read from. Both sides are resolved
    (symlinks, relative segments, '..') before comparing, so a path that only
    looks different from the raw input but points at the same file is still
    caught.
    """
    if isinstance(raw_input_paths, (str, Path)):
        raw_input_paths = [raw_input_paths]

    resolved_output = Path(output_path).resolve()
    for raw_path in raw_input_paths:
        resolved_raw = Path(raw_path).resolve()
        if resolved_output == resolved_raw:
            raise RawDataProtectionError(
                f"refusing to write output to {resolved_output}: "
                f"this is the raw input path ({resolved_raw})"
            )
