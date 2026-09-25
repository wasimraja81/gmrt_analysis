"""Efficient reading of random-groups UVFITS parameter data.

A random-groups UVFITS file stores, per row, a small block of "group
parameters" (baseline, time, source, UV coordinates, ...) immediately
followed by a much larger visibility data block (all channels and
polarizations for that row). Reading parameters through astropy's
high-level ``GroupData.par()`` forces much more of the file into memory
than the parameters alone require -- confirmed directly: a call that only
needed a few megabytes of parameter data pulled tens of gigabytes into
resident memory on the real 389GB GWB file. This module reads only the
parameter columns, via a raw memmap with the file's own per-row stride,
so the (much larger) visibility data is never touched.

Correlation-type-agnostic by design: this module doesn't know or care
whether a row is a cross-correlation or an autocorrelation -- it decodes
whatever baseline value is there and returns it. Filtering by correlation
type, where it belongs at all, happens in a caller that has a specific
reason to (see docs/dev/GWB_PIPELINE_REFACTOR_PLAN.md, T5b).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from data_io.raw_data_access import open_fits_readonly, open_raw_memmap

DEFAULT_RAM_FRACTION_TO_USE = 0.20


def host_total_memory_bytes() -> int:
    """Total physical RAM on this host, in bytes -- POSIX-portable (works on
    the Raspberry Pi target as well as this development host), stdlib only."""
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


@dataclass(frozen=True)
class GroupParamsLayout:
    gcount: int  # number of rows (groups)
    pcount: int  # number of parameter floats per row
    data_floats_per_group: int  # number of visibility-data floats per row
    data_offset: int  # byte offset of the first group's data in the file
    param_names: list[str]  # PTYPE1..PTYPEn, in file order


def read_group_params_layout(fits_path: Path | str) -> GroupParamsLayout:
    """Read the primary header only -- cheap -- to compute the file's row layout."""
    with open_fits_readonly(fits_path) as hdul:
        header = hdul[0].header
        file_info = hdul.fileinfo(0)

        gcount = int(header["GCOUNT"])
        pcount = int(header["PCOUNT"])
        data_floats_per_group = 1
        for axis in range(2, header.get("NAXIS", 1) + 1):
            data_floats_per_group *= int(header.get(f"NAXIS{axis}", 1))
        param_names = [header.get(f"PTYPE{i + 1}", "").strip() for i in range(pcount)]

        return GroupParamsLayout(
            gcount=gcount,
            pcount=pcount,
            data_floats_per_group=data_floats_per_group,
            data_offset=file_info["datLoc"],
            param_names=param_names,
        )


def resolve_all_param_columns(layout: GroupParamsLayout, name: str) -> list[int]:
    """Find every column index matching a parameter name, tolerant of
    UVFITS's own punctuation (e.g. ``UU---SIN``) and case.

    A name can legitimately appear more than once: UVFITS splits a single
    quantity (e.g. ``DATE``) across two PTYPE columns when one float32
    isn't precise enough on its own -- confirmed present in the real GWB
    and GSB files, both list ``PTYPE5`` and ``PTYPE6`` as ``DATE``. Callers
    that need every occurrence use this instead of `resolve_param_column`.
    """
    normalized = [p.upper().replace("-", "").replace("_", "") for p in layout.param_names]
    target = name.upper().replace("-", "").replace("_", "")
    return [i for i, p in enumerate(normalized) if p.startswith(target)]


def resolve_param_column(layout: GroupParamsLayout, name: str) -> int:
    """Find a parameter's first matching column index (see `resolve_all_param_columns`)."""
    matches = resolve_all_param_columns(layout, name)
    if not matches:
        raise ValueError(f"no parameter named like {name!r} in {layout.param_names}")
    return matches[0]


def read_all_param_columns(
    fits_path: Path | str,
    layout: GroupParamsLayout,
    max_chunk_bytes: int | None = None,
    ram_fraction: float = DEFAULT_RAM_FRACTION_TO_USE,
    verbose: bool = False,
) -> np.ndarray:
    """Read every parameter column for every row, as a ``(gcount, pcount)``
    float64 array -- touching only the parameter bytes, never the visibility
    data that follows each one in the file.

    Every row's parameter block sits at a fixed, large stride from the next
    (the visibility data in between). Skipping that gap doesn't make this
    faster on a spinning disk -- confirmed directly: with a stride smaller
    than the disk's own rotational sweep, the platter has to pass over the
    unwanted bytes regardless of whether they're read or skipped, so this
    costs close to what reading the whole file would cost.

    What this function bounds instead is peak memory: rows are read in
    blocks, each through its own small memmap that's closed before the next
    one opens, so at most one block's worth of file-backed pages is
    resident at a time -- 22GB+ resident for a single unblocked pass over
    the real 389GB GWB file, confirmed directly. This pipeline's own target
    includes a Raspberry Pi, so that bound matters as much as correctness.

    Block size comes from a memory budget (`max_chunk_bytes`, or
    `ram_fraction` of the host's total RAM if not given), not from anything
    about the data -- antenna count, integration length, whether the file
    even has a fixed record length at all. Those are per-file facts a
    generic reader must not assume; a larger, memory-budget-sized block is
    chosen only because fewer, bigger reads mean fewer separate file opens.
    """
    floats_per_group = layout.pcount + layout.data_floats_per_group
    row_bytes = floats_per_group * 4  # dtype ">f4" is 4 bytes per float

    if max_chunk_bytes is None:
        max_chunk_bytes = int(host_total_memory_bytes() * ram_fraction)
    chunk_rows = max(1, max_chunk_bytes // row_bytes)
    n_chunks = -(-layout.gcount // chunk_rows)  # ceiling division, for progress reporting only

    started = time.monotonic()
    result = np.empty((layout.gcount, layout.pcount), dtype=np.float64)
    for chunk_num, start in enumerate(range(0, layout.gcount, chunk_rows), start=1):
        stop = min(start + chunk_rows, layout.gcount)
        chunk = open_raw_memmap(
            fits_path,
            dtype=">f4",
            shape=(stop - start, floats_per_group),
            offset=layout.data_offset + start * row_bytes,
        )
        result[start:stop] = chunk[:, : layout.pcount]
        del chunk  # closes this block's own mapping, releasing its pages before the next block opens
        if verbose:
            elapsed = time.monotonic() - started
            print(
                f"read_all_param_columns: chunk {chunk_num}/{n_chunks}, "
                f"rows [{start}, {stop}) of {layout.gcount}, {elapsed:.1f}s elapsed",
                flush=True,
            )
    return result


def read_group_param_columns(
    fits_path: Path | str, layout: GroupParamsLayout, names: list[str]
) -> dict[str, np.ndarray]:
    """Read named parameter columns for every row (see `read_all_param_columns`).

    For a name that occurs more than once (e.g. ``DATE``), this returns only
    the first occurrence -- use `read_all_param_columns` with
    `resolve_all_param_columns` directly to get every occurrence.
    """
    all_params = read_all_param_columns(fits_path, layout)
    result = {}
    for name in names:
        column = resolve_param_column(layout, name)
        result[name] = all_params[:, column]
    return result


def decode_baseline(baseline_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decode the standard AIPS baseline encoding into (ant1, ant2) station numbers.

    baseline = ant1*256 + ant2 for antenna numbers below 256 (GMRT's 30
    antennas always fall in this range); values above 65536 use AIPS's
    extended encoding (baseline - 65536 = ant1*2048 + ant2) for antenna
    numbers up to 2047, for interferometers with more antennas than fit in
    the standard encoding. Decodes any baseline value, including an
    autocorrelation (ant1 == ant2) -- this function only decodes, it never
    filters by correlation type.

    Returned as int16, not because GMRT has few antennas, but because the
    encoding itself can never produce a decoded antenna number above 2047
    (the extended encoding's own ceiling) -- true for any telescope using
    this AIPS convention, so this is the correct width generically, not a
    GMRT-specific shortcut. Chosen deliberately narrow: this pipeline's own
    target includes a Raspberry Pi, and every bit counts across ~4 million
    rows.
    """
    values = np.rint(np.asarray(baseline_values, dtype=np.float64)).astype(np.int64)
    extended = values > 65536
    ant1 = np.where(extended, (values - 65536) // 2048, values // 256)
    ant2 = np.where(extended, (values - 65536) % 2048, values % 256)
    return ant1.astype(np.int16), ant2.astype(np.int16)
