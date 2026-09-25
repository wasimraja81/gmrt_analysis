"""Building a fast row index for a random-groups UVFITS file.

A random-groups UVFITS file records one visibility per row, with no
row-to-source or row-to-scan lookup stored anywhere else in the file: no
"AIPS NX" index/scan table is present in either the real GWB or GSB file
for this observation (checked directly, 2026-09-24 -- both files only
carry AIPS AN, AIPS FQ and AIPS SU). So which rows belong to which
source, and which antenna pair each row is, cannot be read from a header
or a small table; each of the ~4 million rows carries its own SOURCE and
BASELINE value, and building the index means reading that column once
for every row.

Telescope-agnostic by design: this only uses information every
random-groups UVFITS file carries (PTYPE parameters, the AIPS SU table,
the WCS-style frequency/Stokes header keywords). Antenna-level concerns
(dud-antenna resolution, GMRT-specific naming) live in
instruments/gmrt/row_index.py, built on top of this.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits

from data_io.raw_data_access import open_fits_readonly
from data_io.uvfits_group_params import (
    decode_baseline,
    read_all_param_columns,
    read_group_params_layout,
    resolve_all_param_columns,
    resolve_param_column,
)

_STOKES_MAP = {
    1: "I", 2: "Q", 3: "U", 4: "V",
    -1: "RR", -2: "LL", -3: "RL", -4: "LR",
    -5: "XX", -6: "YY", -7: "XY", -8: "YX",
}


@dataclass(frozen=True)
class RowIndex:
    path: str
    gcount: int
    pcount: int
    source_id: np.ndarray  # per row
    jd: np.ndarray  # per row, Julian date
    ant1: np.ndarray  # per row
    ant2: np.ndarray  # per row
    uu_sec: np.ndarray  # per row
    vv_sec: np.ndarray  # per row
    source_ranges: dict[int, list[tuple[int, int]]]  # source id -> [(start, stop), ...), half-open, contiguous runs
    integration_boundaries: np.ndarray  # row indices where a new integration starts, length n_integrations + 1
    id_to_name: dict[int, str]  # from the AIPS SU table
    chan_freqs_hz: np.ndarray  # one per channel
    stokes_labels: list[str]  # one per polarization product
    data_axis_lengths: list[int]  # NAXIS2..NAXISn, in header order, whatever axes this file actually has
    data_axis_types: list[str]  # CTYPE2..CTYPEn, same order -- e.g. "COMPLEX", "STOKES", "FREQ", "IF", "RA", "DEC"
    data_offset: int  # byte offset of the first group's data in the file
    build_time_sec: float


def _read_data_axes(header) -> tuple[list[int], list[str]]:
    """Read every data axis's length and CTYPE label, for however many axes
    this header actually declares (NAXIS2..NAXIS<n>) -- not a fixed count.

    The real GWB file has 7 axes, not the 4 (complex, stokes, freq, IF) the
    random-groups convention is usually described with -- it also carries
    RA and DEC axes (length 1 here, but not necessarily 1 in general, e.g.
    a multi-pointing file) -- confirmed directly, 2026-09-24, not assumed.
    """
    n = int(header.get("NAXIS", 1))
    lengths = [int(header.get(f"NAXIS{i}", 1)) for i in range(2, n + 1)]
    types = [str(header.get(f"CTYPE{i}", "")).strip().upper() for i in range(2, n + 1)]
    return lengths, types


def find_axis(data_axis_types: list[str], ctype_name: str) -> int | None:
    """Index (into `data_axis_lengths`/`data_axis_types`) of the axis whose
    CTYPE matches `ctype_name`, found by label rather than assumed position."""
    try:
        return data_axis_types.index(ctype_name)
    except ValueError:
        return None


def _compute_integration_boundaries(source_id: np.ndarray, jd: np.ndarray) -> np.ndarray:
    """Row indices where a new integration starts -- a new integration begins
    whenever SOURCE or DATE changes from the previous row. Length is
    n_integrations + 1 (the last entry is `gcount`, closing the final run),
    so integration i spans rows [boundaries[i], boundaries[i+1]).

    Not assumed to be uniform in length or antenna content -- see standing
    rule 9 -- this only records where the boundaries fall.
    """
    if len(source_id) == 0:
        return np.array([0], dtype=np.int64)
    changes = np.where((np.diff(source_id) != 0) | (np.diff(jd) != 0))[0] + 1
    return np.concatenate([[0], changes, [len(source_id)]]).astype(np.int64)


def _read_source_id_to_name(fits_path: Path | str) -> dict[int, str]:
    """Read the AIPS SU table's source-id-to-name lookup -- a small, whole-file
    table, not a per-row quantity."""
    with open_fits_readonly(fits_path) as hdul:
        try:
            su = hdul["AIPS SU"]
        except KeyError:
            return {}
        cols = set(su.columns.names)
        id_col = "ID. NO." if "ID. NO." in cols else ("ID_NO." if "ID_NO." in cols else None)
        if id_col is None or "SOURCE" not in cols:
            return {}
        result = {}
        for row in su.data:
            name = row["SOURCE"]
            if isinstance(name, bytes):
                name = name.decode("ascii")
            result[int(row[id_col])] = str(name).strip()
        return result


def _compute_source_ranges(source_id: np.ndarray) -> dict[int, list[tuple[int, int]]]:
    """Group row indices by source id into contiguous runs.

    A source is not guaranteed to occupy one contiguous block of rows --
    the same source can be observed in separate scans at different times,
    confirmed in the real GWB index (source id 4 alone has 9 separate
    blocks) -- so this reports every run, not just a single (start, stop).
    """
    ranges: dict[int, list[tuple[int, int]]] = {}
    for sid in np.unique(source_id):
        indices = np.where(source_id == sid)[0]
        breaks = np.where(np.diff(indices) > 1)[0] + 1
        runs = np.split(indices, breaks)
        ranges[int(sid)] = [(int(run[0]), int(run[-1]) + 1) for run in runs]
    return ranges


def build_row_index(
    fits_path: Path | str,
    max_chunk_bytes: int | None = None,
    verbose: bool = False,
    logger=None,
) -> RowIndex:
    """Read every row's group parameters once and derive a fast, in-memory index.

    This is the one unavoidable full-file pass: every row's SOURCE and
    BASELINE value lives only in that row's own parameter block, with no
    shortcut available in the header or in any small table. Everything
    else this function needs (source names, channel frequencies,
    polarization labels) comes from cheap, whole-file metadata instead.

    `max_chunk_bytes`, `verbose`, and `logger` pass straight through to
    `read_all_param_columns`, which does the actual full-file pass and
    bounds peak memory to about one chunk's size (see its own docstring).
    """
    started = time.monotonic()
    fits_path = Path(fits_path)

    layout = read_group_params_layout(fits_path)
    all_params = read_all_param_columns(
        fits_path, layout, max_chunk_bytes=max_chunk_bytes, verbose=verbose, logger=logger
    )

    source_id = all_params[:, resolve_param_column(layout, "SOURCE")].astype(np.int32)
    ant1, ant2 = decode_baseline(all_params[:, resolve_param_column(layout, "BASELINE")])

    date_columns = resolve_all_param_columns(layout, "DATE")
    jd = all_params[:, date_columns[0]].copy()
    if len(date_columns) > 1:
        jd += all_params[:, date_columns[1]]

    uu_sec = all_params[:, resolve_param_column(layout, "UU")].astype(np.float32)
    vv_sec = all_params[:, resolve_param_column(layout, "VV")].astype(np.float32)

    with open_fits_readonly(fits_path) as hdul:
        header = hdul[0].header
        data_axis_lengths, data_axis_types = _read_data_axes(header)

        stokes_axis = find_axis(data_axis_types, "STOKES")
        if stokes_axis is None:
            raise ValueError(f"no STOKES axis in {fits_path}: CTYPEs are {data_axis_types}")
        nstokes = data_axis_lengths[stokes_axis]
        axis_num = stokes_axis + 2  # data_axis_types[0] is NAXIS2/CTYPE2
        crval_s = int(round(float(header.get(f"CRVAL{axis_num}", -5))))
        cdelt_s = int(round(float(header.get(f"CDELT{axis_num}", -1))))

        freq_axis = find_axis(data_axis_types, "FREQ")
        if freq_axis is None:
            raise ValueError(f"no FREQ axis in {fits_path}: CTYPEs are {data_axis_types}")
        nchan = data_axis_lengths[freq_axis]
        axis_num = freq_axis + 2
        crval_f = float(header.get(f"CRVAL{axis_num}", 0.0))
        cdelt_f = float(header.get(f"CDELT{axis_num}", 1.0))
        crpix_f = float(header.get(f"CRPIX{axis_num}", 1.0))

    chan_freqs_hz = crval_f + (np.arange(1, nchan + 1, dtype=np.float64) - crpix_f) * cdelt_f
    stokes_labels = [_STOKES_MAP.get(crval_s + i * cdelt_s, f"S{crval_s + i * cdelt_s}") for i in range(nstokes)]

    return RowIndex(
        path=str(fits_path),
        gcount=layout.gcount,
        pcount=layout.pcount,
        source_id=source_id,
        jd=jd,
        ant1=ant1,
        ant2=ant2,
        uu_sec=uu_sec,
        vv_sec=vv_sec,
        source_ranges=_compute_source_ranges(source_id),
        integration_boundaries=_compute_integration_boundaries(source_id, jd),
        id_to_name=_read_source_id_to_name(fits_path),
        chan_freqs_hz=chan_freqs_hz,
        stokes_labels=stokes_labels,
        data_axis_lengths=data_axis_lengths,
        data_axis_types=data_axis_types,
        data_offset=layout.data_offset,
        build_time_sec=time.monotonic() - started,
    )


def default_row_index_path(fits_path: Path | str) -> Path:
    """Where a row index for this raw file lives: `<fits_path>.idx.npz`, always
    adjacent to the raw file it indexes, never in a separate work directory --
    confirmed with the user (2026-09-24) as the standing rule, regardless of
    telescope or correlator."""
    return Path(str(fits_path) + ".idx.npz")


def save_row_index(index: RowIndex, path: Path | str) -> Path:
    """Persist a row index to a single .npz file, round-tripping every field
    (not just the large per-row arrays) via `load_row_index`.

    Written atomically: to a temp file in the same directory first, then
    renamed into place -- a process killed mid-write (the real failure this
    project hit, 2026-09-24/25) leaves `path` either absent or with its
    previous complete contents, never a truncated/corrupt file that looks
    present but isn't valid.
    """
    path = Path(path)
    tmp_path = path.with_name(f"{path.name}.tmp{os.getpid()}")
    try:
        with open(tmp_path, "wb") as f:
            np.savez(
                f,
                path_str=index.path,
                gcount=index.gcount,
                pcount=index.pcount,
                source_id=index.source_id,
                jd=index.jd,
                ant1=index.ant1,
                ant2=index.ant2,
                uu_sec=index.uu_sec,
                vv_sec=index.vv_sec,
                integration_boundaries=index.integration_boundaries,
                chan_freqs_hz=index.chan_freqs_hz,
                data_offset=index.data_offset,
                build_time_sec=index.build_time_sec,
                source_ranges_json=json.dumps({str(k): v for k, v in index.source_ranges.items()}),
                id_to_name_json=json.dumps({str(k): v for k, v in index.id_to_name.items()}),
                stokes_labels_json=json.dumps(index.stokes_labels),
                data_axis_lengths_json=json.dumps(index.data_axis_lengths),
                data_axis_types_json=json.dumps(index.data_axis_types),
            )
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return path


def load_row_index(path: Path | str) -> RowIndex:
    """Load a row index previously written by `save_row_index`."""
    with np.load(path, allow_pickle=False) as data:
        source_ranges = {
            int(k): [tuple(r) for r in v]
            for k, v in json.loads(str(data["source_ranges_json"])).items()
        }
        id_to_name = {int(k): v for k, v in json.loads(str(data["id_to_name_json"])).items()}
        return RowIndex(
            path=str(data["path_str"]),
            gcount=int(data["gcount"]),
            pcount=int(data["pcount"]),
            source_id=data["source_id"],
            jd=data["jd"],
            ant1=data["ant1"],
            ant2=data["ant2"],
            uu_sec=data["uu_sec"],
            vv_sec=data["vv_sec"],
            source_ranges=source_ranges,
            integration_boundaries=data["integration_boundaries"],
            id_to_name=id_to_name,
            chan_freqs_hz=data["chan_freqs_hz"],
            stokes_labels=json.loads(str(data["stokes_labels_json"])),
            data_axis_lengths=json.loads(str(data["data_axis_lengths_json"])),
            data_axis_types=json.loads(str(data["data_axis_types_json"])),
            data_offset=int(data["data_offset"]),
            build_time_sec=float(data["build_time_sec"]),
        )
