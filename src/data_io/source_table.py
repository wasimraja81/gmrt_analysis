"""Generic AIPS source-table (SU) reading: source coordinates and
attributes, for any random-groups UVFITS file following the AIPS
convention -- not specific to any one telescope.

`RAEPO`/`DECEPO` (epoch, e.g. J2000) and `RAAPP`/`DECAPP` (apparent, i.e.
precessed/nutated to the date of observation) are both in degrees --
confirmed directly against the real GWB file (2026-09-25): values run
0-360 for RA and match the published coordinates of the sources present
(3C286, 3C345, ...), not hours or radians.

`CALCODE` and the per-IF flux columns (`IFLUX`/`QFLUX`/`UFLUX`/`VFLUX`) are
read as-is, whatever they contain -- for the real GWB file, confirmed
directly, `CALCODE` is empty and every flux column is `[0., 0.]` for all 13
sources.

`CALCODE` encodes calibrator scan intent by letter (D: gain, E: flux
scale, F: bandpass, G: polarization angle). Empty here means no intent was
recorded for any source in this file -- not that none of them are usable
as calibrators.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from data_io.raw_data_access import open_fits_tables


@dataclass(frozen=True)
class Source:
    id: int
    name: str
    ra_epoch_deg: float
    dec_epoch_deg: float
    ra_apparent_deg: float
    dec_apparent_deg: float
    epoch_year: float
    calcode: str
    flux_i_jy: tuple[float, ...]  # one per IF, as stored -- may be all zero/unpopulated
    flux_q_jy: tuple[float, ...]
    flux_u_jy: tuple[float, ...]
    flux_v_jy: tuple[float, ...]
    # The rest of the table (T20; AIPS Memo 117), per IF where so marked, as stored; None or () when absent.
    qualifier: int = 0  # QUAL: the source qualifier number
    freq_offset_hz: tuple[float, ...] = ()  # FREQOFF, per IF
    bandwidth_hz: float | None = None  # BANDWIDTH
    lsr_velocity_m_s: tuple[float, ...] = ()  # LSRVEL, per IF
    rest_freq_hz: tuple[float, ...] = ()  # RESTFREQ, per IF
    pm_ra_deg_per_day: float | None = None  # PMRA
    pm_dec_deg_per_day: float | None = None  # PMDEC


@dataclass(frozen=True)
class SourceTableFrame:
    """The SU table's own keywords: how many IFs its per-IF columns describe
    (NO_IF; the archival files' hold two values each with NO_IF 1), and the
    velocities' reference frame and definition (VELTYP, VELDEF)."""

    n_if: int | None
    velocity_type: str | None
    velocity_definition: str | None


def read_source_table_frame(fits_path: Path | str) -> SourceTableFrame:
    with open_fits_tables(fits_path) as hdul:
        try:
            header = hdul["AIPS SU"].header
        except KeyError:
            return SourceTableFrame(None, None, None)

        def text(key):
            value = header.get(key)
            return str(value).strip() or None if value is not None else None
        return SourceTableFrame(int(header["NO_IF"]) if "NO_IF" in header else None, text("VELTYP"), text("VELDEF"))


def _per_if(row, column: str, cols) -> tuple[float, ...]:
    if column not in cols:
        return ()
    return tuple(float(v) for v in np.atleast_1d(np.asarray(row[column])))


def read_source_table(fits_path: Path | str) -> dict[int, Source]:
    """Read every source in the AIPS SU table, keyed by source id."""
    with open_fits_tables(fits_path) as hdul:
        try:
            su = hdul["AIPS SU"]
        except KeyError:
            return {}
        cols = set(su.columns.names)
        id_col = "ID. NO." if "ID. NO." in cols else ("ID_NO." if "ID_NO." in cols else None)
        if id_col is None or "SOURCE" not in cols:
            return {}

        result: dict[int, Source] = {}
        for row in su.data:
            name = row["SOURCE"]
            if isinstance(name, bytes):
                name = name.decode("ascii")
            calcode = row["CALCODE"] if "CALCODE" in cols else ""
            if isinstance(calcode, bytes):
                calcode = calcode.decode("ascii")

            sid = int(row[id_col])
            result[sid] = Source(
                id=sid,
                name=str(name).strip(),
                ra_epoch_deg=float(row["RAEPO"]) if "RAEPO" in cols else float("nan"),
                dec_epoch_deg=float(row["DECEPO"]) if "DECEPO" in cols else float("nan"),
                ra_apparent_deg=float(row["RAAPP"]) if "RAAPP" in cols else float("nan"),
                dec_apparent_deg=float(row["DECAPP"]) if "DECAPP" in cols else float("nan"),
                epoch_year=float(row["EPOCH"]) if "EPOCH" in cols else float("nan"),
                calcode=str(calcode).strip(),
                flux_i_jy=tuple(float(v) for v in row["IFLUX"]) if "IFLUX" in cols else (),
                flux_q_jy=tuple(float(v) for v in row["QFLUX"]) if "QFLUX" in cols else (),
                flux_u_jy=tuple(float(v) for v in row["UFLUX"]) if "UFLUX" in cols else (),
                flux_v_jy=tuple(float(v) for v in row["VFLUX"]) if "VFLUX" in cols else (),
                qualifier=int(row["QUAL"]) if "QUAL" in cols else 0,
                freq_offset_hz=_per_if(row, "FREQOFF", cols),
                bandwidth_hz=float(row["BANDWIDTH"]) if "BANDWIDTH" in cols else None,
                lsr_velocity_m_s=_per_if(row, "LSRVEL", cols),
                rest_freq_hz=_per_if(row, "RESTFREQ", cols),
                pm_ra_deg_per_day=float(row["PMRA"]) if "PMRA" in cols else None,
                pm_dec_deg_per_day=float(row["PMDEC"]) if "PMDEC" in cols else None,
            )
        return result
