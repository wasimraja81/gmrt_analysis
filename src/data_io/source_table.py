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

from data_io.raw_data_access import open_fits_readonly


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


def read_source_table(fits_path: Path | str) -> dict[int, Source]:
    """Read every source in the AIPS SU table, keyed by source id."""
    with open_fits_readonly(fits_path) as hdul:
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
            )
        return result
