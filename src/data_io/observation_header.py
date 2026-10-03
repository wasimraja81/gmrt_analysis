"""Generic UVFITS primary-header reading: the keywords that describe an
observation (T20), for any random-groups file following the AIPS
convention. Each is None when the file does not give it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from data_io.raw_data_access import open_fits_readonly


@dataclass(frozen=True)
class ObservationHeader:
    telescope: str | None  # TELESCOP
    instrument: str | None  # INSTRUME
    observer: str | None  # OBSERVER
    object_name: str | None  # OBJECT (MULTI for a multi-source file)
    date_obs: str | None  # DATE-OBS
    bunit: str | None  # BUNIT
    epoch: float | None  # EPOCH (or EQUINOX), years
    reference_freq_hz: float | None  # the FREQ axis's CRVAL, which the FQ table's IF FREQ offsets


def read_observation_header(fits_path: Path | str) -> ObservationHeader:
    with open_fits_readonly(fits_path) as hdul:
        header = hdul[0].header

        def text(key: str) -> str | None:
            value = header.get(key)
            return str(value).strip() or None if value is not None else None

        epoch = header.get("EPOCH", header.get("EQUINOX"))
        reference_freq = None
        for i in range(2, int(header.get("NAXIS", 0)) + 1):
            if str(header.get(f"CTYPE{i}", "")).strip() == "FREQ":
                reference_freq = float(header.get(f"CRVAL{i}", 0.0))
        return ObservationHeader(
            telescope=text("TELESCOP"), instrument=text("INSTRUME"), observer=text("OBSERVER"),
            object_name=text("OBJECT"), date_obs=text("DATE-OBS"), bunit=text("BUNIT"),
            epoch=float(epoch) if epoch is not None else None, reference_freq_hz=reference_freq,
        )
