"""What an opened file holds, in a few lines: telescope, time system, span,
integration time, antennas, channels, Stokes and sources with their row
counts (the GUI's data panel; the start of T20's listObs)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from visplot.clock_axis import clock_text


@dataclass(frozen=True)
class FileSummary:
    telescope: str | None
    bunit: str | None
    reference_date: str | None
    time_system: str | None
    n_rows: int
    n_integrations: int
    integration_s: float | None  # median spacing of distinct timestamps
    first_hours: float  # recorded time of the first and last integrations, hours since 0h of the reference date
    last_hours: float
    n_antennas: int
    n_baselines: int  # cross-correlation baselines present
    n_channels: int
    freq_lo_hz: float | None
    freq_hi_hz: float | None
    channel_width_hz: float | None
    stokes_labels: tuple[str, ...]
    sources: tuple[tuple[int, str, int], ...]  # (id, name, rows)
    data_gb: float  # the whole file's visibility data

    def lines(self) -> list[tuple[str, str]]:
        """(label, text) pairs for display."""
        span = f"{clock_text(self.first_hours, 0)} to {clock_text(self.last_hours, 0)}"
        day0 = f" (day 0 = {self.reference_date})" if self.reference_date else ""
        channels = f"{self.n_channels:,}"
        if self.freq_lo_hz is not None:
            channels += f", {self.freq_lo_hz / 1e6:.3f}-{self.freq_hi_hz / 1e6:.3f} MHz"
        if self.channel_width_hz:
            channels += f", {abs(self.channel_width_hz) / 1e3:.3f} kHz wide"
        integration = f"{self.integration_s:.3f} s" if self.integration_s else "one timestamp"
        return [
            ("Telescope", self.telescope or "not given"),
            ("Recorded time", f"{span}{day0}, {self.time_system or 'time system not declared'}"),
            ("Integrations", f"{self.n_integrations:,} of {integration}"),
            ("Rows", f"{self.n_rows:,} ({self.data_gb:.1f} GB of visibilities)"),
            ("Antennas", f"{self.n_antennas} ({self.n_baselines:,} baselines)"),
            ("Channels", channels),
            ("Stokes", ", ".join(self.stokes_labels) or "none"),
            ("Amplitude unit", self.bunit or "BUNIT not given"),
            ("Sources", f"{len(self.sources)}"),
        ]


def summarize(opened) -> FileSummary:
    """The summary of an `OpenedFile` (`visplot.run.open_file`)."""
    index = opened.index
    jd = np.asarray(index.jd)
    stamps = np.unique(jd)
    spacing = np.diff(stamps) * 86400.0
    origin = opened.reference_date_jd
    if origin is None:
        origin = np.floor(stamps[0] - 0.5) + 0.5
    cross = np.asarray(index.ant1) != np.asarray(index.ant2)
    # one integer per (ant1, ant2): np.unique over the rows of a 2-D array took 6.2 s on the GWB
    # file's 4.0M rows, this 0.2 s
    pairs = np.unique(index.ant1[cross].astype(np.int64) * 65536 + index.ant2[cross]) if cross.any() else []
    freqs = np.asarray(index.chan_freqs_hz) if index.chan_freqs_hz is not None else np.array([])
    rows_by_source = {sid: sum(stop - start for start, stop in runs) for sid, runs in index.source_ranges.items()}
    row_bytes = (index.pcount + int(np.prod(index.data_axis_lengths))) * 4
    return FileSummary(
        telescope=opened.telescope, bunit=opened.bunit,
        reference_date=opened.time_reference.reference_date, time_system=opened.time_reference.time_system,
        n_rows=int(index.gcount), n_integrations=len(index.integration_boundaries) - 1,
        integration_s=float(np.median(spacing)) if spacing.size else None,
        first_hours=float((stamps[0] - origin) * 24.0), last_hours=float((stamps[-1] - origin) * 24.0),
        n_antennas=len(np.union1d(index.ant1, index.ant2)), n_baselines=len(pairs),
        n_channels=len(freqs), freq_lo_hz=float(freqs.min()) if freqs.size else None,
        freq_hi_hz=float(freqs.max()) if freqs.size else None,
        channel_width_hz=float(np.median(np.diff(freqs))) if freqs.size > 1 else None,
        stokes_labels=tuple(index.stokes_labels or ()),
        sources=tuple((sid, index.id_to_name.get(sid, str(sid)), rows_by_source[sid]) for sid in sorted(rows_by_source)),
        data_gb=index.gcount * row_bytes / 1e9,
    )
