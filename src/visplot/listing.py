"""listObs (T20): what a file holds, as text, as CASA listobs and AIPS
LISTR's scan listing and PRTAN list it -- from the header, the AN, FQ and SU
tables and the row index, no visibility data read.

Sections (`SECTIONS`, every one by default, as the user chose 2026-10-03):
observation, scans (derived by AIPS INDXR's rules,
`data_io.observation_summary`), sources, spectral setup, antennas. A listing
covers the whole file, or a selection (`ListingSelection`: the rows,
Stokes and channels a request selects, as listobs's `selectdata`).
"""

from __future__ import annotations

import datetime
import os
from dataclasses import dataclass, field

import astropy.units as u
import numpy as np
from astropy.coordinates import Angle
from astropy.time import Time

from data_io.antenna_table import MOUNT_TYPES, antenna_enu_m, read_antenna_feeds
from data_io.astrometry import altaz_deg
from data_io.observation_header import read_observation_header
from data_io.observation_summary import (DEFAULT_GAP_INTEGRATIONS, DEFAULT_LONGEST_SCAN_S, Scan, derive_scans,
                                         integrations_of, rows_per_antenna)
from data_io.row_index import default_row_index_path
from data_io.source_table import read_source_table_frame
from visplot.clock_axis import DEFAULT_TIME_FORMAT, clock_text

SECTIONS = ("observation", "scans", "sources", "spectral", "antennas")
SECTION_TITLES = {"observation": "Observation", "scans": "Scans", "sources": "Sources", "spectral": "Spectral setup",
                  "antennas": "Antennas"}


@dataclass(frozen=True)
class ListingOptions:
    sections: tuple[str, ...] = SECTIONS
    gap_integrations: float = DEFAULT_GAP_INTEGRATIONS  # a gap longer than this many integration times: a new scan
    longest_s: float = DEFAULT_LONGEST_SCAN_S  # a scan on one source longer than this: a new scan
    time_format: str = DEFAULT_TIME_FORMAT


@dataclass(frozen=True)
class ListingSelection:
    """The part of a file listed: its rows (None: every row), Stokes and
    channels (None: all), and its filters as text ("" for the whole file)."""

    row_indices: np.ndarray | None = None
    stokes_labels: tuple[str, ...] | None = None
    channel_indices: np.ndarray | None = None
    text: str = ""


def list_observation(opened, options: ListingOptions = ListingOptions(),
                     selection: ListingSelection | None = None) -> str:
    """The listing of `opened` (a `visplot.run.OpenedFile`), its
    `options.sections` in order."""
    listing = _Listing(opened, options, selection or ListingSelection())
    blocks = [getattr(listing, f"_{name}")() for name in options.sections]
    return "\n\n".join("\n".join(block) for block in blocks) + "\n"


def _table(headers: list[str], rows: list[list[str]], right: set[int]) -> list[str]:
    """Fixed-width columns: those in `right` aligned right."""
    widths = [max(len(str(c)) for c in column) for column in zip(headers, *rows)]

    def line(cells):
        return "  ".join(str(c).rjust(w) if i in right else str(c).ljust(w)
                         for i, (c, w) in enumerate(zip(cells, widths))).rstrip()
    return [line(headers)] + [line(r) for r in rows]


def _fixed(value: float, digits: int) -> str:
    """`value` to `digits` places, a rounded-away negative zero written 0."""
    text = f"{value:.{digits}f}"
    return text[1:] if text.startswith("-") and float(text) == 0 else text


def _indent(lines: list[str]) -> list[str]:
    return ["  " + line if line else line for line in lines]


def _ra(deg: float) -> str:
    return "" if not np.isfinite(deg) else Angle(deg, u.deg).to_string(unit=u.hourangle, sep=":", precision=3, pad=True)


def _dec(deg: float) -> str:
    return "" if not np.isfinite(deg) else Angle(deg, u.deg).to_string(unit=u.deg, sep=":", precision=2, pad=True,
                                                                          alwayssign=True)


@dataclass
class _Listing:
    opened: object
    options: ListingOptions
    selection: ListingSelection
    integrations: object = field(init=False)
    scans: list[Scan] = field(init=False)

    def __post_init__(self):
        index = self.opened.index
        self.integrations = integrations_of(index, self.selection.row_indices)
        self.scans = derive_scans(self.integrations, self.options.gap_integrations, self.options.longest_s)
        origin = self.opened.reference_date_jd
        stamps = np.asarray(index.jd)
        self.origin_jd = origin if origin is not None else float(np.floor(stamps.min() - 0.5) + 0.5)
        self.day0 = datetime.date.fromisoformat(Time(self.origin_jd, format="jd", scale="utc").iso[:10])
        try:
            self.recorded_minus_utc_s = self.opened.time_reference.recorded_minus_utc_s
        except ValueError:
            self.recorded_minus_utc_s = None

    # ---- helpers ------------------------------------------------------------------

    @property
    def n_rows(self) -> int:
        rows = self.selection.row_indices
        return int(self.opened.index.gcount) if rows is None else len(rows)

    def clock(self, jd: float) -> str:
        return clock_text((jd - self.origin_jd) * 24.0, 0, self.options.time_format, self.day0)

    def utc(self, jd: float) -> str:
        if self.recorded_minus_utc_s is None:
            return "UTC unknown"
        return Time(jd - self.recorded_minus_utc_s / 86400.0, format="jd", scale="utc").iso[:19]

    def elevation(self, source_id: int, jd: float) -> str:
        source = self.opened.source_table.get(source_id)
        if source is None or self.recorded_minus_utc_s is None or not np.isfinite(source.dec_apparent_deg):
            return ""
        _, el = altaz_deg(jd - self.recorded_minus_utc_s / 86400.0, source.ra_apparent_deg, source.dec_apparent_deg,
                          self.opened.array_location)
        return f"{float(el):.1f}"

    def source_name(self, source_id: int) -> str:
        source = self.opened.source_table.get(source_id)
        return source.name if source is not None else self.opened.index.id_to_name.get(source_id, str(source_id))

    # ---- sections --------------------------------------------------------------------

    def _observation(self) -> list[str]:
        opened, index = self.opened, self.opened.index
        header = read_observation_header(opened.fits_path)
        size_gb = os.path.getsize(opened.fits_path) / 1e9
        integration_s = self.integrations.integration_s
        jd = self.integrations.jd
        reference = opened.time_reference
        try:
            check = opened.timestamp_check.summary()
        except Exception as err:  # a listing goes on without it
            check = f"timestamps not checked ({type(err).__name__}: {err})"
        described = ", ".join(f"{label} {value}" for label, value in (
            ("instrument", header.instrument), ("observer", header.observer), ("object", header.object_name))
            if value)
        lines = [
            ("File", f"{os.path.basename(opened.fits_path)} ({size_gb:.1f} GB; its row index "
                     f"{default_row_index_path(opened.fits_path).name})"),
            ("Telescope", f"{header.telescope or 'not given'}" + (f" ({described})" if described else "")),
            ("Dates", f"DATE-OBS {header.date_obs or 'not given'}; day 0 {self.day0.isoformat()} "
                      f"({'the reference date, RDATE' if reference.reference_date else 'from the first integration'})"),
            ("Time system", reference.describe()),
            ("", check),
        ]
        if jd.size:
            lines += [
                ("Recorded", f"{self.clock(jd.min())} to {self.clock(jd.max())} "
                             f"({reference.time_system or 'time system not declared'})"),
                ("UTC", f"{self.utc(jd.min())} to {self.utc(jd.max())}"),
                ("Integrations", f"{jd.size:,}" + (f" of {integration_s:.2f} s" if integration_s else "")),
            ]
        row_bytes = (index.pcount + int(np.prod(index.data_axis_lengths))) * 4
        of = f" of {int(index.gcount):,}" if self.selection.row_indices is not None else ""
        lines += [
            ("Rows", f"{self.n_rows:,}{of} ({self.n_rows * row_bytes / 1e9:.1f} GB of visibilities)"),
            ("Amplitude unit", header.bunit or "BUNIT not given"),
            ("Selection", self.selection.text or "the whole file"),
        ]
        width = max(len(label) for label, _ in lines)
        return ["Observation"] + _indent([f"{label.ljust(width)}  {text}" for label, text in lines])

    def _scans(self) -> list[str]:
        integration_s = self.integrations.integration_s or 0.0
        rule = (f"a new scan where the source changes, after a gap of more than {self.options.gap_integrations:g} "
                f"integration times ({self.options.gap_integrations * integration_s:.1f} s), or past "
                f"{self.options.longest_s / 60:g} min on one source (AIPS INDXR's rules; the file has no NX table)")
        rows = [[str(s.number), self.source_name(s.source_id), self.clock(s.start_jd), self.clock(s.end_jd),
                 f"{s.length_s / 60:.1f}", f"{s.n_integrations:,}", f"{s.n_rows:,}",
                 self.elevation(s.source_id, s.start_jd), self.elevation(s.source_id, s.end_jd)] for s in self.scans]
        table = _table(["Scan", "Source", "Start", "End", "Length (min)", "Integrations", "Rows", "El start (deg)",
                        "El end (deg)"], rows, right={0, 4, 5, 6, 7, 8})
        notes = []
        for a, b in zip(self.scans, self.scans[1:]):
            gap_s = (b.start_jd - a.end_jd) * 86400.0
            if a.source_id == b.source_id and gap_s <= self.options.gap_integrations * integration_s:
                notes.append(f"Scans {a.number} and {b.number}: one stretch on {self.source_name(a.source_id)}, cut at "
                             f"{self.options.longest_s / 60:g} min (the longest-scan rule).")
        return [f"Scans ({len(self.scans)}): {rule}"] + _indent(table + notes)

    def _sources(self) -> list[str]:
        opened = self.opened
        frame = read_source_table_frame(opened.fits_path)
        by_source: dict[int, list[Scan]] = {}
        for scan in self.scans:
            by_source.setdefault(scan.source_id, []).append(scan)
        rows = []
        for sid, source in sorted(opened.source_table.items()):
            scans = by_source.get(sid, [])
            epoch = f"J{source.epoch_year:g}" if np.isfinite(source.epoch_year) else ""
            rows.append([str(sid), source.name, str(source.qualifier), source.calcode or "", _ra(source.ra_epoch_deg),
                         _dec(source.dec_epoch_deg), epoch, _ra(source.ra_apparent_deg), _dec(source.dec_apparent_deg),
                         str(len(scans)), f"{sum(s.length_s for s in scans) / 60:.1f}",
                         f"{sum(s.n_rows for s in scans):,}"])
        table = _table(["ID", "Name", "Qual", "Code", "RA", "Dec", "Epoch", "RA (apparent)", "Dec (apparent)", "Scans",
                        "On source (min)", "Rows"], rows, right={0, 2, 9, 10, 11})
        n_if = frame.n_if or 1
        notes = []
        for label, attribute, unit in (("Flux I, Q, U, V", None, "Jy"), ("LSR velocity", "lsr_velocity_m_s", "m/s"),
                                       ("Rest frequency", "rest_freq_hz", "Hz"), ("Frequency offset", "freq_offset_hz",
                                                                                  "Hz")):
            given = []
            for sid, source in sorted(opened.source_table.items()):
                if attribute is None:
                    values = [v for column in (source.flux_i_jy, source.flux_q_jy, source.flux_u_jy, source.flux_v_jy)
                              for v in column[:n_if]]
                else:
                    values = list(getattr(source, attribute)[:n_if])
                if any(v != 0 for v in values):
                    given.append(f"{source.name} {', '.join(f'{v:g}' for v in values)}")
            notes.append(f"{label} ({unit}, {n_if} IF{'s' if n_if != 1 else ''}): "
                         + ("; ".join(given) if given else "none given (all zero)"))
        if frame.velocity_type or frame.velocity_definition:
            notes.append(f"Velocities: {frame.velocity_type or ''} {frame.velocity_definition or ''}".rstrip())
        return [f"Sources ({len(opened.source_table)} in the SU table)"] + _indent(table + notes)

    def _spectral(self) -> list[str]:
        opened, index = self.opened, self.opened.index
        header = read_observation_header(opened.fits_path)
        freqs = np.asarray(index.chan_freqs_hz) if index.chan_freqs_hz is not None else np.array([])
        reference = header.reference_freq_hz or 0.0
        rows = []
        for setup in opened.frequency_setups:
            for k, offset in enumerate(setup.if_offset_hz):
                width = setup.channel_width_hz[k] if k < len(setup.channel_width_hz) else float("nan")
                sideband = setup.sideband[k] if k < len(setup.sideband) else 1
                bandwidth = setup.total_bandwidth_hz[k] if k < len(setup.total_bandwidth_hz) else float("nan")
                rows.append([str(setup.id), str(k + 1), f"{(reference + offset) / 1e6:.3f}",
                             f"{width * sideband / 1e3:.3f}", f"{bandwidth / 1e6:.3f}",
                             {1: "upper", -1: "lower"}.get(sideband, str(sideband))])
        lines = []
        if rows:
            lines += _table(["Setup", "IF", "Frequency (MHz)", "Channel step (kHz)", "Bandwidth (MHz)", "Sideband"],
                            rows, right={0, 1, 2, 3, 4})
            lines.append("Frequency: the reference frequency plus the IF's offset; the step per channel: channel "
                         "width times sideband (AIPS Memo 117)")
            if len(opened.frequency_setups) > 1:
                lines.append(f"{len(opened.frequency_setups)} setups: the row index does not keep each row's (FREQSEL), "
                             "and visplot does not plot this file's visibilities")
        else:
            lines.append("No FQ table: the header's FREQ axis alone")
        chosen = freqs if self.selection.channel_indices is None else freqs[np.asarray(self.selection.channel_indices)]
        if chosen.size:
            of = f" of {freqs.size:,}" if chosen.size != freqs.size else ""
            lines.append(f"Channels: {chosen.size:,}{of}, {chosen[0] / 1e6:.6f} MHz (the first) to "
                         f"{chosen[-1] / 1e6:.6f} MHz (the last)")
        stokes = self.selection.stokes_labels or tuple(index.stokes_labels or ())
        lines.append(f"Stokes: {', '.join(stokes) or 'none'}")
        return [f"Spectral setup (the FQ table; reference frequency {reference / 1e6:.3f} MHz)"] + _indent(lines)

    def _antennas(self) -> list[str]:
        opened = self.opened
        location = opened.array_location
        feeds = read_antenna_feeds(opened.fits_path)
        counts = rows_per_antenna(opened.index, self.selection.row_indices)
        east, north, up = antenna_enu_m(opened.antennas, location)
        rows = []
        for a, e, n, h in zip(opened.antennas, east, north, up):
            feed = feeds.get(a.station_number)
            mount = "" if feed is None or feed.mount_type is None else MOUNT_TYPES.get(feed.mount_type,
                                                                                      str(feed.mount_type))
            offset = "" if feed is None or feed.axis_offset_m is None else f"{feed.axis_offset_m:.3f}"
            polarisations = "" if feed is None else ", ".join(
                f"{kind}" + (f" {angle:g} deg" if angle else "")
                for kind, angle in ((feed.pol_type_a, feed.pol_angle_a_deg), (feed.pol_type_b, feed.pol_angle_b_deg))
                if kind)
            rows.append([str(a.station_number), a.name, _fixed(e, 1), _fixed(n, 1), _fixed(h, 1), mount, offset,
                         polarisations, f"{counts.get(a.station_number, 0):,}"])
        table = _table(["Station", "Name", "East (m)", "North (m)", "Up (m)", "Mount", "Axis offset (m)", "Feeds",
                        "Rows"], rows, right={0, 2, 3, 4, 6, 8})
        geodetic = location.to_geodetic()
        heading = (f"Antennas ({len(opened.antennas)}; positions from the array's reference at latitude "
                   f"{geodetic.lat.deg:.5f} deg, longitude {geodetic.lon.deg:.5f} deg, height {geodetic.height.value:.1f} m)")
        notes = []
        if opened.dud_antennas:
            notes.append(f"Left out: {', '.join(a.name for a in opened.dud_antennas)} (the telescope's structural DUD "
                         "entries of the AN table)")
        return [heading] + _indent(table + notes)
