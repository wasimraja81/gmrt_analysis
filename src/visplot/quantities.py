"""Every quantity a plot can show, in one registry, with the units it can be
shown in.

A quantity is one `Quantity` entry: its name, display name, whether it needs
visibility data read from disk, whether it is a category, and its base
evaluations -- functions evaluating it on a `VisibilityBlock` chunk, each in
one base unit. A unit (`Unit`) is one of those bases times a positive factor:
u in km is its length base (m) times 1e-3, u in Mλ its wavelength base (λ)
times 1e-6. Units of one base differ only by that factor, so ranges found and
cached in the base serve every unit of it. Adding a quantity is adding one
entry.

Evaluated values keep a size-1 axis wherever the quantity does not vary
(a row quantity is shaped (n_rows, 1, 1, ...); frequency is 1 everywhere but
the FREQ axis), so pairing two quantities with numpy broadcasting yields only
the shape the pair needs: hour angle vs time stays one value per row, while
amplitude vs frequency covers every sample. The base decides the shape: u in
s is one value per row, u in kλ one per row and channel.

The visibility quantities (real, imag, amp) are in the file's BUNIT; when
BUNIT is a flux density they can also be shown in Jy, mJy or µJy. Old
quantity names that carried their unit (u_klambda, time_h, ...) stay as
aliases of a quantity and unit (`ALIASES`).

Categories (Stokes, source) evaluate to integer codes; `category_label`
turns a code into its label.
"""

from __future__ import annotations

import datetime
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import astropy.units as u
import numpy as np
from astropy.time import Time

from data_io.astrometry import altaz_deg, hour_angle_hours, local_sidereal_time_hours, parallactic_angle_deg
from data_io.visibility_data import VisibilityBlock

SPEED_OF_LIGHT_M_PER_S = 299_792_458.0
# Sidereal hours per UT1 hour. Used only to count whole sidereal days when
# LST is unwrapped past 24 h, so its precision does not reach the values.
SIDEREAL_PER_SOLAR = 1.002737909350795
# BUNIT spellings astropy does not parse: AIPS writes Jy as 'JY'.
AIPS_UNIT_SPELLINGS = {"JY": "Jy"}


@dataclass(frozen=True)
class QuantityContext:
    """What evaluating a quantity needs beyond the chunk itself, fixed for a
    whole plot: the time origin (so every chunk shares one), the file's
    BUNIT, the selected Stokes labels, and per-source coordinates and names
    indexed by source id."""

    time_reference_jd: float
    bunit: str | None = None
    stokes_labels: tuple[str, ...] = ()
    source_names: dict[int, str] | None = None
    source_ra_deg: np.ndarray | None = None  # indexed by source id
    source_dec_deg: np.ndarray | None = None
    array_location: object | None = None  # astropy EarthLocation
    antenna_names: dict[int, str] | None = None  # station number -> name, for reporting baselines
    time_zone: str | None = None  # the observatory's IANA time zone, for local time
    # The file's reference date (AN RDATE, else DATE-OBS) as the JD of its 0h: clock times
    # count days from it; None: from the first selected integration's date.
    reference_date_jd: float | None = None
    time_system: str | None = None  # AN TIMSYS: the time scale the file declares for its dates
    # Recorded time - UTC, in s (`TimeReference.recorded_minus_utc_s`): subtracted wherever a
    # timestamp is read as UTC (UTC, local time, LST, hour angle, Az/El, parallactic angle).
    recorded_minus_utc_s: float = 0.0


def utc_jd(ctx: QuantityContext, jd):
    """Recorded Julian dates as UTC."""
    return np.asarray(jd) - ctx.recorded_minus_utc_s / 86400.0


def context_from_source_table(time_reference_jd, source_table, array_location, bunit=None, stokes_labels=(),
                              antenna_names=None, time_zone=None, reference_date_jd=None, time_system=None,
                              recorded_minus_utc_s=0.0):
    """A `QuantityContext` with per-source coordinate lookup arrays built once
    from `read_source_table`'s dict (source id -> Source)."""
    max_id = max(source_table) if source_table else 0
    ra = np.full(max_id + 1, np.nan)
    dec = np.full(max_id + 1, np.nan)
    for sid, src in source_table.items():
        ra[sid] = src.ra_apparent_deg
        dec[sid] = src.dec_apparent_deg
    return QuantityContext(
        time_reference_jd=time_reference_jd,
        bunit=bunit,
        stokes_labels=tuple(stokes_labels or ()),
        source_names={sid: src.name for sid, src in source_table.items()},
        source_ra_deg=ra,
        source_dec_deg=dec,
        array_location=array_location,
        antenna_names=antenna_names,
        time_zone=time_zone,
        reference_date_jd=reference_date_jd,
        time_system=time_system,
        recorded_minus_utc_s=recorded_minus_utc_s,
    )


@dataclass(frozen=True)
class Unit:
    """One way to show a quantity: `name` as given on the command line,
    `label` as shown on the axis. The value shown is the quantity's base
    evaluation `base` times `factor` (positive). A clock unit (UTC, local
    time, LST) holds hours and is labelled as a time of day."""

    name: str
    label: str
    base: str
    factor: float = 1.0
    clock: bool = False


@dataclass(frozen=True)
class Quantity:
    name: str
    display_name: str
    description: str  # for the command line's help
    needs_data: bool
    categorical: bool
    # (base name, evaluate(block, ctx)): values in that base's unit
    bases: tuple[tuple[str, Callable[[VisibilityBlock, QuantityContext], np.ndarray]], ...]
    units: tuple[Unit, ...] = ()  # empty: the file's BUNIT (visibility quantities) or none (categories)
    default_unit: str = ""
    # Quantities in the same group are plotted at equal scale by default (--aspect auto)
    # when shown in the same unit, e.g. u against v.
    aspect_group: str | None = None

    @property
    def unit_from_bunit(self) -> bool:
        return self.needs_data and not self.units

    def evaluate_base(self, base: str, block: VisibilityBlock, ctx: QuantityContext) -> np.ndarray:
        for name, evaluate in self.bases:
            if name == base:
                return evaluate(block, ctx)
        raise ValueError(f"{self.name!r} has no base {base!r}")


def _factor(from_unit, to_unit) -> float:
    """The conversion factor, rounded to 15 significant digits so float noise
    (hourangle to deg: 14.999999999999998) does not reach labels or ranges."""
    return float(f"{from_unit.to(to_unit):.15g}")


def _units(base: str, base_unit, *choices) -> tuple[Unit, ...]:
    """Units of one base: (name, label, astropy unit) per choice."""
    return tuple(Unit(name, label, base, _factor(base_unit, target)) for name, label, target in choices)


def _row(values, block: VisibilityBlock) -> np.ndarray:
    values = np.asarray(values)
    return values.reshape((len(values),) + (1,) * len(block.axis_types))


def _along(axis_type: str, values, block: VisibilityBlock) -> np.ndarray:
    if axis_type not in block.axis_types:
        raise ValueError(f"this data has no {axis_type} axis (axes: {block.axis_types})")
    shape = [1] * (1 + len(block.axis_types))
    shape[1 + block.axis_types.index(axis_type)] = len(values)
    return np.asarray(values).reshape(shape)


def _freq_hz(block: VisibilityBlock) -> np.ndarray:
    if block.chan_freqs_hz is None:
        raise ValueError("this data has no channel frequencies (no FREQ axis)")
    return _along("FREQ", block.chan_freqs_hz, block)


def _source_coords(block: VisibilityBlock, ctx: QuantityContext) -> tuple[np.ndarray, np.ndarray]:
    if ctx.source_ra_deg is None or ctx.array_location is None:
        raise ValueError("observing-geometry quantities need the source table and the array location")
    return ctx.source_ra_deg[block.source_id], ctx.source_dec_deg[block.source_id]


def _hour_angle(block, ctx):
    ra, _ = _source_coords(block, ctx)
    return _row(hour_angle_hours(utc_jd(ctx, block.jd), ra, ctx.array_location), block)


def _azimuth(block, ctx):
    ra, dec = _source_coords(block, ctx)
    return _row(altaz_deg(utc_jd(ctx, block.jd), ra, dec, ctx.array_location)[0], block)


def _elevation(block, ctx):
    ra, dec = _source_coords(block, ctx)
    return _row(altaz_deg(utc_jd(ctx, block.jd), ra, dec, ctx.array_location)[1], block)


def _parallactic_angle(block, ctx):
    ra, dec = _source_coords(block, ctx)
    return _row(parallactic_angle_deg(utc_jd(ctx, block.jd), ra, dec, ctx.array_location), block)


def utc_day_start_jd(jd: float) -> float:
    """The JD of 0h UTC on the day of `jd` (a JD starts at noon)."""
    return math.floor(jd - 0.5) + 0.5


def day_origin_jd(ctx: QuantityContext) -> float:
    """The JD of 0h on the date clock times count days from: the file's
    reference date, else the first selected integration's date."""
    if ctx.reference_date_jd is not None:
        return ctx.reference_date_jd
    return utc_day_start_jd(ctx.time_reference_jd)


def _recorded_hours(block, ctx):
    """The DATE random parameters as recorded, in hours since 0h of the
    reference date (day 1 starts at 24)."""
    return _row((block.jd - day_origin_jd(ctx)) * 24.0, block)


def _utc_hours(block, ctx):
    """UTC in hours since 0h UTC of the reference date."""
    return _row((utc_jd(ctx, block.jd) - day_origin_jd(ctx)) * 24.0, block)


@lru_cache(maxsize=None)
def local_time_zone(time_zone: str | None, jd: float) -> tuple[float, str]:
    """(UTC offset in hours, zone abbreviation) of `time_zone` at `jd`.
    Raises ValueError without a zone, or for a zone the system's time zone
    database does not have."""
    if not time_zone:
        raise ValueError("local time needs the observatory's time zone, which is not known here")
    try:
        zone = ZoneInfo(time_zone)
    except (ZoneInfoNotFoundError, ValueError) as err:
        raise ValueError(f"unknown time zone {time_zone!r} (expected an IANA name, e.g. Asia/Kolkata)") from err
    moment = Time(jd, format="jd", scale="utc").to_datetime(timezone=datetime.timezone.utc).astimezone(zone)
    return moment.utcoffset().total_seconds() / 3600.0, moment.tzname()


def local_day_start_jd(ctx: QuantityContext) -> float:
    """The JD (UTC) of 0h local time on the reference date (as a local
    date), at the zone's UTC offset at the first selected integration."""
    offset_h, _ = local_time_zone(ctx.time_zone, float(utc_jd(ctx, ctx.time_reference_jd)))
    return day_origin_jd(ctx) - offset_h / 24.0


def _local_hours(block, ctx):
    """Local time in hours since 0h local time on the reference date, at the
    zone's UTC offset at the first selected integration (for the whole plot:
    a daylight-saving change during the selection does not shift the axis)."""
    return _row((utc_jd(ctx, block.jd) - local_day_start_jd(ctx)) * 24.0, block)


def utc_offset_text(offset_h: float) -> str:
    minutes = round(offset_h * 60)
    sign = "+" if minutes >= 0 else "-"
    return f"UTC{sign}{abs(minutes) // 60:02d}:{abs(minutes) % 60:02d}"


def _lst_hours(block, ctx):
    """Apparent local sidereal time in hours, continuous from the LST at 0h
    UTC of the reference date: each 0h LST passed adds 24 (a sidereal day)."""
    if ctx.array_location is None:
        raise ValueError("LST needs the array location")
    jd, inverse = np.unique(block.jd, return_inverse=True)  # rows share integrations
    jd = utc_jd(ctx, jd)
    ref = day_origin_jd(ctx)
    lst = local_sidereal_time_hours(np.append(jd, ref), ctx.array_location)
    lst, lst_ref = lst[:-1], lst[-1]
    expected = lst_ref + (jd - ref) * 24.0 * SIDEREAL_PER_SOLAR
    unwrapped = lst + 24.0 * np.round((expected - lst) / 24.0)
    return _row(unwrapped[inverse.ravel()], block)


def _uvdist_sec(block):
    return np.hypot(block.uu_sec.astype(np.float64), block.vv_sec.astype(np.float64))


def _baseline_bases(delay_sec):
    """The two bases of u, v, w and uv distance, from the file's values in
    seconds (light travel time): the length in m (one value per row) and in
    wavelengths at each channel's frequency."""
    return (
        ("length", lambda b, c: _row(delay_sec(b) * SPEED_OF_LIGHT_M_PER_S, b)),
        ("wavelength", lambda b, c: _row(delay_sec(b), b) * _freq_hz(b)),
    )


_ANGLE_UNITS = _units("deg", u.deg, ("deg", "deg", u.deg), ("rad", "rad", u.rad))
_BASELINE_UNITS = (
    (Unit("lambda", "λ", "wavelength", 1.0), Unit("klambda", "kλ", "wavelength", 1e-3),
     Unit("Mlambda", "Mλ", "wavelength", 1e-6))
    + _units("length", u.m, ("m", "m", u.m), ("km", "km", u.km))
)
# Help text, shared by quantities with the same units (the help lists them together).
_BASELINE_HELP = ("baseline u, v, w and uv distance sqrt(u^2 + v^2): at each channel's frequency in lambda, "
                  "klambda (default), Mlambda; or m, km")
_DATA_HELP = ("visibility real part, imaginary part, amplitude: in the file's BUNIT; also Jy, mJy, uJy when "
              "BUNIT is a flux density")
_ANGLE_HELP = "azimuth, elevation, parallactic angle: deg (default), rad"
_CATEGORY_HELP = "categories (also for --colorize-by)"

_ALL = [
    # visibility data: stored at the file's float32; amplitude and phase are
    # computed in float64, so a sample on a pixel boundary bins the same way
    # whatever the storage precision
    Quantity("real", "Real", _DATA_HELP, True, False,
             (("value", lambda b, c: b.data.real),), aspect_group="complex"),
    Quantity("imag", "Imag", _DATA_HELP, True, False,
             (("value", lambda b, c: b.data.imag),), aspect_group="complex"),
    Quantity("amp", "Amplitude", _DATA_HELP, True, False,
             (("value", lambda b, c: np.abs(b.data.astype(np.complex128))),)),
    Quantity("phase", "Phase", "visibility phase: deg (default), rad", True, False,
             (("deg", lambda b, c: np.degrees(np.angle(b.data.astype(np.complex128)))),),
             _ANGLE_UNITS, "deg"),
    # per row
    Quantity("time", "Time", "clock time dd:hh:mm:ss, days counted from the file's reference date: "
             "recorded (default; as the file records it), UTC, local (the observatory's time zone), LST; "
             "or time since the first selected integration in h, min, s", False, False,
             (("recorded", _recorded_hours), ("utc", _utc_hours), ("local", _local_hours), ("lst", _lst_hours),
              ("elapsed", lambda b, c: _row((b.jd - c.time_reference_jd) * 24.0, b))),
             (Unit("recorded", "recorded", "recorded", clock=True), Unit("UTC", "UTC", "utc", clock=True),
              Unit("local", "local", "local", clock=True), Unit("LST", "LST", "lst", clock=True))
             + _units("elapsed", u.h, ("h", "h", u.h), ("min", "min", u.min), ("s", "s", u.s)), "recorded"),
    Quantity("u", "U", _BASELINE_HELP, False, False,
             _baseline_bases(lambda b: b.uu_sec), _BASELINE_UNITS, "klambda", aspect_group="uvw"),
    Quantity("v", "V", _BASELINE_HELP, False, False,
             _baseline_bases(lambda b: b.vv_sec), _BASELINE_UNITS, "klambda", aspect_group="uvw"),
    Quantity("w", "W", _BASELINE_HELP, False, False,
             _baseline_bases(lambda b: b.ww_sec), _BASELINE_UNITS, "klambda", aspect_group="uvw"),
    Quantity("uvdist", "UV distance", _BASELINE_HELP, False, False,
             _baseline_bases(_uvdist_sec), _BASELINE_UNITS, "klambda"),
    Quantity("ha", "Hour angle", "hour angle: h (default), deg, rad", False, False, (("ha", _hour_angle),),
             _units("ha", u.hourangle, ("h", "h", u.hourangle), ("deg", "deg", u.deg), ("rad", "rad", u.rad)), "h"),
    Quantity("az", "Azimuth", _ANGLE_HELP, False, False, (("deg", _azimuth),),
             _ANGLE_UNITS, "deg"),
    Quantity("el", "Elevation", _ANGLE_HELP, False, False, (("deg", _elevation),),
             _ANGLE_UNITS, "deg"),
    Quantity("pa", "Parallactic angle", _ANGLE_HELP, False, False,
             (("deg", _parallactic_angle),), _ANGLE_UNITS, "deg"),
    # per channel
    Quantity("freq", "Frequency", "channel frequency: MHz (default), Hz, kHz, GHz", False, False,
             (("hz", lambda b, c: _freq_hz(b)),),
             _units("hz", u.Hz, ("Hz", "Hz", u.Hz), ("kHz", "kHz", u.kHz), ("MHz", "MHz", u.MHz),
                    ("GHz", "GHz", u.GHz)), "MHz"),
    # categories
    Quantity("stokes", "Stokes", _CATEGORY_HELP, False, True,
             (("code", lambda b, c: _along("STOKES", np.arange(len(b.axis_indices.get("STOKES", ()))), b)),)),
    Quantity("source", "Source", _CATEGORY_HELP, False, True,
             (("code", lambda b, c: _row(b.source_id.astype(np.int64), b)),)),
]

QUANTITIES: dict[str, Quantity] = {q.name: q for q in _ALL}

# Earlier names that carried their unit: name -> (quantity, unit).
ALIASES: dict[str, tuple[str, str]] = {
    "phase_deg": ("phase", "deg"),
    "time_h": ("time", "h"),
    "uvdist_m": ("uvdist", "m"),
    "u_klambda": ("u", "klambda"), "v_klambda": ("v", "klambda"), "w_klambda": ("w", "klambda"),
    "uvdist_klambda": ("uvdist", "klambda"),
    "freq_mhz": ("freq", "MHz"),
    "ha_h": ("ha", "h"),
    "az_deg": ("az", "deg"), "el_deg": ("el", "deg"), "pa_deg": ("pa", "deg"),
}


@lru_cache(maxsize=None)
def _flux_unit(bunit: str | None):
    """BUNIT as an astropy flux density unit, or None if it is not one."""
    if not bunit:
        return None
    parsed = u.Unit(AIPS_UNIT_SPELLINGS.get(bunit, bunit), parse_strict="silent")
    if isinstance(parsed, u.UnrecognizedUnit) or not parsed.is_equivalent(u.Jy):
        return None
    return parsed


@lru_cache(maxsize=None)
def _data_units(bunit: str | None) -> tuple[Unit, ...]:
    """The file's own unit (named as BUNIT spells it; unnamed without
    BUNIT), and, when BUNIT is a flux density, Jy, mJy and µJy."""
    own = Unit(bunit or "", bunit or "", "value")
    units = [own]
    flux = _flux_unit(bunit)
    if flux is not None:
        for name, label, target in (("Jy", "Jy", u.Jy), ("mJy", "mJy", u.mJy), ("uJy", "µJy", u.uJy)):
            if name != own.name:
                units.append(Unit(name, label, "value", _factor(flux, target)))
    return tuple(units)


def units_of(name: str, ctx: QuantityContext | None = None) -> tuple[Unit, ...]:
    """The units quantity `name` can be shown in. A category has one,
    unnamed; a visibility quantity's first is the file's BUNIT."""
    q = QUANTITIES[name]
    if q.categorical:
        return (Unit("", "", q.bases[0][0]),)
    if q.unit_from_bunit:
        return _data_units(ctx.bunit if ctx is not None else None)
    return q.units


def resolve_unit(name: str, unit: str | None = None, ctx: QuantityContext | None = None) -> Unit:
    """Quantity `name`'s unit called `unit` (by name, e.g. "klambda", or
    label, e.g. "kλ"); its default if `unit` is None (the file's
    BUNIT for a visibility quantity). Raises ValueError naming the choices."""
    choices = units_of(name, ctx)
    if unit is None:
        default = QUANTITIES[name].default_unit
        return next((choice for choice in choices if choice.name == default), choices[0])
    for choice in choices:
        if unit in (choice.name, choice.label):
            return choice
    q = QUANTITIES[name]
    if q.categorical:
        raise ValueError(f"{name!r} is a category and has no unit; got {unit!r}")
    if q.unit_from_bunit and _flux_unit(ctx.bunit if ctx is not None else None) is None:
        own = ctx.bunit if ctx is not None and ctx.bunit else None
        raise ValueError(
            f"{name!r} is in the file's BUNIT ({own!r}), which is not a flux density, so it cannot be "
            f"converted to {unit!r}" if own else f"{name!r} has no unit here (the file has no BUNIT); got {unit!r}"
        )
    names = ", ".join(choice.name for choice in choices)
    raise ValueError(f"unit {unit!r} does not apply to {name!r}; choose one of: {names}")


# Earlier names whose unit is no longer offered: name -> what to write instead.
RETIRED_NAMES: dict[str, str] = {
    f"{q}_sec": f"{q} with a unit of m, km, lambda, klambda or Mlambda (seconds of light travel time were dropped)"
    for q in ("u", "v", "w")
}


def canonical(name: str, unit: str | None = None) -> tuple[str, str | None]:
    """(quantity, unit) for a quantity name or alias. An alias fixes the
    unit; a different `unit` alongside it raises ValueError, as does a
    retired name."""
    if name in RETIRED_NAMES:
        raise ValueError(f"{name!r} is no longer available: use {RETIRED_NAMES[name]}")
    if name not in ALIASES:
        return name, unit
    quantity, alias_unit = ALIASES[name]
    if unit is not None and resolve_unit(quantity, unit).name != alias_unit:
        raise ValueError(
            f"{name!r} is {quantity!r} in {alias_unit}, but the unit asked for is {unit!r}: "
            f"write {quantity!r} with that unit instead"
        )
    return quantity, alias_unit


def evaluate(name: str, block: VisibilityBlock, ctx: QuantityContext, unit: str | None = None) -> np.ndarray:
    """Quantity (or alias) `name` on `block`, in `unit` (default: the
    quantity's default)."""
    name, unit = canonical(name, unit)
    chosen = resolve_unit(name, unit, ctx)
    values = QUANTITIES[name].evaluate_base(chosen.base, block, ctx)
    return values if chosen.factor == 1.0 else values * chosen.factor


def convert(value: float, name: str, from_unit: str | None, to_unit: str | None,
            ctx: QuantityContext | None = None) -> float:
    """A value of quantity `name` in `from_unit`, in `to_unit`. Raises
    ValueError between units of different bases (e.g. s and kλ, where the
    factor depends on the channel)."""
    a, b = resolve_unit(name, from_unit, ctx), resolve_unit(name, to_unit, ctx)
    if a.base != b.base:
        raise ValueError(f"{name!r} in {a.label} cannot be converted to {b.label} by a fixed factor")
    return value / a.factor * b.factor


def _iso_utc(jd: float, precision: int) -> str:
    time = Time(jd, format="jd", scale="utc")
    time.precision = precision
    return time.iso


def quantity_label(name: str, ctx: QuantityContext | None = None, unit: str | None = None) -> str:
    """Display name plus unit, for an axis label, e.g. "Frequency (MHz)",
    "Amplitude (UNCALIB)" (unit from the file's BUNIT; none shown if the
    file has none). Clock time names the date its days count from, and
    recorded time the file's declared time system; elapsed time names its
    origin."""
    name, unit = canonical(name, unit)
    q = QUANTITIES[name]
    chosen = resolve_unit(name, unit, ctx)
    if ctx is not None and chosen.clock:
        day0 = f"day 0 = {_iso_utc(day_origin_jd(ctx), 0)[:10]}"
        if chosen.base == "recorded":
            system = f", {ctx.time_system}" if ctx.time_system else ""
            return f"{q.display_name} (recorded{system}; {day0})"
        if chosen.base == "local" and ctx.time_zone:
            offset_h, abbreviation = local_time_zone(ctx.time_zone, float(utc_jd(ctx, ctx.time_reference_jd)))
            return f"{q.display_name} ({abbreviation}, {utc_offset_text(offset_h)}; {day0})"
        if chosen.base == "lst":
            return f"{q.display_name} (LST; day 0 at {_iso_utc(day_origin_jd(ctx), 0)[:10]} 0h UTC)"
        return f"{q.display_name} ({chosen.label}; {day0})"
    if chosen.base == "elapsed" and ctx is not None:
        return f"{q.display_name} since {_iso_utc(float(utc_jd(ctx, ctx.time_reference_jd)), 0)} UTC ({chosen.label})"
    return f"{q.display_name} ({chosen.label})" if chosen.label else q.display_name


def value_description(name: str, ctx: QuantityContext | None = None, unit: str | None = None) -> str:
    """What a value in `unit` means, spelled out for a file header, e.g.
    "uvdist in kλ", "time in UTC, hours since 2021-07-24 00:00 UTC"."""
    name, unit = canonical(name, unit)
    chosen = resolve_unit(name, unit, ctx)
    if ctx is not None and chosen.clock:
        origin = _iso_utc(day_origin_jd(ctx), 0)[:16]
        if chosen.base == "recorded":
            system = f" (TIMSYS {ctx.time_system})" if ctx.time_system else ""
            return f"{name} as recorded{system}, in hours since {origin}"
        if chosen.base == "utc":
            return f"{name} in UTC, hours since {origin} UTC"
        if chosen.base == "local" and ctx.time_zone:
            offset_h, abbreviation = local_time_zone(ctx.time_zone, float(utc_jd(ctx, ctx.time_reference_jd)))
            return (f"{name} in local time, hours since {origin} {abbreviation} "
                    f"({ctx.time_zone}, {utc_offset_text(offset_h)})")
        if chosen.base == "lst":
            return f"{name} as LST in hours, from the LST at {origin} UTC, 24 added at each 0h LST"
    if chosen.base == "elapsed" and ctx is not None:
        return f"{name} in {chosen.label} since {_iso_utc(float(utc_jd(ctx, ctx.time_reference_jd)), 3)} UTC"
    return f"{name} in {chosen.label}" if chosen.label else name


def category_label(name: str, code: int, ctx: QuantityContext) -> str:
    if name == "stokes":
        return ctx.stokes_labels[code] if code < len(ctx.stokes_labels) else str(code)
    if name == "source":
        return (ctx.source_names or {}).get(int(code), str(code))
    raise ValueError(f"{name!r} is not a category")
