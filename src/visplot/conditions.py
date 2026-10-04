"""Conditions (`--where`; T26's flagging design, step 2): one comparison each,
tested on every cell of a selection (a row, one baseline at one time, and one
channel). A cell matches when all the conditions of a run hold; an OR is one
run per alternative.

    condition := operand cmp value
               | value cmp operand cmp value       a bounded range, both cmp one direction
    operand   := function "(" product ")"         a data value of one product
               | coordinate                       a value of the row or channel
    function  := amp | phase | real | imag
    product   := RR LL RL LR XX YY XY YX           as the file holds it
               | I Q U V                          derived, as the IAU defines them
    coordinate:= uvdist uu vv ww el az ha pa freq channel time
    cmp       := <  <=  >  >=
    value     := number [unit]                    5, 1e3, -170, 20 deg, 2 klambda, 400 MHz
               | instant scale                    1/17:42:09 UTC, 2021-07-25T17:42:09 recorded

The rules that leave no reading open: a data value names its product; a product
only inside the parentheses, a coordinate only bare; every dimensional quantity
and phase with its unit (no default); amplitude, real and imaginary parts in the
file's BUNIT, writing it optional; `channel` a 0-based index, with no unit; time
an instant with its scale (`recorded`: as the rows record it; `UTC`), counted in
days from the file's day 0 (AN RDATE, else DATE-OBS, else the date of its first
integration) as the plots' time axes count them, or an ISO date;
only <, <=, >, >=. Names are case-insensitive, units case-sensitive.

Parsing (`parse_condition`) needs no file; `resolve_conditions` checks the
conditions against a file: its products, BUNIT, channels, band and time span,
then intersects the conditions on each operand, refusing what cannot hold
before any data is read and warning of what holds for every cell.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace

from astropy.time import Time

from data_io.stokes_products import CIRCULAR, DERIVED, LINEAR, plan_products
from visplot.quantities import QuantityContext, convert, domain_in, domain_text, resolve_unit, units_of
from visplot.range_spec import NUMBER

FUNCTIONS = ("amp", "phase", "real", "imag")
COORDINATES = ("uvdist", "uu", "vv", "ww", "el", "az", "ha", "pa", "freq", "channel", "time")
PRODUCTS = CIRCULAR + LINEAR + DERIVED
OPERATORS = ("<", "<=", ">", ">=")
TIME_SCALES = ("recorded", "UTC")

# The examples, one source for --help, the GUI's hover text and the user guide (each tested).
CONDITION_EXAMPLES = (
    ("amp(V) > 5", "Stokes V's amplitude above 5 (the file's BUNIT); V = RR - LL or -i(XY - YX), as the IAU "
                   "defines it"),
    ("amp(I) < 0.005", "Stokes I's amplitude below 0.005: dead antennas, for example (I = RR + LL or XX + YY)"),
    ("amp(RL) > 3000 UNCALIB", "the RL product's amplitude above 3000, its unit (the file's BUNIT) written out"),
    ("phase(RL) >= 170 deg", "the RL product's phase at or above 170 degrees (phase lies in -180..180 deg)"),
    ("100 < uvdist < 2000 lambda", "uv distance between 100 and 2000 wavelengths, both ends excluded"),
    ("el < 20 deg", "elevation below 20 degrees"),
    ("400 MHz <= freq <= 450 MHz", "channels from 400 to 450 MHz, both ends included"),
    ("channel >= 1800", "channels 1800 onwards (0-based, as --channels counts them)"),
    ("time >= 1/17:42:09 UTC", "from 17:42:09 UTC on day 1 (days counted from the file's day 0, as on the "
                               "plots' time axes)"),
    ("time < 2021-07-25T18:00:00 recorded", "before 18:00:00 on 2021-07-25 as the rows record time (their TIMSYS)"),
)

_DAY_CLOCK = re.compile(r"^(\d+)/(\d{1,2}):(\d{2}):(\d{2}(?:\.\d+)?)$")
_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?$")
_OPERATOR = re.compile(r"<=|>=|<|>")
_FUNCTION = re.compile(r"^([A-Za-z]+)\s*\(\s*([A-Za-z]+)\s*\)$")


@dataclass(frozen=True)
class Operand:
    """What a condition tests: a function of a product, or a coordinate."""

    name: str  # the function (amp, phase, real, imag) or the coordinate
    product: str | None = None  # the product, for a function

    def __str__(self) -> str:
        return f"{self.name}({self.product})" if self.product else self.name


@dataclass(frozen=True)
class Bound:
    value: float
    inclusive: bool


@dataclass(frozen=True)
class Condition:
    """One parsed condition: `operand` within (`lower`, `upper`), each a bound
    or None, in `unit` (None: the file's BUNIT for a data value; for time, the
    instant's scale and `instants` its two texts)."""

    text: str
    operand: Operand
    unit: str | None
    lower: Bound | None
    upper: Bound | None
    instants: tuple[str | None, str | None] = (None, None)  # time: the bounds as written, resolved per file


@dataclass(frozen=True)
class FileFacts:
    """What `resolve_conditions` checks conditions against, from an opened file."""

    stokes_labels: tuple[str, ...]
    bunit: str | None
    n_channels: int
    freq_hz: tuple[float, float]  # the lowest and highest channel frequencies
    jd: tuple[float, float]  # the first and last timestamps, as recorded
    day0_jd: float  # 0h of the file's day 0
    recorded_minus_utc_s: float


@dataclass(frozen=True)
class Interval:
    """An operand's allowed values in one unit: the intersection of its conditions."""

    lower: Bound | None
    upper: Bound | None

    def intersect(self, other: Interval) -> Interval:
        return Interval(_tighter(self.lower, other.lower, max), _tighter(self.upper, other.upper, min))

    @property
    def empty(self) -> bool:
        if self.lower is None or self.upper is None:
            return False
        if self.lower.value != self.upper.value:
            return self.lower.value > self.upper.value
        return not (self.lower.inclusive and self.upper.inclusive)


def _tighter(a: Bound | None, b: Bound | None, pick) -> Bound | None:
    if a is None or b is None:
        return a or b
    if a.value != b.value:
        return a if pick(a.value, b.value) == a.value else b
    return Bound(a.value, a.inclusive and b.inclusive)


def _refuse(text: str, why: str) -> ValueError:
    return ValueError(f"--where {text!r}: {why}")


def _operand(token: str, text: str) -> Operand:
    token = token.strip()
    match = _FUNCTION.match(token)
    if match:
        function, product = match.group(1).lower(), match.group(2).upper()
        if function not in FUNCTIONS:
            raise _refuse(text, f"{match.group(1)!r} is not a function; the functions are {', '.join(FUNCTIONS)}")
        if product not in PRODUCTS:
            raise _refuse(text, f"{match.group(2)!r} is not a product; the products are {', '.join(PRODUCTS)}")
        return Operand(function, product)
    name = token.lower()
    if name in COORDINATES:
        return Operand(name)
    if name in FUNCTIONS:
        raise _refuse(text, f"{token} needs its product, e.g. {name}(RR) or {name}(V)")
    if token.upper() in PRODUCTS:
        hint = f"; the baseline coordinate is {name * 2}" if name in ("u", "v", "w") else ""
        raise _refuse(text, f"a product is written inside a function, e.g. amp({token.upper()}){hint}")
    if name in ("u", "v", "w"):
        raise _refuse(text, f"the baseline coordinate is {name * 2}")
    raise _refuse(text, f"{token!r} is neither a coordinate ({', '.join(COORDINATES)}) nor a function of a "
                        f"product ({', '.join(f + '(...)' for f in FUNCTIONS)})")


def _units_text(name: str) -> str:
    return ", ".join(unit.name for unit in units_of(name))


def _value(token: str, operand: Operand, text: str, range_end: bool = False) -> tuple[float | str, str | None]:
    """(number, unit) from `token`, or for time (instant text, scale). A
    range's end may leave its unit to the other end (`range_end`)."""
    token = token.strip()
    name = operand.name
    if name == "time":
        parts = token.split()
        if len(parts) != 2 or parts[1] not in TIME_SCALES:
            raise _refuse(text, "time is an instant and its scale: d/hh:mm:ss or an ISO date, then recorded or "
                                "UTC, e.g. 1/17:42:09 UTC")
        if not (_DAY_CLOCK.match(parts[0]) or _ISO.match(parts[0])):
            raise _refuse(text, f"{parts[0]!r} is not an instant: write d/hh:mm:ss (days from the file's day 0) "
                                "or an ISO date, e.g. 2021-07-25T17:42:09")
        return parts[0], parts[1]
    match = re.match(rf"^({NUMBER})\s*(\S*)$", token)
    if not match:
        raise _refuse(text, f"{token!r} is not a number with a unit, e.g. 20 deg")
    number, unit = float(match.group(1)), match.group(2) or None
    if not math.isfinite(number):
        raise _refuse(text, f"{token!r} is not a finite number")
    if name == "channel":
        if unit is not None:
            raise _refuse(text, f"channel is a 0-based index and takes no unit; got {unit!r}")
        return number, None
    if name in ("amp", "real", "imag"):
        return number, unit  # the file's BUNIT, checked against the file
    if unit is None:
        if range_end:
            return number, None
        raise _refuse(text, f"{operand} needs its unit, one of: {_units_text(name)}")
    try:
        unit = resolve_unit(name, unit).name
    except ValueError as err:
        raise _refuse(text, str(err)) from None
    return number, unit


def _check_domain(operand: Operand, value: float, unit: str | None, text: str) -> None:
    domain = domain_in(operand.name, unit)
    if domain is None:
        return
    lowest, highest = domain
    if (lowest is not None and value < lowest) or (highest is not None and value > highest):
        where = f" {unit}" if unit else ""
        raise _refuse(text, f"{operand} lies in {domain_text(lowest, highest, unit)}; {value:g}{where} is outside it")


def parse_condition(text: str) -> Condition:
    """One `--where` condition, parsed and checked without a file. Raises
    ValueError saying what is wrong and how to write it."""
    if "=" in text.replace("<=", "").replace(">=", ""):
        raise _refuse(text, "only <, <=, >, >= compare (no equality on measured values)")
    pieces = _OPERATOR.split(text)
    operators = _OPERATOR.findall(text)
    if len(operators) == 1:
        operand = _operand(pieces[0], text)
        value, unit = _value(pieces[1], operand, text)
        op = operators[0]
        is_time = operand.name == "time"  # an instant: its value found per file (`resolve_conditions`)
        bound = Bound(math.nan if is_time else value, op.endswith("="))
        if not is_time:
            _check_domain(operand, value, unit, text)
        if op.startswith(">"):
            return Condition(text, operand, unit, bound, None, (value, None) if is_time else (None, None))
        return Condition(text, operand, unit, None, bound, (None, value) if is_time else (None, None))
    if len(operators) == 2:
        first, second = operators
        if (first in ("<", "<=")) != (second in ("<", "<=")):
            raise _refuse(text, "a range runs one way: a < x < b, or a > x > b")
        operand = _operand(pieces[1], text)
        (a, unit_a), (b, unit_b) = (_value(piece, operand, text, range_end=True) for piece in (pieces[0], pieces[2]))
        if unit_a is not None and unit_b is not None and unit_a != unit_b:
            raise _refuse(text, f"a range takes one unit, on either end or both; got {unit_a} and {unit_b}")
        unit_a = unit_a or unit_b  # written once, it applies to both ends
        if unit_a is None and operand.name not in ("channel", "amp", "real", "imag"):
            raise _refuse(text, f"{operand} needs its unit, one of: {_units_text(operand.name)}")
        ascending = first in ("<", "<=")
        low, high = (a, b) if ascending else (b, a)
        low_op, high_op = (first, second) if ascending else (second, first)
        if operand.name == "time":
            return Condition(text, operand, unit_a, Bound(math.nan, low_op.endswith("=")),
                             Bound(math.nan, high_op.endswith("=")), (low, high))
        _check_domain(operand, low, unit_a, text)
        _check_domain(operand, high, unit_a, text)
        if low > high or (low == high and not (low_op.endswith("=") and high_op.endswith("="))):
            raise _refuse(text, f"the range is empty: its lower end {low:g} is not below its upper end {high:g}")
        return Condition(text, operand, unit_a, Bound(low, low_op.endswith("=")), Bound(high, high_op.endswith("=")))
    if not operators:
        raise _refuse(text, "no comparison: write e.g. amp(V) > 5, or 100 < uvdist < 2000 lambda")
    raise _refuse(text, "at most two comparisons, as a range: a < x < b")


def _instant_jd(instant: str, scale: str, facts: FileFacts) -> float:
    """An instant as a recorded JD (the rows' own time)."""
    match = _DAY_CLOCK.match(instant)
    if match:
        day, hh, mm, ss = int(match.group(1)), int(match.group(2)), int(match.group(3)), float(match.group(4))
        jd = facts.day0_jd + day + (hh * 3600 + mm * 60 + ss) / 86400.0
    else:
        jd = Time(instant, format="isot", scale="tai").jd  # the calendar instant's JD number, in its own scale
    return jd + facts.recorded_minus_utc_s / 86400.0 if scale == "UTC" else jd


@dataclass(frozen=True)
class ResolvedConditions:
    conditions: tuple[Condition, ...]  # in the file's units: data values in BUNIT, time as recorded JD
    intervals: dict  # operand key -> Interval
    warnings: tuple[str, ...]  # conditions holding for every cell


def facts_of(opened) -> FileFacts:
    """`FileFacts` of a `visplot.run.OpenedFile`."""
    index = opened.index
    freqs = index.chan_freqs_hz
    try:
        offset_s = opened.time_reference.recorded_minus_utc_s
    except ValueError:
        offset_s = 0.0
    jd = index.jd
    day0 = opened.day0_jd
    return FileFacts(tuple(index.stokes_labels or ()), opened.bunit, len(freqs) if freqs is not None else 0,
                     (float(min(freqs)), float(max(freqs))) if freqs is not None and len(freqs) else (math.nan, math.nan),
                     (float(jd.min()), float(jd.max())), day0, offset_s)


def resolve_conditions(conditions: list[Condition], facts: FileFacts) -> ResolvedConditions:
    """`conditions` checked against a file and intersected per operand.
    Raises ValueError for a product the file neither holds nor can derive, a
    unit that does not apply, a value outside the file's channels, band or
    time span, and conditions that cannot hold together; warns of conditions
    holding for every cell."""
    ctx = QuantityContext(time_reference_jd=facts.jd[0], bunit=facts.bunit)
    resolved, warnings = [], []
    intervals: dict = {}
    first_text: dict = {}
    for condition in conditions:
        operand, text = condition.operand, condition.text
        if operand.product is not None:
            try:
                plan_products([operand.product], facts.stokes_labels)
            except ValueError as err:
                raise _refuse(text, str(err)) from None
        lower, upper, unit = condition.lower, condition.upper, condition.unit
        if operand.name in ("amp", "real", "imag"):
            try:
                chosen = resolve_unit(operand.name, unit, ctx)
            except ValueError as err:
                raise _refuse(text, str(err)) from None
            factor = chosen.factor  # values in BUNIT
            lower = None if lower is None else Bound(lower.value / factor, lower.inclusive)
            upper = None if upper is None else Bound(upper.value / factor, upper.inclusive)
            unit = None
        elif operand.name == "time":
            low_text, high_text = condition.instants
            lower = None if lower is None else Bound(_instant_jd(low_text, unit, facts), lower.inclusive)
            upper = None if upper is None else Bound(_instant_jd(high_text, unit, facts), upper.inclusive)
            if lower is not None and upper is not None and Interval(lower, upper).empty:
                raise _refuse(text, "the range is empty: its lower end is not before its upper end")
            unit = "jd"
        elif operand.name == "freq":
            lower = None if lower is None else Bound(convert(lower.value, "freq", unit, "Hz"), lower.inclusive)
            upper = None if upper is None else Bound(convert(upper.value, "freq", unit, "Hz"), upper.inclusive)
            unit = "Hz"
        elif condition.unit is not None:
            base = resolve_unit(operand.name, condition.unit).base
            reference = next(u.name for u in units_of(operand.name) if u.base == base)
            lower = None if lower is None else Bound(convert(lower.value, operand.name, unit, reference), lower.inclusive)
            upper = None if upper is None else Bound(convert(upper.value, operand.name, unit, reference), upper.inclusive)
            unit = reference
        file_span = _file_span(operand, facts)
        if file_span is not None:
            for bound in (lower, upper):
                if bound is not None and not (file_span[0] - 1e-9 <= bound.value <= file_span[1] + 1e-9):
                    raise _refuse(text, f"{operand} in this file lies in {_span_text(operand, file_span)}; "
                                        "the condition's value is outside it")
        key = (operand.name, operand.product, unit)
        interval = Interval(lower, upper)
        if key in intervals:
            joined = intervals[key].intersect(interval)
            if joined.empty:
                raise ValueError(f"--where {first_text[key]!r} and --where {text!r} cannot both hold: no "
                                 f"{operand} satisfies both")
            intervals[key] = joined
        else:
            intervals[key], first_text[key] = interval, text
        resolved.append(replace(condition, unit=unit, lower=lower, upper=upper))
        whole = _domain_of(operand, facts, unit)
        if whole is not None and _covers(interval, whole):
            warnings.append(f"--where {text!r} holds for every cell")
    return ResolvedConditions(tuple(resolved), intervals, tuple(warnings))


def _file_span(operand: Operand, facts: FileFacts) -> tuple[float, float] | None:
    if operand.name == "channel":
        return (0.0, float(facts.n_channels - 1))
    if operand.name == "freq":
        return facts.freq_hz
    if operand.name == "time":
        return facts.jd
    return None


def _span_text(operand: Operand, span: tuple[float, float]) -> str:
    if operand.name == "freq":
        return f"{span[0] / 1e6:.6f}..{span[1] / 1e6:.6f} MHz"
    if operand.name == "time":
        first, last = (Time(v, format="jd", scale="tai").isot[:19] for v in span)
        return f"{first}..{last} as recorded"
    return f"{span[0]:g}..{span[1]:g}"


def _domain_of(operand: Operand, facts: FileFacts, unit: str | None) -> tuple[float, float] | None:
    """The whole range an operand can take in `unit`, where it is bounded on
    both sides or from below with nothing above (amplitude, uv distance)."""
    span = _file_span(operand, facts)
    if span is not None:
        return span
    return domain_in(operand.name, unit)


def _covers(interval: Interval, whole: tuple[float | None, float | None]) -> bool:
    lowest, highest = whole
    low_ok = interval.lower is None or (lowest is not None and (interval.lower.value < lowest or (
        interval.lower.value == lowest and interval.lower.inclusive)))
    high_ok = interval.upper is None or (highest is not None and (interval.upper.value > highest or (
        interval.upper.value == highest and interval.upper.inclusive)))
    return low_ok and high_ok and not (interval.lower is None and interval.upper is None)
