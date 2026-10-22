"""Locate sessions: what Locate finds, kept for flag generation (T26's flagging
design, step 2b).

A session is a directory, `<provenance dir>/found/<name>/`, belonging to one
FITS file:

    session.jsonl    append-only, one JSON line per entry: the session's opening
                     (the FITS file it belongs to), then a "run" or "operation"
                     entry (`run_status`) for each run or run operation in turn
    run-001.npz      one Locate run: a JSON header, the rows it searched (packed
    run-002.npz      bits) and its records, one array per field
    .lock            held while an entry is added; runs read their data meanwhile

A run is written as it is found: each field's values go, chunk by chunk, to a
raw file in the run's partial directory. When the run ends they are assembled
into one `.npz` (each raw file deleted once copied, so the run needs little
more than its own size on disk), renamed into place as the next run number,
and its line appended to the log. A run stopped or killed before its rename
leaves the session as it was (a later opening removes its partial files). A
file is renamed into place only once written whole and on disk, and its log
line is a copy of the summary in its header; so a run killed between its
rename and its log line is logged at the next opening from its header, once
every field is checked against the header (`check_run_file`), the line saying
it was written late. A file failing that check is renamed unlogged-run-NNN.npz,
kept and never logged. Run numbers are never reused. Nothing in a session is
edited or removed.

A run has two tables: `cells`, a record per visibility found (row, channel,
product), and `rows`, a record per product of each row found whole (every
searched cell of the row found, or skipped as flagged by the file).

A run is never edited or erased; each is active or set aside (`run_status`,
folding the log's operation entries over its run entries), and the session's
found set, the flag generator's input, is the union of the active ones
(`read_active`). `undo_run`/`restore_run` flip a run's status, each its own
operation entry.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import socket
import zipfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1
LOG_NAME = "session.jsonl"
_RUN_FILE = re.compile(r"^run-(\d+)\.npz$")
_UNLOGGED_FILE = re.compile(r"^unlogged-run-(\d+)\.npz$")
_PARTIAL = re.compile(r"^\.partial-([^-]+)-(\d+)-\d+(\.npz)?$")
_COPY_BYTES = 16 * 1024 * 1024
# A run stops, keeping nothing, before it would leave less than this much of its disk free:
# 1% of the filesystem, at least 1 GB.
RESERVE_FRACTION, RESERVE_MIN_BYTES = 0.01, 10**9


@dataclass(frozen=True)
class FileIdentity:
    """The FITS file a session belongs to: its name, size and number of rows
    (compared); its path when the session was opened, for information only
    (the data may move)."""

    name: str
    size_bytes: int
    gcount: int
    path: str

    @classmethod
    def of(cls, fits_path, gcount: int) -> FileIdentity:
        path = Path(fits_path).resolve()
        return cls(path.name, path.stat().st_size, int(gcount), str(path))

    def same_file(self, other: FileIdentity) -> bool:
        return (self.name, self.size_bytes, self.gcount) == (other.name, other.size_bytes, other.gcount)


class SessionError(ValueError):
    """A session that cannot take this run (another file's, unreadable)."""


class NoSpace(RuntimeError):
    """A run stopped before it would fill its disk; nothing kept."""


def size_text(n_bytes: float) -> str:
    """A size in GB, or MB below one GB."""
    return f"{n_bytes / 1e9:.1f} GB" if n_bytes >= 1e9 else f"{n_bytes / 1e6:.1f} MB"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _host() -> str:
    return re.sub(r"[^A-Za-z0-9.]", "_", socket.gethostname())


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class LocateSession:
    """A session's directory (module docstring). `open` it for a file; it
    is created on first use."""

    def __init__(self, directory: Path, name: str, identity: FileIdentity, created: bool,
                 removed_partials: list[Path], logged_late: list[int], set_aside: list[tuple[Path, str]]):
        self.directory = directory
        self.name = name
        self.identity = identity
        self.created = created
        self.removed_partials = removed_partials  # left by runs stopped or killed before their end
        self.logged_late = logged_late  # runs in place and checked whole, their log lines written at this opening
        self.set_aside = set_aside  # (file, what is wrong): run files failing `check_run_file`, renamed unlogged-...

    @classmethod
    def open(cls, found_dir, name: str, identity: FileIdentity) -> LocateSession:
        """Session `name` under `found_dir` for the file `identity`: created
        if new; refused (SessionError) if it belongs to another file."""
        directory = Path(found_dir) / name
        directory.mkdir(parents=True, exist_ok=True)
        with _locked(directory):
            log = directory / LOG_NAME
            created = not log.exists() or not log.read_text().strip()
            if created:
                _append_line(log, {"entry": "session", "name": name, "format": FORMAT_VERSION,
                                   "created_utc": _now(), "file": asdict(identity)})
            entries = read_log(directory)
            own = FileIdentity(**entries[0]["file"])
            if not own.same_file(identity):
                raise SessionError(
                    f"session {name!r} ({directory}) belongs to {own.name} ({own.size_bytes:,} bytes, "
                    f"{own.gcount:,} rows), and this file is {identity.name} ({identity.size_bytes:,} bytes, "
                    f"{identity.gcount:,} rows): give --session a name for this file's runs")
            removed = _remove_stale_partials(directory)
            logged_late, set_aside = _log_unlogged_runs(directory, entries)
        return cls(directory, name, own, created, removed, logged_late, set_aside)

    def runs(self) -> list[dict]:
        """The log's run entries, in order."""
        return [entry for entry in read_log(self.directory) if entry["entry"] == "run"]

    def status(self) -> dict[int, str]:
        """Each run number's status, "active" or "aside" (`run_status`)."""
        return run_status(read_log(self.directory))

    def active_runs(self) -> list[int]:
        """Run numbers whose status is "active", in run-number order: the
        session's found set, the flag generator's input, is their union."""
        return sorted(n for n, status in self.status().items() if status == "active")

    def run_path(self, number: int) -> Path:
        """Run `number`'s file -- active, aside, or (a derived run's own
        inputs) set aside by the operation that made it."""
        return self.directory / f"run-{number:03d}.npz"

    def new_run(self, fields: dict[str, dict[str, str]]) -> RunWriter:
        """A run being written: `fields` maps each table ("cells", "rows") to
        its fields and their dtypes, e.g. {"cells": {"row": "<u4", ...}}."""
        return RunWriter(self, fields)


def run_status(entries: list[dict]) -> dict[int, str]:
    """Each run number's status, "active" or "aside", folding the log's
    operation entries over its run entries in order: a run starts active
    when its own entry appears; "undo" and the inputs of "intersect"/
    "difference" set a run aside (kept, never edited or removed); "restore"
    makes one active again. A run never in the log (not yet committed, or
    set aside by its own replace before this session was last read) is
    absent, not "aside"."""
    status: dict[int, str] = {}
    for entry in entries:
        if entry["entry"] == "run":
            status[entry["run"]] = "active"
        elif entry["entry"] == "operation":
            op = entry["op"]
            if op == "undo":
                status[entry["target"]] = "aside"
            elif op == "restore":
                status[entry["target"]] = "active"
            elif op in ("intersect", "difference"):
                status[entry["a"]] = status[entry["b"]] = "aside"
    return status


def read_log(directory) -> list[dict]:
    """A session's log entries; a last line cut short (a write killed
    midway) is left out."""
    entries = []
    lines = (Path(directory) / LOG_NAME).read_text().splitlines()
    for number, line in enumerate(lines, 1):
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            if number < len(lines):
                raise SessionError(f"{Path(directory) / LOG_NAME}: line {number} is not JSON") from None
    if not entries or entries[0].get("entry") != "session":
        raise SessionError(f"{Path(directory) / LOG_NAME} does not start with the session's entry")
    return entries


@contextmanager
def _locked(directory: Path):
    with open(directory / ".lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _append_line(log: Path, entry: dict) -> None:
    with open(log, "a") as f:
        f.write(json.dumps(entry) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _remove_stale_partials(directory: Path) -> list[Path]:
    """Partial files of this host's runs that are no longer running."""
    removed = []
    for path in sorted(directory.glob(".partial-*")):
        match = _PARTIAL.match(path.name)
        if match and match.group(1) == _host() and not _pid_alive(int(match.group(2))):
            shutil.rmtree(path) if path.is_dir() else path.unlink()
            removed.append(path)
    return removed


def _run_numbers(directory: Path, entries: list[dict]) -> set[int]:
    """Every run number taken: logged, in place, or set aside."""
    files = {int(m.group(1)) for p in directory.iterdir()
             if (m := _RUN_FILE.match(p.name) or _UNLOGGED_FILE.match(p.name))}
    return files | {entry["run"] for entry in entries if entry["entry"] == "run"}


def _log_unlogged_runs(directory: Path, entries: list[dict]) -> tuple[list[int], list[tuple[Path, str]]]:
    """Run files in place with no log line: a run killed between its rename
    and its log line. A file checked whole (`check_run_file`) is logged
    from its header, the line saying it was written late; a file that is
    not is renamed unlogged-run-NNN.npz, kept and never logged."""
    logged = {entry["run"] for entry in entries if entry["entry"] == "run"}
    logged_late, set_aside = [], []
    for path in sorted(directory.glob("run-*.npz")):
        match = _RUN_FILE.match(path.name)
        if not match or int(match.group(1)) in logged:
            continue
        number = int(match.group(1))
        problem = check_run_file(path)
        if problem is None:
            _append_line(directory / LOG_NAME, {
                **_log_entry(number, path, read_header(path)), "logged_late_utc": _now(),
                "logged_late": "the run's file was in place without this line (its process stopped between the "
                               "two); written at the session's next opening from the file's header"})
            logged_late.append(number)
        else:
            aside = path.with_name(f"unlogged-{path.name}")
            os.replace(path, aside)
            set_aside.append((aside, problem))
    return logged_late, set_aside


def check_run_file(path) -> str | None:
    """What is wrong with a run file, None when it is whole: its header
    readable, and every field its header names present, of its dtype, with
    as many values as its header counts and all their bytes."""
    try:
        header = read_header(path)
        with zipfile.ZipFile(path) as archive:
            for table, fields in header["fields"].items():
                for name, dtype in fields.items():
                    member = f"{table}_{name}.npy"
                    with archive.open(member) as f:
                        version = np.lib.format.read_magic(f)
                        read = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                                else np.lib.format.read_array_header_2_0)
                        shape, _, member_dtype = read(f)
                        header_bytes = f.tell()
                    n = header["counts"][table]
                    if shape != (n,) or member_dtype != np.dtype(dtype):
                        return f"{member}: {shape} of {member_dtype}, the header gives ({n},) of {dtype}"
                    if archive.getinfo(member).file_size != header_bytes + n * np.dtype(dtype).itemsize:
                        return f"{member}: its values are not all there"
    except Exception as err:  # an unreadable file: what is wrong with it
        return f"{type(err).__name__}: {err}"
    return None


def _log_entry(number: int, path: Path, header: dict) -> dict:
    return {"entry": "run", "run": number, "file": path.name, **header["summary"]}


def read_header(path) -> dict:
    """A run file's JSON header."""
    with np.load(path) as run:
        return json.loads(str(run["header"]))


class RunWriter:
    """One run's records, written as they are found (module docstring)."""

    def __init__(self, session: LocateSession, fields: dict[str, dict[str, str]]):
        self.session = session
        self.fields = {table: {name: np.dtype(dtype) for name, dtype in table_fields.items()}
                       for table, table_fields in fields.items()}
        self.partial = session.directory / f".partial-{_host()}-{os.getpid()}-{id(self)}"
        self.partial.mkdir()
        self.counts = {table: 0 for table in self.fields}
        self._files = {(table, name): open(self.partial / f"{table}_{name}.raw", "wb")
                       for table, table_fields in self.fields.items() for name in table_fields}

    @property
    def bytes_written(self) -> int:
        return sum(self.counts[table] * sum(d.itemsize for d in fields.values())
                   for table, fields in self.fields.items())

    def append(self, table: str, columns: dict[str, np.ndarray]) -> None:
        """Records of `table`, one array per field, all of one length."""
        fields = self.fields[table]
        if set(columns) != set(fields):
            raise ValueError(f"{table}: fields {sorted(columns)}, expected {sorted(fields)}")
        lengths = {len(values) for values in columns.values()}
        if len(lengths) != 1:
            raise ValueError(f"{table}: fields of different lengths {sorted(lengths)}")
        n = lengths.pop()
        if not n:
            return
        for name, dtype in fields.items():
            self._files[table, name].write(np.ascontiguousarray(columns[name], dtype=dtype).tobytes())
        self.counts[table] += n

    def check_space(self, fraction_done: float) -> None:
        """Raises NoSpace when this run, projected from `fraction_done` of its
        pass, would leave less than the reserve of its disk free: the rest of
        its records, and its largest field again while the file is assembled."""
        usage = shutil.disk_usage(self.partial)
        reserve = max(RESERVE_MIN_BYTES, int(RESERVE_FRACTION * usage.total))
        written = self.bytes_written
        projected = written / fraction_done if fraction_done > 0 else written
        largest = max(d.itemsize for fields in self.fields.values() for d in fields.values())
        record = max(sum(d.itemsize for d in fields.values()) for fields in self.fields.values())
        needed = (projected - written) + projected * largest / record
        if needed > usage.free - reserve:
            raise NoSpace(
                f"this run would need about {size_text(projected + projected * largest / record)} "
                f"({size_text(written)} written at {fraction_done:.0%} of its pass), and its disk has "
                f"{size_text(usage.free)} free, of which {size_text(reserve)} is kept free; nothing kept: "
                "narrow the search (sources, channels, time) or the conditions")

    def commit(self, header: dict, searched_rows: np.ndarray) -> tuple[int, Path]:
        """The run assembled, renamed into place as the next run number and
        logged. `header` (JSON) carries a "summary", the run's log line;
        `searched_rows` the rows it searched (bool, one per row of the file).
        Returns the run number and its file."""
        for f in self._files.values():
            f.close()
        assembled = self.session.directory / f"{self.partial.name}.npz"
        header = {**header, "format": FORMAT_VERSION, "session": self.session.name, "counts": dict(self.counts),
                  "gcount": self.session.identity.gcount,
                  "fields": {table: {name: dtype.str for name, dtype in fields.items()}
                             for table, fields in self.fields.items()}}
        with zipfile.ZipFile(assembled, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            _write_member(archive, "header", np.array(json.dumps(header)))
            _write_member(archive, "searched_rows", np.packbits(np.asarray(searched_rows, dtype=bool)))
            for (table, name), _ in self._files.items():
                raw = self.partial / f"{table}_{name}.raw"
                with archive.open(f"{table}_{name}.npy", "w", force_zip64=True) as member:
                    np.lib.format.write_array_header_1_0(member, {
                        "descr": np.lib.format.dtype_to_descr(self.fields[table][name]),
                        "fortran_order": False, "shape": (self.counts[table],)})
                    with open(raw, "rb") as source:
                        shutil.copyfileobj(source, member, _COPY_BYTES)
                raw.unlink()
        with open(assembled, "rb+") as f:
            os.fsync(f.fileno())
        shutil.rmtree(self.partial)
        with _locked(self.session.directory):
            number = max(_run_numbers(self.session.directory, read_log(self.session.directory)), default=0) + 1
            path = self.session.directory / f"run-{number:03d}.npz"
            os.replace(assembled, path)
            _fsync_dir(self.session.directory)
            _append_line(self.session.directory / LOG_NAME, _log_entry(number, path, header))
        return number, path

    def abort(self) -> None:
        """Nothing of this run kept."""
        for f in self._files.values():
            f.close()
        shutil.rmtree(self.partial, ignore_errors=True)
        (self.session.directory / f"{self.partial.name}.npz").unlink(missing_ok=True)


def _write_member(archive: zipfile.ZipFile, name: str, array: np.ndarray) -> None:
    with archive.open(f"{name}.npy", "w", force_zip64=True) as member:
        np.lib.format.write_array(member, array, allow_pickle=False)


def _fsync_dir(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@dataclass(frozen=True)
class FoundRun:
    """One run read back: its header, the rows it searched, and its tables."""

    header: dict
    searched_rows: np.ndarray  # bool, one per row of the file
    cells: dict[str, np.ndarray]
    rows: dict[str, np.ndarray]


def read_run(path) -> FoundRun:
    """A run file, read whole (a test's or a small run's reader; the flag
    generator reads field by field)."""
    with np.load(path) as run:
        header = json.loads(str(run["header"]))
        searched = np.unpackbits(run["searched_rows"])[:header["gcount"]].astype(bool)
        tables = {table: {name: run[f"{table}_{name}"] for name in fields}
                  for table, fields in header["fields"].items()}
    return FoundRun(header, searched, tables.get("cells", {}), tables.get("rows", {}))


def undo_run(session: LocateSession, target: int, reason: str | None = None) -> None:
    """Sets run `target` of `session` aside: kept, restorable with
    `restore_run` (`run_status`). `reason` (e.g. "replaced by run 3", used by
    `replace`), kept with the entry. Raises ValueError if `target` was never
    a run of this session, or is already aside."""
    with _locked(session.directory):
        status = run_status(read_log(session.directory))
        if target not in status:
            raise ValueError(f"run {target} is not a run of session {session.name!r}")
        if status[target] == "aside":
            raise ValueError(f"run {target} of session {session.name!r} is already aside")
        entry = {"entry": "operation", "op": "undo", "target": target, "written_utc": _now()}
        if reason is not None:
            entry["reason"] = reason
        _append_line(session.directory / LOG_NAME, entry)


def restore_run(session: LocateSession, target: int) -> None:
    """Makes run `target` of `session` active again (`run_status`). Raises
    ValueError if `target` was never a run of this session, or is already
    active."""
    with _locked(session.directory):
        status = run_status(read_log(session.directory))
        if target not in status:
            raise ValueError(f"run {target} is not a run of session {session.name!r}")
        if status[target] == "active":
            raise ValueError(f"run {target} of session {session.name!r} is already active")
        entry = {"entry": "operation", "op": "restore", "target": target, "written_utc": _now()}
        _append_line(session.directory / LOG_NAME, entry)


def read_active(session: LocateSession) -> list[FoundRun]:
    """The session's found set: its active runs, read whole (`read_run`; a
    small-session reader -- the flag generator reads each active run's file
    field by field instead of holding every one in memory at once). A cell
    found by more than one active run appears once per run that found it;
    merging them is the flag generator's own job."""
    return [read_run(session.run_path(number)) for number in session.active_runs()]
