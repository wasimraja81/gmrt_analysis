"""Located samples written to CSV as they are found.

`LocateCsvWriter` is a `LocateReducer` sink: each chunk's samples are written
as they arrive, so the file holds every located sample while memory holds
one chunk's worth. The file is written as `<name>.<pid>.partial` and renamed
to its name when the pass completes; an interrupted pass removes the
partial file.

Columns give both readable and exact values: baseline by antenna names and
by station numbers, UTC time and the recorded JD, channel index and frequency, Stokes,
the plot's x and y (in the axes' units), the weight, and whether the sample
is a mirrored (conjugate) point. Lines starting with '#' describe the file:
the plot, what x and y hold and in which unit, the box, and (at the end,
once known) the total and the count by baseline.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy.time import Time

from visplot.quantities import utc_jd, value_description

COLUMNS = ("baseline", "ant1", "ant2", "time_utc", "jd_recorded", "channel", "freq_mhz", "stokes",
           "x", "y", "weight", "mirrored", "row", "source")


class LocateCsvWriter:
    def __init__(self, path, locate, ctx, fits_path=None, command: str | None = None):
        self.path = Path(path)
        self.partial = self.path.with_name(f"{self.path.name}.{os.getpid()}.partial")
        self.ctx = ctx
        self.n_written = 0
        self._file = open(self.partial, "w", newline="")
        self._writer = csv.writer(self._file)
        plot = locate.plot
        self._comment(f"located samples of plot: {plot.title}")
        self._comment(f"y: {value_description(plot.y, ctx, plot.y_unit)}")
        self._comment(f"x: {value_description(plot.x, ctx, plot.x_unit)}")
        self._comment(f"box: x {locate.x_box[0]:.10g} to {locate.x_box[1]:.10g}, "
                      f"y {locate.y_box[0]:.10g} to {locate.y_box[1]:.10g}")
        if fits_path is not None:
            self._comment(f"file: {Path(fits_path).resolve()}")
        if command:
            self._comment(f"command: {command}")
        self._comment(f"written: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
        system = f" (TIMSYS {ctx.time_system})" if ctx.time_system else ""
        self._comment(f"jd_recorded: the file's timestamp{system}; time_utc: recorded - "
                      f"{ctx.recorded_minus_utc_s:g} s")
        self._comment("channel -1 and Stokes 'all': the plot does not vary along that axis")
        self._writer.writerow(COLUMNS)

    def _comment(self, text: str) -> None:
        self._file.write(f"# {text}\n")

    def __call__(self, columns: dict) -> None:
        n = len(columns["row"])
        if not n:
            return
        names = self.ctx.antenna_names or {}
        sources = self.ctx.source_names or {}
        pairs = np.stack([columns["ant1"], columns["ant2"]], axis=1)
        unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
        labels = np.array([f"{names.get(int(a), a)}-{names.get(int(b), b)}" for a, b in unique], dtype=object)
        baseline = labels[inverse.ravel()]
        times = Time(utc_jd(self.ctx, columns["jd"]), format="jd", scale="utc").isot
        source = [sources.get(int(s), str(int(s))) for s in columns["source_id"]]
        freq_mhz = columns["freq_hz"] / 1e6
        self._writer.writerows(zip(
            baseline, columns["ant1"].tolist(), columns["ant2"].tolist(), times,
            [f"{v:.10f}" for v in columns["jd"]], columns["channel"].tolist(),
            ["" if np.isnan(f) else f"{f:.6f}" for f in freq_mhz], columns["stokes"],
            [f"{v:.10g}" for v in columns["x"]], [f"{v:.10g}" for v in columns["y"]],
            ["" if np.isnan(w) else f"{w:.6g}" for w in columns["weight"]],
            ["yes" if m else "no" for m in columns["mirrored"]], columns["row"].tolist(), source,
        ))
        self.n_written += n

    def close(self, locate, completed: bool) -> Path | None:
        """Finish the file: with the totals and rename it if the pass
        completed, otherwise remove the partial file. Returns the path
        written, or None."""
        try:
            if completed:
                names = self.ctx.antenna_names or {}
                self._comment(f"total: {locate.n_found} samples in the box, all listed above")
                for (a, b), n in sorted(locate.by_baseline.items(), key=lambda kv: -kv[1]):
                    self._comment(f"baseline {names.get(a, a)}-{names.get(b, b)}: {n}")
        finally:
            self._file.close()
        if completed:
            os.replace(self.partial, self.path)
            return self.path
        self.partial.unlink(missing_ok=True)
        return None


def write_kept(path, locate, ctx, fits_path=None, command=None) -> Path:
    """Write a locate's samples from memory, when all of them were kept
    (`n_found` <= its limit), without reading the selection again."""
    writer = LocateCsvWriter(path, locate, ctx, fits_path=fits_path, command=command)
    for columns in locate.kept_columns():
        writer(columns)
    return writer.close(locate, completed=True)
