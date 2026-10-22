# visplot — User Guide

`visplot` plots and inspects GWB visibilities, from the command line
(`bin/visplot.sh`) or its GUI. This guide covers Locate: finding visibilities
that meet a condition, keeping what each search found, and working with that
across sessions.

## Contents

- [Locate: finding visibilities by condition](#locate-finding-visibilities-by-condition)
- [Sessions: where Locate's runs live](#sessions-where-locates-runs-live)
- [Visibilities the file already flags](#visibilities-the-file-already-flags)
- [Listing a session](#listing-a-session)
- [Run operations: undo and restore](#run-operations-undo-and-restore)

---

## Locate: finding visibilities by condition

`--locate`, given with no box, searches the current selection for every
visibility (a row, channel, product) meeting the conditions you give with
`--where`. It runs without a window, one pass over the data, and writes what
it finds to a run of a Locate session (below).

```bash
bin/visplot.sh obs.FITS --locate --where 'amp(V) > 5' --session dead-antennas
```

Give `--where` more than once for conditions that must all hold together; an
OR is a separate run. A condition is `OPERAND CMP VALUE UNIT` or a range
`A < OPERAND < B`:

| Example | Meaning |
|---|---|
| `amp(V) > 5` | Stokes V's amplitude above 5 (the file's BUNIT) |
| `amp(I) < 0.005` | Stokes I's amplitude below 0.005: dead antennas, for example |
| `amp(RL) > 3000 UNCALIB` | the RL product's amplitude above 3000, its unit written out |
| `phase(RL) >= 170 deg` | the RL product's phase at or above 170 degrees |
| `100 < uvdist < 2000 lambda` | uv distance between 100 and 2000 wavelengths |
| `el < 20 deg` | elevation below 20 degrees |
| `400 MHz <= freq <= 450 MHz` | channels from 400 to 450 MHz |
| `channel >= 1800` | channels 1800 onwards (0-based) |
| `time >= 1/17:42:09 UTC` | from 17:42:09 UTC on day 1 |

> [!IMPORTANT]
> A run needs at least one condition on a data value (`amp(...)`,
> `phase(...)`, `real(...)`, `imag(...)` of a product, or `I`/`Q`/`U`/`V`).
> Conditions on coordinates alone (`el`, `time`, `uvdist`, `channel`, ...)
> narrow where Locate looks, but don't by themselves say what counts as
> found — for those alone, use the plot's own selection options
> (`--el-range`, `--time-range`, `--channels`, ...) instead of `--locate`.

A run reports what it did and what it found as it goes:

```
Locate session demo, new: /path/to/runs/found/demo
Locate where amp(RR) > 0: reads RR and records them at each visibility found; visibilities the file flags skipped
searching 40 of the selection's 40 rows (cross-correlations), 4 channel(s): 0.0 GB to read
locating: 40 / 40 rows (100%), 0.0 GB read, 0:00 elapsed
found 159 visibilities (row, channel) meeting every condition, 39 rows whole; run 1 of session demo: ...
```

"Rows whole" counts rows where every searched channel of that row was found
— a storage shortcut for a row found complete. A row short by even one
channel is kept channel by channel, so Locate's channel-level detail holds
throughout.

---

## Sessions: where Locate's runs live

Each Locate run is added to a session: a named, numbered sequence of runs
kept on disk at `<provenance-dir>/found/<name>/`, one directory per session.
Name it with `--session`:

```bash
bin/visplot.sh obs.FITS --locate --where 'amp(V) > 5' --session dead-antennas
```

| | |
|---|---|
| Default name | the FITS file's name without its extension (any character a session name doesn't take becomes `_`) |
| Allowed characters | letters, digits, `.`, `_`, `-` |
| On disk | `<provenance-dir>/found/<name>/` |

A session belongs to one FITS file, checked by name, size and row count the
first time you give that session name against a given file. Using the same
session name against a different file is refused:

```
session 'demo' (.../found/demo) belongs to obs.fits (1,234 bytes, 40 rows), and this
file is other.fits (5,678 bytes, 60 rows): give --session a name for this file's runs
```

> [!IMPORTANT]
> **A session is files on disk** — nothing about it depends on the
> process that wrote it still running, or on today being the day it was
> written. Point `--session` at the same name tomorrow, from a different
> terminal, a different working directory, even a different machine (if it
> can see the same `--provenance-dir`), and it reopens exactly where it left
> off: the same run numbers, the same found set, ready for another run,
> a listing, or `--undo`/`--restore`. Nothing needs to be re-run to pick a
> session back up.

Every run adds to the session. Nothing a Locate run writes is edited or
erased once committed. A session's **found set** — what flag generation
eventually reads — is the union of whichever runs are currently active
(below).

---

## Visibilities the file already flags

`--flagged` says which visibilities are candidates for a run, alongside the
`--where` conditions:

| `--flagged` | Candidates |
|---|---|
| `exclude` (default) | unflagged visibilities only |
| `only` | already-flagged visibilities only — a diagnosis of what's flagged and why |
| `include` | both |

```bash
bin/visplot.sh obs.FITS --locate --where 'amp(RR) > 0' --flagged only --session demo
```

A visibility counts as already flagged when the file's weight for it is at
or below zero; a derived product (Stokes `I`, `V`, ...) counts as flagged
where any product it's built from does.

---

## Listing a session

`--session NAME` given by itself, with no `--locate`/`--where`, lists that
session's runs instead of running anything:

```bash
bin/visplot.sh obs.FITS --session demo
```

```
Locate session demo: obs.fits, 40 rows
  run  status  conditions   cells found  rows whole  written (UTC)
    1  active  amp(RR) > 0          159          39  2026-10-22T10:32:59+00:00
    2  active  amp(RR) > 0            1           0  2026-10-22T10:33:14+00:00
  found set: the union of runs 1, 2
```

A run's status is `active` (counted in the found set) or `aside` (kept, not
counted — see run operations below). An empty session says `no runs yet`; a
session with every run aside says `found set: empty, no run active`.

---

## Run operations: undo and restore

`--undo RUN` sets a run of a session aside: it stays on disk, out of the
found set, and can be brought back. `--restore RUN` makes an aside run active
again. Both run on the session by itself — no `--where`, no plots:

```bash
bin/visplot.sh obs.FITS --undo 2 --session demo
bin/visplot.sh obs.FITS --restore 2 --session demo
```

Each prints the session's listing afterward, so you see the result
immediately. A run operation is refused if the run doesn't exist in that
session, or is already in the state you asked for:

```
run 2 of session 'demo' is already aside
```

> [!NOTE]
> Undo and restore are the first two run operations built. `--replace`,
> `--intersect`, and `--difference` are designed but not yet available.
