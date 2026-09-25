# GWB Pipeline — User Guide

This guide explains what the GWB calibration pipeline does, from your point of view as
the person running it or reading its results. Each section covers one stage of the
pipeline.

## Contents

- [Run records (provenance)](#run-records-provenance)
- [Reading raw data safely](#reading-raw-data-safely)
- [Finding out what a run did (logs)](#finding-out-what-a-run-did-logs)
- [Finding past runs (the run index)](#finding-past-runs-the-run-index)
- [Running the pipeline](#running-the-pipeline)
- [Building the row index](#building-the-row-index)

---

## Run records (provenance)

Every time a pipeline stage runs, it automatically saves a record of that run: what
stage it was, when it ran, exactly what parameters were used, which version of the code
produced it, which input file it read, what it produced, and whether it succeeded or
failed. You don't have to do anything to get this — it happens on every run, with no
extra flag to remember.

> [!TIP]
> **What this gives you**
> - If a result looks wrong later, you can check exactly what was run to produce it —
>   no reconstructing it from memory or old terminal scrollback.
> - You can hand a collaborator the exact recipe for any result you've produced.
> - If the code had uncommitted changes when a run happened — common mid-tuning, like
>   the threshold-sweeping sessions on the old GSB pipeline — the record captures that
>   too, alongside a copy of the change itself, so nothing is silently unreproducible.

### Where to find it

| Location | What's there |
|---|---|
| `<work_dir>/provenance/<stage name>/<run's ID>.json` | One file per run: parameters, timing, git state, inputs/outputs, outcome. |
| `<work_dir>/provenance/<stage name>/<run's ID>.diff` | Only present if the code had uncommitted changes at the time — a full copy of that change. |

> [!IMPORTANT]
> A run's ID is unique and sortable by time, so listing a stage's folder and reading the
> newest file always gets you the most recent run.

### Example

Once a pipeline stage is wired up to write one of these (starting with primary
calibration), a run record looks like this — trimmed here to the fields that matter most:

```jsonc
{
  "run_id": "20260923T143000Z_a1b2c3d4",
  "stage": "primary_calibration",
  "started_at_utc": "2026-09-23T14:30:00+00:00",
  "finished_at_utc": "2026-09-23T14:47:12+00:00",

  // Exactly which version of the pipeline produced this. If "dirty" is true, see the
  // .diff file alongside this one for exactly what was different from that commit.
  "git": { "commit": "3c1a9f...", "branch": "40_014-gwb", "dirty": false },

  // Every parameter this run used — not just the ones you explicitly set.
  "parameters": { "chan_start": 1731, "chan_end": 1901, "clustering_threshold_jy": 500.0 },

  // What it read and what it produced.
  "inputs": [{ "path": "/data1/gmrt/40_014_25JUL2021/40_014_25jul2021_2.6s_gwb.FITS", "size_bytes": 389403512640 }],
  "outputs": ["/data1/gmrt/40_014/work/primary_calibration/bandpass/3c48_bandpass_final.npz"],

  // Did it work?
  "outcome": { "status": "success", "error_type": null }
}
```

> [!NOTE]
> The input file's identity is recorded as its path and size, not a checksum — hashing a
> 389 GB raw data file on every run would make each run far slower for no practical
> benefit. Path and size are enough to catch "this ran against a different file than last
> time."

---

## Reading raw data safely

The pipeline's data-access layer only ever opens your raw GMRT data file in a read-only
mode, and every output a stage registers is checked automatically against the raw file it
read — a stage cannot register an output that would land on top of its own input.

> [!TIP]
> **What this gives you**
> - The functions the pipeline uses to open your raw data cannot be asked to open it
>   writable — there's no argument or setting that switches that on.
> - If an output path is ever misconfigured to point at the raw input (even indirectly,
>   through a shortcut/symlink or a `..`-containing path), registering that output fails
>   with a clear error instead of silently overwriting your data.

> [!IMPORTANT]
> This check runs automatically the moment a stage registers an output.

---

## Finding out what a run did (logs)

Every run writes its own log file — a plain-text, timestamped record of what the stage
did while it ran, separate from the JSON run record described above. The run record
tells you the parameters and outcome; the log tells you the story of how it got there.

### Where to find it

| Location | What's there |
|---|---|
| `<work_dir>/logs/<stage name>/<run's ID>.log` | The full log for one run, from start to finish. |
| `<work_dir>/logs/<stage name>/<stage name>_latest.log` | Always points at the most recent run for that stage. |

> [!TIP]
> **What this gives you**
> - `tail -f <work_dir>/logs/primary_calibration/primary_calibration_latest.log` follows
>   whichever run is currently in progress, without you needing to know its run ID.
> - The log and the run record for the same run share the same ID, so if you're looking
>   at one, you can find the other by name.
> - If a run fails, the log's last lines say what went wrong, in the same place you'd
>   look to see what the run was doing right before that.

---

## Finding past runs (the run index)

Every run — every stage, from the beginning of the pipeline's use — adds one line to a
single file: `<work_dir>/runs_index.jsonl`. Where the run record and the log are about
one run, this file is about all of them: a running history you can search across.

### Example queries

```bash
# Every run of a given stage, most recent last
grep '"stage": "primary_calibration"' work/runs_index.jsonl

# Only the ones that failed
grep '"stage": "primary_calibration"' work/runs_index.jsonl | grep '"outcome_status": "failed"'

# With jq: stage, clustering threshold, and outcome, side by side
jq -r '[.stage, .parameters.clustering_threshold_jy, .outcome_status] | @tsv' work/runs_index.jsonl
```

> [!TIP]
> **What this gives you**
> - A single place to ask "what did I try, and what happened" across the whole pipeline's
>   history, instead of opening one file per run.
> - Each line points back to its full run record and log file for more detail.

> [!NOTE]
> This file only ever has lines added to it — never rewritten — so it's safe to keep
> around and grow for as long as you use the pipeline.

---

## Running the pipeline

One config file per observation describes what to process and which stages to run.
`bin/run_gwb_pipeline.sh` reads it and runs whichever stages are switched on, in a fixed
order.

```bash
bin/run_gwb_pipeline.sh config/40_014_25jul2021_gwb.yaml
```

### The config file

```yaml
fits_path: /data1/gmrt/40_014_25JUL2021/40_014_25jul2021_2.6s_gwb.FITS
work_dir: /scratch/gmrt/40_014_25JUL2021/work_gwb

stages:
  build_index: true
```

| Field | What it means |
|---|---|
| `fits_path` | Your raw data file. Read-only, never modified. |
| `work_dir` | Where run records, logs, and the run index (described above) get written. Keep this on a fast disk — it's separate from where your raw data lives, and doesn't need to be. |
| `stages` | Which stages to run this time. A stage set to `false`, or left out, is skipped. |

> [!TIP]
> **What this gives you**
> - Re-running the same config reruns exactly the same stages against exactly the same
>   file — no flags to remember or retype.
> - Turning a stage off doesn't remove it from the pipeline; it just skips it for this
>   invocation. Flip it back on any time.
> - As more stages are added over time, they show up here as more entries under
>   `stages:` — the config grows with the pipeline, you don't need a new one.

> [!NOTE]
> `work_dir` isn't the only place output lands — some stages (the row index, described
> below) save their result next to your raw data file instead, because that output
> belongs with the data it describes. `work_dir` is specifically for run records, logs,
> and anything else about *how* a stage ran.

---

## Building the row index

Before the pipeline can select or read specific visibilities efficiently, it needs to
know, for every row in your raw file, which source and which antenna pair it belongs to.
Nothing in the file's header records this — it has to be read once, for every row. The
`build_index` stage does this one-time read and saves the result so nothing has to repeat
it.

Turn it on in your config:

```yaml
stages:
  build_index: true
```

### What you get

| Location | What's there |
|---|---|
| `<fits_path>.idx.npz` | The index itself — always saved next to your raw data file, never under `work_dir`. |

> [!IMPORTANT]
> The first run takes on the order of 30 minutes for a 389 GB file, because every row has
> to be touched once — that per-row information has no summary anywhere else in the file
> to shortcut past. Built once, it's reused automatically from then on — see below.

### It only rebuilds when it needs to

If `<fits_path>.idx.npz` already exists, running the stage again does nothing but confirm
that and record a run — it doesn't repeat the 30-minute read. To force a rebuild anyway
(for instance, if you have reason to think the raw file changed):

```yaml
stages:
  build_index:
    force_rebuild: true
```

### What the log tells you

The stage's log (see [Finding out what a run did](#finding-out-what-a-run-did-logs))
reports progress as it reads, and finishes with a summary — this one is from the
`40_014_25JUL2021` GWB observation:

```
index built: 3959928 rows, 28 active antennas, 2 dead-this-observation: ['C03:04', 'C10:10']
saved index to /data1/gmrt/40_014_25JUL2021/40_014_25jul2021_2.6s_gwb.FITS.idx.npz
```

"Dead-this-observation" antennas are ones your antenna table lists but that have no data
anywhere in this particular file — GMRT antennas taken out of service for maintenance
show up this way, distinct from a handful of antennas with a separate, permanent quirk in
how GMRT's antenna table itself is built (unrelated to whether they had data that day).
You don't need to configure either list; both are worked out from the antenna table and
the data itself.

> [!TIP]
> **What this gives you**
> - A count of which antennas contributed data to this observation, without having to
>   inspect the raw file by hand.
> - If the pipeline's antenna count doesn't match what the data itself shows — the kind
>   of mismatch that can otherwise go unnoticed for a while — the stage stops with a
>   clear error rather than producing a result built on a wrong assumption.
