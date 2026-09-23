# GWB Pipeline — User Guide

This guide explains what the GWB calibration pipeline does, from your point of view as
the person running it or reading its results. Each section covers one stage of the
pipeline.

## Contents

- [Run records (provenance)](#run-records-provenance)
- [Reading raw data safely](#reading-raw-data-safely)

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
