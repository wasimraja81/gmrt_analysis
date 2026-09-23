# src/provenance/

Reproducibility foundation used by every stage.

- `manifest.py` (`RunManifest`) — per-invocation manifest capture: git commit + dirty-diff,
  resolved parameters, `run_id`, input-data identity, outcome. T1.
- `logging_setup.py` — the unified logging convention: per-run timestamped file + a
  `<stage>_latest.log` symlink, wired into `RunManifest` so the log shares the manifest's
  `run_id`. No stage relies on a shell wrapper's `tee` for persistence. T3.
- `run_index.py` — the append-only `runs_index.jsonl` "lab notebook" spanning every stage
  and run, written to by `RunManifest` on exit. T4.

Phase A (foundation) is complete as of T4 — every future stage (T5+) gets a manifest, a
logger, and a run-index entry automatically just by using `RunManifest`.
