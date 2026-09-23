# src/provenance/

Reproducibility foundation used by every stage: per-invocation manifest capture (git
commit + dirty-diff, resolved parameters, RUN_ID, input-data identity, outcome), the
append-only run index (the "lab notebook" layer), and the unified logging setup (one
`logging`-based convention, per-run timestamped file + `_latest` symlink, used by every
stage script — no stage relies on a shell wrapper's `tee` for persistence).

Empty until T1/T3/T4 land — see `../../docs/dev/GWB_PIPELINE_REFACTOR_PLAN.md`.
