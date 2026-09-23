# src/cli/

Stage entry points invoked by `bin/*.sh`. Dispatches into `src/engine/`, wiring up
`src/provenance/` (manifest + logging) and `src/data_io/` (read-only data access) around each
call so no individual stage script has to remember to do that itself.

Empty until the first stage ticket lands — see `../../docs/dev/GWB_PIPELINE_REFACTOR_PLAN.md`.
