# bin/

Thin, GWB-only stage-runner scripts for the new pipeline. Each script does argument
parsing, path setup, and a call into `src/`; the scientific logic itself lives in
`src/engine/`, not here.

Empty until the first stage ticket lands — see `../docs/dev/GWB_PIPELINE_REFACTOR_PLAN.md`.
