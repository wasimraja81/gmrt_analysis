# src/engine/

Telescope-agnostic calibration algorithms: bandpass solve, flagging, clustering,
secondary calibration, split, imaging. No hardcoded antenna counts, antenna names, or
FITS-format assumptions belong here — that's GMRT-specific knowledge, and it lives in
`../instruments/gmrt/` instead, supplied to this layer as plain data (arrays, not file
paths or telescope concepts). See `../../docs/dev/GWB_PIPELINE_REFACTOR_PLAN.md`, standing
rule 8, for the process: check the archived GSB code, understand why it did what it did,
then classify each piece as general or GMRT-specific before writing anything new.

Written fresh — informed by reading the archived GSB engine
(`../../legacy_gsb_40_014/src/modules/ugmrt_query.py` and siblings, plus its design docs
like `../../legacy_gsb_40_014/FLAGGING_DESIGN_NOTES.md`) for behavioral intent, never by
copying files or names forward from it.

- `bandpass_solve.py` — per-channel StefCal-style gain solve. T5a.
