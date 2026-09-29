# config/

GWB-specific pipeline parameters (channel ranges, flagging/clustering thresholds, data
paths). Written fresh for GWB — nothing is silently inherited from the archived
`legacy_gsb_40_014/preprocess_ugmrt.cfg` GSB defaults; every value here should be traceable
to a GWB-specific reason (measurement, tuning run, or explicit decision), not carried over
because it happened to work for GSB.

`requirements.txt` — the frozen `gmrt/` venv dependency list. See
`../docs/dev/ENVIRONMENT_SETUP.md` for rebuild instructions.

`fonts.txt` — the fonts visplot draws with, pinned by version, download URL and SHA-256;
building the venv (`../bin/build_venv.sh`) installs them into it, with the packages of
`requirements.txt` (see the same document).
