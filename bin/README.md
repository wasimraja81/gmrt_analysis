# bin/

Thin, GWB-only scripts. Each does argument parsing and path setup, then calls into
`src/`, where the scientific logic lives.

- `run_gwb_pipeline.sh` — the pipeline's stages (`src/cli/`).
- `visplot.sh` — the plotting tool: the GUI with no arguments, otherwise the command line
  (`src/cli/run_visplot.py`; see `bin/visplot.sh --help`).
- `build_venv.sh` — builds the `gmrt/` venv, all of it: the packages in
  `config/requirements.txt`, the fonts in `config/fonts.txt`, then a check
  (`--clear` builds it afresh; see `../docs/dev/ENVIRONMENT_SETUP.md`).
- `install_fonts.sh` — the font step of `build_venv.sh` on its own (`src/cli/install_fonts.py`;
  `--check` installs nothing and says whether every pinned font is installed).

See `../docs/dev/GWB_PIPELINE_REFACTOR_PLAN.md` for the tickets behind them.
