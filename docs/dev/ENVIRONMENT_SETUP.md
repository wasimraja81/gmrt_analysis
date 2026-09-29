# Environment Setup (Prerequisites)

## Rule: venv-only, never system Python

All Python work in this repo runs inside the `gmrt/` virtualenv at the repo root. Never
`pip install` into system Python for this project — every dependency goes into `gmrt/`.

## Building the venv: `bin/build_venv.sh`

```bash
cd /path/to/gmrt_analysis
bin/build_venv.sh           # create gmrt/ if it is missing, else bring it to the pins
bin/build_venv.sh --clear   # delete gmrt/ and build it afresh
```

One script builds all of it, so no part can be left out:

1. creates `gmrt/` (`python3 -m venv --clear gmrt`, then upgrades pip) -- with `--clear`,
   or when there is no venv yet; `PYTHON=python3.12 bin/build_venv.sh` picks the Python;
2. installs the packages pinned in `config/requirements.txt`;
3. installs the fonts pinned in `config/fonts.txt` (below);
4. checks: the main packages import, and every pinned font is installed from its pin.

Run on a venv that already matches the pins, it changes nothing and ends with the checks,
so it also serves as the verification. To check the fonts alone:
`bin/install_fonts.sh --check`.

## Fonts: `config/fonts.txt`

visplot draws the panel under each plot in a font loaded from its file, never looked up
among the system's fonts, so the same command draws the same pixels on every machine
built this way (T41). `config/fonts.txt` pins each font -- version, the publisher's
download URL, and the download's SHA-256 -- as `config/requirements.txt` pins packages.
The venv build (`bin/build_venv.sh`, step 3; on its own, `bin/install_fonts.sh`)
downloads each into the repository's `tmp/`, checks the SHA-256,
unpacks it unmodified into `gmrt/share/fonts/<name>/` with a `SOURCE.txt` saying where
it came from, and deletes the download; a font already installed from the same pin is
left as it is. The script exists because pip cannot install these fonts: none of the names
tried for TeX Gyre or Liberation on PyPI exists (2026-09-29).
The pinned font is TeX Gyre Heros 2.004 (Helvetica's metric clone, from GUST, under the
GUST Font License); visplot's `--panel-font dejavu-sans` uses matplotlib's own DejaVu
Sans instead. Without the installed font, a visplot run stops and says to run
`bin/build_venv.sh`.

## What's in `config/requirements.txt`

The FITS/numerical/plotting stack used by the calibration and diagnostics code: astropy,
numpy, scipy, matplotlib, scikit-image, imageio, PyYAML, and their transitive
dependencies. Generated via `gmrt/bin/pip freeze` after a clean install — regenerate the
same way after adding a new dependency, so the file always reflects what's installed,
not an aspirational list.

## What's NOT in this venv

**CASA (`casatasks`/`casatools`) is not installed here.** The selfcal/imaging stages
(archived reference: `legacy_gsb_40_014/src/gmrt_selfcal_dev.py`,
`legacy_gsb_40_014/src/moon_selfcal_dev.py`) import `casatasks`, which must come from a
separate CASA environment — identifying and documenting that environment is still
outstanding (not yet needed until the imaging-stage tickets in
`GWB_PIPELINE_REFACTOR_PLAN.md`, Phases F/G).

## History

The venv was found broken on 2026-09-23: it had been built against
`/usr/bin/python3.10`, which was removed from the system during an OS upgrade (system
Python is now 3.12). `gmrt/bin/python` had silently kept resolving to the new system
`python3` symlink, so it ran 3.12 against a `site-packages` built for 3.10 — not just
outdated, unable to `import numpy` at all. Rebuilt clean against Python 3.12 the same day;
current installed versions are current-compatible releases, not pinned to what GSB-era
work originally used (see `legacy_gsb_40_014/ARCHIVE_INDEX.md` for what that was, if a
version-specific regression ever needs comparing against).
