# Environment Setup (Prerequisites)

## Rule: venv-only, never system Python

All Python work in this repo runs inside the `gmrt/` virtualenv at the repo root. Never
`pip install` into system Python for this project — every dependency goes into `gmrt/`.

## Rebuilding from scratch

```bash
cd /path/to/gmrt_analysis
python3 -m venv gmrt --clear
gmrt/bin/python -m pip install --upgrade pip
gmrt/bin/pip install -r config/requirements.txt
```

Verify:

```bash
gmrt/bin/python -c "import numpy, scipy, astropy, matplotlib, skimage, imageio, yaml; print('OK')"
```

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
