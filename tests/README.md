# Tests

Pytest-based unit tests for the ripple-characterisation feature (and any
other pure-numerics code that gets synthetic-data coverage going forward).

## Setup

Install test dependencies into the existing project venv — do not create a
separate Python environment:

```
gmrt/bin/pip install -r requirements-dev.txt
```

## Running

From the repo root:

```
gmrt/bin/python -m pytest -q
```

`pytest.ini` sets `pythonpath = src src/modules`, so tests import project
modules directly (`import ripple_characterisation as rc`,
`from modules import ugmrt_query as q`) without the `sys.path.insert` hack
used by the standalone CLI scripts.

## How this differs from `bin/test_*.sh`

This `tests/` directory covers **pure numerics only** — functions that take
NumPy arrays/plain dicts in and out, tested against synthetic data with known
ground truth (no real UVFITS, no CASA, no file I/O beyond what a test itself
sets up in a temp dir).

End-to-end behaviour against **real data** (real UVFITS files, real bandpass
solutions, real CLI invocations) is covered separately by the shell-based
regression gates in `bin/` (e.g. `bin/run_ripple_characterisation_regression_ci.sh`),
following the existing `bin/test_moon_manifest_equivalence.sh` pattern: real
pipeline artifacts, PASS/FAIL markers, `STRICT_DATA=0` to skip cleanly when
local data isn't present. Those gates are not pytest and are run manually /
as a pre-release check, not automatically.
