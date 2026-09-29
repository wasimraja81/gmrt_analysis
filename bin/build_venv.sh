#!/usr/bin/env bash
# Build the gmrt/ venv, all of it: the Python packages pinned in config/requirements.txt,
# then the fonts pinned in config/fonts.txt (src/cli/install_fonts.py), then a check that
# both are there. See docs/dev/ENVIRONMENT_SETUP.md.
# Usage: bin/build_venv.sh           -- create gmrt/ if it is missing, else bring it to the pins
#        bin/build_venv.sh --clear   -- delete gmrt/ and build it afresh
#        PYTHON=python3.12 bin/build_venv.sh   -- the Python to create it with (default: python3)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="$REPO_ROOT/gmrt"
PYTHON="${PYTHON:-python3}"

case "${1:-}" in
  "" | --clear) ;;
  *) echo "usage: bin/build_venv.sh [--clear]" >&2; exit 2 ;;
esac

if [[ "${1:-}" == "--clear" || ! -x "$VENV/bin/python" ]]; then
  echo "== creating $VENV with $($PYTHON --version)"
  "$PYTHON" -m venv --clear "$VENV"
  "$VENV/bin/python" -m pip install --upgrade pip
fi

echo "== Python packages (config/requirements.txt)"
"$VENV/bin/pip" install -r "$REPO_ROOT/config/requirements.txt"

echo "== fonts (config/fonts.txt)"
"$VENV/bin/python" "$REPO_ROOT/src/cli/install_fonts.py"

echo "== check"
"$VENV/bin/python" -c "import numpy, scipy, astropy, matplotlib, skimage, imageio, yaml, PySide6; print('packages: OK')"
"$VENV/bin/python" "$REPO_ROOT/src/cli/install_fonts.py" --check
