#!/usr/bin/env bash
# Build the gmrt/ venv, all of it: the Python packages pinned in config/requirements.txt,
# then the fonts pinned in config/fonts.txt (src/cli/install_fonts.py), then a check that
# both are there. See docs/dev/ENVIRONMENT_SETUP.md.
# Usage: bin/build_venv.sh           -- create gmrt/ if it is missing, else bring it to the pins
#        bin/build_venv.sh --clear   -- delete gmrt/ and build it afresh
#        PYTHON=python3.12 bin/build_venv.sh   -- the Python to create it with (default: python3)
# Python 3.12 or later: the pins were frozen on Python 3.12.3.

set -euo pipefail

MIN_PYTHON="3.12"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="$REPO_ROOT/gmrt"
PYTHON="${PYTHON:-python3}"

case "${1:-}" in
  "" | --clear) ;;
  *) echo "usage: bin/build_venv.sh [--clear]" >&2; exit 2 ;;
esac

# Refuse a Python older than MIN_PYTHON, before anything is created or deleted.
require_python() {  # $1: the interpreter; $2: what it is, for the message
  local version
  if ! version="$("$1" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null)"; then
    echo "build_venv.sh: $2 ($1) cannot be run; choose one with PYTHON=python$MIN_PYTHON bin/build_venv.sh" >&2
    exit 1
  fi
  if ! "$1" -c "import sys; sys.exit(sys.version_info[:2] < tuple(map(int, '$MIN_PYTHON'.split('.'))))"; then
    echo "build_venv.sh: needs Python $MIN_PYTHON or later; $2 ($1) is Python $version." >&2
    echo "  Choose one with PYTHON=python$MIN_PYTHON bin/build_venv.sh --clear" >&2
    exit 1
  fi
}

if [[ "${1:-}" == "--clear" || ! -x "$VENV/bin/python" ]]; then
  require_python "$PYTHON" "the Python to create the venv with"
  echo "== creating $VENV with $($PYTHON --version)"
  "$PYTHON" -m venv --clear "$VENV"
  "$VENV/bin/python" -m pip install --upgrade pip
fi

require_python "$VENV/bin/python" "the venv's Python"

echo "== Python packages (config/requirements.txt)"
if ! "$VENV/bin/pip" install -r "$REPO_ROOT/config/requirements.txt"; then
  echo "build_venv.sh: the pinned packages did not install on $("$VENV/bin/python" --version)." >&2
  echo "  The pins were frozen on Python 3.12.3; a newer Python may lack a pinned version's" >&2
  echo "  wheel: build with PYTHON=python3.12, or update the pins (docs/dev/ENVIRONMENT_SETUP.md)." >&2
  exit 1
fi

echo "== fonts (config/fonts.txt)"
"$VENV/bin/python" "$REPO_ROOT/src/cli/install_fonts.py"

echo "== check"
"$VENV/bin/python" -c "import numpy, scipy, astropy, matplotlib, skimage, imageio, yaml, PySide6; print('packages: OK')"
"$VENV/bin/python" "$REPO_ROOT/src/cli/install_fonts.py" --check
