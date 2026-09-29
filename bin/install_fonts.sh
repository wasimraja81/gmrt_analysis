#!/usr/bin/env bash
# Thin wrapper: install the fonts pinned in config/fonts.txt into the gmrt/ venv
# (src/cli/install_fonts.py). bin/build_venv.sh runs it as one of its steps; on its own,
# e.g. after a font is added to config/fonts.txt.
# Usage: bin/install_fonts.sh [--check] [--help]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

exec "$REPO_ROOT/gmrt/bin/python3" "$REPO_ROOT/src/cli/install_fonts.py" "$@"
