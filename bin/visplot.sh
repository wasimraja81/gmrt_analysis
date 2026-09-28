#!/usr/bin/env bash
# Thin wrapper: path setup, then dispatch to src/cli/run_visplot.py.
# Usage: bin/visplot.sh [FITS_PATH]                            -- the GUI
#        bin/visplot.sh FITS_PATH --plots PLOT[,PLOT...] [options]  -- see --help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

exec "$REPO_ROOT/gmrt/bin/python3" "$REPO_ROOT/src/cli/run_visplot.py" "$@"
