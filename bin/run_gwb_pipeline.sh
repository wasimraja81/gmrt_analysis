#!/usr/bin/env bash
# Thin wrapper: path setup, then dispatch to src/cli/run_gwb_pipeline.py.
# Usage: bin/run_gwb_pipeline.sh config/<observation>.yaml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

exec "$REPO_ROOT/gmrt/bin/python3" "$REPO_ROOT/src/cli/run_gwb_pipeline.py" "$@"
