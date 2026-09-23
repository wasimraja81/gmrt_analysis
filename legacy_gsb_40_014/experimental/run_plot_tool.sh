#!/usr/bin/env bash
# run_plot_tool.sh — standalone diagnostics plot utility wrapper

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_ACTIVATE="${SCRIPT_DIR}/gmrt/bin/activate"

if [[ -f "${VENV_ACTIVATE}" ]]; then
    # shellcheck source=/dev/null
    source "${VENV_ACTIVATE}"
fi

# Default to non-interactive backend for script usage; caller can override.
if [[ -z "${MPLBACKEND:-}" ]]; then
    export MPLBACKEND=Agg
fi

exec python "${SCRIPT_DIR}/plotVis.py" "$@"
