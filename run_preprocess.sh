#!/usr/bin/env bash
# run_preprocess.sh  –  thin shell wrapper for preprocess_ugmrt.py
#
# Activates the project venv, then passes all arguments straight through to
# the Python driver.  Edit VENV (and optionally CONFIG) below if your paths
# differ.
#
# Usage examples:
#   ./run_preprocess.sh                                              # steps 1-4, writes to disk
#   ./run_preprocess.sh --step 1                                    # index only
#   ./run_preprocess.sh --step 2 --iter-tag iter01                  # bandpass only
#   ./run_preprocess.sh --step 3 --iter-tag iter01                  # diagnostics only
#   ./run_preprocess.sh --step 4 --iter-tag iter01                  # propose flags only
#   ./run_preprocess.sh --step all --iter-tag iter01                # single interactive pass
#   ./run_preprocess.sh --step all --n-iters 5                      # 5-iter interactive loop
#   ./run_preprocess.sh --step all --n-iters 5 --dry-run            # 5-iter loop, preview only
#   ./run_preprocess.sh --step all --auto --n-iters 10              # 10-iter batch, writes to disk
#   ./run_preprocess.sh --step all --auto --n-iters 10 --dry-run    # 10-iter batch, preview only
#   ./run_preprocess.sh --config /other/path.cfg --step all         # custom config

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${SCRIPT_DIR}/gmrt"                   # ← adjust if venv is elsewhere
PYTHON="${VENV}/bin/python"

# Default config file (override with --config on the command line).
CONFIG="${SCRIPT_DIR}/preprocess_ugmrt.cfg"

if [[ ! -x "${PYTHON}" ]]; then
    echo "ERROR: Python not found at ${PYTHON}" >&2
    echo "       Set VENV in $0 to point to your virtual environment." >&2
    exit 1
fi

# matplotlib backend selection:
#   --auto  → force Agg (non-interactive) so the loop never blocks on a GUI
#             window; plots are saved to disk with iteration tags.
#   without --auto → leave MPLBACKEND alone so the system default is used
#             (MacOSX on macOS) and plt.show() opens an interactive window.
# Override at any time by exporting MPLBACKEND yourself before running.
if [[ -z "${MPLBACKEND:-}" ]]; then
    if [[ " $* " =~ " --auto " ]]; then
        export MPLBACKEND=Agg
    fi
fi

# If the caller hasn't already passed --config, inject the default.
if [[ ! " $* " =~ " --config " ]]; then
    set -- --config "${CONFIG}" "$@"
fi

exec "${PYTHON}" "${SCRIPT_DIR}/preprocess_ugmrt.py" "$@"
