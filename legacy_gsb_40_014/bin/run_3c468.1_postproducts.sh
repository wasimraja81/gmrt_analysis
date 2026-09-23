#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "[3c468.1-post] INFO: run_3c468.1_postproducts.sh is deprecated; use run_3c468.1_post_selfcal_artifacts.sh"
exec bash "$SCRIPT_DIR/run_3c468.1_post_selfcal_artifacts.sh" "$@"
