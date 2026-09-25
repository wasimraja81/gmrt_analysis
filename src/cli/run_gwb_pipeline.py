"""Entry point for the GWB pipeline driver -- invoked by `bin/run_gwb_pipeline.sh`.

Usage: gmrt/bin/python3 src/cli/run_gwb_pipeline.py <config.yaml>
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cli.pipeline_stages import load_pipeline_config, run_pipeline  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: run_gwb_pipeline.py <config.yaml>", file=sys.stderr)
        return 1
    config = load_pipeline_config(argv[1])
    run_pipeline(config)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
