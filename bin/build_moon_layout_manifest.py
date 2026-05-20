#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def select_first_existing_rel(run_dir: Path, base_dir: str, candidates: list[str]) -> str:
    for candidate in candidates:
        if (run_dir / base_dir / candidate).is_file():
            return f"{base_dir}/{candidate}"
    return ""


def detect_moon_root(run_dir: Path) -> str:
    target = "casa_selfcal/ghpages_products/moon0520"
    legacy = "casa_selfcal/ghpages_products"
    if (run_dir / target / "no_phasecenter").is_dir() or (run_dir / target / "phasecenter").is_dir():
        return target
    if (run_dir / legacy / "no_phasecenter").is_dir() or (run_dir / legacy / "phasecenter").is_dir():
        return legacy
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Moon gh-pages layout manifest for publish_gh_pages.sh")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--run-ts", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    work_dir = Path(args.work_dir).expanduser().resolve()
    run_dir = work_dir

    moon_products_root = detect_moon_root(run_dir)
    moon_no = f"{moon_products_root}/no_phasecenter" if moon_products_root else ""
    moon_pc = f"{moon_products_root}/phasecenter" if moon_products_root else ""

    has_no = bool(moon_no) and (run_dir / moon_no).is_dir()
    has_pc = bool(moon_pc) and (run_dir / moon_pc).is_dir()

    entries = {
        "no_raw_movie": {
            "entry_id": "moon.s1.movie",
            "resolved_rel": select_first_existing_rel(run_dir, moon_no, ["moon0520_no_phasecenter_raw_selfcal.mp4", "moon0520_no_phasecenter_pre.mp4"]) if has_no else "",
        },
        "no_raw_stack": {
            "entry_id": "moon.s1.stack",
            "resolved_rel": select_first_existing_rel(run_dir, moon_no, ["moon0520_no_phasecenter_phasecorr_raw_selfcal_stack.png", "moon0520_no_phasecenter_phasecorr_destriped_stack_original.png"]) if has_no else "",
        },
        "no_raw_rms_movie": {
            "entry_id": "moon.s1.rms_movie",
            "resolved_rel": select_first_existing_rel(run_dir, moon_no, ["moon0520_observed_center_raw_cumulative_coadd.mp4", "moon0520_no_phasecenter_phasecorr_raw_selfcal_cumulative_coadd.mp4"]) if has_no else "",
        },
        "pc_raw_movie": {
            "entry_id": "moon.s2.movie",
            "resolved_rel": select_first_existing_rel(run_dir, moon_pc, ["moon0520_phasecenter_raw_selfcal.mp4", "moon0520_phasecenter_pre.mp4"]) if has_pc else "",
        },
        "pc_raw_stack": {
            "entry_id": "moon.s2.stack",
            "resolved_rel": select_first_existing_rel(run_dir, moon_pc, ["moon0520_phasecenter_noshift_raw_selfcal_stack.png", "moon0520_phasecenter_noshift_destriped_stack_original.png"]) if has_pc else "",
        },
        "pc_raw_rms_movie": {
            "entry_id": "moon.s2.rms_movie",
            "resolved_rel": select_first_existing_rel(run_dir, moon_pc, ["moon0520_moon_centered_raw_cumulative_coadd.mp4", "moon0520_phasecenter_noshift_raw_selfcal_cumulative_coadd.mp4"]) if has_pc else "",
        },
        "no_dst_movie": {
            "entry_id": "moon.s3.movie",
            "resolved_rel": select_first_existing_rel(run_dir, moon_no, ["moon0520_no_phasecenter_destriped.mp4"]) if has_no else "",
        },
        "no_dst_stack": {
            "entry_id": "moon.s3.stack",
            "resolved_rel": select_first_existing_rel(run_dir, moon_no, ["moon0520_no_phasecenter_phasecorr_destriped_stack.png", "moon0520_no_phasecenter_phasecorr_destriped_stack_destriped.png"]) if has_no else "",
        },
        "no_dst_rms_movie": {
            "entry_id": "moon.s3.rms_movie",
            "resolved_rel": select_first_existing_rel(run_dir, moon_no, ["moon0520_observed_center_destriped_cumulative_coadd.mp4", "moon0520_no_phasecenter_phasecorr_destriped_cumulative_coadd.mp4"]) if has_no else "",
        },
        "pc_dst_movie": {
            "entry_id": "moon.s4.movie",
            "resolved_rel": select_first_existing_rel(run_dir, moon_pc, ["moon0520_phasecenter_destriped.mp4"]) if has_pc else "",
        },
        "pc_dst_stack": {
            "entry_id": "moon.s4.stack",
            "resolved_rel": select_first_existing_rel(run_dir, moon_pc, ["moon0520_phasecenter_noshift_destriped_stack.png", "moon0520_phasecenter_noshift_destriped_stack_destriped.png"]) if has_pc else "",
        },
        "pc_dst_rms_movie": {
            "entry_id": "moon.s4.rms_movie",
            "resolved_rel": select_first_existing_rel(run_dir, moon_pc, ["moon0520_moon_centered_destriped_cumulative_coadd.mp4", "moon0520_phasecenter_noshift_destriped_cumulative_coadd.mp4"]) if has_pc else "",
        },
    }

    doc = {
        "schema": "gmrt-ghpages-layout-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_ts": args.run_ts,
        "moon": {
            "products_root": moon_products_root,
            "no_phasecenter_dir": moon_no,
            "phasecenter_dir": moon_pc,
            "has_no_phasecenter": has_no,
            "has_phasecenter": has_pc,
            "entries": entries,
        },
    }

    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
