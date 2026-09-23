#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path


def detect_workflow_run_id(work_dir: Path, run_ts: str) -> str:
    products_manifest = work_dir / "logs" / f"run_gmrt_40_014_products_{run_ts}.txt"
    if not products_manifest.is_file():
        return ""

    workflow_log: Path | None = None
    lines = products_manifest.read_text(encoding="utf-8").splitlines()
    for idx, line in enumerate(lines):
        if line.strip() == "Workflow log:" and idx + 1 < len(lines):
            candidate = Path(lines[idx + 1].strip()).expanduser()
            if candidate.is_file():
                workflow_log = candidate
            break

    if workflow_log is None or not workflow_log.is_file():
        return ""

    run_id_re = re.compile(r"\bRUN_ID=(\S+)")
    for line in workflow_log.read_text(encoding="utf-8", errors="replace").splitlines():
        match = run_id_re.search(line)
        if match:
            return match.group(1)
    return ""


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


def detect_static_target_dirs(run_dir: Path) -> list[Path]:
    roots = sorted(run_dir.glob("casa_selfcal/*_scan*_stk*"))
    return [p for p in roots if p.is_dir()]


def _first_existing_rel(run_dir: Path, base_dir: str, candidates: list[str]) -> str:
    return select_first_existing_rel(run_dir, base_dir, candidates)


def _first_matching_rel(run_dir: Path, base_dir: str, pattern: str) -> str:
    base_path = run_dir / base_dir
    if not base_path.is_dir():
        return ""
    for match in sorted(base_path.glob(pattern)):
        if match.is_file():
            return str(match.relative_to(run_dir)).replace('\\', '/')
    return ""


def build_static_target_sections(run_dir: Path) -> list[dict]:
    sections: list[dict] = []
    for section_dir in detect_static_target_dirs(run_dir):
        name = section_dir.name
        scan_tag = ""
        stack_size = ""
        if m := re.search(r"(scan\d+)", name):
            scan_tag = m.group(1)
        if m := re.search(r"_stk(\d+)", name):
            stack_size = m.group(1)

        source_label = name
        if m := re.match(r"(.+?)_scan\d+_stk\d+", name):
            source_label = m.group(1)
        context_bits = [f"Source {source_label}"]
        if scan_tag:
            context_bits.append(scan_tag)
        if stack_size:
            context_bits.append(f"imaged per integration (stack size = {stack_size})")
        context_bits.append("static target tracked at phase-tracking centre (unlike moving-target Moon)")

        movie_rel = _first_existing_rel(run_dir, str(section_dir.relative_to(run_dir)), [
            "3c468.1_selfcal_movie.mp4",
            "3c468.1_selfcal_movie.mov",
            "3c468.1_selfcal_movie.gif",
        ])
        stack_rel = _first_existing_rel(run_dir, str(section_dir.relative_to(run_dir)), [
            "3c468.1_stack_mean.png",
            "3c468.1_stack.png",
            "3c468.1_stack_mean_destriped.png",
        ])
        rms_movie_rel = _first_existing_rel(run_dir, str(section_dir.relative_to(run_dir)), [
            f"{name}_cumulative_coadd.mp4",
            f"{name}_cumulative_coadd.mov",
            f"{name}_cumulative_coadd.gif",
            "3c468.1_cumulative_coadd.mp4",
            "3c468.1_cumulative_coadd.mov",
            "3c468.1_cumulative_coadd.gif",
            "3c468.1_cumulative_rms_evolution.mp4",
            "3c468.1_cumulative_rms_evolution.mov",
            "3c468.1_cumulative_rms_evolution.gif",
        ])
        diag_panel_rel = _first_existing_rel(run_dir, str(section_dir.relative_to(run_dir)), [
            f"{scan_tag}_clean_cycle_metrics_panel_5x5.png" if scan_tag else "",
            "scan05_clean_cycle_metrics_panel_5x5.png",
            "scan05_clean_cycle_metrics_panel.png",
        ])
        diag_gain_panel_rel = _first_existing_rel(run_dir, str(section_dir.relative_to(run_dir)), [
            f"{scan_tag}_selfcal_gain_panel_6x5.png" if scan_tag else "",
            "scan05_selfcal_gain_panel_6x5.png",
            "scan05_selfcal_gains_panel_6x5.png",
        ])
        diag_csv_rel = _first_existing_rel(run_dir, str(section_dir.relative_to(run_dir)), [
            f"{scan_tag}_clean_cycle_metrics_per_integration.csv" if scan_tag else "",
            "scan05_clean_cycle_metrics_per_integration.csv",
        ])
        diag_stats_rel = _first_existing_rel(run_dir, str(section_dir.relative_to(run_dir)), [
            f"{scan_tag}_clean_cycle_metrics_summary_stats.csv" if scan_tag else "",
            "scan05_clean_cycle_metrics_summary_stats.csv",
        ])
        diag_flag_total_rel = _first_existing_rel(run_dir, str(section_dir.relative_to(run_dir)), [
            f"{scan_tag}_flag_total_vs_integration.png" if scan_tag else "",
            "scan05_flag_total_vs_integration.png",
        ])
        diag_flag_ant_panel_rel = _first_existing_rel(run_dir, str(section_dir.relative_to(run_dir)), [
            f"{scan_tag}_flag_antenna_panel_6x5.png" if scan_tag else "",
            "scan05_flag_antenna_panel_6x5.png",
        ])
        diag_flag_heatmap_rel = _first_existing_rel(run_dir, str(section_dir.relative_to(run_dir)), [
            f"{scan_tag}_flag_baseline_grid_all_integrations.png" if scan_tag else "",
            "scan05_flag_baseline_grid_all_integrations.png",
        ])
        diag_flag_heatmap_movie_rel = _first_existing_rel(run_dir, str(section_dir.relative_to(run_dir)), [
            f"{scan_tag}_flag_baseline_grid_movie.mp4" if scan_tag else "",
            "scan05_flag_baseline_grid_movie.mp4",
        ])
        diag_flag_heatmap_full_rel = _first_existing_rel(run_dir, str(section_dir.relative_to(run_dir)), [
            f"{scan_tag}_flag_baseline_grid_full_ms.png" if scan_tag else "",
            "scan05_flag_baseline_grid_full_ms.png",
        ])
        if not diag_flag_heatmap_rel:
            for flag_diag_dir in [f"{scan_tag}_flag_diagnostics" if scan_tag else "", "scan05_flag_diagnostics"]:
                if not flag_diag_dir:
                    continue
                diag_flag_heatmap_rel = _first_matching_rel(
                    run_dir,
                    f"{section_dir.relative_to(run_dir)}/{flag_diag_dir}",
                    "*_flag_baseline_grid.png",
                )
                if diag_flag_heatmap_rel:
                    break

        diagnostics_selfcal = {
            "metrics_panel_rel": diag_panel_rel,
            "gain_panel_rel": diag_gain_panel_rel,
            "metrics_csv_rel": diag_csv_rel,
            "summary_csv_rel": diag_stats_rel,
        }
        diagnostics_ms_flag = {
            "representative_heatmap_rel": diag_flag_heatmap_rel,
            "heatmap_movie_rel": diag_flag_heatmap_movie_rel,
            "full_observation_heatmap_rel": diag_flag_heatmap_full_rel,
            "total_vs_integration_rel": diag_flag_total_rel,
            "antenna_panel_rel": diag_flag_ant_panel_rel,
        }

        sections.append({
            "section_id": f"static.{name}",
            "section_name": name,
            "source_label": source_label,
            "scan_tag": scan_tag,
            "stack_size": stack_size,
            "section_title": f"Post-selfcal imaging summary: {source_label}",
            "section_context": " · ".join(context_bits),
            "movie_rel": movie_rel,
            "stack_rel": stack_rel,
            "rms_movie_rel": rms_movie_rel,
            "diagnostics": {
                "selfcal": diagnostics_selfcal,
                "ms_flag": diagnostics_ms_flag,
            },
            "diag_panel_rel": diag_panel_rel,
            "diag_gain_panel_rel": diag_gain_panel_rel,
            "diag_csv_rel": diag_csv_rel,
            "diag_stats_rel": diag_stats_rel,
            "diag_flag_total_rel": diag_flag_total_rel,
            "diag_flag_ant_panel_rel": diag_flag_ant_panel_rel,
            "diag_flag_heatmap_rel": diag_flag_heatmap_rel,
            "diag_flag_heatmap_movie_rel": diag_flag_heatmap_movie_rel,
            "diag_flag_heatmap_full_rel": diag_flag_heatmap_full_rel,
        })
    return sections


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
        "schema": "gmrt-ghpages-layout-v2",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_ts": args.run_ts,
        "workflow_run_id": detect_workflow_run_id(work_dir, args.run_ts),
        "moon": {
            "products_root": moon_products_root,
            "no_phasecenter_dir": moon_no,
            "phasecenter_dir": moon_pc,
            "has_no_phasecenter": has_no,
            "has_phasecenter": has_pc,
            "entries": entries,
        },
        "static_targets": build_static_target_sections(run_dir),
    }

    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
