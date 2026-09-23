# GMRT 40_014 Production Workflow

This document defines the **production-only** workflow for calibrating GMRT 40_014 data in this repository.

It is intended for routine use by operators and collaborators who need reliable, reproducible execution.

---

## 1) Scope

This guide covers:

- the 10-step production calibration chain,
- audit-only verification,
- expected output locations,
- safe recovery and rerun patterns.

This guide does **not** cover experimental scripts, notebook patching utilities, or moon-imaging prototyping.

---

## 2) Production Entry Point

Primary workflow driver:

- `bin/run_gmrt_40_014_calibration_workflow.sh`

Supporting production scripts are invoked internally by this driver.

---

## 3) Prerequisites

Run from repository root:

```bash
cd /Users/raj030/github-wasimraja81/gmrt_analysis
```

Required data location:

```bash
ls -lh "$HOME/DATA/gmrt_40_014/data/40_014_25jul2021_gsb.FITS"
```

Optional sanity check (scripts present):

```bash
ls -1 bin/run_gmrt_40_014_calibration_workflow.sh \
      bin/clean_work_outputs.sh \
      bin/primaryCalibration_example.sh \
      bin/visSplit_3c48_example.sh \
      bin/plotVis_3c48_example.sh \
      bin/visSplit_3c468.1_primary_example.sh \
      bin/plotVis_3c468.1_primary_example.sh \
      bin/clustering_3c468.1_split_example.sh \
      bin/secondaryCalibration_example.sh \
      bin/visSplit_3c468.1_example.sh \
      bin/plotVis_3c468.1_example.sh
```

---

## 4) Copy-Paste Operations

### A. Full production run (clean slate)

```bash
cd /Users/raj030/github-wasimraja81/gmrt_analysis
bash bin/run_gmrt_40_014_calibration_workflow.sh
```

### B. Audit-only verification (no compute rerun)

```bash
cd /Users/raj030/github-wasimraja81/gmrt_analysis
bash bin/run_gmrt_40_014_calibration_workflow.sh --audit-only
```

### C. View latest workflow log

```bash
ls -1t "$HOME/DATA/gmrt_40_014/work/logs"/run_gmrt_40_014_calibration_workflow_*.log | head -1 | xargs -I{} tail -n 80 "{}"
```

### D. View latest product manifest

```bash
ls -1t "$HOME/DATA/gmrt_40_014/work/logs"/run_gmrt_40_014_products_*.txt | head -1 | xargs -I{} cat "{}"
```

---

## 5) Workflow Sequence (Production)

1. Clean work outputs
2. Primary calibration (3C48)
3. Split 3C48 with primary calibration + flags
4. Plot 3C48 diagnostics
5. Split 3C468.1 with primary-only calibration + flags
6. Plot 3C468.1 primary-only diagnostics
7. Derive clustering flags on 3C468.1
8. Secondary calibration with clustering flags
9. Split 3C468.1 with primary + secondary calibration
10. Plot final 3C468.1 diagnostics

The driver enforces strict fail-fast gating: each step must complete and produce expected artifacts before the next step runs.

---

## 6) Expected Outputs

Base output root:

```bash
echo "$HOME/DATA/gmrt_40_014/work"
```

Key artifact groups:

- `primary_calibration/` (bandpass and primary flag tables)
- `split/` (3C48 and 3C468.1 calibrated UVFITS)
- `secondary_calibration/` (clustering products and secondary tables)
- `diagnostics_out/` (PDF/PNG plot products)
- `logs/` (workflow logs and product manifests)

---

## 7) Troubleshooting

### Workflow aborts early

Use the log to identify the failing step and command:

```bash
ls -1t "$HOME/DATA/gmrt_40_014/work/logs"/run_gmrt_40_014_calibration_workflow_*.log | head -1 | xargs -I{} grep -n "STEP\|ABORT\|CHECK FAILED" "{}"
```

### Need quick artifact existence check

```bash
bash bin/run_gmrt_40_014_calibration_workflow.sh --audit-only
```

---

## 8) Repository Organization (Current)

- **Production shell scripts:** `bin/`
- **Production Python apps:** `src/`
- **Reusable Python modules/functions:** `src/modules/`
- **Developer patch/test utilities:** `tools/dev/`
- **Archived/stale docs:** `experimental/docs/`
- **Experimental scripts and notebooks:** `experimental/`

---

## 9) Publishing Curated Outputs to GitHub Pages

Collaborators should generate products from the production workflow on their own machines from the source branch.

GitHub Pages is only for publishing selected static outputs after you inspect them and decide they are worth sharing.

Recommended publish model:

- workflow reproducibility stays on the source branch (`40_014`),
- local run products stay under `$HOME/DATA/gmrt_40_014/work`,
- curated PNG/PDF outputs are copied into a separate `gh-pages` worktree/branch.

Publish the latest successful run:

```bash
cd /Users/raj030/github-wasimraja81/gmrt_analysis
bash bin/publish_gh_pages.sh
```

Moon layout note:

- Moon section rendering is now **strict manifest-only**.
- The publisher auto-generates `logs/moon_layout_manifest_<RUN_TS>.json` when needed.

Publish a specific run timestamp:

```bash
cd /Users/raj030/github-wasimraja81/gmrt_analysis
bash bin/publish_gh_pages.sh --run-ts 20260511_114843
```

Publish and push to the remote `gh-pages` branch:

```bash
cd /Users/raj030/github-wasimraja81/gmrt_analysis
bash bin/publish_gh_pages.sh --run-ts 20260511_114843 --push
```

The publish script:

- creates or reuses a separate `gh-pages` worktree,
- archives the chosen run under `runs/<timestamp>/`,
- refreshes `latest/` to mirror that chosen run,
- regenerates a simple static gallery site,
- commits the update on the `gh-pages` branch.

---

## 11) Moon Manifest Regression Gate (Regular Check)

Run this gate after publish changes and on a regular schedule (e.g. nightly cron on the data host):

```bash
cd /Users/raj030/github-wasimraja81/gmrt_analysis
bash bin/run_moon_manifest_regression_ci.sh
```

What it enforces:

- byte-equivalence between baseline and manifest-rendered Moon HTML,
- negative-control detection,
- sabotage protection (legacy Moon product directories hidden).

Data-host scheduling example (nightly at 02:30):

```bash
30 2 * * * cd /Users/raj030/github-wasimraja81/gmrt_analysis && bash bin/run_moon_manifest_regression_ci.sh >> $HOME/DATA/gmrt_40_014/work/logs/moon_manifest_ci_cron.log 2>&1
```

If running in an environment without data mounts, allow clean skip:

```bash
STRICT_DATA=0 bash bin/run_moon_manifest_regression_ci.sh
```

This means later runs with better labels or styling can be published explicitly, without losing earlier published runs.

---

## 10) Operational Recommendation

For normal operations, run only:

- `bash bin/run_gmrt_40_014_calibration_workflow.sh`
- `bash bin/run_gmrt_40_014_calibration_workflow.sh --audit-only`

Treat everything under `experimental/` and `tools/dev/` as non-production unless explicitly promoted.
