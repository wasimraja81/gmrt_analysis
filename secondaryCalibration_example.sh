#!/usr/bin/env bash

# Secondary calibration example (pipeline_cli interface)
# - Uses config defaults from preprocess_ugmrt.cfg
# - Updates key params through CLI + --set (same style as clustering_example.sh)
# - Uses latest derived secondary flag table via --flagver latest
# - Uses input primary bandpass table via --bpcal
# - No row cap: --max-rows 0

SOURCE="3C468.1"
SRC_TAG="$(printf '%s' "$SOURCE" | tr '[:upper:]' '[:lower:]')"

# Primary table is an INPUT transfer table (can remain 3C48-derived if desired)
PRIMARY_BPCAL=~/DATA/gmrt_40_014/work/3c48_bandpass_25jul_gsb_iterfinal_clustering.npz

python pipeline_cli.py secondary \
--config preprocess_ugmrt.cfg \
--mode phase_only \
--flagver latest \
--solint-mode scan \
--max-rows 0 \
--set "SOURCE='$SOURCE'" \
--set "FLAG_TABLE_SESSION=WORK_DIR / '${SRC_TAG}_flag_table_session.json'" \
--bpcal "$PRIMARY_BPCAL" \
--out ~/DATA/gmrt_40_014/work/${SRC_TAG}_secondary_phase_only.npz

# Alternate examples:
# --mode delay_phase --solint scan --out ~/DATA/gmrt_40_014/work/3c468.1_secondary_delay_phase.npz
# --solint all (single solution over full selected timerange)
# --solint-mode minutes --solint-minutes 10  (fixed-width bins; can span scan boundaries)
# --scan-gap-minutes 2  (optional override for --solint-mode scan; default auto-derived)
