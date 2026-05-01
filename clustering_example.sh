#!/usr/bin/env bash

# Secondary clustering example
# - Applies primary clustering/session flags first (doflag=on)
# - Detects additional flags on secondary
# - Dry-run: no flag-table writes
# - Real-run: merges/writes to secondary session table

python pipeline_cli.py clustering \
--config preprocess_ugmrt.cfg \
--no-dry-run \
--docal on \
--doflag on \
--set "SOURCE='3C468.1'" \
--set "BANDPASS_OUT=WORK_DIR / '3c48_bandpass_25jul_gsb.npz'" \
--set "FLAG_TABLE_PATHS=[WORK_DIR / '3c48_flag_table_session.json']" \
--set "FLAG_TABLE_SESSION=WORK_DIR / '3c468.1_flag_table_session.json'" \
--set "MAX_ROWS_SOLVE=None" \
--set "SOLVE_ELEVATION_MIN_DEG=25.0" \
--set "CLUSTERING_CORR=['V','RR','LL']" \
--set "CLUSTERING_THRESHOLD_JY={'V':5.0,'RR':50.0,'LL':50.0}" \
--set "CLUSTERING_THRESHOLD_LOW_JY={'RR':10.0,'LL':10.0}" \
--save-plots

## Real write run (flags persisted to secondary session table):
# python pipeline_cli.py clustering \
# --config preprocess_ugmrt.cfg \
# --no-dry-run \
# --docal on \
# --doflag on \
# --set "SOURCE='3C468.1'" \
# --set "BANDPASS_OUT=WORK_DIR / '3c48_bandpass_25jul_gsb.npz'" \
# --set "FLAG_TABLE_PATHS=[WORK_DIR / '3c48_flag_table_session.json']" \
# --set "FLAG_TABLE_SESSION=WORK_DIR / '3c468.1_flag_table_session.json'" \
# --set "MAX_ROWS_SOLVE=None" \
# --set "SOLVE_ELEVATION_MIN_DEG=25.0" \
# --set "CLUSTERING_CORR=['V','RR','LL']" \
# --set "CLUSTERING_THRESHOLD_JY={'V':5.0,'RR':50.0,'LL':50.0}" \
# --set "CLUSTERING_THRESHOLD_LOW_JY={'RR':2.0,'LL':2.0}" \
# --save-plots