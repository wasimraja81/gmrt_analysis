#!/usr/bin/env bash

# Gain solution plotting examples (primary + secondary)
# Uses pipeline_cli standard interface.

# 1) All secondary solution tables (auto -> dynamic spectra when multiple tables)
python pipeline_cli.py solplot \
--tables ~/DATA/gmrt_40_014/work/3c468.1_secondary_phase_only_scan*.npz \
--pol RR \
--quantity amp-phase \
--view auto \
--rows 6 --cols 5 \
--out ~/DATA/gmrt_40_014/work/3c468.1_gainplots_dynamic_auto_rr.png

# (reference) Primary bandpass single-table line plot
# python pipeline_cli.py solplot \
# --tables ~/DATA/gmrt_40_014/work/3c48_bandpass_25jul_gsb_iterfinal_clustering.npz \
# --pol RR \
# --quantity amp-phase \
# --view auto \
# --rows 6 --cols 5 \
# --out ~/DATA/gmrt_40_014/work/3c48_gainplots_line_rr.png

# 2) Same multi-table set for LL (auto -> dynamic spectra)
#    Uses the same input tables as plot 1, but LL instead of RR.
python pipeline_cli.py solplot \
--tables ~/DATA/gmrt_40_014/work/3c468.1_secondary_phase_only_scan*.npz \
--pol LL \
--quantity amp-phase \
--view auto \
--rows 6 --cols 5 \
--out ~/DATA/gmrt_40_014/work/3c468.1_gainplots_dynamic_auto_ll.png

# 3) 1D line plot generated as COMPLEX MEAN over all input tables
python pipeline_cli.py solplot \
--tables ~/DATA/gmrt_40_014/work/3c468.1_secondary_phase_only_scan*.npz \
--pol RR \
--quantity amp-phase \
--view line \
--line-from average \
--rows 6 --cols 5 \
--out ~/DATA/gmrt_40_014/work/3c468.1_gainplots_line_meanAllTables_rr_ampPhase.png

# 4) 1D line plot generated from ONE selected table index (time/table index = 0)
python pipeline_cli.py solplot \
--tables ~/DATA/gmrt_40_014/work/3c468.1_secondary_phase_only_scan*.npz \
--pol LL \
--quantity amp-phase \
--view line \
--line-from index \
--time-index 0 \
--rows 6 --cols 5 \
--out ~/DATA/gmrt_40_014/work/3c468.1_gainplots_line_tableIdx0_ll.png
