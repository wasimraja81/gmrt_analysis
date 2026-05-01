python plotVis.py \
--fits ~/DATA/gmrt_40_014/data/40_014_25jul2021_gsb.FITS \
--bpcal ~/DATA/gmrt_40_014/work/3c48_bandpass_25jul_gsb_iterfinal_clustering.npz \
--flag ~/DATA/gmrt_40_014/work/3c468.1_flag_table_session.json \
--outdir ./diagnostics_out \
--index-cache ~/DATA/gmrt_40_014/work/40_014_25jul2021_gsb.index.npz \
--source 3c468.1 \
--elevation-min 25 \
--products RR,LL \
--panels amp_uvdist,phase_uvdist,real_uvdist,imag_uvdist,amp_time,phase_time,real_time,imag_time, amp_freq,phase_freq,real_freq,imag_freq,ri_scatter,vector_avg \
--sample-frac 0.02 \
--overlay-flags \
--multipage both \
--time-range "2021-07-25 21:00:00" "2021-07-26 00:30:00"
#--outfile plotvis_multipanel.png \
