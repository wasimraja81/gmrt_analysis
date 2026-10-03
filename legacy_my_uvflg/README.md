# my_uvflg (2011): flag commands from AIPS UVFND listings

The user's flag-command generator from their thesis work, copied 2026-10-03
from `~/softwares/CURR_DEVEL/FLAG_COMMAND_GENERATION/FLGCMD_GEN/` as a
reference for the flagging design (T26; `docs/dev/GWB_PIPELINE_REFACTOR_PLAN.md`).
The compiled binary (`EXEC/my_uvflg`) is left out; `MAKE/mk_my_uvflg` builds it
with f77.

The 2011 workflow: AIPS UVFND lists the visibilities that meet a condition, one
channel and one Stokes at a time; `my_uvflg` reads that listing and writes AIPS
flag commands.

- `SOURCE/my_uvflg.f`: the program.
- `PAR/`: its parameter files (3C147, 3C286, 3C468.1, 3C48): scan times, scan
  length, sampling time, number of antennas, `maxbad_allowed`, time tolerance,
  input and output paths.
- `DATA/`: two UVFND listings of 3C468.1, channel 11, Stokes I and V.
- `WORK/`: the outputs of `example_run` on the V listing (`3C468.1_V_11.FLG`,
  flag commands; `3C468.1_V_11.SUM`, a summary), `generate_flags_all_chans.sh`
  (one run per channel), and `image42.jpg`, a hand-drawn flowchart of the
  per-baseline step: a baseline with more than `maxbad_allowed` bad points is
  flagged for all times; otherwise its bad times are grouped into intervals
  [t(i1), t(i2)] with t(i2) - t(i1) <= (i2 - i1) * dt + tolerance, each flagged.
- `backme.sh`: the package's backup script.
