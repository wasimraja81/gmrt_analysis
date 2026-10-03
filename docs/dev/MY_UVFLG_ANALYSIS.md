# my_uvflg.f (2011): what it does, and what carries into the flag generator

An analysis of the user's flag-command generator (`legacy_my_uvflg/SOURCE/my_uvflg.f`,
1,079 lines, 20-25 July 2011), for the flag-generation task of T26's flagging design
(`GWB_PIPELINE_REFACTOR_PLAN.md`). Line numbers refer to that file. Read 2026-10-03.

## The workflow it belongs to

AIPS UVFND lists the visibilities of one source, one channel and one Stokes that meet a
condition; `my_uvflg` reads that listing and writes AIPS UVFLG commands (an INTEXT file,
`.FLG`) and a summary (`.SUM`). The listing is made on the calibrator; the flags it
writes cover the calibrator's scan and the target scan after it ("Distribute the bad
vis. into the time bins (cal-scan + target-scan)", line 265), for every source
(`SOURCES ''`). A shell loop runs it once per channel (`WORK/generate_flags_all_chans.sh`,
channels 11-246).

## Inputs

- Arguments: source name, Stokes tag (I or V: which UVFND listing), channel, parameter
  file. The listing read is `<source>_<stokes>_<channel>.UVFND`.
- Parameter file (lines 115-137): `nscans`, the number of time boundaries (scans + 1);
  `scan_len`, minutes, "roughly close to scan-length on the source used for flagging";
  the boundaries as day, hour, minute, second; `tsamp_sec`, the sampling time, "also the
  time within which points around a bad point is flagged"; `nante`; `maxbad_allowed`;
  `tol_sec`; input and output directories.
- Each listing line (format 1140): visibility number, time (d/hh:mm:ss, whole seconds),
  antenna pair, u, v, w, amplitude, phase, weight.

Derived on entry (lines 139-140): `nbase = nante(nante+1)/2` (autocorrelations
included) and `nsamp_per_scan = int(scan_len*60/tsamp_sec) * nbase`, "total vis. (per
chan) per scan".

## The algorithm, for one channel

1. Read and bin (lines 239-278). Each listed visibility is counted for its baseline over
   all times (`ant_mat`) and in the time bin [t0(i), t0(i+1)) holding it
   (`ant_mat3`, `n_badsamp_scan`); its time is kept per baseline and bin
   (`time_scan_sec`). A visibility outside every bin is counted in the total but in no
   bin.
2. A whole bin (lines 341-359): bad when the bin's count of listed visibilities is at
   least `0.7 * nbase * nsamp_per_scan`. It then flags every antenna and baseline for
   the whole bin (`ANTE 0 BASE 0`), and the bin is done.
3. An antenna in a bin (lines 362-414), bad by either of two tests:
   - method 1: the count over all its baselines (its autocorrelation included) is at
     least `0.6 * nante * scan_len*60/tsamp_sec`, "flag ante if bad with more than 60%
     baselines";
   - method 2: more than `0.5 * nante` of its baselines each have more than
     `maxbad_allowed` listed visibilities in the bin.
   A bad antenna is flagged with all its baselines for the whole bin (`ANTE=i BASE 0`).
4. A baseline in a bin (lines 697-881; the flowchart `WORK/image42.jpg`), skipped when
   its second antenna is bad:
   - none listed: nothing;
   - at least `maxbad_allowed` listed: flagged for the whole bin; the antenna's
     baselines of this kind are merged into one command (`ANTE=i BASE=j,k,...`);
   - fewer: its listed times are grouped greedily into runs. From the first time t(i1),
     the run is the longest t(i1)..t(i2) with `t(i2) - t(i1) <= (i2 - i1)*tsamp_sec +
     tol_sec`; each run is flagged over [t(i1) - tsamp_sec, t(i2) + tsamp_sec], and the
     next run starts at i2 + 1.
5. Output: one UVFLG line per decision, all Stokes (`STOKES ''`), this channel only
   (`BCHAN = ECHAN`), `OPCODE 'FLAG'`; and per bin and over all times, antenna-by-
   antenna count matrices in the summary. AIPS's 200-character line limit led to
   dropping the REASON and SOURCES text (the 25 July 2011 caveat).

What reaches the target: the whole-bin decisions of steps 2, 3 and the first case of
step 4, since a bin spans the calibrator scan and the target scan after it and the
commands name no source. The runs of step 4's last case cover only the calibrator's own
bad times, padded by one sample.

## Findings

- The whole-bin threshold (step 2) counts the baselines twice: `nsamp_per_scan` already
  includes `nbase`, and the limit multiplies by `nbase` again. With the parameter file's
  values (30 antennas, 5 min, 2 s): 150 samples x 465 = 69,750 visibilities per bin and
  channel, a limit of 0.7 x 465 x 69,750 = 22.7 million; no bin can reach it. The intent
  reads as 0.7 of a bin's visibilities.
- `nbase` counts autocorrelations (`nante(nante+1)/2`), and method 1 counts an
  antenna's autocorrelation among its baselines; both archival files have none (378
  baselines of 28 antennas).
- Times and the sampling time are whole seconds; the GWB file's integrations are
  2.684 s.
- Fixed limits: 32 antennas, 20 bins, 8,196 listed visibilities per baseline.
- Channels are independent: one run, and one command per decision, per channel; a
  baseline bad in many channels is never seen as one pattern. The cumulative matrix
  over all times is printed and not used in a decision.
- The bins are typed in, one calibrator scan and the target scan after it each; the
  calibrator's scan length is one value for all bins.
- Detection uses one Stokes (I or V) per run; the flags cover all Stokes
  (`STOKES ''`, line 193): a detection flags every product of its row and channel.
  Not in `my_uvflg`: neighbouring channels (each command is one channel, lines
  187-195), time-channel patches, array-wide bursts (the later Python port's), and
  decisions across bins (the all-times matrix is only printed, lines 334-335).
- The 2011 outputs in `WORK/` came from an earlier version: they write
  `SOURCES '3C468.1'` and `REASON 'MY_UVFLG'` where the source now writes empty
  strings, and its merged lines read `BASE= BASE= 2, 3, ...`, which the current source
  does not produce. Its summary counts 6,657 listed visibilities where the V listing in
  `DATA/` holds 7,192 lines; the user (2026-10-03): "The 2011 data was a different set
  of data, so numbers would vary".
- `maxbad_allowed` is compared with `ant_mat3(iante, ibase, iscan)`: the listed
  visibilities of one baseline in one bin, for the run's channel and Stokes, one per
  integration, so a count of the baseline's bad integrations in the calibrator scan.
  A baseline is flagged for the bin at `>= maxbad_allowed` (line 706); antenna method 2
  counts baselines with `> maxbad_allowed` (lines 399, 405).

## Its settings, and where each comes from in the flag generator (proposed)

| `my_uvflg` | 2011 value | proposed source |
|---|---|---|
| bins (`nscans`, boundaries) | typed in | cycles, from the data (defined below), for the calibrator-target pairs the user gives (e.g. 3C468.1, the phase calibrator, with Cas-A in 40_014) |
| `scan_len` | 5 min | from the data: each calibrator scan's own length |
| `tsamp_sec` | 2 s | from the data: the measured integration time |
| `nante`, `nbase` | 30, 465 | from the data: antennas and baselines with rows (autocorrelations only where present) |
| channel | one per run | from the found files: all channels in one run |
| Stokes of detection | one per run | from the found files' conditions, recorded |
| whole-bin fraction | 0.7 | parameter, default 0.7 (as a fraction of the bin's visibilities) |
| antenna, method 1 fraction | 0.6 | parameter, default 0.6 |
| antenna, method 2 fraction | 0.5 | parameter, default 0.5 |
| `maxbad_allowed` | 5 | parameter, a fraction (the user): a baseline's bad integrations in the calibrator scan over its integrations there; the 2011 value 5 of 150 (5 min at 2 s) = 0.033 |
| `tol_sec` | 60 s | parameter, in seconds or integrations |
| padding of a run | one sample | parameter, default one integration |
| directories | parameter file | the found files and the flag file named on the command line |
| 200-character lines | AIPS limit | the AIPS UVFLG export's concern only |

In the flag table's layers, whole-bin decisions are rows (L2) for their channels: a
bin's antenna or baseline flagged in every channel of the found files is L2; in some
channels, L3 bits for those channels; channels flagged in every row, L1.

## Cycles: the bins, from the data

`my_uvflg`'s bin is the span between two consecutive boundary times of its parameter
file, each holding one calibrator scan and the target scan after it (line 265). The
flag generator takes them from the derived scans (T20), named cycles (the user asked
for "bin" to be qualified, 2026-10-03):

- a calibrator run: consecutive scans of the calibrator, no other source between them
  (a gap or the longest-scan cut can split one stretch into two scans);
- a cycle: a calibrator run and the paired target's scans after it, up to the
  calibrator's next run; target scans before the first calibrator run are in no cycle
  (`my_uvflg` ignores points outside its bins). Scans of other sources within that span:
  `my_uvflg` flags them too (`SOURCES ''`, line 192: every source in the bin's span),
  and so does the flag generator (the user, 2026-10-03: "flag all sources within the
  time span of the cycle"; in 40_014, cycle 2 holds 3C303, 3C345, B1929+10 and 3C48
  scans);
- the tiers' counts use the calibrator run's integrations (`visibilities_in_cycle` =
  baselines with rows x the run's integrations, per channel and detection product); the
  cycle's target scans inherit its decisions.

The pair (3C468.1, Cas-A) in the GSB file's listing (2026-10-03):

| cycle | calibrator run (3C468.1) | target scans (Cas-A) | other scans within |
|---|---|---|---|
| 1 | 4, 5 (17:11:33, 17:14:47) | 6, 7 | none |
| 2 | 8, 9 | none | 10-13 (3C303, 3C345, B1929+10, 3C48) |
| 3 | 14 | 15 | none |
| 4-8 | 16, 18, 20, 22, 24 | 17, 19, 21, 23, 25 | none |
| 9 | 26 | none (Moon scans follow) | none |

## Across channels: a vote

The tiers run per channel as ported; then, for each cycle, a whole-cycle, antenna or
baseline decision made in at least a fraction `f_chan` of the searched channels is
applied to all of them (the user agreed, 2026-10-03, default 0.75). The runs of bad
times stay per channel. An extended decision covers the searched channels only (the
user: "Searched channels only").

## Beyond `my_uvflg`: further extension rules

Every flag a rule sets covers all products of its row and channel (as `STOKES ''`).
Adopted in addition (the user, 2026-10-03: "All of B1-B5 should be adopted for the
flagging"), each with parameters and defaults to be defined, and with the same guards:

- B1 neighbouring channels: a detected channel run padded by m channels on each side,
  as runs are padded in time;
- B2 time-channel patches: detections crowding a time x channel patch of one baseline or
  antenna flag the whole patch (the two-dimensional form of the run rule);
- B3 array-wide bursts: an integration with detections on at least `f_burst` of the
  baselines, in at least `f_chan` of the searched channels, flagged for every baseline
  (the Python port's burst tier, its default 0.5);
- B5 across cycles: an antenna or baseline decided in at least `f_cycle` of the cycles
  flagged for the whole observation.

## Porting without such bugs

The user (2026-10-03): "would be good to not have such bugs if we adopt that code in our
flagging scheme" (after the whole-bin finding above). The port:

1. states each limit as a fraction of a named count taken from the data (e.g.
   `bad_in_bin >= whole_bin_fraction * visibilities_in_bin`, with
   `visibilities_in_bin` = baselines with rows x the calibrator scan's integrations,
   defined once), so no count is multiplied in twice;
2. tests every tier with synthetic found lists just below and just above each of its
   limits (the "just above" case fails for a limit no data can reach);
3. reports, for each run, what every tier decided (whole bins, antennas, baselines,
   runs), kept in the run's record, so a tier that never fires is seen.

No reference run against `my_uvflg` itself (the user: "lets avoid the reference run. We
can have tests to verify our implementations. Going to install fortran code will be a
pain - f77 vs missing subroutines etc.").

## The user's answers (2026-10-03)

- `maxbad_allowed` "should be fraction - but find out if this is visibility, antena or
  baseline?": per baseline, as found above; a fraction of the baseline's integrations
  in the calibrator scan.
- "The target-calibrator pair is to be specified by the user. usually the phase
  calibrator is the one that interleaves the target CAS-A observations." In 40_014,
  3C468.1 and Cas-A alternate (the listing's scans 14-26). Which target scans inherit a
  calibrator scan's decisions follows `my_uvflg`: those after it up to the next
  calibrator scan.
- "I think adopting the fortran code as is could be easier. But it would be great to
  do this across channels as well." The tiers are adopted as they are, the whole-bin
  test as it reads (0.7 of the bin's visibilities; to confirm), and the decisions are
  made for all channels in one run, with a pattern across channels to be designed (a
  baseline or antenna flagged in a bin in most channels, flagged in all of them).
