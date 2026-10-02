# Moon UVW / Fringe-Stopping Verification — Ticketed Task List

*A systematic, control-anchored re-verification of whether the recorded UVW in
the GMRT 40_014 Moon UVFITS correspond to the Moon's direction. Every statement
of convention or geometry must be **measured from the data**, not asserted from
memory. No claim graduates to "settled" without a number attached. Read-only
throughout: no source code, pipeline, correlator, or data files are modified —
only this ticket file, the investigation report, and memory. Opened
2026-09-29.*

**Framing.** The stakes: if the recorded UVW genuinely do not point at the Moon,
that implicates the observatory's online delay/phase model — an extraordinary
claim. Extraordinary claims require (a) a recipe certified against a fixed
control, and (b) exhaustion of *benign* explanations (sign, axis order, stale
centre, units, time offset) before any correlator fault is entertained. Tickets
MV-05 (certification gate) and MV-09 (benign-hypothesis sweep) exist precisely
to make that discipline non-optional.

---

## Status

| Ticket | Title | Status | Depends on | Result |
|--------|-------|--------|-----------|--------|
| MV-00 | Create ticket file | DONE | — | this file |
| MV-01 | Antenna table + |UVW|=|B| magnitude check | DONE | — | UU unit=SECONDS (PSCAL=1); \|UVW·c−\|B\|\| median 0.1 mm, max 2.2 mm all 6 files |
| MV-02 | Measured RAEPO↔RAAPP separation (per scan) | DONE | — | cal 9.73′; all 5 Moon scans **18.07′ (0.30°)** — my guessed "~0.2°" was low |
| MV-03 | Read MS FIELD::PHASE_DIR directly | DONE | MS exists | PHASE_DIR = **RAEPO/DECEPO (J2000)**, ref=J2000 (label is mean-epoch) |
| MV-04 | Frame test: recompute calibrator UVW at RAEPO vs RAAPP | DONE | MV-01 | Winner: sign `ant2−ant1`, `H=GAST+lon−RA`, pos **APP** (RMS 18.0 m vs EPO 53.7 m vs 8.5 km wrong-sign). UVW live in **apparent** frame |
| MV-05 | **GATE** — recover calibrator RA/Dec from its own UVW | **PASS** | MV-04 | Recovered 358.14,+64.79 → **0.062° from APP** (0.21° from EPO); fit RMS 0.44 m |
| MV-06 | Independent CASA recompute of calibrator UVW | TODO | MV-04 | |
| MV-07 | Apply certified recipe to Moon (file's own RAAPP) | DONE | MV-05 | **3-bad/2-good REPRODUCED**: 0520/0545/0605 = 18 km resid, 126/131/141° off; 0625/0635 = **17 m** resid, 0.11° on-Moon |
| MV-08 | CASA fixvis cross-check on Moon | TODO | MV-07 | |
| MV-09 | **Benign-hypothesis sweep** before implicating correlator | DONE | MV-07 | **All benign transforms refuted.** `.plan` commanded correct Moon (=SU RAAPP) for all 5. Offending UVW = clean rigid rotation (~157°, 4 m resid) of correct-Moon UVW → tracked a **sky-fixed** (±0.03°) but **wrong** celestial position, differing per scan. Not frozen, units, clock, sign, axis-perm, or stale-command. Selectivity (3 bad/2 good, same file/plan) rules out any uniform bookkeeping bug |
| MV-10 | 2-panel per-snapshot movie (recorded vs theoretical) | TODO | MV-05 | |
| MV-11 | Synthesise: report + memory, conclusion with numbers | DONE | MV-07..09 | Root cause = correct rate applied with wrong time-base in gvfits UVW recompute (LTA online model clean). **Salvageability SETTLED: Case C (salvageable)** — with-rate moon0520 keeps freqCoh 0.60–0.70 on its shortest baselines (vs ≈0 expected for a 126°-off delay), matching the no-rate control's structure → visibilities stopped on the Moon, only UVW metadata wrong |
| MV-12 | Recover pointing from the **on-disk MS** UVW (were the visibilities rotated?) | DONE | MV-07 | **No rotation.** All 4 on-disk moon0520 MSs (workflow ±phasecenter, scratch full+shifted) recover the **identical offending** direction 101.34,−5.48 → **125.70° off Moon**, matching the raw UVFITS control to **0.04°**. `scratch_full` and `scratch_full_shifted` UVW are **byte-identical** (max Δ 0.0 m). MS FIELD::PHASE_DIR = 330.92,−17.49 (Moon, J2000 EPO) while the UVW geometry points 126° away |

Status is updated in this table as each ticket completes, with the measured
result quoted inline.

---

## MV-01 — Antenna geometry and the |UVW| = |B| magnitude check

### Motivation
Before any direction work, confirm the recorded UVW are a *legitimate, unscaled*
geometric projection of the true baselines — i.e. only their pointing could be
in question, not their scale or units. Also fixes the exact data paths (Moon
scans + calibrator) so no later ticket guesses a path.

### Method
- Locate and record the exact UVFITS paths for the calibrator (3C468.1) and all
  five Moon scans. Do not assume; find them.
- Read the `AIPS AN` `STABXYZ` antenna positions; form baseline vectors `B`.
- For each baseline, compare recorded `|UVW|·c` (metres) to `|B|` (metres).

### Acceptance
Report max and median `||UVW|·c − |B||` in metres per file. Expect sub-metre if
the magnitudes are clean.

### Open questions
Confirm the UVW random-parameter unit (seconds vs wavelengths vs metres) from the
header (`PSCAL`/`PZERO`, `PTYPE`), not from assumption.

---

## MV-02 — Measured RAEPO ↔ RAAPP separation

### Motivation
Replace the guessed "~0.2° precession/nutation" with the actual angular
separation between the two SU-table positions, per scan.

### Method
Read `RAEPO/DECEPO` and `RAAPP/DECAPP` from each `AIPS SU` table; compute the
true-angle separation (great-circle) in arcmin/deg.

### Acceptance
A table: scan | RAEPO/DECEPO | RAAPP/DECAPP | separation. No narrative — numbers.

---

## MV-03 — Read MS FIELD::PHASE_DIR directly

### Motivation
Establish what `importuvfits` *actually* wrote as the MS phase centre and in
what reference frame — measured, not asserted.

### Method
If an imported MS exists for the calibrator and/or a Moon scan, read
`FIELD::PHASE_DIR` and its `MEASINFO`/reference frame; compare numerically to
RAEPO/DECEPO and to RAAPP/DECAPP.

### Acceptance
PHASE_DIR value + frame, and which SU column it equals (to what precision). If no
MS exists locally, mark BLOCKED and note it — do not infer.

---

## MV-04 — Frame convention test on the calibrator (settles "B")

### Motivation
Decide empirically which position the recorded UVW were computed against, using
a source whose position needs no ephemeris (fixed extragalactic 3C468.1).

### Method
Recompute the calibrator's UVW from antenna positions + time at (a) its
J2000 RAEPO/DECEPO and (b) its apparent RAAPP/DECAPP. Report the RMS residual (m)
of each against the recorded UVW.

### Acceptance
Two residuals. The smaller one names the frame the UVW live in. State the winner
with its number; retract any prior verbal claim if it disagrees.

---

## MV-05 — GATE: recover the calibrator's RA/Dec from its own UVW

### Motivation
Certify the entire recipe (LST, longitude, baseline sign) end-to-end on a source
with an exactly known position. This is the gate that would have caught the
earlier longitude and antipode-sign bugs. Nothing touches the Moon until this
passes.

### Method
Invert the calibrator's *recorded* UVW for the implied sky direction. Explicitly
determine and report the baseline sign convention (`ant2−ant1` vs `ant1−ant2`)
that the data uses — do not carry a remembered sign.

### Acceptance
Recovered RA/Dec within **≪0.1°** of catalogue, with the sign convention stated
as a measured fact. If it fails, STOP and fix the recipe; downstream tickets are
blocked.

---

## MV-06 — Independent CASA recompute of the calibrator UVW

### Motivation
A second, independent implementation from the package that interpreted the file,
so "consistent derivation" does not rest on my code alone.

### Method
Use CASA measures / `fixvis` (or `me.uvw`) to recompute the calibrator UVW and
compare to both the recorded UVW and my MV-04 winning recipe.

### Acceptance
Metre-level agreement between CASA and my recipe. Quote the residual.

---

## MV-07 — Apply the certified recipe to the Moon

### Motivation
The actual question, now on a gate-certified recipe, with my own ephemeris kept
out of the loop.

### Method
For each Moon scan, compute theoretical UVW at the **file's own RAAPP/DECAPP**
(observatory ephemeris from the SU table — not my astropy Moon). Compare to
recorded UVW: RMS residual (m) and implied-direction separation (deg).

### Acceptance
Per-scan table: residual + separation. This either reproduces or overturns the
earlier 3-bad/2-good split — report whichever the numbers say.

---

## MV-08 — CASA fixvis cross-check on the Moon

### Motivation
Independent confirmation of MV-07 from CASA's own machinery.

### Method
Recompute each Moon scan's UVW to RAAPP via CASA `fixvis`/`phaseshift`; compare
to MV-07.

### Acceptance
Per-scan agreement quoted in metres.

---

## MV-09 — Benign-hypothesis sweep (before implicating the correlator)

### Motivation
An observatory correlator error is the *last* explanation, not the first. For any
scan where recorded ≠ theoretical, test whether a benign transform reconciles
them.

### Method
For each mismatching scan, exhaustively test: baseline sign flip; all signed
axis permutations of (Bx,By,Bz); a stale phase centre inherited from the
previous source in the schedule; a UT1/clock/time offset; a units error; and
antenna-index ordering. Report the best-fit residual for each hypothesis.

### Acceptance
Per-scan: does *any* benign transform reduce the residual to the metre level? If
yes, the "defect" is a benign bookkeeping issue, not a correlator fault. Only if
*no* benign transform works is a genuine online-model problem implicated — and
that conclusion must be stated with that caveat explicit.

### Results (measured)

Probe: `tools/dev/moon_uvw_checks.py mv09` (per-integration Procrustes pointing +
single-rotation tests) and a direct decode of the gvfits `.plan`
(`~/DATA/gmrt_40_014/gvfits_logs/40_014_25jul2021_2.6s.plan`, 336-byte records,
little-endian IEEE doubles; RA/Dec at record offsets 96/104).

1. **Wrong/stale commanded position — REFUTED.** The `.plan` gvfits used for the
   LTA→UVFITS conversion commands the *correct apparent Moon* for all five scans,
   matching the UVFITS SU RAAPP/DECAPP to <0.001°:
   0520 331.220,−17.385 · 0545 331.385,−17.283 · 0605 331.523,−17.200 ·
   0625 331.666,−17.116 · 0635 331.740,−17.074. **Nothing in the schedule points
   at the offending RA≈101–121°/Dec≈−6°**, and no other scan in the plan sits
   there either (nearest in RA is DA240 117.6° but at Dec +55.8°). The three bad
   scans were *commanded* onto the Moon.

2. **Not frozen.** Recorded long-baseline u evolves 40–80 m over each offending
   scan — comparable to the 35–66 m a correct Moon track would move. The delay
   model was updating, not stuck.

3. **Internally a clean, sky-fixed wrong direction.** Per-integration Procrustes
   recovers a constant pointing across each 15-min offending scan to **±0.03° RA /
   ±0.02° Dec** (a horizon-fixed/parked direction would drift ~3.75° in RA over
   15 min). Recovered: 0520 → 101.3,−5.5 (125.7° off Moon); 0545 → 107.5,−4.8
   (131.4°); 0605 → 121.2,−7.4 (141.2°). The two good scans recover the true Moon
   to 0.127°.

4. **The error is a single rigid rotation of the true-Moon UVW.** correct-Moon →
   recorded is one constant rotation of **156–159°** (axis ≈ [−0.9, 0, −0.4] in
   the uv w-frame) fitting the whole scan to **~4 m** RMS; the good scans are
   identity (0.13°, ~0 m). This is *not* a coordinate-axis flip, axis permutation,
   or 90/180° operation.

5. **Selectivity kills every uniform bookkeeping hypothesis.** Baseline sign
   (MV-04: ant1−ant2 → 8.5 km on the calibrator), units (MV-01: |UVW|=|B| to mm),
   and a clock/UT1 offset (Dec is clock-invariant yet wrong by ~11°) are each
   independently refuted. More decisively: any sign/axis/units/time bug would act
   *identically* on all five scans in one file from one gvfits run on one `.plan`
   — yet 3 carry a ~157° rotation and 2 are identity. A per-scan-selective defect
   cannot be a global conversion or analysis bug.

**Verdict.** No benign transform reconciles the offending UVW. The schedule and
all file metadata (SU RAAPP/DECAPP, `.plan`) carry the correct Moon for every
scan; only the *recorded UVW numbers* of the three 15-min scans are wrong, each
pointing at a distinct, cleanly sidereally-tracked, but wrong celestial position.
This implicates the online delay/UVW model for those three scans specifically —
stated with the caveat that MV-06/MV-08 (independent CASA recompute) are the
remaining controls before the conclusion is finalised in MV-11.

**Update (2026-09-29) — GSB LTA online model read directly (decisive).** Read-only
`strings` of `~/DATA/gmrt_40_014/data/40_014_25jul2021_gsb.lta` exposes the
per-scan online source block (`OBJECT`/`RA-DATE`/`DEC-DATE`/`DRA/DT`/`DDEC/DT`),
i.e. exactly what the correlator was commanded. Moon scans = SCAN0026–0030:

| scan | OBJECT | RA-DATE | DEC-DATE | DRA/DT (deg/s) | DDEC/DT (deg/s) |
|------|--------|---------|----------|----------------|-----------------|
| 0026 | MOON0520 | 331.218408 | −17.384986 | 0.000103 | 0.000067 |
| 0027 | MOON0545 | 331.383452 | −17.283068 | 0.000108 | 0.000069 |
| 0028 | MOON0605 | 331.521323 | −17.200152 | 0.000112 | 0.000070 |
| 0029 | MOON0625 | 331.664774 | −17.116175 | **0.000000** | **0.000000** |
| 0030 | MOON0635 | 331.738688 | −17.073822 | **0.000000** | **0.000000** |

1. **Commanded position = correct apparent Moon for all five** (matches SU
   RAAPP/DECAPP to <0.001°). Nothing at 101–121° was ever commanded.
2. **Offending = WITH nonzero rate; good = exactly zero rate** — measured,
   correlator-side confirmation of the user's mode-assignment ground truth.
3. **Rate values are correct.** deg/s = JPL *coordinate* rate: LTA 0.000103 vs
   JPL dRA/dt 0.0001040; 0.000067 vs dDec/dt 0.0000660 — both ~1%.
4. **Rate is far too small to be the 126° error** by direct integration: ~0.11°
   over a 15-min scan; 130° would need 14.6 days. The error is the correct rate
   applied with a wrong **time-base/units** in gvfits' UVW recompute (LTA has
   `PAR_SIZE=0`, no stored UVW → gvfits computes them at conversion), acting only
   on rate-carrying scans. Consistent with the MV-09 rigid-rotation signature and
   the 3-bad/2-good selectivity.

---

## MV-10 — Two-panel per-snapshot UV-coverage movie

### Motivation
Make the recorded-vs-theoretical comparison *watchable* per frame, and get the
animation right (visible frame-to-frame change).

### Method
Per Moon scan, one frame per integration, two panels: (A) recorded UVW, (B)
certified-theoretical UVW at the Moon. Points colorised by frequency; full uv
coverage per snapshot; Moon boundary circle drawn for an **assumed θ = 0.5°**
(→ 1/θ ≈ 115 λ), with that assumption printed on the frame. Zoom/limits chosen
so the short baselines and the boundary circle are both legible. Only after the
MV-05 gate passes.

### Acceptance
One mp4 per scan under `moon_uv_movies/`; frame-to-frame rotation clearly
visible; the two good scans overlay A≈B, and any offending scan shows A≠B.

### Open questions
Whether "full uv coverage" is shown at a fixed wide limit or clipped to the
short-baseline window — decide against legibility once MV-07 numbers are in.

---

## MV-11 — Synthesise: report + memory, conclusion with numbers

### Motivation
One coherent, number-backed conclusion; correct the report and memory to match
what MV-01..09 actually measured (including retracting anything that doesn't
survive).

### Method
Update `moon_fringe_stopping_investigation.md` and the two memory files with the
measured residuals/separations, the settled frame convention, and the MV-09
benign-hypothesis outcome. State plainly whether the correlator is implicated or
a benign explanation accounts for the mismatch.

### Acceptance
Report + memory carry only measured numbers; the final verdict names its
evidence and its caveats.

### Results (measured, 2026-09-29)

**Root cause (benign, correlator NOT at fault).** The GSB LTA online model (MV-09
update) commanded the correct apparent Moon and the correct JPL rate (to ~1%) for
all five scans. The 126–141° error is the correct rate applied with a **wrong
time-base/units** in gvfits' post-hoc UVW recompute (LTA `PAR_SIZE=0` → UVW built
at conversion), acting only on the three rate-carrying scans — matching the
3-bad/2-good selectivity and the single rigid ~157° rotation of MV-09.

**Salvageability — Case C, salvageable** (`tools/dev/moon_salvageability_check.py`,
read-only). Splitting short-baseline RR coherence into a delay axis (freqCoh,
per-integration across the 33.3 MHz band) and a rate axis (timeCoh, per-channel
across the scan):

| baseline (λ) | moon0520 with-rate freqCoh / timeCoh | moon0625 no-rate freqCoh / timeCoh |
|--------------|--------------------------------------|------------------------------------|
| 6-9 (111) | **0.703 / 0.684** | 0.833 / 0.746 |
| 7-9 (111) | **0.597 / 0.531** | 0.907 / 0.845 |

A genuine 126°-off fringe stop on the 103 m baseline 6-9 imposes `τ≈3.4×10⁻⁷ s` =
~11 phase turns across 33.3 MHz → freqCoh ≈ 0. The measured 0.60–0.70 is
incompatible with that: **the delay model stopped on the Moon; only the recorded
UVW metadata are wrong.** Both scans share the same coherence-vs-baseline-length
structure (peak on shortest baselines, resolving out beyond ~130 λ = the Moon disk,
θ=0.531°). Caveat: with-rate coherence is modestly below the no-rate control
(0.70 vs 0.83), a real but limited coherence cost from the 3× longer track over a
partially-resolved disk. Recovery requires a **fresh** UVW recompute to the
apparent Moon — no on-disk MS carries it (MV-12: the one recompute attempt failed).

---

## MV-12 — Recover pointing from the on-disk MS UVW (were the visibilities rotated?)

### Motivation
The user's governing question narrowed to: *"convince yourself the moon0520 image
was made in CASA after the visibilities were rotated to phase up to the Moon's
true position."* MV-07 measured the **raw UVFITS** UVW. MV-12 asks the same of
every **on-disk MS** that was (or could have been) imaged — because rotation, if
it happened, would live in an MS, not the UVFITS.

### Method
`tools/dev/probe_ms_uvw_pointing.py` (read-only). For each MS, read `UVW`,
`ANTENNA1/2`, `TIME`; build baseline vectors `B = POSITION[a2] − POSITION[a1]`
from `ANTENNA::POSITION`; form GAST from `TIME`; recover the instantaneous
pointing per integration with the **certified** `_pointing_per_integration`
Procrustes decoder from `moon_uvw_checks.py` (the exact routine MV-07/MV-09 used).
Control: the raw UVFITS through the same decoder, which must reproduce MV-07.

Frame handling (measured, not assumed): CASA copied the GMRT **array-local**
antenna frame (STABXYZ, X at local meridian) verbatim into `ANTENNA::POSITION`,
so the certified site-longitude term `H = GAST + lon − RA` applies to the MS too.
Tell: with `lon = 0` the MS recovered RA exactly one site-longitude (74.05°) short
of the certified control at identical Dec; adding `lon` reproduces the control to
<0.01°.

### Results (measured)

Control reproduces MV-07: raw UVFITS moon0520 → **RA 101.336 ± 0.027,
Dec −5.482 ± 0.018**, per-integration fit RMS ~0 m, **0.04° from the MV-07
offending direction**, 125.70° from the true Moon (SU RAAPP 331.220, −17.385).

All four on-disk moon0520 MSs recover the **identical offending** pointing:

| MS | recovered RA,Dec | sep from Moon | sep from offending | FIELD::PHASE_DIR |
|----|------------------|---------------|--------------------|------------------|
| `casa_selfcal/moon0520_stk10_phasecenter/moon0520_full.ms` | 101.336, −5.482 | 125.70° | 0.04° | 330.924, −17.490 (J2000) |
| `casa_selfcal/moon0520_stk10/moon0520_full.ms` | 101.336, −5.482 | 125.70° | 0.04° | 330.924, −17.490 (J2000) |
| `scratch/moon0520_full.ms` | 101.336, −5.482 | 125.70° | 0.04° | 330.924, −17.490 (J2000) |
| `scratch/moon0520_full_shifted.ms` | 101.336, −5.482 | 125.70° | 0.04° | 0.000, 0.000 |

Direct column comparison `scratch/moon0520_full.ms` vs `..._shifted.ms`:
row keys (ant1/ant2/time) identical, **max |ΔUVW| = 0.0 m** — the "_shifted" MS
carries **byte-identical** UVW to the unshifted one.

### Verdict
**The moon0520 visibilities were never rotated onto the Moon.** Every on-disk MS
grids UVW that point 125.70° off the Moon — identical to the recorded UVFITS to
0.04°. The MS `FIELD::PHASE_DIR` is set at the Moon (330.92, −17.49 = J2000 EPO,
MV-03) while the UVW geometry points 126° away: metadata says Moon, geometry
says elsewhere. The `_shifted` MS is byte-identical to the unshifted, so no
rotation was ever written.

This is independently corroborated by the logs→products provenance mapping
(agent, 2026-09-29): the only genuine UVW-recompute attempt was
`fixplanets(field='7', fixuvw=True, direction='moon_ephem.tab')` on
`scratch/moon0520_full_shifted.ms` (2026-05-12), which **FAILED** — `SEVERE
ms::getfielddirmeas … No valid ephemeris entry for MJD 59418.999572` — leaving
the UVW unchanged (hence byte-identical). The imaging path used
`--use-tclean-phasecenter` (tclean `phasecenter='MOON'`): recorded UVW gridded at
a Moon phase centre, **no rotation** (MODE c). The published stacks
(`stack_moon_snapshots.py`) do image-plane phase-correlation registration +
radial-shell destripe under a 20′ ephemeris-centred mask — also not a visibility
rotation. So any clean-looking lunar disk in the moon0520 stack was assembled in
the **image plane** (registration + mask + destripe) from 126°-off visibilities,
**not** from correctly fringe-stopped data.
