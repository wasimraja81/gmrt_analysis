# Moon UVW with-rate discrepancy — experiment checklist

**Purpose.** Find the transformation that relates the *recorded* UVW of a with-rate Moon scan to the
*expected* UVW. Each experiment below lists its possible outcomes as tick-boxes, what each outcome
implies, and where to go next. Run the step on your data, fill in the result fields, tick the one
outcome that holds, and follow its "→".

**Observation being explained.**
1. SU table has `PMRA = PMDEC = 0` → recorded UVW match expected.
2. SU table has non-zero `PMRA`/`PMDEC` (fringe-stopped on a moving Moon) → recorded UVW do not match expected.

Nothing in this file has been run. Thresholds are suggestions taken from earlier runs in this repo;
change them if your data justify it.

---

## 0. Terms and symbols (read first; every later step uses these)

| Term | Meaning |
|---|---|
| **Baseline vector b** | Separation between two antennas, from the antenna positions (`STABXYZ`), sign `ant2 − ant1`. A physical vector; its length does not depend on where we look. |
| **UVW** | The same b written in a frame tied to the direction being observed. **W** points at the phase centre, **U** east on the sky, **V** north. One UVW triple per visibility (one baseline at one integration). |
| **Phase centre** | The sky direction the correlator fringe-stopped on. For a moving Moon: position + rate × (time − reference time). |
| **u_rec** | UVW stored in the UVFITS file (converted to metres). |
| **u_exp** | UVW computed independently from b, the time, and the phase centre the correlator was told to track (the LTA source block: `RA-DATE`, `DEC-DATE`, `DRA/DT`, `DDEC/DT`). Standard hour-angle / declination equations, `H = GAST + longitude − RA`. |
| **Control scans** | Calibrator 3C468.1 and the no-rate Moon scans (moon0625, moon0635): `PM = 0`. |
| **Rate scans** | moon0520, moon0545, moon0605: non-zero `PMRA`/`PMDEC`. |
| **Rotation Q** | 3×3 matrix with `u_rec = Q · u_exp`. Entry `Q_jk` is the cosine of the angle between recorded axis j and expected axis k. A proper rotation preserves length and has determinant +1. |
| **Orthogonal Procrustes** | The least-squares rotation between two sets of vectors. Form `M`, the 3×3 sum over all visibilities of (recorded vector) × (expected vector)ᵀ, take its SVD `M = U·S·Vᵀ`, set `Q = U·diag(1,1,det(UVᵀ))·Vᵀ`. |
| **Swing** | Angle between the recorded W axis and the expected W axis, `arccos(Q₃₃)`. It equals the sky separation between the expected phase centre and the direction the recorded UVW point at. |
| **Roll** | Rotation of U and V about W that is left over after the swing is accounted for by the standard frame of the implied direction (defined in E5). |
| **Residual (m)** | RMS over all visibilities of `|Q·u_exp − u_rec|`. |

**Reference values from earlier runs (for comparison only).**
Controls: expected vs recorded ≈ 17 m RMS (calibrator), ~0.1° from the Moon (no-rate scans).
moon0520: fitted rotation 159.3°, axis ≈ (+0.90, +0.04, +0.43) for recorded→expected, fit RMS 2.0 m
(figure title; §3.3 of the investigation says ~4 m — E4 settles which). Swing 125.65°, 131.32°, 141.22°
for moon0520/0545/0605. Implied directions (RA, Dec): (101.34, −5.48), (107.47, −4.77), (121.23, −7.39).

---

## E0 — SU-table inventory (is the rate really the discriminator?)

**Purpose.** Confirm that "non-zero PM ⇔ mismatch" holds scan by scan, and check the units of PM.

**Inputs.** Every UVFITS scan: SU `PMRA`, `PMDEC`, `RAEPO/DECEPO`, `RAAPP/DECAPP`, `EPOCH`; header
`MJD_SRC`, `MJD_REF`, `DATE-OBS`; the `TUNIT` of the PM columns; LTA rates.

**Procedure.** `tools/dev/probe_import_discriminator.py` prints PM and EPOCH per scan. Add the other
fields and the PM column units. Convert LTA rate (deg/s) to the PM unit (AIPS convention is deg/day;
check the file) and compare with the stored PM.

| scan | PMRA | PMDEC | unit | LTA dRA/dt | LTA dDec/dt | ratio PM/LTA (RA, Dec) | recorded matches expected? (E1/E4) |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

**Outcome A — correspondence is one-to-one.**
- [ ] every PM ≠ 0 scan mismatches and every PM = 0 scan matches.
  *Implies:* the rate (or something carried with it) is the trigger. → E1.

**Outcome B — a counter-example exists.**
- [ ] a PM ≠ 0 scan matches, or a PM = 0 scan mismatches.
  *Implies:* PM is not the discriminator. Look for what else differs between the scans (time of day,
  scan length, plan record, source block). Re-plan before E3 onward.

**Outcome C — PM vs LTA ratio (tick one).**
- [ ] ratio ≈ 1 in the same unit: PM written faithfully.
- [ ] ratio ≈ 86400, 3600, 15, 1/cos δ, or another constant: unit or factor error in the writer. *Implies:* a rate-scaling mechanism; carry the constant into E7.
- [ ] ratio varies from scan to scan: PM is derived from something else; record it.

---

## E1 — Validate "expected" on the control scans

**Purpose.** Show that u_exp is right whenever the rate is zero, so a later mismatch is attributable
to the moving-source handling alone.

**Procedure.** Compute u_exp for each control scan and report `RMS(u_exp − u_rec)` in metres.
(`moon_uvw_checks.py mv04` sweeps conventions on the calibrator, `mv05` recovers its RA/Dec from its
own UVW, `mv07` applies the recipe to the Moon scans.)

| control scan | RMS (m) | separation implied direction vs expected (deg) |
|---|---|---|
| | | |

**Outcome A.** - [ ] all controls ≲ 30 m (≲ ~0.1°). *Implies:* frame, baseline sign, time scale and formula are right. → E2.
**Outcome B.** - [ ] a control fails. *Implies:* fix the convention first (baseline sign `ant2 − ant1`; `H = GAST + lon − RA`; `STABXYZ` is in a local-meridian frame; the file's `DATE-OBS` is IST labelled as UTC). Do not continue until the controls pass.

---

## E2 — Lengths

**Purpose.** Decide whether the recorded values differ from expected by a rotation/reflection (length
preserved) or by a scaling.

**Procedure.** Per visibility compare `|u_rec|`, `|u_exp|` and `|b|` (`moon_uvw_checks.py mv01` does the
`|UVW| = |b|` check). Report the largest relative difference per scan.

**Outcome A.** - [ ] `|u_rec| = |b|` to mm on rate scans. *Implies:* length-preserving; rotation or reflection. → E3.
**Outcome B.** - [ ] `|u_rec| ≠ |b|`. *Implies:* a scale or unit factor (seconds vs metres, wavelengths, a frequency factor). Record the factor; → E7 (units).

---

## E3 — General linear map (do not assume a rotation)

**Purpose.** Let the data say what kind of map relates u_exp to u_rec.

**Procedure.** Fit a general 3×3 matrix `L` with `u_rec ≈ L · u_exp` by ordinary least squares over all
visibilities of one rate scan. Report the singular values of `L`, `det L`, and the residual (m).
*(New script needed.)*

| scan | singular values | det L | residual (m) |
|---|---|---|---|
| | | | |

**Outcome A.** - [ ] all three singular values = 1, det = +1. *Implies:* proper rotation. → E4.
**Outcome B.** - [ ] all = 1, det = −1. *Implies:* a reflection is present (an axis sign or a handedness flip), possibly combined with a rotation. Identify it, then repeat E4 on the reflected data.
**Outcome C.** - [ ] singular values differ from 1. *Implies:* scale or shear. Check unit factors per axis; → E7 (units).
**Outcome D.** - [ ] residual stays large (≫ metres) whatever L. *Implies:* not a single linear map: it varies with time or baseline. → E4 per-chunk fit.

---

## E4 — Is it one constant rotation?

**Purpose.** Fit the rotation Q, check how well it explains the whole scan, and check whether it is constant in time.

**Procedure.** (a) Orthogonal Procrustes over the whole scan: Q, angle `arccos((trQ − 1)/2)`, axis
(proportional to `(Q₃₂−Q₂₃, Q₁₃−Q₃₁, Q₂₁−Q₁₂)`), residual. (`moon_uvw_synthesis_compare.py` does this.)
(b) Repeat on thirds of the scan and per integration; plot angle and axis against time. *(Per-chunk
fit: new script.)*

| scan | angle (deg) | axis (U,V,W) | residual (m), whole scan | angle first/mid/last third |
|---|---|---|---|---|
| | | | | |

**Outcome A.** - [ ] residual ≲ 5 m and angle/axis constant to a fraction of a degree. *Implies:* one rigid rotation per scan; built from quantities that do not change over the scan (a fixed direction or a frame convention). → E5.
**Outcome B.** - [ ] angle or axis drifts smoothly with time. *Implies:* the rotation is built from a time-varying quantity. Compare the drift rate with the Moon's rate and with the sidereal rate. → E5, E6.
**Outcome C.** - [ ] angle/axis jumps between segments. *Implies:* processing boundaries (plan records, file chunks). Locate the jump times against the plan records.
**Outcome D.** - [ ] residual large for the whole-scan fit but small per baseline group. *Implies:* baseline- or antenna-dependent defect. Fit per antenna.

---

## E5 — Geometry of Q: swing, roll, and whether the implied direction moves

**Purpose.** Decide whether the recorded UVW are the standard UVW of some *other* sky direction, or use a different axis convention.

**Procedure.**
1. Swing: `arccos(Q₃₃)`. Implied direction: decode the recorded W axis into RA/Dec per integration (`moon_uvw_checks._pointing_per_integration`).
2. Closed-form rotation for the implied direction: the rotation between the standard UVW frames of two directions depends only on the two directions (time and array location cancel). `moon_uvw_rotation_derivation.py` step (4) builds it.
3. Roll: the rotation `Q · Q_closed-formᵀ`. Both map the expected W axis to the implied W axis, so what is left is a rotation about that axis; report its angle. (Step (5) of the script reports a related roll measured from the shortest swing, not from the standard frame; this step uses the standard frame.)
4. Slope of decoded RA and Dec against time (the earlier runs report only the scatter, not the slope).

| scan | swing (deg) | implied RA, Dec | roll vs standard frame (deg) | decoded RA slope (deg/s) | decoded Dec slope (deg/s) |
|---|---|---|---|---|---|
| | | | | | |

**Outcome A (roll).**
- [ ] roll ≈ 0. *Implies:* recorded UVW are the standard UVW of a different direction. Find why the direction differs (E6, E7 rate/time mechanisms).
- [ ] roll stable and non-zero. *Implies:* the U/V axes are defined differently (a frame convention), on top of the direction change. → E7 (frame/roll candidates).
- [ ] roll varies with time. *Implies:* built from a time-varying angle (hour angle, parallactic angle). → E7.

**Outcome B (decoded direction versus time).**
- [ ] slope ≈ 0. *Implies:* the recorded frame is built for a frozen direction.
- [ ] slope ≈ the Moon's rate (0.000103, 0.000067 deg/s for moon0520). *Implies:* the recorded frame moves with the expected one (constant Q relative to a moving frame). A ±0.03° RA scatter over a scan is what a linear ramp at the Moon's rate gives, so the earlier "sky-fixed" reading is not established.
- [ ] slope is something else. *Implies:* record the value and compare with sidereal rate and E6.

---

## E6 — Dependence on the rate (the natural experiment)

**Purpose.** Learn how Q changes with the PM values, using every scan, including controls.

**Procedure.** Fill one row per scan from E4 and E5. Controls must give Q = identity (angle 0).

| scan | PMRA | PMDEC | \|rate\| | position angle of motion | UTC | angle | axis | swing | roll |
|---|---|---|---|---|---|---|---|---|---|
| controls | 0 | 0 | 0 | — | | 0 | — | 0 | 0 |
| moon0520 | | | | | | | | | |
| moon0545 | | | | | | | | | |
| moon0605 | | | | | | | | | |

**Outcome A.** - [ ] angle, swing and roll vary smoothly and go to zero as the rate goes to zero. *Implies:* a continuous function of the rate; fit its form (∝ rate, ∝ rate × time, …).
**Outcome B.** - [ ] angle/axis nearly the same for all rate scans though the rates differ. *Implies:* a switch: any non-zero rate selects a different code path or frame. The transformation is a fixed rotation per frame convention, not a function of the rate.
**Outcome C.** - [ ] parameters follow the scan time, not the rate. *Implies:* a time reference problem (stale reference epoch, UTC/IST, whole-day offset). → E7 (time lever).
**Limit.** Only three rate scans exist here; a two-parameter model has one degree of freedom left over. Say so in the conclusion.

---

## E7 — Candidate mechanisms (each must predict Q, not just fit a residual)

**Purpose.** Test named mechanisms. For each, compute the Q it predicts for every rate scan and compare
with the measured Q. A mechanism counts only if one set of parameters reproduces **all three** rate
scans to metres, **and** gives the identity for zero rate.

| # | Mechanism | Prediction to compute | Reproduces all rate scans? |
|---|---|---|---|
| a | Rate in the wrong unit (deg/day vs deg/s; missing cos δ) | Displaced phase centre = position + (wrong factor) × rate × Δt; build Q | ☐ yes ☐ no ☐ not tested |
| b | Time lever from a stale reference epoch (`MJD_SRC` ~1 day off; UTC vs IST 5.5 h) | Displacement = rate × (t − t_ref); is the displacement parallel to the rate vector? | ☐ yes ☐ no ☐ not tested |
| c | Rate defined in another frame (horizon, ecliptic, galactic) | Rotate the rate vector into that frame; is the displacement parallel to it? | ☐ yes ☐ no ☐ not tested |
| d | Roll equals the parallactic angle, or the position angle of the motion | Compute both angles at scan centre; compare with E5 roll | ☐ yes ☐ no ☐ not tested |
| e | Frame convention switch for moving sources (different axis orientation for non-zero PM) | Compare Q with known frame rotations (obliquity 23.44°, galactic, J2000 ↔ apparent) | ☐ yes ☐ no ☐ not tested |
| f | Other (state) | | |

Already tested, record the result here: rate sign flips with the time lever free
(`moon_uvw_rotation_derivation.py` step 8); wrong sidereal time over ±26 h
(`moon_uvw_synthesis_compare.py`); rate units (`tools/dev/moon_rate_units_check.py`);
benign-hypothesis sweep (`moon_uvw_checks.py mv09`).

**Outcome A.** - [ ] exactly one mechanism reproduces all three scans. *Implies:* that is the transformation; state its formula and parameters. → E9.
**Outcome B.** - [ ] more than one does. *Implies:* underdetermined with this data; E8 or more scans are needed to separate them.
**Outcome C.** - [ ] none does. *Implies:* the cause is outside this list; keep the empirical Q (E10) and widen the candidates.

---

## E8 — Toggle experiment (cleanest; optional)

**Purpose.** Change only the rate and see whether the UVW change.

**Procedure.** Re-run the LTA → UVFITS conversion for the same scan with the rate zeroed in the source
block / plan; separately, impose a synthetic rate on a control scan. Compare u_rec to u_exp for each.

**Outcome A.** - [ ] zero-rate conversion of a rate scan matches u_exp. *Implies:* the rate handling in the converter is the cause; confirm with the synthetic-rate run.
**Outcome B.** - [ ] it still mismatches. *Implies:* the cause co-varies with something else in those scans (time, plan record, source block). Return to E0.
**Outcome C.** - [ ] cannot be run (no converter or plan access). Record and continue.

---

## E9 — Validate

**Purpose.** Make sure the answer is not just a fit to one scan.

**Procedure.**
1. Leave one out: fit parameters on two rate scans, predict the third; report the residual (m).
2. Apply the same mapping to the controls: they must remain the identity.
3. Phase check: with the corrected UVW, the Moon's visibility phase should be ≈ 0 inside the first null (< ~132 λ) on the shortest baselines. A recorded-vs-corrected coherence comparison (`moon_salvageability_check.py`) is the existing indicator.

| held-out scan | residual (m) | controls still identity? | phase ≈ 0 inside first null? |
|---|---|---|---|
| | | | |

**Outcome A.** - [ ] all pass. *Implies:* the transformation is validated for this dataset.
**Outcome B.** - [ ] some scans fit, others do not. *Implies:* the mechanism is incomplete; return to E7 with the failing scan's Q.
**Outcome C.** - [ ] only the empirical per-scan Q works. *Implies:* de-rotating each scan by its own Q repairs this data but predicts nothing for new scans.

---

## E10 — Conclusion (tick one and fill in)

- [ ] **Mechanistic transformation.** Statement: ______________________________________________
  Parameters: ____________  Validated by: E9 outcome A.  Applies to future with-rate scans.
- [ ] **Empirical per-scan rotation.** Q (angle, axis) per scan: ________________  (de-rotate recorded UVW by Qᵀ.)
  Does not predict new scans.
- [ ] **Unresolved.** Last experiment reached: ____  Blocking question: ____________________________

**For imaging.** Whatever the outcome, the expected UVW for a with-rate scan can be recomputed
directly from the source model (E1 recipe). If the visibilities are fringe-stopped on the moving
phase centre, use that recomputed UVW with the matching phase centre in the imaging package.

---

## Script map

| Experiment | Existing script | Needs a new script? |
|---|---|---|
| E0 | `tools/dev/probe_import_discriminator.py` (PM, EPOCH) | add header fields, PM units, LTA ratio |
| E1 | `tools/dev/moon_uvw_checks.py mv04 / mv05 / mv07` | no |
| E2 | `tools/dev/moon_uvw_checks.py mv01` | no |
| E3 | — | yes: general 3×3 least-squares fit, singular values, det |
| E4 | `tools/dev/moon_uvw_synthesis_compare.py` (whole scan) | yes: per-chunk and per-integration Q vs time |
| E5 | `tools/dev/moon_uvw_rotation_derivation.py` steps 2, 4, 5; `moon_uvw_checks._pointing_per_integration` | yes: roll measured against the standard frame; decoded RA/Dec slope vs time |
| E6 | — | yes: assemble the table from E4/E5 |
| E7 | `moon_uvw_rotation_derivation.py` steps 6–9; `moon_rate_units_check.py` | yes: predicted-Q builders for mechanisms a–e |
| E8 | external converter | n/a |
| E9 | `tools/dev/moon_salvageability_check.py` | yes: leave-one-out driver |
