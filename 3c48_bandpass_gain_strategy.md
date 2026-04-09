# 3C48-Based Complex Antenna Bandpass Gain Derivation for uGMRT

## Objective
Estimate per-antenna complex gains as a function of frequency,

\[
G_a(\nu) = A_a(\nu) e^{i\phi_a(\nu)}
\]

using the 3C48 observation as a bandpass calibrator.

Operational constraint:
- Original FITS visibility data are never modified.
- Any copied FITS files are also treated as read-only inputs.
- Bandpass solutions are written to separate calibration files on disk.
- Science and calibrator visibilities are corrected only in memory, on-the-fly.

Given:
- 3C48 is at phase centre (point source model).
- 3C48 is effectively unpolarised for this purpose: RR and LL model visibilities are equal.
- 28 antennas give 378 baselines, so the system is strongly overdetermined.

This is exactly the classical bandpass/self-cal problem with a known calibrator model.

## Measurement Model
For antenna pair \(p,q\), channel \(\nu\), and time \(t\):

\[
V^{obs}_{pq,s}(\nu,t) = g_{p,s}(\nu,t)\, g^*_{q,s}(\nu,t)\, V^{model}_{pq,s}(\nu,t) + n_{pq,s}
\]

where:
- \(s\in\{RR,LL\}\)
- \(g_{p,s}\) is the complex gain of antenna \(p\)
- \(V^{model}_{pq,s}\) is the model visibility of 3C48
- \(n_{pq,s}\) is noise/residual corruption

Because 3C48 is at phase centre and unresolved at these baselines, the ideal model visibility is approximately baseline-independent for each channel:

\[
V^{model}_{pq,RR}(\nu) = V^{model}_{pq,LL}(\nu) = S_{3C48}(\nu)
\]

Hence, for each channel (and time block), solve:

\[
V^{obs}_{pq,s}(\nu) \approx g_{p,s}(\nu) g^*_{q,s}(\nu) S_{3C48}(\nu)
\]

## Unknowns vs Constraints
Per channel, per polarisation:
- Unknown complex gains: 28 antennas (phase reference removes one phase DOF).
- Complex equations: 378 baselines.

So the solve is highly overconstrained and robust if RFI is controlled.

## Flux Scale for 3C48
Use a standard flux density model (for example Perley-Butler 2017) to compute \(S_{3C48}(\nu)\) across channels.

If using the common polynomial form:

\[
\log_{10} S = a_0 + a_1 x + a_2 x^2 + a_3 x^3,\quad x=\log_{10}(\nu_{GHz})
\]

then evaluate at each channel center frequency and convert to Jy.

Important:
- Use the correct coefficient set for the flux scale you adopt.
- Keep this choice fixed for all subsequent target calibration.

## Practical Solve Strategy
### 1) Select calibration data
- Use only 3C48 scans.
- Keep RR and LL separately.
- Exclude known bad antennas/channels/times.

### 2) Pre-flagging
- Flag obvious RFI channels and outlier visibilities (per baseline/time/channel).
- Optionally sigma-clip amplitudes around the median per channel.

### 3) Set model visibilities
- For phase-centre point source model: \(V^{model}_{pq}(\nu)=S_{3C48}(\nu)\).
- If needed, include tiny structure corrections, but usually not required here.

### 4) Solve complex gains (per channel)
For each channel \(\nu\), minimise weighted residual:

\[
\min_{\{g_p\}} \sum_{p<q} w_{pq}\left|V^{obs}_{pq} - g_p g_q^* S_{3C48}(\nu)\right|^2
\]

Use an iterative solver (StEfCal-style updates):

\[
g_p \leftarrow \frac{\sum_q w_{pq} V^{obs}_{pq} g_q M^*_{pq}}{\sum_q w_{pq}|g_q|^2|M_{pq}|^2},\quad M_{pq}=S_{3C48}(\nu)
\]

Repeat to convergence for RR and LL.

### 5) Fix degeneracies
- Choose a reference antenna \(r\).
- Force \(\phi_r(\nu)=0\) for each channel.
- Keep absolute amplitude on flux scale via \(S_{3C48}(\nu)\).

### 6) Enforce/validate RR = LL behavior
Since 3C48 is unpolarised:
- Solve RR and LL independently first.
- Compare \(A^R_a(\nu)\) vs \(A^L_a(\nu)\), and phase offsets.
- Large systematic R/L differences indicate instrumental polarization leakage, differential electronics, or bad data.

### 7) Smooth regularisation in frequency
Bandpass should be spectrally smooth (except real instrumental features):
- Apply low-order smoothing spline or median filter to amplitude and unwrapped phase per antenna.
- Preserve real band-edge roll-offs and known filter features.

### 8) Apply and verify
Apply gains to 3C48 itself, then check:
- Calibrated amplitudes are flat vs uv-distance per channel.
- Residual phases cluster around zero.
- Closure phase near zero for a point source.
- RR/LL calibrated spectra agree.

## Recommended Quality Metrics
Track these per antenna/channel:
- Residual RMS before and after calibration.
- Gain amplitude stability over time.
- Gain phase continuity across channels.
- Fraction of flagged data used in solve.
- Condition indicators (antennas with poor SNR or disconnected baselines).

## Failure Modes and Mitigation
1. Strong RFI in subset of channels
- Use robust weighting and iterative flagging.

2. Bad reference antenna choice
- Pick a stable, high-SNR antenna; re-reference if needed.

3. Time variability during calibrator scan
- Solve in shorter time blocks, then average/smooth.

4. Band-edge instability
- Downweight/flag edge channels before final smoothing.

## Implementation Blueprint for This Dataset
Given your current workflow, a practical implementation is:

1. Read 3C48-only visibilities from the indexed loader.
2. Build per-channel baseline table:
   - baseline list (p,q)
   - complex RR, LL
   - weights/flags
3. Compute \(S_{3C48}(\nu)\) for each channel center.
4. Run iterative gain solve per channel for RR and LL.
5. Re-reference phases to one antenna.
6. Smooth gains across frequency.
7. Save calibration table:
   - antenna_id
   - freq_hz
   - g_rr_real, g_rr_imag
   - g_ll_real, g_ll_imag
   - flags/quality metrics
8. Validate on 3C48 and then apply to science fields.

## Minimal Mathematical Check
After solving, divide observed visibilities by solved gains:

\[
\tilde V_{pq}(\nu)=\frac{V^{obs}_{pq}(\nu)}{g_p(\nu)g_q^*(\nu)}
\]

For a correct solution, \(\tilde V_{pq}(\nu)\) should collapse to \(S_{3C48}(\nu)\) with small noise-like residuals, independent of baseline.

## What This Gives You
Yes, you can derive complex antenna gains as a function of frequency from this 3C48 dataset.

With 28 antennas and 378 baselines, the solve is well-conditioned if you handle RFI and enforce a stable reference.
The result is a full per-antenna bandpass calibration (amplitude and phase) for RR and LL, anchored to the known 3C48 flux spectrum.

## Non-Destructive Workflow
The intended workflow is:

1. Read the FITS visibilities from disk without changing them.
2. Derive complex bandpass gains in memory.
3. Write the gain table to a separate solution file on disk.
4. When analysing any source, load the solution file and divide visibilities by the antenna gains on-the-fly.
5. Keep the original and copied FITS files unchanged at all times.

This keeps calibration products versionable and reversible, and it avoids any risk of corrupting the raw visibility data.

## Iterative Dry-Run Workflow (Current Implementation)
The production notebook now supports iterative, non-destructive refinement:

1. Set `ITER_TAG` (for example `iter01`, `iter02`, `iter03`).
2. Run bandpass derivation with `dry_run=True` to compute gains in memory.
3. Run diagnostics and propose flag updates.
4. Keep merged proposed flag table in memory (`PENDING_FLAG_TABLES`) for the next iteration.
5. Repeat solve+diagnostics with the carried flag state.

Important behavior:
- In dry-run mode, the solve result is computed and returned in memory, but the `.npz` solution file is not written.
- Tagged output paths are still generated, so each iteration has deterministic output naming.
- On-disk flag tables and in-memory pending tables can be merged on-the-fly at solve time.

## Diagnostics-Time Flag Application (Optional)
Diagnostics now supports optional on-the-fly application of merged flag tables before ranking outliers.

Why this matters:
- Without diagnostics-time filtering, already-flagged baselines can still appear in diagnostic rankings.
- With filtering enabled, diagnostics focuses on remnant badness after the current exclusions.

Recommended usage:
- Run one diagnostics pass with merged flags applied to drive next-iteration proposals.
- Optionally save an additional unflagged comparison plot in the same iteration to visualise impact.

This preserves the non-destructive design while improving iterative convergence and interpretability.

## Next Suggested Step
Implement the solver first on one short 3C48 scan block and one frequency chunk, verify residual behavior, then scale to all channels and full scan duration.
