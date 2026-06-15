# Standing-wave ripple — convergence improvement TODOs

---

## Background

The dominant non-noise residual in the gain-corrected visibilities is a spectrally
periodic standing-wave ripple: $g_a(\nu) = g_{\text{smooth}}(\nu)\cdot[1 + A_a\sin(2\pi\nu/P_a + \phi_a)]$.

The smooth-window bandpass solver partially absorbs it but cannot fully remove it
because (a) the smooth window averages it out, and (b) the ripple amplitude/phase
drifts in time.  The residuals therefore contain ripple + noise rather than noise
alone, which inflates the spectral rms convergence metric and makes the stopping
threshold hard to set sensibly.

---

## Step 1 — Immediate: sinusoid-fit cleaned metrics (no solver changes)

### What to add

After computing the baseline-averaged residual spectrum per polarisation
(Stokes-V = Re⟨RR−LL⟩, and the RR/LL model-residual spectra), fit and subtract a
single sinusoid to produce *cleaned* spectra before computing convergence metrics.

For each spectrum $r(\nu)$:

$$r(\nu) = r_{\text{clean}}(\nu) + A\sin\!\bigl(2\pi\nu / P + \phi\bigr)$$

Fit the three free parameters $(A, P, \phi)$ using a bounded nonlinear least-squares
solve (e.g. `scipy.optimize.curve_fit`).  Use the known uGMRT ripple period range
(~5–15 MHz in Band 3/4/5) as a bounded prior on $P$ to prevent fitting noise.

Store three derived quantities per iteration per polarisation:
- `ripple_amplitude_jy` — $A$, the rms-equivalent amplitude of the fitted sinusoid.
  Expected to shrink each iteration as the solver absorbs more of the ripple.
- `cleaned_rms_jy` — $\text{rms}(r_{\text{clean}})$, spectral rms *after* subtraction.
  Converges towards the thermal noise floor.
- `cleaned_median_jy` — $\text{median}(r_{\text{clean}})$ (or weighted mean).

### What each quantity signals

| Quantity | V (Stokes-V) | RR / LL model residuals |
|---|---|---|
| `ripple_amplitude_jy` | Residual standing-wave power not captured by the gain solve.  Should shrink per iteration. | Same; if it stalls while rms is still decreasing, ripple is the dominant floor. |
| `cleaned_rms_jy` | Should approach the thermal noise floor $\sigma_{\text{th}} = \text{SEFD}/\sqrt{2\,\Delta\nu\,\Delta t}$.  Convergence when it stalls. | Should match V rms floor when the flux model is complete. |
| `cleaned_median_jy` | Should be ≈ 0 — calibrators have no net circular polarisation.  Non-zero → leakage or calibration asymmetry. | Should be ≈ 0 — a correct flux model leaves zero mean residual.  Non-zero → flux-scale error or missing source component. |

### Stopping criterion design

**Goal:** avoid requiring the user to specify a flux-density threshold directly,
because the correct threshold depends on the source brightness, bandwidth, and
integration time — all of which vary between datasets.

Instead, express everything in units of the *dataset's own noise floor*, which can
be estimated from the data themselves (e.g. the MAD of the highest-frequency octave
of the cleaned V spectrum, where calibrator flux is lowest and the ripple period is
shortest relative to the channel spacing, making it the most noise-like region).

#### Proposed auto-threshold logic

1. **Estimate the noise floor** $\hat{\sigma}$ from the cleaned V spectrum:

   $$\hat{\sigma} = 1.4826 \cdot \text{MAD}\bigl(r_{\text{clean}}^{V}\bigr)$$

   (the 1.4826 converts MAD to Gaussian-equivalent rms).  This is computed fresh
   each iteration from the current corrected data, so it tracks any overall flux
   scale change.

2. **Define convergence on `cleaned_rms_jy`** as:

   $$\frac{|\text{rms}_{i-1} - \text{rms}_{i}|}{\hat{\sigma}} < \tau_{\text{rms}}$$

   i.e. the *change* in cleaned rms has fallen below $\tau_{\text{rms}}$ noise units.
   A natural choice is $\tau_{\text{rms}} = 0.1$ (10% of the noise floor per
   iteration is negligible further improvement).  This is dimensionless and
   dataset-agnostic.

3. **Define a median-zero check** as:

   $$\frac{|\text{median}_i|}{\hat{\sigma}} < \tau_{\text{med}}$$

   A natural default is $\tau_{\text{med}} = 1.0$ — the median should be within one
   noise unit of zero.  If it is not, flagging/calibration has introduced a bias and
   further iterations are warranted even if the rms has stalled.  This check can also
   fire a *warning* rather than blocking convergence, since a persistent non-zero
   median may indicate a model error that more iterations cannot fix.

4. **Define ripple convergence** as:

   $$\frac{A_i}{\hat{\sigma}} < \tau_{\text{ripple}}$$

   i.e. the residual ripple amplitude has sunk below $\tau_{\text{ripple}}$ noise
   units.  A natural default is $\tau_{\text{ripple}} = 2.0$ — once the ripple is
   sub-2σ it is not meaningfully different from noise.  If $A$ stops decreasing
   *above* this threshold, the solver has reached its ripple-absorption limit and
   further iterations will not help — this is a separate warning ("ripple floor
   reached; consider Level-2 feedback or narrower smooth_window").

5. **Combined stopping condition** (all three must be satisfied simultaneously):

   $$\text{stop} \iff (\Delta\text{rms}/\hat{\sigma} < \tau_{\text{rms}})
     \;\land\; (|\text{median}|/\hat{\sigma} < \tau_{\text{med}})
     \;\land\; (A/\hat{\sigma} < \tau_{\text{ripple}})$$

   This means: the rms is no longer meaningfully changing, the median is unbiased,
   and the residual ripple is noise-level.  All three being satisfied simultaneously
   is a much stronger convergence signal than any single metric.

---

### Precise symbol definitions

#### Criterion 1 — Δrms / σ̂ < τ_rms

$$\frac{|\,\text{rms}_{i-1} - \text{rms}_i\,|}{\hat{\sigma}} < \tau_{\text{rms}}$$

| Symbol | Definition |
|---|---|
| $\text{rms}_i$ | $\sqrt{\frac{1}{N_\nu}\sum_\nu r_{\text{clean}}(\nu)^2}$ — root-mean-square of the sinusoid-subtracted residual spectrum at iteration $i$, computed over $N_\nu$ unflagged channels |
| $\text{rms}_{i-1}$ | Same quantity from the previous iteration |
| $\Delta\text{rms}$ | $|\text{rms}_{i-1} - \text{rms}_i|$ — absolute *change* in cleaned rms between consecutive iterations |
| $\hat{\sigma}$ | Noise floor estimate (see below, shared by all three criteria) |
| $\tau_{\text{rms}}$ | Dimensionless threshold (default 0.1); stop when per-iteration improvement is less than 10 % of the noise floor |

Fires when the solver is no longer meaningfully reducing scatter — further iterations
gain less than a tenth of a noise unit per step.

---

#### Criterion 2 — \|median\| / σ̂ < τ_med

$$\frac{|\,\text{median}_\nu\!\left[r_{\text{clean}}(\nu)\right]|}{\hat{\sigma}} < \tau_{\text{med}}$$

| Symbol | Definition |
|---|---|
| $r_{\text{clean}}(\nu)$ | Sinusoid-subtracted residual spectrum at iteration $i$: $r(\nu) - A\sin(2\pi\nu/P+\phi)$ |
| $\text{median}_\nu[\cdot]$ | Channel-wise median over all unflagged channels — robust to outlier channels, unlike the mean |
| $|\,\cdot\,|$ | Absolute value; we care about any signed bias, not its direction |
| $\hat{\sigma}$ | Noise floor estimate (see below) |
| $\tau_{\text{med}}$ | Dimensionless threshold (default 1.0); stop when the spectral bias is below one noise unit |

Fires when the residual spectrum has no net DC offset.  A persistent non-zero median
means either the flux model is wrong (wrong total flux → non-zero mean in RR/LL
residuals) or calibration has introduced a systematic bias (wrong polarisation balance
→ non-zero mean in V).  This is physically separate from rms: you can have a small
rms but a large bias, or a large rms but zero bias.  A persistent non-zero median
that does not shrink across iterations should fire a *warning* rather than blocking
convergence — it indicates a model error that more iterations cannot fix.

---

#### Criterion 3 — A / σ̂ < τ_ripple

$$\frac{A_i}{\hat{\sigma}} < \tau_{\text{ripple}}$$

| Symbol | Definition |
|---|---|
| $A_i$ | Fitted sinusoid amplitude at iteration $i$ from $r(\nu) = r_{\text{clean}}(\nu) + A\sin(2\pi\nu/P+\phi)$; peak amplitude of the sinusoid in Jy, as returned by the least-squares fit |
| $\hat{\sigma}$ | Noise floor estimate (see below) |
| $\tau_{\text{ripple}}$ | Dimensionless threshold (default 2.0); stop when the residual ripple is below 2 noise units |

Fires when the sinusoid that was subtracted was itself too small to matter — the
solver has already absorbed the ripple to the point where the remaining periodic
component is below the statistical detection threshold.  A value of 2.0 means the
fitted amplitude is less than $2\hat{\sigma}$, which is sub-detection for a single
sinusoid fit.  If $A$ stops decreasing *above* this threshold across iterations, the
solver has reached its ripple-absorption limit and a separate warning should be
emitted: *"ripple floor reached; consider Level-2 feedback or narrowing smooth_window"*.

---

#### The shared noise floor estimate σ̂

$$\hat{\sigma} = 1.4826 \times \text{MAD}_\nu\!\left[r_{\text{clean}}^{V}(\nu)\right]$$

where $\text{MAD}_\nu = \text{median}_\nu\!\left|r_{\text{clean}}^V(\nu) - \text{median}_\nu[r_{\text{clean}}^V(\nu)]\right|$.

| Symbol | Definition |
|---|---|
| $r_{\text{clean}}^V(\nu)$ | Sinusoid-subtracted Stokes-V spectrum Re⟨RR−LL⟩ for the current iteration |
| $\text{MAD}_\nu$ | Median absolute deviation across unflagged channels |
| $1.4826$ | Converts MAD to Gaussian-equivalent standard deviation ($\hat{\sigma} \approx \sigma$ for pure Gaussian noise) |

Computed from the **V spectrum** because V is model-free and has no flux-scale bias —
it is the cleanest estimator of the thermal noise floor available without an external
flux model.  Re-estimated at every iteration so it tracks any global flux-scale change
caused by flagging a bright antenna.  All three criteria share this denominator, so
they are expressed in the same natural unit: *multiples of the current noise floor*.

---

#### User-facing config knobs (minimal set)

| Key | Default | Meaning |
|---|---|---|
| `CONVERGENCE_TAU_RMS` | `0.1` | Δrms / noise-floor threshold. Lower = stricter. |
| `CONVERGENCE_TAU_MEDIAN` | `1.0` | Median / noise-floor tolerance (zero-bias check). |
| `CONVERGENCE_TAU_RIPPLE` | `2.0` | Ripple amplitude / noise-floor threshold. |
| `RIPPLE_PERIOD_RANGE_MHZ` | `[5.0, 20.0]` | Allowed period range for the sinusoid fit. |

The user is only exposed to dimensionless ratios relative to the noise floor — no
Jy thresholds to hand-set.  The existing `CONVERGENCE_EPSILON` (fractional change
threshold on the raw rms) remains available as a fallback for sources without a
flux model.

#### Fallback when no flux model is available

When the source has no PB2017 entry (C3/C4 unavailable):
- Compute $\hat{\sigma}$ from the cleaned V spectrum only.
- Apply the same $\tau$ logic using V metrics alone.
- Log a warning: "No flux model; using V-only metrics for convergence."

---

---

## Step 1b — Ripple characterisation and bandpass correction via 3C468.1

### Motivation

Once primary + secondary calibration is applied to 3C468.1, the baseline-averaged
spectrum should be a smooth power law (spectral index + curvature).  Any residual
standing-wave ripple sits on top of that power law as a **multiplicative modulation**
of the antenna gain bandpass:

$$V_{ij}(\nu) = g_i(\nu)\,g_j^*(\nu)\,V_{ij}^{\text{true}}(\nu)$$

where $g_a(\nu)$ contains both the smooth bandpass envelope and the standing-wave
ripple arising from cable reflections in the signal chain.  The ripple is
**multiplicative** (not additive) because it enters through the transfer function of
the receiving chain before correlation.  This means it can be corrected by
manipulating the bandpass gain table.

Since 3C468.1 has no Perley-Butler model, the power law must be fitted empirically.
The ripple is then the multiplicative residual after the power law is divided out.

---

### Physical model

$$S(\nu) = S_{\text{PL}}(\nu)\times\bigl[1 + r(\nu)\bigr]$$

where:
- $S_{\text{PL}}(\nu)$ is the smooth power-law spectrum with spectral index and
  curvature: $\log S_{\text{PL}} = a_0 + a_1 \log\nu + a_2(\log\nu)^2$ (fit
  globally over all antennas / baseline-averaged spectrum).
- $r(\nu)$ is the harmonic ripple modulation — zero-mean oscillations after the
  power law is divided out.
- Fitting: divide $S(\nu)/S_{\text{PL}}(\nu) - 1 = r(\nu)$, then fit a harmonic
  expansion (multiple sine + cosine terms) to $r(\nu)$.

**Why global power law:**  The source spectral shape is common to all baselines; the
per-antenna ripple appears on top of it.  Since the ripple is shown to be largely
common across antennas (derived from the baseline-averaged spectrum), we assume a
single global power-law + a single global harmonic ripple model as the starting
point.

---

### Harmonic ripple model

Standing waves have harmonics.  Use a Fourier (poly-harmonic) expansion rather than
a single sinusoid:

$$r(\nu) = \sum_{k=1}^{N}\Bigl[A_k\sin\!\Bigl(\frac{2\pi k\nu}{P}\Bigr)
  + B_k\cos\!\Bigl(\frac{2\pi k\nu}{P}\Bigr)\Bigr]$$

Parameters to fit: fundamental period $P$, and amplitudes $(A_k, B_k)$ for
$k = 1 \ldots N$.

| Parameter | Default | Notes |
|---|---|---|
| $N$ (harmonic order) | 3 | Captures fundamental + first two harmonics.  Expose as `RIPPLE_HARMONIC_ORDER`. |
| $P$ search range | `RIPPLE_PERIOD_RANGE_MHZ` = [5, 20] MHz | Use existing config key. |
| Auto-detect $N$ | Optional | Increment $N$ while BIC decreases; stop when adding a harmonic is not justified. |

Fitting strategy:
1. Grid-search or coarse scan over $P$ to find the dominant period.
2. Given $P$, the $(A_k, B_k)$ are linear parameters — solve via ordinary least
   squares (no nonlinear iteration needed once $P$ is fixed).
3. Optionally iterate: refine $P$ with nonlinear fit, then re-solve $(A_k, B_k)$.

---

### Workflow: input → output

```
Input:   3c468.1_primary_secondary_calibrated_flagged.uvfits
         (primary + secondary calibrated and flagged split data)

Step 1:  Load visibilities; compute baseline-averaged real spectrum per polarisation
         (use existing plot_corrected_vector_avg_spectrum computation pathway,
         but return data, not a plot).

Step 2:  Fit global power law in log-log space over unflagged channels:
         log(S) = a0 + a1*log(ν) + a2*(log(ν))^2
         → yields S_PL(ν)

Step 3:  Compute normalised residual: r(ν) = S(ν)/S_PL(ν) - 1
         r(ν) should be zero-mean if the power law fit is good.

Step 4:  Fit harmonic expansion to r(ν):
         - Grid-search for dominant period P in RIPPLE_PERIOD_RANGE_MHZ
         - Solve (A_k, B_k) via linear least squares for k = 1..N
         - Store fitted model r_hat(ν) and residual r_clean(ν) = r(ν) - r_hat(ν)

Step 5:  Diagnostics (plot):
         - S(ν) vs S_PL(ν) overlay
         - r(ν) vs r_hat(ν) overlay
         - r_clean(ν) residual (should approach noise floor)
         - FFT of r(ν) and r_clean(ν) to verify ripple suppression

Step 6:  Build bandpass correction gains:
         g_ripple(ν) = 1 / sqrt(1 + r_hat(ν))
         (square root splits the correction symmetrically across both antennas of
         each baseline: baseline response = g_i * g_j* → each antenna gets 1/sqrt)

Step 7:  Apply to existing bandpass solution:
         g_corrected(ν) = g_existing(ν) * g_ripple(ν)
         Save as new bandpass NPZ table.

Step 8:  Apply corrected bandpass to split data → write:
         3c468.1_ripple_calib.uvfits
         (should show clean power-law spectrum in baseline-averaged view)
```

---

### Implementation location

| Component | Where |
|---|---|
| Data extraction (vector-avg spectrum as array) | `ugmrt_query.get_vector_avg_spectrum(vis, solution)` — new thin function, reuses existing weighted-avg logic from `plot_corrected_vector_avg_spectrum` |
| Power-law fit | `ugmrt_query.fit_power_law_spectrum(freqs_hz, spectrum)` → returns `(a0, a1, a2)` |
| Harmonic ripple fit | `ugmrt_query.fit_harmonic_ripple(freqs_mhz, residual, period_range_mhz, n_harmonics)` → returns `(P, A_k, B_k)` |
| Bandpass correction | `ugmrt_query.apply_ripple_correction_to_bandpass(solution, g_ripple)` → returns new solution dict |
| Top-level entry point | `ugmrt_query.characterise_and_correct_ripple(vis, solution, ...)` → returns corrected solution + diagnostics dict |
| CLI / script driver | New script `bin/run_ripple_correction_3c468.1.sh` calling `src/ripple_correction.py` |
| Output UVFITS | `work/split/3c468.1/3c468.1_ripple_calib.uvfits` |

---

### Acceptance criteria

- Baseline-averaged spectrum of `3c468.1_ripple_calib.uvfits` shows a smooth power
  law with no visible periodic modulation.
- `r_clean(ν)` RMS is at or near the thermal noise floor estimate $\hat{\sigma}$.
- FFT of `r_clean(ν)` shows no dominant peak in the ripple period range.
- Fitted ripple amplitude $A_1 / \hat{\sigma} < \tau_{\text{ripple}}$ (default 2.0).

---

### Open questions / decisions deferred

- **Time dependence**: the current plan uses a single time-averaged spectrum.  If
  the ripple drifts with time or elevation, a per-scan or per-elevation-bin fit may
  be needed in a follow-up pass.
- **Per-antenna vs global**: assuming uniform ripple across antennas for now
  (same $g_{\text{ripple}}$ applied to all).  If antenna-specific ripple signatures
  are revealed by per-baseline fits, this becomes a Level-2 refinement.
- **Route choice**: bandpass manipulation is the primary route.  Direct visibility
  subtraction (additive) is not applicable here because the ripple is multiplicative
  in origin (cable reflections in the signal chain).

---

## Step 2 — Medium term: per-antenna ripple subtraction (Level-2 feedback)

Fit sinusoids per-antenna (not just for the baseline average) and subtract the
fitted ripple from the corrected visibilities *before* the next solve.  This makes
the solver see noise-like residuals in iteration $k+1$, allowing it to converge in
fewer total iterations.

Requires: per-antenna residual time-averaging before fit; minimum unflagged channel
count guard; option to disable (some users may prefer the slow/safe path).

*(No code changes yet — design pending Step 1 results.)*

---

## Step 3 — Long term: joint gain+ripple solve (Level-3)

Augment the bandpass solver to jointly fit $[\mathbf{g}_{\text{smooth}},\, A_a, P_a, \phi_a]$.
Alternating solve (block coordinate descent):

1. Fix $(A_a, P_a, \phi_a)$ → solve smooth gains
2. Fix smooth gains → solve $(A_a, P_a, \phi_a)$
3. Repeat until joint residual converges

Requires constrained $P_a$ prior (from known dish geometry or from Step 2 estimates).
Only warranted if Level-2 stalls.

*(Architecture design pending.)*
