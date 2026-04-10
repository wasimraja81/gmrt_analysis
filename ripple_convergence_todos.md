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
