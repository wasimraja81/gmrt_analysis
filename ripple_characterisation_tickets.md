# Ripple Characterisation — Implementation Tickets

*Implementation ticket breakdown for the standing-wave ripple characterisation
feature designed in `ripple_convergence_todos.md` (Step 1 and Step 1b). This
file is the actionable, individually-implementable ticket list; the physical
model and design narrative live in `ripple_convergence_todos.md` and remain
the reference. Scope here is characterisation, diagnostics, and reporting
only — bandpass correction (steps 6–8 of the original Step 1b workflow) is
explicitly deferred to a future branch; see "Explicitly out of scope" below.*

---

## Status

| Ticket | Title | Status | Depends on | Commit |
|---|---|---|---|---|
| RC-00 | Create this file; cross-link `ripple_convergence_todos.md` | DONE | — | `5f03a08` |
| RC-01 | Introduce pytest: dev-requirements, `pytest.ini`, `tests/` skeleton | DONE | — | `8b3fccc` |
| RC-02 | Factor provenance/book-keeping into `workflow_common.start_run()` | DONE | — | `14b0abf` |
| RC-03 | `mad_sigma()` noise-floor estimator | DONE | RC-01 | `fb582c0` |
| RC-04 | `fit_power_law_spectrum()` | DONE | RC-01 | `878ff8c` |
| RC-05 | `derive_fourier_period_bounds()` + `find_ripple_period_candidates()` | DONE | RC-01 | `caf8244` |
| RC-06 | `fit_harmonic_ripple()` | DONE | RC-01, RC-05 | `a22991a` |
| RC-07 | `period_to_cable_length_m()` | DONE | RC-01 | `ac6602f` |
| RC-08 | Extract `get_vector_avg_spectrum()` in `ugmrt_query.py` | DONE | RC-01 | `e279b1a` |
| RC-09 | Relationship to `run_bandpass_diagnostics()`'s inline fit (decision + optional fix) | DONE | — | `f96af64` |
| RC-10 | `characterise_ripple()` orchestration function | TODO | RC-03…RC-08 | |
| RC-11 | Diagnostics plot for the science user | TODO | RC-10 | |
| RC-12 | JSON + CSV summary schema v2 | TODO | RC-10 | |
| RC-13 | CLI driver: `src/characterise_ripple.py` | TODO | RC-10, RC-11, RC-12, RC-02 | |
| RC-14 | GMRT-engineer markdown report generator | TODO | RC-10, RC-07, RC-12 | |
| RC-15 | Shell regression gate (real-data end-to-end) | TODO | RC-13 | |
| RC-16 | Validation against the prototype's existing JSON outputs | TODO | RC-13 | |
| RC-17 | gh-pages publishing wiring | DEFERRED (follow-on branch) | — | |
| RC-18 | Repo-wide runtime `requirements.txt` | OPTIONAL (not scoped to this branch) | — | |

---

## Architecture

Two-layer split, deliberately deviating from Step 1b's original
"everything in `ugmrt_query.py`" implementation-location table:

1. **`src/modules/ripple_characterisation.py`** (new module) — pure numerics
   only: NumPy arrays and plain dicts in/out, no `vis` dicts, no matplotlib
   dependency for the fit functions themselves. Exercised directly by pytest
   with synthetic ground truth, zero UVFITS/CASA dependency.
2. **One new function in `src/modules/ugmrt_query.py`**: `get_vector_avg_spectrum()`
   (RC-08), factored out of the existing `plot_corrected_vector_avg_spectrum()`
   (line 6193) as a pure, behaviour-preserving refactor — this one must live
   there because it operates on the `vis` dict shape and reuses existing
   weighting/flag logic.
3. **`src/characterise_ripple.py`** (new CLI script) — orchestration/glue
   only: loads data via existing `q.get_or_build_row_index`/
   `q.load_vis_for_source`/`q.load_bandpass_solution`/`q.apply_bandpass_solution`,
   calls `get_vector_avg_spectrum`, hands arrays to `ripple_characterisation.py`,
   writes plot/JSON/CSV/report outputs. Follows the CLI house style already
   established by `src/uv_coverage_movie.py` and
   `src/select_integrations_by_uv_coverage.py`.

**No new Python environment.** All test/runtime dependencies for this work
are installed into the existing project venv at `gmrt/` (`gmrt/bin/pip install
-r requirements-dev.txt`, `gmrt/bin/python -m pytest`). No system-wide pip, no
second virtualenv.

---

## RC-00 — Create this file; cross-link `ripple_convergence_todos.md`

### Motivation

Establish the doc/ticket discipline before any code lands, so every
subsequent ticket has a home and a status row. No GitHub Issues, no
`tickets/`/`docs/` folder exist in this repo — the established convention is
a flat root-level markdown file per topic.

### Implementation location

This file (`ripple_characterisation_tickets.md`); one-line pointer added to
`ripple_convergence_todos.md`'s Step 1b section.

### Acceptance criteria

File exists with the skeleton above; status table lists all RC-01…RC-16 as
`TODO`, RC-17/RC-18 as `DEFERRED`/`OPTIONAL`; `ripple_convergence_todos.md`
has the pointer line.

### Unit / integration tests

None (docs-only ticket).

---

## RC-01 — Introduce pytest: dev-requirements, `pytest.ini`, `tests/` skeleton

### Motivation

This repo has zero test infrastructure today (no pytest, no `tests/`
directory, no CI, no dependency manifest of any kind). Every numerics ticket
below (RC-03…RC-08) is gated on this landing first, since each ships unit
tests.

### Implementation location

- `requirements-dev.txt` (repo root) — `pytest`, `pytest-cov`, loose
  lower-bound pins (e.g. `pytest>=7.0`).
- `pytest.ini` (repo root) — `testpaths = tests`, `pythonpath = src src/modules`.
- `tests/conftest.py` — shared synthetic-data fixtures:
  `make_synthetic_spectrum(freqs_hz, alpha, ripple_components, noise_sigma, seed)`
  and a small synthetic `vis`-shaped dict fixture.
- `tests/unit/test_smoke.py` — trivial collection-works sanity test.
- `tests/README.md` — how to run, how this differs from the shell regression
  gates in `bin/`.

**No new Python environment** — installed into the existing `gmrt/` venv.

### Acceptance criteria

`gmrt/bin/pip install -r requirements-dev.txt && gmrt/bin/python -m pytest -q`
runs cleanly.

### Unit / integration tests

`test_smoke.py::test_pytest_is_wired` — trivial `assert True`, proves
collection + `pythonpath` works.

### Open questions / decisions deferred

No GitHub Actions in this branch (no `.github` directory exists anywhere in
this repo). "CI" here means the local shell gate scripts in `bin/` (RC-15).

---

## RC-02 — Factor provenance/book-keeping into `workflow_common.start_run()`

### Motivation

The `TeeStream` + `.cmd`/`.log` bookkeeping pattern is currently copy-pasted
only inside `src/stack_moon_snapshots.py` (lines ~66-123). The new CLI driver
(RC-13) needs this and shouldn't be a second copy-paste. As its own ticket
it's reusable and independently testable, and also fixes an existing bug
where `sys.stdout`/`sys.stderr` are never restored after the tee is installed.

### Implementation location

`src/modules/workflow_common.py`:
- `class TeeStream` (moved from `stack_moon_snapshots.py`).
- `def start_run(script_path: Path, out_stem: str, provenance_dir: Optional[Path] = None)`
  — creates `provenance_dir/run_<script_stem>_<out_stem>_<run_ts>.{cmd,log}`,
  writes the `.cmd` file (timestamp, cwd, quoted argv), installs `TeeStream`
  on `sys.stdout`/`sys.stderr`, returns a context object exposing `run_ts`,
  `cmd_file`, `log_file`, and a `close()`/context-manager to restore the
  original streams.

`src/stack_moon_snapshots.py` refactored to call `start_run()` instead of its
inline block, as a **pure refactor with no behaviour change** (same
`.cmd`/`.log` filenames and content).

### Acceptance criteria

`stack_moon_snapshots.py`'s provenance output is unchanged pre/post refactor
for a fixed fake clock. `characterise_ripple.py` (RC-13) uses `start_run()`
from day one.

### Unit / integration tests

- `start_run()` creates both files in a temp dir with the expected naming
  pattern.
- `.cmd` file contains the exact quoted argv (test with an argument
  containing a space).
- Writes to `print()` after `start_run()` land in both captured stdout
  (`capsys`) and the `.log` file.
- Restoring `sys.stdout`/`sys.stderr` via the returned context works.

---

## RC-03 — `mad_sigma()` noise-floor estimator

### Motivation

Directly reuses the $\hat\sigma$ design from Step 1 of
`ripple_convergence_todos.md`:

$$\hat{\sigma} = 1.4826 \times \mathrm{MAD}_\nu\!\left[r_{\text{clean}}(\nu)\right]$$

Needed by RC-05 (SNR-threshold peak detection) and by the "is this ripple
significant" framing used throughout the JSON schema and engineer report.

### Implementation location

`src/modules/ripple_characterisation.py::mad_sigma(values: np.ndarray) -> float`

### Workflow: input → output

`values` (1D array, may contain NaN) → scalar $\hat\sigma$ (`nan` if
empty/all-NaN).

### Acceptance criteria

Matches the $\hat\sigma$ definition in `ripple_convergence_todos.md` exactly
(median, MAD, ×1.4826).

### Unit / integration tests

- Seeded Gaussian noise, known `sigma` → `mad_sigma` within a generous
  tolerance of the true value (tolerance scaled to N).
- Same array with one extreme outlier injected → `mad_sigma` changes by <5%,
  contrasted against `np.std` on the same array changing by orders of
  magnitude (documents *why* MAD is used).
- All-NaN input → `nan`, does not raise.
- Empty array → `nan`.
- Constant array (MAD=0) → exactly `0.0`.

---

## RC-04 — `fit_power_law_spectrum()`

### Motivation

Implements Step 1b's power-law-with-curvature model:

$$\log S(\nu) = a_0 + a_1\log\nu + a_2(\log\nu)^2$$

Needed before any ripple residual can be formed
($r(\nu) = S(\nu)/S_{\text{PL}}(\nu) - 1$).

### Implementation location

`src/modules/ripple_characterisation.py::fit_power_law_spectrum(freqs_hz, spectrum_jy, allow_curvature=True) -> dict`

Returns:
```
{
  'coeffs': (a0, a1, a2),      # a2 == 0.0 when allow_curvature=False
  'nu0_hz': float,             # geometric-mean reference frequency
  'alpha_nu0': float,          # local spectral index at nu0
  'beta': float,               # curvature term (0.0 if allow_curvature=False)
  'model_jy': np.ndarray,      # S_PL(nu) evaluated at freqs_hz
  'rss_log': float,
  'mask_n': int,
}
```
Field names (`alpha_nu0`, `beta`, `rss_log`) chosen to match the existing
prototype's JSON keys, so RC-16's validation has a stable field to diff
against.

### Acceptance criteria

Recovers `alpha_nu0`/`beta` exactly on clean synthetic data; raises
`ValueError` on insufficient/invalid data rather than returning a garbage fit.

### Unit / integration tests

- Clean power law (no curvature, no noise) → recovered `alpha_nu0` matches
  input to <1e-6; `beta≈0`.
- Power law + curvature + Gaussian noise on `log(S)` at known σ → recovered
  `(a0,a1,a2)` within a few σ of truth (seeded RNG).
- `allow_curvature=False` on curved input → higher `rss_log` than the curved
  fit on the same data.
- Fewer than 10 valid points → raises `ValueError` with a clear message.
- Negative/zero flux values correctly masked out of the log-log fit (must not
  blow up `np.log`).
- All-NaN spectrum → raises `ValueError`.

### Open questions / decisions deferred

**Finding from unit testing:** curvature (`beta`) is only well-constrained
when the data spans a wide fractional bandwidth — `x = log10(nu/nu0)` must
vary enough that `x^2` carries real signal above the noise. GMRT's actual
GSB band (~16 MHz around ~300-330 MHz, ~5% fractional bandwidth) is far too
narrow for this: over that span `x` only ranges ~±0.02, so `beta*x^2`
contributes ~1e-4 to `log10(S)` — well below realistic per-channel noise.
Confirmed directly: a synthetic test using GMRT's real bandwidth recovered a
`beta` off by orders of magnitude under 1% log-space noise, while the same
test over a wide (100-800 MHz) synthetic band recovered `beta` accurately.
**Implication for RC-10/RC-14:** on real 3C468.1 data, `beta` (curvature)
fitted from a single 16 MHz sub-band should be treated as essentially
unconstrained/noise-dominated, not a genuine physical measurement — the
engineer report (RC-14) and any JSON consumer should not over-interpret a
nonzero fitted `beta` from this data as evidence of real spectral curvature.
This compounds the existing known-model-vs-local-fit `alpha` discrepancy
already flagged for 3C468.1 — RC-10's crosscheck output should be read with
this caveat in mind.

---

## RC-05 — `derive_fourier_period_bounds()` + `find_ripple_period_candidates()`

### Motivation

FFT-seeded period search: detrend → Hann window → FFT → local-maxima
peak-picking with parabolic sub-bin refinement → SNR vs. noise floor →
near-duplicate merge. Strategy carried over from the prior-art prototype's
period-detection idea; implementation is fresh.

### Implementation location

`src/modules/ripple_characterisation.py`:
- `derive_fourier_period_bounds(freqs_hz) -> (period_min_mhz, period_max_mhz)`
  — Nyquist-style bound from channel spacing and total bandwidth
  (`period_min = 2·Δν`, `period_max = span`).
- `find_ripple_period_candidates(freqs_hz, residual, period_min_mhz, period_max_mhz, peak_snr_threshold=10.0) -> dict`
  returning `{'periods_mhz': [...], 'snrs': [...], 'noise_floor_power': float}`.

### Acceptance criteria

Single-sinusoid synthetic residual → true period recovered among the
candidates within a few % of truth; pure-noise input → empty candidate list
at a suitably strict threshold.

### Unit / integration tests

- `derive_fourier_period_bounds`: known `Δν`/span → exact `period_min = 2·Δν`,
  `period_max = span`.
- Single known sinusoid (e.g. `P0=8.0 MHz`) + realistic noise → true period
  recovered among the candidates, `|P_recovered - 8.0| / 8.0 < 0.05`,
  `snr > 10`.
- Two independent sinusoids + noise → both true periods recovered among the
  candidates.
- Pure Gaussian noise, no injected sinusoid → empty candidate list at a
  strict threshold (the "don't fit the noise" guarantee from
  `ripple_convergence_todos.md`).
- Ripple period exactly at `period_min_mhz` boundary → still detected.
- Ripple period exactly at `period_max_mhz` boundary → still detected.
- Fewer than ~16 finite residual points → returns empty lists, does not raise
  (deliberately looser contract than RC-06's).

### Open questions / decisions deferred

**Finding from unit testing — default `peak_snr_threshold` raised from 3.0 to
10.0:** FFT power of Gaussian noise is exponentially distributed
(`P(power > k*median) ~= exp(-k*ln2)` per bin). This function searches every
bin in the requested range at once — for a realistic GMRT channel count
(~128 channels, ~65 rfft bins), a threshold of 3-5 gives a near-certain
chance of at least one spurious candidate from pure noise alone (confirmed
directly: thresholds of 3 and 5 both produced multiple false candidates from
pure Gaussian noise in testing). A Bonferroni-style budget for ~65-250
independent bins requires roughly `ln(n_bins/0.05)/ln(2)` ~ 10-12. The
default is now 10.0; this is a **statistical property of scanning many FFT
bins at once**, not implementation-specific, and should carry through to
RC-13's `--peak-snr-threshold` CLI default.

**Finding from unit testing — short-period ripples leak into the near-DC
(long-period) end of the search range:** a ripple whose period spans only a
few cycles of the total bandwidth (e.g. an 8 MHz period over a 32 MHz span —
only 4 cycles; a real GMRT case is worse still, e.g. an 8-20 MHz ripple over
a 16 MHz GSB band is 1-2 cycles) causes genuine Hann-window mainlobe leakage
into low-frequency (long-period) bins, which can register as an additional,
spurious low-confidence candidate near `period_max_mhz`. This is expected
spectral-leakage behaviour at low cycle counts, not a bug, and is why RC-05
is explicitly a **candidate-seeding** step: RC-06's proper least-squares
harmonic fit (using the injected candidates as seeds) is what separates real
ripple components from seed noise. **Implication for RC-06/RC-10:** the
harmonic fit and/or the significance-vs-noise-floor classification must not
assume every RC-05 candidate is real — low-cycle-count ripples (short
periods relative to a narrow GMRT band) are exactly the regime where this
matters most.

---

## RC-06 — `fit_harmonic_ripple()`

### Motivation

Implements the harmonic Fourier ripple model from `ripple_convergence_todos.md`:

$$r(\nu) = \sum_{k=1}^{N}\Bigl[A_k\sin\!\Bigl(\frac{2\pi k\nu}{P}\Bigr) + B_k\cos\!\Bigl(\frac{2\pi k\nu}{P}\Bigr)\Bigr]$$

Given a fixed period, $(A_k, B_k)$ are linear — solved via OLS; period is
refined by local coordinate descent around the FFT-seeded candidates from
RC-05.

### Implementation location

`src/modules/ripple_characterisation.py::fit_harmonic_ripple(freqs_hz, residual, period_candidates, n_harmonics=1, period_refine_frac=0.15) -> dict`

Single-responsibility split from RC-05: period search and period-given
curve-fitting are separate, separately-testable functions. Builds the
per-component sin/cos design matrix, solves with `np.linalg.lstsq`, does
local period refinement, classifies each component `primary`/`harmonic`/
`independent` via integer-ratio-to-primary-period logic. Returns per-component
`{period_mhz, amplitude, phase_rad, snr, classification, ratio_to_primary}`
plus overall `{rms_before, rms_after, model}`.

### Acceptance criteria

Recovers `N` known synthetic sinusoids' amplitude/phase/period within a few
percent at high SNR; degrades gracefully (no crash) at low SNR.

### Unit / integration tests

- Single known sinusoid, high SNR → recovered `(amplitude, period, phase)`
  within 2% of truth; `rms_after << rms_before`.
- Fundamental + exact 2nd harmonic → second component classified `'harmonic'`
  with `nearest_integer_order=2`.
- Two unrelated periods → both classified `'independent'`.
- Noisy data at SNR≈`peak_snr_threshold` → does not crash; `rms_after <=
  rms_before` always holds.
- Fewer than `max(20, 2·n_harmonics + 4)` finite points → raises `ValueError`
  (contrast explicitly, in the docstring, against RC-05's non-raising
  contract on too-little-data).
- Candidate period exactly at `period_min_mhz` → local refinement window
  never searches below `period_min_mhz` (no OOB/negative period).

---

## RC-07 — `period_to_cable_length_m()`

### Motivation

The one directly reusable "physical hardware" conversion — the GMRT-engineer
report (RC-14) is built entirely around this number.

### Implementation location

`src/modules/ripple_characterisation.py::period_to_cable_length_m(period_mhz: float, velocity_factor: float = 1.0) -> float`

$$L = \frac{\text{velocity\_factor} \times c}{2 f_{\text{ripple}}}$$

(round-trip reflection convention — hence the factor of 2.)

### Acceptance criteria

Exact closed-form match at known inputs; never raises.

### Unit / integration tests

- `period_mhz=10.0, velocity_factor=1.0` → `L ≈ 14.9896 m` (closed-form check
  against `c=299792458.0`).
- `velocity_factor=0.66` → linear scaling check.
- `period_mhz<=0`, `nan`, `velocity_factor<=0`, `nan` → all return `nan`,
  never raise (the engineer report must not crash on a degenerate fit).

---

## RC-08 — Extract `get_vector_avg_spectrum()` from `plot_corrected_vector_avg_spectrum()`

### Motivation

Implements Step 1b's explicit instruction: reuse the existing baseline-averaged
spectrum computation pathway, but return data, not a plot.

### Implementation location

`src/modules/ugmrt_query.py`. New function
`get_vector_avg_spectrum(vis_corrected, pol_label, chan_mask=None) -> np.ndarray`,
factored out of the weighted-average computation inside
`plot_corrected_vector_avg_spectrum()` (line 6193). `plot_corrected_vector_avg_spectrum()`
is refactored to call this new function, as a **pure refactor** — the plot's
output must not change.

### Acceptance criteria

`plot_corrected_vector_avg_spectrum()` produces numerically identical
`real_spec` output before/after the refactor, checked manually against real
data (e.g. via the existing `bin/plotVis_3c468.1_example.sh` invocation).

### Unit / integration tests

- Synthetic `vis_corrected`-shaped fixture (in `tests/conftest.py`) with known
  `vis_complex_corrected`/`weight`/`flagged_corrected` → reproduces a
  hand-computed weighted average.
- All rows flagged for one channel → that channel's output is `nan`, not `0`.
- Zero-weight rows correctly excluded from the weighted average.
- `chan_mask` correctly suppresses edge channels to `nan`.

---

## RC-09 — Relationship to `run_bandpass_diagnostics()`'s inline ripple fit

### Motivation / decision

`run_bandpass_diagnostics()`'s inline `_fit_sinusoid()` (single/double
sinusoid, bounded period 0.5–20 MHz, computed at `ugmrt_query.py:4446`)
answers *"does this iteration's corrected residual still show an obvious
wiggle, for a human watching bandpass-convergence QA right now."* The new
`ripple_characterisation.py` module answers *"what physical cable length is
causing this, and how confident are we."* **These coexist unchanged — nothing
in this branch merges, refactors, or supersedes the inline fit.**

### Optional small fix (bundled here, low-risk)

`pol_results[pol]` (built at `ugmrt_query.py:4530`) has exactly four keys
(`baseline_records`, `antenna_records`, `residual_spectrum_jy`,
`real_spectrum_jy`) — the fitted sinusoid parameters computed at lines
4446–4470 are only ever formatted into a plot-annotation string, never stored
as structured data, even though `experimental/ripple_calibration.py:266`
already defensively reads a `ripple_fit_components` key that has never
existed (silently degrading to `[]`). This ticket populates that key for
real — purely additive, no change to the figure.

### Implementation location

`src/modules/ugmrt_query.py`, inside `run_bandpass_diagnostics()`, at the
point `_popt_r`/`_popt_r2` are computed.

### Acceptance criteria

`pol_results[pol]['ripple_fit_components']` is populated (list of
`{amplitude_jy, period_mhz, phase_rad}`) when a fit succeeds, `[]` when it
doesn't; no change to the figure produced.

### Unit / integration tests

No pytest coverage — this function is I/O/matplotlib-entangled. Verified via
manual before/after figure-hash diff against real data, plus a printed sanity
check of the new key. Marked **optional**; does not block anything
downstream.

---

## RC-10 — `characterise_ripple()` orchestration function

### Motivation

Wires RC-03…RC-08 together end to end into one callable the CLI (RC-13) and
the validation ticket (RC-16) both use.

### Implementation location

`src/characterise_ripple.py`:
```python
def characterise_ripple(
    vis: dict,
    solution: dict | None,
    source: str,
    *,
    physical_model_mode: str = 'fit',   # 'fit' | 'known'
    n_harmonics: int = 1,
    peak_snr_threshold: float = 3.0,
    period_bounds_mode: str = 'fourier',
    velocity_factor: float = 1.0,
    tau_ripple: float = 2.0,
) -> dict:
```

### Workflow: input → output

1. If `solution` given, `q.apply_bandpass_solution(vis, solution)`; else use
   raw `vis`.
2. Per pol in `('RR','LL')`: `q.get_vector_avg_spectrum(...)` (RC-08).
3. `fit_power_law_spectrum(...)` (RC-04), or if `physical_model_mode='known'`,
   evaluate `q._FLUX_MODEL_REGISTRY[source]` directly.
4. `residual = spectrum/model - 1.0`.
5. `derive_fourier_period_bounds(...)` (RC-05) unless `period_bounds_mode='manual'`.
6. `find_ripple_period_candidates(...)` (RC-05).
7. `fit_harmonic_ripple(...)` (RC-06).
8. `mad_sigma(...)` (RC-03) → noise floor; classify each component's
   significance against `tau_ripple`.
9. `period_to_cable_length_m(...)` (RC-07) per component.
10. **Always** compute the known-model-vs-local-fit crosscheck when the
    source has a registry entry, regardless of `physical_model_mode`.

**Default `physical_model_mode` is `'fit'`** (local power-law), not `'known'`
— because anchoring to a possibly-wrong known model risks conflating
flux-scale error with genuine ripple (the 3C468.1 known model disagrees with
a locally-fit power law by Δα≈0.49). `known` mode remains available as a
first-class option; the crosscheck is always computed and shown either way.

### Acceptance criteria

Returns one dict containing everything the plot, JSON/CSV, and engineer-report
writers need; no matplotlib/file-I/O side effects itself.

### Unit / integration tests

Synthetic `vis`-shaped fixture with a known injected power law + known
injected ripple in `vis_complex_corrected` → recovered period/amplitude/
cable-length match truth within tolerance, end to end. Also verified: the
known-model crosscheck is populated whenever the source has a registry
entry, *regardless* of `physical_model_mode` (not gated behind
`mode='known'`); invalid `physical_model_mode`/`period_bounds_mode` and
`known` mode against an unregistered source all raise `ValueError`.

### Open questions / decisions deferred

**Finding from unit testing — a single-pass power-law fit is measurably
biased by the ripple it is fit alongside.** Fitting `fit_power_law_spectrum`
directly against ripple-contaminated data (as this orchestration does in one
non-iterative pass) recovers a spectral index (`alpha_nu0`) that is
noticeably biased toward the ripple's own shape — confirmed directly: a
known `alpha=-0.7` injected alongside a 5%-amplitude, 4-cycle ripple
recovered `alpha_nu0 ~= -0.47` (bias of ~0.23 in a single pass). This is
expected for a non-iterative fit (real pipelines iterate: fit, subtract,
refit), not a bug in RC-04's implementation (which is exact on ripple-free
data — see RC-04's own tests). **This compounds two other findings already
recorded in this doc:** RC-04's note that `beta` (curvature) is
under-constrained over GMRT's narrow (~5%) fractional bandwidth, and the
pre-existing 3C468.1 known-model-vs-local-fit discrepancy
(Delta-alpha ~ 0.49) that motivated defaulting to `physical_model_mode='fit'`
in the first place. **Implication for RC-14 (engineer report):** any
reported `alpha_nu0`/`beta` from a single characterisation run should be
presented as approximate/ripple-contaminated, not a precise independent
spectral measurement — the engineer report's caveats section should note
this alongside the known-model crosscheck caveat, since both point to the
same underlying limitation (a single-pass fit cannot cleanly separate
continuum shape from ripple).

---

## RC-11 — Diagnostics plot for the science user

### Motivation

Multi-panel diagnostic plot, following `plotVis.py`/`ugmrt_query.py`
conventions (`plt.subplots`, `dpi=180, bbox_inches='tight'`,
`mkdir(parents=True, exist_ok=True)`), deliberately simpler than the
prototype's custom annotation-collision solver.

### Implementation location

`src/characterise_ripple.py::plot_ripple_characterisation(result: dict, save_path, title='') -> Figure`

Deliberately *not* in `modules/ripple_characterisation.py` — that module's
own docstring promises "no matplotlib, no file I/O" for its pure-numerics
functions (RC-03…RC-10), so a plotting function with a `save_path` belongs
alongside the orchestration code in `characterise_ripple.py` instead, per
the two-layer split in the "Architecture" section above.

Three-row, two-column (RR/LL) layout:
- Row 1: log-log spectrum + power-law fit overlay + text box of
  `(alpha_nu0, beta, rss_log)`.
- Row 2: linear fractional ripple residual $r(\nu)$ + harmonic model overlay,
  annotated per-component (`period_mhz`, `amplitude`, `cable_length_m`,
  `classification`).
- Row 3: before/after residual with the noise floor drawn as a shaded
  $\pm2\hat\sigma$ band (`ax.axhspan`), making the significance test
  ($A/\hat\sigma < \tau_{\text{ripple}}$) visible directly on the plot.

### Acceptance criteria

One PNG per run; panels self-explanatory without the JSON open alongside;
always renders something informative even in the zero-significant-peaks
case.

### Unit / integration tests

- Runs without raising given a synthetic `result` including the
  zero-components edge case.
- Output file created and non-empty.
- Output directory created if missing.

### Open questions / decisions deferred

**Visual confirmation of a leakage/significance interaction flagged in RC-05.**
Rendering the plot against a near-noiseless synthetic case (known
`alpha=-0.7` continuum + a single injected 8 MHz/5% ripple, `noise_sigma_jy=
1e-4`) recovers the injected component accurately (`P=8.01MHz A=0.0490`,
matching truth), but the harmonic fit also picks up two low-amplitude
artifacts — a spurious ~42 MHz "independent" component and a ~4 MHz component
correctly classified as the injected ripple's own 2nd harmonic — and, because
`tau_ripple` compares component amplitude against `mad_sigma` of the
post-fit residual, the spurious 42 MHz component still crosses the
significance threshold in this pathologically low-noise case. This is the
same mechanism already recorded under RC-05 ("the harmonic fit and/or the
significance-vs-noise-floor classification must not assume every RC-05
candidate is real") — now directly visible on the rendered plot rather than
only in test assertions. Real GMRT data carries substantially more thermal
noise than this synthetic stress case, so the practical severity is expected
to be lower, but the plot's per-component text box intentionally shows every
fitted component (not just ones passing `significant`) precisely so a science
user can visually judge marginal/likely-spurious components themselves rather
than have them silently hidden. **Not fixed here** — a more robust fix (e.g.
BIC-based model selection on the number of harmonics, already noted as
optional in RC-06's original design) is out of scope for this ticket.

---

## RC-12 — JSON + CSV summary schema v2

### Motivation

Closes two gaps found in the prior-art prototype's existing JSON output:

1. **No provenance fields** — the existing summaries have no
   `generated_at`/`run_ts`/`workflow_run_id`/`git_commit`, unlike the
   gh-pages layout-manifest convention.
2. **Raw `NaN` tokens leak into JSON** — confirmed present in
   `experimental/out/ripple_characterisation_src-3c468-1_fit_summary.json`
   (`"spectral_peak_snr": NaN`), which is invalid per the JSON spec.

### Implementation location

`src/modules/ripple_characterisation.py::build_summary_dict(result, *, run_ts, workflow_run_id, git_commit, ...) -> dict`,
`::write_summary_json(summary, path)`, `::write_components_csv(summary, path)`.

Schema `gmrt-ripple-characterisation-v1`. Both a nested JSON (full detail,
one file per run) and a flattened per-`(pol, component)` CSV are written —
resolving the inconsistency between the moon-imaging PNG+CSV convention and
the ripple prototype's PNG+JSON convention by doing both, for different
consumers.

### Acceptance criteria

`generated_at`/`run_ts`/`workflow_run_id`/`git_commit` always present
(`""` when unknown, matching the layout-manifest's own "not found" convention;
`null` used for genuinely not-applicable fields like `bandpass_solution_path`
when none was used, matching the prototype's existing null convention — the
two conventions are used deliberately for different meanings, documented in
the schema). `NaN`/`Inf` never appear as raw tokens in written JSON.

### Unit / integration tests

- `build_summary_dict()` output validates against the schema (key-presence +
  type check).
- `write_components_csv()` round-trips via `csv.DictReader`.
- Missing/`None` optional fields and `NaN`/`Inf` values never produce invalid
  JSON output (explicit regression test for the observed prototype bug).

---

## RC-13 — CLI driver: `src/characterise_ripple.py`

### Motivation

Single entry point, following the house style of `src/uv_coverage_movie.py` /
`src/select_integrations_by_uv_coverage.py`.

### Implementation location

`src/characterise_ripple.py`. Argparse interface: `--fits`, `--source`,
`--index-cache`, `--bandpass-solution` (optional; omitted → characterise raw
data, stated plainly in the report), `--chan-range`, `--elevation-min/max`,
`--physical-model-mode {fit,known}` (default `fit`), `--n-harmonics`,
`--peak-snr-threshold`, `--tau-ripple`, `--velocity-factor`, `--outdir`,
`--outfile-prefix`, `--provenance-dir`. Calls `workflow_common.start_run()`
(RC-02) first thing. Writes exactly four outputs per run: plot PNG, JSON+CSV
summary, engineer report MD.

Add `bin/characterise_ripple_3c468.1_example.sh` as a filled-in example
invocation.

### Acceptance criteria

Running against real data produces all four output files (non-zero size),
exits 0, and `.cmd`/`.log` provenance files exist with the exact invoked
command line.

### Unit / integration tests

Argparse-level unit tests only (invalid choices rejected, required args
enforced). Full end-to-end behaviour is covered by the shell regression gate
(RC-15), not pytest, since it needs real UVFITS/CASA-adjacent I/O.

---

## RC-14 — GMRT-engineer markdown report generator

### Motivation

Brand-new deliverable type — nothing like it exists anywhere in this
codebase today. Designed for a hardware/signal-chain audience: plain
engineering language, minimal calibration-software jargon.

### Implementation location

`src/modules/ripple_characterisation.py::render_engineer_report_md(summary: dict) -> str`
(pure function, returns a string), written to
`<prefix>_engineer_report_<source>_<run_ts>.md` by the CLI (RC-13).

Content: "Bottom line" (period → cable length → amplitude → significance →
confidence rating), a plain-language explanation, a measured-values table, a
**caveats section** that surfaces the known-model-crosscheck discrepancy in
plain language when present (operationalising the 3C468.1 Δα≈0.49 finding as
actionable text, not just a JSON field), a
"what-would-change-this-measurement" section, and a provenance footer.

### Acceptance criteria

Renders without raising given the zero-significant-peaks edge case and the
missing-bandpass-solution edge case.

### Unit / integration tests

Pure string-content assertions (no file I/O needed):
- Synthetic `summary` with one significant RR component, none in LL → report
  contains the RR cable length and explicitly states LL had none.
- Large known-model crosscheck delta present → caveat paragraph present,
  includes the numeric delta.
- No bandpass solution applied → provenance section says "none — raw data".
- No unresolved template placeholders left in the output string.

---

## RC-15 — Shell regression gate (real-data end-to-end)

### Motivation

Task requires the shell-based, `bin/test_moon_manifest_equivalence.sh`-style
PASS/FAIL gate for the full CLI workflow against real UVFITS, separate from
pytest.

### Implementation location

- `bin/test_ripple_characterisation_regression.sh` — runs
  `src/characterise_ripple.py` twice (`--physical-model-mode fit` and
  `known`) against real 3C468.1 data under `$WORK_DIR`, writes a report file
  with markers `RUN_OK=PASS/FAIL`, `OUTPUTS_PRESENT=PASS/FAIL`,
  `JSON_VALID=PASS/FAIL` (strict `json.load`, no raw `NaN`),
  `KNOWN_MODEL_CROSSCHECK_PRESENT=PASS/FAIL`. Supports `STRICT_DATA=0` to
  skip cleanly when `$WORK_DIR` isn't populated.
- `bin/run_ripple_characterisation_regression_ci.sh` — thin wrapper, greps
  each marker.

### Acceptance criteria

Exits 0 against real data with `$WORK_DIR` populated; exits 0 with a `SKIP`
message when it isn't (never a hard failure just because local data isn't
present).

### Unit / integration tests

This *is* the test (shell-level, not pytest).

---

## RC-16 — Validation against the prototype's existing JSON outputs

### Motivation

Explicit sanity check that the fresh implementation isn't wildly different
from the known-working prior-art prototype, using the real JSON files
already on disk as reference baselines.

### Implementation location

`bin/test_ripple_characterisation_validation.sh` (separate from RC-15 — a
different pass/fail basis: numeric-tolerance comparison against two specific
historical files, not schema/presence checks).

Compares fitted dominant period (tolerance ~15–20%, since the fresh
harmonic-selection logic is deliberately simpler than the prototype's) and
fitted `alpha_nu0` (tight tolerance ~1%, plain OLS fit with no heuristic
freedom) against
`experimental/out/ripple_characterisation_src-3c468-1_fit_summary.json` and
the 3C48 equivalent.

### Acceptance criteria

`PERIOD_TOLERANCE_CHECK=PASS`, `ALPHA_TOLERANCE_CHECK=PASS` for both 3C48 and
3C468.1. A large period disagreement is investigated, not automatically
"fixed to match" — the prototype's multi-component selection is
heuristic-heavy and could itself be the one that was wrong.

### Unit / integration tests

Shell-level, real-data-optional-via-`STRICT_DATA=0` skip, same as RC-15.

---

## Explicitly out of scope (this branch)

- Building `g_ripple(ν)`, merging it into any bandpass NPZ gain table, or
  writing a ripple-corrected UVFITS (the back half of the original Step 1b
  design) — **deferred to a future branch**.
- Per-antenna ripple fitting (Step 2 / Level-2) and joint gain+ripple solve
  (Step 3 / Level-3) — already deferred in `ripple_convergence_todos.md`,
  carried forward unchanged.
- Per-scan / elevation-bin time-dependence analysis of the ripple — already
  flagged as future work in the original doc; this branch stays single
  time-averaged-spectrum only.
- **RC-17 (gh-pages publishing wiring)** — deferred to a follow-on branch. No
  stakeholder precedent exists for either the engineer report or the new
  science JSON/CSV/PNG on the existing `publish_gh_pages.sh` surface. RC-12's
  output naming stays manifest-friendly so wiring it in later is additive.
- **RC-18 (repo-wide runtime `requirements.txt`)** — adjacent optional
  cleanup, not part of this branch's ticket set.

---

## Dependency ordering

```
RC-00 (doc setup)            — no dependencies, do first
RC-01 (pytest infra)         ─┐
RC-02 (start_run)            ─┼─ independent of each other, both early
RC-03 mad_sigma               │
RC-04 fit_power_law_spectrum  ├─ all depend only on RC-01, independent of each other
RC-05 period candidates       │
RC-06 harmonic fit  (needs RC-05)
RC-07 cable length            │
RC-08 get_vector_avg_spectrum ┘
RC-09 (decision + optional fix) — independent, can land anytime
RC-10 orchestration          — needs RC-03,04,05,06,07,08
RC-11 plot                   ─┐
RC-12 JSON/CSV                ├─ both need RC-10, independent of each other
RC-13 CLI                    — needs RC-10, RC-11, RC-12, RC-02
RC-14 engineer report        — needs RC-10, RC-07, RC-12
RC-15 shell regression gate  — needs RC-13
RC-16 validation vs prototype— needs RC-13
```

Critical path: RC-01 → {RC-03…RC-08} → RC-10 → {RC-11, RC-12} → RC-13 →
{RC-15, RC-16}, with RC-02/RC-07/RC-09/RC-14 feeding in without blocking the
spine.
