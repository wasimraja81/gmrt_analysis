# Moon Imaging Notes — uGMRT Band 3 / GSB 300 MHz

## Observing Setup

| Parameter | Value |
|---|---|
| Frequency | ~300 MHz (Band 3) |
| Bandwidth | 16 MHz (GSB sub-band, 128 channels) |
| Integration time | 8 s |
| Array | GMRT, up to 30 antennas (~27 usable after flagging) |
| Baseline range | short spacings to ~25 km |

---

## Synthesised Beam and Image Grid

At 300 MHz and GMRT longest baselines (~25 km):

$$\theta_\text{beam} \approx \frac{\lambda}{B_\text{max}} = \frac{1\text{ m}}{25\text{ km}} \approx 8''$$

**Cell size rule**: 3–5 cells across the beam → cell ≤ 2–3″.  
**Adopted**: `cell = 2arcsec`, `imsize = 4096`.

Field of view:
$$\text{FOV} = 4096 \times 2'' = 8192'' \approx 2.3°$$

The Moon's disk diameter is ~0.5° = 1800″, so the image covers the full lunar
disk with ~0.9° margin on each side for background sources and sidelobe
characterisation.

---

## Thermal Noise Estimate

### Single integration (8 s snapshot)

Using the radiometer equation for a synthesis image:

$$\sigma = \frac{T_\text{sys}}{G\,\sqrt{N(N-1)\,\Delta\nu\,\tau\,n_\text{pol}}}$$

With:
- $T_\text{sys} \approx 150$ K (receiver ~50 K + sky ~100 K at 300 MHz, mid-galactic latitude)
- $G = 0.32$ K/Jy (45 m dish, ~50% aperture efficiency)
- $N = 27$ antennas → $N(N-1) = 702$ baselines
- $\Delta\nu = 16 \times 10^6$ Hz
- $\tau = 8$ s
- $n_\text{pol} = 2$ (Stokes I from RR+LL)

$$\sigma_\text{snapshot} = \frac{150}{0.32 \times \sqrt{702 \times 16 \times 10^6 \times 8 \times 2}} \approx \frac{150}{0.32 \times 134{,}000} \approx 3.5 \text{ mJy/beam}$$

> Note: $T_\text{sys}$ can reach 250–300 K toward bright galactic emission or
> at low elevations, increasing noise by up to 2×.

### Full 15-minute scan (~112 integrations stacked)

$$\sigma_\text{scan} = \frac{\sigma_\text{snapshot}}{\sqrt{112}} \approx 330\text{ µJy/beam}$$

### Multiple scans stacked (e.g. 5 × 15 min = 75 min total)

$$\sigma_\text{5scans} \approx \frac{330}{\sqrt{5}} \approx 150\text{ µJy/beam}$$

---

## Moon Surface Brightness at 300 MHz

The Moon emits thermal radiation from its subsurface (~few cm depth at these
wavelengths). At 300 MHz the effective brightness temperature is
$T_b \approx 220$ K.

Total Moon flux density:

$$S_\text{total} = \frac{2 k_B T_b \Omega_\text{moon}}{\lambda^2}$$

With $\Omega_\text{moon} = \pi (0.25°)^2 \approx 6 \times 10^{-5}$ sr and
$\lambda = 1$ m:

$$S_\text{total} \approx 35{,}000 \text{ Jy}$$

Brightness per synthesised beam (beam solid angle $\Omega_\text{beam} \approx
2 \times 10^{-9}$ sr for 8″):

$$S_\text{beam} \approx S_\text{total} \times \frac{\Omega_\text{beam}}{\Omega_\text{moon}} \approx \mathbf{600–800 \text{ mJy/beam}}$$

---

## CLEAN Threshold Assessment

| | Value |
|---|---|
| Per-snapshot thermal noise ($\sigma$) | ~3.5 mJy/beam |
| Adopted CLEAN threshold | 5 mJy (= ~1.4σ) |
| Per-scan stacked noise | ~330 µJy/beam |
| Moon disk centre brightness | ~700 mJy/beam |
| Moon:threshold ratio | ~140× |

**The 5 mJy threshold stops tclean ~1.4× above the per-snapshot noise** —
conservative enough to avoid cleaning artefacts into the model, while the
Moon signal (~700 mJy/beam) is far above it and will be fully cleaned.

For the stacked product one may wish to lower the threshold toward
~1 mJy to allow deeper cleaning of faint lunar structure and background
point sources.

---

## Imaging Strategy: Moon-Tracking Mode

Because the Moon moves ~0.55″/s (~300″ per 15-min scan), the standard
MFS imaging of a full scan would smear the Moon into a ~300″ arc.

**Adopted strategy**: per-integration phase shift + image + stack.

1. Split each integration (8 s) into a sub-MS.
2. Phase-shift visibilities (`phaseshift`/`fixvis`) to the Moon's J2000
   position at that integration's midpoint epoch.
3. Run tclean with `specmode='mfs'` on the phase-shifted sub-MS.
4. Export FITS image; stack all integrations (mean or median).

The Moon is always at the image phase centre; background sources drift
across the image and are partially suppressed in the mean stack (analogous
to a dithered observation).

**W-projection**: not required. With the Moon at phase centre and the
image spanning only ±1°, w-term phase errors are <0.1 rad for all
GMRT baselines at 300 MHz. Use `--wproject` only if limb artefacts appear.

---

## Recommended `casa_moon_imaging_example.sh` Parameters

```
cell       = 2arcsec
imsize     = 4096         # 2.3° FOV, fully covers lunar disk (0.5°)
threshold  = 5mJy         # ~1.4× per-snapshot noise; lower to 1mJy for deep stack
niter      = 6000
scales     = 0,10,30,60,120  # pixels; 10px = 20" ≈ 2× beam
weighting  = briggs, robust=0.5
uvmin      = 0 klambda    # include all baselines (no lower cut needed)
uvmax      = none         # wide open
```
