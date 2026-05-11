from pathlib import Path
import numpy as np
from astropy.io import fits
from casacore.tables import table

IMG = Path('/Users/raj030/github-wasimraja81/gmrt_analysis/casa_out/moon/moon0520__int0057_recovered.image.fits')
MS = Path('/Users/raj030/github-wasimraja81/gmrt_analysis/casa_out/moon/moon0520.ms')
CELL_ARCSEC = 2.0

# --- image Fourier analysis: find strongest non-DC spatial mode ---
with fits.open(IMG, memmap=True) as hdul:
    img = np.asarray(hdul[0].data, dtype=np.float32)

img = np.nan_to_num(img, nan=0.0)
ny, nx = img.shape
img = img - np.median(img)
win_y = np.hanning(ny)[:, None]
win_x = np.hanning(nx)[None, :]
imgw = img * win_y * win_x

F = np.fft.fftshift(np.fft.fft2(imgw))
P = np.abs(F) ** 2

cy, cx = ny // 2, nx // 2
yy, xx = np.indices((ny, nx))
rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

# suppress DC and very-high-k noise
mask = (rr >= 8) & (rr <= min(nx, ny) * 0.2)
Pm = np.where(mask, P, 0.0)
peak_idx = np.unravel_index(np.argmax(Pm), Pm.shape)
py, px = peak_idx

# choose one side of conjugate pair
du_pix = px - cx
dv_pix = py - cy
k_pix = np.sqrt(du_pix**2 + dv_pix**2)
period_pix = (np.inf if k_pix == 0 else nx / k_pix)

# map pixel frequency to uv in wavelengths
cell_rad = np.deg2rad(CELL_ARCSEC / 3600.0)
u0 = du_pix / (nx * cell_rad)
v0 = dv_pix / (ny * cell_rad)

# stripe direction in image plane is perpendicular to Fourier vector
stripe_angle_deg = (np.degrees(np.arctan2(dv_pix, du_pix)) + 90.0) % 180.0

# --- MS analysis: find baselines whose UVW cluster near this mode ---
with table(str(MS), readonly=True) as t:
    uvw = t.getcol('UVW')  # shape (nrow,3), meters
    a1 = t.getcol('ANTENNA1')
    a2 = t.getcol('ANTENNA2')

with table(str(MS / 'SPECTRAL_WINDOW'), readonly=True) as spw:
    chan_freq = spw.getcol('CHAN_FREQ')[0]  # Hz

with table(str(MS / 'ANTENNA'), readonly=True) as ant:
    names = ant.getcol('NAME')

# wavelength per channel
c = 299792458.0
lam = c / chan_freq
# use channel-average lambda for MFS-like proxy
lam0 = float(np.mean(lam))

u = uvw[:, 0] / lam0
v = uvw[:, 1] / lam0

# metric to target uv mode and its conjugate
# (u,v) and (-u,-v) are equivalent for real-image stripe pattern
d1 = np.hypot(u - u0, v - v0)
d2 = np.hypot(u + u0, v + v0)
d = np.minimum(d1, d2)

# also require orientation consistency within 20 degrees to reduce false picks
ang = np.degrees(np.arctan2(v, u))
target_ang = np.degrees(np.arctan2(v0, u0))
def angdiff(a, b):
    x = (a - b + 180.0) % 360.0 - 180.0
    return np.abs(x)

ok = (angdiff(ang, target_ang) < 20.0) | (angdiff(ang, target_ang + 180.0) < 20.0)

# nearest 2% UV samples by distance among orientation-consistent points
cand_idx = np.where(ok)[0]
if cand_idx.size == 0:
    cand_idx = np.arange(d.size)
q = np.quantile(d[cand_idx], 0.02)
sel = np.where((d <= q) & ok)[0]
if sel.size < 30:
    sel = np.argsort(d)[:max(30, int(0.01 * d.size))]

# aggregate by baseline pair
pairs = {}
for i in sel:
    p = tuple(sorted((int(a1[i]), int(a2[i]))))
    pairs[p] = pairs.get(p, 0) + 1

ranked = sorted(pairs.items(), key=lambda kv: kv[1], reverse=True)

print('IMAGE_FILE', IMG.name)
print('PEAK_FFT_PIXEL_DUV', int(du_pix), int(dv_pix))
print('EST_PERIOD_PIX', float(period_pix))
print('EST_UV_LAMBDA', float(u0), float(v0), float(np.hypot(u0, v0)))
print('EST_STRIPE_ANGLE_DEG', float(stripe_angle_deg))
print('MS_MEAN_FREQ_MHZ', float(np.mean(chan_freq) / 1e6))
print('TOP_BASELINES')
for (i, j), cnt in ranked[:15]:
    print(f'{names[i]}-{names[j]}\trows={cnt}')
