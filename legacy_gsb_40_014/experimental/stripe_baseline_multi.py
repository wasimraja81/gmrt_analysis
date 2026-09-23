from pathlib import Path
import numpy as np
from astropy.io import fits
from casacore.tables import table

MS = Path('/Users/raj030/github-wasimraja81/gmrt_analysis/casa_out/moon/moon0520.ms')
IMGS = [
    '/Users/raj030/github-wasimraja81/gmrt_analysis/casa_out/moon/moon0520__int0001_recovered.image.fits',
    '/Users/raj030/github-wasimraja81/gmrt_analysis/casa_out/moon/moon0520__int0057_recovered.image.fits',
    '/Users/raj030/github-wasimraja81/gmrt_analysis/casa_out/moon/moon0520__int0113_recovered.image.fits',
]
cell_rad = np.deg2rad(2.0 / 3600.0)

with table(str(MS), readonly=True) as t:
    uvw = t.getcol('UVW')
    a1 = t.getcol('ANTENNA1')
    a2 = t.getcol('ANTENNA2')
with table(str(MS / 'SPECTRAL_WINDOW'), readonly=True) as spw:
    freq = spw.getcol('CHAN_FREQ')[0]
with table(str(MS / 'ANTENNA'), readonly=True) as ant:
    names = ant.getcol('NAME')

lam0 = 299792458.0 / float(np.mean(freq))
u = uvw[:, 0] / lam0
v = uvw[:, 1] / lam0

for imgp in IMGS:
    with fits.open(imgp, memmap=True) as h:
        img = np.asarray(h[0].data, dtype=np.float32)

    ny, nx = img.shape
    img = np.nan_to_num(img, nan=0.0)
    img = img - np.median(img)

    window = np.hanning(ny)[:, None] * np.hanning(nx)[None, :]
    F = np.fft.fftshift(np.fft.fft2(img * window))
    P = np.abs(F) ** 2

    cy, cx = ny // 2, nx // 2
    yy, xx = np.indices((ny, nx))
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    P[(rr < 8) | (rr > 0.2 * nx)] = 0

    py, px = np.unravel_index(np.argmax(P), P.shape)
    du = px - cx
    dv = py - cy

    u0 = du / (nx * cell_rad)
    v0 = dv / (ny * cell_rad)
    d = np.minimum(np.hypot(u - u0, v - v0), np.hypot(u + u0, v + v0))

    sel = np.argsort(d)[:300]
    pairs = {}
    for i in sel:
        p = tuple(sorted((int(a1[i]), int(a2[i]))))
        pairs[p] = pairs.get(p, 0) + 1

    top = sorted(pairs.items(), key=lambda kv: kv[1], reverse=True)[:8]
    print(f'\nIMG {Path(imgp).name} du,dv=({du},{dv}) |uv|={np.hypot(u0, v0):.1f} lambda')
    for (i, j), cnt in top:
        print(f'  {names[i]}-{names[j]} rows={cnt}')
