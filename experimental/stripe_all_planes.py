from pathlib import Path
import numpy as np
from astropy.io import fits
from casacore.tables import table

cube = Path('/Users/raj030/github-wasimraja81/gmrt_analysis/casa_out/moon/moon0520_uvmin0.000kl_moontrack_cube_recovered.image.fits')
ms = Path('/Users/raj030/github-wasimraja81/gmrt_analysis/casa_out/moon/moon0520.ms')
cell_rad = np.deg2rad(2.0 / 3600.0)

with fits.open(cube, memmap=True) as h:
    data = np.asarray(h[0].data, dtype=np.float32)

with table(str(ms), readonly=True) as t:
    uvw = t.getcol('UVW')
    a1 = t.getcol('ANTENNA1')
    a2 = t.getcol('ANTENNA2')
    times = t.getcol('TIME')

with table(str(ms / 'SPECTRAL_WINDOW'), readonly=True) as s:
    freq = s.getcol('CHAN_FREQ')[0]
with table(str(ms / 'ANTENNA'), readonly=True) as a:
    names = a.getcol('NAME')
    stations = a.getcol('STATION')

lam = 299792458.0 / float(np.mean(freq))
all_u = uvw[:, 0] / lam
all_v = uvw[:, 1] / lam

uniq_times = np.unique(times)
if len(uniq_times) != data.shape[0]:
    print(f'warning: unique MS times={len(uniq_times)} cube planes={data.shape[0]}')
ntime = min(len(uniq_times), data.shape[0])

pair_vote = {}
pair_rows = {}
plane_reports = []

for idx in range(ntime):
    img = data[idx]
    if not np.isfinite(img).any():
        continue
    img = np.nan_to_num(img, nan=0.0)
    ny, nx = img.shape
    img = img - np.median(img)
    window = np.hanning(ny)[:, None] * np.hanning(nx)[None, :]
    F = np.fft.fftshift(np.fft.fft2(img * window))
    P = np.abs(F) ** 2
    cy, cx = ny // 2, nx // 2
    yy, xx = np.indices((ny, nx))
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    P[(rr < 8) | (rr > 0.2 * nx)] = 0.0
    py, px = np.unravel_index(np.argmax(P), P.shape)
    du = px - cx
    dv = py - cy
    if du == 0 and dv == 0:
        continue
    u0 = du / (nx * cell_rad)
    v0 = dv / (ny * cell_rad)

    # Only use rows from the same integration time.
    tm = uniq_times[idx]
    m = times == tm
    if not np.any(m):
        continue
    u = all_u[m]
    v = all_v[m]
    aa1 = a1[m]
    aa2 = a2[m]

    d = np.minimum(np.hypot(u - u0, v - v0), np.hypot(u + u0, v + v0))
    if d.size == 0:
        continue
    sel = np.argsort(d)[: min(50, d.size)]
    local_pairs = {}
    for j in sel:
        p = tuple(sorted((int(aa1[j]), int(aa2[j]))))
        local_pairs[p] = local_pairs.get(p, 0) + 1
    ranked = sorted(local_pairs.items(), key=lambda kv: kv[1], reverse=True)
    if not ranked:
        continue
    best_pair, best_cnt = ranked[0]
    pair_vote[best_pair] = pair_vote.get(best_pair, 0) + 1
    pair_rows[best_pair] = pair_rows.get(best_pair, 0) + best_cnt
    plane_reports.append((idx + 1, du, dv, np.hypot(u0, v0), best_pair, best_cnt))

print(f'planes_analyzed {len(plane_reports)}')
print('top_recurring_baselines')
for (i, j), votes in sorted(pair_vote.items(), key=lambda kv: (-kv[1], -pair_rows[kv[0]]))[:15]:
    print(f'{names[i]}-{names[j]} ({stations[i]}-{stations[j]})\tvotes={votes}\trowsum={pair_rows[(i, j)]}')

print('sample_plane_matches')
for rec in plane_reports[:10]:
    idx, du, dv, uvmag, (i, j), cnt = rec
    print(f'plane={idx}\tdu,dv=({du},{dv})\t|uv|={uvmag:.1f}\t{names[i]}-{names[j]}\trows={cnt}')
