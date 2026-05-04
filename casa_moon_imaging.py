#!/usr/bin/env python3
"""CASA imaging helper for split Moon UVFITS files.

Designed for per-scan imaging of the calibrated Moon UVFITS files produced by
`visSplit_moon_example.sh`.  Each scan is imported to a CASA MeasurementSet and
imaged independently with a minimum UV cut of 0.12 kλ by default, which keeps
baselines sensitive to the lunar disk while discarding shorter spacings that
probe larger-than-Moon structure.

This script can be run either:
  1. with a Python that has `casatasks` available, or
  2. through a CASA launcher, e.g.
       casa --nogui --nologger -c casa_moon_imaging.py --fits ...
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, cast

import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import EarthLocation, get_body
from astropy.time import Time


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Import and image split Moon UVFITS files with CASA')
    p.add_argument('--fits', nargs='+', required=True, help='One or more split Moon UVFITS files')
    p.add_argument('--outdir', default='./casa_out/moon', help='Output directory for MS and image products')
    p.add_argument('--cell', default='4arcsec', help='Image cell size (default: 4arcsec)')
    p.add_argument('--imsize', nargs='+', type=int, default=[1024], help='Image size in pixels: one value or two values')
    p.add_argument('--stokes', default='I', help='Output Stokes parameter (default: I)')
    p.add_argument('--spw', default='',
                   help='Optional CASA spw channel selection (default: all channels), e.g. "0:0~127"')
    p.add_argument('--expected-nchan', type=int, default=None,
                   help='Optional guard: fail if input UVFITS NAXIS4 != expected value')
    p.add_argument('--uvmin-klambda', type=float, default=0.12, help='Minimum uv cut in kλ (default: 0.12)')
    p.add_argument('--uvmax-klambda', type=float, default=None, help='Optional maximum uv cut in kλ')
    p.add_argument('--weighting', default='briggs', choices=['natural', 'uniform', 'briggs'], help='CASA weighting scheme')
    p.add_argument('--robust', type=float, default=0.0, help='Briggs robust parameter (default: 0.0)')
    p.add_argument('--niter', type=int, default=5000, help='Maximum CLEAN iterations (default: 5000)')
    p.add_argument('--cycleniter', type=int, default=500, help='Minor iterations per major cycle/chunk (default: 500)')
    p.add_argument('--cycles-per-report', type=int, default=1, help='Major cycles per progress report chunk (default: 1)')
    p.add_argument('--threshold', default='0mJy', help='CLEAN threshold, e.g. 5mJy (default: 0mJy)')
    p.add_argument('--scales', default='0,5,15,45,135', help='CASA multiscale scales in pixels (default: 0,5,15,45,135)')
    p.add_argument('--deconvolver', default='multiscale', choices=['multiscale', 'hogbom', 'clark'],
                   help='CASA deconvolver (default: multiscale)')
    p.add_argument('--wprojplanes', type=int, default=64,
                   help='Number of W-projection planes (default: 64).')
    p.add_argument('--moon-track-per-integration', action='store_true',
                   help='Image each integration separately after phasing visibilities to per-integration Moon position, then stack.')
    p.add_argument('--integration-step', type=int, default=1,
                   help='Use every Nth integration in moon-track mode (default: 1, i.e. all).')
    p.add_argument('--integration-niter', type=int, default=300,
                   help='tclean niter per integration in moon-track mode (default: 300).')
    p.add_argument('--integration-cycleniter', type=int, default=100,
                   help='tclean cycleniter per integration in moon-track mode (default: 100).')
    p.add_argument('--integration-nmajor', type=int, default=2,
                   help='Maximum major cycles per integration in moon-track mode (default: 2; -1 for auto).')
    p.add_argument('--stack-method', choices=['mean', 'median'], default='mean',
                   help='Stacking method for per-integration FITS images (default: mean).')
    p.add_argument('--keep-integration-products', action='store_true',
                   help='Keep per-integration image products in moon-track mode (default: delete after stacking).')
    p.add_argument('--overwrite', action='store_true', help='Remove existing MS/image products before re-running')
    p.add_argument('--keep-ms', action='store_true', help='Keep imported MeasurementSets after imaging')
    p.add_argument('--export-fits', action='store_true', help='Export CASA .image products to FITS')
    return p.parse_args()


def _import_casa_tasks():
    try:
        from casatasks import exportfits, importuvfits, split, tclean  # type: ignore
    except Exception as exc:  # pragma: no cover - import path depends on local CASA install
        raise SystemExit(
            'Could not import CASA tasks. Run this with a CASA-enabled Python or use '\
            '`casa --nogui --nologger -c casa_moon_imaging.py ...`. '\
            f'Original import error: {exc}'
        )

    phaseshift_task = None
    fixvis_task = None
    try:
        from casatasks import phaseshift as phaseshift_task  # type: ignore
    except Exception:
        phaseshift_task = None
    try:
        from casatasks import fixvis as fixvis_task  # type: ignore
    except Exception:
        fixvis_task = None

    if phaseshift_task is None and fixvis_task is None:
        raise SystemExit('Neither casatasks.phaseshift nor casatasks.fixvis is available for phase shifting.')

    return importuvfits, tclean, exportfits, split, fixvis_task, phaseshift_task


def _normalize_imsize(values: Iterable[int]) -> list[int]:
    vals = [int(v) for v in values]
    if len(vals) == 1:
        return [vals[0], vals[0]]
    if len(vals) == 2:
        return vals
    raise ValueError('--imsize expects one value or two values')


def _parse_scales(text: str) -> list[int]:
    vals = [int(v.strip()) for v in str(text).split(',') if v.strip()]
    if not vals:
        raise ValueError('--scales must contain at least one integer')
    if 0 not in vals:
        vals.insert(0, 0)
    vals = sorted(set(vals))
    return vals


_STOP_CODES: dict[int, str] = {
    0: 'not yet converged',
    1: 'niter reached',
    2: 'threshold reached',
    3: 'force stop / divergence guard',
    4: 'user stop',
}


def _run_tclean_with_progress(
    tclean,
    *,
    label: str,
    total_niter: int,
    cycleniter: int,
    cycles_per_report: int,
    **tclean_kwargs,
) -> dict[str, Any]:
    """Run CASA tclean in restart chunks and print major-cycle style progress."""
    total_niter = int(total_niter)
    cycleniter = max(1, int(cycleniter))
    cycles_per_report = max(1, int(cycles_per_report))

    chunk_niter = cycleniter * cycles_per_report
    iters_done = 0
    major_done = 0
    first_call = True
    result: dict[str, Any] = {}

    estimated_majors = max(1, (total_niter + cycleniter - 1) // cycleniter)
    print(
        f'[CASA] tclean starting  label={label}  '
        f'niter={total_niter}  cycleniter={cycleniter}  '
        f'(~{estimated_majors} major cycles expected)'
    )

    while iters_done < total_niter:
        remaining = total_niter - iters_done
        this_chunk = min(chunk_niter, remaining)

        call_kwargs = dict(tclean_kwargs)
        call_kwargs['niter'] = this_chunk
        call_kwargs['cycleniter'] = cycleniter

        if first_call:
            call_kwargs['calcpsf'] = True
            call_kwargs['calcres'] = True
            call_kwargs['restart'] = False
            first_call = False
        else:
            call_kwargs['calcpsf'] = False
            call_kwargs['calcres'] = False
            call_kwargs['restart'] = True

        result = tclean(**call_kwargs) or {}

        chunk_iters = int(result.get('iterdone', this_chunk))
        iters_done += chunk_iters

        nmajor = int(result.get('nmajordone', 0))
        major_by_iter = max(1, (iters_done + cycleniter - 1) // cycleniter)
        major_done = max(major_done, major_by_iter, nmajor)

        peak_res = float('nan')
        summaryminor = result.get('summaryminor', None)
        if isinstance(summaryminor, dict):
            try:
                any_chan = next(iter(summaryminor.values()))
                any_pol = next(iter(any_chan.values()))
                any_field = next(iter(any_pol.values()))
                peaks = any_field.get('peakRes', [])
                if peaks:
                    peak_res = float(peaks[-1])
            except Exception:
                pass

        stop_code = int(result.get('stopcode', -1))
        stop_desc = str(result.get('stopDescription') or _STOP_CODES.get(stop_code, f'code {stop_code}'))

        pct = 100.0 * iters_done / total_niter
        print(
            f'[CASA]   major cycle ~{major_done:3d}/{estimated_majors}  |  '
            f'iter {iters_done:6d}/{total_niter} ({pct:5.1f}%)  |  '
            f'peak residual {peak_res:.4g} Jy/beam  |  {stop_desc}'
        )
        sys.stdout.flush()

        if chunk_iters < this_chunk:
            print(f'[CASA] tclean stopping early: {stop_desc}')
            break

        if chunk_iters == 0:
            print('[CASA] tclean returned 0 iterations done - stopping loop.')
            break

    print(f'[CASA] tclean finished  label={label}  total_iter_done={iters_done}')
    return result


def _uvrange_expr(uvmin_klambda: float, uvmax_klambda: float | None) -> str:
    if uvmax_klambda is None:
        return f'>{uvmin_klambda:.3f}klambda'
    return f'{uvmin_klambda:.3f}~{uvmax_klambda:.3f}klambda'


def _remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _remove_imagename_products(imagename: Path) -> None:
    for p in imagename.parent.glob(imagename.name + '.*'):
        _remove_path(p)


def _read_uvfits_nchan(fits_path: Path) -> int | None:
    try:
        with fits.open(fits_path) as hdul:
            primary = cast(Any, hdul[0])
            val = primary.header.get('NAXIS4', None)
            return int(val) if val is not None else None
    except Exception:
        return None


def _normalize_uvfits_for_casa(fits_path: Path, outdir: Path, label: str) -> Path:
    """Normalize UVFITS metadata so CASA importuvfits can ingest problematic
    GMRT/AIPS UVFITS variants.

    Steps:
      1) patch a temp copy so pyuvdata can read frame metadata,
      2) force timesys=UTC in-memory,
      3) re-write UVFITS with pyuvdata.
    """
    try:
        from pyuvdata import UVData
    except Exception as exc:
        raise RuntimeError(
            'CASA import fallback needs pyuvdata installed. '
            f'Original import error: {exc}'
        )

    readfix = outdir / f'{label}__casa_readfix.uvfits'
    normalized = outdir / f'{label}__casa_norm.uvfits'

    if readfix.exists():
        _remove_path(readfix)
    if normalized.exists():
        _remove_path(normalized)

    shutil.copy2(fits_path, readfix)
    with fits.open(readfix, mode='update') as hdul:
        primary = cast(Any, hdul[0])
        primary.header['RADESYS'] = 'fk5'
        if 'AIPS AN' in hdul:
            an_hdu = cast(Any, hdul['AIPS AN'])
            an = an_hdu.header
            an['FRAME'] = 'ITRF'
            an['XYZHAND'] = 'RIGHT'
        hdul.flush(output_verify='ignore')

    uv = UVData()
    uv.read_uvfits(str(readfix))
    setattr(uv, 'timesys', 'UTC')
    uv.write_uvfits(str(normalized), force_phase=True)

    _remove_path(readfix)
    return normalized


def _get_unique_integration_times_mjd(ms_path: Path) -> tuple[np.ndarray, float]:
    from casatools import table  # type: ignore

    tb = table()
    tb.open(str(ms_path))
    try:
        times_sec = np.asarray(tb.getcol('TIME'), dtype=np.float64)
    finally:
        tb.close()

    unique_sec = np.unique(np.round(times_sec, 6))
    if unique_sec.size > 1:
        cadence_sec = float(np.median(np.diff(unique_sec)))
    else:
        cadence_sec = 8.0
    return unique_sec / 86400.0, cadence_sec


def _to_casa_timerange(mjd_center: float, half_width_sec: float) -> str:
    t0 = Time(mjd_center - half_width_sec / 86400.0, format='mjd', scale='utc')
    t1 = Time(mjd_center + half_width_sec / 86400.0, format='mjd', scale='utc')
    dt0 = cast(Any, t0.to_datetime(timezone=None))
    dt1 = cast(Any, t1.to_datetime(timezone=None))
    s0 = dt0.strftime('%Y/%m/%d/%H:%M:%S.%f')[:-3]
    s1 = dt1.strftime('%Y/%m/%d/%H:%M:%S.%f')[:-3]
    return f'{s0}~{s1}'


def _moon_phasecenter_j2000(mjd_utc: float) -> str:
    from casatools import measures, quanta  # type: ignore

    gmrt = EarthLocation.from_geodetic(lon=74.0497 * u.deg, lat=19.0965 * u.deg, height=650 * u.m)
    t = Time(mjd_utc, format='mjd', scale='utc', location=gmrt)
    # Compute the topocentric apparent Moon position at GMRT, then convert it
    # to a CASA-compatible J2000 direction using CASA measures at the same
    # observatory and epoch.  Direct ICRS/J2000 from astropy under-tracks the
    # Moon here, while direct APP failed in phaseshift on this MS.
    moon_app = cast(Any, get_body('moon', t, location=gmrt))
    app_ra = moon_app.ra.to_string(unit=u.hour, sep='hms', precision=3, pad=True)
    app_dec = moon_app.dec.to_string(unit=u.deg, sep='dms', precision=3, alwayssign=True, pad=True)

    me = measures()
    qa = quanta()
    me.doframe(me.position('WGS84', '74.0497deg', '19.0965deg', '650m'))
    me.doframe(me.epoch('UTC', t.isot))
    app_dir = me.direction('APP', app_ra, app_dec)
    j2000_dir = me.measure(app_dir, 'J2000')

    ra = qa.formxxx(j2000_dir['m0'], format='hms', prec=3)
    dec = qa.formxxx(j2000_dir['m1'], format='dms', prec=3)
    return f'J2000 {ra} {dec}'


def _image_moon_per_integration(
    *,
    ms_path: Path,
    imagename_base: Path,
    label: str,
    split_task,
    fixvis_task,
    phaseshift_task,
    tclean_task,
    exportfits_task,
    args: argparse.Namespace,
) -> None:
    times_mjd, cadence_sec = _get_unique_integration_times_mjd(ms_path)
    step = max(1, int(args.integration_step))
    selected = times_mjd[::step]
    total = len(selected)
    half_width_sec = max(0.1, 0.45 * cadence_sec)

    print(
        f'[CASA] moon-track mode: {label} integrations={len(times_mjd)} '
        f'(using every {step} -> {total}), cadence~{cadence_sec:.2f}s, '
        f'niter/int={int(args.integration_niter)}, cycleniter/int={int(args.integration_cycleniter)}, '
        f'nmajor/int={int(args.integration_nmajor)}, wprojplanes={int(args.wprojplanes)}'
    )

    fits_paths: list[Path] = []
    cube: list[np.ndarray] = []
    header0 = None
    for idx, mjd in enumerate(selected, start=1):
        timerange = _to_casa_timerange(float(mjd), half_width_sec)
        phasecenter = _moon_phasecenter_j2000(float(mjd))

        int_ms = imagename_base.parent / f'{label}__int{idx:04d}.ms'
        phased_ms = imagename_base.parent / f'{label}__int{idx:04d}_phased.ms'
        int_imagename = imagename_base.parent / f'{label}__int{idx:04d}'

        _remove_path(int_ms)
        _remove_path(phased_ms)
        _remove_imagename_products(int_imagename)

        split_task(
            vis=str(ms_path),
            outputvis=str(int_ms),
            datacolumn='data',
            timerange=timerange,
            spw=str(args.spw or ''),
            keepflags=False,
        )
        if not int_ms.exists():
            continue

        if phaseshift_task is not None:
            phaseshift_task(
                vis=str(int_ms),
                outputvis=str(phased_ms),
                phasecenter=phasecenter,
            )
        elif fixvis_task is not None:
            fixvis_task(
                vis=str(int_ms),
                outputvis=str(phased_ms),
                phasecenter=phasecenter,
            )
        else:
            raise RuntimeError('No CASA phase-shift task available (phaseshift/fixvis).')

        tclean_task(
            vis=str(phased_ms),
            imagename=str(int_imagename),
            spw=str(args.spw or ''),
            phasecenter=phasecenter,
            imsize=_normalize_imsize(args.imsize),
            cell=args.cell,
            specmode='mfs',
            deconvolver=args.deconvolver,
            scales=_parse_scales(args.scales),
            weighting=args.weighting,
            robust=args.robust,
            niter=int(args.integration_niter),
            cycleniter=int(args.integration_cycleniter),
            threshold=args.threshold,
            stokes=args.stokes,
            uvrange=_uvrange_expr(args.uvmin_klambda, args.uvmax_klambda),
            gridder='wproject',
            wprojplanes=int(args.wprojplanes),
            interactive=False,
            nmajor=int(args.integration_nmajor),
            pbcor=False,
            calcpsf=True,
            calcres=True,
            savemodel='none',
        )

        image_path = Path(str(int_imagename) + '.image')
        fits_out = Path(str(int_imagename) + '.image.fits')
        if image_path.exists():
            try:
                exportfits_task(imagename=str(image_path), fitsimage=str(fits_out), overwrite=True)
            except Exception as exc:
                print(f'[CASA] warning: exportfits failed for integration {idx}: {exc}')
            if fits_out.exists():
                fits_paths.append(fits_out)
                with fits.open(fits_out) as hdul:
                    primary = cast(Any, hdul[0])
                    data = np.asarray(primary.data, dtype=np.float32)
                    if data.ndim == 4:
                        data = data[0, 0, :, :]
                    elif data.ndim != 2:
                        data = np.squeeze(data)
                    if header0 is None:
                        header0 = primary.header.copy()
                    cube.append(data)
            else:
                print(f'[CASA] warning: no FITS image for integration {idx}, skipping in stack')
        else:
            print(f'[CASA] warning: no .image product for integration {idx}, skipping in stack')

        if idx % 5 == 0 or idx == total:
            print(f'[CASA] moon-track progress: integration {idx}/{total}')

        _remove_path(int_ms)
        _remove_path(phased_ms)
        if not bool(args.keep_integration_products):
            _remove_imagename_products(int_imagename)

    if not cube:
        raise RuntimeError(f'No per-integration images generated for {label}')

    stack = np.stack(cube, axis=0)
    n_ints = len(fits_paths)
    if args.stack_method == 'median':
        out_data = np.nanmedian(stack, axis=0)
    else:
        out_data = np.nanmean(stack, axis=0)

    hdr = header0 if header0 is not None else fits.Header()

    def _write_stack(suffix: str, data: np.ndarray, method_label: str) -> None:
        p = Path(str(imagename_base) + f'_moontrack_{suffix}.image.fits')
        if p.exists() and args.overwrite:
            _remove_path(p)
        h = hdr.copy()
        h['HISTORY'] = f'Moon per-integration stacked image ({method_label}), n={n_ints}'
        fits.PrimaryHDU(data=data.astype(np.float32), header=h).writeto(p, overwrite=bool(args.overwrite))
        print(f'[CASA] moon-track stacked image: {p.name} (n={n_ints})')

    # Primary stack (mean or median as chosen)
    _write_stack(args.stack_method, out_data, args.stack_method)

    # RMS per pixel across integrations
    rms_data = np.sqrt(np.nanmean(stack ** 2, axis=0))
    _write_stack('rms', rms_data, 'rms')

    # Max per pixel across integrations
    max_data = np.nanmax(stack, axis=0)
    _write_stack('max', max_data, 'max')


def main() -> int:
    args = _parse_args()
    importuvfits, tclean, exportfits, split_task, fixvis_task, phaseshift_task = _import_casa_tasks()

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    imsize = _normalize_imsize(args.imsize)
    scales = _parse_scales(args.scales)
    uvrange = _uvrange_expr(args.uvmin_klambda, args.uvmax_klambda)
    spw_sel = str(args.spw or '').strip()
    spw_label = spw_sel if spw_sel else 'all'

    for fits_path_str in args.fits:
        fits_path = Path(fits_path_str).expanduser().resolve()
        if not fits_path.exists():
            raise SystemExit(f'Input UVFITS not found: {fits_path}')

        input_nchan = _read_uvfits_nchan(fits_path)
        if input_nchan is not None:
            print(f'[CASA] input UVFITS channels (NAXIS4): {input_nchan}')
            if args.expected_nchan is not None and int(input_nchan) != int(args.expected_nchan):
                raise SystemExit(
                    f'Input channel count mismatch for {fits_path.name}: '
                    f'NAXIS4={input_nchan}, expected={int(args.expected_nchan)}'
                )
        elif args.expected_nchan is not None:
            print('[CASA] warning: could not read input NAXIS4 for --expected-nchan check')

        stem = fits_path.stem
        if stem.endswith('.uvfits'):
            stem = Path(stem).stem
        label = stem.replace('_calibrated', '')

        ms_path = outdir / f'{label}.ms'
        imagename = outdir / f'{label}_uvmin{args.uvmin_klambda:.3f}kl'

        if args.overwrite:
            _remove_path(ms_path)
            _remove_imagename_products(imagename)

        if not ms_path.exists():
            print(f'[CASA] importuvfits: {fits_path.name} -> {ms_path.name}')
            try:
                importuvfits(fitsfile=str(fits_path), vis=str(ms_path))
            except Exception as exc:
                print(f'[CASA] importuvfits failed, trying pyuvdata normalization fallback: {exc}')
                norm_fits = _normalize_uvfits_for_casa(fits_path, outdir, label)
                print(f'[CASA] importuvfits (normalized): {norm_fits.name} -> {ms_path.name}')
                importuvfits(fitsfile=str(norm_fits), vis=str(ms_path))
                _remove_path(norm_fits)
        else:
            print(f'[CASA] reusing existing MS: {ms_path}')

        print(
            f'[CASA] tclean: label={label} imsize={imsize} cell={args.cell} '
            f'uvrange={uvrange} stokes={args.stokes} spw={spw_label}'
        )
        if args.moon_track_per_integration:
            _image_moon_per_integration(
                ms_path=ms_path,
                imagename_base=imagename,
                label=label,
                split_task=split_task,
                fixvis_task=fixvis_task,
                phaseshift_task=phaseshift_task,
                tclean_task=tclean,
                exportfits_task=exportfits,
                args=args,
            )
        else:
            _run_tclean_with_progress(
                tclean,
                label=label,
                total_niter=args.niter,
                cycleniter=args.cycleniter,
                cycles_per_report=args.cycles_per_report,
                vis=str(ms_path),
                imagename=str(imagename),
                spw=spw_sel,
                imsize=imsize,
                cell=args.cell,
                specmode='mfs',
                deconvolver=args.deconvolver,
                scales=scales,
                weighting=args.weighting,
                robust=args.robust,
                threshold=args.threshold,
                stokes=args.stokes,
                uvrange=uvrange,
                gridder='wproject',
                wprojplanes=int(args.wprojplanes),
                interactive=False,
                pbcor=False,
                savemodel='none',
            )

        image_path = Path(str(imagename) + '.image')
        if (not args.moon_track_per_integration) and args.export_fits and image_path.exists():
            fits_out = Path(str(imagename) + '.image.fits')
            if fits_out.exists() and args.overwrite:
                _remove_path(fits_out)
            print(f'[CASA] exportfits: {fits_out.name}')
            exportfits(imagename=str(image_path), fitsimage=str(fits_out), overwrite=bool(args.overwrite))

        if not args.keep_ms:
            _remove_path(ms_path)
            print(f'[CASA] removed MS: {ms_path.name}')

    print(f'[CASA] done. outputs in {outdir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
