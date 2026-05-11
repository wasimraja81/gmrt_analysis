from pathlib import Path"""Patch casa_moon_imaging.py to add cross-scan stacking.""""""Patch casa_moon_imaging.py to add cross-scan stacking."""

































































































































print(f'\nAll {ok} patches applied and written to {p}')p.write_text(src)print('P5 ok')ok += 1src = src.replace(a5, b5, 1)assert a5 in src, 'P5 not found'b5 = fn + "def main() -> int:"a5 = "def main() -> int:""""    _write('rms', rms, 'rms')    _write(method, combined, method)        print(f'[CASA] cross-scan stack written: {out.name}  (n_scans={n})')        astrofits.writeto(str(out), data.astype(np.float32), hdr)        hdr['HISTORY'] = f'Moon cross-scan stack ({label}), n_scans={n}'        hdr = ref_header.copy()                return                print(f'[CASA] cross-scan stack: skipping existing {out.name} (use --overwrite)')            else:                out.unlink()            if overwrite:        if out.exists():        out = outdir / f'moon_allscans_{suffix}.image.fits'    def _write(suffix, data, label):    outdir = Path(outdir)    rms = np.sqrt(np.nanmean(arr ** 2, axis=0))        combined = np.nanmean(arr, axis=0)    else:        combined = np.nanmedian(arr, axis=0)    if method == 'median':    arr = np.stack(cube, axis=0)        cube.append(data)                ref_header = hdul[0].header.copy()            if ref_header is None:            data = hdul[0].data.astype(np.float32)        with astrofits.open(str(fp)) as hdul:    for fp in fits_paths:    cube, ref_header = [], None    print(f'[CASA] cross-scan stack: combining {n} scan images (method={method})')    n = len(fits_paths)    from astropy.io import fits as astrofits    import numpy as np    \"\"\"      moon_allscans_rms.image.fits       - per-pixel RMS across scans      moon_allscans_<method>.image.fits  - combined stack    Outputs written to outdir:    were processed with identical parameters by this script.    Each image must share the same WCS/cell/imsize — guaranteed when all scans    \"\"\"Stack per-scan moontrack FITS images across all input scans.def _cross_scan_stack(fits_paths, outdir, method='mean', overwrite=False):fn = """# P5: helper function before main()print('P4 ok')ok += 1src = src.replace(a4, b4, 1)assert a4 in src, 'P4 not found')    "    return 0"    "    print(f'[CASA] done. outputs in {outdir}')\n"    "\n"    "        _cross_scan_stack(scan_stack_fits, outdir, args.cross_scan_stack_method, args.overwrite)\n"    "    if args.moon_track_per_integration and args.cross_scan_stack and scan_stack_fits:\n"b4 = ()    "    return 0"    "    print(f'[CASA] done. outputs in {outdir}')\n"a4 = (# P4: call cross-scan stack after loopprint('P3 ok')ok += 1src = src.replace(a3, b3, 1)assert a3 in src, 'P3 not found')    "            _run_tclean_with_progress("    "        else:\n"    "                print(f'[CASA] warning: per-scan stack not found: {_candidate.name}')\n"    "            else:\n"    "                scan_stack_fits.append(_candidate)\n"    "            if _candidate.exists():\n"    "            _candidate = Path(str(imagename) + f'_moontrack_{args.stack_method}.image.fits')\n"    "            )\n"b3 = (a3 = "            )\n        else:\n            _run_tclean_with_progress("# P3: collect per-scan resultprint('P2 ok')ok += 1src = src.replace(a2, b2, 1)assert a2 in src, 'P2 not found')    "    for fits_path_str in args.fits:"    "    scan_stack_fits = []  # per-scan moontrack stacked FITS for cross-scan stack\n"b2 = (a2 = "    for fits_path_str in args.fits:"# P2: collection list before per-scan loopprint('P1 ok')ok += 1src = src.replace(a1, b1, 1)assert a1 in src, 'P1 not found')    "    return p.parse_args()"    "                   help='Stacking method for cross-scan combination (default: mean).')\n"    "    p.add_argument('--cross-scan-stack-method', choices=['mean', 'median'], default='mean',\n"    "                   help='Stack per-scan moon-track images into a single combined output (default: True).')\n"    "    p.add_argument('--cross-scan-stack', action=argparse.BooleanOptionalAction, default=True,\n"    "    p.add_argument('--export-fits', action='store_true', help='Export CASA .image products to FITS')\n"b1 = ()    "    return p.parse_args()"    "    p.add_argument('--export-fits', action='store_true', help='Export CASA .image products to FITS')\n"a1 = (# P1: new CLI argsok = 0src = p.read_text()p = Path(__file__).parent / 'casa_moon_imaging.py'








































































































print(f'\nAll {ok} patches applied and written to {p}')p.write_text(src)src = src.replace(a5, b5, 1); ok += 1; print('P5 ok')assert a5 in src, 'P5 not found'b5 = fn + "def main() -> int:"a5 = "def main() -> int:""""    _write('rms', rms, 'rms')    _write(method, combined, method)        print(f'[CASA] cross-scan stack written: {out.name}  (n_scans={n})')        astrofits.writeto(str(out), data.astype(np.float32), hdr)        hdr['HISTORY'] = f'Moon cross-scan stack ({label}), n_scans={n}'        hdr = ref_header.copy()                return                print(f'[CASA] cross-scan stack: skipping existing {out.name} (use --overwrite)')            else:                out.unlink()            if overwrite:        if out.exists():        out = outdir / f'moon_allscans_{suffix}.image.fits'    def _write(suffix, data, label):    outdir = Path(outdir)    rms = np.sqrt(np.nanmean(arr ** 2, axis=0))    combined = np.nanmedian(arr, axis=0) if method == 'median' else np.nanmean(arr, axis=0)    arr = np.stack(cube, axis=0)  # (n_scans, stokes, freq, y, x)        cube.append(data)                ref_header = hdul[0].header.copy()            if ref_header is None:            data = hdul[0].data.astype(np.float32)        with astrofits.open(str(fp)) as hdul:    for fp in fits_paths:    cube, ref_header = [], None    print(f'[CASA] cross-scan stack: combining {n} scan images (method={method})')    n = len(fits_paths)    from astropy.io import fits as astrofits    import numpy as np    \"\"\"      moon_allscans_rms.image.fits       - per-pixel RMS across scans      moon_allscans_<method>.image.fits  - combined stack    Outputs written to outdir:    were processed with identical parameters by this script.    Each image must share the same WCS/cell/imsize — guaranteed when all scans    \"\"\"Stack per-scan moontrack FITS images across all input scans.def _cross_scan_stack(fits_paths, outdir, method='mean', overwrite=False):fn = """\# ── 5. insert helper function before main() ───────────────────────────────────src = src.replace(a4, b4, 1); ok += 1; print('P4 ok')assert a4 in src, 'P4 not found'      "    return 0")      "    print(f'[CASA] done. outputs in {outdir}')\n"      "\n"      "        _cross_scan_stack(scan_stack_fits, outdir, args.cross_scan_stack_method, args.overwrite)\n"b4 = ("    if args.moon_track_per_integration and args.cross_scan_stack and scan_stack_fits:\n"a4 = "    print(f'[CASA] done. outputs in {outdir}')\n    return 0"# ── 4. call cross-scan stack after the loop ───────────────────────────────────src = src.replace(a3, b3, 1); ok += 1; print('P3 ok')assert a3 in src, 'P3 not found'      "            _run_tclean_with_progress(")      "        else:\n"      "                print(f'[CASA] warning: per-scan stack not found: {_candidate.name}')\n"      "            else:\n"      "                scan_stack_fits.append(_candidate)\n"      "            if _candidate.exists():\n"      "            _candidate = Path(str(imagename) + f'_moontrack_{args.stack_method}.image.fits')\n"b3 = ("            )\n"a3 = "            )\n        else:\n            _run_tclean_with_progress("# ── 3. collect per-scan result after _image_moon_per_integration call ─────────src = src.replace(a2, b2, 1); ok += 1; print('P2 ok')assert a2 in src, 'P2 not found'      "    for fits_path_str in args.fits:")b2 = ("    scan_stack_fits = []  # per-scan moontrack stacked FITS for cross-scan stack\n"a2 = "    for fits_path_str in args.fits:"# ── 2. collection list before per-scan loop ───────────────────────────────────src = src.replace(a1, b1, 1); ok += 1; print('P1 ok')assert a1 in src, f'P1 not found; snippet={repr(a1[:80])}'      "    return p.parse_args()")      "                   help='Stacking method for cross-scan combination (default: mean).')\n"      "    p.add_argument('--cross-scan-stack-method', choices=['mean', 'median'], default='mean',\n"      "                   help='Stack per-scan moon-track images into a single combined output (default: True).')\n"      "    p.add_argument('--cross-scan-stack', action=argparse.BooleanOptionalAction, default=True,\n"b1 = ("    p.add_argument('--export-fits', action='store_true', help='Export CASA .image products to FITS')\n"      "    return p.parse_args()")a1 = ("    p.add_argument('--export-fits', action='store_true', help='Export CASA .image products to FITS')\n"# ── 1. new CLI args after --export-fits ───────────────────────────────────────ok = 0src = p.read_text()p = Path(__file__).parent / 'casa_moon_imaging.py'from pathlib import Pathfrom pathlib import Path

src_path = Path(__file__).parent / 'casa_moon_imaging.py'
src = src_path.read_text()

# ── 1. Add two new CLI args ────────────────────────────────────────────────────
OLD1 = (
    "    p.add_argument('--keep-ms', action='store_true', "
    "help='Keep imported MeasurementSets after imaging')\n\n"
    "    return p.parse_args()"
)
NEW1 = (
    "    p.add_argument('--keep-ms', action='store_true', "
    "help='Keep imported MeasurementSets after imaging')\n"
    "    p.add_argument('--cross-scan-stack', action=argparse.BooleanOptionalAction, default=True,\n"
    "                   help='Stack per-scan moon-track images into a single combined output (default: True).')\n"
    "    p.add_argument('--cross-scan-stack-method', choices=['mean', 'median'], default='mean',\n"
    "                   help='Stacking method for cross-scan combination (default: mean).')\n\n"
    "    return p.parse_args()"
)
assert OLD1 in src, f"PATCH1 not found"
src = src.replace(OLD1, NEW1, 1)
print("PATCH1 applied")

# ── 2. Initialise collection list before the per-scan for-loop ────────────────
OLD2 = "    for fits_path_str in args.fits:"
NEW2 = (
    "    scan_stack_fits: list[Path] = []"
    "  # per-scan moontrack stacked FITS collected for cross-scan stack\n"
    "    for fits_path_str in args.fits:"
)
assert OLD2 in src, "PATCH2 not found"
src = src.replace(OLD2, NEW2, 1)
print("PATCH2 applied")

# ── 3. Collect per-scan stacked FITS after _image_moon_per_integration ─────────
OLD3 = """\
            )
        else:
            _run_tclean_with_progress("""
NEW3 = """\
            )
            # collect this scan's stacked image for cross-scan combination
            _candidate = Path(str(imagename) + f'_moontrack_{args.stack_method}.image.fits')
            if _candidate.exists():
                scan_stack_fits.append(_candidate)
            else:
                print(f'[CASA] warning: expected per-scan stack not found: {_candidate.name}')
        else:
            _run_tclean_with_progress("""
assert OLD3 in src, "PATCH3 not found"
src = src.replace(OLD3, NEW3, 1)
print("PATCH3 applied")

# ── 4. Call cross-scan stack after the per-scan loop ──────────────────────────
OLD4 = (
    "    print(f'[CASA] done. outputs in {outdir}')\n"
    "    return 0"
)
NEW4 = (
    "    # ── cross-scan stack ──────────────────────────────────────────────────\n"
    "    if args.moon_track_per_integration and args.cross_scan_stack and scan_stack_fits:\n"
    "        _cross_scan_stack(scan_stack_fits, outdir, args.cross_scan_stack_method, args.overwrite)\n\n"
    "    print(f'[CASA] done. outputs in {outdir}')\n"
    "    return 0"
)
assert OLD4 in src, "PATCH4 not found"
src = src.replace(OLD4, NEW4, 1)
print("PATCH4 applied")

# ── 5. Insert _cross_scan_stack() function before main() ──────────────────────
CROSS_SCAN_FN = '''\
def _cross_scan_stack(
    fits_paths: list,
    outdir,
    method: str = 'mean',
    overwrite: bool = False,
) -> None:
    """Stack per-scan moontrack FITS images across all input scans.

    Each image must share the same WCS, cell size and imsize — guaranteed when
    all scans were processed with identical imaging parameters by this script.

    Outputs written to outdir:
      moon_allscans_<method>.image.fits  — combined stack
      moon_allscans_rms.image.fits       — per-pixel RMS across scans
    """
    import numpy as np
    from astropy.io import fits as astrofits

    n = len(fits_paths)
    print(f'[CASA] cross-scan stack: combining {n} scan images (method={method})')

    cube = []
    ref_header = None
    for p in fits_paths:
        with astrofits.open(str(p)) as hdul:
            data = hdul[0].data.astype(np.float32)
            if ref_header is None:
                ref_header = hdul[0].header.copy()
        cube.append(data)

    arr = np.stack(cube, axis=0)  # shape (n_scans, stokes, freq, y, x)

    if method == 'median':
        combined = np.nanmedian(arr, axis=0)
    else:
        combined = np.nanmean(arr, axis=0)

    rms = np.sqrt(np.nanmean(arr ** 2, axis=0))

    outdir = Path(outdir)

    def _write(suffix: str, data: np.ndarray, label: str) -> None:
        out = outdir / f'moon_allscans_{suffix}.image.fits'
        if out.exists():
            if overwrite:
                out.unlink()
            else:
                print(
                    f'[CASA] cross-scan stack: skipping existing {out.name} '
                    f'(use --overwrite to replace)'
                )
                return
        hdr = ref_header.copy()
        hdr['HISTORY'] = f'Moon cross-scan stacked image ({label}), n_scans={n}'
        astrofits.writeto(str(out), data.astype(np.float32), hdr)
        print(f'[CASA] cross-scan stack written: {out.name}  (n_scans={n})')

    _write(method, combined, method)
    _write('rms', rms, 'rms')


'''

OLD5 = "def main() -> int:"
NEW5 = CROSS_SCAN_FN + "def main() -> int:"
assert OLD5 in src, "PATCH5 not found"
src = src.replace(OLD5, NEW5, 1)
print("PATCH5 applied")

src_path.write_text(src)
print(f"\nAll 5 patches written to {src_path}")
