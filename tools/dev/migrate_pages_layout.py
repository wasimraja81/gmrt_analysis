import pathlib, shutil

pages_dir = pathlib.Path('/Users/raj030/github-wasimraja81/gmrt_analysis_gh_pages')
run_dirs = [
    pages_dir / '40_014/runs/20260511_174446',
    pages_dir / '40_014/latest',
]

DIR_RENAMES = [
    ('diagnostics_out/3c48_primary_calibration',         'diagnostics_out/primary/3c48/solve'),
    ('diagnostics_out/3c48_split_calibrated',            'diagnostics_out/primary/3c48/selfcheck'),
    ('diagnostics_out/3c468.1_primary_split_calibrated', 'diagnostics_out/secondary/3c468.1/transfer'),
    ('secondary_calibration/clustering',                 'diagnostics_out/secondary/3c468.1/flagging'),
    ('diagnostics_out/3c468.1_split_calibrated',         'diagnostics_out/secondary/3c468.1/final_qa'),
]

HTML_REPLACEMENTS = [
    ('diagnostics_out/3c48_primary_calibration/',         'diagnostics_out/primary/3c48/solve/'),
    ('diagnostics_out/3c48_split_calibrated/',            'diagnostics_out/primary/3c48/selfcheck/'),
    ('diagnostics_out/3c468.1_primary_split_calibrated/', 'diagnostics_out/secondary/3c468.1/transfer/'),
    ('secondary_calibration/clustering/',                 'diagnostics_out/secondary/3c468.1/flagging/'),
    ('diagnostics_out/3c468.1_split_calibrated/',         'diagnostics_out/secondary/3c468.1/final_qa/'),
]

for run_dir in run_dirs:
    print(f'\n=== {run_dir.name} ===')
    for old_rel, new_rel in DIR_RENAMES:
        src = run_dir / old_rel
        dst = run_dir / new_rel
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            print(f'  mv {old_rel} -> {new_rel}')
        else:
            print(f'  skip: {old_rel}')

    sc = run_dir / 'secondary_calibration'
    if sc.exists():
        remaining = [p for p in sc.rglob('*') if p.is_file()]
        if not remaining:
            shutil.rmtree(str(sc))
            print('  removed empty secondary_calibration/')
        else:
            print(f'  WARNING: secondary_calibration/ still has: {[str(p.relative_to(run_dir)) for p in remaining]}')

    html_path = run_dir / 'index.html'
    if html_path.exists():
        text = html_path.read_text()
        for old, new in HTML_REPLACEMENTS:
            text = text.replace(old, new)
        html_path.write_text(text)
        print('  patched index.html')
