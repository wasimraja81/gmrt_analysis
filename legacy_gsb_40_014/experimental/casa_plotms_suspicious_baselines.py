#!/usr/bin/env python
"""Create plotms diagnostics for suspicious Moon baselines.

Run with a CASA-enabled Python, or via:
  casa --nogui --nologger -c casa_plotms_suspicious_baselines.py --vis ...
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _import_casa_tasks():
    try:
        from casatasks import flagdata, split  # type: ignore
    except Exception as exc:
        raise SystemExit(
            'Could not import CASA tasks. Run with CASA-enabled Python or use '
            '`casa --nogui --nologger -c casa_plotms_suspicious_baselines.py ...`. '
            f'Original import error: {exc}'
        )

    plotms_task = None
    try:
        from casaplotms import plotms as plotms_task  # type: ignore
    except Exception:
        plotms_task = None
    if plotms_task is None:
        try:
            from casatasks import plotms as plotms_task  # type: ignore
        except Exception:
            plotms_task = globals().get('plotms', None)
    if plotms_task is None:
        raise SystemExit('Could not import CASA plotms task.')

    return split, flagdata, plotms_task


def _plot(plotms_task, *, vis: str, plotfile: Path, title: str, antenna: str = '', xaxis: str = 'time', yaxis: str = 'amp', showgui: bool = False) -> None:
    if plotfile.exists():
        plotfile.unlink()
    plotms_task(
        vis=vis,
        xaxis=xaxis,
        yaxis=yaxis,
        antenna=antenna,
        avgchannel='4096',
        avgtime='8',
        correlation='RR,LL',
        coloraxis='baseline',
        plotrange=[],
        showgui=showgui,
        showlegend=True,
        customsymbol=True,
        symbolshape='circle',
        symbolsize=4,
        title=title,
        xlabel=xaxis,
        ylabel=yaxis,
        clearplots=True,
        plotfile=str(plotfile),
        expformat='png',
        exprange='current',
        overwrite=True,
    )
    print(f'[plotms] wrote {plotfile}')


def main() -> int:
    p = argparse.ArgumentParser(description='plotms diagnostics for suspicious Moon baselines')
    p.add_argument('--vis', required=True, help='Input MS path')
    p.add_argument('--outdir', default='./diagnostics_out/moon_plotms', help='Output directory for PNGs and optional flagged MS copy')
    p.add_argument('--suspicious-baselines', default='29&30;1&25;18&30;15&27', help='CASA antenna baseline selection string')
    p.add_argument('--make-excluded-copy', action='store_true', help='Make a copy of the MS and flag suspicious baselines there')
    p.add_argument('--excluded-vis', default=None, help='Path for flagged/excluded MS copy (default: <outdir>/<visstem>_suspects_flagged.ms)')
    p.add_argument('--showgui', action='store_true', help='Open plotms GUI in addition to writing PNGs')
    args = p.parse_args()

    split, flagdata, plotms_task = _import_casa_tasks()

    vis = Path(args.vis).expanduser().resolve()
    if not vis.exists():
        raise SystemExit(f'Input MS not found: {vis}')

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    suspects = str(args.suspicious_baselines).strip()
    visstem = vis.name[:-3] if vis.name.endswith('.ms') else vis.name

    print(f'[plotms] vis={vis}')
    print(f'[plotms] suspicious baselines={suspects}')

    _plot(
        plotms_task,
        vis=str(vis),
        plotfile=outdir / f'{visstem}_all_time_amp.png',
        title='All baselines: amplitude vs time',
        xaxis='time',
        yaxis='amp',
        showgui=bool(args.showgui),
    )
    _plot(
        plotms_task,
        vis=str(vis),
        plotfile=outdir / f'{visstem}_all_uvdist_amp.png',
        title='All baselines: amplitude vs uvdist',
        xaxis='uvdist',
        yaxis='amp',
        showgui=False,
    )
    _plot(
        plotms_task,
        vis=str(vis),
        antenna=suspects,
        plotfile=outdir / f'{visstem}_suspects_time_amp.png',
        title=f'Suspicious baselines only: {suspects}',
        xaxis='time',
        yaxis='amp',
        showgui=False,
    )
    _plot(
        plotms_task,
        vis=str(vis),
        antenna=suspects,
        plotfile=outdir / f'{visstem}_suspects_uvdist_amp.png',
        title=f'Suspicious baselines only: {suspects}',
        xaxis='uvdist',
        yaxis='amp',
        showgui=False,
    )

    if args.make_excluded_copy:
        excluded_vis = Path(args.excluded_vis).expanduser().resolve() if args.excluded_vis else (outdir / f'{visstem}_suspects_flagged.ms')
        if excluded_vis.exists():
            print(f'[plotms] removing existing excluded copy: {excluded_vis}')
            import shutil
            shutil.rmtree(excluded_vis)
        print(f'[plotms] creating excluded copy: {excluded_vis}')
        split(vis=str(vis), outputvis=str(excluded_vis), datacolumn='all')
        print(f'[plotms] flagging suspicious baselines in copy: {suspects}')
        flagdata(vis=str(excluded_vis), mode='manual', antenna=suspects, flagbackup=False)

        _plot(
            plotms_task,
            vis=str(excluded_vis),
            plotfile=outdir / f'{visstem}_excluded_time_amp.png',
            title='Suspicious baselines flagged/excluded: amplitude vs time',
            xaxis='time',
            yaxis='amp',
            showgui=False,
        )
        _plot(
            plotms_task,
            vis=str(excluded_vis),
            plotfile=outdir / f'{visstem}_excluded_uvdist_amp.png',
            title='Suspicious baselines flagged/excluded: amplitude vs uvdist',
            xaxis='uvdist',
            yaxis='amp',
            showgui=False,
        )

    print(f'[plotms] done. outputs in {outdir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
