#!/usr/bin/env python3
"""
Generate a frame-by-frame movie showing UV coverage evolution over time.

Each frame shows the instantaneous UV plane at a single integration, allowing
visualization of how baseline coverage changes as Earth rotates during observation.

Usage:
    python uv_coverage_movie.py --fits <uvfits_file> --source <SOURCE> \
        --output <output.mp4> [--framerate 5] [--dpi 100] [--fps-limit 5]

Example:
    python uv_coverage_movie.py \
        --fits ~/DATA/gmrt_40_014/work/split/moon/moon0520_primary_secondary_calibrated_flagged.uvfits \
        --source MOON0520 \
        --output uv_coverage_movie.mp4
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import Normalize

# Import from project modules
try:
    from modules import ugmrt_query as q
except ImportError:
    print("ERROR: Could not import modules. Make sure you're running from the src/ directory", file=sys.stderr)
    sys.exit(1)


def create_uv_movie(
    fits_path,
    source_name,
    output_mp4,
    index_cache=None,
    moon_diameter_deg=None,
    uvdist_max=350,
    framerate=5,
    dpi=100,
):
    """
    Create a frame-by-frame UV coverage movie.

    Parameters
    ----------
    fits_path : str or Path
        Path to input UVFITS file
    source_name : str
        Source name to extract
    output_mp4 : str or Path
        Output MP4 filename
    index_cache : str or Path, optional
        Path to cached row index. If None, will be created.
    moon_diameter_deg : float, optional
        Moon/target angular diameter in degrees. If provided, shows critical UV circle.
    uvdist_max : float
        Maximum UV distance to plot (in wavelengths)
    framerate : int
        Framerate for output movie
    dpi : int
        DPI for frame rendering
    """

    fits_path = Path(fits_path)
    output_mp4 = Path(output_mp4)

    if not fits_path.exists():
        raise FileNotFoundError(f"FITS file not found: {fits_path}")

    print(f"[uv_coverage_movie] Loading {source_name} from {fits_path.name}...")

    # Build index
    cache_path = Path(index_cache) if index_cache else fits_path.with_suffix('.uvfits.row_index_cache.npz')
    index = q.get_or_build_row_index(
        str(fits_path),
        cache_path=str(cache_path),
        force_rebuild=False,
    )

    # Load visibility data
    vis = q.load_vis_for_source(index, source=source_name, stokes=('RR', 'LL'))

    uu_sec = vis['uu_sec']
    vv_sec = vis['vv_sec']
    freqs_hz = vis['freqs_hz']
    flagged = vis['flagged']
    jd = vis['jd']

    freqs_mhz = freqs_hz / 1e6
    unique_times = np.unique(jd)
    n_integrations = len(unique_times)

    print(f"[uv_coverage_movie] {n_integrations} integrations, {len(freqs_hz)} channels")

    # Create frame directory
    frame_dir = output_mp4.parent / f"{output_mp4.stem}_frames"
    frame_dir.mkdir(exist_ok=True)
    print(f"[uv_coverage_movie] Writing frames to {frame_dir}")

    # UV limit circle (if moon diameter provided)
    uv_circle = None
    if moon_diameter_deg is not None:
        moon_diameter_rad = np.radians(moon_diameter_deg)
        uv_circle = 1.0 / moon_diameter_rad

    # Normalize colors across all frames
    norm = Normalize(vmin=freqs_mhz.min(), vmax=freqs_mhz.max())

    # Create each frame
    for frame_idx, current_time in enumerate(unique_times):
        if (frame_idx + 1) % max(1, n_integrations // 10) == 0:
            print(f"  Frame {frame_idx + 1}/{n_integrations}...")

        # Extract data for this integration only
        time_mask = (jd == current_time)

        u_frame = []
        v_frame = []
        freq_frame = []

        for chan_idx in range(len(freqs_hz)):
            freq = freqs_hz[chan_idx]
            u_lambda = uu_sec * freq
            v_lambda = vv_sec * freq
            uvdist = np.sqrt(u_lambda**2 + v_lambda**2)

            unflagged_mask = ~(flagged[:, chan_idx, 0] | flagged[:, chan_idx, 1])
            mask = (uvdist >= 0) & (uvdist <= uvdist_max) & unflagged_mask & time_mask

            u_chan = u_lambda[mask]
            v_chan = v_lambda[mask]

            if len(u_chan) > 0:
                u_frame.extend(u_chan)
                v_frame.extend(v_chan)
                freq_frame.extend([freq / 1e6] * len(u_chan))

        if len(u_frame) == 0:
            continue

        u_frame = np.array(u_frame)
        v_frame = np.array(v_frame)
        freq_frame = np.array(freq_frame)

        # Create figure
        fig, ax = plt.subplots(figsize=(15, 13), dpi=dpi)

        scatter = ax.scatter(u_frame, v_frame, s=2, alpha=0.5, c=freq_frame,
                            cmap='viridis', norm=norm)
        ax.scatter(-u_frame, -v_frame, s=2, alpha=0.5, c=freq_frame,
                  cmap='viridis', norm=norm)

        # Add UV circle if moon diameter provided
        if uv_circle is not None:
            circle = Circle((0, 0), uv_circle, fill=False, edgecolor='red',
                           linewidth=3.5, linestyle='--')
            ax.add_patch(circle)

        ax.set_xlabel('U (wavelengths)', fontsize=13, fontweight='bold')
        ax.set_ylabel('V (wavelengths)', fontsize=13, fontweight='bold')

        title = f'UV Coverage: {source_name}\n'
        if freqs_mhz.size > 0:
            title += f'{freqs_mhz.mean():.1f} MHz ({len(freqs_hz)} channels), '
        title += f'{n_integrations} integrations'
        if moon_diameter_deg is not None:
            title += f'\nTarget diameter: {moon_diameter_deg}°'

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal')

        cbar = plt.colorbar(scatter, ax=ax, label='Frequency (MHz)', format='%.1f')

        # Info text
        textstr = f'Integration {frame_idx + 1}/{n_integrations} ({len(u_frame)} visibilities)'
        if uv_circle is not None:
            textstr += f'\nUV circle: {uv_circle:.1f}λ'

        props = dict(boxstyle='round', facecolor='lightcyan', alpha=0.85)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=props, family='monospace')

        ax.set_xlim(-uvdist_max, uvdist_max)
        ax.set_ylim(-uvdist_max, uvdist_max)

        # Save frame
        frame_file = frame_dir / f'frame_{frame_idx:03d}.png'
        fig.savefig(frame_file, bbox_inches='tight')
        plt.close(fig)

    # Create movie with ffmpeg
    print(f"[uv_coverage_movie] Creating MP4 with ffmpeg...")
    import subprocess

    cmd = [
        'ffmpeg',
        '-framerate', str(framerate),
        '-i', str(frame_dir / 'frame_%03d.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-y',  # Overwrite output file
        str(output_mp4),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("ERROR: ffmpeg failed", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)

    print(f"✅ Movie created: {output_mp4}")
    print(f"   Frames: {frame_dir}")
    print(f"   Duration: {n_integrations / framerate:.1f} seconds at {framerate} fps")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--fits', required=True,
        help='Input UVFITS file'
    )
    parser.add_argument(
        '--source', required=True,
        help='Source name to extract'
    )
    parser.add_argument(
        '--output', '-o', required=True,
        help='Output MP4 filename'
    )
    parser.add_argument(
        '--index-cache',
        help='Path to row index cache (default: auto-generated from FITS filename)'
    )
    parser.add_argument(
        '--moon-diameter', type=float, default=None,
        help='Moon/target angular diameter in degrees (shows as red circle in UV plane)'
    )
    parser.add_argument(
        '--uvdist-max', type=float, default=350,
        help='Maximum UV distance to plot in wavelengths (default: 350)'
    )
    parser.add_argument(
        '--framerate', type=int, default=5,
        help='Framerate for output movie (default: 5 fps)'
    )
    parser.add_argument(
        '--dpi', type=int, default=100,
        help='DPI for frame rendering (default: 100)'
    )

    args = parser.parse_args()

    try:
        create_uv_movie(
            fits_path=args.fits,
            source_name=args.source,
            output_mp4=args.output,
            index_cache=args.index_cache,
            moon_diameter_deg=args.moon_diameter,
            uvdist_max=args.uvdist_max,
            framerate=args.framerate,
            dpi=args.dpi,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
