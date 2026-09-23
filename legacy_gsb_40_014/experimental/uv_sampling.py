#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import ugmrt_query as q
from plotVis import _get_product_data, _parse_csv_list, _sample_mask


def _sanitize_for_filename(text: str) -> str:
	out = ''.join(ch if ch.isalnum() or ch in '._-' else '_' for ch in str(text).strip().lower())
	out = out.strip('_')
	return out or 'source'


def _slice_rows(vis: dict, row_keep: np.ndarray) -> dict:
	nrows = len(vis['jd'])
	out = {}
	for key, value in vis.items():
		if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == nrows:
			out[key] = value[row_keep]
		else:
			out[key] = value
	return out


def _guess_source_name(fits_path: Path, available_sources: list[str]) -> str:
	guessed = fits_path.name.split('_', 1)[0].upper()
	if guessed in available_sources:
		return guessed
	if len(available_sources) == 1:
		return available_sources[0]
	raise ValueError(f'Could not resolve source for {fits_path.name}. Available: {available_sources}')


def _build_product_colors(products: list[str]) -> dict[str, str]:
	color_cycle = plt.rcParams.get('axes.prop_cycle', None)
	cycle_colors = color_cycle.by_key().get('color', []) if color_cycle is not None else []
	if not cycle_colors:
		cycle_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
	return {prod: cycle_colors[i % len(cycle_colors)] for i, prod in enumerate(products)}



def _source_group_tag(source_names: list[str]) -> str:
	uniq = sorted({_sanitize_for_filename(str(name)) for name in source_names if str(name).strip()})
	if not uniq:
		return 'sources'
	if len(uniq) == 1:
		return uniq[0]
	# If all names share a common alphabetic prefix (e.g. "moon0520", "moon0545"
	# are all UVFITS source-name entries for successive Moon scans), collapse to
	# "{prefix}_{N}scans" rather than the verbose "moon0520_moon0545_plus3" form.
	prefixes = [re.match(r'^([a-zA-Z]+)', s) for s in uniq]
	if all(m for m in prefixes):  # every name must start with letters
		common_prefixes = {m.group(1).lower() for m in prefixes}
		if len(common_prefixes) == 1:
			prefix = common_prefixes.pop()
			return f'{prefix}_{len(uniq)}scans'
	if len(uniq) <= 3:
		return '_'.join(uniq)
	return '_'.join(uniq[:3]) + f'_plus{len(uniq) - 3}'



def _jd_to_utc_label(jd_value: float) -> str:
	if not np.isfinite(jd_value):
		return 'UTC: n/a'
	unix_seconds = (float(jd_value) - 2440587.5) * 86400.0
	dt = datetime.fromtimestamp(unix_seconds, tz=timezone.utc)
	return dt.strftime('UTC %Y-%m-%d %H:%M:%S')


def _movie_duration_seconds(movie_path: Path) -> float:
        cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(movie_path),
        ]
        out = subprocess.check_output(cmd, text=True).strip()
        dur = float(out)
        if not np.isfinite(dur) or dur <= 0:
                raise ValueError(f'Invalid duration for {movie_path}: {out}')
        return dur


def _render_uv_sampling(
	ax,
	vis: dict,
	products: list[str],
	sample_frac: float,
	overlay_flags: bool,
	seed: int,
	title: str,
	show_stats: bool,
	product_colors: dict[str, str],
	uv_good_marker_size: float,
	uv_flag_marker_size: float,
	uv_lim_override: float | None = None,
) -> tuple[int, int]:
	product_payload = []
	for prod in products:
		z, f = _get_product_data(prod, vis, None, None)
		product_payload.append((str(prod), z, f))

	if not product_payload:
		raise ValueError('No products available for uv-sampling render.')

	uu_sec = np.asarray(vis['uu_sec'], dtype=np.float64)
	vv_sec = np.asarray(vis['vv_sec'], dtype=np.float64)
	freqs_hz = np.asarray(vis['freqs_hz'], dtype=np.float64)
	uu_kl_cell = (uu_sec[:, np.newaxis] * freqs_hz[np.newaxis, :]) / 1e3
	vv_kl_cell = (vv_sec[:, np.newaxis] * freqs_hz[np.newaxis, :]) / 1e3

	z0 = product_payload[0][1]
	shared_sample_mask = _sample_mask(z0.shape, sample_frac, seed=seed)

	good_cell_masks = []
	flag_cell_masks = []
	flagged_by_prod = {}

	for prod, z, f in product_payload:
		finite = np.isfinite(z.real) & np.isfinite(z.imag)
		good = (~f) & finite
		bad = f

		use = good & shared_sample_mask
		bad_s = bad & shared_sample_mask

		good_cell_masks.append(use)
		flag_cell_masks.append(bad_s)

		n_sampled_cells = int(np.count_nonzero(shared_sample_mask))
		n_flag_sampled = int(np.count_nonzero(bad_s))
		flagged_by_prod[prod] = (n_flag_sampled, n_sampled_cells)

	good_stack = np.stack(good_cell_masks, axis=0)
	flag_stack = np.stack(flag_cell_masks, axis=0)

	good_counts = np.sum(good_stack, axis=0)
	flag_counts = np.sum(flag_stack, axis=0)

	sampled_good_cells = good_counts > 0
	shared_good_cells = good_counts > 1
	sampled_flag_cells = flag_counts > 0
	shared_flag_cells = flag_counts > 1

	same_good = len(good_cell_masks) <= 1 or all(np.array_equal(good_cell_masks[0], m) for m in good_cell_masks[1:])
	same_flag = len(flag_cell_masks) <= 1 or all(np.array_equal(flag_cell_masks[0], m) for m in flag_cell_masks[1:])

	legend_handles = []
	prod_labels = [p for p, _, _ in product_payload]

	if same_good:
		ug = uu_kl_cell[sampled_good_cells]
		vg = vv_kl_cell[sampled_good_cells]
		ax.scatter(ug, vg, s=uv_good_marker_size, alpha=0.52, c='tab:blue', linewidths=0)
		ax.scatter(-ug, -vg, s=uv_good_marker_size, alpha=0.28, c='tab:blue', linewidths=0)
	else:
		for prod, cell_mask in zip(prod_labels, good_cell_masks):
			unique_cells = cell_mask & (good_counts == 1)
			if np.any(unique_cells):
				color = product_colors.get(prod, None)
				u = uu_kl_cell[unique_cells]
				v = vv_kl_cell[unique_cells]
				ax.scatter(u, v, s=uv_good_marker_size, alpha=0.62, c=color, linewidths=0)
				ax.scatter(-u, -v, s=uv_good_marker_size, alpha=0.34, c=color, linewidths=0)
				legend_handles.append(
					Line2D([0], [0], marker='o', linestyle='None', markersize=4,
						   markerfacecolor=color, markeredgecolor=color, label=f'{prod} unique')
				)

		if np.any(shared_good_cells):
			us = uu_kl_cell[shared_good_cells]
			vs = vv_kl_cell[shared_good_cells]
			ax.scatter(us, vs, s=uv_good_marker_size, alpha=0.42, c='0.35', linewidths=0)
			ax.scatter(-us, -vs, s=uv_good_marker_size, alpha=0.22, c='0.35', linewidths=0)
			legend_handles.append(
				Line2D([0], [0], marker='o', linestyle='None', markersize=4,
					   markerfacecolor='0.35', markeredgecolor='0.35', label='shared (multi-product)')
			)

	if overlay_flags:
		if same_flag:
			if np.any(sampled_flag_cells):
				uf = uu_kl_cell[sampled_flag_cells]
				vf = vv_kl_cell[sampled_flag_cells]
				ax.scatter(uf, vf, s=uv_flag_marker_size, alpha=0.34, c='red', linewidths=0)
				ax.scatter(-uf, -vf, s=uv_flag_marker_size, alpha=0.18, c='red', linewidths=0)
		else:
			for prod, cell_mask in zip(prod_labels, flag_cell_masks):
				unique_cells = cell_mask & (flag_counts == 1)
				if np.any(unique_cells):
					color = product_colors.get(prod, None)
					uf = uu_kl_cell[unique_cells]
					vf = vv_kl_cell[unique_cells]
					ax.scatter(uf, vf, s=uv_flag_marker_size, alpha=0.26, c=color, linewidths=0)
					ax.scatter(-uf, -vf, s=uv_flag_marker_size, alpha=0.14, c=color, linewidths=0)
			if np.any(shared_flag_cells):
				uf = uu_kl_cell[shared_flag_cells]
				vf = vv_kl_cell[shared_flag_cells]
				ax.scatter(uf, vf, s=uv_flag_marker_size, alpha=0.34, c='red', linewidths=0)
				ax.scatter(-uf, -vf, s=uv_flag_marker_size, alpha=0.18, c='red', linewidths=0)

	sampled_points = int(np.count_nonzero(sampled_good_cells))
	total_points = int(good_counts.size)

	if show_stats:
		stats_lines = [f'Plotting {sampled_points:,} out of {total_points:,} sampled uv-points']
		stats_lines.extend([
			f'% flagged_{p}={((100.0 * nf / nt) if nt > 0 else 0.0):.2f}%'
			for p, (nf, nt) in flagged_by_prod.items()
		])
		ax.text(
			0.01,
			0.99,
			'\n'.join(stats_lines),
			transform=ax.transAxes,
			fontsize=7,
			va='top',
			ha='left',
			linespacing=1.3,
			bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7),
		)

	if legend_handles:
		ax.legend(handles=legend_handles, loc='upper right', fontsize=7, frameon=True)

	if uv_lim_override is not None and np.isfinite(float(uv_lim_override)) and float(uv_lim_override) > 0:
		uv_lim = float(uv_lim_override)
	else:
		uv_for_limits = []
		if np.any(sampled_good_cells):
			uv_for_limits.append(np.abs(uu_kl_cell[sampled_good_cells]))
			uv_for_limits.append(np.abs(vv_kl_cell[sampled_good_cells]))
		if np.any(sampled_flag_cells):
			uv_for_limits.append(np.abs(uu_kl_cell[sampled_flag_cells]))
			uv_for_limits.append(np.abs(vv_kl_cell[sampled_flag_cells]))
		if not uv_for_limits:
			uv_for_limits = [np.abs(uu_kl_cell), np.abs(vv_kl_cell)]

		uv_max = max(float(np.nanmax(arr)) for arr in uv_for_limits if arr.size)
		if not np.isfinite(uv_max) or uv_max <= 0:
			uv_max = 1.0
		uv_lim = 1.02 * uv_max

	ax.set_xlabel('u (kλ)')
	ax.set_ylabel('v (kλ)')
	ax.set_aspect('equal', adjustable='box')
	ax.set_xlim(-uv_lim, uv_lim)
	ax.set_ylim(-uv_lim, uv_lim)
	ax.grid(True, alpha=0.25)
	ax.set_title(title)

	return sampled_points, total_points


def _movie_from_frames(frames_dir: Path, movie_path: Path, fps: int) -> None:
	pattern = str(frames_dir / 'uvsnap_%04d.png')
	cmd = [
		'ffmpeg',
		'-y',
		'-framerate',
		str(int(fps)),
		'-i',
		pattern,
		'-vf',
		'scale=trunc(iw/2)*2:trunc(ih/2)*2',
		'-c:v',
		'prores_ks',
		'-profile:v',
		'3',
		'-pix_fmt',
		'yuv422p10le',
		str(movie_path),
	]
	subprocess.run(cmd, check=True)


def _load_vis(fits_path: Path, source: str | None, chan_range: tuple[int, int] | None, index_cache: Path | None = None) -> tuple[str, dict]:
	index = q.get_or_build_row_index(
		str(fits_path),
		cache_path=str(index_cache) if index_cache else None,
		force_rebuild=False,
		validation_mode='fast',
		write_cache=True,
		override_dud_names=None,
	)

	id_to_name = index.get('id_to_name', {})
	available_sources = sorted(id_to_name.values()) if id_to_name else []
	if not available_sources:
		raise ValueError(f'No source found in index for {fits_path}')

	resolved_source = source
	if not resolved_source:
		resolved_source = _guess_source_name(fits_path, available_sources)
	elif resolved_source not in available_sources:
		if len(available_sources) == 1:
			print(f'[uv] source "{resolved_source}" not found; using "{available_sources[0]}"')
			resolved_source = available_sources[0]
		else:
			raise ValueError(f'Source "{resolved_source}" not found. Available: {available_sources}')

	vis = q.load_vis_for_source(
		index,
		source=resolved_source,
		stokes=('RR', 'LL'),
		chan_range=chan_range,
		flag_all_corrs_if_any_rawvis_flagged=True,
	)
	return resolved_source, vis


def _global_uv_lim(vis: dict) -> float:
	freqs_hz = np.asarray(vis['freqs_hz'], dtype=np.float64)
	uu_kl = np.asarray(vis['uu_sec'], dtype=np.float64)[:, np.newaxis] * freqs_hz[np.newaxis, :] / 1e3
	vv_kl = np.asarray(vis['vv_sec'], dtype=np.float64)[:, np.newaxis] * freqs_hz[np.newaxis, :] / 1e3
	uv_max = max(float(np.nanmax(np.abs(uu_kl))), float(np.nanmax(np.abs(vv_kl))))
	if not np.isfinite(uv_max) or uv_max <= 0:
		uv_max = 1.0
	return 1.02 * uv_max


def _generate_single(
	fits_path: Path,
	outdir: Path,
	products: list[str],
	sample_frac: float,
	overlay_flags: bool,
	seed: int,
	time_step: int,
	chan_range: tuple[int, int] | None,
	fps: int,
	uv_good_marker_size: float,
	uv_flag_marker_size: float,
	fixed_limits: bool,
	skip_movie: bool,
	source: str | None = None,
	index_cache: Path | None = None,
	whole_name: str = 'uv_sampling_whole.png',
	frames_dirname: str = 'snapshots',
	movie_name: str = 'uv_sampling_movie_prores.mov',
) -> tuple[str, dict, Path, Path, Path]:
	resolved_source, vis = _load_vis(fits_path, source=source, chan_range=chan_range, index_cache=index_cache)
	product_colors = _build_product_colors(products)

	src_tag = _sanitize_for_filename(resolved_source)
	source_outdir = outdir / f'{src_tag}_uv_sampling'
	source_outdir.mkdir(parents=True, exist_ok=True)
	whole_path = source_outdir / f'{src_tag}_{whole_name}'
	frames_dir = source_outdir / f'{src_tag}_{frames_dirname}'
	movie_path = source_outdir / f'{src_tag}_{movie_name}'
	frames_dir.mkdir(parents=True, exist_ok=True)

	uv_lim = _global_uv_lim(vis) if fixed_limits else None

	fig, ax = plt.subplots(figsize=(8, 8), dpi=180)
	sampled_points, total_points = _render_uv_sampling(
		ax=ax,
		vis=vis,
		products=products,
		sample_frac=sample_frac,
		overlay_flags=overlay_flags,
		seed=seed,
		title='uv sampling (sampled cells) - whole scan',
		show_stats=True,
		product_colors=product_colors,
		uv_good_marker_size=uv_good_marker_size,
		uv_flag_marker_size=uv_flag_marker_size,
		uv_lim_override=uv_lim,
	)
	fig.tight_layout()
	fig.savefig(whole_path, dpi=180, bbox_inches='tight')
	plt.close(fig)
	print(f'[uv] whole scan saved: {whole_path}')
	print(f'[uv] whole scan stats: plotted {sampled_points:,} / {total_points:,} sampled uv-points')

	jd = np.asarray(vis['jd'], dtype=np.float64)
	unique_jd = np.unique(jd)

	# Split unique timestamps into contiguous scan groups (gap > 5x median dt)
	if len(unique_jd) > 1:
		diffs = np.diff(unique_jd)
		median_dt = np.median(diffs)
		gap_mask = diffs > 5.0 * median_dt
		split_points = np.where(gap_mask)[0] + 1
		scan_groups = np.split(unique_jd, split_points)
	else:
		scan_groups = [unique_jd]

	multi_scan = len(scan_groups) > 1
	print(f'[uv] detected {len(scan_groups)} scan group(s)')

	movie_paths = []
	for scan_idx, scan_jd in enumerate(scan_groups, start=1):
		scan_tag = f'scan{scan_idx:02d}_' if multi_scan else ''
		scan_frames_dir = source_outdir / f'{src_tag}_{scan_tag}{frames_dirname}'
		scan_movie_path = source_outdir / f'{src_tag}_{scan_tag}{movie_name}'
		scan_frames_dir.mkdir(parents=True, exist_ok=True)

		selected_jd = scan_jd[::max(1, int(time_step))]
		span_min = (scan_jd[-1] - scan_jd[0]) * 1440.0
		print(f'[uv] scan {scan_idx}: {len(selected_jd)} frames, {span_min:.1f} min')

		for idx, jd_val in enumerate(selected_jd, start=1):
			row_keep = jd == jd_val
			vis_t = _slice_rows(vis, row_keep)
			frame_jd = float(np.nanmean(np.asarray(vis_t['jd'], dtype=np.float64)))

			fig, ax = plt.subplots(figsize=(8, 8), dpi=180)
			_render_uv_sampling(
				ax=ax,
				vis=vis_t,
				products=products,
				sample_frac=sample_frac,
				overlay_flags=overlay_flags,
				seed=seed,
				title=f'uv snapshot {idx:04d}' + (f' (scan {scan_idx})' if multi_scan else ''),
				show_stats=False,
				product_colors=product_colors,
				uv_good_marker_size=uv_good_marker_size,
				uv_flag_marker_size=uv_flag_marker_size,
				uv_lim_override=uv_lim,
			)
			ax.text(
				0.99,
				0.99,
				_jd_to_utc_label(frame_jd),
				transform=ax.transAxes,
				fontsize=8,
				va='top',
				ha='right',
				bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7),
			)
			fig.tight_layout()
			frame_path = scan_frames_dir / f'uvsnap_{idx:04d}.png'
			fig.savefig(frame_path, dpi=180, bbox_inches='tight')
			plt.close(fig)

			if idx % 10 == 0 or idx == len(selected_jd):
				print(f'[uv] scan {scan_idx} frames: {idx}/{len(selected_jd)}')

		print(f'[uv] frame set saved: {scan_frames_dir}')

		if not skip_movie:
			_movie_from_frames(frames_dir=scan_frames_dir, movie_path=scan_movie_path, fps=fps)
			print(f'[uv] movie saved: {scan_movie_path}')
			movie_paths.append(scan_movie_path)
		else:
			print('[uv] movie generation skipped (--skip-movie)')

	final_movie_path = movie_paths[0] if movie_paths else movie_path
	return resolved_source, vis, whole_path, frames_dir, final_movie_path


def _concat_vis_rowwise(vis_list: list[dict]) -> dict:
	if not vis_list:
		raise ValueError('No visibilities to combine.')

	ref = vis_list[0]
	ref_nrows = len(ref['jd'])
	out = {}
	for key, first in ref.items():
		if isinstance(first, np.ndarray) and first.ndim >= 1 and first.shape[0] == ref_nrows:
			out[key] = np.concatenate([vis[key] for vis in vis_list], axis=0)
		else:
			out[key] = first
	return out


def _generate_combined_plot(
        fits_paths: list[Path],
        outdir: Path,
        products: list[str],
        sample_frac: float,
        overlay_flags: bool,
        seed: int,
        chan_range: tuple[int, int] | None,
        uv_good_marker_size: float,
        uv_flag_marker_size: float,
        out_name: str | None = None,
        title: str | None = None,
) -> Path:
        loaded = []
        source_names = []
        for fits_path in fits_paths:
                source, vis = _load_vis(
                        fits_path,
                        source=None,
                        chan_range=chan_range,
                        index_cache=fits_path.with_name(f'{fits_path.name}.row_index_cache.npz'),
                )
                loaded.append(vis)
                source_names.append(source)
                print(f'[uv-combined] loaded {fits_path.name} as {source}: {len(vis["jd"]):,} rows')

        combined_vis = _concat_vis_rowwise(loaded)
        uv_lim = _global_uv_lim(combined_vis)
        product_colors = _build_product_colors(products)

        combined_tag = _source_group_tag(source_names)
        combined_outdir = outdir / f'{combined_tag}_uv_sampling'
        combined_outdir.mkdir(parents=True, exist_ok=True)

        resolved_out_name = out_name or f'{combined_tag}_combined_uv_sampling.png'
        resolved_title = title or f'Combined uv sampling across {len(source_names)} file(s)'

        out_path = combined_outdir / resolved_out_name
        fig, ax = plt.subplots(figsize=(8, 8), dpi=180)
        sampled_points, total_points = _render_uv_sampling(
                ax=ax,
                vis=combined_vis,
                products=products,
                sample_frac=sample_frac,
                overlay_flags=overlay_flags,
                seed=seed,
                title=resolved_title,
                show_stats=True,
                product_colors=product_colors,
                uv_good_marker_size=uv_good_marker_size,
                uv_flag_marker_size=uv_flag_marker_size,
                uv_lim_override=uv_lim,
        )
        fig.tight_layout()
        fig.savefig(out_path, dpi=180, bbox_inches='tight')
        plt.close(fig)

        print(f'[uv-combined] combined sources: {", ".join(source_names)}')
        print(f'[uv-combined] combined plot saved: {out_path}')
        print(f'[uv-combined] combined stats: plotted {sampled_points:,} / {total_points:,} sampled uv-points')
        return out_path


def _generate_montage(outdir: Path, pattern: str, output_name: str, tile_width: int, fps: int, source_tag: str = 'sources', source_filters: list[str] | None = None) -> Path:
        movie_files = sorted(path for path in outdir.glob(pattern) if path.is_file())
        if source_filters:
                allowed = {_sanitize_for_filename(str(s)) for s in source_filters if str(s).strip()}
                movie_files = [
                        p for p in movie_files
                        if _sanitize_for_filename(p.parent.name.removesuffix('_uv_sampling')) in allowed
                ]
        if not movie_files:
                raise ValueError(f'No movie files matched after filtering: {outdir / pattern}')
        print(f'[uv-montage] using {len(movie_files)} movie file(s)')
        for movie_file in movie_files:
                print(f'  - {movie_file.name}')

        target_duration = max(_movie_duration_seconds(path) for path in movie_files)
        ffmpeg_cmd = ['ffmpeg', '-y']
        filter_parts = []

        for index, movie_file in enumerate(movie_files):
                ffmpeg_cmd.extend(['-stream_loop', '-1', '-i', str(movie_file)])
                tag = movie_file.name.removesuffix('_uv_sampling_movie_prores.mov')
                filter_parts.append(
                        f'[{index}:v]setpts=PTS-STARTPTS,fps={fps},scale={tile_width}:-2,'
                        f'trim=duration={target_duration:.6f},'
                        f"drawtext=text='{tag}':x=20:y=20:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=8[v{index}]"
                )

        ncols = 3
        n_movies = len(movie_files)
        nrows = (n_movies + ncols - 1) // ncols
        n_cells = ncols * nrows
        n_blanks = n_cells - n_movies

        for _bi in range(n_blanks):
                blank_index = n_movies + _bi
                ffmpeg_cmd.extend(['-f', 'lavfi', '-i', f'color=c=black:s={tile_width}x720:r={fps}:d={target_duration:.6f}'])
                filter_parts.append(
                        f"[{blank_index}:v]setpts=PTS-STARTPTS,trim=duration={target_duration:.6f},drawtext=text='':x=20:y=20:fontsize=28:fontcolor=white@0.35[v{blank_index}]"
                )

        layout_cells = []
        for _r in range(nrows):
                for _c in range(ncols):
                        xpos = '0' if _c == 0 else '+'.join(f'w{_c2}' for _c2 in range(_c))
                        ypos = '0' if _r == 0 else '+'.join(f'h{_r2 * ncols}' for _r2 in range(_r))
                        layout_cells.append(f'{xpos}_{ypos}')
        layout_str = '|'.join(layout_cells)

        stream_labels = [f'[v{i}]' for i in range(n_cells)]
        filter_parts.append(''.join(stream_labels) + f'xstack=inputs={n_cells}:layout={layout_str}[vout]')
        filter_complex = ';'.join(filter_parts)

        tag = _sanitize_for_filename(source_tag)
        resolved_output_name = output_name if output_name.startswith(f'{tag}_') else f'{tag}_{output_name}'
        output_path = outdir / resolved_output_name
        ffmpeg_cmd.extend([
                '-filter_complex', filter_complex,
                '-map', '[vout]',
                '-c:v', 'prores_ks',
                '-profile:v', '3',
                '-pix_fmt', 'yuv422p10le',
                str(output_path),
        ])
        subprocess.run(ffmpeg_cmd, check=True)
        print(f'[uv-montage] montage saved: {output_path}')
        return output_path



def _resolve_fits_paths(fits_dir: Path, pattern: str) -> list[Path]:
	fits_paths = sorted(path for path in fits_dir.glob(pattern) if path.is_file())
	if not fits_paths:
		raise ValueError(f'No files matched: {fits_dir / pattern}')
	return fits_paths


def _add_common_plot_args(parser: argparse.ArgumentParser) -> None:
	parser.add_argument('--products', default='RR,LL')
	parser.add_argument('--sample-frac', type=float, default=0.01)
	parser.add_argument('--overlay-flags', action='store_true')
	parser.add_argument('--seed', type=int, default=100)
	parser.add_argument('--chan-range', nargs=2, type=int, default=None, metavar=('START', 'END'))
	parser.add_argument('--uv-good-marker-size', type=float, default=0.075)
	parser.add_argument('--uv-flag-marker-size', type=float, default=0.03)


def main() -> int:
	parser = argparse.ArgumentParser(description='Unified source uv-sampling pipeline')
	subparsers = parser.add_subparsers(dest='command', required=True)

	single = subparsers.add_parser('single', help='Generate whole+snapshots+movie for one source file')
	single.add_argument('--fits', required=True)
	single.add_argument('--index-cache', default=None)
	single.add_argument('--source', default=None, help='Source name override for FITS with multiple sources')
	single.add_argument('--outdir', default='./diagnostics_out')
	single.add_argument('--time-step', type=int, default=1)
	single.add_argument('--fps', type=int, default=8)
	single.add_argument('--fixed-limits', action='store_true', default=True)
	single.add_argument('--skip-movie', action='store_true')
	_add_common_plot_args(single)

	combined = subparsers.add_parser('combined', help='Generate combined UV-sampling plot across files/sources')
	combined.add_argument('--fits', nargs='+', required=True)
	combined.add_argument('--outdir', default='./diagnostics_out')
	combined.add_argument('--out-name', default=None)
	combined.add_argument('--title', default=None)
	_add_common_plot_args(combined)

	montage = subparsers.add_parser('montage', help='Generate 3x2 montage movie from per-scan movies')
	montage.add_argument('--outdir', default='./diagnostics_out')
	montage.add_argument('--pattern', default='**/*_uv_sampling_movie_prores.mov')
	montage.add_argument('--output-name', default='uv_sampling_montage_3x2_prores.mov')
	montage.add_argument('--tile-width', type=int, default=640)
	montage.add_argument('--fps', type=int, default=8)

	batch = subparsers.add_parser('batch', help='Run full uv-sampling pipeline over matching files')
	batch.add_argument('--fits-dir', default='~/DATA/gmrt_40_014/work/split')
	batch.add_argument('--pattern', default='*_calibrated.uvfits')
	batch.add_argument('--outdir', default='./diagnostics_out')
	batch.add_argument('--time-step', type=int, default=1)
	batch.add_argument('--fps', type=int, default=8)
	batch.add_argument('--fixed-limits', action='store_true', default=True)
	batch.add_argument('--skip-movie', action='store_true')
	batch.add_argument('--skip-combined', action='store_true')
	batch.add_argument('--make-montage', action='store_true')
	_add_common_plot_args(batch)

	args = parser.parse_args()

	if args.command == 'single':
		outdir = Path(args.outdir).expanduser().resolve()
		outdir.mkdir(parents=True, exist_ok=True)
		fits_path = Path(args.fits).expanduser().resolve()
		index_cache = Path(args.index_cache).expanduser().resolve() if args.index_cache else None
		products = _parse_csv_list(args.products)
		chan_range = tuple(args.chan_range) if args.chan_range else None
		_generate_single(
			fits_path=fits_path,
			outdir=outdir,
			products=products,
			sample_frac=float(args.sample_frac),
			overlay_flags=bool(args.overlay_flags),
			seed=int(args.seed),
			time_step=int(args.time_step),
			chan_range=chan_range,
			fps=int(args.fps),
			uv_good_marker_size=float(args.uv_good_marker_size),
			uv_flag_marker_size=float(args.uv_flag_marker_size),
			fixed_limits=bool(args.fixed_limits),
			skip_movie=bool(args.skip_movie),
			source=args.source,
			index_cache=index_cache,
		)
		return 0

	if args.command == 'combined':
		outdir = Path(args.outdir).expanduser().resolve()
		outdir.mkdir(parents=True, exist_ok=True)
		fits_paths = [Path(item).expanduser().resolve() for item in args.fits]
		_generate_combined_plot(
			fits_paths=fits_paths,
			outdir=outdir,
			products=_parse_csv_list(args.products),
			sample_frac=float(args.sample_frac),
			overlay_flags=bool(args.overlay_flags),
			seed=int(args.seed),
			chan_range=tuple(args.chan_range) if args.chan_range else None,
			uv_good_marker_size=float(args.uv_good_marker_size),
			uv_flag_marker_size=float(args.uv_flag_marker_size),
			out_name=args.out_name,
			title=args.title,
		)
		return 0

	if args.command == 'montage':
		outdir = Path(args.outdir).expanduser().resolve()
		movie_files = sorted(path for path in outdir.glob(str(args.pattern)) if path.is_file())
		inferred_sources = [path.name.split('_', 1)[0] for path in movie_files]
		_generate_montage(
			outdir=outdir,
			pattern=str(args.pattern),
			output_name=str(args.output_name),
			tile_width=int(args.tile_width),
			fps=int(args.fps),
			source_tag=_source_group_tag(inferred_sources),
			source_filters=inferred_sources,
		)
		return 0

	if args.command == 'batch':
		outdir = Path(args.outdir).expanduser().resolve()
		outdir.mkdir(parents=True, exist_ok=True)
		fits_dir = Path(args.fits_dir).expanduser().resolve()
		fits_paths = _resolve_fits_paths(fits_dir, args.pattern)
		products = _parse_csv_list(args.products)
		chan_range = tuple(args.chan_range) if args.chan_range else None

		print(f'[uv] found {len(fits_paths)} files')
		resolved_sources = []
		for fits_path in fits_paths:
			source_guess = fits_path.name.split('_', 1)[0].upper()
			index_cache = fits_path.with_name(f'{fits_path.name}.row_index_cache.npz')
			print(f'[uv] processing {fits_path.name} (source={source_guess})')
			resolved_source, *_ = _generate_single(
				fits_path=fits_path,
				outdir=outdir,
				products=products,
				sample_frac=float(args.sample_frac),
				overlay_flags=bool(args.overlay_flags),
				seed=int(args.seed),
				time_step=int(args.time_step),
				chan_range=chan_range,
				fps=int(args.fps),
				uv_good_marker_size=float(args.uv_good_marker_size),
				uv_flag_marker_size=float(args.uv_flag_marker_size),
				fixed_limits=bool(args.fixed_limits),
				skip_movie=bool(args.skip_movie),
				source=source_guess,
				index_cache=index_cache if index_cache.exists() else None,
			)
			resolved_sources.append(resolved_source)

		if not args.skip_combined:
			_generate_combined_plot(
				fits_paths=fits_paths,
				outdir=outdir,
				products=products,
				sample_frac=float(args.sample_frac),
				overlay_flags=bool(args.overlay_flags),
				seed=int(args.seed),
				chan_range=chan_range,
				uv_good_marker_size=float(args.uv_good_marker_size),
				uv_flag_marker_size=float(args.uv_flag_marker_size),
			)

		if args.make_montage:
			_generate_montage(
				outdir=outdir,
				pattern='**/*_uv_sampling_movie_prores.mov',
				output_name='uv_sampling_montage_3x2_prores.mov',
				tile_width=640,
				fps=int(args.fps),
				source_tag=_source_group_tag(resolved_sources),
				source_filters=resolved_sources,
			)
		print(f'[uv] done. outputs under: {outdir}')
		return 0

	raise SystemExit('Unsupported command')


if __name__ == '__main__':
	raise SystemExit(main())
