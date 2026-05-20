#!/usr/bin/env bash

# Publish selected GMRT 40_014 diagnostics to a separate gh-pages worktree.
#
# Design goals:
# - Keep source/reproducibility on the main branch.
# - Keep published artifacts on a dedicated gh-pages branch/worktree.
# - Publish a curated static site for one chosen run at a time.
# - Preserve older published runs under runs/<timestamp>/ while updating latest/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_WORK_DIR="${WORK_DIR:-$HOME/DATA/gmrt_40_014/work}"
DEFAULT_PAGES_DIR="${PAGES_DIR:-${REPO_ROOT}_gh_pages}"
DEFAULT_BRANCH="gh-pages"
DEFAULT_LATEST_ALIAS="latest"
DEFAULT_SITE_TITLE="GMRT 40_014 Calibration Diagnostics"
DEFAULT_PROJECT_SUBDIR="40_014"

WORK_DIR="$DEFAULT_WORK_DIR"
PAGES_DIR="$DEFAULT_PAGES_DIR"
PUBLISH_BRANCH="$DEFAULT_BRANCH"
LATEST_ALIAS="$DEFAULT_LATEST_ALIAS"
SITE_TITLE="$DEFAULT_SITE_TITLE"
PROJECT_SUBDIR="$DEFAULT_PROJECT_SUBDIR"
MANIFEST_PATH=""
RUN_TS=""
PUSH=0
DRY_RUN=0
OPEN_AFTER=1
PUBLISH_FILES=()
GHPAGES_LAYOUT_MANIFEST="${GHPAGES_LAYOUT_MANIFEST:-}"
MANIFEST_TRACE_PATH="${MANIFEST_TRACE_PATH:-}"

usage() {
	cat <<EOF
Usage: bash bin/publish_gh_pages.sh [options]

Publish selected PNG/PDF outputs from a workflow run to a separate gh-pages worktree.

Options:
  --manifest PATH      Publish the run described by this manifest file.
  --run-ts TS          Publish the run with timestamp TS (looks up the matching manifest).
  --work-dir PATH      Override work directory (default: $DEFAULT_WORK_DIR).
  --pages-dir PATH     Override gh-pages worktree path (default: $DEFAULT_PAGES_DIR).
  --branch NAME        Publication branch name (default: $DEFAULT_BRANCH).

  --project-subdir DIR Project subdirectory under the gh-pages root (default: $DEFAULT_PROJECT_SUBDIR).
  --site-title TEXT    Site title used in generated HTML.
  --push               Push the publication branch after commit.
	--open               Open the committed index.html in the browser after publishing (default).
	--no-open            Do not open index.html after publishing.
  --dry-run            Show planned actions without writing files or committing.
  -h, --help           Show this help.

Examples:
  bash bin/publish_gh_pages.sh
  bash bin/publish_gh_pages.sh --run-ts 20260511_114843
  bash bin/publish_gh_pages.sh --manifest "$HOME/DATA/gmrt_40_014/work/logs/run_gmrt_40_014_products_20260511_114843.txt" --push

Environment knobs:
	GHPAGES_LAYOUT_MANIFEST=PATH Use explicit Moon layout manifest JSON.
EOF
}

log() {
	echo "[publish-gh-pages] $*"
}

die() {
	echo "[publish-gh-pages] ERROR: $*" >&2
	exit 1
}

copy_file() {
	local src="$1"
	local dst="$2"
	if [[ "$DRY_RUN" -eq 1 ]]; then
		log "DRY-RUN copy $src -> $dst"
		return 0
	fi
	mkdir -p "$(dirname "$dst")"
	cp -f "$src" "$dst"
}

escape_html() {
	python - "$1" <<'PY'
import html
import sys
print(html.escape(sys.argv[1], quote=True))
PY
}

render_csv_table_page() {
	local run_dir="$1"
	local rel_csv="$2"
	local table_title="$3"
	local abs_csv="$run_dir/$rel_csv"
	[[ -f "$abs_csv" ]] || return 1

	local rel_html="${rel_csv%.csv}.table.html"
	if [[ "$rel_html" == "$rel_csv" ]]; then
		rel_html="${rel_csv}.table.html"
	fi
	local abs_html="$run_dir/$rel_html"
	mkdir -p "$(dirname "$abs_html")"

	python - "$abs_csv" "$abs_html" "$table_title" <<'PY'
import csv
import html
import os
import sys

csv_path, html_path, table_title = sys.argv[1], sys.argv[2], sys.argv[3]

with open(csv_path, newline='', encoding='utf-8') as handle:
	rows = list(csv.reader(handle))

header = rows[0] if rows else []
data_rows = rows[1:] if len(rows) > 1 else []
csv_name = os.path.basename(csv_path)

quantity_notes = []
if csv_name.endswith('_per_integration.csv'):
	quantity_notes = [
		('run_id', 'Workflow run identifier from the launcher.'),
		('selfcal_name', 'Selfcal directory name for this scan/source.'),
		('scan_tag', 'Scan label used to group integrations.'),
		('start_integration', 'Starting integration index within the scan.'),
		('cycle', 'Selfcal cycle label (for example sc1…sc4, final).'),
		('cycle_index', 'Numeric ordering of the selfcal cycle.'),
		('peak_jy_per_beam', 'Peak pixel value in the cleaned image for that cycle.'),
		('residual_peak_jy_per_beam', 'Peak residual pixel value after model subtraction.'),
		('model_sum_jy', 'Integrated model flux estimate for the cycle.'),
		('dr_peak_over_residual', 'Peak-to-residual dynamic range = peak_jy_per_beam / residual_peak_jy_per_beam.'),
		('nminor_cycles_total', 'Total minor cycles performed by tclean.'),
		('nmajor_cycles_used', 'Total major cycles used by tclean.'),
	]
elif csv_name.endswith('_summary_stats.csv'):
	quantity_notes = [
		('cycle', 'Cycle label summarised across integrations.'),
		('n', 'Number of independent selfcal runs / time snapshots contributing to the cycle; the table reports descriptive statistics across these 38 per-run measurements, and does not use stacked or combined images.'),
		('*_min', 'Minimum value across the rows contributing to this cycle.'),
		('*_max', 'Maximum value across the rows contributing to this cycle.'),
		('*_std', 'Sample standard deviation across the rows contributing to this cycle.'),
		('*_median', 'Median across the rows contributing to this cycle.'),
		('dr_peak_over_residual_*', 'Peak-to-residual dynamic range summary; not peak/off-source RMS.'),
	]

def _cells(tag: str, values: list[str]) -> str:
	return ''.join(f'<{tag}>{html.escape(str(value), quote=True)}</{tag}>' for value in values)

parts = [
	'<!doctype html>',
	'<html lang="en">',
	'<head>',
	'  <meta charset="utf-8">',
	'  <meta name="viewport" content="width=device-width, initial-scale=1">',
	f'  <title>{html.escape(table_title, quote=True)}</title>',
	'  <style>',
	'    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 1rem auto; max-width: 96vw; padding: 0 1rem; line-height: 1.45; }',
	'    .meta { margin-bottom: 0.9rem; font-size: 0.92rem; opacity: 0.9; }',
	'    .table-wrap { overflow: auto; border: 1px solid #8884; border-radius: 8px; }',
	'    table { border-collapse: collapse; width: max-content; min-width: 100%; }',
	'    th, td { border: 1px solid #8884; padding: 0.35rem 0.5rem; font-size: 0.86rem; white-space: nowrap; }',
	'    th { position: sticky; top: 0; background: #f6f8fa; text-align: left; }',
	'    tr:nth-child(even) td { background: #8881; }',
	'    @media (prefers-color-scheme: dark) { th { background: #1f2328; } }',
	'  </style>',
	'</head>',
	'<body>',
	f'  <h1>{html.escape(table_title, quote=True)}</h1>',
	f'  <p class="meta"><a href="{html.escape(csv_name, quote=True)}">Download raw CSV</a> · Rows: {len(data_rows)} · Columns: {len(header)}</p>',
	'  <h2>Quantity index</h2>',
	'  <ul>',
]

if quantity_notes:
	for key, desc in quantity_notes:
		parts.append(f'    <li><strong>{html.escape(key, quote=True)}</strong>: {html.escape(desc, quote=True)}</li>')
else:
	parts.append('    <li>No quantity index available for this table.</li>')

parts.extend([
	'  </ul>',
	'  <p class="meta"><strong>Note:</strong> the DR column shown here is the table’s internal peak/residual metric. It is <em>not</em> the peak-to-off-source-RMS value shown in the cumulative movie inset.</p>',
	'  <div class="table-wrap">',
	'    <table>',
])

if header:
	parts.extend([
		'      <thead>',
		f'        <tr>{_cells("th", header)}</tr>',
		'      </thead>',
	])

parts.append('      <tbody>')
for row in data_rows:
	parts.append(f'        <tr>{_cells("td", row)}</tr>')
parts.extend([
	'      </tbody>',
	'    </table>',
	'  </div>',
	'</body>',
	'</html>',
])

with open(html_path, 'w', encoding='utf-8') as handle:
	handle.write('\n'.join(parts) + '\n')
PY

	echo "$rel_html"
}

trace_manifest_entry() {
	local section_id="$1"
	local entry_id="$2"
	local resolved_rel="$3"
	[[ -n "$MANIFEST_TRACE_PATH" ]] || return 0
	mkdir -p "$(dirname "$MANIFEST_TRACE_PATH")"
	printf '{"section_id":"%s","entry_id":"%s","resolved_rel":"%s"}\n' "$section_id" "$entry_id" "$resolved_rel" >> "$MANIFEST_TRACE_PATH"
}

load_moon_layout_manifest() {
	local manifest_path="$1"
	local run_ts="$2"
	[[ -f "$manifest_path" ]] || return 1

	local fields
	if ! fields="$(python - "$manifest_path" "$run_ts" <<'PY'
import json
import sys

manifest_path = sys.argv[1]
run_ts = sys.argv[2]
with open(manifest_path, 'r', encoding='utf-8') as handle:
    data = json.load(handle)

if run_ts and data.get('run_ts') and str(data.get('run_ts')) != run_ts:
    raise SystemExit(f"manifest run_ts mismatch: expected {run_ts}, got {data.get('run_ts')}")

moon = data.get('moon', {})
entries = moon.get('entries', {})

def sval(key, default=''):
    val = moon.get(key, default)
    return '' if val is None else str(val)

def bval(key):
    return '1' if bool(moon.get(key, False)) else '0'

def eval_entry(key):
    node = entries.get(key, {})
    if isinstance(node, dict):
        return '' if node.get('resolved_rel') is None else str(node.get('resolved_rel', ''))
    return ''

out = {
    'moon_products_root': sval('products_root', ''),
    'moon_no_phasecenter_dir': sval('no_phasecenter_dir', ''),
    'moon_phasecenter_dir': sval('phasecenter_dir', ''),
    'moon_has_no_phasecenter': bval('has_no_phasecenter'),
    'moon_has_phasecenter': bval('has_phasecenter'),
    'no_raw_mp4_rel': eval_entry('no_raw_movie'),
    'no_raw_stack_rel': eval_entry('no_raw_stack'),
    'no_raw_cum_mp4_rel': eval_entry('no_raw_rms_movie'),
    'pc_raw_mp4_rel': eval_entry('pc_raw_movie'),
    'pc_raw_stack_rel': eval_entry('pc_raw_stack'),
    'pc_raw_cum_mp4_rel': eval_entry('pc_raw_rms_movie'),
    'no_dst_mp4_rel': eval_entry('no_dst_movie'),
    'no_dst_stack_rel': eval_entry('no_dst_stack'),
    'no_dst_cum_mp4_rel': eval_entry('no_dst_rms_movie'),
    'pc_dst_mp4_rel': eval_entry('pc_dst_movie'),
    'pc_dst_stack_rel': eval_entry('pc_dst_stack'),
    'pc_dst_cum_mp4_rel': eval_entry('pc_dst_rms_movie'),
}

static_targets = data.get('static_targets', [])
out['static_count'] = str(len(static_targets))
for idx, section in enumerate(static_targets, start=1):
	if not isinstance(section, dict):
		continue
	prefix = f'static_{idx}_'
	out[prefix + 'section_id'] = str(section.get('section_id', ''))
	out[prefix + 'section_name'] = str(section.get('section_name', ''))
	out[prefix + 'source_label'] = str(section.get('source_label', ''))
	out[prefix + 'scan_tag'] = str(section.get('scan_tag', ''))
	out[prefix + 'stack_size'] = str(section.get('stack_size', ''))
	out[prefix + 'section_title'] = str(section.get('section_title', ''))
	out[prefix + 'section_context'] = str(section.get('section_context', ''))
	out[prefix + 'movie_rel'] = str(section.get('movie_rel', ''))
	out[prefix + 'stack_rel'] = str(section.get('stack_rel', ''))
	out[prefix + 'rms_movie_rel'] = str(section.get('rms_movie_rel', ''))
	out[prefix + 'diag_panel_rel'] = str(section.get('diag_panel_rel', ''))
	out[prefix + 'diag_csv_rel'] = str(section.get('diag_csv_rel', ''))
	out[prefix + 'diag_stats_rel'] = str(section.get('diag_stats_rel', ''))

for key, val in out.items():
    print(f"{key}\t{val}")
PY
)"; then
		return 1
	fi

	local key val
	while IFS=$'\t' read -r key val; do
		case "$key" in
			moon_products_root) moon_products_root="$val" ;;
			moon_no_phasecenter_dir) moon_no_phasecenter_dir="$val" ;;
			moon_phasecenter_dir) moon_phasecenter_dir="$val" ;;
			moon_has_no_phasecenter) moon_has_no_phasecenter="$val" ;;
			moon_has_phasecenter) moon_has_phasecenter="$val" ;;
			no_raw_mp4_rel) no_raw_mp4_rel="$val" ;;
			no_raw_stack_rel) no_raw_stack_rel="$val" ;;
			no_raw_cum_mp4_rel) no_raw_cum_mp4_rel="$val" ;;
			pc_raw_mp4_rel) pc_raw_mp4_rel="$val" ;;
			pc_raw_stack_rel) pc_raw_stack_rel="$val" ;;
			pc_raw_cum_mp4_rel) pc_raw_cum_mp4_rel="$val" ;;
			no_dst_mp4_rel) no_dst_mp4_rel="$val" ;;
			no_dst_stack_rel) no_dst_stack_rel="$val" ;;
			no_dst_cum_mp4_rel) no_dst_cum_mp4_rel="$val" ;;
			pc_dst_mp4_rel) pc_dst_mp4_rel="$val" ;;
			pc_dst_stack_rel) pc_dst_stack_rel="$val" ;;
			pc_dst_cum_mp4_rel) pc_dst_cum_mp4_rel="$val" ;;
			static_count) static_count="$val" ;;
			static_*_section_id) eval "$key=\"$val\"" ;;
			static_*_section_name) eval "$key=\"$val\"" ;;
			static_*_source_label) eval "$key=\"$val\"" ;;
			static_*_scan_tag) eval "$key=\"$val\"" ;;
			static_*_stack_size) eval "$key=\"$val\"" ;;
			static_*_section_title) eval "$key=\"$val\"" ;;
			static_*_section_context) eval "$key=\"$val\"" ;;
			static_*_movie_rel) eval "$key=\"$val\"" ;;
			static_*_stack_rel) eval "$key=\"$val\"" ;;
			static_*_rms_movie_rel) eval "$key=\"$val\"" ;;
			static_*_diag_panel_rel) eval "$key=\"$val\"" ;;
			static_*_diag_csv_rel) eval "$key=\"$val\"" ;;
			static_*_diag_stats_rel) eval "$key=\"$val\"" ;;
		esac
	done <<< "$fields"

	return 0
}

ensure_moon_layout_manifest() {
	if [[ -n "$GHPAGES_LAYOUT_MANIFEST" ]]; then
		[[ -f "$GHPAGES_LAYOUT_MANIFEST" ]] || return 1
		return 0
	fi

	local auto_manifest="$WORK_DIR/logs/moon_layout_manifest_${RUN_TS}.json"
	if python3 "$REPO_ROOT/bin/build_gmrt_40_014_layout_manifest.py" \
		--work-dir "$WORK_DIR" \
		--run-ts "$RUN_TS" \
		--output "$auto_manifest"; then
		GHPAGES_LAYOUT_MANIFEST="$auto_manifest"
		log "auto-generated moon layout manifest: $GHPAGES_LAYOUT_MANIFEST"
		return 0
	fi

	return 1
}

find_latest_manifest() {
	ls -1t "$WORK_DIR"/logs/run_gmrt_40_014_products_*.txt 2>/dev/null | head -1
}

resolve_manifest() {
	if [[ -n "$MANIFEST_PATH" && -n "$RUN_TS" ]]; then
		die "use either --manifest or --run-ts, not both"
	fi

	if [[ -n "$RUN_TS" ]]; then
		MANIFEST_PATH="$WORK_DIR/logs/run_gmrt_40_014_products_${RUN_TS}.txt"
	fi

	if [[ -z "$MANIFEST_PATH" ]]; then
		MANIFEST_PATH="$(find_latest_manifest)"
	fi

	[[ -n "$MANIFEST_PATH" ]] || die "no workflow manifest found under $WORK_DIR/logs"
	[[ -f "$MANIFEST_PATH" ]] || die "manifest not found: $MANIFEST_PATH"

	if [[ -z "$RUN_TS" ]]; then
		RUN_TS="$(basename "$MANIFEST_PATH" | sed -E 's/^run_gmrt_40_014_products_([0-9]{8}_[0-9]{6})\.txt$/\1/')"
	fi

	[[ "$RUN_TS" =~ ^[0-9]{8}_[0-9]{6}$ ]] || die "could not determine run timestamp from manifest: $MANIFEST_PATH"
}

ensure_pages_worktree() {
	local remote_has_branch=0
	local local_has_branch=0

	if [[ -d "$PAGES_DIR/.git" || -f "$PAGES_DIR/.git" ]]; then
		log "using existing pages worktree: $PAGES_DIR"
		return 0
	fi

	if git -C "$REPO_ROOT" show-ref --verify --quiet "refs/heads/$PUBLISH_BRANCH"; then
		local_has_branch=1
	fi

	if git -C "$REPO_ROOT" ls-remote --exit-code --heads origin "$PUBLISH_BRANCH" >/dev/null 2>&1; then
		remote_has_branch=1
	fi

	if [[ "$DRY_RUN" -eq 1 ]]; then
		if [[ "$local_has_branch" -eq 1 || "$remote_has_branch" -eq 1 ]]; then
			log "DRY-RUN would create worktree at $PAGES_DIR for existing branch $PUBLISH_BRANCH"
		else
			log "DRY-RUN would create orphan $PUBLISH_BRANCH worktree at $PAGES_DIR"
		fi
		return 0
	fi

	if [[ "$local_has_branch" -eq 0 && "$remote_has_branch" -eq 1 ]]; then
		git -C "$REPO_ROOT" fetch origin "$PUBLISH_BRANCH:$PUBLISH_BRANCH"
		local_has_branch=1
	fi

	if [[ "$local_has_branch" -eq 1 ]]; then
		git -C "$REPO_ROOT" worktree add "$PAGES_DIR" "$PUBLISH_BRANCH"
		return 0
	fi

	git -C "$REPO_ROOT" worktree add --detach "$PAGES_DIR"
	git -C "$PAGES_DIR" checkout --orphan "$PUBLISH_BRANCH"
	git -C "$PAGES_DIR" rm -rf . >/dev/null 2>&1 || true
	find "$PAGES_DIR" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
}

collect_publish_files() {
	local diagnostics_dir="$WORK_DIR/diagnostics_out"
	local selfcal_dir="$WORK_DIR/casa_selfcal"
	local logs_dir="$WORK_DIR/logs"
	local workflow_log=""

	PUBLISH_FILES=()
	while IFS= read -r file_path; do
		[[ -n "$file_path" ]] && PUBLISH_FILES+=("$file_path")
	done < <(
		{
			if [[ -d "$diagnostics_dir" ]]; then
				find "$diagnostics_dir" -type f \( -name '*.png' -o -name '*.pdf' -o -name '*.gif' -o -name '*.mp4' -o -name '*.mov' \)
			fi
			if [[ -d "$selfcal_dir" ]]; then
				find "$selfcal_dir" -type f \( \
					-name '*.gif' -o -name '*.mp4' -o -name '*.mov' -o \
					-name 'scan*_clean_cycle_metrics_panel*.png' -o \
					-name 'scan*_clean_cycle_metrics*.csv' -o \
					-name '*destripe*.png' -o -name '*destrip*.png' -o \
					-name '*convergence*.png' -o -name '*iter_progression*.png' -o \
					-name '*iter_progression*.gif' -o -name '*iter_progression*.mp4' -o \
					-name '*stack*.png' -o \
					-name '*selfcal_movie*.png' -o -name '*selfcal_movie*.gif' -o -name '*selfcal_movie*.mp4' -o -name '*selfcal_movie*.mov' -o \
					-name '*before_after_compare*.png' -o -name '*before_after_compare*.gif' -o -name '*before_after_compare*.mp4' -o -name '*before_after_compare*.mov' -o \
					-name '*coadd*.png' -o -name '*coadd*.gif' -o -name '*coadd*.mp4' -o -name '*coadd*.mov' -o \
					-name '*cumulative*.png' -o -name '*cumulative*.gif' -o -name '*cumulative*.mp4' -o -name '*cumulative*.mov' -o \
					-name '*rms*.png' -o -name '*rms*.pdf' -o -name '*rms*.csv' \
				\)
			fi
			if [[ -d "$logs_dir" ]]; then
				find "$logs_dir" -maxdepth 1 -type f \( \
					-name '*moon0520*.cmd' -o -name '*moon0520*.log' -o \
					-name '*moon_selfcal*.cmd' -o -name '*moon_selfcal*.log' -o \
					-name 'clustering_moon0520*.cmd' -o -name 'clustering_moon0520*.log' \
				\)
			fi
		} | sort -u
	)

	workflow_log="$(awk '/^Workflow log:/{getline; print; exit}' "$MANIFEST_PATH")"
	if [[ -n "$workflow_log" && -f "$workflow_log" ]]; then
		PUBLISH_FILES+=("$workflow_log")
	fi
	PUBLISH_FILES+=("$MANIFEST_PATH")

	((${#PUBLISH_FILES[@]} > 0)) || die "no publishable files found for run $RUN_TS"
}

render_moon_debug_index() {
	local run_dir="$1"
	local rel_case_dir="$2"
	local title="$3"
	local debug_root="$run_dir/$rel_case_dir/destripe_debug"
	local debug_file="$run_dir/$rel_case_dir/debug.html"
	local case_root="$run_dir/$rel_case_dir"
	local html_title
	html_title="$(escape_html "$title")"

	[[ -d "$debug_root" ]] || return 0

	cat > "$debug_file" <<EOF
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>${html_title}</title>
  <style>
    :root { color-scheme: light dark; }
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 2rem auto; max-width: 1160px; padding: 0 1rem; line-height: 1.5; }
    .card { border: 1px solid #8884; border-radius: 10px; padding: 0.8rem; background: #8881; margin-bottom: 1rem; }
	img, video { max-width: 100%; height: auto; border-radius: 6px; display: block; }
	.movie-thumb { max-width: 420px; margin: 0 auto; cursor: zoom-in; }
    a { text-decoration: none; }
    .tiny { opacity: 0.9; font-size: 0.92rem; }
  </style>
</head>
<body>
	<p><a href="../../../../index.html">← Back to run summary</a></p>
  <h1>${html_title}</h1>
  <p class="tiny">Detailed destriping diagnostics are intentionally hidden from the landing page and collected here.</p>
EOF

	while IFS= read -r stack_dir; do
		local stack_name conv_png iter_gif iter_mp4 rel_conv rel_gif rel_mp4 html_name
		stack_name="$(basename "$stack_dir")"
		html_name="$(escape_html "$stack_name")"
		conv_png="$stack_dir/${stack_name}_converge_rms.png"
		iter_gif="$stack_dir/${stack_name}_iter_progression.gif"
		iter_mp4="$stack_dir/${stack_name}_iter_progression.mp4"
		cat >> "$debug_file" <<EOF
  <div class="card">
    <h2>${html_name}</h2>
EOF
		if [[ -f "$conv_png" ]]; then
			rel_conv="${conv_png#"$case_root/"}"
			cat >> "$debug_file" <<EOF
    <p><a href="${rel_conv}">Convergence plot (PNG)</a></p>
    <img src="${rel_conv}" alt="${html_name} convergence plot" loading="lazy">
EOF
		fi
		if [[ -f "$iter_gif" || -f "$iter_mp4" ]]; then
			cat >> "$debug_file" <<'EOF'
    <p>
EOF
			if [[ -f "$iter_gif" ]]; then
				rel_gif="${iter_gif#"$case_root/"}"
				cat >> "$debug_file" <<EOF
      <a href="${rel_gif}">Iteration GIF</a>
EOF
			fi
			if [[ -f "$iter_mp4" ]]; then
				rel_mp4="${iter_mp4#"$case_root/"}"
				if [[ -f "$iter_gif" ]]; then
					cat >> "$debug_file" <<'EOF'
      ·
EOF
				fi
				cat >> "$debug_file" <<EOF
      <a href="${rel_mp4}">Iteration MP4</a>
EOF
			fi
			cat >> "$debug_file" <<'EOF'
    </p>
EOF
		fi
		cat >> "$debug_file" <<'EOF'
  </div>
EOF
	done < <(find "$debug_root" -mindepth 1 -maxdepth 1 -type d | sort)

	cat >> "$debug_file" <<'EOF'
</body>
</html>
EOF
}

render_post_selfcal_section() {
	local tmp_file="$1"
	local section_title="$2"
	local section_context="$3"
	local movie_rel="$4"
	local stack_rel="$5"
	local rms_movie_rel="$6"
	local diag_panel_rel="$7"
	local diag_csv_rel="$8"
	local diag_stats_rel="$9"
	local diag_csv_table_rel="${10}"
	local diag_stats_table_rel="${11}"
	local section_note_html="${12}"
	local footer_link="${13}"
	local footer_label="${14}"
	local html_section_title html_section_context

	html_section_title="$(escape_html "$section_title")"
	html_section_context="$(escape_html "$section_context")"

	cat >> "$tmp_file" <<EOF
    <div class="moon-section">
      <h3>${html_section_title}</h3>
      <p class="tiny">${html_section_context}</p>
EOF
	if [[ -n "$section_note_html" ]]; then
		cat >> "$tmp_file" <<EOF
      ${section_note_html}
EOF
	fi

	cat >> "$tmp_file" <<'EOF'
      <div class="panel-grid">
        <div class="panel-card">
          <p><strong>Movie</strong></p>
EOF
	if [[ -n "$movie_rel" ]]; then
		cat >> "$tmp_file" <<EOF
          <video controls preload="metadata" onclick="window.open('${movie_rel}','_blank')" title="Click to open full-size">
            <source src="${movie_rel}" type="video/mp4">
          </video>
EOF
	else
		cat >> "$tmp_file" <<'EOF'
          <p class="tiny">Not generated for this source/run.</p>
EOF
	fi

	cat >> "$tmp_file" <<'EOF'
        </div>
        <div class="panel-card">
          <p><strong>Stacked image</strong></p>
EOF
	if [[ -n "$stack_rel" ]]; then
		cat >> "$tmp_file" <<EOF
          <a href="${stack_rel}"><img src="${stack_rel}" alt="${html_section_title} stacked image" loading="lazy"></a>
EOF
	else
		cat >> "$tmp_file" <<'EOF'
          <p class="tiny">Not generated for this source/run.</p>
EOF
	fi

	cat >> "$tmp_file" <<'EOF'
        </div>
        <div class="panel-card">
          <p><strong>RMS evolution</strong></p>
EOF
	if [[ -n "$rms_movie_rel" ]]; then
		cat >> "$tmp_file" <<EOF
          <video controls preload="metadata" onclick="window.open('${rms_movie_rel}','_blank')" title="Click to open full-size">
            <source src="${rms_movie_rel}" type="video/mp4">
          </video>
EOF
	else
		cat >> "$tmp_file" <<'EOF'
          <p class="tiny">Not generated for this source/run.</p>
EOF
	fi

	cat >> "$tmp_file" <<'EOF'
        </div>
EOF

	if [[ -n "$diag_panel_rel" || -n "$diag_csv_rel" || -n "$diag_stats_rel" || -n "$diag_csv_table_rel" || -n "$diag_stats_table_rel" ]]; then
		cat >> "$tmp_file" <<'EOF'
        <div class="panel-card">
          <p><strong>Selfcal diagnostics</strong></p>
EOF
		if [[ -n "$diag_panel_rel" ]]; then
			cat >> "$tmp_file" <<EOF
          <a href="${diag_panel_rel}"><img src="${diag_panel_rel}" alt="${html_section_title} selfcal diagnostics panel" loading="lazy"></a>
EOF
		fi
		if [[ -n "$diag_csv_rel" || -n "$diag_stats_rel" ]]; then
			cat >> "$tmp_file" <<'EOF'
          <p>
EOF
			if [[ -n "$diag_csv_rel" ]]; then
				if [[ -n "$diag_csv_table_rel" ]]; then
					cat >> "$tmp_file" <<EOF
            <a href="${diag_csv_table_rel}">Per-integration table</a> · <a href="${diag_csv_rel}">CSV</a>
EOF
				else
					cat >> "$tmp_file" <<EOF
            <a href="${diag_csv_rel}">Per-integration CSV</a>
EOF
				fi
			fi
			if [[ -n "$diag_csv_rel" && -n "$diag_stats_rel" ]]; then
				cat >> "$tmp_file" <<'EOF'
            ·
EOF
			fi
			if [[ -n "$diag_stats_rel" ]]; then
				if [[ -n "$diag_stats_table_rel" ]]; then
					cat >> "$tmp_file" <<EOF
            <a href="${diag_stats_table_rel}">Summary stats table</a> · <a href="${diag_stats_rel}">CSV</a>
EOF
				else
					cat >> "$tmp_file" <<EOF
            <a href="${diag_stats_rel}">Summary stats CSV</a>
EOF
				fi
			fi
			cat >> "$tmp_file" <<'EOF'
          </p>
EOF
		fi
		if [[ -z "$diag_panel_rel" && -z "$diag_csv_rel" && -z "$diag_stats_rel" ]]; then
			cat >> "$tmp_file" <<'EOF'
          <p class="tiny">Not generated for this source/run.</p>
EOF
		fi
		cat >> "$tmp_file" <<'EOF'
        </div>
EOF
	fi

	cat >> "$tmp_file" <<'EOF'
      </div>
EOF
	if [[ -n "$footer_link" && -n "$footer_label" ]]; then
		local html_footer_label
		html_footer_label="$(escape_html "$footer_label")"
		cat >> "$tmp_file" <<EOF
      <p class="tiny"><a href="${footer_link}">${html_footer_label}</a></p>
EOF
	fi

	cat >> "$tmp_file" <<'EOF'
    </div>
EOF
}

render_run_index() {
	local run_dir="$1"
	local run_rel="$2"
	local published_at="$3"
	local source_commit="$4"
	local source_branch="$5"
	local workflow_log_name="$6"
	local manifest_name="$7"
	local back_link="${8:-../../index.html}"
	local base_path
	# Derive base_path from back_link so all relative nav URLs stay consistent
	# e.g. back_link="../../index.html" → base_path="../../"
	base_path="${back_link%index.html}"
	local tmp_file="$run_dir/index.html"
	local html_title html_run_rel
	html_title="$(escape_html "$SITE_TITLE")"
	html_run_rel="$(escape_html "$run_rel")"

	cat > "$tmp_file" <<EOF
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>${html_title} — ${RUN_TS}</title>
  <style>
    :root { color-scheme: light dark; }
		body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 2rem auto; max-width: 1160px; padding: 0 1rem; line-height: 1.5; }
    h1, h2 { line-height: 1.2; }
    h1 { margin-bottom: 0.45rem; }
    h2 { margin-bottom: 0.55rem; }
    .meta { padding: 1rem; border: 1px solid #8884; border-radius: 10px; background: #8881; }
		.summary { margin-top: 1rem; padding: 1rem; border: 1px solid #8884; border-radius: 10px; background: #8881; }
		.scheme { width: 100%; border-collapse: collapse; margin-top: 0.8rem; }
		.scheme th, .scheme td { border: 1px solid #8884; padding: 0.45rem 0.6rem; text-align: left; vertical-align: top; }
		.scheme th { background: #8882; }
		.stage { margin-top: 1.2rem; padding: 1.1rem; border: 1px solid #8884; border-radius: 10px; background: #8881; }
		.stage-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; }
		.panel-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; align-items: start; }
		.moon-section .panel-grid { align-items: stretch; }
    .card { border: 1px solid #8884; border-radius: 10px; padding: 0.9rem; background: #8881; }
		.panel-card { border: 1px solid #8884; border-radius: 10px; padding: 0.75rem; background: #8881; }
		.moon-section .panel-card { display: flex; flex-direction: column; }
		.moon-section .panel-card > a { flex: 1; display: flex; align-items: center; justify-content: center; }
		.moon-section { margin-top: 1rem; border: 1px solid #8884; border-radius: 12px; background: #8881; padding: 1rem; }
		.moon-section h3 { margin-top: 0; margin-bottom: 0.25rem; }
		.card p { margin: 0.35rem 0; }
		.panel-card p { margin: 0.2rem 0 0.5rem 0; }
    img { max-width: 100%; height: auto; border-radius: 6px; display: block; }
		.panel-card img, .panel-card video { width: 100%; border-radius: 8px; }
		.moon-section .panel-card img { width: auto; max-width: 100%; margin-left: auto; margin-right: auto; }
		.panel-card video { cursor: pointer; }
    a { text-decoration: none; }
    code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
		ul { padding-left: 1.1rem; }
		.tiny { opacity: 0.9; font-size: 0.92rem; }
		.gallery { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 0.9rem; }
		.gallery-item { border: 1px solid #8884; border-radius: 10px; padding: 0.55rem; background: #8881; }
		.caption { margin-top: 0.4rem; font-size: 0.85rem; opacity: 0.88; word-break: break-word; }
  </style>
</head>
<body>
  <h1>${html_title}</h1>
  <p class="tiny"><strong>Published:</strong> ${published_at}</p>
  <div class="meta">
    <p><strong>Run timestamp:</strong> <code>${RUN_TS}</code></p>
    <p><strong>Published at:</strong> <code>${published_at}</code></p>
    <p><strong>Source branch:</strong> <code>${source_branch}</code></p>
    <p><strong>Source commit:</strong> <code>${source_commit}</code></p>
    <p><strong>Published folder:</strong> <code>${html_run_rel}</code></p>
    <p><strong>Manifest:</strong> <a href="logs/${manifest_name}">${manifest_name}</a></p>
EOF

	if [[ -n "$workflow_log_name" ]]; then
		cat >> "$tmp_file" <<EOF
    <p><strong>Workflow log:</strong> <a href="logs/${workflow_log_name}">${workflow_log_name}</a></p>
EOF
	fi

	cat >> "$tmp_file" <<'EOF'
  </div>

	<div class="summary">
		<h2>What this report shows</h2>
		<p>This page summarizes how the GMRT 40_014 visibility data evolves through calibration. Each section contains inline previews and direct links to full-resolution files for zooming and detailed inspection.</p>
		<table class="scheme">
			<thead>
				<tr>
					<th>Section</th>
					<th>Calibrator / Product</th>
					<th>What is shown</th>
					<th>Why it matters</th>
				</tr>
			</thead>
			<tbody>
				<tr>
					<td>Primary bandpass calibration diagnostics</td>
					<td>3C48</td>
					<td>Iterative solution-solving process: baseline-averaged spectra, Stokes-V and spectral deviation flags, bandpass solution plots</td>
					<td>Establishes robust instrumental bandpass calibration and RFI rejection baseline</td>
				</tr>
				<tr>
					<td>Primary calibration self-check</td>
					<td>3C48</td>
					<td>Calibrated visibility plots after applying primary solutions back to 3C48</td>
					<td>Verifies that the primary calibration solutions are self-consistent and well-behaved</td>
				</tr>
				<tr>
					<td>Primary calibration transfer check</td>
					<td>3C468.1</td>
					<td>Calibrated visibility plots after transferring primary solutions to 3C468.1</td>
					<td>Checks that primary calibration transfers cleanly to the target field calibrator</td>
				</tr>
				<tr>
					<td>Advanced flagging diagnostics — clustering algorithm</td>
					<td>3C468.1</td>
					<td>Before/after views of the clustering algorithm on spectrum, Stokes-V, and uv-distance</td>
					<td>Quantifies the impact of advanced flagging before secondary phase solving</td>
				</tr>
				<tr>
					<td>Full calibration QA</td>
					<td>3C468.1</td>
					<td>Final calibrated visibility plots after applying both primary and secondary solutions with clustering-derived flags</td>
					<td>Represents the end-to-end calibration quality of the target field data products</td>
				</tr>
			</tbody>
		</table>
	</div>

	<div class="stage">
		<h2>Primary bandpass calibration diagnostics (3C48)</h2>
		<p>These plots show the solution-solving process for the primary bandpass calibration using 3C48. We iteratively solve for the bandpass, identify and flag bad data based on two criteria — high Stokes-V emission (a proxy for RFI) and deviation of the baseline-averaged spectrum from the Perley–Butler source model — and repeat until stable, well-behaved solutions are reached. The section includes both the per-iteration flagging diagnostic plots and the final bandpass solution plots.</p>
		<div class="stage-grid">
EOF

	while IFS= read -r rel_png; do
		local rel_pdf card_title html_card_title
		rel_pdf="${rel_png%_page*.png}.pdf"
		card_title="$(basename "$rel_png")"
		html_card_title="$(escape_html "$card_title")"
		cat >> "$tmp_file" <<EOF
    <div class="card">
      <p><strong>${html_card_title}</strong></p>
      <a href="${rel_png}"><img src="${rel_png}" alt="${html_card_title}"></a>
EOF
		if [[ -f "$run_dir/$rel_pdf" ]]; then
			cat >> "$tmp_file" <<EOF
      <p><a href="${rel_png}" target="_blank" rel="noopener">Open full-resolution PNG</a> · <a href="${rel_pdf}" target="_blank" rel="noopener">Open PDF</a></p>
EOF
		else
			cat >> "$tmp_file" <<EOF
      <p><a href="${rel_png}" target="_blank" rel="noopener">Open full-resolution PNG</a></p>
EOF
		fi
		cat >> "$tmp_file" <<'EOF'
    </div>
EOF
	done < <(cd "$run_dir" && find diagnostics_out/primary/3c48/solve -type f -name '*.png' 2>/dev/null | sort)

	if ! (cd "$run_dir" && find diagnostics_out/primary/3c48/solve -type f -name '*.png' | grep -q .); then
		cat >> "$tmp_file" <<'EOF'
      <p>No PNG diagnostics found for this stage.</p>
EOF
	fi

	cat >> "$tmp_file" <<'EOF'
    </div>
  </div>

  <div class="stage">
    <h2>Primary calibration self-check (3C48)</h2>
    <p>These diagnostic plots probe the quality of the primary calibration by applying the derived solutions back to 3C48 itself. Good calibration should yield clean, well-structured visibilities consistent with the expected source behaviour. Residual structure or anomalous baselines here indicate issues in the primary solve.</p>
    <div class="stage-grid">
EOF

	while IFS= read -r rel_png; do
		local rel_pdf card_title html_card_title
		rel_pdf="${rel_png%_page*.png}.pdf"
		card_title="$(basename "$rel_png")"
		html_card_title="$(escape_html "$card_title")"
		cat >> "$tmp_file" <<EOF
    <div class="card">
      <p><strong>${html_card_title}</strong></p>
      <a href="${rel_png}" target="_blank" rel="noopener"><img src="${rel_png}" alt="${html_card_title}"></a>
EOF
		if [[ -f "$run_dir/$rel_pdf" && "$rel_pdf" != "$rel_png" ]]; then
			cat >> "$tmp_file" <<EOF
      <p><a href="${rel_png}" target="_blank" rel="noopener">Open full-resolution PNG</a> · <a href="${rel_pdf}" target="_blank" rel="noopener">Open PDF</a></p>
EOF
		else
			cat >> "$tmp_file" <<EOF
      <p><a href="${rel_png}" target="_blank" rel="noopener">Open full-resolution PNG</a></p>
EOF
		fi
		cat >> "$tmp_file" <<'EOF'
    </div>
EOF
	done < <(cd "$run_dir" && find diagnostics_out/primary/3c48/selfcheck -type f -name '*.png' 2>/dev/null | sort)

	cat >> "$tmp_file" <<'EOF'
    </div>
  </div>

  <div class="stage">
    <h2>Primary calibration transfer check (3C468.1)</h2>
    <p>These plots show the QA for the primary calibration when its solutions are transferred to and applied to 3C468.1, the secondary calibrator. This tests whether the primary calibration is stable and transferable across sources before any secondary self-calibration is attempted.</p>
    <div class="stage-grid">
EOF

	while IFS= read -r rel_png; do
		local rel_pdf card_title html_card_title
		rel_pdf="${rel_png%_page*.png}.pdf"
		card_title="$(basename "$rel_png")"
		html_card_title="$(escape_html "$card_title")"
		cat >> "$tmp_file" <<EOF
    <div class="card">
      <p><strong>${html_card_title}</strong></p>
      <a href="${rel_png}" target="_blank" rel="noopener"><img src="${rel_png}" alt="${html_card_title}"></a>
EOF
		if [[ -f "$run_dir/$rel_pdf" && "$rel_pdf" != "$rel_png" ]]; then
			cat >> "$tmp_file" <<EOF
      <p><a href="${rel_png}" target="_blank" rel="noopener">Open full-resolution PNG</a> · <a href="${rel_pdf}" target="_blank" rel="noopener">Open PDF</a></p>
EOF
		else
			cat >> "$tmp_file" <<EOF
      <p><a href="${rel_png}" target="_blank" rel="noopener">Open full-resolution PNG</a></p>
EOF
		fi
		cat >> "$tmp_file" <<'EOF'
    </div>
EOF
	done < <(cd "$run_dir" && find diagnostics_out/secondary/3c468.1/transfer -type f -name '*.png' 2>/dev/null | sort)

	cat >> "$tmp_file" <<'EOF'
    </div>
  </div>

  <div class="stage">
    <h2>Advanced flagging diagnostics — clustering algorithm (3C468.1)</h2>
    <p>These plots show the effect of the clustering-based advanced flagging algorithm applied to 3C468.1. Side-by-side before/after views are shown for the baseline-averaged spectrum, Stokes-V, and uv-distance diagnostic planes. This flagging step is applied prior to secondary phase calibration to remove residual RFI and outlier baselines that survived the primary flagging iterations.</p>
    <div class="stage-grid">
  <div style="grid-column:1/-1;font-weight:600;border-bottom:1px solid #8884;padding-bottom:0.3rem;margin-bottom:0.2rem">Before flagging</div>
EOF

	while IFS= read -r rel_png; do
		local card_title html_card_title
		card_title="$(basename "$rel_png")"
		html_card_title="$(escape_html "$card_title")"
		cat >> "$tmp_file" <<EOF
    <div class="card">
      <p><strong>${html_card_title}</strong></p>
      <a href="${rel_png}" target="_blank" rel="noopener"><img src="${rel_png}" alt="${html_card_title}"></a>
      <p><a href="${rel_png}" target="_blank" rel="noopener">Open full-resolution PNG</a></p>
    </div>
EOF
	done < <(cd "$run_dir" && find diagnostics_out/secondary/3c468.1/flagging -type f -name '*_before.png' 2>/dev/null | sort)

	cat >> "$tmp_file" <<'EOF'
  <div style="grid-column:1/-1;font-weight:600;border-bottom:1px solid #8884;padding-bottom:0.3rem;margin-bottom:0.2rem;margin-top:0.6rem">After flagging</div>
EOF

	while IFS= read -r rel_png; do
		local card_title html_card_title
		card_title="$(basename "$rel_png")"
		html_card_title="$(escape_html "$card_title")"
		cat >> "$tmp_file" <<EOF
    <div class="card">
      <p><strong>${html_card_title}</strong></p>
      <a href="${rel_png}" target="_blank" rel="noopener"><img src="${rel_png}" alt="${html_card_title}"></a>
      <p><a href="${rel_png}" target="_blank" rel="noopener">Open full-resolution PNG</a></p>
    </div>
EOF
	done < <(cd "$run_dir" && find diagnostics_out/secondary/3c468.1/flagging -type f -name '*_after.png' 2>/dev/null | sort)

	cat >> "$tmp_file" <<'EOF'
    </div>
  </div>

  <div class="stage">
    <h2>Full calibration QA (3C468.1)</h2>
    <p>These plots show the end-to-end calibration quality for 3C468.1 after applying both the primary bandpass/gain solutions and the secondary phase calibration derived post-clustering. This is the final QA checkpoint: well-calibrated data should show clean, phase-coherent visibilities across all baselines and channels.</p>
    <div class="stage-grid">
EOF

	while IFS= read -r rel_png; do
		local rel_pdf card_title html_card_title
		rel_pdf="${rel_png%_page*.png}.pdf"
		card_title="$(basename "$rel_png")"
		html_card_title="$(escape_html "$card_title")"
		cat >> "$tmp_file" <<EOF
    <div class="card">
      <p><strong>${html_card_title}</strong></p>
      <a href="${rel_png}" target="_blank" rel="noopener"><img src="${rel_png}" alt="${html_card_title}"></a>
EOF
		if [[ -f "$run_dir/$rel_pdf" && "$rel_pdf" != "$rel_png" ]]; then
			cat >> "$tmp_file" <<EOF
      <p><a href="${rel_png}" target="_blank" rel="noopener">Open full-resolution PNG</a> · <a href="${rel_pdf}" target="_blank" rel="noopener">Open PDF</a></p>
EOF
		else
			cat >> "$tmp_file" <<EOF
      <p><a href="${rel_png}" target="_blank" rel="noopener">Open full-resolution PNG</a></p>
EOF
		fi
		cat >> "$tmp_file" <<'EOF'
    </div>
EOF
	done < <(cd "$run_dir" && find diagnostics_out/secondary/3c468.1/final_qa -type f -name '*.png' 2>/dev/null | sort)

	# ── Target QA: Moon scans ─────────────────────────────────────────────────
	# Iterate over all per-source subdirs under diagnostics_out/target/
	local moon_srcs=()
	while IFS= read -r d; do
		[[ -n "$d" ]] && moon_srcs+=("$d")
	done < <(cd "$run_dir" && find diagnostics_out/target -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort)

	if (( ${#moon_srcs[@]} > 0 )); then
		cat >> "$tmp_file" <<'EOF'
  </div>

  <div class="stage">
    <h2>Target QA — Moon scans</h2>
    <p>Visibility diagnostics for each Moon scan, shown in three calibration states: raw (flags only), primary-calibrated (bandpass/gain applied), and fully calibrated (primary + secondary phase solutions). All variants have both the primary flag table and the secondary clustering flag table applied at split time.</p>
EOF

		for moon_src_rel in "${moon_srcs[@]}"; do
			local moon_tag
			moon_tag="$(basename "$moon_src_rel")"
			local html_moon_tag
			html_moon_tag="$(escape_html "$(printf '%s' "$moon_tag" | tr '[:lower:]' '[:upper:]')")"  # e.g. MOON0520

			cat >> "$tmp_file" <<EOF
    <h3 style="margin-top:1.2rem">${html_moon_tag}</h3>
EOF

			local sub_label sub_path sub_desc
			for sub_label in "raw:Raw (flags only)" "transfer:Primary calibrated" "final_qa:Primary + secondary calibrated"; do
				sub_path="${sub_label%%:*}"
				sub_desc="${sub_label#*:}"
				local html_sub_desc
				html_sub_desc="$(escape_html "$sub_desc")"
				local dir_path="${moon_src_rel}/${sub_path}"

				# Check if any PNGs exist for this sub-dir
				local png_count=0
				png_count=$(cd "$run_dir" && find "$dir_path" -type f -name '*.png' 2>/dev/null | wc -l | tr -d ' ')

				if (( png_count == 0 )); then
					continue
				fi

				cat >> "$tmp_file" <<EOF
    <p style="margin-top:0.8rem"><strong>${html_sub_desc}</strong></p>
    <div class="stage-grid">
EOF

				while IFS= read -r rel_png; do
					local rel_pdf card_title html_card_title
					rel_pdf="${rel_png%_page*.png}.pdf"
					card_title="$(basename "$rel_png")"
					html_card_title="$(escape_html "$card_title")"
					cat >> "$tmp_file" <<EOF
    <div class="card">
      <p><strong>${html_moon_tag} — ${html_sub_desc}</strong></p>
      <a href="${rel_png}" target="_blank" rel="noopener"><img src="${rel_png}" alt="${html_card_title}"></a>
EOF
					if [[ -f "$run_dir/$rel_pdf" && "$rel_pdf" != "$rel_png" ]]; then
						cat >> "$tmp_file" <<EOF
      <p><a href="${rel_png}" target="_blank" rel="noopener">Open full-resolution PNG</a> · <a href="${rel_pdf}" target="_blank" rel="noopener">Open PDF</a></p>
EOF
					else
						cat >> "$tmp_file" <<EOF
      <p><a href="${rel_png}" target="_blank" rel="noopener">Open full-resolution PNG</a></p>
EOF
					fi
					cat >> "$tmp_file" <<'EOF'
    </div>
EOF
				done < <(cd "$run_dir" && find "$dir_path" -type f -name '*.png' 2>/dev/null | sort)

				cat >> "$tmp_file" <<'EOF'
    </div>
EOF
			done
		done

		cat >> "$tmp_file" <<'EOF'
  </div>
EOF
	else
		cat >> "$tmp_file" <<'EOF'
  </div>
EOF
	fi

	# ── Moon imaging diagnostics ──────────────────────────────────────────────
	local moon_imaging_dir
	moon_imaging_dir="$run_dir/diagnostics_out/moon_imaging"
	if [ -d "$moon_imaging_dir" ]; then
		cat >> "$tmp_file" <<'EOF'

  <h2>Moon imaging diagnostics</h2>
  <p class="tiny">Per-integration trajectory and selfcal products for the MOON0520 scan.</p>
  <div class="gallery">
EOF
		while IFS= read -r rel_png; do
			local alt_text
			alt_text="$(escape_html "$(basename "$rel_png" .png)")"
			cat >> "$tmp_file" <<EOF
    <div class="gallery-item">
      <a href="${rel_png}"><img src="${rel_png}" alt="${alt_text}" loading="lazy"></a>
      <div class="caption">${alt_text}</div>
    </div>
EOF
		done < <(cd "$run_dir" && find diagnostics_out/moon_imaging -type f -name '*.png' 2>/dev/null | sort)
		cat >> "$tmp_file" <<'EOF'
  </div>
EOF
	fi


	# ── Moon stage command/log provenance files ───────────────────────────────
	if (cd "$run_dir" && find logs -maxdepth 1 -type f \( -name '*moon0520*.cmd' -o -name '*moon0520*.log' -o -name '*moon_selfcal*.cmd' -o -name '*moon_selfcal*.log' -o -name 'clustering_moon0520*.cmd' -o -name 'clustering_moon0520*.log' \) 2>/dev/null | grep -q .); then
		cat >> "$tmp_file" <<'EOF'

  <h2>Provenance & reproducibility</h2>
  <p class="tiny">Stage command and log manifests kept in structured directories for clean audit trail.</p>
  <ul>
    <li><a href="logs/">Browse command/log manifests →</a></li>
  </ul>
EOF
	fi

	local moon_products_root=""
	local moon_no_phasecenter_dir=""
	local moon_phasecenter_dir=""
	local moon_has_no_phasecenter="0"
	local moon_has_phasecenter="0"
	local no_raw_mp4_rel=""
	local no_raw_stack_rel=""
	local no_raw_cum_mp4_rel=""
	local pc_raw_mp4_rel=""
	local pc_raw_stack_rel=""
	local pc_raw_cum_mp4_rel=""
	local no_dst_mp4_rel=""
	local no_dst_stack_rel=""
	local no_dst_cum_mp4_rel=""
	local pc_dst_mp4_rel=""
	local pc_dst_stack_rel=""
	local pc_dst_cum_mp4_rel=""
	local static_count="0"
	if ! load_moon_layout_manifest "$GHPAGES_LAYOUT_MANIFEST" "$RUN_TS"; then
		die "Moon layout manifest could not be loaded: $GHPAGES_LAYOUT_MANIFEST"
	fi

	trace_manifest_entry "moon.section1" "moon.s1.movie" "$no_raw_mp4_rel"
	trace_manifest_entry "moon.section1" "moon.s1.stack" "$no_raw_stack_rel"
	trace_manifest_entry "moon.section1" "moon.s1.rms_movie" "$no_raw_cum_mp4_rel"
	trace_manifest_entry "moon.section2" "moon.s2.movie" "$pc_raw_mp4_rel"
	trace_manifest_entry "moon.section2" "moon.s2.stack" "$pc_raw_stack_rel"
	trace_manifest_entry "moon.section2" "moon.s2.rms_movie" "$pc_raw_cum_mp4_rel"
	trace_manifest_entry "moon.section3" "moon.s3.movie" "$no_dst_mp4_rel"
	trace_manifest_entry "moon.section3" "moon.s3.stack" "$no_dst_stack_rel"
	trace_manifest_entry "moon.section3" "moon.s3.rms_movie" "$no_dst_cum_mp4_rel"
	trace_manifest_entry "moon.section4" "moon.s4.movie" "$pc_dst_mp4_rel"
	trace_manifest_entry "moon.section4" "moon.s4.stack" "$pc_dst_stack_rel"
	trace_manifest_entry "moon.section4" "moon.s4.rms_movie" "$pc_dst_cum_mp4_rel"

	if [[ "$moon_has_no_phasecenter" == "1" ]]; then
		render_moon_debug_index "$run_dir" "$moon_no_phasecenter_dir" "Observed phase-centre destriping debug"
	fi
	if [[ "$moon_has_phasecenter" == "1" ]]; then
		render_moon_debug_index "$run_dir" "$moon_phasecenter_dir" "Moon-centred tClean destriping debug"
	fi

	if [[ -n "$moon_products_root" ]]; then
		local moon_section3_note
		moon_section3_note='<p class="tiny"><strong>Caution:</strong> residual destriping artifacts can bias the automatic image-alignment shifts. Registration here uses FFT-based phase-correlation, so non-astronomical stripe power can introduce alignment error.</p>'

		cat >> "$tmp_file" <<'EOF'

  <div class="stage">
    <h2>Moon post-selfcal imaging & stacking summary</h2>
    <p>Four imaging/processing modes organized by geometry, then processing stage. Each section shows the observation movie, stacked image, and RMS evolution diagnostics. Detailed destriping diagnostics are linked separately at the bottom of each section.</p>
EOF

		if [[ "$moon_has_no_phasecenter" == "1" ]]; then
			render_post_selfcal_section "$tmp_file" \
				"1. Observed phase-centre – Raw selfcal" \
				"Moon drifts across the image. Stacking uses shift-then-add to align moon before co-adding." \
				"$no_raw_mp4_rel" "$no_raw_stack_rel" "$no_raw_cum_mp4_rel" \
				"" "" "" "" "" "" "" ""
		fi

		if [[ "$moon_has_phasecenter" == "1" ]]; then
			render_post_selfcal_section "$tmp_file" \
				"2. Moon-centred – Raw selfcal" \
				"Moon stays near image center. Stacking uses direct (no-shift) co-addition." \
				"$pc_raw_mp4_rel" "$pc_raw_stack_rel" "$pc_raw_cum_mp4_rel" \
				"" "" "" "" "" "" "" ""
		fi

		if [[ "$moon_has_no_phasecenter" == "1" ]]; then
			render_post_selfcal_section "$tmp_file" \
				"3. Observed phase-centre – Destriped" \
				"Same geometry as Section 1, but with striping artifacts removed." \
				"$no_dst_mp4_rel" "$no_dst_stack_rel" "$no_dst_cum_mp4_rel" \
				"" "" "" "" "" "$moon_section3_note" "${moon_no_phasecenter_dir}/debug.html" "→ Destriping debug details"
		fi

		if [[ "$moon_has_phasecenter" == "1" ]]; then
			render_post_selfcal_section "$tmp_file" \
				"4. Moon-centred – Destriped" \
				"Same geometry as Section 2, but with striping artifacts removed." \
				"$pc_dst_mp4_rel" "$pc_dst_stack_rel" "$pc_dst_cum_mp4_rel" \
				"" "" "" "" "" "" "${moon_phasecenter_dir}/debug.html" "→ Destriping debug details"
		fi

		cat >> "$tmp_file" <<'EOF'
  </div>
EOF
	fi

	# ── Manifest-driven static-target sections ───────────────────────────────
	if [[ "$static_count" =~ ^[0-9]+$ ]] && [[ "$static_count" -gt 0 ]]; then
		cat >> "$tmp_file" <<'EOF'

  <div class="stage">
    <h2>Post-selfcal imaging summary: 3C468.1</h2>
    <p>Per-scan post-selfcal products for 3C468.1, rendered from the layout manifest in the same panel order as the Moon sections.</p>
EOF
		for ((i=1; i<=static_count; i++)); do
			local pfx="static_${i}_"
			local section_title_var="${pfx}section_title"
			local section_context_var="${pfx}section_context"
			local movie_var="${pfx}movie_rel"
			local stack_var="${pfx}stack_rel"
			local rms_movie_var="${pfx}rms_movie_rel"
			local diag_panel_var="${pfx}diag_panel_rel"
			local diag_csv_var="${pfx}diag_csv_rel"
			local diag_stats_var="${pfx}diag_stats_rel"
			local section_title="${!section_title_var}"
			local section_context="${!section_context_var}"
			local movie_rel="${!movie_var}"
			local stack_rel="${!stack_var}"
			local rms_movie_rel="${!rms_movie_var}"
			local diag_panel_rel="${!diag_panel_var}"
			local diag_csv_rel="${!diag_csv_var}"
			local diag_stats_rel="${!diag_stats_var}"
			local diag_csv_table_rel=""
			local diag_stats_table_rel=""
			if [[ -n "$diag_csv_rel" && -f "$run_dir/$diag_csv_rel" ]]; then
				diag_csv_table_rel="$(render_csv_table_page "$run_dir" "$diag_csv_rel" "${section_title} · Per-integration clean-cycle metrics")" || diag_csv_table_rel=""
			fi
			if [[ -n "$diag_stats_rel" && -f "$run_dir/$diag_stats_rel" ]]; then
				diag_stats_table_rel="$(render_csv_table_page "$run_dir" "$diag_stats_rel" "${section_title} · Clean-cycle summary statistics")" || diag_stats_table_rel=""
			fi

			render_post_selfcal_section "$tmp_file" \
				"$section_title" "$section_context" \
				"$movie_rel" "$stack_rel" "$rms_movie_rel" \
				"$diag_panel_rel" "$diag_csv_rel" "$diag_stats_rel" \
				"$diag_csv_table_rel" "$diag_stats_table_rel" "" "" ""
		done
		cat >> "$tmp_file" <<'EOF'
  </div>
EOF
	fi

	cat >> "$tmp_file" <<'EOF'

  <h2>All PDF products</h2>
  <p class="tiny">Useful for high-quality zoom/export and documentation snapshots.</p>
  <ul>
EOF

	while IFS= read -r rel_pdf; do
		local html_name
		html_name="$(escape_html "$(basename "$rel_pdf")")"
		cat >> "$tmp_file" <<EOF
    <li><a href="${rel_pdf}">${html_name}</a></li>
EOF
	done < <(cd "$run_dir" && find diagnostics_out -type f -name '*.pdf' 2>/dev/null | sort)

	cat >> "$tmp_file" <<'EOF'
  </ul>
</body>
</html>
EOF
}

render_root_index() {
	local root_file="$PAGES_DIR/$PROJECT_SUBDIR/index.html"
	local html_title
	html_title="$(escape_html "$SITE_TITLE")"
	mkdir -p "$(dirname "$root_file")"

	cat > "$root_file" <<EOF
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>${html_title}</title>
  <style>
    :root { color-scheme: light dark; }
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 2rem auto; max-width: 1000px; padding: 0 1rem; line-height: 1.5; }
    .card { border: 1px solid #8884; border-radius: 10px; padding: 1rem; background: #8881; margin-bottom: 1rem; }
    code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  </style>
</head>
<body>
  <h1>${html_title}</h1>
  <p>This site publishes curated static diagnostics from selected GMRT 40_014 workflow runs. Reproducibility lives on the source branch; this site only mirrors approved outputs.</p>
  <div class="card">
    <p><strong>Latest published run:</strong> <a href="${LATEST_ALIAS}/index.html">${LATEST_ALIAS}</a></p>
  </div>
  <h2>Published runs</h2>
  <ul>
EOF

	while IFS= read -r run_dir_name; do
		local html_name
		html_name="$(escape_html "$run_dir_name")"
		cat >> "$root_file" <<EOF
    <li><a href="runs/${run_dir_name}/index.html"><code>${html_name}</code></a></li>
EOF
	done < <(find "$PAGES_DIR/$PROJECT_SUBDIR/runs" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; 2>/dev/null | sort -r)

	cat >> "$root_file" <<'EOF'
  </ul>
</body>
</html>
EOF
}

publish_run() {
	local source_commit source_branch published_at run_dir run_rel workflow_log manifest_name workflow_log_name

	source_commit="$(git -C "$REPO_ROOT" rev-parse HEAD)"
	source_branch="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)"
	published_at="$(date '+%Y-%m-%d %H:%M:%S %Z')"
	run_rel="${PROJECT_SUBDIR}/${LATEST_ALIAS}"
	run_dir="$PAGES_DIR/$run_rel"

	collect_publish_files
	if [[ -n "$GHPAGES_LAYOUT_MANIFEST" && -f "$GHPAGES_LAYOUT_MANIFEST" ]]; then
		PUBLISH_FILES+=("$GHPAGES_LAYOUT_MANIFEST")
	fi
	workflow_log="$(awk '/^Workflow log:/{getline; print; exit}' "$MANIFEST_PATH")"
	manifest_name="$(basename "$MANIFEST_PATH")"
	workflow_log_name=""
	if [[ -n "$workflow_log" && -f "$workflow_log" ]]; then
		workflow_log_name="$(basename "$workflow_log")"
	fi

	if [[ "$DRY_RUN" -eq 1 ]]; then
		log "DRY-RUN would publish run $RUN_TS to $run_rel (latest only)" >&2
	else
		rm -rf "$run_dir"
		mkdir -p "$run_dir"
	fi

	local src rel dst
	for src in "${PUBLISH_FILES[@]}"; do
		if [[ "$src" == "$MANIFEST_PATH" || "$(basename "$src")" == "$manifest_name" ]]; then
			dst="$run_dir/logs/$manifest_name"
		elif [[ -n "$workflow_log_name" && "$(basename "$src")" == "$workflow_log_name" ]]; then
			dst="$run_dir/logs/$workflow_log_name"
		elif [[ "$src" == "$WORK_DIR/"* ]]; then
			rel="${src#"$WORK_DIR/"}"
			dst="$run_dir/$rel"
		else
			dst="$run_dir/misc/$(basename "$src")"
		fi
		copy_file "$src" "$dst"
	done

	if [[ "$DRY_RUN" -eq 1 ]]; then
		log "DRY-RUN would publish $run_rel"
		if [[ "$OPEN_AFTER" -eq 1 ]]; then
			log "DRY-RUN would open $PAGES_DIR/$run_rel/index.html in browser"
		else
			log "  (opening disabled via --no-open)"
		fi
		return 0
	fi

	# Publish directly to latest/ (no historical run browsing exposed)
	render_run_index "$run_dir" "$run_rel" "$published_at" "$source_commit" "$source_branch" "$workflow_log_name" "$manifest_name" "../index.html"

	# Create a simple project landing page at the root
	local project_root="$PAGES_DIR/$PROJECT_SUBDIR"
	cat > "$project_root/index.html" <<'EOF'
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GMRT 40_014 Calibration Diagnostics</title>
  <meta http-equiv="refresh" content="0; url=latest/">
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; padding: 2rem; margin: 0; }
    a { color: #0969da; text-decoration: none; }
    a:hover { text-decoration: underline; }
  </style>
</head>
<body>
  <h1>GMRT 40_014 Calibration Diagnostics</h1>
  <p>Redirecting to the <a href="latest/">latest published run</a>...</p>
</body>
</html>
EOF

	git -C "$PAGES_DIR" add .
	if git -C "$PAGES_DIR" diff --cached --quiet; then
		log "no gh-pages changes to commit"
	else
		git -C "$PAGES_DIR" commit -m "Publish GMRT 40_014 run ${RUN_TS}"
		if [[ "$PUSH" -eq 1 ]]; then
			git -C "$PAGES_DIR" push -u origin "$PUBLISH_BRANCH"
		fi
	fi

	if [[ "$OPEN_AFTER" -eq 1 ]]; then
		open "$run_dir/index.html"
	fi
}

while [[ $# -gt 0 ]]; do
	case "$1" in
		--manifest)
			MANIFEST_PATH="$2"
			shift 2
			;;
		--run-ts)
			RUN_TS="$2"
			shift 2
			;;
		--work-dir)
			WORK_DIR="$2"
			shift 2
			;;
		--pages-dir)
			PAGES_DIR="$2"
			shift 2
			;;
		--branch)
			PUBLISH_BRANCH="$2"
			shift 2
			;;
		--latest-alias)
			LATEST_ALIAS="$2"
			shift 2
			;;
		--project-subdir)
			PROJECT_SUBDIR="$2"
			shift 2
			;;
		--site-title)
			SITE_TITLE="$2"
			shift 2
			;;
		--push)
			PUSH=1
			shift
			;;
		--dry-run)
			DRY_RUN=1
			shift
			;;
		--open)
			OPEN_AFTER=1
			shift
			;;
		--no-open)
			OPEN_AFTER=0
			shift
			;;
		-h|--help)
			usage
			exit 0
			;;
		*)
			die "unknown argument: $1"
			;;
	esac
done

resolve_manifest

if ! ensure_moon_layout_manifest; then
	die "Moon layout manifest could not be prepared"
fi

log "repo root: $REPO_ROOT"
log "work dir: $WORK_DIR"
log "manifest: $MANIFEST_PATH"
log "run timestamp: $RUN_TS"
if [[ -n "$GHPAGES_LAYOUT_MANIFEST" ]]; then
	log "moon layout manifest: $GHPAGES_LAYOUT_MANIFEST"
fi
log "pages dir: $PAGES_DIR"
log "publish branch: $PUBLISH_BRANCH"

ensure_pages_worktree
publish_run

if [[ "$DRY_RUN" -eq 1 ]]; then
	log "dry run complete"
else
	log "publish complete"
	log "pages worktree: $PAGES_DIR"
	if [[ "$PUSH" -eq 0 ]]; then
		log "branch committed locally only; rerun with --push to publish remotely"
	fi
fi