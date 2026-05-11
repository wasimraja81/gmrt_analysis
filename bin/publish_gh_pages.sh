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

WORK_DIR="$DEFAULT_WORK_DIR"
PAGES_DIR="$DEFAULT_PAGES_DIR"
PUBLISH_BRANCH="$DEFAULT_BRANCH"
LATEST_ALIAS="$DEFAULT_LATEST_ALIAS"
SITE_TITLE="$DEFAULT_SITE_TITLE"
MANIFEST_PATH=""
RUN_TS=""
PUSH=0
DRY_RUN=0
PUBLISH_FILES=()

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
  --latest-alias NAME  Alias directory for the most recently published run (default: $DEFAULT_LATEST_ALIAS).
  --site-title TEXT    Site title used in generated HTML.
  --push               Push the publication branch after commit.
  --dry-run            Show planned actions without writing files or committing.
  -h, --help           Show this help.

Examples:
  bash bin/publish_gh_pages.sh
  bash bin/publish_gh_pages.sh --run-ts 20260511_114843
  bash bin/publish_gh_pages.sh --manifest "$HOME/DATA/gmrt_40_014/work/logs/run_gmrt_40_014_products_20260511_114843.txt" --push
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
	local clustering_dir="$WORK_DIR/secondary_calibration/clustering"
	local workflow_log=""

	PUBLISH_FILES=()
	while IFS= read -r file_path; do
		[[ -n "$file_path" ]] && PUBLISH_FILES+=("$file_path")
	done < <(
		{
			if [[ -d "$diagnostics_dir" ]]; then
				find "$diagnostics_dir" -type f \( -name '*.png' -o -name '*.pdf' \)
			fi
			if [[ -d "$clustering_dir" ]]; then
				find "$clustering_dir" -maxdepth 1 -type f \( -name '*.png' -o -name '*.pdf' \)
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

render_run_index() {
	local run_dir="$1"
	local run_rel="$2"
	local published_at="$3"
	local source_commit="$4"
	local source_branch="$5"
	local workflow_log_name="$6"
	local manifest_name="$7"
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
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 2rem auto; max-width: 1100px; padding: 0 1rem; line-height: 1.5; }
    h1, h2 { line-height: 1.2; }
    .meta { padding: 1rem; border: 1px solid #8884; border-radius: 10px; background: #8881; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 1rem; }
    .card { border: 1px solid #8884; border-radius: 10px; padding: 0.8rem; background: #8881; }
    img { max-width: 100%; height: auto; border-radius: 6px; display: block; }
    a { text-decoration: none; }
    code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    ul { padding-left: 1.2rem; }
  </style>
</head>
<body>
  <p><a href="../../index.html">← all published runs</a> · <a href="../../${LATEST_ALIAS}/index.html">latest</a></p>
  <h1>${html_title}</h1>
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

  <h2>Diagnostics</h2>
  <div class="grid">
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
      <p><a href="${rel_png}">PNG</a> · <a href="${rel_pdf}">PDF</a></p>
EOF
		else
			cat >> "$tmp_file" <<EOF
      <p><a href="${rel_png}">PNG</a></p>
EOF
		fi
		cat >> "$tmp_file" <<'EOF'
    </div>
EOF
	done < <(cd "$run_dir" && find diagnostics_out secondary_calibration/clustering -type f -name '*.png' 2>/dev/null | sort)

	cat >> "$tmp_file" <<'EOF'
  </div>

  <h2>PDF products</h2>
  <ul>
EOF

	while IFS= read -r rel_pdf; do
		local html_name
		html_name="$(escape_html "$(basename "$rel_pdf")")"
		cat >> "$tmp_file" <<EOF
    <li><a href="${rel_pdf}">${html_name}</a></li>
EOF
	done < <(cd "$run_dir" && find diagnostics_out secondary_calibration/clustering -type f -name '*.pdf' 2>/dev/null | sort)

	cat >> "$tmp_file" <<'EOF'
  </ul>
</body>
</html>
EOF
}

render_root_index() {
	local root_file="$PAGES_DIR/index.html"
	local html_title
	html_title="$(escape_html "$SITE_TITLE")"

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
	done < <(find "$PAGES_DIR/runs" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; 2>/dev/null | sort -r)

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
	run_rel="runs/${RUN_TS}"
	run_dir="$PAGES_DIR/$run_rel"

	collect_publish_files
	workflow_log="$(awk '/^Workflow log:/{getline; print; exit}' "$MANIFEST_PATH")"
	manifest_name="$(basename "$MANIFEST_PATH")"
	workflow_log_name=""
	if [[ -n "$workflow_log" && -f "$workflow_log" ]]; then
		workflow_log_name="$(basename "$workflow_log")"
	fi

	if [[ "$DRY_RUN" -eq 1 ]]; then
		log "DRY-RUN would publish run $RUN_TS into $run_dir"
	else
		mkdir -p "$run_dir"
		rm -rf "$run_dir/diagnostics_out" "$run_dir/secondary_calibration" "$run_dir/logs"
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
		log "DRY-RUN would refresh $LATEST_ALIAS/ and regenerate site indexes"
		return 0
	fi

	render_run_index "$run_dir" "$run_rel" "$published_at" "$source_commit" "$source_branch" "$workflow_log_name" "$manifest_name"
	rm -rf "$PAGES_DIR/$LATEST_ALIAS"
	mkdir -p "$PAGES_DIR/$LATEST_ALIAS"
	cp -R "$run_dir"/. "$PAGES_DIR/$LATEST_ALIAS/"
	render_root_index

	git -C "$PAGES_DIR" add .
	if git -C "$PAGES_DIR" diff --cached --quiet; then
		log "no gh-pages changes to commit"
		return 0
	fi

	git -C "$PAGES_DIR" commit -m "Publish GMRT 40_014 run ${RUN_TS}"
	if [[ "$PUSH" -eq 1 ]]; then
		git -C "$PAGES_DIR" push -u origin "$PUBLISH_BRANCH"
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
log "repo root: $REPO_ROOT"
log "work dir: $WORK_DIR"
log "manifest: $MANIFEST_PATH"
log "run timestamp: $RUN_TS"
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