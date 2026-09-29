"""Install the fonts visplot draws with into the venv (T41) -- a step of
`bin/build_venv.sh`, and on its own `bin/install_fonts.sh`.

Each font pinned in `config/fonts.txt` is downloaded from its publisher into
the repository's tmp/, checked against its SHA-256, and unpacked unmodified
into `<venv>/share/fonts/<name>/` beside a SOURCE.txt saying where it came
from; the download is then deleted. A font already installed from the same
pin is left as it is. Nothing is installed at system level, and nothing is
taken from the system's fonts.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visplot.fonts import FONTS_CONFIG, REPO_ROOT, installed_dir, read_pins  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install(config: Path, prefix: Path, download_dir: Path) -> list[str]:
    """Install every font in `config` into the venv at `prefix`; returns a
    line per font saying what was done. Raises SystemExit on a checksum
    mismatch (the download is removed, nothing installed)."""
    done = []
    for pin in read_pins(config):
        target = installed_dir(pin.name, prefix)
        stamp = target / "SOURCE.txt"
        if stamp.is_file() and stamp.read_text() == pin.stamp():
            done.append(f"{pin.name} {pin.version}: already installed in {target}")
            continue
        download_dir.mkdir(parents=True, exist_ok=True)
        archive = download_dir / pin.url.rsplit("/", 1)[-1]
        with urllib.request.urlopen(pin.url) as response, open(archive, "wb") as out:
            shutil.copyfileobj(response, out)
        sha256 = _sha256(archive)
        if sha256 != pin.sha256:
            archive.unlink()
            raise SystemExit(f"{pin.name}: {pin.url} has SHA-256 {sha256}; {config} pins {pin.sha256}; "
                             f"nothing installed")
        staging = target.with_name(f".{pin.name}.installing")
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(staging)
        (staging / "SOURCE.txt").write_text(pin.stamp())
        shutil.rmtree(target, ignore_errors=True)
        staging.rename(target)
        archive.unlink()
        done.append(f"{pin.name} {pin.version}: installed in {target} "
                    f"({len([p for p in target.iterdir() if p.name != 'SOURCE.txt'])} files)")
    if download_dir.is_dir() and not any(download_dir.iterdir()):
        download_dir.rmdir()
    return done


def check(config: Path, prefix: Path) -> tuple[bool, list[str]]:
    """Whether every font in `config` is installed from its pin in the venv
    at `prefix`, and a line per font."""
    ok, lines = True, []
    for pin in read_pins(config):
        stamp = installed_dir(pin.name, prefix) / "SOURCE.txt"
        installed = stamp.is_file() and stamp.read_text() == pin.stamp()
        ok &= installed
        lines.append(f"{pin.name} {pin.version}: " + ("installed" if installed else
                                                      "NOT installed from its pin; run bin/build_venv.sh"))
    return ok, lines


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="bin/install_fonts.sh", description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default=str(FONTS_CONFIG), help="the pinned fonts (default: config/fonts.txt)")
    parser.add_argument("--prefix", default=sys.prefix, help="the venv to install into (default: the one running this)")
    parser.add_argument("--download-dir", default=str(REPO_ROOT / "tmp" / "fonts_download"),
                        help="where downloads wait to be checked (default: the repository's tmp/fonts_download)")
    parser.add_argument("--check", action="store_true",
                        help="install nothing: say whether every pinned font is installed (exit status 1 if not)")
    args = parser.parse_args(argv[1:])
    if args.check:
        ok, lines = check(Path(args.config), Path(args.prefix))
        print("\n".join(lines))
        return 0 if ok else 1
    for line in install(Path(args.config), Path(args.prefix), Path(args.download_dir)):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
