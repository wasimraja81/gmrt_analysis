"""The fonts visplot draws the panel under each plot with (T41), each loaded
from a known file -- never looked up among the system's fonts -- so the same
command draws the same pixels on every machine built the same way.

- `tex-gyre-heros` (the default): TeX Gyre Heros, the Helvetica metric clone
  (Helvetica is CERN ROOT's text font), pinned in `config/fonts.txt` and
  installed into the venv (`<venv>/share/fonts/`) from its publisher, GUST,
  as a step of building the venv (`bin/build_venv.sh`).
- `dejavu-sans`: DejaVu Sans, the font matplotlib ships with (its version
  goes with matplotlib's, in `config/requirements.txt`).

A request chooses one with --panel-font, so its command records it.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import matplotlib
from matplotlib.font_manager import FontProperties

REPO_ROOT = Path(__file__).resolve().parents[2]
FONTS_CONFIG = REPO_ROOT / "config" / "fonts.txt"

# --panel-font name -> (family, file: in the venv's share/fonts/<name>/, or None for matplotlib's own)
PANEL_FONTS = {
    "tex-gyre-heros": ("TeX Gyre Heros", "texgyreheros-regular.otf"),
    "dejavu-sans": ("DejaVu Sans", None),
}
DEFAULT_PANEL_FONT = "tex-gyre-heros"


class FontNotInstalled(ValueError):
    """A --panel-font whose file is not in the venv."""


@dataclass(frozen=True)
class FontPin:
    """One line of `config/fonts.txt`."""

    name: str
    version: str
    sha256: str
    url: str

    def stamp(self) -> str:
        """What an installed font records of where it came from."""
        return f"{self.name} {self.version}\n{self.url}\nsha256 {self.sha256}\n"


def read_pins(path=FONTS_CONFIG) -> list[FontPin]:
    pins = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            name, version, sha256, url = line.split()
            pins.append(FontPin(name, version, sha256.lower(), url))
    return pins


def installed_dir(name: str, prefix=None) -> Path:
    """Where the font installer (`src/cli/install_fonts.py`) puts font `name`
    in the venv at `prefix` (default: this Python's)."""
    return Path(prefix or sys.prefix) / "share" / "fonts" / name


def font_file(name: str) -> Path:
    """The file of --panel-font `name`. Raises FontNotInstalled when it is
    not in the venv, saying how to install it."""
    family, filename = PANEL_FONTS[name]
    if filename is None:
        return Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
    path = installed_dir(name) / filename
    if not path.is_file():
        raise FontNotInstalled(f"--panel-font {name}: {family} is not installed in this venv ({path}); run "
                               f"bin/build_venv.sh (it installs config/fonts.txt), or choose --panel-font dejavu-sans")
    return path


@lru_cache(maxsize=64)
def font_properties(name: str, size: float) -> FontProperties:
    return FontProperties(fname=str(font_file(name)), size=size)


@lru_cache(maxsize=16)
def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
