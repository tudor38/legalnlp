"""
Config loader for the legalnlp app.

The TOML config lives at the project root: config/app.toml
It is loaded once at import time and exposed as CFG.
"""

import tomllib
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "app.toml"


def _load() -> dict:
    with open(_CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


CFG: dict = _load()
