"""holabot_station.utils

Small shared helpers used across the repository.

Design goals:
- Keep dependencies minimal (standard library only).
- Be safe for Windows-first development, while remaining portable to Linux.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


_LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}


def normalize_level(level: str) -> str:
    lvl = (level or "INFO").upper().strip()
    return lvl if lvl in _LEVELS else "INFO"


def log(level: str, msg: str, cfg_level: str = "INFO") -> None:
    """Lightweight logger (stdout)."""

    level = normalize_level(level)
    cfg_level = normalize_level(cfg_level)
    if _LEVELS[level] < _LEVELS[cfg_level]:
        return

    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {level:<7} {msg}", flush=True)


@dataclass
class FpsCounter:
    """Tracks FPS over a moving 1-second window."""

    window_seconds: float = 1.0
    _t0: float = 0.0
    _frames: int = 0
    _fps: float = 0.0

    def __post_init__(self) -> None:
        self._t0 = time.time()

    def tick(self, n: int = 1) -> float:
        """Call once per processed frame; returns current FPS estimate."""

        self._frames += int(n)
        now = time.time()
        dt = now - self._t0
        if dt >= self.window_seconds:
            self._fps = self._frames / max(dt, 1e-9)
            self._frames = 0
            self._t0 = now
        return self._fps

    @property
    def fps(self) -> float:
        return float(self._fps)


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def repo_root_from_src(file_path: Path) -> Path:
    """Assumes this file lives under <repo>/src/holabot_station/*.py."""

    p = file_path.resolve()
    # .../<repo>/src/holabot_station/utils.py -> parents[2] == <repo>
    return p.parents[2]


def with_src_on_pythonpath(repo_root: Path) -> Dict[str, str]:
    """Return a copy of environment with <repo>/src added to PYTHONPATH.

    Useful when spawning child processes without installing the package.
    """

    env = dict(os.environ)
    src_path = str((repo_root / "src").resolve())
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_path + (os.pathsep + existing if existing else "")
    return env


def python_executable() -> str:
    return sys.executable


def is_windows() -> bool:
    return os.name == "nt"
