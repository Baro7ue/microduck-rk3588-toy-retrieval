"""Repository and board paths. Code uses these instead of absolute paths.

Set by env.sh on the Mac (MD_ROOT, MD_RUNS) and by the board launchers (MD_BOARD_ROOT);
the fallbacks below keep a plain checkout working and match the board layout ~/microduck/.
"""
from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    env = os.environ.get('MD_ROOT')
    return Path(env) if env else Path(__file__).resolve().parents[2]


def runs_dir() -> Path:
    env = os.environ.get('MD_RUNS')
    return Path(env) if env else repo_root() / 'runs'


def board_root() -> Path:
    env = os.environ.get('MD_BOARD_ROOT')
    return Path(env) if env else Path.home() / 'microduck'
