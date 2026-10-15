"""Stage 1 scoring helpers: the Stage 0 ones, plus reading the arena body's truth.csv.

truth.csv (sim/arena_body.py) = Stage 0's columns + mouth_open (0..1 target), mouth_q (the angle
reported to robotd), holding (toy short name or empty) and x/y/z/yaw for every toy
(bus, henry, ketchup, croc, thomas).
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

from microduck_toy.stage0.metrics import (  # noqa: F401  (re-exported for the Stage 1 scripts)
    FALL_HOLD_S, at, clean, continuity, down, fall_time, first_hold, label_spans, point_in,
    read_states, relative, settle_time, standing, summary, window, wrap,
)

TEXT_COLUMNS = ('mark', 'holding')
INT_COLUMNS = ('lfoot', 'rfoot', 'other_floor', 'torque', 'released')


def read_truth(path: Path) -> list[dict]:
    rows = []
    with Path(path).open(newline='') as handle:
        for raw in csv.DictReader(handle):
            row = {}
            for key, value in raw.items():
                if key in TEXT_COLUMNS:
                    row[key] = value or ''
                elif key in INT_COLUMNS:
                    row[key] = int(value)
                else:
                    row[key] = float(value)
            rows.append(row)
    return rows


def at_mark(rows: list[dict], t: float) -> dict:
    """The last row at or before t: the world as it was when a mark was stamped. Use this, not
    at(), for a mark that ends a trial: the next trial's reset can teleport the robot a few
    milliseconds later, and the first row after the mark may already show it."""
    found = None
    for row in rows:
        if row['mono'] > t:
            break
        found = row
    if found is None:
        if not rows:
            raise ValueError('no rows')
        return rows[0]
    return found


def toy_in_trunk(row: dict, toy: str) -> tuple[float, float]:
    """A toy's (forward, left) in the trunk's heading frame at this row, metres."""
    return point_in(row, row[f'{toy}_x'], row[f'{toy}_y'])


def displacement(a: dict, b: dict, toy: str) -> float:
    return math.dist((a[f'{toy}_x'], a[f'{toy}_y'], a[f'{toy}_z']),
                     (b[f'{toy}_x'], b[f'{toy}_y'], b[f'{toy}_z']))


def rate(values: list[bool]) -> dict:
    n = len(values)
    k = sum(1 for v in values if v)
    return {'k': k, 'n': n, 'rate': (k / n) if n else None}
