"""Stage 0 scoring from probe truth rows (truth.csv). Pure functions; no robot, no sockets.

Times are the probe's `mono` column (time.monotonic() on the Mac), the same clock the test
scripts write into events.jsonl. Standing follows microduck-lab's bench_getup gate
(behaviors/getup.py `_getup_hold_raw`, Apache-2.0): both feet down, nothing else on the
floor, trunk upright (gravity z < -0.85) and tall (z >= STAND z - 0.045); a "full" stand also
has the jaw at >= 80 % of its STAND keyframe height.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import statistics

# truth.csv, written by sim/probe_body.py. cam_pitch is the head camera's depression below the
# horizon (rad, positive = looking down); tip_* is the MJCF mouth_tip site, jaw_z the jaw_soft body.
COLUMNS = (
    't', 'mono', 'x', 'y', 'z', 'yaw', 'gx', 'gy', 'gz', 'vx', 'vy', 'vz', 'wz',
    'tip_x', 'tip_y', 'tip_z', 'jaw_z', 'cam_x', 'cam_y', 'cam_z', 'cam_pitch',
    'neck_q', 'head_q', 'lfoot', 'rfoot', 'other_floor', 'torque', 'kp', 'released', 'mark',
)
FALL_GZ = -0.5          # robotd's fall_gravity_z
FALL_LOW_Z = 0.06       # trunk this low is on the floor whatever the gravity says
FALL_HOLD_S = 0.2       # robotd's fall_debounce_ms
UP_GZ = -0.85           # lab STAND_GZ
STAND_DROP = 0.045      # lab STAND_DROP
HEAD_UP_FRAC = 0.80     # lab HEAD_UP_FRAC
INT_COLUMNS = ('lfoot', 'rfoot', 'other_floor', 'torque', 'released')


def read_truth(path: Path) -> list[dict]:
    rows = []
    with Path(path).open(newline='') as handle:
        for raw in csv.DictReader(handle):
            row = {}
            for key, value in raw.items():
                if key == 'mark':
                    row[key] = value or ''
                elif key in INT_COLUMNS:
                    row[key] = int(value)
                else:
                    row[key] = float(value)
            rows.append(row)
    return rows


def wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def upright(row: dict, refs: dict) -> bool:
    return (row['gz'] < UP_GZ and row['z'] >= refs['stand_trunk_z'] - STAND_DROP
            and row['other_floor'] == 0)


def standing(row: dict, refs: dict, full: bool = False) -> bool:
    if not (row['lfoot'] and row['rfoot'] and upright(row, refs)):
        return False
    return not full or row['jaw_z'] >= HEAD_UP_FRAC * refs['stand_jaw_z']


def down(row: dict) -> bool:
    return row['gz'] > FALL_GZ or row['z'] < FALL_LOW_Z


def window(rows: list[dict], start: float, end: float = math.inf) -> list[dict]:
    return [r for r in rows if start <= r['mono'] <= end]


def first_hold(rows: list[dict], predicate, start: float, hold_s: float,
               end: float = math.inf) -> float | None:
    """Start time of the first run of rows (from `start`) where predicate holds for hold_s."""
    run_start = None
    for row in window(rows, start, end):
        if predicate(row):
            if run_start is None:
                run_start = row['mono']
            if row['mono'] - run_start >= hold_s - 1e-9:
                return run_start
        else:
            run_start = None
    return None


def fall_time(rows: list[dict], start: float, end: float = math.inf) -> float | None:
    """Onset of the first fall (down for FALL_HOLD_S) at or after start."""
    return first_hold(rows, down, start, FALL_HOLD_S, end)


def holds(rows: list[dict], predicate, start: float, duration: float,
          gap_s: float = 0.1) -> bool:
    """Every row in [start, start + duration] passes, and the rows cover the whole span."""
    span = window(rows, start, start + duration)
    if not span or span[0]['mono'] - start > gap_s or start + duration - span[-1]['mono'] > gap_s:
        return False
    times = [r['mono'] for r in span]
    if any(b - a > gap_s for a, b in zip(times, times[1:])):
        return False
    return all(predicate(r) for r in span)


def at(rows: list[dict], t: float) -> dict:
    """The first row at or after t (the last row if t is past the end)."""
    for row in rows:
        if row['mono'] >= t:
            return row
    if not rows:
        raise ValueError('no rows')
    return rows[-1]


def relative(origin: dict, row: dict) -> dict:
    """row's trunk pose in origin's heading frame: forward, left (m) and yaw change (rad)."""
    dx, dy = row['x'] - origin['x'], row['y'] - origin['y']
    c, s = math.cos(origin['yaw']), math.sin(origin['yaw'])
    return {'forward': c * dx + s * dy, 'left': -s * dx + c * dy,
            'dyaw': wrap(row['yaw'] - origin['yaw'])}


def point_in(origin: dict, x: float, y: float) -> tuple[float, float]:
    """A world point in origin's trunk heading frame (forward, left), metres."""
    dx, dy = x - origin['x'], y - origin['y']
    c, s = math.cos(origin['yaw']), math.sin(origin['yaw'])
    return c * dx + s * dy, -s * dx + c * dy


def unwrapped_yaw(rows: list[dict]) -> list[float]:
    out, total = [], 0.0
    for i, row in enumerate(rows):
        if i:
            total += wrap(row['yaw'] - rows[i - 1]['yaw'])
        out.append(total)
    return out


def settle_time(rows: list[dict], start: float, window_s: float = 0.5,
                max_move: float = 0.003, max_turn: float = math.radians(0.5),
                end: float = math.inf) -> float | None:
    """Start of the first window_s-long stretch (from `start`) in which the trunk moved <
    max_move and turned < max_turn: the moment it was still. None if never before `end`."""
    span = window(rows, start, end)
    yaws = unwrapped_yaw(span)
    j = 0
    for i, row in enumerate(span):
        if row['mono'] - start < window_s:
            continue
        while span[j]['mono'] < row['mono'] - window_s:
            j += 1
        piece = span[j:i + 1]
        x0, y0 = piece[0]['x'], piece[0]['y']
        moved = max(math.hypot(r['x'] - x0, r['y'] - y0) for r in piece)
        turned = max(yaws[j:i + 1]) - min(yaws[j:i + 1])
        if moved < max_move and turned < max_turn:
            return piece[0]['mono']
    return None


def continuity(rows: list[dict]) -> dict:
    """Is truth.csv complete? One row per 20 ms batch of sim time, and wall-clock gaps."""
    sim_gaps = [b['t'] - a['t'] for a, b in zip(rows, rows[1:])]
    wall_gaps = [b['mono'] - a['mono'] for a, b in zip(rows, rows[1:])]
    return {'rows': len(rows),
            'span_s': rows[-1]['mono'] - rows[0]['mono'] if rows else 0.0,
            'missing_batches': sum(round(g / 0.02) - 1 for g in sim_gaps if g > 0.03),
            'max_wall_gap_s': max(wall_gaps, default=0.0)}


def summary(values: list[float]) -> dict:
    values = [v for v in values if v is not None and math.isfinite(v)]
    if not values:
        return {'n': 0}
    return {'n': len(values), 'mean': statistics.fmean(values),
            'sd': statistics.stdev(values) if len(values) > 1 else 0.0,
            'min': min(values), 'max': max(values)}


def mark_times(rows: list[dict], prefix: str = '') -> dict[str, float]:
    """mark label -> mono of the row that carries it (labels joined with '|')."""
    found = {}
    for row in rows:
        for label in filter(None, row['mark'].split('|')):
            if label.startswith(prefix) and label not in found:
                found[label] = row['mono']
    return found


def read_states(path: Path) -> list[dict]:
    """state.jsonl reduced to what scoring needs: mono, daemon t, policy label, fallen."""
    states = []
    with Path(path).open() as handle:
        for line in handle:
            state = json.loads(line)
            states.append({'mono': state['mono'], 't': state['t'], 'policy': state['policy'],
                           'fallen': state['safety']['fallen'], 'limp': state['safety']['limp']})
    return states


def label_spans(states: list[dict]) -> list[dict]:
    """Consecutive runs of one policy label: [{'policy', 'start', 'end'}]."""
    spans = []
    for state in states:
        if spans and spans[-1]['policy'] == state['policy']:
            spans[-1]['end'] = state['mono']
        else:
            spans.append({'policy': state['policy'], 'start': state['mono'], 'end': state['mono']})
    return spans


def clean(value):
    """JSON-safe copy: non-finite floats become None."""
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
