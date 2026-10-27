"""Closed-loop vision check: every box YOLO gave the brain while searching, against the truth.

Stage 2 measured the detector offline (rendered test sets: mAP50, search radius 1.8 m, zero
confusion within 1.6 m). This asks the same of the frames the brain actually used in Stage 3: the
spin frames and the standing looks (brain.jsonl `spin_frame` and `look` events log every box with
its class and its ground point in the level trunk frame).

    per frame   the truth row at the frame's moment; each toy on the field, flat on the floor
                (z < 0.06, not held), within the search radius, inside the +-22.5 deg picture and
                above the picture's bottom edge (episode.in_picture) is "visible"
    per box     matched to a toy by where it puts it relative to the robot (so odometry drift
                does not count). A standing look: bearing within LOOK_BEARING_TOL_DEG and range
                within max(RANGE_TOL_M, RANGE_TOL_FRAC x true range). A spin frame: bearing only,
                within SPIN_BEARING_TOL_DEG (the brain uses spin frames for bearings; their ground
                points are rough while turning, Mac 3ab: |range error| p50 0.19 m at 1.0-1.4 m,
                p90 1.4 m at 1.4-1.8 m, reported as spin_range_err_m). Same class = correct; only
                another visible toy there = wrong class (true -> said); nothing = phantom
    misses      visible toys with no correct box in that frame

Stream frames are dated by their first byte, so a frame taken while turning is a little older than
its date and its bearings lead the truth by wz x lag (Mac 3ab before the camera fix: +5.0 deg
median, spin_bearing_err_deg). Numbers only, no verdict: they sit beside Stage 2's offline ones.

    python -m microduck_toy.stage3.vision_loop runs/<dir>     (a finished run, printed as a table)
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
from pathlib import Path

from microduck_toy.stage1 import metrics
from microduck_toy.stage3 import episode as ep
from microduck_toy.stage3.world import wrap

TOYS = ('bus', 'henry', 'ketchup', 'croc', 'thomas')
TARGETS = ('bus', 'henry', 'ketchup')
RADIUS_M = 1.8              # configs/vision/stage2_facts.json search radius
HALF_FOV_DEG = 22.5         # the 360-px axis of the 45 deg picture
EDGE_MARGIN_DEG = 1.5
FIELD_HALF_M = 1.0          # stage1.json arena.half_size_m: parked toys are outside it
LOOK_BEARING_TOL_DEG = 10.0
SPIN_BEARING_TOL_DEG = 12.0
RANGE_TOL_M = 0.15
RANGE_TOL_FRAC = 0.3
BINS = ((0.0, 0.6), (0.6, 1.0), (1.0, 1.4), (1.4, RADIUS_M))


class Truth:
    """truth.csv rows looked up by time (bisect; metrics.at_mark scans the whole run)."""

    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.monos = [r['mono'] for r in rows]

    def at(self, mono: float) -> dict:
        i = bisect.bisect_right(self.monos, mono) - 1
        return self.rows[max(0, min(i, len(self.rows) - 1))]


def relative(row: dict, toy: str) -> tuple[float, float]:
    """(range m, bearing rad) of the toy from the trunk, by the truth."""
    dx, dy = row[f'{toy}_x'] - row['x'], row[f'{toy}_y'] - row['y']
    return math.hypot(dx, dy), wrap(math.atan2(dy, dx) - row['yaw'])


def visible(row: dict, toy: str, view: str) -> bool:
    if row[f'{toy}_z'] >= 0.06 or abs(row[f'{toy}_x']) >= FIELD_HALF_M or abs(row[f'{toy}_y']) >= FIELD_HALF_M:
        return False
    rng, brg = relative(row, toy)
    if rng > RADIUS_M or abs(math.degrees(brg)) > HALF_FOV_DEG - EDGE_MARGIN_DEG:
        return False
    return ep.in_picture(row, toy, view)


def near(box: dict, rng: float, brg: float, spin: bool) -> bool:
    off = abs(math.degrees(wrap(box['bearing'] - brg)))
    if spin:
        return off <= SPIN_BEARING_TOL_DEG
    return off <= LOOK_BEARING_TOL_DEG and abs(box['dist'] - rng) <= max(RANGE_TOL_M, RANGE_TOL_FRAC * rng)


def judge_frame(row: dict, seen: list[dict], view: str, spin: bool = False) -> dict:
    """One frame: what was visible, and what each toy box was."""
    toys = {t: relative(row, t) for t in TOYS}
    shown = [t for t in TOYS if visible(row, t, view)]
    boxes, found = [], set()
    for box in seen:
        if box.get('cls') not in TOYS:
            continue
        if box.get('dist') is None or box.get('bearing') is None:
            boxes.append({'cls': box['cls'], 'kind': 'unlocated'})
            continue
        own = toys[box['cls']]
        if near(box, *own, spin):
            kind = 'duplicate' if box['cls'] in found else 'correct'
            found.add(box['cls'])
            boxes.append({'cls': box['cls'], 'kind': kind, 'dist': own[0],
                          'bearing_err_deg': round(math.degrees(wrap(box['bearing'] - own[1])), 2),
                          'range_err_m': round(box['dist'] - own[0], 3)})
            continue
        here = [t for t in shown if t != box['cls'] and near(box, *toys[t], spin)]
        if here:
            true = min(here, key=lambda t: abs(wrap(box['bearing'] - toys[t][1])))
            boxes.append({'cls': box['cls'], 'kind': 'wrong_class', 'true': true, 'dist': toys[true][0]})
        else:
            boxes.append({'cls': box['cls'], 'kind': 'phantom', 'dist': box['dist']})
    return {'visible': {t: round(toys[t][0], 3) for t in shown}, 'found': sorted(found & set(shown)),
            'boxes': boxes}


def frames_of(log: list[dict]) -> list[dict]:
    """(kind, moment, view, boxes) of every search frame the brain logged."""
    out = []
    for e in log:
        if e['event'] == 'spin_frame':
            out.append({'kind': 'spin', 'mono': e.get('mono_frame') or e['mono'],
                        'view': e.get('view') or 'window', 'seen': e.get('seen') or []})
        elif e['event'] == 'look':
            # standing: frames logged before the look's own stamp carry it if they have none
            for f in e.get('frames_seen') or []:
                out.append({'kind': 'look', 'mono': f.get('mono') or e['mono'],
                            'view': e.get('view') or 'window', 'seen': f.get('seen') or []})
    return out


def judge_run(run_dir: Path, episodes: list[dict], rows: list[dict]) -> list[dict]:
    truth = Truth(rows)
    out = []
    for rec in episodes:
        for f in frames_of(ep.read_brain_log(run_dir, rec)):
            j = judge_frame(truth.at(f['mono']), f['seen'], f['view'], f['kind'] == 'spin')
            out.append({'label': rec['label'], 'kind': f['kind'], **j})
    return out


def rate(k: int, n: int) -> dict:
    return {'k': k, 'n': n, 'rate': round(k / n, 3) if n else None}


def summarise(judged: list[dict]) -> dict:
    if not judged:
        return {'frames': 0}
    counts = {'spin': sum(f['kind'] == 'spin' for f in judged),
              'look': sum(f['kind'] == 'look' for f in judged)}
    boxes = [b for f in judged for b in f['boxes']]

    def recall(cls_set, lo=0.0, hi=RADIUS_M):
        vis = [(t, d) for f in judged for t, d in f['visible'].items() if t in cls_set and lo <= d < hi + 1e-9]
        hit = [(t, d) for f in judged for t, d in f['visible'].items()
               if t in cls_set and lo <= d < hi + 1e-9 and t in f['found']]
        return rate(len(hit), len(vis))

    def precision(cls_set, lo=0.0, hi=math.inf):
        mine = [b for b in boxes if b['cls'] in cls_set and b['kind'] != 'unlocated'
                and lo <= b['dist'] < hi + 1e-9]
        return rate(sum(b['kind'] == 'correct' for b in mine), len(mine))

    spin_ok = [b for f in judged if f['kind'] == 'spin' for b in f['boxes'] if b['kind'] == 'correct']
    bearing = sorted(b['bearing_err_deg'] for b in spin_ok)

    def range_err(lo, hi):
        errs = sorted(abs(b['range_err_m']) for b in spin_ok if lo <= b['dist'] < hi + 1e-9)
        return {'m': f'{lo:.1f}-{hi:.1f}', 'n': len(errs), 'abs_p50': ep.pct(errs, 0.5), 'abs_p90': ep.pct(errs, 0.9)}

    wrong: dict[str, int] = {}
    for b in boxes:
        if b['kind'] == 'wrong_class':
            key = f"{b['true']}->{b['cls']}"
            wrong[key] = wrong.get(key, 0) + 1
    return {
        'frames': counts,
        'recall_targets': recall(TARGETS), 'precision_targets': precision(TARGETS, hi=RADIUS_M),
        'recall_all': recall(TOYS), 'precision_all': precision(TOYS, hi=RADIUS_M),
        'by_class': {t: {'recall': recall((t,)), 'precision': precision((t,), hi=RADIUS_M)} for t in TOYS},
        'by_distance': [{'m': f'{lo:.1f}-{hi:.1f}', 'recall': recall(TOYS, lo, hi),
                         'precision': precision(TOYS, lo, hi)} for lo, hi in BINS],
        'wrong_class': dict(sorted(wrong.items(), key=lambda kv: -kv[1])),
        'wrong_class_within_1_6m': sum(1 for b in boxes if b['kind'] == 'wrong_class' and b['dist'] <= 1.6),
        'phantom': sum(b['kind'] == 'phantom' for b in boxes),
        'beyond_radius_boxes': sum(1 for b in boxes if b['kind'] != 'unlocated' and b['dist'] > RADIUS_M),
        'duplicates': sum(b['kind'] == 'duplicate' for b in boxes),
        'unlocated': sum(b['kind'] == 'unlocated' for b in boxes),
        'spin_bearing_err_deg': {'n': len(bearing), 'p50': ep.pct(bearing, 0.5),
                                 'abs_p90': ep.pct(sorted(abs(v) for v in bearing), 0.9)},
        'spin_range_err_m': [range_err(lo, hi) for lo, hi in BINS],
        'tolerance': {'look': f'bearing {LOOK_BEARING_TOL_DEG} deg, range max({RANGE_TOL_M} m, '
                              f'{RANGE_TOL_FRAC} x range)', 'spin': f'bearing {SPIN_BEARING_TOL_DEG} deg',
                      'visible': f'on the floor, inside the field, <= {RADIUS_M} m, |bearing| <= '
                                 f'{HALF_FOV_DEG - EDGE_MARGIN_DEG} deg, above the picture bottom'},
        'note': 'spin and standing-look frames only (the B segment logs no per-frame boxes); '
                'report numbers beside Stage 2 offline, no verdict',
    }


def check_run(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    trials = [json.loads(line) for line in (run_dir / 'trials.jsonl').read_text().splitlines() if line.strip()]
    episodes = [t for t in trials if not t['part'].startswith('m_') and 'label' in t]
    rows = metrics.read_truth(run_dir / 'truth.csv')
    return summarise(judge_run(run_dir, episodes, rows))


def print_table(s: dict) -> None:
    if not s.get('frames'):
        print('没有转圈或站立看的帧。')
        return
    f = s['frames']
    print(f"闭环视觉：转圈 {f['spin']} 帧，站着看 {f['look']} 帧（框与真值对比；转圈只比方位 ±{SPIN_BEARING_TOL_DEG:.0f}°，"
          f"站着看比方位 ±{LOOK_BEARING_TOL_DEG:.0f}° 和距离）")

    def r(x):
        return '—' if x['rate'] is None else f"{x['rate']:.1%}（{x['k']}/{x['n']}）"
    print(f"  目标玩具：召回 {r(s['recall_targets'])}，精确 {r(s['precision_targets'])}")
    print(f"  全部玩具：召回 {r(s['recall_all'])}，精确 {r(s['precision_all'])}")
    for row in s['by_distance']:
        print(f"  {row['m']} m：召回 {r(row['recall'])}，精确 {r(row['precision'])}")
    for t, v in s['by_class'].items():
        print(f"  {t:8s} 召回 {r(v['recall'])}，精确 {r(v['precision'])}")
    print(f"  认错类：{s['wrong_class'] or '无'}（1.6 m 内 {s['wrong_class_within_1_6m']} 个）；"
          f"凭空多框 {s['phantom']}；重复框 {s['duplicates']}")
    b = s['spin_bearing_err_deg']
    if b['n']:
        print(f"  转圈方位误差：中位 {b['p50']:+.1f}°（正 = 画面比机器人状态晚），|误差| p90 {b['abs_p90']:.1f}°")
        print('  转圈测距 |误差| 中位/p90：' + '；'.join(
            f"{r['m']} m {r['abs_p50']}/{r['abs_p90']} m" for r in s['spin_range_err_m'] if r['n']))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('run', type=Path)
    a = p.parse_args(argv)
    s = check_run(a.run)
    print_table(s)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
