"""Bird's-eye trace of each episode: where the robot went, and where the brain thought it went.

    episodes/<label>/trace.png   one episode: the field and walls, the tray, the five toys where
                                 they started (and an arrow if one was moved), the true trunk path
                                 (truth.csv), the brain's path (robotd odometry through the brain's
                                 own start pose and its tray re-anchors, state.jsonl; the brain's
                                 logged poses when there is no state.jsonl, as on the board), and
                                 the moments: falls, ToF stops, mouth closes, the confirmed spot,
                                 the end
    traces.png                   every episode of the run small, side by side, to spot drift fast

The two paths start at the same point; the gap that opens between them is the odometry error the
brain carries (Mac batch 1 2026-10-26: 0.2-0.35 m by the handover, from the stalled physics). The
labels are English so the pictures go into the report as they are.

    python -m microduck_toy.stage3.trace runs/<dir> [--label ab_3000_bus]
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
from pathlib import Path

from microduck_toy.stage1 import metrics
from microduck_toy.stage3 import episode as ep
from microduck_toy.stage3.world import FieldPose

COLOURS = {'bus': '#e0b000', 'henry': '#2e8b57', 'ketchup': '#d62728', 'croc': '#7a8f2a',
           'thomas': '#1f6fb4'}
TOYS = tuple(COLOURS)
FIELD_HALF_M = 1.0              # stage1.json arena.half_size_m
TRAY_M = 0.3                    # stage1.json arena.tray.size_m
MAX_POINTS = 1500


def window(rec: dict) -> tuple[float, float]:
    return rec['start'], rec.get('judge_mark') or rec['end_mark']


def truth_path(rows: list[dict], monos: list[float], t0: float, t1: float) -> list[tuple]:
    i, j = bisect.bisect_left(monos, t0), bisect.bisect_right(monos, t1)
    step = max(1, (j - i) // MAX_POINTS)
    return [(r['x'], r['y']) for r in rows[i:j:step]]


def odometry(state_path: Path, t0: float, t1: float, every: int = 5) -> list[tuple]:
    """(mono, (x, y, yaw)) of robotd's odometry in the window, from the run's state.jsonl."""
    out = []
    if not state_path.exists():
        return out
    with state_path.open() as f:
        for k, line in enumerate(f):
            if k % every:
                continue
            try:
                state = json.loads(line)
                mono = float(state['mono'])
            except (ValueError, KeyError, TypeError):
                continue
            if mono < t0:
                continue
            if mono > t1:
                break
            o = state.get('odom')
            if o:
                out.append((mono, (float(o['position'][0]), float(o['position'][1]), float(o['yaw']))))
    return out


def brain_path(log: list[dict], odom: list[tuple]) -> list[tuple]:
    """The brain's own idea of where it was: odometry through its start pose, shifted by each tray
    re-anchor from its moment on; without odometry, the poses it logged."""
    start = next((e for e in log if e['event'] == 'start'), None)
    if start is None:
        return []
    if not odom:
        return [tuple(e['pose'][:2]) for e in log if isinstance(e.get('pose'), list) and len(e['pose']) >= 2]
    shifts = [(e['mono'], e['reanchor_m']) for e in log if e['event'] == 'tray_look' and e.get('reanchor_m')]
    pose = FieldPose(start['start'], start['odom0'])
    out, k = [], 0
    for mono, o in odom:
        while k < len(shifts) and shifts[k][0] <= mono:
            dx, dy = shifts[k][1]
            pose.start = (pose.start[0] + dx, pose.start[1] + dy, pose.start[2])
            k += 1
        out.append(pose.of_odom(o)[:2])
    return out


def moments(log: list[dict], rows: list[dict], monos: list[float], t0: float, t1: float) -> dict:
    def row_at(t):
        return rows[max(0, min(bisect.bisect_right(monos, t) - 1, len(rows) - 1))]
    falls = [row_at(f['onset']) for f in ep.falls_in(rows, t0, t1)]
    return {
        'falls': [(r['x'], r['y']) for r in falls],
        'tof': [(row_at(e['mono'])['x'], row_at(e['mono'])['y']) for e in log if e['event'] == 'tof_stop'],
        'closes': [(row_at(e['mono'])['tip_x'], row_at(e['mono'])['tip_y']) for e in log if e['event'] == 'pick_close'],
        'confirmed': [tuple(e['spot']['xy']) for e in log if e['event'] == 'confirmed' and e.get('spot', {}).get('xy')],
        'end': next(({'end': e.get('end'), 'elapsed_s': e.get('elapsed_s'), 'closes': e.get('closes')}
                     for e in log if e['event'] == 'end'), None),
    }


def draw(ax, rec: dict, rows: list[dict], monos: list[float], log: list[dict], odom: list[tuple],
         tray_centre, small: bool = False) -> None:
    from matplotlib.patches import Rectangle
    t0, t1 = window(rec)
    i0 = max(0, min(bisect.bisect_left(monos, t0), len(rows) - 1))
    i1 = max(0, min(bisect.bisect_right(monos, t1) - 1, len(rows) - 1))
    first, last = rows[i0], rows[i1]
    h = FIELD_HALF_M
    ax.add_patch(Rectangle((-h, -h), 2 * h, 2 * h, fill=False, lw=1.5, ec='#555555'))
    if tray_centre:
        cx, cy = tray_centre
        ax.add_patch(Rectangle((cx - TRAY_M / 2, cy - TRAY_M / 2), TRAY_M, TRAY_M, fc='#d9d9d9', ec='#777777'))
    target = rec.get('target')
    for toy in TOYS:
        x, y = first[f'{toy}_x'], first[f'{toy}_y']
        if abs(x) > h or abs(y) > h:
            continue                                  # parked outside the field
        big = toy == target
        ax.plot(x, y, 'o', ms=(9 if big else 6) * (0.6 if small else 1), color=COLOURS[toy],
                mec='black' if big else COLOURS[toy], mew=1.5 if big else 0.5, zorder=5)
        xe, ye = last[f'{toy}_x'], last[f'{toy}_y']
        if math.hypot(xe - x, ye - y) > 0.03:
            ax.annotate('', (xe, ye), (x, y), arrowprops={'arrowstyle': '->', 'color': COLOURS[toy],
                                                         'lw': 1.0, 'ls': ':'}, zorder=4)
        if not small:
            ax.text(x + 0.04, y + 0.04, toy + (' (target)' if big else ''), fontsize=7, color=COLOURS[toy])
    true = truth_path(rows, monos, t0, t1)
    if true:
        ax.plot(*zip(*true), '-', color='#222222', lw=1.4 if not small else 0.9, label='true path', zorder=3)
        ax.plot(*true[0], '^', color='#222222', ms=7 if not small else 4, zorder=6)
    believed = brain_path(log, odom)
    if believed:
        ax.plot(*zip(*believed), '--', color='#2a62c9', lw=1.2 if not small else 0.8,
                label="brain's path", zorder=3)
    m = moments(log, rows, monos, t0, t1)
    for key, style, label in (('falls', dict(marker='X', color='#c00000', ms=10), 'fall'),
                              ('tof', dict(marker='s', color='#ff8c00', ms=6, mfc='none'), 'ToF stop'),
                              ('closes', dict(marker='+', color='black', ms=8, mew=1.5), 'mouth close'),
                              ('confirmed', dict(marker='o', color=COLOURS.get(target, 'k'), ms=12, mfc='none', mew=1.2),
                               'confirmed spot (brain)')):
        pts = m[key]
        if pts:
            if small:
                style = {**style, 'ms': style['ms'] * 0.6}
            ax.plot(*zip(*pts), ls='none', label=label, zorder=7, **style)
    ax.plot(last['x'], last['y'], 's', color='#222222', ms=6 if not small else 3, zorder=6)
    end = m['end'] or {}
    title = f"{rec['label']}  {end.get('end') or '-'}"
    if not small:
        title += f"  {end.get('elapsed_s') or '-'} s  closes {end.get('closes') or 0}  falls {len(m['falls'])}"
        if believed and true:
            title += f"\nend gap brain vs truth {math.dist(believed[-1], true[-1]):.2f} m"
    ax.set_title(title, fontsize=6 if small else 9)
    ax.set_xlim(-h - 0.05, h + 0.05)
    ax.set_ylim(-h - 0.05, h + 0.05)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=5 if small else 7)
    if not small:
        ax.set_xlabel('x (m)', fontsize=8)
        ax.set_ylabel('y (m)', fontsize=8)
        ax.legend(fontsize=6.5, loc='lower left', framealpha=0.85)


def plot_run(run_dir: Path, label: str | None = None, rows: list[dict] | None = None) -> list[Path]:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run_dir = Path(run_dir)
    trials = [json.loads(line) for line in (run_dir / 'trials.jsonl').read_text().splitlines() if line.strip()]
    episodes = [t for t in trials if 'label' in t and t.get('start') is not None
                and t.get('end_mark') is not None and (label is None or t['label'] == label)]
    if not episodes:
        return []
    meta = json.loads((run_dir / 'summary.json').read_text()).get('meta', {}) \
        if (run_dir / 'summary.json').exists() else {}
    tray = meta.get('tray_centre')
    rows = rows if rows is not None else metrics.read_truth(run_dir / 'truth.csv')
    monos = [r['mono'] for r in rows]
    state_path = run_dir / 'state.jsonl'
    drawn, out = [], []
    for rec in episodes:
        log = ep.read_brain_log(run_dir, rec)
        odom = [] if rec.get('offset') else odometry(state_path, *window(rec))
        fig, ax = plt.subplots(figsize=(5.2, 5.4), dpi=130)
        draw(ax, rec, rows, monos, log, odom, tray)
        fig.tight_layout()
        path = run_dir / 'episodes' / rec['label'] / 'trace.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
        out.append(path)
        drawn.append((rec, log, odom))
    if label is None and len(drawn) > 1:
        cols = min(6, len(drawn))
        lines = math.ceil(len(drawn) / cols)
        fig, axes = plt.subplots(lines, cols, figsize=(2.3 * cols, 2.45 * lines), dpi=110, squeeze=False)
        for ax in axes.flat[len(drawn):]:
            ax.axis('off')
        for ax, (rec, log, odom) in zip(axes.flat, drawn):
            draw(ax, rec, rows, monos, log, odom, tray, small=True)
        fig.suptitle(f'{run_dir.name}: true path (black) vs brain path (blue dashed)', fontsize=9)
        fig.tight_layout()
        path = run_dir / 'traces.png'
        fig.savefig(path)
        plt.close(fig)
        out.append(path)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('run', type=Path)
    p.add_argument('--label', default=None, help='one episode only')
    a = p.parse_args(argv)
    paths = plot_run(a.run, a.label)
    print(f'俯视轨迹图 {len(paths)} 张' + (f'；总览：{paths[-1]}' if paths and a.label is None else ''))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
