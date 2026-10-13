"""Stage 0c: does the robot get back up, and which stand slot should it run?

    python -m microduck_toy.stage0.s0c_getup --config s2

Parts (plan section 3):
  c1  stand 10 s at each head pose H0..H3
  c2  walking at 0.2 m/s with head H1, pushed forward/backward/left/right: a sweep from
      0.6 to 1.5 m/s finds the smallest push that knocks it down, then 3 pushes per direction
  c3  started lying on its back, front and side
  c4  after every get-up: hold 5 s, then walk 3 s at 0.2 m/s with head H1
When a fall is seen (truth, or robotd's safety.fallen) the script zeroes the velocity and
sets the head to H0, as the Stage 3 state machine will.
"""
from __future__ import annotations

import sys
import time

from microduck_toy.stage0 import common, metrics

DIRECTIONS = {'forward': (1.0, 0.0), 'backward': (-1.0, 0.0),
              'left': (0.0, 1.0), 'right': (0.0, -1.0)}
SWEEP = (0.6, 0.75, 0.9, 1.05, 1.2, 1.35, 1.5)
LYING = ('back', 'front', 'side')
HEADS = ('H0', 'H1', 'H2', 'H3')
WALK_VX = 0.2
GETUP_LIMIT_S = 10.0
UP_HOLD_S = 0.5
STAND_KEEP_S = 5.0
WALK_AFTER_S = 3.0
PASS_GETUPS = 13 / 15


def walk_and_push(s, direction: str, speed: float, label: str) -> tuple[float, float | None]:
    s.reset(head='H1')
    s.robot.move(WALK_VX, 0.0, 0.0)
    time.sleep(2.0)
    ux, uy = DIRECTIONS[direction]
    pushed = s.mark(label)
    s.probe.call('push', vx=ux * speed, vy=uy * speed)
    return pushed, s.watch_fall(3.0)


def get_up(s, label: str) -> dict:
    """After a fall: zero the command, wait for a full stand, keep it, then walk again."""
    s.robot.move(0.0, 0.0, 0.0)
    s.robot.head('H0')
    fallen = s.mark(f'{label}_fallen')
    up, run_start = None, None
    while time.monotonic() < fallen + GETUP_LIMIT_S + UP_HOLD_S:
        row = s.probe.truth()
        if metrics.standing(row, s.refs, full=True):
            run_start = run_start or row['mono']
            if row['mono'] - run_start >= UP_HOLD_S:
                up = run_start
                break
        else:
            run_start = None
        time.sleep(0.02)
    times = {'fallen_seen': fallen}
    if up is None:
        times['gave_up'] = s.mark(f'{label}_no_getup')
        return times
    times['up_seen'] = s.mark(f'{label}_up')
    time.sleep(STAND_KEEP_S)
    s.robot.head('H1')
    times['walk_start'] = s.mark(f'{label}_walk')
    s.robot.move(WALK_VX, 0.0, 0.0)
    time.sleep(WALK_AFTER_S)
    s.robot.move(0.0, 0.0, 0.0)
    times['walk_end'] = s.mark(f'{label}_walk_end')
    time.sleep(1.0)
    return times


def c1(s) -> None:
    for head in HEADS:
        s.reset(head=head)
        start = s.mark(f'c1_{head}')
        time.sleep(3.0 if s.args.quick else 10.0)
        s.trial({'part': 'c1', 'head': head, 'start': start, 'end': s.mark(f'c1_{head}_end')})
        print(f'c1 {head} 完成', flush=True)


def c2(s) -> None:
    reps = s.args.reps or (1 if s.args.quick else 3)
    directions = ('forward', 'left') if s.args.quick else tuple(DIRECTIONS)
    for direction in directions:
        threshold = s.args.push
        for speed in () if threshold else SWEEP:
            pushed, fell = walk_and_push(s, direction, speed, f'c2_sweep_{direction}_{speed}')
            s.trial({'part': 'c2_sweep', 'direction': direction, 'speed': speed,
                     'pushed': pushed, 'fell_seen': fell})
            print(f'c2 扫描 {direction} {speed} m/s：{"推倒" if fell else "没倒"}', flush=True)
            if fell:
                threshold = speed
                break
        if threshold is None:
            continue
        for rep in range(1, reps + 1):
            speed = threshold
            while True:
                label = f'c2_{direction}_{rep}_{speed}'
                pushed, fell = walk_and_push(s, direction, speed, label)
                record = {'part': 'c2', 'direction': direction, 'rep': rep, 'speed': speed,
                          'pushed': pushed, 'fell_seen': fell}
                if fell:
                    record.update(get_up(s, label))
                s.trial(record)
                print(f'c2 {direction} 第 {rep} 次 {speed} m/s：'
                      f'{"推倒" if fell else "没倒"}', flush=True)
                if fell or speed >= SWEEP[-1]:
                    break
                speed = round(speed + 0.15, 2)


def c3(s) -> None:
    for kind in LYING:
        s.reset(head='H0')
        posed = s.mark(f'c3_{kind}')
        s.probe.call('pose', kind=kind)
        record = {'part': 'c3', 'kind': kind, 'posed': posed}
        record.update(get_up(s, f'c3_{kind}'))
        s.trial(record)
        print(f'c3 {kind} 完成', flush=True)


PARTS = {'c1': c1, 'c2': c2, 'c3': c3}


def score_getup(rows, refs, record, onset) -> dict:
    """Plan section 3: a full stand within 10 s of the fall, kept 5 s, then walking again."""
    out = {'onset': onset}
    if onset is None:
        return {**out, 'success': False, 'why': 'no fall'}
    stand = lambda r: metrics.standing(r, refs, full=True)  # noqa: E731
    up = metrics.first_hold(rows, stand, onset, UP_HOLD_S, end=onset + GETUP_LIMIT_S + UP_HOLD_S)
    if up is None or up - onset > GETUP_LIMIT_S:
        return {**out, 'success': False, 'why': 'no full stand within 10 s'}
    out['getup_s'] = up - onset
    kept = metrics.holds(rows, lambda r: metrics.upright(r, refs), up, STAND_KEEP_S)
    out['kept_5s'] = kept
    if 'walk_start' not in record:
        return {**out, 'success': False, 'why': 'script did not see the stand'}
    walk = metrics.window(rows, record['walk_start'], record['walk_end'])
    out['walk_m'] = metrics.relative(walk[0], walk[-1])['forward'] if walk else None
    walk_fall = metrics.fall_time(rows, record['walk_start'], record['walk_end'] + 1.0)
    out['walk_fell'] = walk_fall is not None
    out['success'] = bool(kept and walk_fall is None and (out['walk_m'] or 0) > 0.1)
    if not out['success']:
        out['why'] = 'fell again' if walk_fall else ('not kept 5 s' if not kept else 'did not walk')
    return out


def score(s, rows) -> dict:
    states = metrics.read_states(s.dir / 'state.jsonl')
    spans = metrics.label_spans(states)
    refs = s.refs
    result = {'c1': [], 'c2_sweep': [], 'c2': [], 'c3': []}
    for record in s.trials:
        part = record['part']
        if part == 'c1':
            fell = metrics.fall_time(rows, record['start'], record['end'])
            span = metrics.window(rows, record['start'], record['end'])
            result['c1'].append({'head': record['head'], 'fell': fell is not None,
                                 'jaw_z': metrics.summary([r['jaw_z'] for r in span]),
                                 'other_floor_rows': sum(r['other_floor'] > 0 for r in span)})
        elif part == 'c2_sweep':
            fell = metrics.fall_time(rows, record['pushed'], record['pushed'] + 3.5)
            result['c2_sweep'].append({k: record[k] for k in ('direction', 'speed')}
                                      | {'fell': fell is not None})
        else:
            if part == 'c2':
                onset = metrics.fall_time(rows, record['pushed'], record['pushed'] + 3.5)
                if onset is None and record.get('fell_seen') is None:
                    result['c2'].append({'direction': record['direction'],
                                         'speed': record['speed'], 'knocked_down': False})
                    continue
                base = {'direction': record['direction'], 'rep': record['rep'],
                        'speed': record['speed'], 'knocked_down': True}
            else:
                onset = record['posed']
                base = {'kind': record['kind']}
            scored = score_getup(rows, refs, record, onset)
            end = record.get('walk_end', record.get('gave_up', onset or 0) + 1.0)
            scored['policies'] = sorted({sp['policy'] for sp in spans
                                         if sp['end'] >= (onset or 0) and sp['start'] <= end})
            result[part].append(base | scored)
    getups = [r for r in result['c2'] if r.get('knocked_down')] + result['c3']
    ok = [r for r in getups if r.get('success')]
    c1_ok = not any(r['fell'] for r in result['c1']) if result['c1'] else None
    limp_in_c1 = [sp for sp in spans if sp['policy'] == 'limp_fall'
                  and any(t['part'] == 'c1' and t['start'] <= sp['start'] <= t['end']
                          for t in s.trials)]
    times = [r['getup_s'] for r in ok]
    verdict = {
        'c1_no_fall': c1_ok,
        'getups': f'{len(ok)}/{len(getups)}',
        'getup_rate': len(ok) / len(getups) if getups else None,
        'getup_s': metrics.summary(times),
        'false_limp_in_c1': len(limp_in_c1),
        'pass': bool(c1_ok and getups and len(ok) / len(getups) >= PASS_GETUPS
                     and not limp_in_c1),
        'note': 'plan: c1 no fall; >= 13 of 15 full get-ups within 10 s that walk again; '
                '0 false limp triggers (0a data is checked too)',
    }
    if s.args.quick or not {'c1', 'c2', 'c3'} <= {t['part'] for t in s.trials}:
        verdict['pass'] = None
        verdict['note'] += '. pass is null for --quick or a partial run (e.g. the board re-run)'
    return {'verdict': verdict, 'parts': result}


def options(p) -> None:
    p.add_argument('--reps', type=int, default=None, help='c2 pushes per direction (default 3)')
    p.add_argument('--push', type=float, default=None,
                   help='c2 push speed in m/s: skip the sweep (board re-run uses the Mac value)')


def main(argv=None) -> int:
    return common.main('c', __doc__, PARTS, score, argv, extra=options)


if __name__ == '__main__':
    sys.exit(main())
