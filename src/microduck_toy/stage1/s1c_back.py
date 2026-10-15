"""Stage 1C: the two things the inching controller needs that Stage 0 did not measure.

    python -m microduck_toy.stage1.s1c_back [--parts c_back,c_odom,c_trim] [--quick]

Parts (plan section 5, revision 1: no side-step test, decision 2B):
  c_back   backward pulses at H2: vx -0.3 and -0.4 for 0.3 s and 0.5 s, 10 each. Used when a pick
           misses and the robot backs off to look again; it need not be precise.
  c_odom   forward and turn pulses at H2 (vx 0.25 x 0.3/0.5 s, wz +0.6 x 0.3 s, wz -1.0 x 0.3 s),
           10 each. Every pulse of both parts compares robotd's contact odometry (robot.state odom,
           which a real robot has) with the truth, from the settled stand before to the settled
           stand after.
  c_trim   added 2026-10-15 after the first Mac run: every backward pulse turned the robot 9-16 deg
           left and vx 0.25 x 0.3 s forward pulses turned it about 5 deg left. The same pulses with
           a right-turn bias, 10 each, to find pulses that go straight: vx 0.25 x 0.3 s with wz
           -0.2/-0.4/-0.6, vx 0.25 x 0.5 s with wz -0.1/-0.2/-0.3, vx -0.4 x 0.5 s with wz
           -0.2/-0.4/-0.6.
Usable (plan): backward = some pulse that backs >= 5 cm every time with <= 5 deg of turn and no fall;
straight forward (c_trim) = |mean turn| <= 2 deg and every turn within 5 deg, no fall;
odometry = per-pulse error sd <= 0.5 cm forward and left, and 90% of heading errors <= 1 deg.
"""
from __future__ import annotations

import math
import sys
import time

from microduck_toy.stage1 import common, metrics

BACK = [(-0.3, 0.3), (-0.3, 0.5), (-0.4, 0.3), (-0.4, 0.5)]
ODOM = [('vx', 0.25, 0.3), ('vx', 0.25, 0.5), ('wz', 0.6, 0.3), ('wz', -1.0, 0.3)]
# (cloud twin, walk_17500 standing: wz -0.3 x 0.3 s still +3.7 deg, -0.6 already -6.5 deg)
TRIM = [(0.25, -0.2, 0.3), (0.25, -0.4, 0.3), (0.25, -0.6, 0.3),
        (0.25, -0.1, 0.5), (0.25, -0.2, 0.5), (0.25, -0.3, 0.5),
        (-0.4, -0.2, 0.5), (-0.4, -0.4, 0.5), (-0.4, -0.6, 0.5)]
STRAIGHT_MEAN_DEG = 2.0
RESET_EVERY = 5
BACK_MIN_M = 0.05
BACK_MAX_TURN_DEG = 5.0
ODOM_SD_M = 0.005
ODOM_YAW_DEG = 1.0


def odom(state: dict) -> dict:
    o = state['odom']
    return {'x': float(o['position'][0]), 'y': float(o['position'][1]), 'yaw': float(o['yaw'])}


def pulse(s, part: str, label: str, command: tuple, seconds: float, rep: int, extra: dict) -> None:
    if rep % RESET_EVERY == 0:
        s.reset(head='H2')
    s.wait_settled(3.0)
    before_state = s.robot.snapshot()
    start = s.mark(f'{part}_{label}_{rep}')
    s.robot.move(*command)
    time.sleep(seconds)
    s.robot.move(0.0, 0.0, 0.0)
    stop = s.mark(f'{part}_{label}_{rep}_stop')
    settled = s.wait_settled(3.0)
    end = s.mark(f'{part}_{label}_{rep}_end')
    s.trial({'part': part, 'label': label, 'rep': rep, 'command': list(command),
             'seconds': seconds, 'start': start, 'stop': stop, 'end': end,
             'settled_after_s': settled, 'odom_before': odom(before_state),
             'odom_after': odom(s.robot.snapshot()), **extra})


def c_back(s) -> None:
    reps = 2 if s.args.quick else 10
    for vx, seconds in BACK:
        for rep in range(reps):
            pulse(s, 'c_back', f'vx{vx:+.1f}x{seconds}', (vx, 0.0, 0.0), seconds, rep,
                  {'vx': vx})
        print(f'c_back vx {vx} × {seconds} s 完成', flush=True)


def c_odom(s) -> None:
    reps = 2 if s.args.quick else 10
    for axis, value, seconds in ODOM:
        command = (value, 0.0, 0.0) if axis == 'vx' else (0.0, 0.0, value)
        for rep in range(reps):
            pulse(s, 'c_odom', f'{axis}{value:+.2f}x{seconds}', command, seconds, rep,
                  {'axis': axis, 'value': value})
        print(f'c_odom {axis} {value} × {seconds} s 完成', flush=True)


def c_trim(s) -> None:
    reps = 2 if s.args.quick else 10
    for vx, wz, seconds in TRIM:
        for rep in range(reps):
            pulse(s, 'c_trim', f'vx{vx:+.2f}wz{wz:+.1f}x{seconds}', (vx, 0.0, wz), seconds, rep,
                  {'vx': vx, 'wz': wz})
        print(f'c_trim vx {vx} wz {wz} × {seconds} s 完成', flush=True)


PARTS = {'c_back': c_back, 'c_odom': c_odom, 'c_trim': c_trim}


def in_frame(before: dict, after: dict) -> dict:
    dx, dy = after['x'] - before['x'], after['y'] - before['y']
    c, sn = math.cos(before['yaw']), math.sin(before['yaw'])
    return {'forward': c * dx + sn * dy, 'left': -sn * dx + c * dy,
            'dyaw': metrics.wrap(after['yaw'] - before['yaw'])}


def score_pulse(rows, record) -> dict:
    a, b = metrics.at_mark(rows, record['start']), metrics.at_mark(rows, record['end'])
    truth = metrics.relative(a, b)
    odo = in_frame(record['odom_before'], record['odom_after'])
    return {'label': record['label'], 'rep': record['rep'],
            'fell': metrics.fall_time(rows, record['start'], record['end']) is not None,
            'forward_m': truth['forward'], 'left_m': truth['left'],
            'dyaw_deg': math.degrees(truth['dyaw']),
            'odom_forward_m': odo['forward'], 'odom_left_m': odo['left'],
            'odom_dyaw_deg': math.degrees(odo['dyaw']),
            'err_forward_m': odo['forward'] - truth['forward'],
            'err_left_m': odo['left'] - truth['left'],
            'err_dyaw_deg': math.degrees(metrics.wrap(odo['dyaw'] - truth['dyaw'])),
            'settled_after_s': record.get('settled_after_s')}


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    k = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[k]


def pulse_table(results: list[dict]) -> dict:
    """Per pulse label: what it does (truth), and whether it is a usable backward pulse
    (>= 5 cm back every time, <= 5 deg) or a straight forward one (|mean| <= 2 deg, all <= 5)."""
    table = {}
    for label in dict.fromkeys(r['label'] for r in results):
        rs = [r for r in results if r['label'] == label]
        turns = [r['dyaw_deg'] for r in rs]
        falls = sum(r['fell'] for r in rs)
        entry = {'forward_m': metrics.summary([r['forward_m'] for r in rs]),
                 'left_m': metrics.summary([r['left_m'] for r in rs]),
                 'turn_deg': metrics.summary(turns), 'falls': falls}
        if label.startswith('vx-'):
            back = [-r['forward_m'] for r in rs]
            entry['back_m'] = metrics.summary(back)
            entry['usable'] = bool(rs and min(back) >= BACK_MIN_M and not falls
                                   and max(abs(t) for t in turns) <= BACK_MAX_TURN_DEG)
        else:
            entry['straight'] = bool(rs and not falls
                                     and abs(sum(turns) / len(turns)) <= STRAIGHT_MEAN_DEG
                                     and max(abs(t) for t in turns) <= BACK_MAX_TURN_DEG)
        table[label] = entry
    return table


def score(s, rows) -> dict:
    parts: dict[str, list] = {}
    for record in s.trials:
        parts.setdefault(record['part'], []).append(score_pulse(rows, record))
    verdict = {}
    quick = s.args.quick
    if 'c_back' in parts:
        table = pulse_table(parts['c_back'])
        verdict['c_back'] = {'pulses': table,
                             'usable': None if quick else any(t['usable'] for t in table.values())}
    if 'c_trim' in parts:
        table = pulse_table(parts['c_trim'])
        verdict['c_trim'] = {
            'pulses': table,
            'straight_forward': None if quick else [k for k, t in table.items()
                                                    if t.get('straight')],
            'usable_backward': None if quick else [k for k, t in table.items()
                                                   if t.get('usable')]}
    all_pulses = [r for rs in parts.values() for r in rs if not r['fell']]
    if all_pulses:
        sd = lambda key: metrics.summary([r[key] for r in all_pulses])  # noqa: E731
        yaw_abs = [abs(r['err_dyaw_deg']) for r in all_pulses]
        per_label = {}
        for label in dict.fromkeys(r['label'] for r in all_pulses):
            rs = [r for r in all_pulses if r['label'] == label]
            per_label[label] = {k: metrics.summary([r[k] for r in rs])
                                for k in ('err_forward_m', 'err_left_m', 'err_dyaw_deg')}
        fwd, left = sd('err_forward_m'), sd('err_left_m')
        verdict['c_odom'] = {
            'pulses': len(all_pulses), 'err_forward_m': fwd, 'err_left_m': left,
            'err_dyaw_deg_p90_abs': percentile(yaw_abs, 0.9), 'per_pulse': per_label,
            'usable': None if quick else bool(fwd['sd'] <= ODOM_SD_M and left['sd'] <= ODOM_SD_M
                                              and percentile(yaw_abs, 0.9) <= ODOM_YAW_DEG),
        }
    verdict['note'] = ('plan 1C: not pass/fail but which tools the C segment may use; usable is '
                       'null for --quick')
    return {'verdict': verdict, 'parts': parts}


def main(argv=None) -> int:
    return common.main('c', __doc__, PARTS, score, argv)


if __name__ == '__main__':
    sys.exit(main())
