"""Stage 0a: can walk_17500 walk with the head down, and how precisely can it inch?

    python -m microduck_toy.stage0.s0a_walk --config s2

Parts (plan section 4; head poses H0 (0, 0), H1 head +0.6, H2 head +1.0, H3 neck -0.4 head +0.8):
  a5   standing at each head pose: mouth tip and camera relative to the trunk
  a1v  vx 0.05..0.40 for 6 s at H0, H1, H2: forward dead band and speed
  a1w  wz ±0.25..±1.0 for 6 s at H1, cold (from standing) and warm (after 2 s at 0.3 m/s)
  a2   1 m straight at H1 and H2, vx 0.2 and 0.3, heading PI on (kp 1, ki 0.25), 3 each
  a3f  forward pulses at H2 from standstill: vx 0.25/0.30 x 0.2/0.3/0.5/0.8 s, 10 each
  a3t  turn pulses at H2: wz ±0.6/±1.0 x 0.3/0.5 s, 10 each
  a4   stop: 2 s at 0.3 m/s, then zero; coast distance and time to stand still (H0, H1, H2)
The lab's walker_facts.py measured the same table on alpha_walking (microduck-lab, Apache-2.0).
"""
from __future__ import annotations

import math
import sys
import time

from microduck_toy.control.heading_hold import HeadingConfig, HeadingHold
from microduck_toy.stage0 import common, metrics

VX_SWEEP = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
WZ_SWEEP = (0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1.0, -1.0)
SWEEP_S = 6.0
STEADY_FROM_S = 2.0
PULSE_VX = (0.25, 0.30)
PULSE_WZ = (0.6, -0.6, 1.0, -1.0)
PULSE_S = (0.2, 0.3, 0.5, 0.8)       # 0.2 s added to the plan's three, to look below 3 cm
TURN_PULSE_S = (0.3, 0.5)
SETTLE_MAX_S = 3.0
STEP_MAX_M = 0.03
STEP_SD_M = 0.01
TURN_MAX_DEG = 5.0


def reps(s, full: int, quick: int = 2) -> int:
    return quick if s.args.quick else full


def a5(s) -> None:
    for head in ('H0', 'H1', 'H2', 'H3'):
        s.reset(head=head)
        s.wait_settled(SETTLE_MAX_S)
        start = s.mark(f'a5_{head}')
        time.sleep(2.0)
        s.trial({'part': 'a5', 'head': head, 'start': start, 'end': s.mark(f'a5_{head}_end')})
    print('a5 完成', flush=True)


def a1v(s) -> None:
    speeds = (0.10, 0.30) if s.args.quick else VX_SWEEP
    for head in ('H0', 'H1', 'H2'):
        for vx in speeds:
            s.reset(head=head)
            start = s.mark(f'a1v_{head}_{vx}')
            s.robot.move(vx, 0.0, 0.0)
            time.sleep(SWEEP_S)
            s.robot.move(0.0, 0.0, 0.0)
            end = s.mark(f'a1v_{head}_{vx}_end')
            s.wait_settled(SETTLE_MAX_S)
            s.trial({'part': 'a1v', 'head': head, 'vx': vx, 'start': start, 'end': end})
        print(f'a1v {head} 完成', flush=True)


def a1w(s) -> None:
    rates = (0.5, -1.0) if s.args.quick else WZ_SWEEP
    for warm in (False, True):
        for wz in rates:
            s.reset(head='H1')
            if warm:
                s.robot.move(0.3, 0.0, 0.0)
                time.sleep(2.0)
            start = s.mark(f'a1w_{"warm" if warm else "cold"}_{wz}')
            s.robot.move(0.0, 0.0, wz)
            time.sleep(SWEEP_S)
            s.robot.move(0.0, 0.0, 0.0)
            end = s.mark(f'a1w_{wz}_end')
            s.wait_settled(SETTLE_MAX_S)
            s.trial({'part': 'a1w', 'head': 'H1', 'wz': wz, 'warm': warm,
                     'start': start, 'end': end})
        print(f'a1w {"热" if warm else "冷"}启动 完成', flush=True)


def a2(s) -> None:
    for head in ('H1', 'H2'):
        for vx in (0.2, 0.3):
            for rep in range(1, reps(s, 3, 1) + 1):
                origin = s.reset(head=head)
                hold = HeadingHold(HeadingConfig(kp=1.0, ki=0.25))
                start = s.mark(f'a2_{head}_{vx}_{rep}')
                last_t, reached, deadline = None, None, time.monotonic() + 15.0
                while time.monotonic() < deadline:
                    state = s.robot.wait(lambda st: last_t is None or st['t'] > last_t, 1.0,
                                         'next robot.state')
                    dt = 0.02 if last_t is None else state['t'] - last_t
                    last_t = state['t']
                    yaw = hold.update(state['imu']['gyro'], state['safety']['gravity'], dt,
                                      (vx, 0.0, 0.0))
                    s.robot.move(vx, 0.0, yaw)
                    if metrics.relative(origin, s.probe.truth())['forward'] >= 1.0:
                        reached = s.mark(f'a2_{head}_{vx}_{rep}_1m')
                        break
                    if s.robot.snapshot()['safety']['fallen']:
                        break
                s.robot.move(0.0, 0.0, 0.0)
                end = s.mark(f'a2_{head}_{vx}_{rep}_end')
                s.wait_settled(SETTLE_MAX_S)
                s.trial({'part': 'a2', 'head': head, 'vx': vx, 'rep': rep, 'start': start,
                         'reached': reached, 'end': end})
        print(f'a2 {head} 完成', flush=True)


def pulse(s, label: str, command: tuple[float, float, float], seconds: float) -> dict:
    s.wait_settled(SETTLE_MAX_S)
    start = s.mark(label)
    s.robot.move(*command)
    time.sleep(seconds)
    s.robot.move(0.0, 0.0, 0.0)
    released = s.mark(f'{label}_zero')
    s.wait_settled(SETTLE_MAX_S)
    return {'start': start, 'released': released}


def a3f(s) -> None:
    for vx in PULSE_VX:
        for seconds in PULSE_S:
            s.reset(head='H2')
            for rep in range(1, reps(s, 10) + 1):
                times = pulse(s, f'a3f_{vx}_{seconds}_{rep}', (vx, 0.0, 0.0), seconds)
                s.trial({'part': 'a3f', 'head': 'H2', 'vx': vx, 'seconds': seconds,
                         'rep': rep, **times})
            print(f'a3f vx {vx} × {seconds} s 完成', flush=True)


def a3t(s) -> None:
    for wz in PULSE_WZ:
        for seconds in TURN_PULSE_S:
            s.reset(head='H2')
            for rep in range(1, reps(s, 10) + 1):
                times = pulse(s, f'a3t_{wz}_{seconds}_{rep}', (0.0, 0.0, wz), seconds)
                s.trial({'part': 'a3t', 'head': 'H2', 'wz': wz, 'seconds': seconds,
                         'rep': rep, **times})
            print(f'a3t wz {wz} × {seconds} s 完成', flush=True)


def a4(s) -> None:
    for head in ('H0', 'H1', 'H2'):
        for rep in range(1, reps(s, 3, 1) + 1):
            s.reset(head=head)
            start = s.mark(f'a4_{head}_{rep}')
            s.robot.move(0.3, 0.0, 0.0)
            time.sleep(2.0)
            s.robot.move(0.0, 0.0, 0.0)
            stop = s.mark(f'a4_{head}_{rep}_zero')
            s.wait_settled(4.0)
            s.trial({'part': 'a4', 'head': head, 'rep': rep, 'start': start, 'released': stop})
        print(f'a4 {head} 完成', flush=True)


PARTS = {'a5': a5, 'a1v': a1v, 'a1w': a1w, 'a2': a2, 'a3f': a3f, 'a3t': a3t, 'a4': a4}


# ── scoring ──────────────────────────────────────────────────────────────

def fell(rows, start, end) -> bool:
    return metrics.fall_time(rows, start, end) is not None


def score_a5(rows, record) -> dict:
    span = metrics.window(rows, record['start'], record['end'])
    tips = [metrics.point_in(r, r['tip_x'], r['tip_y']) for r in span]
    mean = lambda xs: metrics.summary(xs).get('mean')  # noqa: E731
    return {'head': record['head'],
            'tip_forward': mean([t[0] for t in tips]), 'tip_left': mean([t[1] for t in tips]),
            'tip_z': mean([r['tip_z'] for r in span]), 'cam_z': mean([r['cam_z'] for r in span]),
            'cam_pitch_down_rad': mean([r['cam_pitch'] for r in span]),
            'neck_q': mean([r['neck_q'] for r in span]), 'head_q': mean([r['head_q'] for r in span]),
            'trunk_z': mean([r['z'] for r in span]), 'fell': fell(rows, record['start'], record['end'])}


def score_sweep(rows, record) -> dict:
    start, end = record['start'], record['start'] + SWEEP_S
    span = metrics.window(rows, start, end)
    origin, steady = span[0], metrics.at(rows, start + STEADY_FROM_S)
    final = metrics.relative(origin, span[-1])
    yaws = metrics.unwrapped_yaw(span)
    i2 = next(i for i, r in enumerate(span) if r['mono'] >= steady['mono'])
    out = {k: record[k] for k in ('head', 'vx', 'wz', 'warm') if k in record}
    out.update({'forward_m': final['forward'], 'left_m': final['left'],
                'yaw_deg': math.degrees(yaws[-1]),
                'fell': fell(rows, start, record['end'] + 1.0)})
    span_s = span[-1]['mono'] - steady['mono']
    if 'vx' in record:
        out['steady_mps'] = (final['forward'] - metrics.relative(origin, steady)['forward']) / span_s
    else:
        out['steady_radps'] = (yaws[-1] - yaws[i2]) / span_s
    return out


def score_a2(rows, record) -> dict:
    end = record['reached'] or record['end']
    span = metrics.window(rows, record['start'], end)
    origin, last = span[0], span[-1]
    rel = metrics.relative(origin, last)
    return {'head': record['head'], 'vx': record['vx'], 'rep': record['rep'],
            'reached_1m': record['reached'] is not None,
            'time_s': (record['reached'] - record['start']) if record['reached'] else None,
            'left_at_end_m': rel['left'], 'yaw_drift_deg': math.degrees(rel['dyaw']),
            'max_left_m': max(abs(metrics.relative(origin, r)['left']) for r in span),
            'fell': fell(rows, record['start'], record['end'] + 1.0)}


def score_pulse(rows, record) -> dict:
    origin = metrics.at(rows, record['start'])
    still = metrics.settle_time(rows, record['released'], end=record['released'] + SETTLE_MAX_S)
    final = metrics.at(rows, still if still is not None else record['released'] + SETTLE_MAX_S)
    rel = metrics.relative(origin, final)
    out = {k: record[k] for k in ('head', 'vx', 'wz', 'seconds', 'rep') if k in record}
    out.update({'forward_m': rel['forward'], 'left_m': rel['left'],
                'yaw_deg': math.degrees(rel['dyaw']),
                'still_after_zero_s': (still - record['released']) if still else None,
                'fell': fell(rows, record['start'], record['released'] + SETTLE_MAX_S)})
    return out


def score_stop(rows, record) -> dict:
    """Coast after the command goes to zero, measured from that moment."""
    stop = metrics.at(rows, record['released'])
    still = metrics.settle_time(rows, record['released'], end=record['released'] + 4.0)
    final = metrics.at(rows, still if still is not None else record['released'] + 4.0)
    rel = metrics.relative(stop, final)
    return {'head': record['head'], 'rep': record['rep'], 'coast_m': rel['forward'],
            'coast_left_m': rel['left'], 'coast_yaw_deg': math.degrees(rel['dyaw']),
            'speed_at_zero_mps': math.hypot(stop['vx'], stop['vy']),
            'still_after_zero_s': (still - record['released']) if still is not None else None,
            'fell': fell(rows, record['start'], record['released'] + 4.0)}


def group(results, keys):
    groups = {}
    for r in results:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    return groups


def pulse_table(results, keys, value, least) -> list[dict]:
    """Per pulse setting: summary of `value`; all_moved = every repeat went the commanded way
    by more than `least` (forward for vx, the sign of wz for turns)."""
    table = []
    for key, items in group(results, keys).items():
        values = [r[value] for r in items]
        sign = math.copysign(1.0, items[0].get('wz', 1.0))
        table.append({**dict(zip(keys, key)), **{f'{value}_{k}': v for k, v in
                      metrics.summary(values).items()},
                      'all_moved': all(sign * v > least for v in values),
                      'falls': sum(r['fell'] for r in items)})
    return table


def score(s, rows) -> dict:
    states = metrics.read_states(s.dir / 'state.jsonl')
    spans = metrics.label_spans(states)
    scorers = {'a5': score_a5, 'a1v': score_sweep, 'a1w': score_sweep, 'a2': score_a2,
               'a3f': score_pulse, 'a3t': score_pulse, 'a4': score_stop}
    parts = {}
    for record in s.trials:
        parts.setdefault(record['part'], []).append(scorers[record['part']](rows, record))
    a4 = parts.get('a4', [])
    verdict = {}
    if 'a1v' in parts:
        h1 = [r for r in parts['a1v'] if r['head'] == 'H1' and r['vx'] <= 0.3 + 1e-9]
        verdict['a1_H1_no_fall'] = not any(r['fell'] for r in h1)
        for head in ('H0', 'H1', 'H2'):
            dead = [r['vx'] for r in parts['a1v'] if r['head'] == head and r['steady_mps'] < 0.01]
            verdict[f'forward_dead_band_{head}'] = max(dead) if dead else None
    if 'a2' in parts:
        verdict['a2_H1_no_fall'] = not any(r['fell'] for r in parts['a2'] if r['head'] == 'H1')
        verdict['a2_all_reached_1m'] = all(r['reached_1m'] for r in parts['a2'])
    if 'a3f' in parts:
        table = pulse_table(parts['a3f'], ('vx', 'seconds'), 'forward_m', 0.005)
        usable = [t for t in table if t['all_moved'] and t['falls'] == 0]
        best = min(usable, key=lambda t: t['forward_m_mean'], default=None)
        verdict['a3f_table'] = table
        verdict['min_step'] = best
        verdict['min_step_ok'] = bool(best and best['forward_m_mean'] <= STEP_MAX_M
                                      and best['forward_m_sd'] <= STEP_SD_M)
    if 'a3t' in parts:
        table = pulse_table(parts['a3t'], ('wz', 'seconds'), 'yaw_deg', 0.5)
        usable = [t for t in table if t['all_moved'] and t['falls'] == 0]
        best = min(usable, key=lambda t: abs(t['yaw_deg_mean']), default=None)
        verdict['a3t_table'] = table
        verdict['min_turn'] = best
        verdict['min_turn_ok'] = bool(best and abs(best['yaw_deg_mean']) <= TURN_MAX_DEG)
    if 'a3f' in parts or 'a3t' in parts:
        verdict['a3_H2_no_fall'] = not any(r['fell'] for p in ('a3f', 'a3t')
                                          for r in parts.get(p, []))
    if a4:
        verdict['stop'] = {head: {'coast_m': metrics.summary([r['coast_m'] for r in a4
                                                            if r['head'] == head]),
                                  'still_after_zero_s': metrics.summary(
                                      [r['still_after_zero_s'] for r in a4 if r['head'] == head])}
                           for head in ('H0', 'H1', 'H2')}
    falls = sum(r.get('fell', False) for p in parts.values() for r in p)
    limp = [sp for sp in spans if sp['policy'] == 'limp_fall']
    verdict['falls_total'] = falls
    verdict['limp_fall_spans'] = len(limp)
    verdict['false_limp'] = max(0, len(limp) - falls)
    checks = [verdict[k] for k in ('a1_H1_no_fall', 'a2_H1_no_fall', 'a3_H2_no_fall',
                                   'min_step_ok', 'min_turn_ok') if k in verdict]
    verdict['pass'] = None if s.args.quick or len(checks) < 5 else all(checks)
    verdict['note'] = ('plan: H1 a1 (vx <= 0.3) and a2 without falls; H2 a3 without falls; '
                       'min forward step <= 3 cm with sd <= 1 cm; min turn <= 5 deg. '
                       'pass is null for --quick or a partial run')
    return {'verdict': verdict, 'parts': parts}


def main(argv=None) -> int:
    return common.main('a', __doc__, PARTS, score, argv)


if __name__ == '__main__':
    sys.exit(main())
