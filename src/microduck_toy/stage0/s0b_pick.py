"""Stage 0b: does the official GroundPick run cleanly in our robotd, and where does the beak go?

    python -m microduck_toy.stage0.s0b_pick --config s2

Parts (plan section 5): b_H0 and b_H2 trigger `robot.do ground_pick` 10 times each, from a
settled stand with head H0 or H2 (H2 is how an inching approach ends). After every pick the
robot stands 3 s and then walks 0.5 m at 0.2 m/s. Needs alpha_ground_pick.onnx in the
ground_pick slot. Phase timing (microduck_ground_pick_env_cfg.py:96-111): 4 s cycle, beak
down by phase 0.375, held to 0.425, up by 0.80; robotd hands back at phase 0.7.
"""
from __future__ import annotations

import math
import sys
import time

from microduck_toy.stage0 import common, metrics

PERIOD_S = 4.0          # robotd's walking-mode ground_pick_period default
WALK_VX = 0.2
WALK_M = 0.5
TIP_MAX_Z = 0.03
SPREAD_SD_M = 0.01
STABLE_WITHIN_S = 1.0


def pick(s, head: str, rep: int) -> dict:
    s.reset(head=head)
    s.wait_settled(3.0)
    label = f'b_{head}_{rep}'
    record = {'part': f'b_{head}', 'head': head, 'rep': rep, 'trigger': s.mark(label)}
    record['reply'] = s.robot.request('robot.do', {'skill': 'ground_pick'})
    try:
        s.robot.wait(lambda st: st['policy'] == 'ground_pick', 1.5, 'ground_pick start')
        s.robot.wait(lambda st: st['policy'] != 'ground_pick', 6.0, 'ground_pick hand-back')
    except TimeoutError as error:
        record['error'] = str(error)
    record['handback_seen'] = s.mark(f'{label}_handback')
    fell = s.watch_fall(3.0)
    if fell is None:
        origin = s.probe.truth()
        record['walk_start'] = s.mark(f'{label}_walk')
        s.robot.move(WALK_VX, 0.0, 0.0)
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            if metrics.relative(origin, s.probe.truth())['forward'] >= WALK_M:
                break
            if s.robot.snapshot()['safety']['fallen']:
                break
            time.sleep(0.02)
        s.robot.move(0.0, 0.0, 0.0)
        record['walk_end'] = s.mark(f'{label}_walk_end')
        s.wait_settled(3.0)
    else:
        s.robot.move(0.0, 0.0, 0.0)
        s.robot.head('H0')
        record['fell_seen'] = fell
        time.sleep(3.0)
    return record


def b_parts(head: str):
    def part(s) -> None:
        for rep in range(1, (2 if s.args.quick else 10) + 1):
            s.trial(pick(s, head, rep))
            print(f'b {head} 第 {rep} 次完成', flush=True)
    return part


PARTS = {'b_H0': b_parts('H0'), 'b_H2': b_parts('H2')}


def score_pick(rows, spans, refs, record) -> dict:
    out = {'head': record['head'], 'rep': record['rep']}
    span = next((sp for sp in spans if sp['policy'] == 'ground_pick'
                 and record['trigger'] <= sp['start'] <= record['trigger'] + 1.5), None)
    end_all = record.get('walk_end', record['handback_seen'] + 3.0) + 1.0
    out['fell'] = metrics.fall_time(rows, record['trigger'], end_all) is not None
    if span is None:
        return {**out, 'ran': False}
    start, handback = span['start'], span['end'] + 0.02
    origin = metrics.at(rows, record['trigger'])
    during = metrics.window(rows, start, handback + 0.5)
    low = min(during, key=lambda r: r['tip_z'])
    tip_forward, tip_left = metrics.point_in(origin, low['tip_x'], low['tip_y'])
    drift = metrics.relative(origin, metrics.at(rows, handback))
    stand = lambda r: metrics.standing(r, refs)  # noqa: E731
    stable = metrics.first_hold(rows, stand, handback, 0.5, end=handback + 3.0)
    posture = {f'{dt:+.0f}s': {'gz': metrics.at(rows, handback + dt)['gz'],
                               'z': metrics.at(rows, handback + dt)['z']}
               for dt in (-1.0, 0.0, 1.0)}
    out.update({
        'ran': True, 'duration_s': handback - start,
        'handback_phase': (handback - start) / PERIOD_S,
        'tip_min_z': low['tip_z'], 'tip_min_phase': (low['mono'] - start) / PERIOD_S,
        'tip_forward_m': tip_forward, 'tip_left_m': tip_left,
        'trunk_drift_forward_m': drift['forward'], 'trunk_drift_left_m': drift['left'],
        'trunk_drift_yaw_deg': math.degrees(drift['dyaw']),
        'stable_after_handback_s': (stable - handback) if stable is not None else None,
        'posture_around_handback': posture,
    })
    if 'walk_start' in record:
        walk = metrics.window(rows, record['walk_start'], record['walk_end'])
        out['walk_m'] = metrics.relative(walk[0], walk[-1])['forward'] if walk else None
    return out


def score(s, rows) -> dict:
    spans = metrics.label_spans(metrics.read_states(s.dir / 'state.jsonl'))
    parts = {}
    for record in s.trials:
        parts.setdefault(record['part'], []).append(score_pick(rows, spans, s.refs, record))
    verdict = {}
    for name, results in parts.items():
        ran = [r for r in results if r['ran']]
        tips = [r['tip_min_z'] for r in ran]
        forward = metrics.summary([r['tip_forward_m'] for r in ran])
        left = metrics.summary([r['tip_left_m'] for r in ran])
        stable_ok = all(r['stable_after_handback_s'] is not None
                        and r['stable_after_handback_s'] <= STABLE_WITHIN_S for r in ran)
        walked = all((r.get('walk_m') or 0) >= WALK_M - 0.1 for r in ran)
        verdict[name] = {
            'picks': f'{len(ran)}/{len(results)}', 'falls': sum(r['fell'] for r in results),
            'tip_min_z': metrics.summary(tips), 'tip_forward_m': forward, 'tip_left_m': left,
            'tip_min_phase': metrics.summary([r['tip_min_phase'] for r in ran]),
            'handback_phase': metrics.summary([r['handback_phase'] for r in ran]),
            'stable_within_1s': stable_ok, 'walked_after': walked,
            'pass': None if s.args.quick else bool(
                ran and len(ran) == len(results) and not any(r['fell'] for r in results)
                and stable_ok and walked and max(tips) <= TIP_MAX_Z
                and forward['sd'] <= SPREAD_SD_M and left['sd'] <= SPREAD_SD_M),
        }
    verdict['note'] = ('plan: 10/10 without falls per head pose, stable within 1 s of the '
                       'hand-back, walks after; beak tip lowest <= 3 cm; landing sd <= 1 cm '
                       'forward and left. pass is null for --quick')
    return {'verdict': verdict, 'parts': parts}


def check_ground_pick(s) -> None:
    slots = {slot['slot']: slot for slot in s.meta['policies']['slots']}
    if not slots.get('ground_pick', {}).get('path'):
        raise RuntimeError('robotd has no ground_pick policy loaded; use a stage0 config')


def main(argv=None) -> int:
    return common.main('b', __doc__, PARTS, score, argv, check=check_ground_pick)


if __name__ == '__main__':
    sys.exit(main())
