"""Stage 1B: the beak. Mouth open/close, weld, let go; and the pick itself when the stop is perfect.

    python -m microduck_toy.stage1.s1b_grasp [--parts b_mech,b_phi,b_ideal,b_offset,b_fall] [--quick]

Parts (plan section 4). The toy is put in place by the test port (a fixture: truth is used only to
place toys and to score):
  b_mech    no GroundPick: H0 standing, mouth open, a toy put at the beak tip, mouth closed. Expect
            a weld, the toy following the head (H0 -> H2 -> H0), the mouth reading "held", and the
            toy dropping when the mouth opens. A toy 6 cm off must be out of reach. 3 targets x 2.
  b_phi     the close phase: 0.29, 0.33, 0.38 x 3 targets x 3 picks from the ideal spot; the
            phase with the smallest mean beak-to-toy distance at the close is used by the parts
            after it in the same run (else configs/control/stage1.json pick.phi_close).
  b_ideal   H2, settled, toy centre 8.0 cm ahead and 1.5 cm left of the trunk (Stage 0's beak
            landing), toy heading 0/45/90/135 deg: 10 picks per target.
  b_offset  the same with the toy moved 1, 2, 3, 4, 5 cm in 8 directions in turn: 5 picks each.
  b_fall    a toy held (as b_mech) while walking at 0.2 m/s with head H1, then a push as in
            Stage 0 c2 (the speeds that knocked s2 over: right 0.6, forward/left/backward 1.05 m/s,
            from stage0_facts.json); directions in that order, one per try, until the robot falls.
            The toy must drop on the fall. (A push while standing still at 1.4 m/s did not knock
            velstand over in 3 of 3 on the Mac, 2026-10-15.)
Pass (plan): b_ideal 30/30; stands within 1 s of the hand-back; held toys rise >= 3 cm; b_offset
successes per offset within 1 of the expected sum of 1 - (e / 4 cm)^2; b_mech and b_fall all as
expected. Needs alpha_ground_pick.onnx in the ground_pick slot (s2) for b_phi/b_ideal/b_offset.
"""
from __future__ import annotations

import math
import sys
import time

from microduck_toy.control.pick import ground_pick
from microduck_toy.scene import toys as toy_lib
from microduck_toy.stage1 import common, metrics

TARGETS = [toy_lib.SHORT[n] for n in toy_lib.targets()]
LIFT_M = 0.03
STABLE_WITHIN_S = 1.0
OFFSETS_CM = (1, 2, 3, 4, 5)
FALL_WALK_VX = 0.2           # Stage 0 c2 pushed while walking, head H1
FALL_ORDER = ('right', 'forward', 'left', 'backward')     # weakest side first (Stage 0 0c)
FALL_DIRECTIONS = {'forward': (1.0, 0.0), 'backward': (-1.0, 0.0),
                   'left': (0.0, 1.0), 'right': (0.0, -1.0)}
FALL_START = (-0.5, 0.0, 0.0)  # 0.4 m of walking stays clear of the tray (x 0.5-0.8 m)
FALL_WATCH_S = 3.0


def phi_close(s) -> float:
    return float(s.meta.get('phi_close_chosen', s.cfg['pick']['phi_close']))


def grasp_entries(s, before: int) -> list[dict]:
    return s.grasps_since(before)


def one_pick(s, part: str, toy: str, rep: int, forward: float, left: float, yaw_rel: float,
             phi: float, extra: dict | None = None) -> dict:
    pick_cfg = s.cfg['pick']
    s.reset(head='H2')
    s.wait_settled(3.0)
    s.place_in_trunk(toy, forward, left, yaw_rel)
    time.sleep(0.3)
    before = s.grasp_count()
    label = f'{part}_{toy}_{rep}'
    record = {'part': part, 'toy': toy, 'rep': rep, 'forward': forward, 'left': left,
              'yaw_rel': yaw_rel, 'start': s.mark(label), **(extra or {})}
    result = ground_pick(s.robot, phi, float(pick_cfg['period_s']),
                         float(s.cfg['grasp']['held_threshold_deg']),
                         confirm_s=float(pick_cfg['confirm_after_s']),
                         mark=lambda tag: s.mark(f'{label}_{tag}'))
    record['pick'] = result
    record['grasps'] = grasp_entries(s, before)
    s.robot.mouth(1.0)                      # let go, then check it fell
    time.sleep(1.0)
    record['released'] = s.mark(f'{label}_released')
    s.robot.mouth(0.0)
    return record


def b_phi(s) -> None:
    pick_cfg = s.cfg['pick']
    fwd, left = float(pick_cfg['ideal_toy_forward_m']), float(pick_cfg['ideal_toy_left_m'])
    reps = 1 if s.args.quick else 3
    errors: dict[float, list] = {}
    for phi in pick_cfg['phi_close_sweep']:
        for toy in (TARGETS[:1] if s.args.quick else TARGETS):
            for rep in range(reps):
                rec = one_pick(s, 'b_phi', toy, rep, fwd, left, math.radians(45 * rep), float(phi),
                               {'phi': float(phi)})
                s.trial(rec)
                close = next((g for g in rec['grasps'] if 'nearest_exy' in g), None)
                if close is not None:
                    errors.setdefault(float(phi), []).append(close['nearest_exy'])
        print(f'b_phi {phi} 完成', flush=True)
    if errors:
        best = min(errors, key=lambda p: sum(errors[p]) / len(errors[p]))
        s.meta['phi_close_chosen'] = best
        s.meta['phi_close_errors_m'] = {str(k): v for k, v in errors.items()}
        print(f'选定闭嘴相位 φ = {best}', flush=True)


def b_ideal(s) -> None:
    pick_cfg = s.cfg['pick']
    fwd, left = float(pick_cfg['ideal_toy_forward_m']), float(pick_cfg['ideal_toy_left_m'])
    reps = 2 if s.args.quick else 10
    for toy in TARGETS:
        for rep in range(reps):
            s.trial(one_pick(s, 'b_ideal', toy, rep, fwd, left, math.radians(45 * (rep % 4)),
                             phi_close(s)))
        print(f'b_ideal {toy} 完成', flush=True)


def b_offset(s) -> None:
    pick_cfg = s.cfg['pick']
    fwd, left = float(pick_cfg['ideal_toy_forward_m']), float(pick_cfg['ideal_toy_left_m'])
    reps = 1 if s.args.quick else 5
    k = 0
    for cm in OFFSETS_CM:
        for rep in range(reps):
            angle = math.radians(45 * (k % 8))
            toy = TARGETS[k % 3]
            k += 1
            df, dl = cm / 100 * math.cos(angle), cm / 100 * math.sin(angle)
            s.trial(one_pick(s, 'b_offset', toy, rep, fwd + df, left + dl,
                             math.radians(60 * (k % 3)), phi_close(s),
                             {'offset_cm': cm, 'offset_dir_deg': 45 * ((k - 1) % 8)}))
        print(f'b_offset {cm} cm 完成', flush=True)


def hold_at_tip(s, toy: str) -> dict:
    """Mouth open, toy at the beak tip, mouth closed (standing; no GroundPick)."""
    s.robot.mouth(1.0)
    time.sleep(0.4)
    row = s.probe.truth()
    s.probe.call('toy', name=toy, x=row['tip_x'], y=row['tip_y'], yaw=row['yaw'], z=row['tip_z'])
    s.robot.mouth(0.0)
    time.sleep(0.6)
    return row


def b_mech(s) -> None:
    held_threshold = float(s.cfg['grasp']['held_threshold_deg'])
    for toy in TARGETS:
        for rep in range(1 if s.args.quick else 2):
            s.reset(head='H0')
            s.wait_settled(2.0)
            before = s.grasp_count()
            record = {'part': 'b_mech', 'toy': toy, 'rep': rep, 'start': s.mark(f'b_mech_{toy}_{rep}')}
            hold_at_tip(s, toy)
            record['closed'] = s.mark('closed')
            record['mouth_deg_closed'] = math.degrees(s.robot.snapshot()['joints'][9])
            record['held_read'] = record['mouth_deg_closed'] > held_threshold
            s.robot.head('H2')
            time.sleep(1.5)
            record['h2'] = s.mark('h2')
            s.robot.head('H0')
            time.sleep(1.5)
            record['h0'] = s.mark('h0')
            s.robot.mouth(1.0)
            time.sleep(1.5)
            record['opened'] = s.mark('opened')
            s.robot.mouth(0.0)
            time.sleep(0.3)
            record['mouth_deg_empty'] = math.degrees(s.robot.snapshot()['joints'][9])
            # Out of reach: the toy 6 cm beside the beak tip, at its height.
            s.robot.mouth(1.0)
            time.sleep(0.4)
            row = s.probe.truth()
            c, sn = math.cos(row['yaw']), math.sin(row['yaw'])
            s.probe.call('toy', name=toy, x=row['tip_x'] - 0.06 * sn, y=row['tip_y'] + 0.06 * c,
                         yaw=row['yaw'], z=row['tip_z'])
            s.robot.mouth(0.0)
            time.sleep(0.6)
            record['far_closed'] = s.mark('far_closed')
            record['grasps'] = grasp_entries(s, before)
            s.trial(record)
        print(f'b_mech {toy} 完成', flush=True)


def watch_fall(s, duration: float) -> float | None:
    """Time a fall is first seen (truth down for FALL_HOLD_S, the rule the arena body lets go
    by), else None after `duration`."""
    down_since = None
    end = time.monotonic() + duration
    while time.monotonic() < end:
        row = s.probe.truth()
        if metrics.down(row):
            down_since = down_since or time.monotonic()
            if time.monotonic() - down_since >= metrics.FALL_HOLD_S:
                return down_since
        else:
            down_since = None
        time.sleep(0.02)
    return None


def b_fall(s) -> None:
    speeds = s.facts['getup']['push_to_fall_mps_mac_s2']
    for toy in (TARGETS[:1] if s.args.quick else TARGETS):
        for attempt, direction in enumerate(FALL_ORDER):
            s.reset(head='H1', robot=FALL_START)
            s.wait_settled(2.0)
            before = s.grasp_count()
            record = {'part': 'b_fall', 'toy': toy, 'attempt': attempt, 'direction': direction,
                      'speed': float(speeds[direction]),
                      'start': s.mark(f'b_fall_{toy}_{direction}')}
            hold_at_tip(s, toy)
            s.robot.move(FALL_WALK_VX, 0.0, 0.0)
            time.sleep(2.0)
            ux, uy = FALL_DIRECTIONS[direction]
            record['push'] = s.mark(f'push_{direction}')
            s.probe.call('push', vx=ux * record['speed'], vy=uy * record['speed'])
            record['fell_seen'] = watch_fall(s, FALL_WATCH_S)
            s.robot.move(0.0, 0.0, 0.0)
            time.sleep(1.0)
            record['end'] = s.mark('end')
            record['grasps'] = grasp_entries(s, before)
            record['falls_seen'] = [f for f in s.robot.falls if f['mono'] >= record['push']]
            s.trial(record)
            if record['fell_seen'] is not None:
                time.sleep(3.0)       # velstand gets up (s2); reset() teleports otherwise
                break
        print(f'b_fall {toy} 完成', flush=True)


PARTS = {'b_mech': b_mech, 'b_phi': b_phi, 'b_ideal': b_ideal, 'b_offset': b_offset,
         'b_fall': b_fall}


def close_entry(record: dict) -> dict | None:
    return next((g for g in record.get('grasps', []) if 'nearest_exy' in g), None)


def score_pick(s, rows, record) -> dict:
    out = {k: record.get(k) for k in ('toy', 'rep', 'phi', 'offset_cm', 'offset_dir_deg')}
    pick = record.get('pick') or {}
    out['ran'] = bool(pick.get('ran'))
    g = close_entry(record)
    out['result'] = g and g['result']
    out['grasped'] = bool(g and g['result'] == 'grasped')
    out['e_xy_m'] = g and g.get('nearest_exy')
    out['e_z_m'] = g and g.get('nearest_ez')
    out['p'] = g and g.get('p')
    out['close_phase'] = pick.get('close_phase')
    out['handback_phase'] = pick.get('handback_phase')
    out['held_read'] = pick.get('held')
    out['mouth_deg'] = pick.get('mouth_deg')
    if 'handback' in pick:
        hb = pick['handback']
        stand = lambda r: metrics.standing(r, s.refs)  # noqa: E731
        stable = metrics.first_hold(rows, stand, hb, 0.5, end=hb + 3.0)
        out['stable_after_handback_s'] = (stable - hb) if stable is not None else None
        toy = record['toy']
        rest = s.meta['toys'][toy]['rest_z']
        span = metrics.window(rows, hb, pick.get('confirm', hb + 1.0))
        out['lift_m'] = (max(r[f'{toy}_z'] for r in span) - rest) if span else None
        dropped = metrics.at_mark(rows, record['released'])
        out['dropped'] = dropped[f'{toy}_z'] < rest + 0.01 and dropped['holding'] == ''
    out['fell'] = metrics.fall_time(rows, record['start'], record.get('released', record['start'] + 8)) \
        is not None
    return out


def score_mech(s, rows, record) -> dict:
    toy = record['toy']
    results = [g['result'] for g in record['grasps'] if 'nearest_exy' in g]
    closed, h2, h0, opened = (metrics.at_mark(rows, record[k])
                               for k in ('closed', 'h2', 'h0', 'opened'))
    rest = s.meta['toys'][toy]['rest_z']
    return {'toy': toy, 'rep': record['rep'], 'results': results,
            'held_read': record['held_read'], 'mouth_deg_closed': record['mouth_deg_closed'],
            'mouth_deg_empty': record['mouth_deg_empty'],
            'follows_head': abs((h2[f'{toy}_z'] - h0[f'{toy}_z']) - (h2['tip_z'] - h0['tip_z'])) < 0.01
            and closed['holding'] == toy,
            'dropped_on_open': opened['holding'] == '' and opened[f'{toy}_z'] < rest + 0.03,
            # closes: the grasp, the empty close after letting go, the 6 cm one
            'ok': results == ['grasped', 'out_of_reach', 'out_of_reach'] and record['held_read']
            and record['mouth_deg_empty'] < float(s.cfg['grasp']['held_threshold_deg'])}


def score_fall(s, rows, record) -> dict:
    toy = record['toy']
    released = [g for g in record['grasps'] if g.get('result') == 'released_fell']
    fell = metrics.fall_time(rows, record['push'], record['end']) is not None
    pushed = metrics.at_mark(rows, record['push'])
    end = metrics.at_mark(rows, record['end'])
    return {'toy': toy, 'attempt': record.get('attempt', 0), 'direction': record.get('direction'),
            'speed': record.get('speed'), 'held_at_push': pushed['holding'] == toy,
            'fell': fell, 'released_on_fall': bool(released),
            'held_at_end': end['holding'] == toy,
            'guard_saw_fall': bool(record.get('falls_seen')),
            'ok': fell and bool(released) and pushed['holding'] == toy}


def fall_check(results: list[dict]) -> dict:
    """Per toy: the try that knocked it over must have let the toy go; tries that did not
    knock it over count how often a held toy stayed in the beak through a recovered push."""
    toys = list(dict.fromkeys(r['toy'] for r in results))
    falls = [r for r in results if r['fell']]
    no_fall = [r for r in results if not r['fell']]
    out = {'toys': len(toys), 'toys_fell': len({r['toy'] for r in falls}),
           'released_on_fall': f'{sum(r["ok"] for r in falls)}/{len(falls)}',
           'pushes_recovered': len(no_fall),
           'kept_through_recovery': sum(r['held_at_end'] for r in no_fall),
           'tries': [{k: r[k] for k in ('toy', 'direction', 'speed', 'fell', 'released_on_fall')}
                     for r in results]}
    if len(out['tries']) and out['toys_fell'] < len(toys):
        out['pass'] = None      # some toy never fell: the release rule was not exercised
    else:
        out['pass'] = bool(falls) and all(r['ok'] for r in falls)
    return out


def offset_check(results: list[dict]) -> dict:
    out = {}
    for cm in OFFSETS_CM:
        rs = [r for r in results if r.get('offset_cm') == cm and r['ran']]
        expected = sum(max(0.0, 1 - (r['e_xy_m'] / 0.04) ** 2) for r in rs
                       if r['e_xy_m'] is not None and r['e_xy_m'] < 0.04)
        got = sum(r['grasped'] for r in rs)
        out[f'{cm}cm'] = {'n': len(rs), 'grasped': got, 'expected': round(expected, 2),
                          'within_1': abs(got - expected) <= 1.0}
    return out


def score(s, rows) -> dict:
    parts: dict[str, list] = {}
    for record in s.trials:
        part = record['part']
        if part == 'b_mech':
            parts.setdefault(part, []).append(score_mech(s, rows, record))
        elif part == 'b_fall':
            parts.setdefault(part, []).append(score_fall(s, rows, record))
        else:
            parts.setdefault(part, []).append(score_pick(s, rows, record))
    verdict = {}
    quick = s.args.quick
    if 'b_mech' in parts:
        res = parts['b_mech']
        verdict['b_mech'] = {'ok': f'{sum(r["ok"] for r in res)}/{len(res)}',
                             'pass': None if quick else all(r['ok'] and r['follows_head']
                                                            and r['dropped_on_open'] for r in res)}
    if 'b_fall' in parts:
        check = fall_check(parts['b_fall'])
        verdict['b_fall'] = {**check, 'pass': None if quick else check['pass']}
    if 'b_phi' in parts:
        res = [r for r in parts['b_phi'] if r['ran']]
        by_phi = {}
        for r in res:
            by_phi.setdefault(r['phi'], []).append(r)
        verdict['b_phi'] = {str(k): {'n': len(v), 'grasped': sum(r['grasped'] for r in v),
                                     'e_xy_m': metrics.summary([r['e_xy_m'] for r in v
                                                                if r['e_xy_m'] is not None]),
                                     'e_z_m': metrics.summary([r['e_z_m'] for r in v
                                                               if r['e_z_m'] is not None])}
                            for k, v in sorted(by_phi.items())}
        verdict['b_phi']['chosen'] = s.meta.get('phi_close_chosen')
    if 'b_ideal' in parts:
        res = parts['b_ideal']
        ran = [r for r in res if r['ran']]
        stable = all(r.get('stable_after_handback_s') is not None
                     and r['stable_after_handback_s'] <= STABLE_WITHIN_S for r in ran)
        lifted = all((r.get('lift_m') or 0) >= LIFT_M for r in ran if r['grasped'])
        verdict['b_ideal'] = {
            'grasped': f'{sum(r["grasped"] for r in res)}/{len(res)}',
            'held_read_matches': sum(r['held_read'] == r['grasped'] for r in ran),
            'falls': sum(r['fell'] for r in res), 'stable_within_1s': stable,
            'lifted_3cm': lifted, 'dropped_on_open': sum(bool(r.get('dropped')) for r in ran),
            'e_xy_m': metrics.summary([r['e_xy_m'] for r in ran if r['e_xy_m'] is not None]),
            'pass': None if quick else bool(ran and len(ran) == len(res)
                                            and all(r['grasped'] for r in res)
                                            and stable and lifted
                                            and not any(r['fell'] for r in res)),
        }
    if 'b_offset' in parts:
        check = offset_check(parts['b_offset'])
        verdict['b_offset'] = {**check, 'pass': None if quick else all(
            c['within_1'] for c in check.values())}
    verdict['note'] = ('plan 1B: ideal spot 30/30, stable within 1 s of the hand-back, held toys '
                       'rise >= 3 cm; offsets: successes within 1 of the 1 - (e/4 cm)^2 sum; '
                       'fall: the try that knocks the robot over lets the toy go (pass null when '
                       'no try knocked a toy\'s robot over); pass is null for --quick')
    return {'verdict': verdict, 'parts': parts}


def check_ground_pick(s) -> None:
    chosen = (s.args.parts or ','.join(PARTS)).split(',')
    if not {'b_phi', 'b_ideal', 'b_offset'} & set(chosen):
        return
    slots = {slot['slot']: slot for slot in s.meta['policies']['slots']}
    if not slots.get('ground_pick', {}).get('path'):
        raise RuntimeError('robotd has no ground_pick policy loaded; start it with the s2 config')


def main(argv=None) -> int:
    return common.main('b', __doc__, PARTS, score, argv, check=check_ground_pick)


if __name__ == '__main__':
    sys.exit(main())
