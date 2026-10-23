"""Stage 1G: carry a held toy. Stand, walk and turn with it in the beak; it must stay there.

    python -m microduck_toy.stage1.s1g_carry [--reps 3] [--quick]

Plan section 9: after a grasp, stand 5 s; walk 0.5 m at H0 and at H1, each at vx 0.2 and 0.3 m/s;
turn in place 90 deg left and 90 deg right; 3 targets x 3. The toys weigh 13-29 g: GroundPick was
trained with 10-40 g in the beak, walk_17500 and velstand were not, so this has to be measured.

One trial = load, stand, one route, let go:
  load   the robot at ROUTE_START facing +x, head H2, the toy put at 1B's ideal spot (8.0 cm
         ahead, 1.5 cm left of the trunk; 1B: 30/30 grasped there) at heading 0/60/120 deg by
         rep, GroundPick at stage1_facts grasp.phi_close. Up to LOAD_PICKS picks; if none holds,
         the test-port fixture of 1B b_mech (mouth open, toy at the beak tip, mouth closed), so
         every trial carries. The trial records which one loaded the toy.
  stand  head H0, STAND_S.
  route  LEGS in order: walk 0.5 m at H0 0.2, turn left 90, H0 0.3, left 90, H1 0.2, right 90,
         H1 0.3, right 90. Walks and turns are the B segment's (control/approach.py: heading PI
         and turn on robotd's odometry, nothing from the simulator); settle_s standing between
         moves. The route stays inside x -0.6..-0.1, y -0.5..0.5 (walls at +-1 m, tray x >= 0.5).
         A fall seen by the fall guard ends the route (the arena body lets the toy go on a fall).
  end    mouth open 1 s (let go), mouth closed.
After every step the robot's own check is read too: the mouth joint from robot.state above
grasp.held_threshold_deg = holding (what the state machine will use; no truth).
Scored on truth: the arena body's `holding` column, falls (Stage 0 rule), each walk's distance,
drift and time, each turn's angle, the toy's sway in the beak (toy - beak tip, against the start of
the step) and the trunk's largest tilt.
Pass (plan): 0 falls and 0 drops. Fallback (plan): a fall -> carry at H1 and 0.2 m/s only and
measure again; still falling -> stop and discuss. So the verdict also lists every (head, speed).
"""
from __future__ import annotations

import json
import math
import sys
import time

from microduck_toy.control import approach
from microduck_toy.control.pick import ground_pick, mouth_angle
from microduck_toy.control.robot import UPRIGHT
from microduck_toy.paths import repo_root
from microduck_toy.scene import toys as toy_lib
from microduck_toy.stage1 import common, metrics, s1b_grasp

TARGETS = [toy_lib.SHORT[n] for n in toy_lib.targets()]
ROUTE_START = (-0.6, -0.5, 0.0)
STAND_S = 5.0
WALK_M = 0.5
TURN_DEG = 90.0
LOAD_PICKS = 3
TOY_YAW_DEG = (0.0, 60.0, 120.0)            # by rep
# (kind, head, value): walks carry vx (m/s), turns carry the angle (deg, left > 0).
LEGS = (('walk', 'H0', 0.2), ('turn', 'H0', TURN_DEG),
        ('walk', 'H0', 0.3), ('turn', 'H0', TURN_DEG),
        ('walk', 'H1', 0.2), ('turn', 'H1', -TURN_DEG),
        ('walk', 'H1', 0.3), ('turn', 'H1', -TURN_DEG))


def stage1_facts() -> dict:
    return json.loads((repo_root() / 'configs' / 'control' / 'stage1_facts.json').read_text())


def leg_name(kind: str, head: str, value: float) -> str:
    if kind == 'walk':
        return f'walk_{head}_{value:.1f}'
    return 'turn_left' if value > 0 else 'turn_right'


def held_read(s) -> bool:
    """The robot's own check: a toy in the beak stops the mouth short of closed."""
    return math.degrees(mouth_angle(s.robot.snapshot())) > float(s.cfg['grasp']['held_threshold_deg'])


def fallen(s, falls0: int) -> bool:
    return len(s.robot.falls) > falls0 or bool(s.robot.snapshot()['safety'].get('fallen'))


def load(s, toy: str, rep: int, phi: float, label: str) -> dict:
    """Get the toy into the beak: GroundPick from the ideal spot, else the fixture."""
    pick_cfg = s.cfg['pick']
    fwd, left = float(pick_cfg['ideal_toy_forward_m']), float(pick_cfg['ideal_toy_left_m'])
    out = {'picks': [], 'by': None}
    for k in range(LOAD_PICKS):
        s.reset(head='H2', robot=ROUTE_START)
        s.wait_settled(3.0)
        s.place_in_trunk(toy, fwd, left, math.radians(TOY_YAW_DEG[rep % len(TOY_YAW_DEG)]))
        time.sleep(0.3)
        before = s.grasp_count()
        result = ground_pick(s.robot, phi, float(pick_cfg['period_s']),
                             float(s.cfg['grasp']['held_threshold_deg']),
                             confirm_s=float(pick_cfg['confirm_after_s']),
                             mark=lambda tag: s.mark(f'{label}_load{k}_{tag}'))
        result['grasps'] = s.grasps_since(before)
        out['picks'].append(result)
        if result.get('held') and not result.get('fallen'):
            out['by'] = 'pick'
            return out
        s.robot.mouth(1.0)                  # let go of anything half-held before the next try
        time.sleep(0.8)
        s.robot.mouth(0.0)
    s.reset(head='H0', robot=ROUTE_START)
    s.wait_settled(3.0)
    s1b_grasp.hold_at_tip(s, toy)
    out['by'] = 'fixture'
    return out


def run_leg(s, kind: str, head: str, value: float, cfg: dict) -> dict:
    if kind == 'walk':
        return approach.walk(s.robot, WALK_M, 0.0, {**cfg, 'walk_vx_mps': value})
    return approach.turn(s.robot, math.radians(value), cfg)


def trial(s, toy: str, rep: int, phi: float) -> dict:
    label = f'g_{toy}_{rep}'
    cfg, settle = s.cfg['approach'], float(s.cfg['inching']['settle_s'])
    record = {'part': 'g_carry', 'toy': toy, 'rep': rep, 'start': s.mark(label)}
    record['load'] = load(s, toy, rep, phi, label)
    s.robot.head('H0')
    falls0 = len(s.robot.falls)
    record['carry'] = s.mark(f'{label}_carry')
    record['held_read_at_carry'] = held_read(s)
    steps = []
    time.sleep(STAND_S)
    steps.append({'name': 'stand', 'kind': 'stand', 'head': 'H0', 'begin': record['carry'],
                  'end': s.mark(f'{label}_stand_end'), 'held_read': held_read(s)})
    for kind, head, value in LEGS:
        if fallen(s, falls0):
            break
        name = leg_name(kind, head, value)
        if head != steps[-1]['head']:
            s.robot.head(head)
            time.sleep(settle)
        step = {'name': name, 'kind': kind, 'head': head, 'value': value,
                'begin': s.mark(f'{label}_{name}')}
        step['move'] = run_leg(s, kind, head, value, cfg)
        time.sleep(settle)
        step['end'] = s.mark(f'{label}_{name}_end')
        step['held_read'] = held_read(s)
        steps.append(step)
    record['steps'] = steps
    record['fell_seen'] = s.robot.falls[falls0:]
    if record['fell_seen']:
        s.robot.move(0.0, 0.0, 0.0)
        s.robot.head('H0')
        try:
            s.robot.wait(lambda st: st['policy'] in UPRIGHT and not st['safety'].get('fallen'),
                         8.0, 'velstand up')
        except TimeoutError:
            s.events.log('warning', text='still down 8 s after a fall')
    record['end'] = s.mark(f'{label}_end')
    s.robot.mouth(1.0)
    time.sleep(1.0)
    s.robot.mouth(0.0)
    return record


def g_carry(s) -> None:
    phi = float(stage1_facts()['grasp']['phi_close'])
    s.meta['g'] = {'phi_close': phi, 'route_start': list(ROUTE_START), 'legs': [list(l) for l in LEGS],
                   'stand_s': STAND_S, 'walk_m': WALK_M}
    reps = 1 if s.args.quick else int(s.args.reps)
    toys = TARGETS[:1] if s.args.quick else TARGETS
    for rep in range(reps):
        for toy in toys:
            s.trial(trial(s, toy, rep, phi))
            r = s.trials[-1]
            print(f'g_carry {toy} {rep + 1}/{reps}: loaded by {r["load"]["by"]}, '
                  f'{len(r["steps"]) - 1}/{len(LEGS)} legs, falls seen {len(r["fell_seen"])}, '
                  f'still held (robot) {r["steps"][-1]["held_read"]}', flush=True)


PARTS = {'g_carry': g_carry}


# ── scoring (truth only) ────────────────────────────────────────────────

def tilt_deg(row: dict) -> float:
    g = math.sqrt(row['gx'] ** 2 + row['gy'] ** 2 + row['gz'] ** 2) or 1.0
    return math.degrees(math.acos(max(-1.0, min(1.0, -row['gz'] / g))))


def in_beak(row: dict, toy: str) -> tuple[float, float, float]:
    return (row[f'{toy}_x'] - row['tip_x'], row[f'{toy}_y'] - row['tip_y'],
            row[f'{toy}_z'] - row['tip_z'])


def score_step(rows: list[dict], toy: str, step: dict) -> dict:
    span = metrics.window(rows, step['begin'], step['end'])
    a, b = metrics.at_mark(rows, step['begin']), metrics.at_mark(rows, step['end'])
    out = {'name': step['name'], 'kind': step['kind'], 'head': step['head'],
           'value': step.get('value'), 'seconds': step['end'] - step['begin'],
           'held': bool(span) and all(r['holding'] == toy for r in span),
           'held_read': step['held_read'],
           'fell': metrics.fall_time(rows, step['begin'], step['end']) is not None,
           'max_tilt_deg': max((tilt_deg(r) for r in span), default=None)}
    if span and all(r['holding'] == toy for r in span):
        ref = in_beak(span[0], toy)
        out['sway_m'] = max(math.dist(in_beak(r, toy), ref) for r in span)
    rel = metrics.relative(a, b)
    if step['kind'] == 'walk':
        out.update(forward_m=rel['forward'], drift_m=rel['left'], dyaw_deg=math.degrees(rel['dyaw']))
    elif step['kind'] == 'turn':
        out.update(turned_deg=math.degrees(rel['dyaw']), moved_m=math.hypot(rel['forward'], rel['left']))
    return out


def score_trial(s, rows, record) -> dict:
    toy = record['toy']
    carry = metrics.at_mark(rows, record['carry'])
    steps = [score_step(rows, toy, st) for st in record['steps']]
    route = metrics.window(rows, record['carry'], record['steps'][-1]['end'])
    picks = record['load']['picks']
    return {'toy': toy, 'rep': record['rep'], 'loaded_by': record['load']['by'],
            'load_picks': len(picks),
            'held_at_carry': carry['holding'] == toy,
            'held_read_at_carry': record['held_read_at_carry'],
            'legs_done': len(record['steps']) - 1,
            'dropped': carry['holding'] == toy and any(r['holding'] != toy for r in route),
            'fell': metrics.fall_time(rows, record['carry'], record['end']) is not None,
            'falls_seen': len(record['fell_seen']),
            'robot_read_matches': all(st['held'] == st['held_read'] for st in steps),
            'steps': steps}


def verdict(results: list[dict], quick: bool) -> dict:
    steps = [st for r in results for st in r['steps']]
    out = {'trials': len(results),
           'loaded_by': {k: sum(r['loaded_by'] == k for r in results) for k in ('pick', 'fixture')},
           'falls': sum(r['fell'] for r in results),
           'drops': sum(r['dropped'] for r in results),
           'not_held_at_carry': sum(not r['held_at_carry'] for r in results),
           'routes_complete': sum(r['legs_done'] == len(LEGS) for r in results),
           'robot_read_matches': sum(r['robot_read_matches'] for r in results),
           'stand_sway_m': metrics.summary([st.get('sway_m') for st in steps if st['kind'] == 'stand']),
           'by_step': {}}
    for name in ['stand'] + list(dict.fromkeys(leg_name(*leg) for leg in LEGS)):
        ss = [st for st in steps if st['name'] == name]
        entry = {'n': len(ss), 'falls': sum(st['fell'] for st in ss),
                 'not_held': sum(not st['held'] for st in ss),
                 'sway_m': metrics.summary([st.get('sway_m') for st in ss]),
                 'max_tilt_deg': metrics.summary([st['max_tilt_deg'] for st in ss])}
        if name.startswith('walk'):
            entry.update(forward_m=metrics.summary([st['forward_m'] for st in ss]),
                         drift_m=metrics.summary([st['drift_m'] for st in ss]),
                         seconds=metrics.summary([st['seconds'] for st in ss]))
        elif name.startswith('turn'):
            entry.update(turned_deg=metrics.summary([st['turned_deg'] for st in ss]))
        out['by_step'][name] = entry
    ok = out['falls'] == 0 and out['drops'] == 0 and out['not_held_at_carry'] == 0
    out['pass'] = None if quick or not results else ok
    return out


def score(s, rows) -> dict:
    results = [score_trial(s, rows, r) for r in s.trials]
    v = verdict(results, bool(s.args.quick))
    v['note'] = ('plan 1G: 0 falls and 0 drops over 3 targets x 3 (stand 5 s, H0 and H1 walks at '
                 '0.2 and 0.3 m/s, 90 deg turns both ways); a fall -> carry at H1 0.2 m/s only and '
                 'measure again. In the sim a weld only lets go when the mouth opens or the robot '
                 'falls, so a drop here means a fall or a mouth command; real slip is not modelled')
    return {'results': results, 'verdict': v}


def extra_args(p) -> None:
    p.add_argument('--reps', type=int, default=3, help='trials per target toy (plan: 3)')


def check_ground_pick(s) -> None:
    slots = {slot['slot']: slot for slot in s.meta['policies']['slots']}
    if not slots.get('ground_pick', {}).get('path'):
        raise RuntimeError('robotd has no ground_pick policy loaded; start it with the s2 config')


def main(argv=None) -> int:
    return common.main('g', __doc__, PARTS, score, argv, check=check_ground_pick, extra=extra_args)


if __name__ == '__main__':
    sys.exit(main())
