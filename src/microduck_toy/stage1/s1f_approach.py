"""Stage 1F: approach and pick from a seeded start, with InchingController.decide in the loop.

    python -m microduck_toy.stage1.s1f_approach [--parts f_main,f_blind] [--per-target 10]
                                                [--blind-starts 10] [--controller module:Class]
                                                [--quick]
    (needs the arena body with its camera: run_arena_body.zsh --camera)

Start (plan section 8, seeds 2000 + i, fixed): the robot at (-0.5, 0) facing delta ~ U(-10, 10)
deg; the target toy on the x axis 0.25 +- 0.03 m beyond the beak tip at H0 (7.9 cm ahead of the
trunk, stage0_facts head_geometry_mac), left ~ U(-5, 5) cm, toy heading ~ U(0, 180) deg; the other
toys parked. Starts go bus, henry, ketchup, bus, ... so --per-target 3 (the board) is the first 9.
f_main  per start, up to 3 mouth closes. One approach = B segment + C segment
        (control/approach.py, decide() from --controller, default InchingController). C says
        pick -> control.pick.ground_pick at stage1_facts grasp.phi_close; a close that does not
        hold, or C says retreat -> the retreat pulse (B10), approach again; abort (a fall) ->
        head H0, wait for velstand, approach again. At most 6 approaches or 120 s per start.
f_blind control group (plan 1E, a prototype here; the comparison is Stage 4): B segment, then
        no more looks: turn and walk on odometry so the toy should end at the target, head H2,
        pick. Same 3-close rule. --blind-starts N runs the first N starts of f_main (default 10;
        30 = the same starts as f_main, for a paired comparison).
Scored on truth only: per close, the beak-to-toy distance and the weld result; per pick, the
toy's position in the trunk frame at the pick trigger vs the target (8.0, 1.5 cm).
Pass (plan): grasps per close >= 50% -> grasp stays the main line, else switch to pushing.
Watched: RMS park error <= 2 cm, starts holding within 3 closes >= 80%, mean time per start
<= 40 s, 0 falls.
"""
from __future__ import annotations

import importlib
import json
import math
import random
import sys
import time

from microduck_toy.control import approach, inching, ranging
from microduck_toy.control.inching import ABORT, PICK, RETREAT
from microduck_toy.control.pick import ground_pick
from microduck_toy.control.robot import UPRIGHT
from microduck_toy.paths import repo_root
from microduck_toy.scene import toys as toy_lib
from microduck_toy.stage1 import common, metrics

TARGETS = [toy_lib.SHORT[n] for n in toy_lib.targets()]
ROBOT_AT = (-0.5, 0.0)
SEED_BASE = 2000
GAP_M = (0.22, 0.28)
LEFT_M = 0.05
ROBOT_YAW_DEG = 10.0
MAX_CLOSES, MAX_APPROACHES, MAX_START_S = 3, 6, 120.0
SUCCESS_MIN, PARK_RMS_M, WITHIN_3_MIN, START_S_MAX = 0.5, 0.02, 0.8, 40.0


def stage1_facts() -> dict:
    return json.loads((repo_root() / 'configs' / 'control' / 'stage1_facts.json').read_text())


def starts(per_target: int) -> list[dict]:
    out = []
    for k in range(per_target):
        for j, toy in enumerate(TARGETS):
            i = k * len(TARGETS) + j
            rng = random.Random(SEED_BASE + i)
            out.append({'i': i, 'seed': SEED_BASE + i, 'toy': toy,
                        'gap_m': rng.uniform(*GAP_M), 'left_m': rng.uniform(-LEFT_M, LEFT_M),
                        'toy_yaw_deg': rng.uniform(0.0, 180.0),
                        'robot_yaw_deg': rng.uniform(-ROBOT_YAW_DEG, ROBOT_YAW_DEG)})
    return out


def make_controller(spec: str | None, stage1: dict, facts: dict):
    cls = inching.InchingController
    if spec:
        module, name = spec.split(':')
        cls = getattr(importlib.import_module(module), name)
    return cls.from_config(stage1['inching'], facts)


class Kit:
    """What one run needs: configs, eyes, controller, retreat pulse."""

    def __init__(self, s):
        self.stage1 = s.cfg
        self.facts = stage1_facts()
        self.tip_m = s.facts['head_geometry_mac']['H0']['tip_forward_cm'] / 100
        self.phi = float(self.facts['grasp']['phi_close'])
        self.ctl = make_controller(s.args.controller, self.stage1, self.facts)
        self.retreat = inching.load_pulses(self.facts, [self.stage1['inching']['retreat']])[
            self.stage1['inching']['retreat']]

        def next_boxes(after: int):
            view = s.probe.call('view', after=after, rgb=False, timeout=5.0)
            return view['seq'], view['boxes']

        self.eyes = approach.Eyes(next_boxes, ranging.CameraModel.from_config(s.cfg['camera']),
                                  self.facts['ranging'])
        s.meta['f'] = {'controller': s.args.controller or 'microduck_toy.control.inching:InchingController',
                       'pulses': sorted(self.ctl.pulses), 'phi_close': self.phi,
                       'aim_right_of_toy_m': self.stage1['inching']['aim_right_of_toy_m'],
                       'tip_forward_m': self.tip_m}


def place(s, kit: Kit, start: dict) -> None:
    yaw = math.radians(start['robot_yaw_deg'])
    toy_xy = (ROBOT_AT[0] + kit.tip_m + start['gap_m'], ROBOT_AT[1] + start['left_m'])
    s.reset(head='H0', robot=(ROBOT_AT[0], ROBOT_AT[1], yaw),
            toys={start['toy']: [toy_xy[0], toy_xy[1], math.radians(start['toy_yaw_deg'])]})
    s.wait_settled(3.0)


def pick(s, kit: Kit, label: str) -> dict:
    pick_cfg = s.cfg['pick']
    before = s.grasp_count()
    result = ground_pick(s.robot, kit.phi, float(pick_cfg['period_s']),
                         float(s.cfg['grasp']['held_threshold_deg']),
                         confirm_s=float(pick_cfg['confirm_after_s']),
                         mark=lambda tag: s.mark(f'{label}_{tag}'))
    result['grasps'] = s.grasps_since(before)
    return result


def after_fall(s) -> None:
    s.robot.move(0.0, 0.0, 0.0)
    s.robot.head('H0')
    try:
        s.robot.wait(lambda st: st['policy'] in UPRIGHT and not st['safety'].get('fallen'), 8.0,
                     'velstand up')
    except TimeoutError:
        s.events.log('warning', text='still down 8 s after a fall')
    time.sleep(1.0)


def blind_move(s, kit: Kit, toy_at) -> dict:
    """Lab-style last part: from one look, turn and walk on odometry so the toy should end at the
    target; no more looks."""
    cfg = s.cfg['approach']
    tx, ty = s.cfg['inching']['target_m']
    x, y = toy_at
    r = math.hypot(x, y)
    side = math.sqrt(max(0.0, r * r - ty * ty))
    beta = math.atan2(y, x) - math.atan2(ty, side)
    moves = []
    if abs(beta) > math.radians(cfg['turn_tol_deg']):
        moves.append(approach.turn(s.robot, beta, cfg))
        beta = 0.0
    moves.append(approach.walk(s.robot, side - tx, beta, cfg))
    s.robot.head(cfg['c_head'])
    time.sleep(s.cfg['inching']['settle_s'])
    return {'from': list(toy_at), 'moves': moves}


def run_start(s, kit: Kit, start: dict, blind: bool) -> dict:
    part = 'f_blind' if blind else 'f_main'
    place(s, kit, start)
    toy = start['toy']
    plane_z = float(s.meta['toys'][toy]['rest_z'])
    record = {'part': part, **start, 'start': s.mark(f'{part}_{start["i"]}'), 'approaches': []}
    falls0 = len(s.robot.falls)
    closes, held = 0, False
    while (closes < MAX_CLOSES and len(record['approaches']) < MAX_APPROACHES
           and time.monotonic() - record['start'] < MAX_START_S and not held):
        n = len(record['approaches'])
        label = f'{part}_{start["i"]}_{n}'
        ap = {'n': n, 'begin': s.mark(label)}
        record['approaches'].append(ap)
        ap['b'] = approach.b_segment(s.robot, kit.eyes, toy, plane_z, s.cfg)
        if ap['b']['end'] != 'ready':
            ap['end'] = f'b_{ap["b"]["end"]}'
            approach.send_pulse(s.robot, kit.retreat)
            continue
        if blind:
            ap['blind'] = blind_move(s, kit, ap['b']['steps'][-1]['toy'])
            end = PICK
        else:
            ap['c'] = approach.c_segment(s.robot, kit.eyes, toy, plane_z, kit.ctl, s.cfg)
            end = ap['c']['end']
        ap['end'] = end
        if end == PICK:
            ap['pick'] = pick(s, kit, label)
            if not ap['pick'].get('ran'):
                ap['end'] = 'pick_not_run'      # robotd has no ground_pick slot: stop this start
                break
            closes += 1
            held = bool(ap['pick'].get('held'))
            if not held:
                time.sleep(0.5)
                approach.send_pulse(s.robot, kit.retreat)
        elif end == ABORT:
            after_fall(s)
        else:                                   # retreat or runaway
            approach.send_pulse(s.robot, kit.retreat)
    record['closes'], record['held'] = closes, held
    record['falls_seen'] = s.robot.falls[falls0:]
    record['end'] = s.mark(f'{part}_{start["i"]}_end')
    s.robot.mouth(1.0)                          # let go before the next start
    time.sleep(0.8)
    s.robot.mouth(0.0)
    return record


def part_for(blind: bool):
    def part(s) -> None:
        kit = Kit(s)
        per = 1 if s.args.quick else int(s.args.per_target)
        chosen = starts(per)
        if blind:
            n = 1 if s.args.quick else int(s.args.blind_starts)
            chosen = starts(max(per, -(-n // len(TARGETS))))[:n]
        for k, start in enumerate(chosen):
            s.trial(run_start(s, kit, start, blind))
            r = s.trials[-1]
            print(f'{r["part"]} {k + 1}/{len(chosen)} {r["toy"]}: closes {r["closes"]}, '
                  f'held {r["held"]}, approaches {len(r["approaches"])}', flush=True)
    return part


PARTS = {'f_main': part_for(False), 'f_blind': part_for(True)}


# ── scoring (truth only) ────────────────────────────────────────────────

def grasp_p(err, tol: float) -> float:
    """The weld model's chance if the beak landed exactly on the target: 1 - (e / tol)^2."""
    return max(0.0, 1.0 - (math.hypot(*err) / tol) ** 2)


def close_entry(grasps: list[dict]) -> dict | None:
    return next((g for g in grasps if 'nearest_exy' in g), None)


def score_start(s, rows, record) -> dict:
    tx, ty = s.cfg['inching']['target_m']
    toy = record['toy']
    out = {k: record[k] for k in ('part', 'i', 'toy', 'gap_m', 'left_m', 'toy_yaw_deg',
                                  'robot_yaw_deg', 'closes', 'held')}
    picks, ends, pulses = [], [], []
    for ap in record['approaches']:
        ends.append(ap.get('end'))
        if 'c' in ap:
            pulses.append(ap['c'].get('pulses', 0))
        if 'pick' not in ap:
            continue
        p = ap['pick']
        g = close_entry(p.get('grasps', []))
        entry = {'ran': bool(p.get('ran')), 'grasped': bool(g and g.get('result') == 'grasped'),
                 'result': g and g.get('result'), 'e_xy_m': g and g.get('nearest_exy'),
                 'p': g and g.get('p'), 'held_read': p.get('held')}
        if 'trigger' in p:
            fwd, left = metrics.toy_in_trunk(metrics.at_mark(rows, p['trigger']), toy)
            entry['park_err_m'] = [fwd - tx, left - ty]
            entry['close_after_start_s'] = p.get('close', p['trigger']) - record['start']
        picks.append(entry)
    out['picks'] = picks
    out['approach_ends'] = ends
    out['pulses'] = pulses
    out['fell'] = metrics.fall_time(rows, record['start'], record['end']) is not None
    out['falls_seen'] = len(record.get('falls_seen', []))
    out['start_s'] = record['end'] - record['start']
    out['held_within_3'] = any(pk['grasped'] for pk in [pk for pk in picks if pk['ran']][:3])
    out['grasp_p_at_trigger'] = [grasp_p(pk['park_err_m'], s.cfg['grasp']['tol_xy_m'])
                                 for pk in picks if pk.get('park_err_m')]
    return out


def verdict(results: list[dict]) -> dict:
    park = [pk['park_err_m'] for r in results for pk in r['picks'] if pk.get('park_err_m')]
    picks = [pk for r in results for pk in r['picks'] if pk['ran']]
    ends: dict[str, int] = {}
    for r in results:
        for e in r['approach_ends']:
            ends[str(e)] = ends.get(str(e), 0) + 1
    pulses = [p for r in results for p in r['pulses']]
    out = {
        'starts': len(results),
        'closes': len(picks),
        'grasped': sum(pk['grasped'] for pk in picks),
        'success_per_close': (sum(pk['grasped'] for pk in picks) / len(picks)) if picks else None,
        'park_err_rms_m': (math.sqrt(sum(a * a + b * b for a, b in park) / len(park)) if park else None),
        'park_err_mean_m': ([sum(a for a, _ in park) / len(park), sum(b for _, b in park) / len(park)]
                            if park else None),
        'held_within_3': metrics.rate([r['held_within_3'] for r in results]),
        'mean_start_s': (sum(r['start_s'] for r in results) / len(results)) if results else None,
        'falls': sum(r['fell'] for r in results),
        'approach_ends': ends,
        'mean_pulses_per_c': (sum(pulses) / len(pulses)) if pulses else None,
        'park_grasp_p_mean': (lambda v: sum(v) / len(v) if v else None)(
            [p for r in results for p in r['grasp_p_at_trigger']]),
        'by_toy': {},
    }
    for toy in TARGETS:
        tp = [pk for r in results if r['toy'] == toy for pk in r['picks'] if pk['ran']]
        out['by_toy'][toy] = {'closes': len(tp), 'grasped': sum(pk['grasped'] for pk in tp)}
    if out['success_per_close'] is None:
        out['pass'] = None
    else:
        out['pass'] = out['success_per_close'] >= SUCCESS_MIN
    out['watch'] = {
        'park_rms_le_2cm': out['park_err_rms_m'] is not None and out['park_err_rms_m'] <= PARK_RMS_M,
        'held_within_3_ge_80pct': (out['held_within_3']['rate'] or 0) >= WITHIN_3_MIN,
        'mean_start_le_40s': out['mean_start_s'] is not None and out['mean_start_s'] <= START_S_MAX,
        'no_falls': out['falls'] == 0,
    }
    return out


def score(s, rows) -> dict:
    parts: dict[str, list] = {}
    for record in s.trials:
        parts.setdefault(record['part'], []).append(score_start(s, rows, record))
    out = {'results': parts, 'verdict': {name: verdict(rs) for name, rs in parts.items()}}
    out['verdict']['note'] = ('plan 1F: grasps per close >= 50% keeps grasping as the main line; '
                              'watched: RMS park error <= 2 cm, starts holding within 3 closes >= 80%, '
                              'mean time per start <= 40 s, 0 falls. f_blind is the control-group '
                              'prototype (formal comparison in Stage 4)')
    return out


def extra_args(p) -> None:
    p.add_argument('--per-target', type=int, default=10, help='starts per target toy (board: 3)')
    p.add_argument('--blind-starts', type=int, default=10,
                   help='f_blind: the first N starts of f_main (default 10)')
    p.add_argument('--controller', default=None,
                   help='module:Class for decide() (default microduck_toy.control.inching:InchingController)')


def check_camera(s) -> None:
    if not s.meta.get('camera'):
        raise RuntimeError('the arena body has no camera; start run_arena_body.zsh with --camera')


def main(argv=None) -> int:
    return common.main('f', __doc__, PARTS, score, argv, check=check_camera, extra=extra_args)


if __name__ == '__main__':
    sys.exit(main())
