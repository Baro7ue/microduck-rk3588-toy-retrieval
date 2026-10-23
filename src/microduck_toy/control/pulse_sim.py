"""Closed-loop check of the C segment controller without MuJoCo, from Stage 1 measurements only:
each pulse is drawn from its measured mean and sd (1C), each look has the measured ranging error
(1D), and the toy is seen only where the H2 camera sees it (1D's camera pose).

    python -m microduck_toy.control.pulse_sim [--episodes 500] [--seed 1] [--pulses F2,F5,L2]
                                              [--aim 0.03] [--controller module:Class] [--show 3]

An episode starts where the B segment is assumed to leave the toy (stage1.json inching.sim) and
ends when the controller says pick, retreat or abort. A pick is scored on the truth: parked = the
toy really was in the window; grasp_p = the weld model's chance 1 - (e / 4 cm)^2 with e the
distance from the beak's landing point (stage1.json grasp.tol_xy_m). The real 1F run replaces all
of this; the sim is for writing and tuning decide().
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import statistics

from microduck_toy.control import ranging
from microduck_toy.control.inching import (ABORT, PICK, RETREAT, InchingController, Look,
                                           move_toy)
from microduck_toy.paths import repo_root


def load_configs() -> tuple[dict, dict]:
    control = repo_root() / 'configs/control'
    stage1 = json.loads((control / 'stage1.json').read_text())
    facts = json.loads((control / 'stage1_facts.json').read_text())
    return stage1, facts


class H2View:
    """Is the toy centre in the H2 image, far enough from the edges for a usable box?"""

    def __init__(self, camera_cfg: dict, frame: dict, plane_z: float, margin_px: float,
                 bottom_px: float):
        self.cam = ranging.CameraModel.from_config(camera_cfg)
        self.state = {'safety': {'gravity': frame['gravity']},
                      'odom': {'position': [0.0, 0.0, frame['trunk_z_m']]},
                      'frames': {'camera': {'pos': frame['camera_pos_m'],
                                            'quat': frame['camera_quat']}}}
        self.plane_z = plane_z
        self.margin = margin_px
        self.bottom = bottom_px

    @classmethod
    def from_configs(cls, stage1: dict, facts: dict) -> 'H2View':
        sim = stage1['inching']['sim']
        return cls(stage1['camera'], facts['ranging']['H2']['camera_frame'], sim['toy_centre_z_m'],
                   sim['edge_margin_px'], sim['bottom_row_px'])

    def pixel(self, toy) -> tuple[float, float] | None:
        return ranging.project(self.cam, self.state, (toy[0], toy[1], self.plane_z))

    def visible(self, toy) -> bool:
        px = self.pixel(toy)
        if px is None:
            return False
        col, row = px
        return (self.margin <= col <= self.cam.width - self.margin
                and self.margin <= row <= self.bottom)


def sample_pulse(pulse, rng: random.Random) -> tuple[float, float, float]:
    """One draw of (forward_m, left_m, yaw_rad) from the measured mean and sd."""
    return (rng.gauss(*pulse.forward_m), rng.gauss(*pulse.left_m), rng.gauss(*pulse.yaw_rad))


def grasp_p(err_m: float, tol_m: float) -> float:
    return max(0.0, 1.0 - (err_m / tol_m) ** 2)


def run_episode(ctl: InchingController, start, rng: random.Random, stage1: dict, view: H2View,
                show: bool = False) -> dict:
    """One C segment from `start` (the toy's true position). Returns how it ended and the truth."""
    inch, sim = stage1['inching'], stage1['inching']['sim']
    frames, step_s = int(inch['frames']), float(inch['settle_s'])
    look_s = frames / float(sim['camera_fps'])
    bias = (rng.gauss(0, sim['bias_sd_m']), rng.gauss(0, sim['bias_sd_m']))
    toy, elapsed, actions = tuple(start), 0.0, []
    ctl.reset()
    end = 'runaway'
    for _ in range(int(sim['max_decisions'])):
        elapsed += look_s
        if view.visible(toy):
            seen = frames
            obs = (toy[0] + bias[0] + rng.gauss(0, sim['look_sd_m']),
                   toy[1] + bias[1] + rng.gauss(0, sim['look_sd_m']))
        else:
            seen, obs = 0, None
        action = ctl.decide(Look(toy=obs, seen=seen, frames=frames, elapsed_s=round(elapsed, 3)))
        if action not in ctl.actions():
            raise ValueError(f'decide() returned {action!r}, not one of {sorted(ctl.actions())}')
        actions.append(action)
        if show:
            print(f'  t {elapsed:5.1f} s  toy {toy[0] * 100:6.2f} {toy[1] * 100:6.2f} cm  '
                  f'seen {obs is not None}  -> {action}')
        if action in (PICK, RETREAT, ABORT):
            end = action
            break
        pulse = ctl.pulses[action]
        toy = move_toy(toy, *sample_pulse(pulse, rng))
        elapsed += pulse.seconds + step_s
    target = ctl.target
    err = math.hypot(*target.error(toy))
    out = {'start': list(start), 'end': end, 'pulses': sum(a in ctl.pulses for a in actions),
           'elapsed_s': round(elapsed, 2), 'actions': actions, 'truth_end_m': [toy[0], toy[1]],
           'unseen_at_end': not view.visible(toy)}
    if end == PICK:
        out.update(err_m=err, parked=target.inside(toy),
                   grasp_p=grasp_p(err, stage1['grasp']['tol_xy_m']))
    return out


def starts(n: int, rng: random.Random, stage1: dict, aim: float | None = None) -> list:
    """Where the B segment is assumed to leave the toy: forward uniform in start_forward_m, left
    uniform in aim +- start_left_spread_m (aim = inching.aim_right_of_toy_m: B aims the trunk that
    far right of the toy, so the toy starts that far left of the centre line)."""
    inch, sim = stage1['inching'], stage1['inching']['sim']
    aim = inch['aim_right_of_toy_m'] if aim is None else aim
    lo, hi = sim['start_forward_m']
    spread = sim['start_left_spread_m']
    return [(rng.uniform(lo, hi), rng.uniform(aim - spread, aim + spread)) for _ in range(n)]


def summarise(results: list[dict]) -> dict:
    n = len(results)
    picks = [r for r in results if r['end'] == PICK]
    ends = {}
    for r in results:
        key = r['end']
        if key == RETREAT:
            key = 'retreat_unseen' if r['unseen_at_end'] else 'retreat_seen'
        ends[key] = ends.get(key, 0) + 1
    return {
        'episodes': n,
        'parked': round(sum(r['parked'] for r in picks) / n, 3),
        'picked': round(len(picks) / n, 3),
        'wrong_picks': round(sum(not r['parked'] for r in picks) / max(1, len(picks)), 3),
        'grasp_p_per_close': round(statistics.mean(r['grasp_p'] for r in picks), 3) if picks else 0.0,
        'err_cm_rms_at_pick': (round(100 * math.sqrt(statistics.mean(r['err_m'] ** 2 for r in picks)), 2)
                               if picks else None),
        'mean_pulses': round(statistics.mean(r['pulses'] for r in results), 2),
        'mean_s': round(statistics.mean(r['elapsed_s'] for r in results), 1),
        'ends': dict(sorted(ends.items())),
    }


def simulate(ctl: InchingController, episodes: int = 500, seed: int = 1, stage1: dict | None = None,
             facts: dict | None = None, aim: float | None = None, show: int = 0) -> dict:
    """Run `episodes` C segments with one controller; same seed, same starts and same draws."""
    if stage1 is None or facts is None:
        stage1, facts = load_configs()
    view = H2View.from_configs(stage1, facts)
    rng = random.Random(seed)
    results = []
    for i, start in enumerate(starts(episodes, rng, stage1, aim)):
        if i < show:
            print(f'episode {i}: start {start[0] * 100:.1f} {start[1] * 100:.1f} cm')
        results.append(run_episode(ctl, start, rng, stage1, view, show=i < show))
    return {'summary': summarise(results), 'results': results}


def make_controller(spec: str | None, stage1: dict, facts: dict, pulses: list[str] | None):
    cfg = dict(stage1['inching'])
    if pulses:
        cfg['pulses'] = pulses
    cls = InchingController
    if spec:
        module, name = spec.split(':')
        cls = getattr(importlib.import_module(module), name)
    return cls.from_config(cfg, facts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--episodes', type=int, default=500)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--pulses', help='comma list, default stage1.json inching.pulses')
    ap.add_argument('--aim', type=float, help='metres; default inching.aim_right_of_toy_m')
    ap.add_argument('--controller', help='module:Class with from_config(cfg, facts); default '
                                         'microduck_toy.control.inching:InchingController')
    ap.add_argument('--show', type=int, default=0, help='print every look of the first N episodes')
    args = ap.parse_args(argv)
    stage1, facts = load_configs()
    pulses = args.pulses.split(',') if args.pulses else None
    ctl = make_controller(args.controller, stage1, facts, pulses)
    try:
        out = simulate(ctl, args.episodes, args.seed, stage1, facts, args.aim, args.show)
    except NotImplementedError as error:
        print(f'not written yet: {error}')
        return 2
    print(json.dumps({'pulses': sorted(ctl.pulses), **out['summary']}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
