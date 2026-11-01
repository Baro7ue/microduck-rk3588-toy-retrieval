"""The Stage 3 brain: one episode as a state machine over robot.state, camera boxes and the ToF.

    brain = Brain(robot, eyes, tof, cfg, target='bus', start=(x, y, yaw), log_path=...)
    result = brain.run()          # {'end': 'dropped' | 'held' | 'grasp_fail' | ..., ...}

On the board (plan 3F), one process per episode:

    python -m microduck_toy.stage3.brain --target bus --start X,Y,YAW --socket <robotd.sock> \
        --camera <mac>:7901 --tof <mac>:7801 --backend int8 --out <dir>

It is a loop over a switch: each state is one method that calls actions already measured in
Stages 1 and 2 and returns the next state (plan section 3):

    SPIN       turn in place at head 0.15 (window view); record what each frame shows
    LOOK       stand at H0 facing a candidate, 5 frames; confirm the target within 1.6 m
    GOTO       walk one leg (<= 0.5 m) towards the B segment start, around known toys
    APPROACH   control/approach.py b_segment with YOLO boxes (H1)
    LAST       control/approach.py blind_move (the B segment's last look, then odometry only; H2)
    PICK       control/pick.py ground_pick; held -> CARRY, else back off and APPROACH (<= 3 closes)
    CARRY      H1 to the chosen side's staging point, square up facing the tray
    TRAY_LOOK  H1 tray box -> where the tray centre is
    DROP       H0 walk until the trunk is d from the centre, open the beak, end
    FALLEN     (any state) wait for velstand, then carry on from what the mouth says
    RECOVER    the toy came out in a fall: look at the feet, else re-search

Global: a ToF stop on any walk (back off, route around it) and the 180 s clock. The brain reads no
simulator truth: the known start pose is given (question 1A), everything else comes from the robot.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import threading
import time

from microduck_toy.control import approach, ranging
from microduck_toy.control.inching import load_pulses
from microduck_toy.control.pick import ground_pick, mouth_angle
from microduck_toy.control.robot import Events, Robot, UPRIGHT
from microduck_toy.paths import repo_root
from microduck_toy.stage3 import shots as shots_lib, tray as tray_lib
from microduck_toy.stage3.world import (Field, FieldPose, Sightings, approach_point, bearing_to,
                                        plan_route, to_field, wrap)

TARGETS = ('bus', 'henry', 'ketchup')
STATES = ('SPIN', 'LOOK', 'GOTO', 'APPROACH', 'LAST', 'PICK', 'CARRY', 'TRAY_LOOK', 'DROP',
          'FALLEN', 'RECOVER', 'RELOCATE')
ENDS = ('dropped', 'held', 'grasp_fail', 'search_fail', 'timeout', 'error')


class Fallen(Exception):
    pass


class Timeout(Exception):
    pass


class Ended(Exception):
    def __init__(self, reason: str, **detail):
        super().__init__(reason)
        self.reason, self.detail = reason, detail


def load_configs(root: Path | None = None, override: str | Path | None = None) -> dict:
    """The four frozen configs. override (Stage 5 room demo, e.g. configs/demo/room.json, relative to
    the repo): its 'stage1' and 'stage3' sections are deep-merged over those two (stage5/room.py)."""
    root = root or repo_root()
    read = lambda p: json.loads((root / p).read_text())  # noqa: E731
    configs = {'stage3': read('configs/control/stage3.json'),
               'stage1': read('configs/control/stage1.json'),
               'facts1': read('configs/control/stage1_facts.json'),
               'stage2': read('configs/vision/stage2.json')}
    if override:
        from microduck_toy.stage5 import room
        doc = room.load(root / override)
        for name in ('stage1', 'stage3'):
            configs[name] = room.deep_merge(configs[name], doc.get(name, {}))
    return configs


def points_of(obstacles: list) -> list:
    """Route obstacles as choose_side wants them: toy entries as they are, ToF ((x, y), r) as (x, y)."""
    return [o for o in obstacles if not isinstance(o[0], (tuple, list))] + \
           [o[0] for o in obstacles if isinstance(o[0], (tuple, list))]


class Brain:
    def __init__(self, robot, eyes, tof, configs: dict, target: str, start, odom0=None,
                 log_path: Path | None = None, until: str = 'delivered', start_state: str = 'SPIN',
                 holding: bool = False, known=(), drop_d: float | None = None, side: str | None = None,
                 on_state=None, echo=None, frames_dir: Path | None = None):
        if target not in TARGETS:
            raise ValueError(f'{target!r} is not a target toy')
        if start_state not in STATES:
            raise ValueError(f'unknown start state {start_state!r}')
        self.robot, self.eyes, self.tof = robot, eyes, tof
        self.cfg, self.s1, self.f1 = configs['stage3'], configs['stage1'], configs['facts1']
        self.classes = list(configs['stage2']['classes'])
        self.target, self.until = target, until
        self.start_state, self.holding = start_state, holding
        self.fld = Field.from_config(self.s1, self.cfg)
        self.cam = ranging.CameraModel.from_config(self.s1['camera'])
        state = robot.snapshot()
        self.pose = FieldPose(start, odom0 if odom0 is not None else approach.odom_of(state))
        self.seen = Sightings(cluster_m=float(self.cfg['search']['cluster_m']))
        for cls, xy in known:                    # a fixture's prior map (3C c_route), else nothing
            self.seen.add(cls, xy, 1.0, 0.0).status = 'seen'
        self.extra: list = []                    # ToF stop points, [(x, y), r]
        self.drop_d = float(drop_d if drop_d is not None else self.cfg['drop']['d_m'])
        self.side_forced = side
        self.retreat = load_pulses(self.f1, [self.s1['inching']['retreat']])[self.s1['inching']['retreat']]
        self.tof_retreat = load_pulses(self.f1, [self.cfg['tof']['retreat']])[self.cfg['tof']['retreat']]
        self.plane = {k: float(v) for k, v in self.cfg['plane_z_m'].items() if k != 'note'}
        self.log_file = Path(log_path).open('a') if log_path else None
        self.lock = threading.Lock()
        self.on_state, self.echo = on_state, echo
        self.state = None
        self.candidate = None
        self.toy_at = None
        self.side = None
        self.tray = None
        self.closes = 0
        self.spins = 0
        self.spin_left = float(self.cfg['search']['spin_deg'])
        self.approach_tries = 0
        self.tray_turns = 0
        self.tray_moves = 0
        self.tray_scans = 0
        self.pose_doubt = False                  # set by a fall, cleared by a YOLO tray look
        self.lost = False                        # the toy came out of the beak and is not found yet
        self.recover_looks = 0                   # RECOVER looks since it came out
        self.falls_handled = len(robot.falls)
        self.history: list = []
        self.tof_stops: list = []
        self.picks: list = []
        self.t0 = None
        self.frames_dir = Path(frames_dir) if frames_dir else None
        self.shots = 0
        if self.frames_dir is not None and eyes is not None:
            eyes.shoot = self.shot

    # ── bookkeeping ─────────────────────────────────────────────────────

    def shot(self, sight, tag: str) -> None:
        """Save the picture a decision rested on, with its boxes (stage3/shots.py); never fatal."""
        if self.frames_dir is None or self.shots >= shots_lib.MAX_PER_EPISODE:
            return
        try:
            got = self.eyes.picture(sight)
            if got is None:
                return
            rgb, y0 = got
            name = f'{self.shots:02d}_{tag}_s{sight.seq}.jpg'
            title = f'{self.state or "-"} {tag}  target {self.target}  seq {sight.seq}'
            shots_lib.save(self.frames_dir / name, rgb, sight.boxes, title, self.target, y0)
            self.shots += 1
        except Exception as error:          # a picture is evidence, not control: carry on
            self.log('shot_error', tag=tag, error=f'{type(error).__name__}: {error}')

    def log(self, event: str, **fields) -> None:
        now = time.monotonic()
        row = {'mono': now, 'el': None if self.t0 is None else round(now - self.t0, 3),
               'event': event, **fields}
        with self.lock:
            if self.log_file is not None:
                self.log_file.write(json.dumps(row, default=_jsonable) + '\n')
                self.log_file.flush()
        if self.echo is not None and event in ('state', 'end', 'tof_stop', 'fallen', 'confirmed'):
            self.echo(row)

    def elapsed(self) -> float:
        return time.monotonic() - self.t0

    def here(self) -> tuple[float, float, float]:
        return self.pose.of(self.robot.snapshot())

    def check(self) -> None:
        """Between actions: a new fall, or the episode clock."""
        state = self.robot.snapshot()
        if len(self.robot.falls) > self.falls_handled or state['safety'].get('fallen'):
            raise Fallen()
        if self.elapsed() > float(self.cfg['episode']['max_s']):
            raise Timeout()

    def walk_stop(self, state: dict):
        """approach.walk's stop hook: fall, clock, ToF (in that order)."""
        if state['safety'].get('fallen') or len(self.robot.falls) > self.falls_handled:
            return 'fallen'
        if self.elapsed() > float(self.cfg['episode']['max_s']):
            return 'timeout'
        if self.tof is not None:
            hit = self.tof.stop_reason()
            if hit is not None:
                return {'tof': hit, 'pose': self.pose.of(state)}
        return None

    def holding_now(self) -> bool:
        return math.degrees(mouth_angle(self.robot.snapshot())) > float(self.s1['grasp']['held_threshold_deg'])

    def head(self, pose) -> None:
        self.robot.head(tuple(pose) if isinstance(pose, list) else pose)

    def obstacles(self, skip=None) -> list:
        return self.seen.obstacles(self.target, skip=skip) + self.extra

    # ── motion with the global checks ───────────────────────────────────

    def turn(self, angle: float) -> dict:
        out = approach.turn(self.robot, angle, self.s1['approach'])
        self.check()
        return out

    def walk(self, distance: float, bearing: float, vx: float) -> dict:
        if self.tof is not None:
            self.tof.arm()
        out = approach.walk(self.robot, distance, bearing, {**self.s1['approach'], 'walk_vx_mps': vx},
                            stop=self.walk_stop)
        stopped = out.get('stopped')
        if stopped == 'fallen':
            raise Fallen()
        if stopped == 'timeout':
            raise Timeout()
        if isinstance(stopped, dict):
            self.on_tof(stopped)
        self.check()
        return out

    def on_tof(self, stopped: dict) -> None:
        hit, pose = stopped['tof'], stopped['pose']
        xy = to_field(pose, hit['point'])
        self.tof_stops.append({'el': round(self.elapsed(), 3), 'pose': list(pose), 'hit': hit,
                               'xy': list(xy), 'state': self.state})
        self.log('tof_stop', hit=hit, xy=xy, pose=pose, state=self.state)
        time.sleep(0.3)
        approach.send_pulse(self.robot, self.tof_retreat)
        self.extra.append((xy, float(self.cfg['route']['tof_obstacle_m'])))

    def step_towards(self, goal, vx: float, leg_max: float, obstacles) -> dict:
        """One leg of the route to `goal`: turn to its first waypoint, walk <= leg_max."""
        x, y, yaw = self.here()
        route = plan_route((x, y), goal, obstacles, self.fld) or [tuple(goal)]
        # A walk shorter than walk_slide_m is all coast and never starts (cloud 3D: a post-fall pose
        # 4 cm outside the keep-in square looped on its 4 cm entry step), so such a waypoint is passed.
        near = float(self.s1['approach']['walk_slide_m']) + 0.02
        while len(route) > 1 and math.dist((x, y), route[0]) < near:
            route.pop(0)
        w = route[0]
        b = bearing_to((x, y, yaw), w)
        out = {'route': [list(p) for p in route], 'from': [x, y, yaw]}
        if abs(b) > math.radians(self.s1['approach']['turn_tol_deg']):
            out['turn'] = self.turn(b)
            x, y, yaw = self.here()
            b = bearing_to((x, y, yaw), w)
        length = min(math.dist((x, y), w), leg_max)
        n0 = len(self.tof_stops)
        out['walk'] = self.walk(length, b, vx)
        out['tof_stop'] = len(self.tof_stops) > n0
        return out

    # ── the loop ────────────────────────────────────────────────────────

    def run(self) -> dict:
        self.t0 = time.monotonic()
        self.log('start', target=self.target, start=self.pose.start, odom0=self.pose.odom0,
                 until=self.until, start_state=self.start_state, holding=self.holding,
                 drop_d=self.drop_d, known=[s.as_dict() for s in self.seen.spots])
        state, end = self.start_state, None
        while end is None:
            self.enter(state)
            try:
                state = getattr(self, 's_' + state.lower())()
            except Fallen:
                state = 'FALLEN'
            except Timeout:
                end = Ended('timeout')
            except Ended as done:
                end = done
            except Exception as error:      # anything unexpected ends the episode, logged
                end = Ended('error', error=f'{type(error).__name__}: {error}')
        self.leave()
        try:
            self.robot.move(0.0, 0.0, 0.0)
        except Exception:
            pass
        result = {'end': end.reason, **end.detail, 'target': self.target,
                  'elapsed_s': round(self.elapsed(), 3), 'closes': self.closes,
                  'holding': self.holding, 'spins': self.spins, 'history': self.history,
                  'tof_stops': self.tof_stops, 'picks': self.picks,
                  'falls': [f for f in self.robot.falls if f['mono'] >= self.t0],
                  'candidate': self.candidate.as_dict() if self.candidate else None,
                  'side': {k: v for k, v in (self.side or {}).items() if k != 'route'},
                  'tray': self.tray, 'drop_d': self.drop_d,
                  'sightings': [s.as_dict() for s in self.seen.spots]}
        try:                                # where the brain thinks it ended (Stage 5: the next start)
            result['pose'] = [round(float(v), 5) for v in self.here()]
        except Exception:
            result['pose'] = None
        self.log('end', **{k: result[k] for k in ('end', 'elapsed_s', 'closes', 'holding')},
                 detail=end.detail)
        return result

    def enter(self, state: str) -> None:
        self.leave()
        self.state = state
        self.history.append([state, round(self.elapsed(), 3), None])
        self.log('state', state=state, pose=self.here())
        if self.on_state is not None:
            self.on_state(state)

    def leave(self) -> None:
        if self.history and self.history[-1][2] is None:
            self.history[-1][2] = round(self.elapsed(), 3)

    def end(self, reason: str, **detail):
        raise Ended(reason, **detail)

    # ── SPIN / LOOK / RELOCATE ──────────────────────────────────────────

    def note(self, sight, weight: float, cluster_per_m: float = 0.0) -> list[dict]:
        """Every box of a frame into the sighting table (field frame); returns what was seen."""
        cfg = self.cfg['search']
        pose = self.pose.of(sight.state)
        out = []
        for box in sight.boxes:
            cls = box['cls']
            if cls not in self.plane:
                continue
            body = self.eyes.ground(sight, box, self.plane[cls])
            bearing = self.eyes.bearing(sight, box)
            rec = {'cls': cls, 'conf': box['conf'], 'bearing': bearing}
            if body is not None:
                dist = math.hypot(*body)
                xy = to_field(pose, body)
                rec.update(body=body, xy=xy, dist=dist)
                if cls != 'tray' and dist <= float(cfg['max_range_m']):
                    radius = float(cfg['cluster_m']) + cluster_per_m * dist
                    spot = self.add_spot(cls, xy, weight, dist, radius)
                    rec['spot'] = spot.id
            out.append(rec)
        return out

    def add_spot(self, cls, xy, weight, dist, radius):
        old = self.seen.cluster_m
        self.seen.cluster_m = radius
        try:
            return self.seen.add(cls, xy, weight, dist)
        finally:
            self.seen.cluster_m = old

    def next_candidate(self):
        here = self.here()
        spots = [s for s in self.seen.of_class(self.target, ('seen', 'target'))
                 if s.frames >= int(self.cfg['search']['spin_hits'])]
        if not spots:
            return None
        return max(spots, key=lambda s: (s.status == 'target', s.weight,
                                         -math.dist(s.xy, here[:2])))

    def s_spin(self) -> str:
        cfg = self.cfg['search']
        self.head(cfg['spin_head'])
        time.sleep(0.5)
        self.check()
        wz = float(cfg['spin_wz_radps'])
        _, _, yaw_prev = approach.odom_of(self.robot.snapshot())
        turned = 0.0
        found = None
        self.eyes.frame('search', fresh=True)
        self.robot.move(0.0, 0.0, wz)
        try:
            while turned < math.radians(self.spin_left):
                self.check()
                sight = self.eyes.frame('search')
                _, _, yaw = approach.odom_of(self.robot.snapshot())
                turned += abs(wrap(yaw - yaw_prev))
                yaw_prev = yaw
                seen = self.note(sight, float(cfg['spin_weight']), float(cfg['spin_cluster_per_m']))
                self.log('spin_frame', seq=sight.seq, mono_frame=sight.mono, pose=self.pose.of(sight.state),
                         turned_deg=round(math.degrees(turned), 1), seen=seen, view=sight.view, ms=sight.ms)
                found = self.next_candidate()
                if found is not None:
                    self.shot(sight, 'spin_found')
                    break
        finally:
            self.robot.move(0.0, 0.0, 0.0)
        self.spin_left = max(0.0, self.spin_left - math.degrees(turned))
        time.sleep(0.3)
        if found is not None:
            self.candidate = found
            return 'LOOK'
        self.spins += 1
        self.spin_left = float(cfg['spin_deg'])
        if self.spins >= int(cfg['max_spins']):
            self.end('search_fail', spins=self.spins)
        return 'RELOCATE'

    def s_relocate(self) -> str:
        """Walk to the nearest free relocate spot (plan: a free spot near the middle), then spin."""
        cfg = self.cfg['search']
        x, y, _ = self.here()
        spots = [tuple(p) for p in cfg['relocate_spots']
                 if math.dist(p, (x, y)) >= float(cfg['relocate_min_m'])
                 and all(math.dist(p, o if not isinstance(o[0], (tuple, list)) else o[0]) > 0.25
                         for o in self.obstacles())]
        if not spots:
            return 'SPIN'
        goal = min(spots, key=lambda p: math.dist(p, (x, y)))
        self.head(self.cfg['goto']['head'])
        for _ in range(6):
            x, y, _ = self.here()
            if math.dist(goal, (x, y)) <= 0.1:
                break
            leg = self.step_towards(goal, float(self.cfg['goto']['vx_mps']),
                                    float(self.cfg['goto']['leg_max_m']), self.obstacles())
            self.log('relocate_leg', goal=goal, **leg)
        return 'SPIN'

    def s_look(self) -> str:
        cfg = self.cfg['search']
        spot = self.candidate
        if spot is None:
            return 'SPIN'
        pose = self.here()
        # The search window starts about 0.33 m from the trunk at H0 (its bottom edge): a nearer spot
        # (a GOTO leg that went long) is not in the picture, so the B segment's H1 looks take it.
        near = math.dist(spot.xy, pose[:2])
        if near < float(cfg['look_near_m']):
            self.log('look', spot=spot.id, result='near', dist=near)
            return 'APPROACH'
        b = bearing_to(pose, spot.xy)
        if abs(b) > math.radians(cfg['look_turn_tol_deg']):
            self.turn(b)
        self.head(cfg['look_head'])
        time.sleep(float(cfg['look_settle_s']))
        self.check()
        self.eyes.frame('search', fresh=True)
        hits, others, frames = [], {}, []
        view = None
        for _ in range(int(cfg['look_frames'])):
            sight = self.eyes.frame('search')
            view = sight.view
            seen = self.note(sight, 1.0)
            frames.append({'seq': sight.seq, 'mono': sight.mono, 'pose': self.pose.of(sight.state), 'seen': seen})
            near_t = [r for r in seen if r['cls'] == self.target and 'xy' in r
                      and math.dist(r['xy'], spot.xy) <= float(cfg['look_match_m'])]
            if near_t:
                hits.append(min(near_t, key=lambda r: math.dist(r['xy'], spot.xy))['xy'])
            for cls in {r['cls'] for r in seen if r['cls'] not in (self.target, 'tray') and 'xy' in r
                        and math.dist(r['xy'], spot.xy) <= float(cfg['look_match_m'])}:
                others[cls] = others.get(cls, 0) + 1
        spot.looks += 1
        pose = self.here()
        need = int(cfg['look_min_seen'])
        verdict = {'spot': spot.id, 'hits': len(hits), 'others': others, 'frames': len(frames), 'view': view}
        if len(hits) >= need:
            spot.x = sum(p[0] for p in hits) / len(hits)
            spot.y = sum(p[1] for p in hits) / len(hits)
            dist = math.dist(spot.xy, pose[:2])
            verdict['dist'] = dist
            if dist <= float(cfg['confirm_m']):
                spot.status = 'target'
                verdict['result'] = 'confirmed'
                self.shot(sight, 'look_confirmed')
                self.log('look', **verdict, frames_seen=frames)
                self.log('confirmed', spot=spot.as_dict(), dist=dist)
                near = dist <= float(self.cfg['goto']['b_start_m']) + float(self.cfg['goto']['b_start_tol_m'])
                return 'APPROACH' if near else 'GOTO'
            verdict['result'] = 'far'
            self.shot(sight, 'look_far')
            self.log('look', **verdict, frames_seen=frames)
            return 'GOTO'
        dist = math.dist(spot.xy, pose[:2])
        # A confirmed target spot never turns into a distractor (no confusion within confirm_m, Stage 2):
        # another toy there means the odometry (a fall) or the toy moved, so it only counts as a miss.
        if (others and max(others.values()) >= need and dist <= float(cfg['confirm_m'])
                and spot.status != 'target'):
            spot.status = 'distractor'
            verdict['result'] = 'distractor'
        else:
            spot.misses += 1
            verdict['result'] = 'miss'
            if spot.misses >= int(cfg['max_misses']):
                spot.status = 'dropped'
                verdict['result'] = 'dropped'
        self.shot(sight, 'look_' + verdict['result'])
        self.log('look', **verdict, dist=dist, frames_seen=frames)
        self.candidate = self.next_candidate()
        return 'LOOK' if self.candidate is not None else 'SPIN'

    # ── GOTO / APPROACH / LAST / PICK ───────────────────────────────────

    def s_goto(self) -> str:
        cfg = self.cfg['goto']
        spot = self.candidate
        x, y, yaw = self.here()
        dist = math.dist(spot.xy, (x, y))
        if spot.status == 'target' and dist <= float(cfg['b_start_m']) + float(cfg['b_start_tol_m']):
            return 'APPROACH'
        obstacles = self.obstacles(skip=spot)
        plain = [o for o in obstacles if not isinstance(o[0], (tuple, list))]
        goal = approach_point((x, y), spot.xy, float(cfg['b_start_m']), plain, self.fld,
                              float(cfg['corridor_m']))
        if goal is None:
            a = math.atan2(y - spot.y, x - spot.x)
            goal = (spot.x + float(cfg['b_start_m']) * math.cos(a), spot.y + float(cfg['b_start_m']) * math.sin(a))
        self.head(cfg['head'])
        leg = self.step_towards(goal, float(cfg['vx_mps']), float(cfg['leg_max_m']), obstacles)
        self.log('goto_leg', goal=goal, spot=spot.as_dict(), **leg)
        return 'LOOK'

    def s_approach(self) -> str:
        spot = self.candidate
        self.eyes.prefer = lambda st: self.pose.body(st, spot.xy)
        b = approach.b_segment(self.robot, self.eyes, self.target, self.plane[self.target], self.s1,
                               stop=self.walk_stop)
        self.log('b_segment', end=b['end'], steps=b['steps'], spot=spot.as_dict(), pose=self.here())
        stopped = b.get('stopped')
        if stopped == 'fallen':
            raise Fallen()
        if stopped == 'timeout':
            raise Timeout()
        if isinstance(stopped, dict):
            self.on_tof(stopped)
            return 'LOOK'
        self.check()
        if b['end'] == 'ready':
            self.toy_at = b['steps'][-1]['toy']
            spot.x, spot.y = self.pose.point(self.robot.snapshot(), self.toy_at)
            self.approach_tries = 0
            return 'LAST'
        self.approach_tries += 1
        approach.send_pulse(self.robot, self.retreat)
        if self.approach_tries >= int(self.cfg['goto']['max_approach_tries']):
            spot.status = 'dropped'
            self.approach_tries = 0
            self.candidate = self.next_candidate()
            return 'LOOK' if self.candidate is not None else 'SPIN'
        return 'LOOK'

    def s_last(self) -> str:
        out = approach.blind_move(self.robot, self.toy_at, self.s1, stop=self.walk_stop)
        self.log('blind_move', **out)
        stopped = out.get('stopped')
        if stopped == 'fallen':
            raise Fallen()
        if stopped == 'timeout':
            raise Timeout()
        if isinstance(stopped, dict):
            self.on_tof(stopped)
            return 'LOOK'
        self.check()
        return 'PICK'

    def s_pick(self) -> str:
        pick = self.s1['pick']
        res = ground_pick(self.robot, float(self.f1['grasp']['phi_close']), float(pick['period_s']),
                          float(self.s1['grasp']['held_threshold_deg']),
                          confirm_s=float(pick['confirm_after_s']),
                          mark=lambda tag: (self.log(tag), time.monotonic())[1])
        self.closes += 1 if res.get('ran') else 0
        self.picks.append({**res, 'n': self.closes, 'el': round(self.elapsed(), 3)})
        self.log('pick', **res, closes=self.closes)
        if not res.get('ran'):
            self.end('error', error=f"ground_pick did not run: {res.get('error')}")
        self.check()
        if res.get('held'):
            self.holding = True
            if self.until == 'held':
                self.end('held')
            return 'CARRY'
        if self.closes >= int(self.cfg['episode']['max_closes']):
            self.end('grasp_fail')
        time.sleep(0.5)
        approach.send_pulse(self.robot, self.retreat)
        self.check()
        return 'APPROACH'

    # ── CARRY / TRAY_LOOK / DROP ────────────────────────────────────────

    def s_carry(self) -> str:
        cfg = self.cfg['carry']
        self.head(cfg['head'])
        x, y, yaw = self.here()
        skip = self.candidate
        obstacles = self.obstacles(skip=skip)
        plain = points_of(obstacles)
        if self.side_forced:
            point = tray_lib.staging_point(self.side_forced, self.fld, float(cfg['staging_m']))
            side = {'side': self.side_forced, 'staging': point, 'face_yaw': tray_lib.facing(self.side_forced),
                    'route': plan_route((x, y), point, obstacles, self.fld) or [point]}
        else:
            side = tray_lib.choose_side((x, y), plain, self.fld, cfg)
            if side['side'] is None and self.extra:
                # Every ToF stop is a wall (hits under 5 cm do not stop) and walls are in the keep-in
                # square already; on post-fall odometry their field points are wrong and can wall in
                # every side (Mac 3D: 15 stops in 80 s, then no route), so they are dropped.
                self.log('no_route', dropped_tof_obstacles=len(self.extra))
                self.extra = []
                obstacles = self.obstacles(skip=skip)
                plain = points_of(obstacles)
                side = tray_lib.choose_side((x, y), plain, self.fld, cfg)
            if side['side'] is None:
                # Still none: the nearest side straight, the ToF guarding the walk.
                name = min(cfg['sides'], key=lambda k: math.dist(
                    (x, y), tray_lib.staging_point(k, self.fld, float(cfg['staging_m']))))
                point = tray_lib.staging_point(name, self.fld, float(cfg['staging_m']))
                self.log('no_route', straight_to=name)
                side = {'side': name, 'staging': point, 'face_yaw': tray_lib.facing(name),
                        'route': [point]}
        if self.side is None or self.side['side'] != side['side']:
            self.log('side', **{k: v for k, v in side.items() if k != 'route'}, route=side['route'])
        self.side = side
        if math.dist((x, y), side['staging']) <= float(cfg['arrive_m']):
            turn = wrap(side['face_yaw'] - yaw)
            if abs(turn) > math.radians(self.s1['approach']['turn_tol_deg']) * 0.5:
                self.log('square_up', **self.turn(turn))
            return 'TRAY_LOOK'
        leg = self.step_towards(side['staging'], float(cfg['vx_mps']), float(cfg['leg_max_m']),
                                obstacles)
        holding = self.holding_now()
        self.log('carry_leg', side=side['side'], holding=holding, **leg)
        if not holding:
            self.holding, self.lost, self.recover_looks = False, True, 0
            return 'RECOVER'
        return 'CARRY'

    def s_tray_look(self) -> str:
        cfg = self.cfg['tray_look']
        self.head(cfg['head'])
        time.sleep(float(cfg['settle_s']))
        self.check()
        self.eyes.frame('approach', fresh=True)
        looks = []
        for _ in range(int(cfg['frames'])):
            sight = self.eyes.frame('approach')
            looks.append((sight.state, sight.boxes))
        state = self.robot.snapshot()
        expected = self.pose.body(state, self.fld.tray_centre)
        doubt = self.pose_doubt
        accept = float(cfg['accept_after_fall_m'] if doubt else cfg['accept_m'])
        # Stage 5 room demo only: the round tray is ranged by its own rule (tray.measure_tray).
        round_tray = {'shape': 'round'} if self.s1['arena']['tray'].get('shape') == 'round' else {}
        m = tray_lib.measure_tray(looks, self.cam, {**cfg, 'accept_m': accept}, self.fld.tray_half_m,
                                  float(self.s1['arena']['tray']['rim_height_m']), expected,
                                  near_z_m=self.plane['tray'], **round_tray)
        if doubt and m['source'] == 'yolo' and m['used'] < int(cfg['min_frames_after_fall']):
            m.update(centre=list(expected), source='odometry', too_few=True)
        m['boxes'] = [[b for b in boxes if b['cls'] == 'tray'] for _, boxes in looks]
        self.shot(sight, 'tray_' + m['source'])
        m['pose_doubt'] = doubt
        # After a fall the tray may be well off to the side (cloud 3D: 88 deg right after two falls,
        # out of the +-22.5 deg picture, and the toy went down on odometry 0.4 m away): look around.
        scans = cfg['scan_after_fall_deg']
        if doubt and m['source'] != 'yolo' and self.tray_scans < len(scans):
            turn = float(scans[self.tray_scans])
            self.tray_scans += 1
            self.tray = m
            self.log('tray_look', **m, recarry=False, scan_deg=turn)
            self.turn(math.radians(turn))
            return 'TRAY_LOOK'
        recarry = False
        if m['source'] == 'yolo':
            shift = self.pose.reanchor(state, m['centre'], self.fld.tray_centre)
            m['reanchor_m'] = [round(v, 4) for v in shift]
            self.pose_doubt = False
            # After a fall the "staging point" reached on bad odometry is not the real one, and looking
            # at the tray from off its normal ranges it short (cloud 3D: 4-6 cm): go there for real.
            recarry = (doubt and math.hypot(*shift) > float(self.cfg['carry']['arrive_m'])
                       and self.tray_moves < int(cfg['max_moves']))
        self.tray = m
        self.log('tray_look', **m, recarry=recarry)
        if recarry:
            self.tray_moves += 1
            return 'CARRY'
        bearing = math.atan2(m['centre'][1], m['centre'][0])
        if abs(bearing) > math.radians(cfg['turn_tol_deg']) and self.tray_turns < int(cfg['max_turns']):
            self.tray_turns += 1
            self.turn(bearing)
            return 'TRAY_LOOK'
        # Too far for the short drop walk (a wrong staging point after a fall): walk up, look again.
        far = math.hypot(*m['centre']) - float(self.cfg['carry']['staging_m'])
        if far > float(cfg['range_tol_m']) and self.tray_moves < int(cfg['max_moves']):
            self.tray_moves += 1
            self.log('tray_closer', **self.walk(far, bearing, float(self.cfg['carry']['vx_mps'])))
            if not self.holding_now():
                self.holding = False
                return 'RECOVER'
            return 'TRAY_LOOK'
        return 'DROP'

    def s_drop(self) -> str:
        cfg = self.cfg['drop']
        # The tray look was taken standing here; odometry since then is a turn at most.
        centre = self.tray['centre']
        self.head(cfg['head'])
        time.sleep(0.5)
        w = tray_lib.drop_walk(centre, self.drop_d)
        out = {'plan': w}
        b = w['bearing']
        if abs(b) > math.radians(cfg['turn_tol_deg']):
            out['turn'] = self.turn(b)
            b = 0.0
        out['walk'] = self.walk(w['distance'], b, float(cfg['vx_mps']))
        time.sleep(float(cfg['stand_s']))
        self.check()
        out['open'] = time.monotonic()
        self.log('drop_open', **out)
        self.robot.mouth(1.0)
        time.sleep(float(cfg['open_s']))
        self.robot.mouth(0.0)
        time.sleep(0.3)
        still = self.holding_now()
        self.holding = still
        self.end('dropped', still_holding=still, drop=out)

    # ── FALLEN / RECOVER ────────────────────────────────────────────────

    def s_fallen(self) -> str:
        cfg = self.cfg['fall']
        self.robot.move(0.0, 0.0, 0.0)
        self.head('H0')
        t_fall = time.monotonic()
        try:
            self.robot.wait(lambda s: s['policy'] in UPRIGHT and not s['safety'].get('fallen'),
                            float(cfg['getup_wait_s']), 'velstand up')
            up = True
        except TimeoutError:
            up = False
        time.sleep(float(cfg['settle_s']))
        self.falls_handled = len(self.robot.falls)
        self.pose_doubt = True                   # contact odometry while tumbling is not to be trusted
        held = self.holding_now()
        self.log('fallen', up=up, waited_s=round(time.monotonic() - t_fall, 3), was_holding=self.holding,
                 holding=held, pose=self.here())
        if self.elapsed() > float(self.cfg['episode']['max_s']):
            raise Timeout()
        if not up:
            return 'FALLEN'
        if self.holding and held:
            return 'CARRY'
        if self.holding or self.lost:            # it came out in this fall or an earlier one: at our feet
            if self.holding:
                self.recover_looks = 0
            self.holding, self.lost = False, True
            return 'RECOVER'
        if self.candidate is not None and self.candidate.status in ('seen', 'target'):
            return 'LOOK'
        return 'SPIN'

    def s_recover(self) -> str:
        """The toy came out of the beak: look at the feet, H2 first (it sees the ground from about 8 cm
        ahead of the trunk, H1 only from about 12 cm), then H1. Seen -> approach it, backing off a
        pulse first if it is too close to aim at; not seen -> forget the pick spot, search afresh."""
        cfg = self.cfg['recover']
        self.recover_looks += 1
        hits = []
        for head in cfg['heads']:
            self.head(head)
            time.sleep(float(cfg['settle_s']))
            self.check()
            self.eyes.frame('approach', fresh=True)
            hits = []
            for _ in range(int(cfg['frames'])):
                sight = self.eyes.frame('approach')
                for box in sight.of(self.target)[:1]:
                    body = self.eyes.ground(sight, box, self.plane[self.target])
                    if body is not None:
                        body = self.eyes.corrections[head].apply(body)
                        hits.append((body, self.pose.point(sight.state, body)))
            self.shot(sight, f'recover_{head}')
            self.log('recover_look', head=head, hits=[xy for _, xy in hits])
            if len(hits) >= int(self.cfg['search']['look_min_seen']):
                break
        if len(hits) >= int(self.cfg['search']['look_min_seen']):
            self.lost = False
            xy = (sum(p[0] for _, p in hits) / len(hits), sum(p[1] for _, p in hits) / len(hits))
            forward = sum(b[0] for b, _ in hits) / len(hits)
            if forward < float(cfg['backoff_below_m']):
                self.log('recover_backoff', forward_m=forward)
                approach.send_pulse(self.robot, self.retreat)
            if self.candidate is None:
                self.candidate = self.seen.add(self.target, xy, 1.0, 0.0)
            self.candidate.x, self.candidate.y = xy
            self.candidate.status = 'target'
            return 'APPROACH'
        approach.send_pulse(self.robot, self.retreat)
        if self.recover_looks < int(cfg['looks']):
            return 'RECOVER'                     # it may be under the beak (cloud 3D: 5 cm ahead): back off, look again
        self.lost = False
        # The candidate is where the toy was picked, not where it fell (and after a fall the odometry is
        # off too): forget it and search afresh with a full spin budget.
        if self.candidate is not None:
            self.candidate.status = 'dropped'
            self.candidate = None
        self.spins, self.spin_left = 0, float(self.cfg['search']['spin_deg'])
        return 'SPIN'


def _jsonable(value):
    if hasattr(value, 'as_dict'):
        return value.as_dict()
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    return str(value)


# ── the board process (plan 3F) ─────────────────────────────────────────────

def make_eyes(args, configs: dict, history, probe=None, log=None):
    """The camera source and the detector named on the command line."""
    from microduck_toy.stage2 import deploy
    from microduck_toy.stage3 import eyes as eyes_lib
    s2, s3 = configs['stage2'], configs['stage3']
    if args.source == 'probe':
        source = eyes_lib.ProbeSource(probe, rgb=args.eyes != 'oracle')
    else:
        host, port = args.camera.rsplit(':', 1)
        source = eyes_lib.StreamSource((host, int(port)), float(s3['camera']['stream_lag_s']))
    if args.eyes == 'oracle':
        if args.source != 'probe':
            raise SystemExit('--eyes oracle needs --source probe (the arena body test port)')
        detector = eyes_lib.Oracle()
    else:
        model = Path(args.model or (repo_root() / s3['detector']['models'][args.backend]))
        live = deploy.LiveDetector(args.backend, model, s2['classes'], float(s2['detect']['conf']),
                                   float(s2['detect']['iou']), threads=int(s3['detector']['cpu_threads']))
        live.warm()
        detector = eyes_lib.Yolo(live, int(s2['input']['window_margin_rows']))
        if args.eyes == 'colour':           # Stage 4E: the toys by colour threshold, the tray by YOLO
            detector = eyes_lib.Colour(detector, s2['baselines']['colour'], float(s2['detect']['conf']))
    cam = ranging.CameraModel.from_config(configs['stage1']['camera'])
    return eyes_lib.Eyes(source, detector, history, cam, configs['facts1']['ranging'], log=log)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--target', required=True, choices=TARGETS)
    ap.add_argument('--start', required=True, help='known start pose X,Y,YAW (field frame, rad)')
    ap.add_argument('--socket', required=True, help='robotd control socket')
    ap.add_argument('--camera', default='127.0.0.1:7901')
    ap.add_argument('--source', choices=('stream', 'probe'), default='stream')
    ap.add_argument('--probe', default='127.0.0.1:7811', help='with --source probe')
    ap.add_argument('--eyes', choices=('yolo', 'oracle', 'colour'), default='yolo')
    ap.add_argument('--backend', choices=('cpu', 'fp16', 'int8'), default='int8')
    ap.add_argument('--model', default=None)
    ap.add_argument('--tof', default='127.0.0.1:7801', help="sim body HOST:PORT for ToF, or 'none'")
    ap.add_argument('--until', choices=('held', 'delivered'), default='delivered')
    ap.add_argument('--drop-d', type=float, default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--frames', action='store_true',
                    help="save the robot's-eye pictures to OUT/frames (needs Pillow)")
    ap.add_argument('--config-override', default=None,
                    help='Stage 5 room demo: configs/demo/room.json (its stage1/stage3 sections '
                         'deep-merged over the frozen configs)')
    ap.add_argument('--side', default=None, choices=('x-', 'x+', 'y+', 'y-'),
                    help='Stage 5 room demo: carry to this side of the tray (default: the shortest '
                         'free side, stage3 carry.sides)')
    return ap


def main(argv=None) -> int:
    from microduck_toy.stage3.eyes import StateHistory
    from microduck_toy.stage3.tof import TofReader
    args = parser().parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    configs = load_configs(override=args.config_override)
    start = tuple(float(v) for v in args.start.split(','))
    events = Events(out / 'events.jsonl')
    history = StateHistory()
    robot = Robot(args.socket, out, events, on_state=history)
    probe = None
    if args.source == 'probe':
        from microduck_toy.stage0.link import Probe
        host, port = args.probe.rsplit(':', 1)
        probe = Probe(host, int(port))
    eyes = tof = None
    status = 1

    def echo(row):
        print(json.dumps({'sync': row['mono'], 'event': row['event'], 'state': row.get('state'),
                          'end': row.get('end')}), flush=True)

    try:
        robot.wait(lambda s: s['policy'] in UPRIGHT, 10.0, 'upright policy')
        eyes = make_eyes(args, configs, history, probe)
        if args.tof != 'none':
            host, port = args.tof.rsplit(':', 1)
            tof = TofReader((host, int(port)), robot, configs['stage3']['tof'])
            tof.start()
        echo({'mono': time.monotonic(), 'event': 'ready'})
        brain = Brain(robot, eyes, tof, configs, args.target, start, log_path=out / 'brain.jsonl',
                      until=args.until, drop_d=args.drop_d, side=args.side, echo=echo,
                      frames_dir=out / 'frames' if args.frames else None)
        result = brain.run()
        result['latency'] = eyes.latency()
        (out / 'result.json').write_text(json.dumps(result, indent=1, default=_jsonable))
        status = 0 if result['end'] != 'error' else 2
    except KeyboardInterrupt:
        status = 130
    finally:
        for thing in (tof, eyes):
            if thing is not None:
                thing.close()
        try:
            robot.mouth(0.0)
            robot.head('H0')
        finally:
            robot.close()
            events.close()
            if probe is not None:
                probe.sock.close()      # not probe.close(): that would end the runner's truth.csv
    return status


if __name__ == '__main__':
    sys.exit(main())
