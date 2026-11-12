"""The Stage 3 brain's state machine without a simulator: a fake robot that teleports, fake eyes
with scripted boxes, and the measured actions (turn, walk, ground_pick, pulses) stubbed out."""
from __future__ import annotations

import copy
import math
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest
from unittest import mock

from microduck_toy.stage3 import brain as brain_lib
from microduck_toy.stage3 import tray as tray_lib
from microduck_toy.stage3.brain import Brain, Fallen
from microduck_toy.stage3.eyes import Sight

CONFIGS = brain_lib.load_configs()
S3 = CONFIGS['stage3']


class FakeRobot:
    """robot.state with odometry the test moves by hand."""

    def __init__(self, x=0.0, y=0.0, yaw=0.0):
        self.x, self.y, self.yaw = x, y, yaw
        self.falls: list = []
        self.fallen = False
        self.policy = 'stand'          # robotd slot names (control/robot.py UPRIGHT)
        self.calls: list = []

    def snapshot(self):
        return {'odom': {'position': [self.x, self.y, 0.12], 'yaw': self.yaw},
                'safety': {'fallen': self.fallen, 'gravity': [0.0, 0.0, -1.0]},
                'policy': self.policy, 'frames': {}, 'imu': {'quat': [1.0, 0.0, 0.0, 0.0]}}

    def head(self, pose):
        self.calls.append(('head', pose))

    def move(self, vx, vy, wz):
        self.calls.append(('move', vx, vy, wz))

    def mouth(self, v):
        self.calls.append(('mouth', v))

    def wait(self, cond, timeout, what):
        state = self.snapshot()
        if cond(state):
            return state
        raise TimeoutError(what)

    # the stubs for approach.turn / approach.walk move the robot here
    def turn_by(self, angle):
        self.yaw = math.atan2(math.sin(self.yaw + angle), math.cos(self.yaw + angle))

    def walk_by(self, distance, bearing):
        heading = self.yaw + bearing
        self.x += distance * math.cos(heading)
        self.y += distance * math.sin(heading)


class FakeEyes:
    """Each frame: the boxes script(robot) returns, with box['body'] = (forward, left) on the floor."""

    def __init__(self, robot, script, before=None):
        self.robot, self.script, self.before = robot, script, before
        self.seq = 0
        self.prefer = None
        self.corrections = {'H1': mock.Mock(apply=lambda p: p), 'H2': mock.Mock(apply=lambda p: p)}

    def frame(self, view, fresh=False, timeout=5.0):
        if self.before is not None:
            self.before(self.robot)
        self.seq += 1
        boxes = self.script(self.robot, self.seq)
        state = copy.deepcopy(self.robot.snapshot())
        return Sight(self.seq, state, boxes, view, {}, 0.0, 0.0)

    def ground(self, sight, box, plane_z, point='centre'):
        return box.get('body')

    def bearing(self, sight, box):
        b = box.get('body')
        return None if b is None else math.atan2(b[1], b[0])


def body_of(robot, xy):
    c, s = math.cos(robot.yaw), math.sin(robot.yaw)
    dx, dy = xy[0] - robot.x, xy[1] - robot.y
    return (c * dx + s * dy, -s * dx + c * dy)


def toy_at(cls, xy):
    """A script that shows the toy at field xy in every frame, wherever the robot is."""
    return lambda robot, seq: [{'cls': cls, 'conf': 0.9, 'xyxy': [0, 0, 1, 1], 'body': body_of(robot, xy)}]


def make_brain(robot, script=lambda r, s: [], before=None, **kw):
    kw.setdefault('start', (robot.x, robot.y, robot.yaw))
    eyes = FakeEyes(robot, script, before)
    b = Brain(robot, eyes, None, CONFIGS, kw.pop('target', 'bus'), **kw)
    b.t0 = 0.0
    return b


class StubbedMotion(unittest.TestCase):
    """time.sleep is skipped; turn/walk teleport the fake robot; the clock never runs out."""

    def setUp(self):
        patches = [mock.patch('microduck_toy.stage3.brain.time.sleep'),
                   mock.patch.object(Brain, 'elapsed', lambda self: 1.0),
                   mock.patch.object(Brain, 'turn', lambda self, a: (self.robot.turn_by(a), {'turned': a})[1]),
                   mock.patch.object(Brain, 'walk', lambda self, d, b, vx: (self.robot.walk_by(d, b), {'walked': d})[1]),
                   mock.patch('microduck_toy.stage3.brain.approach.send_pulse')]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)


class LookTests(StubbedMotion):
    def candidate(self, b, xy, frames=2):
        spot = b.seen.add('bus', xy, 0.25, 1.0)
        spot.frames = frames
        b.candidate = spot
        return spot

    def test_near_target_confirmed_goes_to_the_b_segment(self):
        robot = FakeRobot()
        b = make_brain(robot, toy_at('bus', (0.55, 0.0)))
        spot = self.candidate(b, (0.5, 0.05))
        self.assertEqual(b.s_look(), 'APPROACH')
        self.assertEqual(spot.status, 'target')
        self.assertAlmostEqual(spot.x, 0.55)

    def test_farther_target_confirmed_walks_first(self):
        robot = FakeRobot()
        b = make_brain(robot, toy_at('bus', (1.2, 0.0)))
        self.candidate(b, (1.15, 0.0))
        self.assertEqual(b.s_look(), 'GOTO')
        self.assertEqual(b.candidate.status, 'target')

    def test_beyond_confirm_distance_is_not_confirmed(self):
        robot = FakeRobot()
        b = make_brain(robot, toy_at('bus', (1.8, 0.0)))
        spot = self.candidate(b, (1.8, 0.0))
        self.assertEqual(b.s_look(), 'GOTO')
        self.assertEqual(spot.status, 'seen')

    def test_distractor_at_the_spot(self):
        robot = FakeRobot()
        b = make_brain(robot, toy_at('thomas', (0.8, 0.0)))
        spot = self.candidate(b, (0.8, 0.0))
        self.assertEqual(b.s_look(), 'SPIN')
        self.assertEqual(spot.status, 'distractor')
        self.assertIn((0.8, 0.0), [tuple(round(v, 6) for v in o) for o in b.obstacles()])

    def test_a_confirmed_target_spot_is_not_relabelled(self):
        robot = FakeRobot()
        b = make_brain(robot, toy_at('thomas', (0.8, 0.0)))
        spot = self.candidate(b, (0.8, 0.0))
        spot.status = 'target'                   # confirmed before a fall moved the odometry
        b.s_look()
        self.assertEqual(spot.status, 'target')
        self.assertEqual(spot.misses, 1)

    def test_two_empty_looks_drop_the_candidate(self):
        robot = FakeRobot()
        b = make_brain(robot)
        spot = self.candidate(b, (0.8, 0.0))
        self.assertEqual(b.s_look(), 'LOOK')          # one miss: still the best candidate
        self.assertEqual(b.s_look(), 'SPIN')
        self.assertEqual(spot.status, 'dropped')

    def test_turns_to_a_candidate_off_to_the_side(self):
        robot = FakeRobot()
        b = make_brain(robot, toy_at('bus', (0.0, 0.6)))
        self.candidate(b, (0.0, 0.6))
        b.s_look()
        self.assertAlmostEqual(robot.yaw, math.pi / 2)

    def test_a_candidate_below_the_window_goes_to_the_b_segment_without_a_look(self):
        robot = FakeRobot()
        b = make_brain(robot)                    # nothing in any frame: the window cannot see it
        spot = self.candidate(b, (0.26, 0.0))    # Mac 3ab: a GOTO leg ended 0.26 m from the target
        self.assertEqual(b.s_look(), 'APPROACH')
        self.assertEqual((b.eyes.seq, spot.misses, spot.status), (0, 0, 'seen'))

    def test_needs_spin_hits_to_be_a_candidate(self):
        robot = FakeRobot()
        b = make_brain(robot)
        b.seen.add('bus', (1.0, 0.0), 0.25, 1.0)
        self.assertIsNone(b.next_candidate())
        b.seen.add('bus', (1.0, 0.0), 0.25, 1.0)
        self.assertIsNotNone(b.next_candidate())


class SpinTests(StubbedMotion):
    def test_stops_on_the_second_sighting(self):
        robot = FakeRobot()
        step = math.radians(20)
        toy = (0.0, -1.0)                      # 270 deg to the left

        def script(r, seq):
            ahead = body_of(r, toy)
            return toy_at('bus', toy)(r, seq) if abs(math.atan2(ahead[1], ahead[0])) < math.radians(22.5) else []
        b = make_brain(robot, script, before=lambda r: r.turn_by(step))
        self.assertEqual(b.s_spin(), 'LOOK')
        self.assertLess(math.dist(b.candidate.xy, toy), 1e-6)
        self.assertEqual(b.spins, 0)
        self.assertIn(('move', 0.0, 0.0, 0.0), robot.calls)

    def test_empty_spins_relocate_then_fail(self):
        robot = FakeRobot()
        b = make_brain(robot, before=lambda r: r.turn_by(math.radians(30)))
        self.assertEqual(b.s_spin(), 'RELOCATE')
        with self.assertRaises(brain_lib.Ended) as done:
            b.s_spin()
        self.assertEqual(done.exception.reason, 'search_fail')

    def test_relocate_walks_to_the_nearest_free_spot(self):
        robot = FakeRobot()
        b = make_brain(robot)
        self.assertEqual(b.s_relocate(), 'SPIN')
        x, y, _ = b.here()
        self.assertLessEqual(min(math.dist((x, y), p) for p in S3['search']['relocate_spots']), 0.1)
        self.assertGreaterEqual(math.dist((x, y), (0.0, 0.0)), S3['search']['relocate_min_m'] - 0.1)


class PickTests(StubbedMotion):
    def run_brain(self, results, **kw):
        robot = FakeRobot(-0.3, 0.0, 0.0)
        b = make_brain(robot, start_state='PICK', **kw)
        b.toy_at = (0.08, 0.015)
        b.candidate = b.seen.add('bus', (-0.22, 0.0), 1.0, 0.0)
        it = iter(results)
        with mock.patch('microduck_toy.stage3.brain.ground_pick', lambda *a, **k: next(it)), \
             mock.patch.object(Brain, 's_approach', lambda self: 'LAST'), \
             mock.patch.object(Brain, 's_last', lambda self: 'PICK'):
            return b.run()

    def test_three_misses_end_in_grasp_fail(self):
        res = self.run_brain([{'ran': True, 'held': False}] * 3)
        self.assertEqual((res['end'], res['closes']), ('grasp_fail', 3))
        self.assertEqual([h[0] for h in res['history']], ['PICK', 'APPROACH', 'LAST'] * 2 + ['PICK'])

    def test_held_until_held_ends_held(self):
        res = self.run_brain([{'ran': True, 'held': False}, {'ran': True, 'held': True}], until='held')
        self.assertEqual((res['end'], res['closes'], res['holding']), ('held', 2, True))

    def test_held_goes_to_carry(self):
        robot = FakeRobot()
        b = make_brain(robot, start_state='PICK')
        with mock.patch('microduck_toy.stage3.brain.ground_pick', lambda *a, **k: {'ran': True, 'held': True}):
            self.assertEqual(b.s_pick(), 'CARRY')

    def test_pick_that_did_not_run_is_an_error(self):
        res = self.run_brain([{'ran': False, 'error': 'no slot'}])
        self.assertEqual(res['end'], 'error')
        self.assertIn('no slot', res['error'])


class FallTests(StubbedMotion):
    def fallen_brain(self, holding, mouth_deg, up=True):
        robot = FakeRobot()
        robot.policy = 'stand' if up else 'limp'
        b = make_brain(robot, holding=holding)
        patch = mock.patch('microduck_toy.stage3.brain.mouth_angle', lambda s: math.radians(mouth_deg))
        patch.start()
        self.addCleanup(patch.stop)
        return b

    def test_still_holding_carries_on(self):
        b = self.fallen_brain(True, 3.0)       # closed on a 2 cm toy: -5 + 4 x 2 deg
        self.assertEqual(b.s_fallen(), 'CARRY')

    def test_toy_lost_in_the_fall_looks_at_the_feet(self):
        b = self.fallen_brain(True, -5.0)      # closed on nothing
        self.assertEqual(b.s_fallen(), 'RECOVER')
        self.assertFalse(b.holding)

    def test_not_holding_goes_back_to_the_candidate(self):
        b = self.fallen_brain(False, -5.0)
        b.candidate = b.seen.add('bus', (1.0, 0.0), 1.0, 1.0)
        self.assertEqual(b.s_fallen(), 'LOOK')
        b.candidate = None
        self.assertEqual(b.s_fallen(), 'SPIN')

    def test_fall_while_recovering_recovers_again(self):
        b = self.fallen_brain(False, -5.0)
        b.lost = True
        b.candidate = b.seen.add('bus', (1.0, 0.0), 1.0, 1.0)
        self.assertEqual(b.s_fallen(), 'RECOVER')

    def test_recover_sees_the_toy_at_the_feet(self):
        robot = FakeRobot()
        b = make_brain(robot, toy_at('bus', (0.25, 0.0)), holding=False)
        b.lost = True
        b.candidate = b.seen.add('bus', (1.0, 0.0), 1.0, 1.0)
        b.candidate.status = 'target'
        self.assertEqual(b.s_recover(), 'APPROACH')
        self.assertAlmostEqual(b.candidate.x, 0.25)
        self.assertFalse(b.lost)

    def test_recover_backs_off_from_a_toy_under_the_beak(self):
        robot = FakeRobot()
        b = make_brain(robot, toy_at('bus', (0.06, 0.0)), holding=False)
        b.lost = True
        brain_lib.approach.send_pulse.reset_mock()
        self.assertEqual(b.s_recover(), 'APPROACH')
        brain_lib.approach.send_pulse.assert_called_once()
        self.assertEqual([c for c in robot.calls if c[0] == 'head'], [('head', 'H2')])

    def test_recover_without_the_toy_backs_off_and_looks_again_then_searches_afresh(self):
        robot = FakeRobot()
        b = make_brain(robot, holding=False)
        b.lost, b.spins = True, 1
        old = b.candidate = b.seen.add('bus', (1.0, 0.0), 1.0, 1.0)
        old.status = 'target'
        self.assertEqual(b.s_recover(), 'RECOVER')
        self.assertTrue(b.lost)
        self.assertEqual(b.s_recover(), 'SPIN')
        self.assertEqual(old.status, 'dropped')
        self.assertIsNone(b.candidate)
        self.assertEqual(b.spins, 0)
        self.assertFalse(b.lost)

    def test_not_up_yet_waits_again(self):
        b = self.fallen_brain(False, -5.0, up=False)
        self.assertEqual(b.s_fallen(), 'FALLEN')

    def test_fall_in_any_state_goes_to_fallen(self):
        robot = FakeRobot()
        b = make_brain(robot, start_state='GOTO')
        seen = []

        def goto(self):
            raise Fallen()

        def fallen(self):
            seen.append(self.state)
            self.end('timeout')
        with mock.patch.object(Brain, 's_goto', goto), mock.patch.object(Brain, 's_fallen', fallen):
            res = b.run()
        self.assertEqual(seen, ['FALLEN'])
        self.assertEqual([h[0] for h in res['history']], ['GOTO', 'FALLEN'])

    def test_check_sees_a_new_fall(self):
        robot = FakeRobot()
        b = make_brain(robot)
        b.check()
        robot.falls.append({'mono': 1.0})
        with self.assertRaises(Fallen):
            b.check()


class BoardImportTests(unittest.TestCase):
    def test_brain_imports_without_the_simulator(self):
        """The board has no MuJoCo: the brain process must not import it (plan 3F)."""
        code = textwrap.dedent('''
            import importlib.abc, sys
            class Block(importlib.abc.MetaPathFinder):
                def find_spec(self, name, path, target=None):
                    if name.split('.')[0] in ('mujoco', 'mjlab_microduck', 'glfw', 'torch', 'ultralytics'):
                        raise ImportError('blocked ' + name)
            sys.meta_path.insert(0, Block())
            import microduck_toy.stage3.brain, microduck_toy.stage3.eyes, microduck_toy.stage2.deploy
        ''')
        env = {**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'src')}
        done = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)


class LoopTests(unittest.TestCase):
    def test_unexpected_error_ends_the_episode(self):
        robot = FakeRobot()
        b = make_brain(robot)
        with mock.patch.object(Brain, 's_spin', lambda self: 1 / 0):
            res = b.run()
        self.assertEqual(res['end'], 'error')
        self.assertIn('ZeroDivisionError', res['error'])
        self.assertEqual(robot.calls[-1], ('move', 0.0, 0.0, 0.0))

    def test_clock_ends_the_episode(self):
        robot = FakeRobot()
        b = make_brain(robot)
        with mock.patch('microduck_toy.stage3.brain.time.sleep'), \
             mock.patch.object(Brain, 'elapsed', lambda self: S3['episode']['max_s'] + 1):
            res = b.run()
        self.assertEqual(res['end'], 'timeout')

    def test_unknown_target_or_state(self):
        with self.assertRaises(ValueError):
            make_brain(FakeRobot(), target='croc')
        with self.assertRaises(ValueError):
            make_brain(FakeRobot(), start_state='DANCE')


class CarryTests(StubbedMotion):
    def setUp(self):
        super().setUp()
        p = mock.patch.object(Brain, 'holding_now', lambda self: True)
        p.start()
        self.addCleanup(p.stop)

    def carry(self, robot, **kw):
        b = make_brain(robot, start_state='CARRY', holding=True, **kw)
        states = []
        for _ in range(20):
            states.append(b.s_carry())
            if states[-1] != 'CARRY':
                break
        return b, states

    def test_carries_to_the_nearest_side_and_squares_up(self):
        robot = FakeRobot(-0.6, 0.1, 2.0)
        b, states = self.carry(robot)
        self.assertEqual(states[-1], 'TRAY_LOOK')
        self.assertEqual(b.side['side'], 'x-')
        self.assertLessEqual(math.dist((robot.x, robot.y), b.side['staging']), S3['carry']['arrive_m'])
        self.assertLess(abs(math.atan2(math.sin(robot.yaw), math.cos(robot.yaw))), math.radians(5))

    def test_goes_round_a_known_toy_in_the_x_minus_lane(self):
        cx, cy = CONFIGS['stage1']['arena']['tray']['centre_m']
        robot = FakeRobot(-0.6, 0.0, 0.0)
        b, states = self.carry(robot, known=[('croc', (cx - 0.3, cy))])
        self.assertEqual(states[-1], 'TRAY_LOOK')
        self.assertNotEqual(b.side['side'], 'x-')
        face = tray_lib.facing(b.side['side'])
        self.assertLess(abs(math.atan2(math.sin(robot.yaw - face), math.cos(robot.yaw - face))), math.radians(5))

    def test_a_start_just_outside_the_keep_in_square_still_gets_going(self):
        slide = CONFIGS['stage1']['approach']['walk_slide_m']
        fld = make_brain(FakeRobot(0.0, 0.0, 0.0)).fld
        lim = fld.half_m - fld.keep_in_m
        robot = FakeRobot(-0.09, -lim - 0.035, 1.42)         # cloud 3D: about 4 cm out after a fall
        coast = lambda self, d, b, vx: (self.robot.walk_by(d, b) if d > slide else None, {'walked': d})[1]
        with mock.patch.object(Brain, 'walk', coast):        # walks shorter than the coast never start
            b, states = self.carry(robot)
        self.assertEqual(states[-1], 'TRAY_LOOK')

    def test_tof_stops_that_wall_in_every_side_are_dropped(self):
        robot = FakeRobot(-0.2, 0.0, 0.0)
        b = make_brain(robot, start_state='CARRY', holding=True)
        b.extra = [((-0.2, 0.0), 0.5)]           # post-fall ToF points (Mac 3D) stood in for a wall
        logged = []
        b.log = lambda event, **f: logged.append((event, f))
        real = tray_lib.choose_side

        def walled(start, obstacles, fld, cfg):
            # every side is blocked while any ToF point is still in the list
            if b.extra:
                return {'side': None, 'options': []}
            return real(start, obstacles, fld, cfg)
        states = []
        with mock.patch.object(tray_lib, 'choose_side', walled):
            for _ in range(20):
                states.append(b.s_carry())
                if states[-1] != 'CARRY':
                    break
        self.assertEqual(states[-1], 'TRAY_LOOK')
        self.assertEqual(b.extra, [])
        self.assertIn(('no_route', {'dropped_tof_obstacles': 1}), logged)

    def test_with_no_route_at_all_the_nearest_side_is_walked_to_straight(self):
        robot = FakeRobot(-0.2, -0.5, 0.0)
        with mock.patch.object(tray_lib, 'choose_side', lambda *a, **k: {'side': None, 'options': []}):
            b, states = self.carry(robot)
        self.assertEqual(states[-1], 'TRAY_LOOK')
        self.assertEqual(b.side['side'], 'x-')          # staging (0.2, 0) is the nearest from (-0.2, -0.5)

    def test_dropped_on_the_way_recovers(self):
        robot = FakeRobot(-0.6, 0.0, 0.0)
        b = make_brain(robot, start_state='CARRY', holding=True)
        with mock.patch.object(Brain, 'holding_now', lambda self: False):
            self.assertEqual(b.s_carry(), 'RECOVER')
        self.assertFalse(b.holding)


class TrayLookTests(StubbedMotion):
    def test_measures_then_drops(self):
        robot = FakeRobot(0.2, 0.0, 0.0)
        b = make_brain(robot, start_state='TRAY_LOOK', holding=True)
        self.assertEqual(b.s_tray_look(), 'DROP')         # no tray box: odometry centre, straight ahead
        self.assertEqual(b.tray['source'], 'odometry')
        self.assertAlmostEqual(b.tray['centre'][0], 0.45)

    def test_turns_when_the_tray_is_off_to_the_side(self):
        robot = FakeRobot(0.2, 0.0, math.radians(-30))
        b = make_brain(robot, start_state='TRAY_LOOK', holding=True)
        self.assertEqual(b.s_tray_look(), 'TRAY_LOOK')
        self.assertAlmostEqual(robot.yaw, 0.0, places=6)

    def fake_measure(self, centre):
        """measure_tray stand-in: the tray box at `centre` (forward, left), kept if within accept_m."""
        seen = []

        def measure(looks, cam, cfg, half, rim, expected, near_z_m=0.0):
            seen.append(cfg['accept_m'])
            if math.dist(centre, expected) > cfg['accept_m']:
                return {'frames': 5, 'used': 0, 'rejected': 5, 'expected': list(expected),
                        'centre': list(expected), 'source': 'odometry'}
            return {'frames': 5, 'used': 5, 'rejected': 0, 'expected': list(expected),
                    'centre': list(centre), 'source': 'yolo'}
        return seen, mock.patch('microduck_toy.stage3.tray.measure_tray', measure)

    def test_after_a_fall_the_real_tray_resets_the_pose_and_carries_again(self):
        robot = FakeRobot(-0.1, 0.0, 0.0)                  # really 0.75 m from the centre
        b = make_brain(robot, start=(0.2, 0.0, 0.0), start_state='TRAY_LOOK', holding=True)
        b.pose_doubt = True                                # odometry says staging point (0.45 m)
        seen, patch = self.fake_measure([0.75, 0.0])
        with patch:
            self.assertEqual(b.s_tray_look(), 'CARRY')
        self.assertEqual(seen, [S3['tray_look']['accept_after_fall_m']])
        self.assertEqual(b.tray['reanchor_m'], [-0.3, 0.0])
        self.assertFalse(b.pose_doubt)
        self.assertAlmostEqual(b.here()[0], -0.1)          # the pose is the real one now

    def test_a_tray_a_little_far_is_walked_up_to(self):
        robot = FakeRobot(0.03, 0.0, 0.0)                  # 0.62 m from the centre, no fall
        b = make_brain(robot, start=(0.03, 0.0, 0.0), start_state='TRAY_LOOK', holding=True)
        seen, patch = self.fake_measure([0.62, 0.0])
        with patch, mock.patch.object(Brain, 'holding_now', lambda self: True):
            self.assertEqual(b.s_tray_look(), 'TRAY_LOOK')
            self.assertAlmostEqual(robot.x, 0.2)           # walked 0.17 m up to the staging range
            self.assertEqual(b.s_tray_look(), 'DROP')      # second look: in range, no second walk
        self.assertEqual(seen, [S3['tray_look']['accept_m']] * 2)

    def test_without_a_fall_a_far_tray_box_is_not_trusted(self):
        robot = FakeRobot(-0.2, 0.0, 0.0)
        b = make_brain(robot, start=(0.2, 0.0, 0.0), start_state='TRAY_LOOK', holding=True)
        seen, patch = self.fake_measure([0.85, 0.0])
        with patch:
            self.assertEqual(b.s_tray_look(), 'DROP')
        self.assertEqual(b.tray['source'], 'odometry')
        self.assertNotIn('reanchor_m', b.tray)

    def seen_when_facing(self, robot):
        """measure_tray stand-in: the real tray box only while the tray is inside the +-22.5 deg picture."""
        cx, cy = CONFIGS['stage1']['arena']['tray']['centre_m']

        def measure(looks, cam, cfg, half, rim, expected, near_z_m=0.0):
            dx, dy = cx - robot.x, cy - robot.y
            c, s = math.cos(robot.yaw), math.sin(robot.yaw)
            body = (c * dx + s * dy, -s * dx + c * dy)
            if abs(math.atan2(body[1], body[0])) > math.radians(22.5) or math.dist(body, expected) > cfg['accept_m']:
                return {'frames': 5, 'used': 0, 'centre': list(expected), 'source': 'odometry'}
            return {'frames': 5, 'used': 5, 'centre': list(body), 'source': 'yolo'}
        return mock.patch('microduck_toy.stage3.tray.measure_tray', measure)

    def test_after_a_fall_a_tray_out_of_the_picture_is_found_by_turning(self):
        cx, cy = CONFIGS['stage1']['arena']['tray']['centre_m']
        robot = FakeRobot(cx, cy - 0.45, 0.0)             # the tray is 90 deg to the left
        b = make_brain(robot, start=(cx - 0.45, cy, 0.0), start_state='TRAY_LOOK', holding=True)
        b.pose_doubt = True                                # odometry says: at the staging point, facing it
        with self.seen_when_facing(robot):
            states = [b.s_tray_look() for _ in range(3)]
        self.assertEqual(states, ['TRAY_LOOK', 'TRAY_LOOK', 'CARRY'])
        self.assertEqual(b.tray['source'], 'yolo')
        self.assertAlmostEqual(math.degrees(robot.yaw), 90.0, places=4)
        self.assertAlmostEqual(b.here()[0], cx, places=4)  # re-anchored on the real tray
        self.assertAlmostEqual(b.here()[1], cy - 0.45, places=4)

    def test_after_a_fall_the_scan_ends_and_the_drop_goes_on_odometry(self):
        cx, cy = CONFIGS['stage1']['arena']['tray']['centre_m']
        robot = FakeRobot(cx + 0.5, cy, 0.0)               # the tray is behind: never in the picture
        b = make_brain(robot, start=(cx - 0.45, cy, 0.0), start_state='TRAY_LOOK', holding=True)
        b.pose_doubt = True
        with self.seen_when_facing(robot):
            states = []
            for _ in range(10):
                states.append(b.s_tray_look())
                if states[-1] != 'TRAY_LOOK':
                    break
        self.assertEqual(b.tray_scans, len(S3['tray_look']['scan_after_fall_deg']))
        self.assertEqual(states[-1], 'DROP')
        self.assertEqual(b.tray['source'], 'odometry')

    def test_without_a_fall_there_is_no_scan(self):
        robot = FakeRobot(0.2, 0.0, 0.0)
        b = make_brain(robot, start_state='TRAY_LOOK', holding=True)
        self.assertEqual(b.s_tray_look(), 'DROP')
        self.assertEqual(b.tray_scans, 0)

    def test_a_fall_marks_the_pose_doubtful(self):
        robot = FakeRobot()
        b = make_brain(robot, start_state='SPIN')
        with mock.patch.object(Brain, 'holding_now', lambda self: False):
            b.s_fallen()
        self.assertTrue(b.pose_doubt)

    def test_drop_walks_to_d_and_opens(self):
        robot = FakeRobot(0.2, 0.0, 0.0)
        b = make_brain(robot, start_state='DROP', holding=True, drop_d=0.21)
        b.tray = {'centre': [0.45, 0.0], 'source': 'yolo'}
        with mock.patch.object(Brain, 'holding_now', lambda self: False), \
             self.assertRaises(brain_lib.Ended) as done:
            b.s_drop()
        self.assertEqual(done.exception.reason, 'dropped')
        self.assertAlmostEqual(robot.x, 0.65 - 0.21)
        self.assertIn(('mouth', 1.0), robot.calls)
        self.assertEqual(robot.calls[-1], ('mouth', 0.0))

    def test_the_stage3_drop_has_no_v2_steps(self):
        self.assertNotIn('staged', S3['drop'])
        self.assertNotIn('leave', S3['drop'])
        self.assertEqual(make_brain(FakeRobot()).fld.tray_exit_m, 0.0)


# ── Stage 5 v2 (configs/demo/room.json): staged drop, settle, leave the tray ──

ROOM = brain_lib.load_configs(override='configs/demo/room.json')
RS3 = ROOM['stage3']
STILL = {'settle_s': 0.0, 'still': True}


def make_room_brain(robot, **kw):
    kw.setdefault('start', (robot.x, robot.y, robot.yaw))
    b = Brain(robot, FakeEyes(robot, lambda r, s: []), None, ROOM, kw.pop('target', 'bus'), **kw)
    b.t0 = 0.0
    return b


def pulse_moves(robot, script=None):
    """approach.send_pulse stand-in: the fake robot goes the pulse's mean forward distance, or what
    script(pulse) returns: (forward m, turn rad)."""
    def send(rob, pulse):
        fwd, turn = script(pulse) if script else (pulse.forward_m[0], 0.0)
        robot.walk_by(fwd, 0.0)
        robot.turn_by(turn)
        robot.calls.append(('pulse', pulse.name))
    return mock.patch('microduck_toy.stage3.brain.approach.send_pulse', send)


class StagedDropTests(StubbedMotion):
    CX, CY = ROOM['stage1']['arena']['tray']['centre_m']

    def setUp(self):
        super().setUp()
        for p in (mock.patch.object(Brain, 'settle', lambda self: dict(STILL)),
                  mock.patch.object(Brain, 'holding_now', lambda self: False)):
            p.start()
            self.addCleanup(p.stop)

    def dropper(self, back=0.45):
        robot = FakeRobot(self.CX - back, self.CY, 0.0)            # facing the tray from x-
        b = make_room_brain(robot, start_state='DROP', holding=True)
        b.tray = {'centre': [back, 0.0], 'source': 'yolo'}
        return robot, b

    def run_drop(self, b):
        with self.assertRaises(brain_lib.Ended) as done:
            b.s_drop()
        self.assertEqual(done.exception.reason, 'dropped')
        return done.exception.detail['drop']

    def test_walks_to_the_standoff_creeps_in_and_opens_inside_the_window(self):
        robot, b = self.dropper()
        st, d = RS3['drop']['staged'], RS3['drop']['d_m']
        opened = []
        b.robot.mouth = lambda v: (opened.append(math.dist((robot.x, robot.y), (self.CX, self.CY)))
                                   if v == 1.0 else None)
        with pulse_moves(robot):
            drop = self.run_drop(b)
        acts = [s['act'] for s in drop['steps']]
        self.assertEqual(acts[-1], 'open')
        self.assertTrue(set(acts[:-1]) <= {st['pulse_long'], st['pulse_short']}, acts)
        self.assertAlmostEqual(drop['steps'][0]['range_m'], st['standoff_m'], places=6)
        self.assertGreaterEqual(opened[0], d - st['deep_tol_m'])
        self.assertLessEqual(opened[0], d + st['far_tol_m'])

    def test_then_backs_out_of_the_tray_and_turns_to_the_field(self):
        robot, b = self.dropper()
        with pulse_moves(robot):
            drop = self.run_drop(b)
        leave = drop['leave']
        self.assertGreaterEqual(leave['range_m'], RS3['drop']['leave']['clear_m'])
        self.assertLessEqual(leave['pulses'], RS3['drop']['leave']['max_pulses'])
        self.assertIn('turn', leave)
        here = b.here()
        self.assertLess(abs(brain_lib.bearing_to(here, (0.0, 0.0))), math.radians(1))

    def test_a_step_into_the_tray_is_backed_off_before_it_opens(self):
        robot, b = self.dropper()
        lurched = []

        def script(pulse):                         # the first forward pulse lurches 18 cm in
            if pulse.forward_m[0] > 0 and not lurched:
                lurched.append(1)
                return 0.18, 0.0
            return pulse.forward_m[0], 0.0
        with pulse_moves(robot, script):
            drop = self.run_drop(b)
        acts = [s['act'] for s in drop['steps']]
        self.assertIn('retreat', acts)
        self.assertEqual(drop['retreats'], 1)
        self.assertEqual(acts[-1], 'open')
        self.assertGreaterEqual(drop['steps'][-1]['range_m'],
                                RS3['drop']['d_m'] - RS3['drop']['staged']['deep_tol_m'])

    def scripted(self, ranges):
        """Run the staged drop with tray_range scripted: [(range m, bearing deg), ...] per settle."""
        robot, b = self.dropper()
        seq = iter([(r, math.radians(a)) for r, a in ranges])
        last = [None]

        def tray_range(self_):
            last[0] = next(seq, last[0])
            return last[0]
        with mock.patch.object(Brain, 'tray_range', tray_range), pulse_moves(robot), \
                mock.patch.object(Brain, 'leave_tray', lambda self_, cfg: {}):
            drop = self.run_drop(b)
        return [s['act'] for s in drop['steps']]

    def test_turned_off_the_centre_in_the_window_backs_off_turns_and_comes_in_again(self):
        st = RS3['drop']['staged']
        acts = self.scripted([(0.25, 0.0), (0.12, 30.0), (0.225, 30.0), (0.225, 0.0), (0.115, 0.0)])
        self.assertEqual(acts, [st['pulse_long'], 'retreat', 'turn', st['pulse_long'], 'open'])

    def test_no_turning_once_the_toes_are_over_the_rim(self):
        st = RS3['drop']['staged']
        acts = self.scripted([(0.18, 15.0), (0.12, 15.0)])
        self.assertEqual(acts, [st['pulse_long'], 'open'])

    def test_short_of_the_window_by_little_takes_the_short_pulse(self):
        st, d = RS3['drop']['staged'], RS3['drop']['d_m']
        acts = self.scripted([(d + st['long_above_m'] - 0.005, 0.0), (d, 0.0)])
        self.assertEqual(acts, [st['pulse_short'], 'open'])

    # v3: out of steps or retreats short of the window, the beak stays shut (v2 opened there:
    # runs/20261112T034551Z_s5_v2_validation, Mac 5000 ketchup 133 deg off, board 5002 ketchup 103).

    @staticmethod
    def stuck(pulse):
        """Creeps that do not move the robot (it never reaches the window); retreats as measured."""
        return (0.0, 0.0) if pulse.forward_m[0] > 0 else (pulse.forward_m[0], 0.0)

    def range_of(self, robot):
        return math.dist((robot.x, robot.y), (self.CX, self.CY))

    def test_out_of_steps_short_of_the_window_the_beak_stays_shut(self):
        st = RS3['drop']['staged']
        robot, b = self.dropper()
        with pulse_moves(robot, self.stuck):
            out = b.staged_drop(RS3['drop'], b.tray['centre'])
        self.assertEqual(out['result'], 'out_of_steps')
        self.assertEqual(len(out['steps']), st['max_steps'] + 1)
        self.assertNotIn(('mouth', 1.0), robot.calls)

    def test_out_of_retreats_the_beak_stays_shut(self):
        st = RS3['drop']['staged']
        robot, b = self.dropper()

        def deep(pulse):                           # every creep lurches past the window
            return (0.20, 0.0) if pulse.forward_m[0] > 0 else (pulse.forward_m[0], 0.0)
        with pulse_moves(robot, deep):
            out = b.staged_drop(RS3['drop'], b.tray['centre'])
        self.assertEqual(out['result'], 'out_of_retreats')
        self.assertEqual(out['retreats'], st['max_retreats'])
        self.assertNotIn(('mouth', 1.0), robot.calls)

    def test_not_ready_backs_out_facing_the_tray_and_looks_again(self):
        st = RS3['drop']['staged']
        robot, b = self.dropper()
        with pulse_moves(robot, self.stuck), mock.patch.object(Brain, 'holding_now', lambda self: True):
            nxt = b.s_drop()
        self.assertEqual(nxt, 'TRAY_LOOK')
        self.assertEqual(b.drop_redos, 1)
        self.assertNotIn(('mouth', 1.0), robot.calls)
        self.assertGreaterEqual(self.range_of(robot), st['redo_back_m'])
        here = b.here()
        self.assertLess(abs(brain_lib.bearing_to(here, (self.CX, self.CY))), math.radians(1))

    def test_redo_with_the_toy_gone_recovers(self):
        robot, b = self.dropper()
        with pulse_moves(robot, self.stuck):     # holding_now False (setUp): it came out
            nxt = b.s_drop()
        self.assertEqual(nxt, 'RECOVER')
        self.assertTrue(b.lost)

    def test_after_the_redos_it_gives_up_outside_the_tray(self):
        st = RS3['drop']['staged']
        robot, b = self.dropper()
        b.drop_redos = st['max_redos']
        opened_at = []
        b.robot.mouth = lambda v: opened_at.append(self.range_of(robot)) if v == 1.0 else None
        with pulse_moves(robot, self.stuck), mock.patch.object(Brain, 'holding_now', lambda self: True), \
                self.assertRaises(brain_lib.Ended) as done:
            b.s_drop()
        self.assertEqual(done.exception.reason, 'drop_fail')
        self.assertEqual(done.exception.detail['not_ready'], 'out_of_steps')
        drop = done.exception.detail['drop']
        self.assertFalse(drop['redo'])
        self.assertTrue(drop['back']['clear'])
        self.assertEqual(len(opened_at), 1)
        self.assertGreaterEqual(opened_at[0], st['redo_back_m'])
        self.assertTrue(drop['leave']['clear'])

    def test_a_fall_while_backing_out_for_a_redo_goes_to_fallen_with_the_redo_counted(self):
        robot, b = self.dropper()
        calls = []

        def settle(self_):
            calls.append(1)
            if len(calls) > RS3['drop']['staged']['max_steps'] + 2:   # the second back_out step
                raise Fallen()
            return dict(STILL)
        with pulse_moves(robot, self.stuck), mock.patch.object(Brain, 'settle', settle), \
                self.assertRaises(Fallen):
            b.s_drop()
        self.assertEqual(b.drop_redos, 1)
        self.assertNotIn(('mouth', 1.0), robot.calls)

    # v3 leave: back straight out, turn only with the toes off the rim, never turn round in the tray.

    def leaver(self, r=None, yaw=0.0):
        robot, b = self.dropper(back=r if r is not None else RS3['drop']['d_m'])
        robot.yaw = yaw
        return robot, b

    def test_leave_squares_up_to_the_centre_only_with_the_toes_off_the_rim(self):
        robot, b = self.leaver()

        def drift(pulse):                          # each B10 turns the robot 20 deg
            return (pulse.forward_m[0], math.radians(20)) if pulse.forward_m[0] < 0 else (pulse.forward_m[0], 0.0)
        with pulse_moves(robot, drift):
            leave = b.leave_tray(RS3['drop']['leave'])
        acts = [s['act'] for s in leave['steps']]
        self.assertEqual(acts, ['B10', 'B10', 'turn', 'B10', 'clear'])
        turn = leave['steps'][2]
        self.assertGreaterEqual(turn['range_m'], RS3['drop']['staged']['turn_above_m'])
        self.assertTrue(leave['clear'])
        self.assertIn('turn', leave)               # then it faces the field

    def test_spinning_on_the_rim_stops_not_clear_and_does_not_turn_round(self):
        robot, b = self.leaver()

        def spin(pulse):                           # board 5000 ketchup: no range, all yaw
            return (0.0, math.radians(46))
        with pulse_moves(robot, spin):
            leave = b.leave_tray(RS3['drop']['leave'])
        acts = [s['act'] for s in leave['steps']]
        self.assertEqual(acts, ['B10', 'B10', 'no_gain'])
        self.assertFalse(leave['clear'])
        self.assertEqual(leave['result'], 'no_gain')
        self.assertNotIn('turn', leave)

    def test_out_of_pulses_short_of_clear_does_not_turn_round(self):
        cfg = RS3['drop']['leave']
        robot, b = self.leaver()
        with pulse_moves(robot, lambda p: (-0.03, 0.0)):
            leave = b.leave_tray(cfg)
        self.assertEqual(leave['pulses'], cfg['max_pulses'])
        self.assertEqual(leave['result'], 'out_of_pulses')
        self.assertFalse(leave['clear'])
        self.assertNotIn('turn', leave)
        self.assertLess(leave['range_m'], cfg['clear_m'])

    def test_with_the_centre_behind_it_walks_forward_out(self):
        robot, b = self.leaver(r=0.15, yaw=math.pi)            # Mac 5000 ketchup: B10 went in
        with pulse_moves(robot):
            leave = b.leave_tray(RS3['drop']['leave'])
        steps = leave['steps']
        self.assertEqual(steps[0]['act'], RS3['drop']['staged']['pulse_long'])
        self.assertGreater(steps[1]['range_m'], steps[0]['range_m'])
        self.assertTrue(leave['clear'])
        self.assertTrue(all(s['range_m'] >= RS3['drop']['staged']['turn_above_m']
                            for s in steps if s['act'] == 'turn'))

    def test_a_fall_while_leaving_is_logged_not_raised(self):
        robot, b = self.dropper(back=RS3['drop']['d_m'])           # standing where the beak opened
        calls = []

        def fell_second(self_):
            calls.append(1)
            if len(calls) > 1:
                raise Fallen()
            return dict(STILL)
        with mock.patch.object(Brain, 'settle', fell_second), pulse_moves(robot):
            leave = b.leave_tray(RS3['drop']['leave'])
        self.assertTrue(leave['fallen'])
        self.assertFalse(leave['clear'])
        self.assertEqual(leave['pulses'], 1)
        self.assertEqual(len(leave['steps']), 1)                   # what was done before the fall

    def test_the_clock_running_out_while_leaving_is_logged_not_raised(self):
        robot, b = self.dropper(back=RS3['drop']['d_m'])
        with mock.patch.object(Brain, 'elapsed', lambda self_: RS3['episode']['max_s'] + 1.0), \
                pulse_moves(robot):
            leave = b.leave_tray(RS3['drop']['leave'])
        self.assertTrue(leave['timeout'])
        self.assertFalse(leave['clear'])
        self.assertEqual(leave['pulses'], 0)


class StagedActTests(unittest.TestCase):
    """brain.staged_act, the v3 decision table, on the room config (d_m 0.11)."""
    st, d = RS3['drop']['staged'], RS3['drop']['d_m']

    def act(self, r, aim_deg, still=True, retreats=0):
        return brain_lib.staged_act(r, math.radians(aim_deg), still, self.d, self.st, retreats)

    def test_opens_only_in_the_window_aimed_and_still(self):
        self.assertEqual(self.act(self.d, 0.0), 'open')
        self.assertEqual(self.act(self.d, 0.0, still=False), 'wait')
        self.assertEqual(self.act(self.d, self.st['aim_open_deg'] + 5), 'retreat')

    def test_too_deep_retreats_and_then_is_out_of_retreats_not_open(self):
        deep = self.d - self.st['deep_tol_m'] - 0.01
        self.assertEqual(self.act(deep, 0.0), 'retreat')
        self.assertEqual(self.act(deep, 0.0, retreats=self.st['max_retreats']), 'out_of_retreats')
        self.assertEqual(self.act(self.d, 60.0, retreats=self.st['max_retreats']), 'out_of_retreats')

    def test_no_creeping_with_the_centre_off_to_the_side_over_the_rim(self):
        # Mac 5000 ketchup k=6,7: 12.6 cm, centre 125 deg off: v2 sent F2 (away from the centre).
        self.assertEqual(self.act(0.1258, 124.5), 'retreat')
        self.assertEqual(self.act(0.18, 15.0), self.st['pulse_long'])      # aimed enough: creep
        self.assertEqual(self.act(0.18, 40.0), 'retreat')                   # not: out, turn there

    def test_turns_only_with_the_toes_off_the_rim(self):
        self.assertEqual(self.act(self.st['turn_above_m'] + 0.01, 15.0), 'turn')
        self.assertEqual(self.act(self.st['turn_above_m'] + 0.01, 170.0), 'turn')
        self.assertEqual(self.act(self.st['turn_above_m'] - 0.01, 170.0), 'retreat')

    def test_away_is_back_with_the_centre_ahead_forward_with_it_behind(self):
        b = make_room_brain(FakeRobot())
        self.assertEqual(b.away(math.radians(80))[0], ROOM['stage1']['inching']['retreat'])
        self.assertEqual(b.away(math.radians(-122))[0], self.st['pulse_long'])


class SettleTests(unittest.TestCase):
    """Brain.settle on a robot.state stream: the fake clock is robot.state t."""

    class Stream(FakeRobot):
        def __init__(self, moving_s, vx=0.1, wz=0.0):
            super().__init__(0.0, 0.0, 0.0)
            self.t, self.moving_s, self.vx, self.wz = 0.0, moving_s, vx, wz

        def next_states(self, count, timeout=2.0):
            self.t += 0.02
            if self.t <= self.moving_s:
                self.x += self.vx * 0.02
                self.turn_by(self.wz * 0.02)
            s = self.snapshot()
            s['t'] = self.t
            return [s]

    def settle(self, robot):
        b = make_room_brain(robot)
        with mock.patch('microduck_toy.stage3.brain.time.monotonic', lambda: robot.t):
            return b.settle()

    def test_waits_for_the_walk_to_stop(self):
        cfg = RS3['drop']['settle']
        robot = self.Stream(moving_s=0.6)
        out = self.settle(robot)
        self.assertTrue(out['still'])
        self.assertGreaterEqual(out['settle_s'], 0.6 + cfg['window_s'] * 0.8 - 0.05)
        self.assertLess(out['settle_s'], cfg['max_s'])

    def test_a_robot_that_keeps_turning_is_not_still(self):
        robot = self.Stream(moving_s=99.0, vx=0.0, wz=math.radians(20))
        out = self.settle(robot)
        self.assertFalse(out['still'])
        self.assertGreaterEqual(out['settle_s'], RS3['drop']['settle']['max_s'])

    def test_already_still_waits_min_s(self):
        robot = self.Stream(moving_s=0.0)
        out = self.settle(robot)
        self.assertTrue(out['still'])
        self.assertLess(out['settle_s'], 0.5)

    def test_a_fall_while_settling(self):
        robot = self.Stream(moving_s=0.0)
        robot.fallen = True
        with self.assertRaises(Fallen):
            self.settle(robot)


if __name__ == '__main__':
    unittest.main()
