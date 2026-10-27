"""Stage 3 visual checks without a simulator: the closed-loop YOLO check (vision_loop.py), the
bird's-eye traces (trace.py) and the robot's-eye pictures (shots.py, Brain.shot)."""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from microduck_toy.stage3 import episode, shots, vision_loop as vl
from microduck_toy.stage3.eyes import Sight
from test_stage3_brain import FakeRobot, StubbedMotion, make_brain
from test_stage3_judge import row

trace = None
if importlib.util.find_spec('matplotlib') is not None:
    from microduck_toy.stage3 import trace
PIL = importlib.util.find_spec('PIL')


def standing(at=0.0, **toys):
    """H0 standing at (at, 0) facing +x (camera 8 cm ahead, 0.232 m up, 14 deg down)."""
    r = row(**toys)
    r.update(x=at, cam_x=at + 0.08, cam_y=0.0, cam_z=0.232, cam_pitch=0.246)
    return r


def box(cls, forward, left):
    return {'cls': cls, 'conf': 0.9, 'bearing': math.atan2(left, forward), 'dist': math.hypot(forward, left)}


class VisionLoopTests(unittest.TestCase):
    def test_correct_wrong_class_phantom_and_miss(self):
        r = standing(-0.3, bus=(0.7, 0.0, 0.015), henry=(0.5, 0.25, 0.015), croc=(0.9, -0.3, 0.015))
        j = vl.judge_frame(r, [box('bus', 1.02, 0.01),            # the bus where it is
                               box('croc', 0.8, 0.25),            # croc said where henry is
                               box('thomas', 0.6, -0.15)], 'window')   # nothing there
        self.assertEqual(sorted(j['visible']), ['bus', 'croc', 'henry'])
        kinds = [(b['cls'], b['kind'], b.get('true')) for b in j['boxes']]
        self.assertEqual(kinds, [('bus', 'correct', None), ('croc', 'wrong_class', 'henry'),
                                 ('thomas', 'phantom', None)])
        self.assertEqual(j['found'], ['bus'])                       # henry and croc were missed

    def test_toys_off_the_picture_are_not_counted_as_missed(self):
        r = standing(-0.9, bus=(-0.64, 0.0, 0.015),               # below the window (Mac 3ab looks)
                     henry=(-0.1, 0.5, 0.015),                     # 32 deg left, out of the picture
                     ketchup=(0.95, 0.0, 0.015),                   # beyond the 1.8 m radius
                     croc=(0.0, 0.0, 0.12))                        # lifted (in the beak)
        self.assertEqual(vl.judge_frame(r, [], 'window')['visible'], {})
        parked = standing(-0.3, thomas=(1.1, 0.2, 0.012))          # outside the walls
        self.assertEqual(vl.judge_frame(parked, [], 'window')['visible'], {})

    def test_a_spin_box_is_judged_by_bearing_alone(self):
        r = standing(-0.6, bus=(0.8, 0.0, 0.015))
        far_off = box('bus', 2.4, 0.05)                            # 1 m long: turning ranges poorly
        self.assertEqual(vl.judge_frame(r, [far_off], 'window', spin=True)['boxes'][0]['kind'], 'correct')
        self.assertEqual(vl.judge_frame(r, [far_off], 'window', spin=False)['boxes'][0]['kind'], 'phantom')

    def test_summary_rates_and_confusions(self):
        two = standing(-0.3, bus=(0.7, 0.0, 0.015), henry=(0.5, 0.25, 0.015))
        frames = [{'kind': 'spin', **vl.judge_frame(two, [box('bus', 1.0, 0.0), box('croc', 0.8, 0.25)],
                                                    'window', True)},
                  {'kind': 'look', **vl.judge_frame(standing(-0.3, bus=(0.7, 0.0, 0.015)), [], 'window')}]
        s = vl.summarise(frames)
        self.assertEqual(s['frames'], {'spin': 1, 'look': 1})
        self.assertEqual(s['recall_targets'], {'k': 1, 'n': 3, 'rate': 0.333})   # bus 1/2, henry 0/1
        self.assertEqual(s['precision_all'], {'k': 1, 'n': 2, 'rate': 0.5})
        self.assertEqual(s['wrong_class'], {'henry->croc': 1})
        self.assertEqual(s['wrong_class_within_1_6m'], 1)

    def test_frames_come_from_spins_and_looks(self):
        log = [{'event': 'spin_frame', 'mono': 2.0, 'mono_frame': 1.9, 'seen': [], 'view': 'window'},
               {'event': 'look', 'mono': 5.0, 'view': 'window',
                'frames_seen': [{'mono': 4.5, 'seen': []}, {'seen': []}]},       # an old log: no mono
               {'event': 'b_segment', 'mono': 6.0}]
        self.assertEqual([(f['kind'], f['mono']) for f in vl.frames_of(log)],
                         [('spin', 1.9), ('look', 4.5), ('look', 5.0)])

    def test_board_frames_are_moved_onto_the_mac_clock(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / 'episodes' / 'e_1').mkdir(parents=True)
            (run / 'episodes' / 'e_1' / 'brain.jsonl').write_text(
                json.dumps({'event': 'spin_frame', 'mono': 2.0, 'mono_frame': 1.9, 'seen': [], 'view': 'window'}) + '\n'
                + json.dumps({'event': 'look', 'mono': 5.0, 'view': 'window',
                              'frames_seen': [{'mono': 4.5, 'seen': []}]}) + '\n')
            log = episode.read_brain_log(run, {'label': 'e_1', 'offset': 100.0})   # board clock + offset
            monos = [f['mono'] for f in vl.frames_of(log)]
            self.assertAlmostEqual(monos[0], 101.9)
            self.assertAlmostEqual(monos[1], 104.5)


@unittest.skipIf(trace is None, 'matplotlib not installed')
class TraceTests(unittest.TestCase):
    def test_the_brain_path_follows_odometry_and_its_tray_reanchor(self):
        log = [{'event': 'start', 'mono': 0.0, 'start': [0.5, 0.0, math.pi / 2], 'odom0': [1.0, 1.0, 0.0]},
               {'event': 'tray_look', 'mono': 2.0, 'reanchor_m': [0.1, -0.05]}]
        odom = [(1.0, (2.0, 1.0, 0.0)), (3.0, (2.0, 1.0, 0.0))]      # 1 m forward on odometry = +y
        path = trace.brain_path(log, odom)
        self.assertAlmostEqual(path[0][0], 0.5)
        self.assertAlmostEqual(path[0][1], 1.0)
        self.assertAlmostEqual(path[1][0], 0.6)
        self.assertAlmostEqual(path[1][1], 0.95)

    def test_without_odometry_the_logged_poses_are_the_path(self):
        log = [{'event': 'start', 'mono': 0.0, 'start': [0.0, 0.0, 0.0], 'odom0': [0, 0, 0]},
               {'event': 'state', 'mono': 1.0, 'pose': [0.1, 0.2, 0.0]}]
        self.assertEqual(trace.brain_path(log, []), [(0.1, 0.2)])

    def test_a_run_gets_a_trace_per_episode_and_an_overview(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            rows = []
            for i in range(200):
                r = row(bus=(0.5, 0.2, 0.015))
                r.update(mono=i * 0.02, x=0.002 * i, y=0.0, z=0.18, gz=-1.0, tip_x=0.0, tip_y=0.0)
                rows.append(r)
            trials = []
            for label in ('ab_3000_bus', 'ab_3001_bus'):
                trials.append({'part': 'ab', 'label': label, 'target': 'bus', 'start': 0.0, 'end_mark': 3.9})
                (run / 'episodes' / label).mkdir(parents=True)
                (run / 'episodes' / label / 'brain.jsonl').write_text(json.dumps(
                    {'mono': 0.0, 'event': 'start', 'start': [0, 0, 0], 'odom0': [0, 0, 0]}) + '\n')
            (run / 'trials.jsonl').write_text(''.join(json.dumps(t) + '\n' for t in trials))
            with mock.patch.object(trace.ep, 'falls_in', lambda rows, a, b: []):
                paths = trace.plot_run(run, rows=rows)
            self.assertEqual([p.name for p in paths], ['trace.png', 'trace.png', 'traces.png'])
            self.assertTrue(all(p.stat().st_size > 1000 for p in paths))


@unittest.skipIf(PIL is None, 'Pillow not installed')
class ShotTests(StubbedMotion):
    def test_draw_puts_a_title_band_over_the_upright_frame(self):
        rgb = np.zeros((640, 360, 3), np.uint8)
        img = shots.draw(rgb, [{'cls': 'bus', 'conf': 0.91, 'xyxy': [100, 300, 160, 340]}], 'LOOK', 'bus', 200)
        self.assertEqual(img.size, (360, 640 + shots.TITLE_H))
        self.assertEqual(img.getpixel((100, shots.TITLE_H + 320)), (224, 176, 0))   # the bus box edge

    def brain_with_pictures(self, tmp, picture):
        b = make_brain(FakeRobot(), frames_dir=Path(tmp) / 'frames')
        b.eyes.picture = picture
        return b

    def test_the_brain_saves_numbered_pictures_and_stops_at_the_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = self.brain_with_pictures(tmp, lambda s: (np.zeros((640, 360, 3), np.uint8), None))
            self.assertIs(b.eyes.shoot.__func__, type(b).shot)             # B-segment looks shoot too
            sight = Sight(7, {}, [], 'window', {}, 0.0, 0.0)
            for _ in range(shots.MAX_PER_EPISODE + 3):
                b.shot(sight, 'look_confirmed')
            names = sorted(p.name for p in (Path(tmp) / 'frames').iterdir())
            self.assertEqual(len(names), shots.MAX_PER_EPISODE)
            self.assertEqual(names[0], '00_look_confirmed_s7.jpg')

    def test_a_failed_picture_is_logged_not_raised(self):
        with tempfile.TemporaryDirectory() as tmp:
            def broken(sight):
                raise ValueError('no pixels')
            b = self.brain_with_pictures(tmp, broken)
            logged = []
            b.log = lambda event, **f: logged.append(event)
            b.shot(Sight(1, {}, [], 'window', {}, 0.0, 0.0), 'spin_found')
            self.assertEqual(logged, ['shot_error'])


if __name__ == '__main__':
    unittest.main()
