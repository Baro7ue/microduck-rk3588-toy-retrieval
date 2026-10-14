"""Stage 0 scoring on synthetic truth rows (no simulator, no robotd)."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import tempfile
import unittest

from microduck_toy.stage0 import common, metrics
from microduck_toy.stage0.metrics import COLUMNS
from microduck_toy.stage0.s0a_walk import score_pulse, score_stop
from microduck_toy.stage0.s0b_pick import score_pick
from microduck_toy.stage0.s0c_getup import score_getup

REFS = {'stand_trunk_z': 0.12, 'stand_jaw_z': 0.2326}


def row(mono, **over):
    base = {'t': mono, 'mono': mono, 'x': 0.0, 'y': 0.0, 'z': 0.117, 'yaw': 0.0,
            'gx': 0.0, 'gy': 0.0, 'gz': -1.0, 'vx': 0.0, 'vy': 0.0, 'vz': 0.0, 'wz': 0.0,
            'tip_x': 0.08, 'tip_y': 0.0, 'tip_z': 0.21, 'jaw_z': 0.234,
            'cam_x': 0.06, 'cam_y': 0.0, 'cam_z': 0.23, 'cam_pitch': 0.0,
            'neck_q': 0.35, 'head_q': 0.35, 'lfoot': 1, 'rfoot': 1, 'other_floor': 0,
            'torque': 1, 'kp': 200.0, 'released': 1, 'mark': ''}
    base.update(over)
    return base


def series(start, end, **over):
    n = int(round((end - start) / 0.02))
    return [row(round(start + 0.02 * i, 6), **over) for i in range(n)]


class StandingTests(unittest.TestCase):
    def test_lab_gate(self):
        self.assertTrue(metrics.standing(row(0), REFS, full=True))
        self.assertFalse(metrics.standing(row(0, lfoot=0), REFS))
        self.assertFalse(metrics.standing(row(0, other_floor=1), REFS))
        self.assertFalse(metrics.standing(row(0, gz=-0.8), REFS))
        self.assertFalse(metrics.standing(row(0, z=0.07), REFS))   # below 0.12 - 0.045
        crouch = row(0, jaw_z=0.15)                                 # head-down half crouch
        self.assertTrue(metrics.standing(crouch, REFS))
        self.assertFalse(metrics.standing(crouch, REFS, full=True))

    def test_fall_needs_two_tenths_of_a_second(self):
        rows = series(0, 1) + series(1, 1.1, gz=0.0) + series(1.1, 2) + series(2, 3, gz=0.1)
        self.assertEqual(metrics.fall_time(rows, 0), 2.0)
        self.assertIsNone(metrics.fall_time(rows, 0, 1.9))
        low = series(0, 0.5, z=0.05)
        self.assertEqual(metrics.fall_time(low, 0), 0.0)

    def test_holds_requires_coverage(self):
        rows = series(0, 2)
        self.assertTrue(metrics.holds(rows, lambda r: True, 0.5, 1.0))
        self.assertFalse(metrics.holds(rows, lambda r: True, 1.5, 1.0))     # runs past the data
        gap = series(0, 1) + series(1.5, 3)
        self.assertFalse(metrics.holds(gap, lambda r: True, 0.5, 1.5))


class GeometryTests(unittest.TestCase):
    def test_relative_and_point_in_use_the_origin_heading(self):
        origin = row(0, x=1.0, y=1.0, yaw=math.pi / 2)
        moved = row(1, x=1.0, y=1.3, yaw=math.pi / 2 + 0.1)
        rel = metrics.relative(origin, moved)
        self.assertAlmostEqual(rel['forward'], 0.3)
        self.assertAlmostEqual(rel['left'], 0.0)
        self.assertAlmostEqual(rel['dyaw'], 0.1)
        forward, left = metrics.point_in(origin, 0.9, 1.0)
        self.assertAlmostEqual(forward, 0.0)
        self.assertAlmostEqual(left, 0.1)

    def test_unwrapped_yaw_crosses_pi(self):
        rows = [row(i * 0.02, yaw=metrics.wrap(3.0 + 0.1 * i)) for i in range(10)]
        self.assertAlmostEqual(metrics.unwrapped_yaw(rows)[-1], 0.9)

    def test_settle_time_returns_when_stillness_began(self):
        moving = [row(round(i * 0.02, 6), x=0.01 * i) for i in range(50)]      # 1 s at 0.5 m/s
        still = series(1.0, 2.5, x=0.49)
        settled = metrics.settle_time(moving + still, 0.0)
        self.assertAlmostEqual(settled, 0.98, places=6)   # last moving row is already at x=0.49
        self.assertIsNone(metrics.settle_time(moving, 0.0))

    def test_free_until_stops_before_the_probe_holds_again(self):
        rows = series(0, 1) + series(1, 2, released=0)
        self.assertAlmostEqual(metrics.free_until(rows, 0.0, 5.0), 0.98)
        self.assertEqual(metrics.free_until(rows, 0.0, 0.5), 0.5)


class FileTests(unittest.TestCase):
    def test_truth_and_states_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'truth.csv'
            with path.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=COLUMNS)
                writer.writeheader()
                writer.writerow(row(0.0, mark='a|b'))
                writer.writerow(row(0.02))
            rows = metrics.read_truth(path)
            self.assertEqual(rows[0]['lfoot'], 1)
            self.assertIsInstance(rows[0]['lfoot'], int)
            self.assertEqual(metrics.mark_times(rows), {'a': 0.0, 'b': 0.0})
            states = Path(tmp) / 'state.jsonl'
            states.write_text(''.join(json.dumps(
                {'mono': i * 0.02, 't': i * 0.02, 'policy': p,
                 'safety': {'fallen': False, 'limp': False}}) + '\n'
                for i, p in enumerate(['stand', 'stand', 'ground_pick', 'ground_pick', 'stand'])))
            spans = metrics.label_spans(metrics.read_states(states))
            self.assertEqual([s['policy'] for s in spans], ['stand', 'ground_pick', 'stand'])
            self.assertEqual((spans[1]['start'], spans[1]['end']), (0.04, 0.06))

    def test_rescore_falls_back_to_the_trials_kept_in_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta = {'refs': REFS, 'quick': False}
            trials = [{'part': 'a3t', 'start': 1.0, 'released': 1.5}]
            (Path(tmp) / 'summary.json').write_text(json.dumps({'meta': meta, 'trials': trials}))
            (Path(tmp) / 'trials.jsonl').write_text('')
            self.assertEqual(common.Rescored(tmp).trials, trials)
            (Path(tmp) / 'summary.json').write_text(json.dumps({'meta': meta}))
            with self.assertRaises(RuntimeError):
                common.Rescored(tmp)

    def test_continuity_counts_missing_batches(self):
        rows = series(0, 1) + series(1.06, 2)
        self.assertEqual(metrics.continuity(rows)['missing_batches'], 3)   # 1.00, 1.02, 1.04
        self.assertEqual(metrics.continuity(series(0, 1))['missing_batches'], 0)

    def test_clean_replaces_non_finite(self):
        self.assertEqual(metrics.clean({'a': [math.nan, 1.0], 'b': math.inf}),
                         {'a': [None, 1.0], 'b': None})


class ScoringTests(unittest.TestCase):
    def test_getup_success_and_failures(self):
        down = series(0, 2, gz=0.1, z=0.07, lfoot=0, rfoot=0, other_floor=2)
        up = series(2, 9)
        walk = [row(round(9 + 0.02 * i, 6), x=0.004 * i) for i in range(150)]
        rows = down + up + walk + series(12, 13.5, x=0.6)
        record = {'walk_start': 9.0, 'walk_end': 12.0}
        result = score_getup(rows, REFS, record, 0.0)
        self.assertTrue(result['success'], result)
        self.assertAlmostEqual(result['getup_s'], 2.0)
        self.assertFalse(score_getup(down + series(2, 13.5, z=0.07), REFS, record, 0.0)['success'])
        late = series(0, 11, gz=0.1) + series(11, 25)
        self.assertEqual(score_getup(late, REFS, record, 0.0)['why'], 'no full stand within 10 s')

    def test_pick_finds_the_lowest_beak_and_the_handback(self):
        rows = series(0, 1)
        for i in range(140):                       # 2.8 s pick, beak lowest at 1.5 s into it
            t = round(1 + 0.02 * i, 6)
            depth = max(0.0, 1 - abs(t - 2.5) / 1.5)
            rows.append(row(t, tip_z=0.21 - 0.19 * depth, tip_x=0.08 + 0.02 * depth,
                            tip_y=0.01 * depth))
        rows += series(3.8, 9)
        spans = [{'policy': 'stand', 'start': 0.0, 'end': 0.98},
                 {'policy': 'ground_pick', 'start': 1.0, 'end': 3.78},
                 {'policy': 'stand', 'start': 3.8, 'end': 9.0}]
        record = {'head': 'H0', 'rep': 1, 'trigger': 0.98, 'handback_seen': 3.82}
        result = score_pick(rows, spans, REFS, record)
        self.assertTrue(result['ran'])
        self.assertAlmostEqual(result['tip_min_z'], 0.02)
        self.assertAlmostEqual(result['tip_min_phase'], 1.5 / 4.0)
        self.assertAlmostEqual(result['handback_phase'], 0.7)
        self.assertAlmostEqual(result['tip_forward_m'], 0.10)
        self.assertAlmostEqual(result['tip_left_m'], 0.01)
        self.assertAlmostEqual(result['stable_after_handback_s'], 0.0)
        self.assertFalse(result['fell'])

    # Turn pulse at yaw -1.5 rad: 0.5 s at -0.4 rad/s, then still creeping when the next block's
    # reset puts the robot back at yaw 0 (board a3t -0.6 x 0.5 s rep 10, 2026-10-14).
    def creeping_turn(self, cut):
        rows = series(0, 1, yaw=-1.5)
        rows += [row(round(1 + 0.02 * i, 6), yaw=-1.5 - 0.4 * 0.02 * i) for i in range(25)]
        rows += [row(round(1.5 + 0.02 * i, 6), yaw=-1.7 - 0.05 * 0.02 * i)
                 for i in range(int(round((cut - 1.5) / 0.02)))]
        return rows

    def test_pulse_ignores_the_next_reset(self):
        rows = self.creeping_turn(3.0) + series(3.0, 4.0, released=0) + series(4.0, 6.0)
        record = {'head': 'H2', 'wz': -0.6, 'seconds': 0.5, 'rep': 10,
                  'start': 1.0, 'released': 1.5}
        result = score_pulse(rows, record)
        self.assertAlmostEqual(result['yaw_deg'], math.degrees(-0.2 - 0.05 * 1.48), places=6)
        self.assertIsNone(result['still_after_zero_s'])
        self.assertAlmostEqual(result['window_s'], 1.48)

    def test_pulse_stops_at_the_next_trial(self):
        rows = self.creeping_turn(3.0)
        rows += [row(round(3 + 0.02 * i, 6), yaw=-1.775 - 0.01 * i) for i in range(25)]
        rows += series(3.5, 6.0, yaw=-2.025)
        record = {'head': 'H2', 'wz': -0.6, 'seconds': 0.5, 'rep': 9,
                  'start': 1.0, 'released': 1.5}
        unbounded = score_pulse(rows, record)['yaw_deg']
        self.assertAlmostEqual(unbounded, math.degrees(-0.525), places=6)
        bounded = score_pulse(rows, record, limit=3.0)
        self.assertAlmostEqual(bounded['yaw_deg'], math.degrees(-0.275), places=6)
        self.assertAlmostEqual(bounded['window_s'], 1.5)

    def test_pulse_that_settles_reads_the_still_pose(self):
        rows = self.creeping_turn(2.0) + series(2.0, 6.0, yaw=-1.725)
        record = {'head': 'H2', 'wz': -0.6, 'seconds': 0.5, 'rep': 1,
                  'start': 1.0, 'released': 1.5}
        result = score_pulse(rows, record, limit=5.0)
        # still from 1.84 s: what is left of the creep (-1.717 -> -1.725) is under 0.5 deg
        self.assertAlmostEqual(result['yaw_deg'], math.degrees(-0.217), places=6)
        self.assertAlmostEqual(result['still_after_zero_s'], 0.34)
        self.assertAlmostEqual(result['window_s'], 3.0)

    def test_stop_ignores_the_next_reset(self):
        rows = series(0, 1)
        rows += [row(round(1 + 0.02 * i, 6), x=0.002 * i, vx=0.1) for i in range(150)]
        rows += series(4.0, 5.0, released=0) + series(5.0, 9.0)
        record = {'head': 'H0', 'rep': 1, 'start': 0.0, 'released': 1.0}
        result = score_stop(rows, record)
        self.assertAlmostEqual(result['coast_m'], 0.002 * 149, places=6)
        self.assertIsNone(result['still_after_zero_s'])


if __name__ == '__main__':
    unittest.main()
