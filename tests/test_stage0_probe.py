"""probe_body.py on the real scene, stepped by hand (no robotd, no real-time loop, no viewer)."""
from __future__ import annotations

import json
import math
from pathlib import Path
import socket
import tempfile
import threading
import unittest

try:
    import mujoco
    from mjlab_microduck.sim import body_server as sim
except ImportError:  # the test needs the microduck_rl environment (MD_PY)
    sim = None

if sim is not None:
    from microduck_toy.sim import bootstrap_body, probe_body
from microduck_toy.stage0 import metrics


@unittest.skipIf(sim is None, 'mujoco / mjlab_microduck not importable')
class ProbeBodyTests(unittest.TestCase):
    def setUp(self):
        probe_body.ProbeWorld.probe_address = ('127.0.0.1', 0)
        self.world = probe_body.ProbeWorld(sim.DEFAULT_SCENE, 1)
        body = bootstrap_body.StartupBody(self.world, 0)
        body.place(None, sim.HOME_TRUNK_Z, offset_y=0.0)
        body.support_ended = True      # as after robotd's first policy tick
        self.world.bodies.append(body)
        mujoco.mj_forward(self.world.model, self.world.data)
        self.body = body
        self.world.step(4)

    def tearDown(self):
        self.world.server.shutdown()
        self.world.server.server_close()

    def op(self, name, **params):
        done, reply = threading.Event(), {}
        self.world.ops.put((name, params, done, reply))
        self.world.step(4)
        self.assertTrue(done.is_set())
        self.assertNotIn('error', reply)
        return reply

    def ask(self, request):
        with socket.create_connection(self.world.server.server_address[:2], timeout=5) as client:
            stream = client.makefile('rwb')
            stream.write((json.dumps(request) + '\n').encode())
            stream.flush()
            return json.loads(stream.readline())

    def test_truth_row_has_every_column_and_the_stand_references(self):
        row = self.world.last_row
        self.assertEqual(set(row), set(metrics.COLUMNS))
        self.assertAlmostEqual(row['gz'], -1.0, places=3)
        self.assertGreater(row['tip_x'], 0.04)         # mouth tip ahead of the trunk
        refs = self.ask({'op': 'info'})['refs']
        self.assertEqual(refs['source'], 'STAND keyframe')
        self.assertAlmostEqual(refs['stand_trunk_z'], 0.12, places=3)

    def test_lying_poses_rest_just_above_the_floor(self):
        for kind, gx_sign in (('back', -1), ('front', 1)):
            reply = self.op('pose', kind=kind, tilt_deg=95)
            with self.world.lock:
                lowest = self.world._lowest_point()
            row = self.world.last_row
            self.assertAlmostEqual(row['gz'], -math.cos(math.radians(95)), places=2, msg=kind)
            self.assertEqual(math.copysign(1, row['gx']), gx_sign, kind)
            self.assertLess(lowest, 0.01, kind)
            self.assertGreater(reply['trunk_z'], 0.0)
        self.op('pose', kind='side', tilt_deg=95)
        self.assertGreater(abs(self.world.last_row['gy']), 0.9)

    def test_stand_pose_holds_then_releases(self):
        self.op('pose', kind='stand', x=0.3, y=-0.2, yaw=1.0, hold_s=0.1)
        row = self.world.last_row
        self.assertEqual(row['released'], 0)
        self.assertAlmostEqual(row['x'], 0.3, places=3)
        self.assertAlmostEqual(row['yaw'], 1.0, places=3)
        for _ in range(10):
            self.world.step(4)
        self.assertEqual(self.world.last_row['released'], 1)

    def test_push_adds_trunk_velocity_in_the_heading_frame(self):
        self.op('pose', kind='stand', yaw=math.pi / 2, hold_s=0.0)
        self.world.step(4)
        before = self.world.data.qvel[self.body.trunk_dof: self.body.trunk_dof + 2].copy()
        # Applied between batches without stepping, so no physics runs in between.
        self.world.ops.put(('push', {'vx': 0.8, 'vy': 0.0}, threading.Event(), {}))
        self.world._apply_ops()
        after = self.world.data.qvel[self.body.trunk_dof: self.body.trunk_dof + 2]
        self.assertAlmostEqual(after[0] - before[0], 0.0, places=6)
        self.assertAlmostEqual(after[1] - before[1], 0.8, places=6)   # forward is +y at yaw 90°

    def test_record_and_mark_over_the_test_port(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'truth.csv'
            self.assertEqual(self.ask({'op': 'record', 'path': str(path)})['recording'], str(path))
            self.world.step(4)
            self.assertEqual(self.ask({'op': 'mark', 'label': 'hello'})['marked'], 'hello')
            for _ in range(5):
                self.world.step(4)
            self.ask({'op': 'record', 'path': None})
            rows = metrics.read_truth(path)
            self.assertEqual(len(rows), 6)
            self.assertEqual([r['mark'] for r in rows].count('hello'), 1)
            meta = json.loads(path.with_suffix('.meta.json').read_text())
            self.assertEqual(meta['columns'], list(metrics.COLUMNS))
            self.assertIn('error', self.ask({'op': 'bogus'}))


if __name__ == '__main__':
    unittest.main()
