"""arena_body.py on the Stage 1 field, stepped by hand (no robotd, no camera, no real-time loop)."""
from __future__ import annotations

import json
import math
from pathlib import Path
import threading
import time
import unittest

try:
    import mujoco
    from mjlab_microduck.sim import body_server as sim
except ImportError:  # the test needs the microduck_rl environment (MD_PY)
    sim = None

if sim is not None:
    from microduck_toy.sim import arena_body, bootstrap_body, probe_body
    from microduck_toy.stage1 import metrics

REPO = Path(__file__).resolve().parents[1]
PATCHED = ('build_world', 'Body', 'World', 'Camera')


@unittest.skipIf(sim is None, 'mujoco / mjlab_microduck not importable')
class ArenaBodyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved = {name: getattr(sim, name) for name in PATCHED}
        config = json.loads((REPO / 'configs/control/stage1.json').read_text())
        arena_body.install('grid', config)

    @classmethod
    def tearDownClass(cls):
        for name, value in cls.saved.items():
            setattr(sim, name, value)

    def setUp(self):
        arena_body.ArenaWorld.probe_address = ('127.0.0.1', 0)
        self.world = arena_body.ArenaWorld(sim.DEFAULT_SCENE, 1)
        body = arena_body.ArenaBody(self.world, 0)
        body.place(None, sim.HOME_TRUNK_Z, offset_y=0.0)
        body.support_ended = True
        self.world.bodies.append(body)
        mujoco.mj_forward(self.world.model, self.world.data)
        self.body = body
        self.world.step(4)

    def tearDown(self):
        self.world.server.shutdown()
        self.world.server.server_close()

    def op(self, op_name, /, **params):
        done, reply = threading.Event(), {}
        self.world.ops.put((op_name, params, done, reply))
        self.world.step(4)
        self.assertTrue(done.is_set())
        self.assertNotIn('error', reply)
        return reply

    def mouth(self, fraction):
        target = arena_body.MOUTH_CLOSED + fraction * (arena_body.MOUTH_OPEN - arena_body.MOUTH_CLOSED)
        with self.world.lock:
            self.body.mouth_target = target
        self.world.step(2)

    def toy_at_tip(self, name, dx=0.0, dy=0.0):
        row = self.world.last_row
        self.op('toy', name=name, x=row['tip_x'] + dx, y=row['tip_y'] + dy, yaw=row['yaw'],
                z=row['tip_z'])

    def test_truth_row_has_the_beak_and_every_toy(self):
        row = self.world.last_row
        self.assertEqual(set(row), set(arena_body.COLUMNS))
        for short in ('bus', 'henry', 'ketchup', 'croc', 'thomas'):
            self.assertLess(row[f'{short}_y'], -1.0)          # parked outside the field
        self.assertEqual(row['holding'], '')

    def test_layout_places_toys_and_parks_the_rest(self):
        self.op('layout', robot=[0.2, -0.3, 0.5], toys={'bus': [0.5, 0.4, 1.0], 'croc': [-0.5, 0.2, 0.0]},
                hold_s=1.0)
        row = self.world.last_row
        self.assertAlmostEqual(row['x'], 0.2, places=2)
        self.assertAlmostEqual(row['yaw'], 0.5, places=2)
        self.assertAlmostEqual(row['bus_x'], 0.5, places=3)
        self.assertAlmostEqual(row['bus_y'], 0.4, places=3)
        self.assertLess(row['henry_y'], -1.0)

    def test_close_at_the_tip_welds_then_open_drops(self):
        self.mouth(1.0)
        self.toy_at_tip('henry')
        self.mouth(0.0)
        self.assertEqual(self.world.holding, 'Thomas_Friends_Woodan_Railway_Henry')
        entry = self.world.grasp_log[-1]
        self.assertEqual(entry['result'], 'grasped')
        self.assertLess(entry['nearest_exy'], 0.005)
        self.assertGreater(math.degrees(self.body.mouth_reported), -1.5)   # reads as holding
        self.world.step(4)
        self.assertEqual(self.world.last_row['holding'], 'henry')
        self.mouth(1.0)
        self.assertIsNone(self.world.holding)
        self.assertEqual(self.world.grasp_log[-1]['result'], 'released_opened')

    def test_close_6cm_away_is_out_of_reach_and_reads_empty(self):
        self.mouth(1.0)
        self.toy_at_tip('bus', dy=0.06)
        self.mouth(0.0)
        self.assertIsNone(self.world.holding)
        self.assertEqual(self.world.grasp_log[-1]['result'], 'out_of_reach')
        self.assertAlmostEqual(self.body.mouth_reported, arena_body.MOUTH_CLOSED)

    def test_reset_pose_releases_the_toy(self):
        self.mouth(1.0)
        self.toy_at_tip('ketchup')
        self.mouth(0.0)
        self.assertIsNotNone(self.world.holding)
        self.op('pose', kind='stand')
        self.assertIsNone(self.world.holding)
        self.assertEqual(self.world.grasp_log[-1]['result'], 'released_reset')

    def copy_cost(self, camera) -> float:
        """One render() call from the physics thread, in milliseconds."""
        start = time.perf_counter()
        camera.render(self.world)
        return (time.perf_counter() - start) * 1000.0

    def drawn(self, camera, after: int) -> dict:
        with camera.cond:
            self.assertTrue(camera.cond.wait_for(
                lambda: camera.view is not None and camera.view['seq'] > after, timeout=60.0))
            return camera.view

    def test_the_camera_draws_on_a_worker_and_segments_only_for_a_view(self):
        camera = arena_body.ArenaCamera(self.world.model, f'{self.body.prefix}head_camera')
        camera.fps = 0.0                     # draw only when a "view" op is waiting
        camera.wanted = True                 # a view op: this frame needs the oracle boxes too
        held = self.copy_cost(camera)
        view = self.drawn(camera, -1)
        self.assertIsNotNone(view['boxes'])
        self.assertFalse(camera.wanted)
        draw_ms = camera.draw_stats()['draw_ms_p50']
        # The physics thread only copies the scene; the drawing itself happens on the worker.
        self.assertLess(held, draw_ms / 2.0, f'copy {held:.1f} ms vs draw {draw_ms:.1f} ms')
        camera.fps = 1000.0                  # a stream frame: RGB only, no segmentation
        self.copy_cost(camera)
        self.assertIsNone(self.drawn(camera, view['seq'])['boxes'])

    def test_info_names_the_floor_and_toys(self):
        info = self.world.info()
        self.assertFalse(info['camera'])
        self.assertEqual(info['floor'], 'grid')
        self.assertEqual(sorted(info['toys']), ['bus', 'croc', 'henry', 'ketchup', 'thomas'])


if __name__ == '__main__':
    unittest.main()
