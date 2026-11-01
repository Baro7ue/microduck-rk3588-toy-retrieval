"""Stage 5 room demo without a robot: the config merge, the toy layout, the toy copies in the scene,
the brain's config override, the board command, the launcher (with the fakes of test_launchers.py),
and fix2: the round tray, its tray look, the drop sides and the judge (knocked out, drop_missed)."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
from pathlib import Path
import socket
import unittest

import numpy as np

from microduck_toy.control import ranging
from microduck_toy.stage3 import brain as brain_lib, episode
from microduck_toy.stage5 import room

try:
    import mujoco
    from microduck_toy.scene import arena
    from microduck_toy.scene import toys as toy_lib
except ImportError:          # the board's Python has no MuJoCo
    mujoco = None

import test_launchers
from test_launchers import ZSH, put

REPO = Path(__file__).resolve().parents[1]
RL = REPO / 'third_party' / 'microduck_rl'


def as_f32(rgba):
    return tuple(np.float32(v) for v in rgba)


class RoomConfigTests(unittest.TestCase):
    def tearDown(self):
        if mujoco is not None:
            toy_lib.register_copies(None)     # room.layout registers the copies; others see five toys

    def test_deep_merge_merges_dicts_and_replaces_the_rest(self):
        base = {'a': {'x': 1, 'y': [1, 2]}, 'b': 2}
        out = room.deep_merge(base, {'a': {'y': [3]}, 'c': 4})
        self.assertEqual(out, {'a': {'x': 1, 'y': [3]}, 'b': 2, 'c': 4})
        self.assertEqual(base, {'a': {'x': 1, 'y': [1, 2]}, 'b': 2})

    def test_the_room_changes_only_what_it_names(self):
        doc = room.load()
        s1 = room.merged(doc, 'stage1')
        frozen = json.loads((REPO / 'configs/control/stage1.json').read_text())
        self.assertEqual(s1['arena']['half_size_m'], 1.5)
        self.assertEqual(s1['arena']['tray']['centre_m'], [1.15, 0.0])
        self.assertEqual(s1['arena']['tray']['rim_height_m'], frozen['arena']['tray']['rim_height_m'])
        self.assertEqual(s1['camera'], frozen['camera'])
        self.assertFalse(s1['arena']['cast_shadows'])

    def test_layout_is_one_per_seed_and_keeps_its_gaps(self):
        doc = room.load()
        d = doc['demo']
        a, b = room.layout(doc, 5000), room.layout(doc, 5000)
        self.assertEqual(a, b)
        self.assertNotEqual(a['toys'], room.layout(doc, 5001)['toys'])
        toys = a['toys']
        self.assertEqual(len(toys), 12)
        self.assertEqual(sorted(t for t in toys if t.startswith('bus')), ['bus', 'bus_2'])
        self.assertEqual(sorted(t for t in toys if t.startswith('croc')), ['croc', 'croc_2', 'croc_3'])
        tray = doc['stage1']['arena']['tray']['centre_m']
        pts = [v[:2] for v in toys.values()]
        for p in pts:
            self.assertLessEqual(max(abs(p[0]), abs(p[1])), d['toy_field_m'])
            self.assertGreaterEqual(math.dist(p, tray), d['tray_clear_m'])
            self.assertGreaterEqual(math.dist(p, d['robot'][:2]), d['start_clear_m'])
        for i, p in enumerate(pts):
            for q in pts[i + 1:]:
                self.assertGreaterEqual(math.dist(p, q), d['toy_gap_m'])

    def test_furniture_stays_between_the_field_and_the_walls(self):
        arena_cfg = room.merged(room.load(), 'stage1')['arena']
        field, walls = arena_cfg['half_size_m'], arena_cfg['room_half_m']
        for piece in arena_cfg['furniture']:
            for box in piece['boxes']:
                (x, y, _), (hx, hy, _) = box['pos'], box['size']
                inner = max(abs(x) - hx, abs(y) - hy)        # nearest edge, by the farther axis
                self.assertGreaterEqual(inner, field, piece['name'])
                self.assertLessEqual(max(abs(x) + hx, abs(y) + hy), walls + 1e-9, piece['name'])
                # A dark grey box on the floor reads as the tray (cloud check 2026-10-31: the first
                # TV stand, 41 of 216 frames); the black TV screen hangs 0.53 m up and was never boxed.
                rgb = (box.get('rgba') or piece['rgba'])[:3]
                on_floor = box['pos'][2] - box['size'][2] < 0.05
                self.assertFalse(on_floor and max(rgb) - min(rgb) < 0.08 and max(rgb) < 0.45,
                                 f'{piece["name"]}: dark grey reads as the tray')


class BrainOverrideTests(unittest.TestCase):
    def test_frozen_configs_without_an_override(self):
        configs = brain_lib.load_configs()
        self.assertEqual(configs['stage1']['arena']['half_size_m'], 1.0)
        self.assertNotIn('copies', configs['stage1']['arena'])

    def test_override_merges_stage1_and_stage3(self):
        configs = brain_lib.load_configs(override='configs/demo/room.json')
        self.assertEqual(configs['stage1']['arena']['half_size_m'], 1.5)
        self.assertEqual(configs['stage3']['search']['max_spins'], 3)
        self.assertEqual(len(configs['stage3']['search']['relocate_spots']), 8)
        self.assertEqual(configs['stage3']['episode']['max_s'], 240.0)
        self.assertEqual(configs['stage3']['search']['spin_deg'],
                         brain_lib.load_configs()['stage3']['search']['spin_deg'])

    def test_parser_takes_the_override(self):
        args = brain_lib.parser().parse_args(['--target', 'bus', '--start=-0.9,0,0', '--socket', 's',
                                              '--out', 'o', '--config-override', 'configs/demo/room.json'])
        self.assertEqual(args.config_override, 'configs/demo/room.json')

    def test_demo_options_do_not_swallow_the_runners(self):
        from microduck_toy.stage5 import demo
        own, args = demo.parse(['--target', 'board', '--source', 'stream', '--board-bundle', '/b',
                                '--board-out', '/o', '--seed', '5001'])
        self.assertIsNone(own.targets)
        self.assertEqual((own.seed, args.target, args.board_bundle), (5001, 'board', '/b'))
        own, args = demo.parse(['--targets', 'henry,bus'])
        self.assertEqual((own.targets, args.target), ('henry,bus', 'mac'))

    def test_board_command_passes_the_override_only_for_the_demo(self):
        args = episode.parser().parse_args(['3e', '--target', 'board', '--board-bundle', '/b',
                                            '--board-out', '/o'])
        args.backend = 'int8'
        plain = episode.board_command(args, '/o/e', 'bus', (0.0, 0.0, 0.0), 'delivered')
        self.assertNotIn('--config-override', plain)
        args.config_override = 'configs/demo/room.json'
        demo = episode.board_command(args, '/o/e', 'bus', (-0.9, 0.0, 0.0), 'delivered')
        self.assertTrue(demo.endswith('--config-override configs/demo/room.json'))
        self.assertIn('--start=-0.90000,0.00000,0.00000', demo)
        self.assertNotIn('--side', demo)
        args.side = 'y+'
        sided = episode.board_command(args, '/o/e', 'bus', (-0.9, 0.0, 0.0), 'delivered')
        self.assertTrue(sided.endswith('--config-override configs/demo/room.json --side=y+'))

    def test_brain_takes_a_drop_side(self):
        base = ['--target', 'bus', '--start=0,0,0', '--socket', 's', '--out', 'o']
        self.assertIsNone(brain_lib.parser().parse_args(base).side)
        self.assertEqual(brain_lib.parser().parse_args(base + ['--side=y-']).side, 'y-')
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            brain_lib.parser().parse_args(base + ['--side', 'left'])

    def test_room_drop_sides_and_round_tray(self):
        doc = room.load()
        sides = doc['demo']['drop_sides']
        self.assertEqual(len(set(sides)), len(sides))
        self.assertTrue(set(sides) <= set(room.merged(doc, 'stage3')['carry']['sides']))
        tray = room.merged(doc, 'stage1')['arena']['tray']
        self.assertEqual((tray['shape'], tray['size_m'], tray['rim_height_m']), ('round', 0.3, 0.01))
        self.assertNotIn('shape', brain_lib.load_configs()['stage1']['arena']['tray'])


class RoundTrayJudgeTests(unittest.TestCase):
    """In the round tray (truth), what a run knocked out, and drop_missed."""
    S3 = room.merged(room.load(), 'stage3')
    TRAY = room.merged(room.load(), 'stage1')['arena']['tray']
    C = (1.15, 0.0)

    def row(self, **toys):
        r = {}
        for name in ('bus', 'henry', 'ketchup'):
            x, y, z = toys.get(name, (0.0, 0.0, 0.01))
            r.update({f'{name}_x': x, f'{name}_y': y, f'{name}_z': z})
        return r

    def test_inside_the_rim_and_low_is_in(self):
        inn = lambda **kw: room.in_tray(self.row(**kw), 'bus', self.C, self.TRAY, self.S3)  # noqa: E731
        r = self.TRAY['size_m'] / 2                               # 0.15; the rim's inner face 0.144
        self.assertTrue(inn(bus=(1.15 + r - 0.03, 0.0, 0.012)))
        self.assertTrue(inn(bus=(1.15 + 0.6 * r, 0.6 * r, 0.012)))    # 0.85 r from the centre
        self.assertFalse(inn(bus=(1.15 + 0.13, 0.13, 0.012)))     # 0.184: past the rim (a square's corner)
        self.assertFalse(inn(bus=(1.15 - r + 0.002, 0.0, 0.03)))  # lying across the rim
        self.assertFalse(inn(bus=(1.15, 0.0, 0.15)))              # still in the beak above it
        square = self.TRAY | {'shape': 'square'}
        corner = self.row(bus=(1.15 + 0.13, 0.13, 0.012))
        self.assertTrue(room.in_tray(corner, 'bus', self.C, square, self.S3))

    TOYS = ['bus', 'henry', 'ketchup']

    def judge(self, row, target, end='dropped'):
        from microduck_toy.stage5 import demo
        return demo.judge(self.rec(row, target, end), [], self.TOYS, {'bus'}, self.S3, self.C, self.TRAY)

    def rec(self, row, target, end='dropped'):
        return {'judge_row': row, 'target': target, 'start': 0.0, 'end_mark': 1.0, 'side': 'y+',
                'start_told': [0.0, 0.0, 0.0], 'start_truth': [0.0, 0.0, 0.0],
                'brain': {'end': end, 'pose': [0.9, 0.0, 0.0], 'elapsed_s': 50.0, 'closes': 1}}

    def test_a_drop_that_knocks_an_earlier_toy_out(self):
        row = self.row(bus=(0.95, 0.0, 0.012), henry=(1.15, 0.12, 0.012))
        j = self.judge(row, 'henry')
        self.assertEqual((j['outcome'], j['new_in_tray'], j['knocked_out'], j['in_tray']),
                         ('delivered', ['henry'], ['bus'], ['henry']))
        self.assertEqual(j['side'], 'y+')

    def test_dropped_but_nothing_new_is_drop_missed(self):
        row = self.row(bus=(1.03, 0.0, 0.012), ketchup=(0.97, 0.05, 0.012))
        j = self.judge(row, 'ketchup')
        self.assertEqual((j['outcome'], j['new_in_tray'], j['knocked_out']), ('drop_missed', [], []))
        self.assertEqual(self.judge(row, 'ketchup', end='timeout')['outcome'], 'timeout')


class RoundTrayLookTests(unittest.TestCase):
    """tray.measure_tray shape 'round' on a synthetic round-tray box (the silhouette of its top and
    bottom circles, projected like test_stage3_tray.tray_box)."""
    R, H = 0.15, 0.05

    def box(self, state, centre):
        from test_stage3_tray import CAM
        pts = []
        for k in range(720):
            a = math.radians(k / 2)
            for z in (0.0, self.H):
                rim = (centre[0] + self.R * math.cos(a), centre[1] + self.R * math.sin(a), z)
                p = ranging.project(CAM, state, rim)
                if p is not None:
                    pts.append(p)
        clip = lambda v, hi: max(0.0, min(hi, v))  # noqa: E731
        cols, rows = [p[0] for p in pts], [p[1] for p in pts]
        return {'cls': 'tray', 'conf': 0.9, 'xyxy': [clip(min(cols), CAM.width), clip(min(rows), CAM.height),
                                                     clip(max(cols), CAM.width), clip(max(rows), CAM.height)]}

    def test_round_rule_finds_the_centre_where_the_square_rule_is_off(self):
        from test_stage3_tray import CAM, h1_like
        from microduck_toy.stage3 import tray
        cfg = json.loads((REPO / 'configs/control/stage3.json').read_text())['tray_look']
        st = h1_like()
        worst_square = 0.0
        for left in (-0.24, -0.2, 0.17, 0.2):
            centre = (0.45, left)
            b = self.box(st, centre)
            self.assertTrue(b['xyxy'][0] > 3 or b['xyxy'][2] < CAM.width - 3)    # a side is in view
            m = tray.measure_tray([(st, [b])], CAM, cfg, self.R, self.H, (0.45, left - 0.03), shape='round')
            self.assertEqual(m['source'], 'yolo')
            self.assertAlmostEqual(m['centre'][0], 0.45, delta=0.005)
            self.assertAlmostEqual(m['centre'][1], left, delta=0.005)
            sq = tray.measure_tray([(st, [b])], CAM, cfg, self.R, self.H, (0.45, left - 0.03))
            worst_square = max(worst_square, abs(sq['centre'][1] - left))
        self.assertGreater(worst_square, 0.01)


@unittest.skipIf(mujoco is None or not (RL / 'src/mjlab_microduck/robot/microduck/scene.xml').is_file(),
                 'mujoco or microduck_rl not available')
class RoomSceneTests(unittest.TestCase):
    def tearDown(self):
        toy_lib.register_copies(None)        # other tests see the five toys again

    def test_copies_are_bodies_of_their_own(self):
        cfg = room.merged(room.load(), 'stage1')['arena']
        model = arena.build_spec(cfg, RL, 'wood').compile()
        ids = arena.ArenaIds(model)
        self.assertEqual(len(ids.toys), 12)
        self.assertEqual(toy_lib.SHORT['SCHOOL_BUS@2'], 'bus_2')
        self.assertEqual(toy_lib.kind_of('croc_3'), 'croc')
        self.assertEqual(toy_lib.resolve('thomas_3'), toy_lib.LONG['thomas'] + '@3')
        parks = [t['park'] for t in ids.toys.values()]
        self.assertEqual(len(set(parks)), 12)
        toy_lib.register_copies(None)
        self.assertNotIn('bus_2', toy_lib.LONG)
        self.assertEqual(len(toy_lib.scene_toys()), 5)
        self.assertTrue(all(abs(y) > cfg['room_half_m'] for _, y in parks))
        self.assertTrue(all(model.light_castshadow[i] == 0 for i in range(model.nlight)))
        self.assertGreater(sum(1 for g in range(model.ngeom) if model.geom(g).name.startswith('furn_')), 10)

    def test_the_room_tray_is_round(self):
        cfg = room.merged(room.load(), 'stage1')['arena']
        tray = cfg['tray']
        model = arena.build_spec(cfg, RL, 'wood').compile()
        ids = arena.ArenaIds(model)
        names = [model.geom(g).name for g in ids.tray_geoms]
        walls = [g for g in ids.tray_geoms if model.geom(g).name.startswith('tray_wall_')]
        self.assertEqual(len(walls), tray['segments'])
        self.assertIn('tray_base', names)
        base = model.geom('tray_base')
        self.assertEqual(base.type[0], mujoco.mjtGeom.mjGEOM_CYLINDER)
        self.assertAlmostEqual(base.size[0], tray['size_m'] / 2)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        r, c = tray['size_m'] / 2, ids.tray_centre
        for g in walls:
            self.assertAlmostEqual(model.geom_size[g][2] * 2, tray['rim_height_m'])
            centre = data.geom_xpos[g][:2] - c
            normal = centre / np.linalg.norm(centre)
            outward = data.geom_xmat[g].reshape(3, 3)[:2, 0]               # the box's thin axis
            self.assertAlmostEqual(abs(float(normal @ outward)), 1.0, places=6)
            self.assertAlmostEqual(float(np.linalg.norm(centre)) + model.geom_size[g][0], r, places=6)
            # the outer face's half-length reaches the next box: no gap round the circle
            self.assertGreaterEqual(model.geom_size[g][1], r * math.tan(math.pi / len(walls)) - 1e-9)
        self.assertTrue(all(tuple(model.geom_rgba[g]) == as_f32(arena.TRAY_RGBA) for g in ids.tray_geoms))

    def test_the_stage1_arena_is_unchanged(self):
        cfg = json.loads((REPO / 'configs/control/stage1.json').read_text())['arena']
        toy_lib.register_copies(None)
        model = arena.build_spec(cfg, RL, 'grid').compile()
        self.assertEqual(len(arena.ArenaIds(model).toys), 5)
        self.assertTrue(any(model.light_castshadow))
        self.assertEqual(model.geom('wall_px').size[1], cfg['half_size_m'] + cfg['wall_thickness_m'])
        self.assertEqual(sorted(model.geom(g).name for g in arena.ArenaIds(model).tray_geoms),
                         ['tray_base', 'tray_rim_nx', 'tray_rim_ny', 'tray_rim_px', 'tray_rim_py'])


@unittest.skipUnless(ZSH, 'zsh not installed')
class RoomDemoLauncherTests(unittest.TestCase):
    setUp_base = test_launchers.LauncherTests.setUp
    tearDown = test_launchers.LauncherTests.tearDown
    run_script = test_launchers.LauncherTests.run_script
    calls = test_launchers.LauncherTests.calls
    runs = test_launchers.LauncherTests.runs

    def setUp(self):
        self.setUp_base()
        for name in ('yolo11n_toys_split_320.onnx', 'yolo11n_toys_fp16.rknn', 'yolo11n_toys_int8.rknn'):
            put(self.root / 'models/yolo/yolo11n_toys' / name, name)

    def unix_socket(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(str(path))
        self.addCleanup(sock.close)

    def modules(self):
        return [(args[args.index('-m') + 1], args[args.index('-m') + 2:]) for args, _ in self.calls('python')]

    def test_mac_runs_the_demo_module(self):
        self.unix_socket(Path(self.env['MD_MAC_STATE']) / 'robotd.sock')
        result = self.run_script('scripts/stage5/run_room_demo.zsh', '--seed', '5001', FAKE_LSOF_STATUS='0')
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        (module, args), = self.modules()
        self.assertEqual(module, 'microduck_toy.stage5.demo')
        self.assertEqual(args, ['--source', 'stream', '--seed', '5001'])

    def test_board_sends_a_bundle_with_the_room_config(self):
        self.unix_socket(self.board / 'run/robotd.sock')
        result = self.run_script('scripts/stage5/run_room_demo.zsh', '--board', FAKE_LSOF_STATUS='0')
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        run = next(p for p in self.runs() if p.name.endswith('_s5_board_bundle'))
        bundle = self.board / 'bundles' / run.name
        for rel in ('microduck_toy/stage5/room.py', 'microduck_toy/stage3/brain.py',
                    'configs/demo/room.json', 'configs/control/stage3.json',
                    'models/yolo/yolo11n_toys/yolo11n_toys_int8.rknn', 'SHA256SUMS'):
            self.assertTrue((bundle / rel).is_file(), rel)
        self.assertIn('configs/demo/room.json', (bundle / 'SHA256SUMS').read_text())
        (module, args), = self.modules()
        self.assertEqual(module, 'microduck_toy.stage5.demo')
        self.assertEqual(args[:4], ['--target', 'board', '--source', 'stream'])
        self.assertEqual(args[args.index('--board-bundle') + 1], str(bundle))

    def test_stops_without_the_body_or_robotd(self):
        result = self.run_script('scripts/stage5/run_room_demo.zsh', FAKE_LSOF_STATUS='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('--room', result.stderr + result.stdout)
        self.assertEqual(self.modules(), [])
        result = self.run_script('scripts/stage5/run_room_demo.zsh', FAKE_LSOF_STATUS='0')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('run_robotd.zsh s2', result.stderr + result.stdout)

    def test_arena_body_room_mode(self):
        result = self.run_script('scripts/sim/stage1/run_arena_body.zsh', '--room', '--camera', '--board')
        self.assertEqual(result.returncode, 0, result.stderr)
        args, _ = self.calls('mjpython')[-1]
        root = str(self.root.resolve())
        self.assertEqual(args[2:4], ['--floor', 'wood'])                  # the room's default floor
        self.assertEqual(args[args.index('--room') + 1], root + '/configs/demo/room.json')
        self.assertEqual(args[args.index('--host') + 1], '192.168.50.1')
        self.run_script('scripts/sim/stage1/run_arena_body.zsh', '--room', '--floor', 'grid')
        self.assertEqual(self.calls('mjpython')[-1][0][2:4], ['--floor', 'grid'])
        self.run_script('scripts/sim/stage1/run_arena_body.zsh')
        self.assertNotIn('--room', self.calls('mjpython')[-1][0])

    def test_video_needs_a_room_run(self):
        run = self.root / 'runs' / '20261031T000000Z_s5_room_demo'
        run.mkdir(parents=True)
        result = self.run_script('scripts/stage5/make_demo_video.zsh', str(run))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('--room', result.stderr + result.stdout)
        put(run / 'truth.qpos.f64', '')
        result = self.run_script('scripts/stage5/make_demo_video.zsh', str(run), '--fps', '10')
        self.assertEqual(result.returncode, 0, result.stderr)
        (module, args), = self.modules()
        self.assertEqual(module, 'microduck_toy.stage5.replay')
        self.assertEqual(args, [str(run), '--fps', '10'])

    def test_refuses_runner_options_it_sets_itself(self):
        result = self.run_script('scripts/stage5/run_room_demo.zsh', '--episodes', '3000:bus')
        self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()
