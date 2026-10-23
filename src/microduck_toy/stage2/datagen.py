"""Stage 2 (plan 2A): render detector training and test images of the Stage 1 field, labelled from
segmentation, through the same colour path and input mapping the board uses.

    python -m microduck_toy.stage2.datagen --out runs/<UTC>_s2a_dataset [--splits train,val]
                                           [--count N] [--workers 4]

One frame = one seed. `sample()` turns the seed into a scene description (floor, light, robot pose
and head, which toys are where, which input view) without MuJoCo; `Renderer.render()` poses the
robot kinematically (STAND keyframe, no robotd), renders the head camera and its segmentation, runs
the UYVY round trip, cuts the 320x320 input and writes YOLO labels:

    OUT/images/<split>/<id>.png    the 320x320 input, BGR PNG (what the detector sees)
    OUT/labels/<split>/<id>.txt    YOLO rows: class cx cy w h (0-1 of 320)
    OUT/meta/<split>.jsonl         per frame: the scene description, mapping, every object's frame
                                   and input box, distances, the camera state for ranging, drops
    OUT/data.yaml, OUT/dataset.json

Views (configs/vision/stage2.json "datagen"):
    search     head 0 to +0.3, horizon window; the robot and one object are placed together so
               the object is d m ahead and inside the 45 deg view, with p_far of the distances
               in 1.2-2.0 m (answer 4B); most other toys are moved into the view too
    empty      a search view with every toy in view moved out (about 10% of frames)
    approach   H1 (head +0.6), whole frame letterboxed; one toy 0.15-0.7 m ahead, +-0.15 m left,
               most other toys moved to 0.22-0.9 m ahead inside the view
Splits use disjoint seed ranges and may override the view mix (test_far: far search views only,
for the recall curve out to 2.0 m); Stage 4's 20 formal layouts are seeds 1000-1019 of a different
generator (scene/layout.py) and never appear here.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from multiprocessing import Pool
import os
from pathlib import Path
import time

import numpy as np

from microduck_toy.paths import repo_root
from microduck_toy.vision import uyvy, views

CFG_PATH = Path('configs/vision/stage2.json')
STAGE1_PATH = Path('configs/control/stage1.json')
SHORT_TOYS = ['bus', 'henry', 'ketchup', 'croc', 'thomas']


def load_configs() -> tuple[dict, dict]:
    root = repo_root()
    return (json.loads((root / CFG_PATH).read_text()),
            json.loads((root / STAGE1_PATH).read_text()))


# ----------------------------------------------------------------------------- sampling (no MuJoCo)

def _inside(p, lim) -> bool:
    return abs(p[0]) <= lim and abs(p[1]) <= lim


def _dist(p, q) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def _place_toys(rng, names, arena_cfg, layout_cfg, avoid=()) -> dict | None:
    half = float(arena_cfg['half_size_m'])
    tray = tuple(arena_cfg['tray']['centre_m'])
    lim = half - float(layout_cfg['wall_clear_m'])
    out = {}
    for name in names:
        for _ in range(500):
            p = (float(rng.uniform(-lim, lim)), float(rng.uniform(-lim, lim)))
            if (_dist(p, tray) >= float(layout_cfg['tray_clear_m'])
                    and all(_dist(p, q) >= float(layout_cfg['toy_gap_m']) for q in out.values())
                    and all(_dist(p, q) >= 0.2 for q in avoid)):
                out[name] = p
                break
        else:
            return None
    return {k: [v[0], v[1], float(rng.uniform(-math.pi, math.pi))] for k, v in out.items()}


def _robot_ok(p, toys, arena_cfg, wall_clear, rcfg, skip=None) -> bool:
    half = float(arena_cfg['half_size_m'])
    tray = tuple(arena_cfg['tray']['centre_m'])
    if not _inside(p, half - wall_clear):
        return False
    if _dist(p, tray) < float(arena_cfg['tray']['size_m']) / 2 + float(rcfg['tray_extra_m']):
        return False
    return all(_dist(p, t[:2]) >= float(rcfg['toy_clear_m']) for n, t in toys.items() if n != skip)


def _light(rng, lcfg) -> dict:
    if rng.random() < float(lcfg['p_default']):
        return {'default': True}
    tilt = math.radians(float(rng.uniform(*lcfg['sun_tilt_deg'])))
    az = float(rng.uniform(-math.pi, math.pi))
    return {'default': False,
            'headlight_diffuse': float(rng.uniform(*lcfg['headlight_diffuse'])),
            'headlight_ambient': float(rng.uniform(*lcfg['headlight_ambient'])),
            'sun_diffuse': float(rng.uniform(*lcfg['sun_diffuse'])),
            'sun_dir': [math.sin(tilt) * math.cos(az), math.sin(tilt) * math.sin(az),
                        -math.cos(tilt)]}


def _crowd(rng, robot, toys, focus, dist_m, half_bearing, p_move, arena_cfg, layout_cfg,
           rcfg, p_far=0.0, far_m=None) -> dict:
    """Move each non-focus toy, with probability p_move, to a free spot in front of the robot
    (distance from dist_m, or far_m with probability p_far; bearing within +-half_bearing), so a
    frame carries more labels than a uniform layout gives. The distance is drawn once per toy, so
    the moved toys keep its spread; the Stage 1 layout rules still hold; a toy that finds no free
    spot for its distance in 200 tries stays where it was."""
    half = float(arena_cfg['half_size_m'])
    tray = tuple(arena_cfg['tray']['centre_m'])
    lim = half - float(layout_cfg['wall_clear_m'])
    out = {n: list(t) for n, t in toys.items()}
    for name in sorted(out):
        if name == focus or rng.random() >= p_move:
            continue
        band = far_m if far_m is not None and rng.random() < p_far else dist_m
        d = float(rng.uniform(*band))
        for _ in range(200):
            b = robot[2] + float(rng.uniform(-half_bearing, half_bearing))
            q = (robot[0] + d * math.cos(b), robot[1] + d * math.sin(b))
            if (_inside(q, lim) and _dist(q, tray) >= float(layout_cfg['tray_clear_m'])
                    and _dist(q, robot[:2]) >= float(rcfg['toy_clear_m'])
                    and all(_dist(q, t[:2]) >= float(layout_cfg['toy_gap_m'])
                            for n, t in out.items() if n != name)):
                out[name] = [q[0], q[1], float(rng.uniform(-math.pi, math.pi))]
                break
    return out


def _face_tray(rng, band, toys, arena_cfg, rcfg) -> list | None:
    """Robot d metres from the tray centre, facing it (the tray does not move)."""
    fxy = tuple(arena_cfg['tray']['centre_m'])
    for _ in range(300):
        d = float(rng.uniform(*band))
        phi = float(rng.uniform(-math.pi, math.pi))
        p = (fxy[0] - d * math.cos(phi), fxy[1] - d * math.sin(phi))
        if _robot_ok(p, toys, arena_cfg, float(rcfg['wall_clear_m']), rcfg):
            return [p[0], p[1], phi]
    return None


def _face_toy(rng, band, focus, toys, arena_cfg, layout_cfg, rcfg):
    """Robot and focus toy placed together, the toy d metres straight ahead, d uniform in band.

    Placing the robot around a toy that already lies somewhere uniform almost never gives
    1.6-2.0 m (that needs a toy near one corner and the robot near the opposite one), so the
    toy moves with the robot. A d that fits nowhere in 20000 tries is drawn again."""
    half = float(arena_cfg['half_size_m'])
    tray = tuple(arena_cfg['tray']['centre_m'])
    lim = half - float(layout_cfg['wall_clear_m'])
    rlim = half - float(rcfg['wall_clear_m'])
    others = {n: t for n, t in toys.items() if n != focus}
    for _ in range(20):
        d = float(rng.uniform(*band))
        for _ in range(20000):
            p = (float(rng.uniform(-rlim, rlim)), float(rng.uniform(-rlim, rlim)))
            phi = float(rng.uniform(-math.pi, math.pi))
            q = (p[0] + d * math.cos(phi), p[1] + d * math.sin(phi))
            if (_inside(q, lim) and _dist(q, tray) >= float(layout_cfg['tray_clear_m'])
                    and all(_dist(q, t[:2]) >= float(layout_cfg['toy_gap_m']) for t in others.values())
                    and _robot_ok(p, others, arena_cfg, float(rcfg['wall_clear_m']), rcfg)):
                moved = dict(toys)
                moved[focus] = [q[0], q[1], float(rng.uniform(-math.pi, math.pi))]
                return [p[0], p[1], phi], moved
    return None, toys


def in_view(robot, point, half_fov=math.radians(25.0)) -> bool:
    """Roughly inside the camera's 45 deg horizontal view (a little margin), in front."""
    dx, dy = point[0] - robot[0], point[1] - robot[1]
    bearing = math.atan2(dy, dx) - robot[2]
    bearing = math.atan2(math.sin(bearing), math.cos(bearing))
    return abs(bearing) <= half_fov and math.hypot(dx, dy) > 0.05


def sample(seed: int, split: str, cfg: dict, stage1: dict) -> dict:
    """The scene for one frame, from its seed only."""
    g = cfg['datagen']
    arena_cfg, layout_cfg = stage1['arena'], stage1['layout']
    rng = np.random.default_rng(int(seed))
    sc = g['splits'][split]
    # A split may override the view mix (test_far: far search views only).
    p_search = float(sc.get('p_search', g['p_search']))
    p_empty = float(sc.get('p_empty_in_search', g['p_empty_in_search']))
    p_tray = float(sc.get('p_tray_focus', g['search']['p_tray_focus']))
    p_far = float(sc.get('p_far', g['search']['p_far']))
    floors = sc['floors']
    floor = floors[int(rng.integers(len(floors)))]
    light = _light(rng, g['light'])
    jit = g['jitter']
    jitter = {'pitch': math.radians(float(rng.uniform(-1, 1)) * float(jit['pitch_deg'])),
              'roll': math.radians(float(rng.uniform(-1, 1)) * float(jit['roll_deg'])),
              'dz': float(rng.uniform(-1, 1)) * float(jit['height_m'])}
    lo, hi = g['toys_per_frame']
    for _attempt in range(200):
        k = int(rng.integers(lo, hi + 1))
        names = [SHORT_TOYS[i] for i in sorted(rng.choice(len(SHORT_TOYS), size=k, replace=False))]
        toys = _place_toys(rng, names, arena_cfg, layout_cfg)
        if toys is None:
            continue
        view = 'search' if rng.random() < p_search else 'approach'
        if view == 'search':
            s = g['search']
            head = float(rng.uniform(*s['head_pitch_rad']))
            win = int(rng.integers(-int(s['window_jitter_rows']), int(s['window_jitter_rows']) + 1))
            if rng.random() < p_empty:
                view = 'empty'
                for _ in range(200):
                    lim = float(arena_cfg['half_size_m']) - float(g['robot']['wall_clear_m'])
                    p = (float(rng.uniform(-lim, lim)), float(rng.uniform(-lim, lim)))
                    if _robot_ok(p, toys, arena_cfg, float(g['robot']['wall_clear_m']), g['robot']):
                        robot = [p[0], p[1], float(rng.uniform(-math.pi, math.pi))]
                        break
                else:
                    continue
                parked = [n for n, t in toys.items() if in_view(robot, t, math.radians(30))]
                toys = {n: t for n, t in toys.items() if n not in parked}
                focus = None
            else:
                focus = 'tray' if rng.random() < p_tray else str(rng.choice(names))
                band = s['far_m'] if rng.random() < p_far else s['any_m']
                if focus == 'tray':
                    robot = _face_tray(rng, band, toys, arena_cfg, g['robot'])
                else:
                    robot, toys = _face_toy(rng, band, focus, toys, arena_cfg, layout_cfg, g['robot'])
                if robot is None:
                    continue
                yaw = robot[2] + float(rng.uniform(-1, 1)) * float(s['yaw_off_rad'])
                robot[2] = math.atan2(math.sin(yaw), math.cos(yaw))
                toys = _crowd(rng, robot, toys, focus, s['any_m'],
                              math.radians(float(s['extra_bearing_deg'])),
                              float(s['p_extra_in_view']), arena_cfg, layout_cfg, g['robot'],
                              p_far, s['far_m'])
            frame = {'view': view, 'head': [0.0, head], 'window_jitter_rows': win, 'focus': focus}
        else:
            a = g['approach']
            head = float(rng.uniform(*a['head_pitch_rad']))
            focus = str(rng.choice(names))
            robot = None
            for _ in range(300):
                fwd = float(rng.uniform(*a['forward_m']))
                left = float(rng.uniform(*a['left_m']))
                yaw = float(rng.uniform(-math.pi, math.pi))
                tx, ty = toys[focus][:2]
                c, s_ = math.cos(yaw), math.sin(yaw)
                p = (tx - (c * fwd - s_ * left), ty - (s_ * fwd + c * left))
                if _robot_ok(p, toys, arena_cfg, float(a['robot_wall_clear_m']), g['robot'],
                             skip=focus):
                    robot = [p[0], p[1], yaw]
                    break
            if robot is None:
                continue
            toys = _crowd(rng, robot, toys, focus, a['extra_dist_m'],
                          math.radians(float(a['extra_bearing_deg'])), float(a['p_extra_in_view']),
                          arena_cfg, layout_cfg, g['robot'])
            frame = {'view': 'approach', 'head': [0.0, head], 'window_jitter_rows': 0,
                     'focus': focus}
        return {'seed': int(seed), 'split': split, 'floor': floor, 'light': light,
                'robot': [round(v, 5) for v in robot], 'jitter': jitter,
                'toys': {n: [round(v, 5) for v in t] for n, t in toys.items()}, **frame}
    raise RuntimeError(f'seed {seed}: no scene after 200 attempts')


# ----------------------------------------------------------------------------- rendering

def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


def _axis_quat(axis, angle):
    s = math.sin(angle / 2)
    return [math.cos(angle / 2), axis[0] * s, axis[1] * s, axis[2] * s]


def label_objects(boxes, truth: dict, trunk_xy, heading: float, cam_xy, mapping, cfg: dict):
    """Segmentation boxes in the upright frame -> (YOLO labels, objects, dropped).

    truth: {name: (x, y, z, half extents or None)} in the world; trunk_xy and heading place the
    robot, cam_xy the camera (for dist_cam_m). An object outside the input, or smaller than
    label.min_side_px / min_area_px there, is dropped (kept in `dropped` with the reason)."""
    lab = cfg['datagen']['label']
    classes = cfg['classes']
    c, s = math.cos(heading), math.sin(heading)
    labels, objects, dropped = [], [], []
    for b in boxes:
        name = b['name']
        if name not in truth:
            continue  # a parked toy seen over the wall (cannot happen inside the field)
        box_in = views.to_input_box(b['xyxy'], mapping)
        x, y, z, half = truth[name]
        dx, dy = x - trunk_xy[0], y - trunk_xy[1]
        obj = {'name': name, 'cls': classes.index(name), 'xyxy_frame': b['xyxy'],
               'area_frame_px': b['area_px'], 'cut_frame': b['cut'],
               'xyxy_input': [round(v, 2) for v in box_in] if box_in else None,
               'dist_cam_m': round(math.hypot(x - cam_xy[0], y - cam_xy[1]), 4),
               'fwd_m': round(c * dx + s * dy, 4), 'left_m': round(-s * dx + c * dy, 4),
               'half_z_m': round(half[2], 4) if half else None}
        if box_in is None:
            dropped.append({**obj, 'why': 'outside input'})
            continue
        w, h = box_in[2] - box_in[0], box_in[3] - box_in[1]
        if min(w, h) < float(lab['min_side_px']) or w * h < float(lab['min_area_px']):
            dropped.append({**obj, 'why': 'too small'})
            continue
        objects.append(obj)
        n = views.INPUT
        labels.append((obj['cls'], (box_in[0] + box_in[2]) / 2 / n, (box_in[1] + box_in[3]) / 2 / n,
                       w / n, h / n))
    return labels, objects, dropped


def yolo_rows(labels) -> str:
    return ''.join(f'{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n' for c, x, y, w, h in labels)


class Renderer:
    """One compiled field per floor; the head camera and segmentation at 640x360."""

    def __init__(self, rl_root: Path, cfg: dict, stage1: dict):
        self.rl_root, self.cfg, self.stage1 = Path(rl_root), cfg, stage1
        self.scenes = {}

    def scene(self, floor: str):
        if floor not in self.scenes:
            import mujoco
            from microduck_toy.scene import arena
            from microduck_toy.scene import render as scene_render
            spec = arena.build_spec(self.stage1['arena'], self.rl_root,
                                    'wood' if floor == 'wood' else 'grid')
            if floor == 'carpet':
                c = self.cfg['datagen']['carpet']
                tex = spec.add_texture(name='carpet', type=mujoco.mjtTexture.mjTEXTURE_2D,
                                       builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
                                       mark=mujoco.mjtMark.mjMARK_RANDOM, random=float(c['random']),
                                       rgb1=list(c['rgb1']), markrgb=list(c['markrgb']),
                                       width=512, height=512)
                ground = spec.material('groundplane')
                ground.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
            elif floor not in ('grid', 'wood'):
                raise ValueError(f'unknown floor {floor!r}')
            r = scene_render.ArenaRender.__new__(scene_render.ArenaRender)
            r.model = spec.compile()
            r.data = mujoco.MjData(r.model)
            r.ids = arena.ArenaIds(r.model)
            r.owners = scene_render.owners(r.model, r.ids)
            r.cam = scene_render.turned_camera(r.model)
            r.key = r.model.key('STAND').id
            r.rgb = mujoco.Renderer(r.model, height=scene_render.RAW_H, width=scene_render.RAW_W)
            r.seg = mujoco.Renderer(r.model, height=scene_render.RAW_H, width=scene_render.RAW_W)
            r.seg.enable_segmentation_rendering()
            m = r.model
            default = {'hd': m.vis.headlight.diffuse.copy(), 'ha': m.vis.headlight.ambient.copy(),
                       'sd': m.light_diffuse[0].copy(), 'dir': m.light_dir[0].copy()}
            self.scenes[floor] = (r, default)
        return self.scenes[floor]

    def _set_light(self, r, default, light):
        m = r.model
        if light['default']:
            m.vis.headlight.diffuse[:] = default['hd']
            m.vis.headlight.ambient[:] = default['ha']
            m.light_diffuse[0] = default['sd']
            m.light_dir[0] = default['dir']
        else:
            m.vis.headlight.diffuse[:] = light['headlight_diffuse']
            m.vis.headlight.ambient[:] = light['headlight_ambient']
            m.light_diffuse[0] = light['sun_diffuse']
            m.light_dir[0] = light['sun_dir']

    def pose(self, scene: dict):
        """Pose robot and toys; returns the ArenaRender with mj_forward done."""
        import mujoco
        from microduck_toy.scene import arena
        from microduck_toy.scene import toys as toy_lib
        r, default = self.scene(scene['floor'])
        self._set_light(r, default, scene['light'])
        x, y, yaw = scene['robot']
        r.pose(tuple(scene['head']), x=x, y=y, yaw=yaw)
        j = scene['jitter']
        q = _quat_mul(_axis_quat((0, 0, 1), yaw),
                      _quat_mul(_axis_quat((0, 1, 0), j['pitch']), _axis_quat((1, 0, 0), j['roll'])))
        r.data.qpos[3:7] = q
        r.data.qpos[2] += j['dz']
        for short, (tx, ty, tyaw) in scene['toys'].items():
            arena.place_toy(r.model, r.data, r.ids, toy_lib.resolve(short), tx, ty, tyaw)
        mujoco.mj_forward(r.model, r.data)
        return r

    @staticmethod
    def camera_cv(r) -> tuple[np.ndarray, np.ndarray]:
        """Camera position and rotation (upright-image OpenCV axes -> world)."""
        xmat = r.data.cam_xmat[r.cam].reshape(3, 3)
        # MuJoCo camera: x right, y up, looking along -z of the raw image; upright = rot90 k=3:
        # upright column follows raw up (+y), upright row follows raw right (+x).
        rot = np.stack([xmat[:, 1], xmat[:, 0], -xmat[:, 2]], axis=1)
        return r.data.cam_xpos[r.cam].copy(), rot

    def render(self, scene: dict) -> tuple[np.ndarray, list[dict], dict]:
        """(320x320 BGR input, labels, meta) for one scene."""
        import mujoco
        from microduck_toy.scene import arena
        from microduck_toy.scene import toys as toy_lib
        r = self.pose(scene)
        # The first RGB render after a pose differs from the next ones (up to ~90 levels on a few
        # hundred pixels, OSMesa, MuJoCo 3.10); the sim body streams frames continuously, so the
        # camera always shows the settled image. Render once to settle, keep the second.
        r.rgb.update_scene(r.data, camera=r.cam)
        r.rgb.render()
        rgb, boxes = r.view()
        frame = uyvy.round_trip_upright(rgb)
        cam_pos, cam_rot = self.camera_cv(r)
        horizon = views.horizon_row(cam_rot)
        if scene['view'] == 'approach':
            mapping = views.letterbox_mapping()
        else:
            margin = int(self.cfg['input']['window_margin_rows']) + int(scene['window_jitter_rows'])
            mapping = views.window_mapping(horizon, margin)
        img = views.to_input(np.ascontiguousarray(frame[:, :, ::-1]), mapping)

        # Truth for every object: world position, distances from camera and trunk.
        trunk = r.data.qpos[:3].copy()
        tq = r.data.qpos[3:7].copy()
        tmat = np.zeros(9)
        mujoco.mju_quat2Mat(tmat, tq)
        tmat = tmat.reshape(3, 3)
        hyaw = math.atan2(tmat[1, 0], tmat[0, 0])
        truth = {}
        for short in scene['toys']:
            x, y, z, _ = arena.toy_pose(r.data, r.ids, toy_lib.resolve(short))
            truth[short] = (x, y, z, toy_lib.catalogue()[toy_lib.resolve(short)].half)
        tray_xy = r.ids.tray_centre
        truth['tray'] = (float(tray_xy[0]), float(tray_xy[1]), 0.0, None)

        labels, objects, dropped = label_objects(boxes, truth, trunk[:2], hyaw, cam_pos[:2], mapping,
                                                 self.cfg)

        # The camera as robot.state reports it (frames.camera in the trunk frame, OpenCV upright
        # axes), so ranging.ground_point can run on these frames in 2E.
        rel = tmat.T @ (cam_pos - trunk)
        crot = tmat.T @ cam_rot
        cq = np.zeros(4)
        mujoco.mju_mat2Quat(cq, crot.flatten())
        state = {'frames': {'camera': {'pos': [round(v, 6) for v in rel],
                                       'quat': [round(v, 7) for v in cq]}},
                 'imu': {'quat': [round(float(v), 7) for v in tq]},
                 'odom': {'position': [0.0, 0.0, round(float(trunk[2]), 6)]}}
        meta = {**scene, 'mapping': mapping.as_dict(),
                'horizon_row': round(horizon, 2) if horizon is not None else None,
                'objects': objects, 'dropped': dropped, 'state': state}
        return img, labels, meta


# ----------------------------------------------------------------------------- dataset

def frame_id(split: str, seed: int) -> str:
    return f'{split}_{seed:06d}'


_worker = {}


def _init_worker(rl_root: str):
    cfg, stage1 = load_configs()
    _worker['r'] = Renderer(Path(rl_root), cfg, stage1)
    _worker['cfg'], _worker['stage1'] = cfg, stage1


def _job(args):
    out, split, seed = args
    import cv2
    cfg, stage1, rend = _worker['cfg'], _worker['stage1'], _worker['r']
    scene = sample(seed, split, cfg, stage1)
    img, labels, meta = rend.render(scene)
    fid = frame_id(split, seed)
    out = Path(out)
    cv2.imwrite(str(out / 'images' / split / f'{fid}.png'), img)
    (out / 'labels' / split / f'{fid}.txt').write_text(yolo_rows(labels))
    return {'id': fid, **meta}


def build(out: Path, splits: list[str], count: int | None, workers: int, rl_root: Path) -> dict:
    cfg, _ = load_configs()
    jobs = []
    for split in splits:
        sc = cfg['datagen']['splits'][split]
        n = int(count if count is not None else sc['count'])
        (out / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out / 'labels' / split).mkdir(parents=True, exist_ok=True)
        jobs += [(str(out), split, int(sc['seed_base']) + i) for i in range(n)]
    (out / 'meta').mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    metas: dict[str, list] = {s: [] for s in splits}
    with Pool(workers, initializer=_init_worker, initargs=(str(rl_root),)) as pool:
        for i, meta in enumerate(pool.imap(_job, jobs, chunksize=4)):
            metas[meta['split']].append(meta)
            if (i + 1) % 200 == 0:
                print(f'{i + 1}/{len(jobs)} frames, {time.time() - t0:.0f} s', flush=True)
    for split, rows in metas.items():
        rows.sort(key=lambda m: m['seed'])
        with open(out / 'meta' / f'{split}.jsonl', 'w') as f:
            for m in rows:
                f.write(json.dumps(m) + '\n')
    names = cfg['classes']
    yaml = ['path: .', *(f'{s}: images/{s}' for s in splits if s in ('train', 'val', 'test')),
            f'nc: {len(names)}', 'names:', *(f'  {i}: {n}' for i, n in enumerate(names))]
    (out / 'data.yaml').write_text('\n'.join(yaml) + '\n')
    summary = summarise(metas, cfg)
    import mujoco
    summary.update({'created_utc': dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                    'seconds': round(time.time() - t0, 1), 'workers': workers,
                    'mujoco': mujoco.__version__, 'host': os.uname().nodename,
                    'config_sha256': hashlib.sha256((repo_root() / CFG_PATH).read_bytes()).hexdigest()})
    (out / 'dataset.json').write_text(json.dumps(summary, indent=1) + '\n')
    return summary


def summarise(metas: dict, cfg: dict) -> dict:
    out = {}
    for split, rows in metas.items():
        inst = {n: 0 for n in cfg['classes']}
        bands = {n: {'<0.6': 0, '0.6-1.2': 0, '>1.2': 0} for n in cfg['classes']}
        views_ = {}
        for m in rows:
            views_[m['view']] = views_.get(m['view'], 0) + 1
            for o in m['objects']:
                inst[o['name']] += 1
                d = math.hypot(o['fwd_m'], o['left_m'])  # from the trunk, as stage2.evaluate bins
                bands[o['name']]['<0.6' if d < 0.6 else '0.6-1.2' if d <= 1.2 else '>1.2'] += 1
        out[split] = {'frames': len(rows), 'views': views_, 'instances': inst, 'by_distance': bands,
                      'empty_frames': sum(1 for m in rows if not m['objects']),
                      'dropped': sum(len(m['dropped']) for m in rows)}
    return {'splits': out}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--splits', default='train,val,test,test_unseen,test_far')
    ap.add_argument('--count', type=int, default=None, help='frames per split (default: config)')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument('--rl-root', type=Path, default=repo_root() / 'third_party' / 'microduck_rl')
    args = ap.parse_args(argv)
    summary = build(args.out, args.splits.split(','), args.count, args.workers, args.rl_root)
    print(json.dumps(summary['splits'], indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
