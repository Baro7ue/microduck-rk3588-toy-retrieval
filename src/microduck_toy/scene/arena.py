"""The Stage 1 field as a MuJoCo model: upstream scene.xml (robot + floor) plus walls, a tray, the
five toys, and the grasp plumbing.

    spec = build_spec(cfg)            # MjSpec; cfg = configs/control/stage1.json "arena"
    model = spec.compile()
    ids = ArenaIds(model)             # toy bodies/geoms, tray geoms, weld ids, park spots

Nothing in the upstream repositories changes: the scene file is read through MjSpec and extended in
memory. Field layout (world frame, the robot's HOME spawn at the origin facing +x):

    walls     a square of side 2 * half_size_m centred on the origin, wall_height_m tall
    tray      a base plate and four rims, centre at tray.centre_m (default +x wall centre,
              0.2 m between the tray edge and the wall); tray.shape "round" (Stage 5 room demo
              only) makes it round, diameter size_m, rim rim_height_m tall
    toys      a free joint each; one visual mesh (no collision) and one collision box (not drawn).
              Spawned parked outside the field; scripts move them in through the probe.
    grasp     one inactive weld jaw_soft <-> toy per toy, and a contact exclude for the same pair
              (idea from microduck-lab world/compose.py:702-716, Apache-2.0: a rigid jaw hull sits on
              top of a toy and holds the beak above it, which no real soft beak does)

Adding free-joint bodies pads every keyframe with the toys at the origin (MuJoCo 3.3/3.10). The
body server never resets to a keyframe for the toys; scripts place them after every reset.
"""
from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from microduck_toy.scene import bowl
from microduck_toy.scene import toys as toy_lib

UPSTREAM_SCENE = Path('src/mjlab_microduck/robot/microduck/scene.xml')
WOOD_RGB = (0.72, 0.58, 0.42)     # the Stage 0 d4 "light wood" floor (curves.py FLOORS)
WALL_RGBA = (0.85, 0.85, 0.82, 1.0)
TRAY_RGBA = (0.30, 0.30, 0.34, 1.0)
PARK_Y = -1.6                     # outside the field, in a row along -y
PARK_X0 = -0.6
PARK_DX = 0.3


def scene_path(rl_root: Path) -> Path:
    return Path(rl_root) / UPSTREAM_SCENE


def _add_box(parent, name, size, pos, rgba, group=0, contype=1, conaffinity=1, mass=None):
    kwargs = dict(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=list(size), pos=list(pos),
                  rgba=list(rgba), group=group, contype=contype, conaffinity=conaffinity)
    if mass is not None:
        kwargs['mass'] = mass
    return parent.add_geom(**kwargs)


def _round_tray(body, r: float, wall_h: float, wall_t: float, base_t: float, segments: int) -> None:
    """Stage 5 room demo: a round tray of outer radius r on the tray body, same colour as the
    square one (the frozen YOLO knows the tray by it). A disc base and a rim of `segments` thin
    boxes round the circle; each box is as long as the outer chord's tangent span, so the boxes
    overlap a little on the inside and leave no gap on the outside."""
    body.add_geom(name='tray_base', type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[r, base_t / 2, 0],
                  pos=[0, 0, base_t / 2], rgba=list(TRAY_RGBA))
    half_len = r * math.tan(math.pi / segments)
    mid = r - wall_t / 2
    for k in range(segments):
        a = 2 * math.pi * k / segments
        body.add_geom(name=f'tray_wall_{k:02d}', type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[wall_t / 2, half_len, wall_h / 2],
                      pos=[mid * math.cos(a), mid * math.sin(a), wall_h / 2],
                      quat=[math.cos(a / 2), 0, 0, math.sin(a / 2)], rgba=list(TRAY_RGBA))


def bowl_profile(tray: dict) -> dict:
    """The bowl's outer wall (Stage 5 room demo, tray.shape 'bowl'): scene/bowl.py profile."""
    return bowl.profile(tray)


def _bowl_tray(body, tray: dict, base_t: float, segments: int) -> None:
    """Stage 5 room demo: a bowl, same colour as the trays. A disc base of the floor radius and a cone
    of `segments` thin boxes from (base_r, 0) to (lip_r, h) on their outer face. Each box is as wide
    as the lip's tangent span, so the boxes leave no gap at the lip and overlap further down. With
    tray.priority set, the bowl's friction (tray.friction) is the contact's, whatever the toy's
    (MuJoCo: the higher geom priority wins), so a toy that lands on the wall slides in."""
    p = bowl_profile(tray)
    lip_r, base_r, h, t, alpha = p['lip_r'], p['base_r'], p['h'], p['t'], p['alpha']
    extra = {}
    if 'friction' in tray:
        extra['friction'] = [float(tray['friction']), 0.005, 0.0001]
    if 'priority' in tray:
        extra['priority'] = int(tray['priority'])
    body.add_geom(name='tray_base', type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[base_r, base_t / 2, 0],
                  pos=[0, 0, base_t / 2], rgba=list(TRAY_RGBA), **extra)
    slant = math.hypot(lip_r - base_r, h)
    half_len = lip_r * math.tan(math.pi / segments)
    # The box's local z runs up the slope, its local x is the outer face's normal (out and down):
    # a turn of beta = 90 deg - alpha about the tangent, then the azimuth about z.
    beta = math.pi / 2 - alpha
    mid_r = (base_r + lip_r) / 2 - t / 2 * math.sin(alpha)
    mid_z = h / 2 + t / 2 * math.cos(alpha)
    for k in range(segments):
        a = 2 * math.pi * k / segments
        quat = np.zeros(4)
        mujoco.mju_mulQuat(quat, np.array([math.cos(a / 2), 0, 0, math.sin(a / 2)]),
                           np.array([math.cos(beta / 2), 0, math.sin(beta / 2), 0]))
        body.add_geom(name=f'tray_wall_{k:02d}', type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[t / 2, half_len, slant / 2],
                      pos=[mid_r * math.cos(a), mid_r * math.sin(a), mid_z],
                      quat=list(quat), rgba=list(TRAY_RGBA), **extra)


def park_spot(index: int, y: float = PARK_Y) -> tuple[float, float]:
    return PARK_X0 + PARK_DX * index, y


def build_spec(cfg: dict, rl_root: Path, floor: str = 'grid') -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(scene_path(rl_root)))
    # The head camera renders 640x360; segmentation renders share the same buffer size.
    spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, 640)
    spec.visual.global_.offheight = max(spec.visual.global_.offheight, 480)

    if floor == 'wood':
        ground = spec.material('groundplane')
        ground.textures = [''] * len(ground.textures)
        ground.rgba = list(WOOD_RGB) + [1.0]
    elif floor != 'grid':
        raise ValueError(f'unknown floor {floor!r} (grid or wood)')

    world = spec.worldbody
    # The room demo (Stage 5) puts its walls at room_half_m, outside the field the brain walks in
    # (half_size_m), with furniture in between; without it the walls are the field's edge.
    half = float(cfg.get('room_half_m', cfg['half_size_m']))
    h = float(cfg['wall_height_m'])
    t = float(cfg['wall_thickness_m'])
    rgba = tuple(cfg.get('wall_rgba', WALL_RGBA))
    park_y = PARK_Y if 'room_half_m' not in cfg else -(half + 0.6)
    for name, pos, size in (
        ('wall_px', (half + t / 2, 0, h / 2), (t / 2, half + t, h / 2)),
        ('wall_nx', (-half - t / 2, 0, h / 2), (t / 2, half + t, h / 2)),
        ('wall_py', (0, half + t / 2, h / 2), (half, t / 2, h / 2)),
        ('wall_ny', (0, -half - t / 2, h / 2), (half, t / 2, h / 2)),
    ):
        _add_box(world, name, size, pos, rgba)

    tray = cfg['tray']
    side, rim_h = float(tray['size_m']), float(tray['rim_height_m'])
    rim_t, base_t = float(tray['rim_thickness_m']), float(tray['base_thickness_m'])
    cx, cy = (float(v) for v in tray['centre_m'])
    body = world.add_body(name='tray', pos=[cx, cy, 0.0])
    s = side / 2
    if tray.get('shape', 'square') == 'round':
        _round_tray(body, s, rim_h, rim_t, base_t, int(tray.get('segments', 32)))
    elif tray.get('shape') == 'bowl':
        _bowl_tray(body, tray, base_t, int(tray.get('segments', 32)))
    else:
        _add_box(body, 'tray_base', (s, s, base_t / 2), (0, 0, base_t / 2), TRAY_RGBA)
        for name, pos, size in (
            ('tray_rim_px', (s - rim_t / 2, 0, rim_h / 2), (rim_t / 2, s, rim_h / 2)),
            ('tray_rim_nx', (-s + rim_t / 2, 0, rim_h / 2), (rim_t / 2, s, rim_h / 2)),
            ('tray_rim_py', (0, s - rim_t / 2, rim_h / 2), (s - rim_t, rim_t / 2, rim_h / 2)),
            ('tray_rim_ny', (0, -s + rim_t / 2, rim_h / 2), (s - rim_t, rim_t / 2, rim_h / 2)),
        ):
            _add_box(body, name, size, pos, TRAY_RGBA)

    # Stage 5 room demo (optional keys): no shadows, furniture boxes and copies of the toys. With the
    # room's tall walls and furniture the directional light's shadow map draws striped bands on the
    # floor (OSMesa, MuJoCo 3.10), and the frozen YOLO took the stripes for a bus (cloud check
    # 2026-10-31: 10 phantom buses in 216 frames, none with shadows off).
    if not cfg.get('cast_shadows', True):
        for light in spec.lights:
            light.castshadow = False
    for piece in cfg.get('furniture', []):
        for k, box in enumerate(piece['boxes']):
            _add_box(world, f'furn_{piece["name"]}_{k}', box['size'], box['pos'],
                     tuple(box.get('rgba', piece.get('rgba', (0.6, 0.6, 0.6, 1.0)))))
    toy_lib.register_copies(cfg.get('copies'))

    made = set()
    for i, (name, toy) in enumerate(toy_lib.scene_toys().items()):
        if toy.name not in made:            # one texture, material and mesh per kind of toy
            made.add(toy.name)
            tex = spec.add_texture(name=f'tex_{toy.name}', type=mujoco.mjtTexture.mjTEXTURE_2D,
                                   file=str(toy.texture))
            mat = spec.add_material(name=f'mat_{toy.name}', specular=0.3, shininess=0.3)
            mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
            spec.add_mesh(name=f'mesh_{toy.name}', file=str(toy.mesh), scale=[toy.scale] * 3)
        px, py = park_spot(i, park_y)
        b = world.add_body(name=name, pos=[px, py, toy.rest_z])
        b.add_freejoint(name=f'{name}_free')
        b.add_geom(name=f'{name}_visual', type=mujoco.mjtGeom.mjGEOM_MESH,
                   meshname=f'mesh_{toy.name}', material=f'mat_{toy.name}',
                   pos=list(toy.mesh_pos), quat=list(toy.mesh_quat),
                   contype=0, conaffinity=0, group=2, density=0)
        # Group 3: collision only, not drawn by the default renderer or viewer.
        _add_box(b, f'{name}_box', toy.half, (0, 0, 0), (1, 0, 1, 0.3), group=3,
                 mass=toy.mass_kg)
        spec.add_equality(name=f'hold_{name}', type=mujoco.mjtEq.mjEQ_WELD,
                          objtype=mujoco.mjtObj.mjOBJ_BODY, name1='jaw_soft', name2=name,
                          active=False)
        ex = spec.add_exclude(name=f'beak_{name}')
        ex.bodyname1, ex.bodyname2 = 'jaw_soft', name
    return spec


class ArenaIds:
    """Ids the body server and the renderers need, resolved once per compiled model."""

    def __init__(self, model: mujoco.MjModel):
        self.toys = {}
        for name, toy in toy_lib.scene_toys().items():
            body = model.body(name).id
            joint = model.body_jntadr[body]
            self.toys[name] = {
                'body': body,
                'visual': model.geom(f'{name}_visual').id,
                'box': model.geom(f'{name}_box').id,
                'eq': model.equality(f'hold_{name}').id,
                'qadr': int(model.jnt_qposadr[joint]),
                'vadr': int(model.jnt_dofadr[joint]),
                'park': tuple(float(v) for v in model.body_pos[body][:2]),   # where build_spec put it
                'rest_z': toy.rest_z,
                'half': toy.half,
            }
        self.tray = model.body('tray').id
        self.tray_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == self.tray]
        self.tray_centre = model.body_pos[self.tray][:2].copy()
        self.jaw = model.body('jaw_soft').id
        self.tip = model.site('mouth_tip').id


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def place_toy(model, data, ids: ArenaIds, name: str, x: float, y: float, yaw: float,
              z: float | None = None) -> None:
    """Put a toy at rest at (x, y), long axis along `yaw`; zero its velocity."""
    toy = ids.toys[name]
    qa, va = toy['qadr'], toy['vadr']
    data.qpos[qa: qa + 3] = [x, y, toy['rest_z'] + 0.0005 if z is None else z]
    data.qpos[qa + 3: qa + 7] = yaw_quat(yaw)
    data.qvel[va: va + 6] = 0.0


def park_toy(model, data, ids: ArenaIds, name: str) -> None:
    x, y = ids.toys[name]['park']
    place_toy(model, data, ids, name, x, y, 0.0)


def toy_pose(data, ids: ArenaIds, name: str) -> tuple[float, float, float, float]:
    """(x, y, z, yaw) of a toy's box centre."""
    qa = ids.toys[name]['qadr']
    w, qx, qy, qz = (float(v) for v in data.qpos[qa + 3: qa + 7])
    yaw = math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return float(data.qpos[qa]), float(data.qpos[qa + 1]), float(data.qpos[qa + 2]), yaw
