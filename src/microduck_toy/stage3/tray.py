"""The tray: which side to drop from, the staging point, the tray look, and the drop walk.

    side = choose_side(robot_xy, obstacles, fld, cfg)        # 'x-', 'y+' or 'y-' + staging point
    m = measure_tray(boxes_and_states, cam, cfg, expected)   # tray centre (forward, left) from H1
    walk = drop_walk(m['centre'], d)                         # turn + walk so trunk-to-centre = d

Plan 3 (CARRY, TRAY_LOOK, DROP) and 3C. The robot carries the toy to a staging point on one side's
normal, staging_m from the tray centre (lab aim_range 0.42 and basket_normal, microduck-lab@8dad921
brain/tidy.py TidyParams, Apache-2.0, reference only), squares up facing the centre, looks once at
H1, then walks blind until its trunk is drop_d_m from the centre and opens the beak. The +x side is
never used: only 0.2 m of floor between the tray and the wall.

The tray look, at H1 from the staging point (the near rim about 0.3 m ahead): the bottom edge of the
tray box is where the near rim meets the floor, so its ground point gives the distance to the near
rim, and the centre lies half a tray side beyond it. Sideways, the near rim is wider than the image
at that distance (H1 sees about +-12 cm there), so the box is usually cut at both sides; a side that
is not cut gives that side rim, and with both cut the centre is within the band the picture leaves
for it, so the odometry estimate is kept, clipped to that band. Plan 3 risk table: only a tray box whose
centre lands within accept_m of where odometry expects the tray is used.
"""
from __future__ import annotations

import math

from microduck_toy.control import ranging
from microduck_toy.stage3.world import Field, plan_route, route_length, seg_point_dist

NORMALS = {'x-': (-1.0, 0.0), 'y+': (0.0, 1.0), 'y-': (0.0, -1.0), 'x+': (1.0, 0.0)}


def staging_point(side: str, fld: Field, staging_m: float) -> tuple[float, float]:
    nx, ny = NORMALS[side]
    cx, cy = fld.tray_centre
    return cx + nx * staging_m, cy + ny * staging_m


def facing(side: str) -> float:
    """Field yaw that faces the tray centre from that side's staging point."""
    nx, ny = NORMALS[side]
    return math.atan2(-ny, -nx)


def corridor_blocked(side: str, fld: Field, staging_m: float, half_width: float,
                     obstacles) -> list:
    """Known toys in the lane from the staging point to the tray edge (half_width each side)."""
    sx, sy = staging_point(side, fld, staging_m)
    cx, cy = fld.tray_centre
    nx, ny = NORMALS[side]
    ex, ey = cx + nx * fld.tray_half_m, cy + ny * fld.tray_half_m
    return [o for o in obstacles if seg_point_dist((sx, sy), (ex, ey), o) < half_width]


def choose_side(robot_xy, obstacles, fld: Field, cfg: dict) -> dict:
    """The drop side: among cfg['sides'] whose staging point is inside the field and whose lane
    holds no known toy, the one with the shortest route; if every lane is blocked, the shortest
    route anyway (the robot squares up there and the blocked lane is noted)."""
    staging_m, half = float(cfg['staging_m']), float(cfg['lane_half_m'])
    options = []
    for side in cfg['sides']:
        point = staging_point(side, fld, staging_m)
        if not fld.inside(point):
            continue
        route = plan_route(robot_xy, point, obstacles, fld)
        if route is None:
            continue
        blocked = corridor_blocked(side, fld, staging_m, half, obstacles)
        options.append({'side': side, 'staging': point, 'route': route, 'blocked': len(blocked),
                        'length_m': route_length(robot_xy, route), 'face_yaw': facing(side)})
    if not options:
        return {'side': None, 'options': []}
    clear = [o for o in options if o['blocked'] == 0]
    best = min(clear or options, key=lambda o: o['length_m'])
    return {**best, 'options': [{k: o[k] for k in ('side', 'blocked', 'length_m')} for o in options]}


def measure_tray(looks: list, cam: ranging.CameraModel, cfg: dict, tray_half_m: float,
                 rim_z_m: float, expected, near_z_m: float = 0.0, shape: str = 'square') -> dict:
    """The tray centre (forward, left) in the level trunk frame from standing looks at H1.

    looks: [(state, boxes)] per frame, boxes [{'cls', 'conf', 'xyxy'}] in the upright image.
    expected: (forward, left) of the centre from odometry. Returns the measured centre (median
    over the frames that saw an acceptable tray box) or the expected one, and how it was got.

    The box's bottom edge is where the near rim's outer face meets the floor (near_z_m; 3P: z = 0
    lands 0.7 cm short of the rim on average, the rim top 1.8 cm). A side rim runs straight ahead,
    so it is widest in the picture at its near corner when it lies on its own side of the camera
    axis, and at its far corner (the far rim's top, rim_z_m, the box's top edge) when it lies across
    the axis.

    shape 'round' (Stage 5 room demo, a round tray of radius tray_half_m, rim top at rim_z_m): a
    side of the box is where the camera's view plane grazes the rim's top, so the centre is
    tray_half_m from that plane's line at z = rim_z_m (cloud check 2026-10-31: the square rule
    put the round tray's centre 1.3-3.1 cm off sideways)."""
    edge = float(cfg['cut_margin_px'])
    accept = float(cfg['accept_m'])
    w, mid = cam.width, cam.width / 2
    fwds, lefts, used, rejected = [], [], 0, 0

    def ground(state, col, row, z):
        return ranging.ground_point(cam, state, (col, row, col, row), z)

    def tangent_left(state, col, y0, y1, fwd, sign):
        a, b = ground(state, col, y1, rim_z_m), ground(state, col, y0, rim_z_m)
        if a is None or b is None or math.dist(a, b) < 1e-3:
            return None
        ux, uy = (b[0] - a[0]) / math.dist(a, b), (b[1] - a[1]) / math.dist(a, b)
        nx, ny = -uy, ux
        if abs(ny) < 0.2:
            return None
        both = [a[1] + (k * tray_half_m - (fwd - a[0]) * nx) / ny for k in (-1.0, 1.0)]
        return min(both) if sign < 0 else max(both)       # the centre is inside the grazed side

    for state, boxes in looks:
        trays = [b for b in boxes if b['cls'] == 'tray']
        best = None
        for b in sorted(trays, key=lambda b: -(b['xyxy'][2] - b['xyxy'][0]) * (b['xyxy'][3] - b['xyxy'][1])):
            x0, y0, x1, y1 = b['xyxy']
            near = ground(state, (x0 + x1) / 2, y1, near_z_m)
            if near is None:
                continue
            fwd = near[0] + tray_half_m
            sides = []
            for col, sign in ((x0, -1.0), (x1, 1.0)):      # left side of the image = the tray's +y rim
                if (sign < 0 and col <= edge) or (sign > 0 and col >= w - edge):
                    continue                                 # cut by the image edge
                if shape == 'round':
                    left = tangent_left(state, col, y0, y1, fwd, sign)
                    if left is not None:
                        sides.append(left)
                    continue
                same_side = col < mid if sign < 0 else col > mid
                if not same_side and y0 <= edge:
                    continue                                 # its far corner is above the picture
                p = ground(state, col, y1, near_z_m) if same_side else ground(state, col, y0, rim_z_m)
                if p is not None:
                    sides.append(p[1] + sign * tray_half_m)
            if sides:
                left = sum(sides) / len(sides)
            else:
                # both sides cut: the near rim spans the picture, so the centre is in that band
                lo, hi = ground(state, 0.0, y1, near_z_m), ground(state, float(w), y1, near_z_m)
                band = (lo[1] - tray_half_m, hi[1] + tray_half_m) if lo and hi else (-tray_half_m, tray_half_m)
                left = max(min(band), min(max(band), expected[1]))
            if math.hypot(fwd - expected[0], left - expected[1]) > accept:
                rejected += 1
                continue
            best = (fwd, left, len(sides))
            break
        if best is not None:
            fwds.append(best[0])
            lefts.append(best[1])
            used += 1
    out = {'frames': len(looks), 'used': used, 'rejected': rejected, 'expected': list(expected)}
    if used:
        fwds.sort()
        lefts.sort()
        out.update(centre=[fwds[len(fwds) // 2], lefts[len(lefts) // 2]], source='yolo')
    else:
        out.update(centre=list(expected), source='odometry')
    return out


def drop_walk(centre, d: float) -> dict:
    """From the tray centre (forward, left): the bearing to turn to and the distance to walk so the
    trunk ends d from the centre on the line towards it."""
    f, l = centre
    r = math.hypot(f, l)
    return {'bearing': math.atan2(l, f), 'distance': max(0.0, r - d), 'range_m': r}
