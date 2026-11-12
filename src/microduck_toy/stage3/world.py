"""The brain's picture of the field: where it is, what it has seen, and how to walk around it.

    pose = FieldPose(start=(x, y, yaw), odom0=odom_of(state))   # the known start (plan question 1A)
    x, y, yaw = pose.of(state)                                  # odometry -> field
    xy = pose.point(state, (forward, left))                     # a level-trunk point -> field
    seen = Sightings(cluster_m=0.15)                            # one entry per toy-sized spot
    route = plan_route((x, y), goal, obstacles, field)           # waypoints around known toys

Field frame = the arena's world frame (configs/control/stage1.json arena: walls at +-half_size_m,
tray centre on the +x side). Only robot.state goes in; nothing here reads the simulator, so the same
code runs on the board.

Odometry: robotd's contact odometry (robot.state odom: position and yaw) counts from wherever robotd
started. At the episode start the brain is told its field pose (plan 3, question 1A: "like a robot
leaving its charging dock") and ties that pose to the odometry reading of that moment; every later
pose is the odometry change since then, turned into the field frame.

Routes: a visibility graph over points around each obstacle (circles: known toys, a ToF stop) and
around the tray (a square grown by tray_inflate_m), kept inside the walls by keep_in_m, solved with
Dijkstra. Small on purpose: at most five toys and one tray are ever known (plan 3, GOTO and CARRY).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import math


def wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def odom_of(state: dict) -> tuple[float, float, float]:
    o = state['odom']
    return float(o['position'][0]), float(o['position'][1]), float(o['yaw'])


class FieldPose:
    """Odometry -> field, anchored on the known start pose."""

    def __init__(self, start, odom0):
        self.start = tuple(float(v) for v in start)
        self.odom0 = tuple(float(v) for v in odom0)
        self.turn = wrap(self.start[2] - self.odom0[2])

    def of_odom(self, odom) -> tuple[float, float, float]:
        dx, dy = odom[0] - self.odom0[0], odom[1] - self.odom0[1]
        c, s = math.cos(self.turn), math.sin(self.turn)
        return (self.start[0] + c * dx - s * dy, self.start[1] + s * dx + c * dy,
                wrap(self.start[2] + odom[2] - self.odom0[2]))

    def of(self, state: dict) -> tuple[float, float, float]:
        return self.of_odom(odom_of(state))

    def point(self, state: dict, body_xy) -> tuple[float, float]:
        """A point in the level trunk frame (forward, left) -> field."""
        return to_field(self.of(state), body_xy)

    def body(self, state: dict, xy) -> tuple[float, float]:
        """A field point -> (forward, left) in the level trunk frame."""
        return to_body(self.of(state), xy)

    def reanchor(self, state: dict, body_xy, field_xy) -> tuple[float, float]:
        """Shift the field position (not the heading) so the landmark seen at body_xy lies at its
        known field_xy: the tray puts back the odometry a fall threw off. Returns the shift."""
        fx, fy = self.point(state, body_xy)
        dx, dy = field_xy[0] - fx, field_xy[1] - fy
        self.start = (self.start[0] + dx, self.start[1] + dy, self.start[2])
        return dx, dy


def to_field(pose, body_xy) -> tuple[float, float]:
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    f, l = body_xy
    return x + c * f - s * l, y + s * f + c * l


def to_body(pose, xy) -> tuple[float, float]:
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = xy[0] - x, xy[1] - y
    return c * dx + s * dy, -s * dx + c * dy


def bearing_to(pose, xy) -> float:
    """Angle (rad, left > 0) from the robot's heading to a field point."""
    return wrap(math.atan2(xy[1] - pose[1], xy[0] - pose[0]) - pose[2])


# ── what has been seen ──────────────────────────────────────────────────────

@dataclass
class Spot:
    """One toy-sized place: the detections of one class that landed within cluster_m of it."""
    cls: str
    x: float
    y: float
    weight: float = 0.0
    frames: int = 0
    looks: int = 0               # standing looks that included it
    nearest_m: float = math.inf  # closest the robot was when it was seen
    status: str = 'seen'         # seen / target / distractor / dropped
    misses: int = 0              # standing looks at it that did not see the target
    id: int = 0

    @property
    def xy(self) -> tuple[float, float]:
        return self.x, self.y

    def add(self, xy, weight: float, distance: float) -> None:
        total = self.weight + weight
        self.x = (self.x * self.weight + xy[0] * weight) / total
        self.y = (self.y * self.weight + xy[1] * weight) / total
        self.weight = total
        self.frames += 1
        self.nearest_m = min(self.nearest_m, distance)

    def as_dict(self) -> dict:
        return {'id': self.id, 'cls': self.cls, 'xy': [round(self.x, 4), round(self.y, 4)],
                'frames': self.frames, 'looks': self.looks, 'weight': round(self.weight, 3),
                'nearest_m': None if math.isinf(self.nearest_m) else round(self.nearest_m, 3),
                'status': self.status, 'misses': self.misses}


@dataclass
class Sightings:
    """The sighting table (plan 3, LOOK): one Spot per class and place."""
    cluster_m: float = 0.15
    spots: list = field(default_factory=list)

    def add(self, cls: str, xy, weight: float, distance: float) -> Spot:
        best, best_d = None, self.cluster_m
        for s in self.spots:
            if s.cls != cls or s.status == 'dropped':
                continue
            d = math.dist(s.xy, xy)
            if d <= best_d:
                best, best_d = s, d
        if best is None:
            best = Spot(cls, float(xy[0]), float(xy[1]), id=len(self.spots))
            self.spots.append(best)
        best.add(xy, weight, distance)
        return best

    def of_class(self, cls: str, statuses=('seen', 'target')) -> list[Spot]:
        return [s for s in self.spots if s.cls == cls and s.status in statuses]

    def near(self, xy, radius: float, classes=None) -> list[Spot]:
        return [s for s in self.spots if s.status != 'dropped' and math.dist(s.xy, xy) <= radius
                and (classes is None or s.cls in classes)]

    def obstacles(self, target_cls: str, skip: Spot | None = None) -> list[tuple[float, float]]:
        """Toy places to walk around: every non-target spot, every rejected target-class spot."""
        out = []
        for s in self.spots:
            if s is skip or s.status == 'dropped':
                continue
            if s.cls != target_cls or s.status == 'distractor':
                out.append(s.xy)
        return out


# ── routes ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Field:
    half_m: float                 # walls at +-half_m
    keep_in_m: float              # the trunk stays this far from the walls
    tray_centre: tuple
    tray_half_m: float            # half the tray side
    tray_inflate_m: float         # extra clearance around the tray
    clear_m: float                # clearance around a known toy
    tray_exit_m: float = 0.0      # > 0 (Stage 5 v2): a start in the tray box steps out first (plan_route)

    @classmethod
    def from_config(cls, stage1: dict, stage3: dict) -> 'Field':
        arena, route = stage1['arena'], stage3['route']
        return cls(half_m=float(arena['half_size_m']), keep_in_m=float(route['keep_in_m']),
                   tray_centre=tuple(float(v) for v in arena['tray']['centre_m']),
                   tray_half_m=float(arena['tray']['size_m']) / 2,
                   tray_inflate_m=float(route['tray_inflate_m']), clear_m=float(route['clear_m']),
                   tray_exit_m=float(route.get('tray_exit_m', 0.0)))

    def inside(self, p, margin: float = 0.0) -> bool:
        lim = self.half_m - self.keep_in_m + margin
        return abs(p[0]) <= lim + 1e-9 and abs(p[1]) <= lim + 1e-9

    def tray_box(self, extra: float = 0.0) -> tuple[float, float, float, float]:
        cx, cy = self.tray_centre
        h = self.tray_half_m + self.tray_inflate_m + extra
        return cx - h, cy - h, cx + h, cy + h


def seg_point_dist(a, b, p) -> float:
    ax, ay = a
    bx, by = b
    vx, vy = bx - ax, by - ay
    n = vx * vx + vy * vy
    t = 0.0 if n < 1e-12 else max(0.0, min(1.0, ((p[0] - ax) * vx + (p[1] - ay) * vy) / n))
    return math.hypot(ax + t * vx - p[0], ay + t * vy - p[1])


def seg_hits_box(a, b, box) -> bool:
    """Does segment a-b cross the open axis-aligned box (x0, y0, x1, y1)? (Liang-Barsky)"""
    x0, y0, x1, y1 = box
    dx, dy = b[0] - a[0], b[1] - a[1]
    lo, hi = 0.0, 1.0
    for p, q in ((-dx, a[0] - x0), (dx, x1 - a[0]), (-dy, a[1] - y0), (dy, y1 - a[1])):
        if abs(p) < 1e-12:
            if q <= 0:
                return False
            continue
        t = q / p
        if p < 0:
            lo = max(lo, t)
        else:
            hi = min(hi, t)
        if lo >= hi:
            return False
    return hi - lo > 1e-9


def in_box(p, box) -> bool:
    return box[0] < p[0] < box[2] and box[1] < p[1] < box[3]


def segment_clear(a, b, circles, boxes, fld: Field) -> bool:
    if not (fld.inside(a, 0.02) and fld.inside(b, 0.02)):
        return False
    for c, r in circles:
        if seg_point_dist(a, b, c) < r - 1e-9:
            return False
    return not any(seg_hits_box(a, b, box) for box in boxes)


def tray_exit(start, fld: Field) -> tuple[float, float] | None:
    """Where a start inside the tray box (tray half + tray_inflate_m) leaves it: straight away from
    the tray centre to fld.tray_exit_m outside the box; if that point is outside the keep-in square
    (the +x side is 0.2 m from the wall), straight out sideways along the axis closest to that
    direction, kept in the square. None if no way out stays in the field."""
    cx, cy = fld.tray_centre
    sx, sy = float(start[0]), float(start[1])
    h = fld.tray_half_m + fld.tray_inflate_m + fld.tray_exit_m
    dx, dy = sx - cx, sy - cy
    if math.hypot(dx, dy) > 1e-6:
        m = max(abs(dx), abs(dy))
        p = (cx + dx / m * h, cy + dy / m * h)          # on the square of half h, away from the centre
        if fld.inside(p):
            return p
    lim = fld.half_m - fld.keep_in_m
    box = fld.tray_box()
    axes = [(-1.0, 0.0), (0.0, 1.0), (0.0, -1.0), (1.0, 0.0)]
    axes.sort(key=lambda a: -(a[0] * dx + a[1] * dy))
    for ux, uy in axes:
        p = (cx + ux * h, sy) if ux else (sx, cy + uy * h)
        p = (max(-lim, min(lim, p[0])), max(-lim, min(lim, p[1])))
        if not in_box(p, box):
            return p
    return None


def plan_route(start, goal, obstacles, fld: Field, avoid_tray: bool = True,
               ring: int = 8) -> list[tuple[float, float]] | None:
    """Waypoints from start to goal (start excluded, goal included), or None when no route.

    obstacles: field points of known toys (circles of fld.clear_m) or [(x, y), r] pairs. An obstacle
    (or the tray) that already contains the start or the goal is left out: the robot may leave a
    place it is standing in, and may stop next to the toy it is walking to. A start outside the
    keep-in square (odometry after a fall, cloud 3D) first steps straight back into it. With
    fld.tray_exit_m > 0 (Stage 5 v2) a start inside the tray box first steps straight out of it
    (tray_exit) and the rest of the route goes round the tray: left out, the line from a drop spot
    to the next target crossed the tray (board run 20261112T025144Z, the bus kicked out).
    """
    if not fld.inside(start, 0.02):
        lim = fld.half_m - fld.keep_in_m
        entry = (max(-lim, min(lim, float(start[0]))), max(-lim, min(lim, float(start[1]))))
        rest = plan_route(entry, goal, obstacles, fld, avoid_tray, ring)
        return None if rest is None else [entry] + rest
    if avoid_tray and fld.tray_exit_m > 0:
        box = fld.tray_box()
        if in_box(start, box) and not in_box(goal, box):
            out = tray_exit(start, fld)
            if out is not None:
                rest = plan_route(out, goal, obstacles, fld, avoid_tray, ring)
                return None if rest is None else [out] + rest
    circles = []
    for o in obstacles:
        (c, r) = (o if isinstance(o[0], (tuple, list)) else (o, fld.clear_m))
        c = (float(c[0]), float(c[1]))
        if math.dist(c, start) < r or math.dist(c, goal) < r:
            continue
        circles.append((c, float(r)))
    boxes = []
    if avoid_tray:
        box = fld.tray_box()
        if not (in_box(start, box) or in_box(goal, box)):
            boxes.append(box)
    start, goal = tuple(map(float, start)), tuple(map(float, goal))
    if segment_clear(start, goal, circles, boxes, fld):
        return [goal]
    nodes = [start, goal]
    pad = 1.0 / math.cos(math.pi / ring)       # a ring polygon's edges stay outside the circle
    for c, r in circles:
        rr = (r + 0.02) * pad
        for k in range(ring):
            a = 2 * math.pi * k / ring + math.pi / ring
            nodes.append((c[0] + rr * math.cos(a), c[1] + rr * math.sin(a)))
    for box in boxes:
        x0, y0, x1, y1 = box
        e = 0.03
        nodes += [(x0 - e, y0 - e), (x0 - e, y1 + e), (x1 + e, y0 - e), (x1 + e, y1 + e)]
    ok = [n for n in nodes[2:] if fld.inside(n) and all(math.dist(n, c) >= r for c, r in circles)
          and not any(in_box(n, b) for b in boxes)]
    nodes = nodes[:2] + ok
    best = {0: 0.0}
    prev: dict[int, int] = {}
    heap = [(0.0, 0)]
    done = set()
    while heap:
        d, i = heapq.heappop(heap)
        if i in done:
            continue
        done.add(i)
        if i == 1:
            break
        for j in range(len(nodes)):
            if j in done or j == i:
                continue
            if not segment_clear(nodes[i], nodes[j], circles, boxes, fld):
                continue
            nd = d + math.dist(nodes[i], nodes[j])
            if nd < best.get(j, math.inf) - 1e-12:
                best[j] = nd
                prev[j] = i
                heapq.heappush(heap, (nd, j))
    if 1 not in prev:
        return None
    path, i = [], 1
    while i != 0:
        path.append(nodes[i])
        i = prev[i]
    return path[::-1]


def route_length(start, route) -> float:
    total, p = 0.0, start
    for q in route:
        total += math.dist(p, q)
        p = q
    return total


def approach_point(robot_xy, toy_xy, distance: float, obstacles, fld: Field,
                   corridor_m: float, step_deg: float = 30.0) -> tuple[float, float] | None:
    """Where the B segment should start: `distance` from the toy, on the side the robot comes from
    if that is clear, else the nearest direction (in step_deg steps) whose start point is inside
    the field, off the tray, and whose straight line to the toy passes no known toy closer than
    corridor_m. (The B segment walks straight at the toy from there.)"""
    base = math.atan2(robot_xy[1] - toy_xy[1], robot_xy[0] - toy_xy[0])
    box = fld.tray_box()
    for k in range(int(180 / step_deg) + 1):
        for sign in ((1,) if k == 0 else (1, -1)):
            a = base + sign * math.radians(k * step_deg)
            p = (toy_xy[0] + distance * math.cos(a), toy_xy[1] + distance * math.sin(a))
            if not fld.inside(p) or in_box(p, box) or seg_hits_box(p, toy_xy, fld.tray_box(-fld.tray_inflate_m)):
                continue
            if any(seg_point_dist(p, toy_xy, o) < corridor_m for o in obstacles):
                continue
            return p
    return None
