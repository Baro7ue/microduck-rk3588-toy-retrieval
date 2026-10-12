# Moved 2026-10-11 from work/ball_follow_camera.py (original kept unchanged in archive/2026-10_ball_follow/work/); only paths and imports changed.
"""Camera-ray geometry for the simulator's upright 360x640 image."""
import math

FOCAL_PX = 434.55844122715706  # Raw 360px height / (2*tan(45deg/2)); rotation preserves focal length.
PRINCIPAL = (180., 320.)


def _numbers(values, length):
    if not isinstance(values, (list, tuple)) or len(values) != length:
        raise ValueError('finite numeric sequence required')
    try:
        if any(isinstance(value, bool) or not math.isfinite(value) for value in values):
            raise ValueError('finite numeric sequence required')
        return tuple(float(value) for value in values)
    except (TypeError, OverflowError) as error:
        raise ValueError('finite numeric sequence required') from error


def normalize_quat(quat):
    """Validate scalar-first near-unit input; normalize only ordinary numeric rounding."""
    quat = _numbers(quat, 4)
    norm = math.hypot(*quat)
    if not .95 <= norm <= 1.05:
        raise ValueError('camera quaternion must have unit norm within 0.05')
    return tuple(value/norm for value in quat)


def _multiply(a, b):
    w, x, y, z = a
    v, i, j, k = b
    return (w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
            w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v)


def _rotate(quat, vector):
    w, x, y, z = quat
    vx, vy, vz = vector
    tx, ty, tz = 2*(y*vz-z*vy), 2*(z*vx-x*vz), 2*(x*vy-y*vx)
    return (vx+w*tx+y*tz-z*ty, vy+w*ty+z*tx-x*tz, vz+w*tz+x*ty-y*tx)


def world_camera_quat(state):
    """Camera→world = trunk→world * camera→trunk; absent legacy poses return None."""
    if state is None:
        return None
    if not isinstance(state, dict):
        raise ValueError('robot state must be an object')
    quats = []
    for path in (('imu', 'quat'), ('frames', 'camera', 'quat')):
        value = state
        for key in path:
            if value is None:
                return None
            if not isinstance(value, dict):
                raise ValueError('camera pose must be an object')
            if key not in value:
                return None
            value = value[key]
        quats.append(normalize_quat(value))
    return normalize_quat(_multiply(*quats))


def body_target_error(box, camera_world_quat, trunk_world_quat):
    """Return horizontal image-equivalent error relative to the trunk's world heading."""
    x1, y1, x2, y2 = _numbers(box, 4)
    if x1 >= x2 or y1 >= y2:
        raise ValueError('positive box dimensions required')
    camera, trunk = normalize_quat(camera_world_quat), normalize_quat(trunk_world_quat)
    ray = (((x1+x2)/2-PRINCIPAL[0])/FOCAL_PX,
           ((y1+y2)/2-PRINCIPAL[1])/FOCAL_PX, 1.)
    world_ray = _rotate(camera, ray)
    forward = _rotate(trunk, (1., 0., 0.))
    horizontal = math.hypot(forward[0], forward[1])
    if (not all(math.isfinite(value) for value in (*world_ray, *forward)) or
            horizontal <= 1e-6):
        raise ValueError('trunk must have a finite horizontal heading')
    if math.hypot(world_ray[0], world_ray[1]) <= 1e-6:
        raise ValueError('target ray must have a horizontal direction')
    fx, fy = forward[0]/horizontal, forward[1]/horizontal
    along = fx*world_ray[0]+fy*world_ray[1]
    left = -fy*world_ray[0]+fx*world_ray[1]
    if not math.isfinite(along) or not math.isfinite(left) or along <= 1e-12:
        raise ValueError('target ray must face forward along the trunk heading')
    error = -FOCAL_PX*left/(PRINCIPAL[0]*along)
    bearing = math.atan2(left, along)
    if not all(math.isfinite(value) for value in (error, bearing)):
        raise ValueError('nonfinite body target direction')
    return error, bearing


def predict_box(last_box, previous_quat, current_quat):
    """Rotate the previous center ray into the current camera, preserving raw box size."""
    x1, y1, x2, y2 = _numbers(last_box, 4)
    if x1 >= x2 or y1 >= y2:
        raise ValueError('positive box dimensions required')
    previous, current = normalize_quat(previous_quat), normalize_quat(current_quat)
    ray = (((x1+x2)/2-PRINCIPAL[0])/FOCAL_PX,
           ((y1+y2)/2-PRINCIPAL[1])/FOCAL_PX, 1.)
    inverse = (current[0], -current[1], -current[2], -current[3])
    rx, ry, rz = _rotate(_multiply(inverse, previous), ray)
    if not all(math.isfinite(value) for value in (rx, ry, rz)) or rz <= 1e-12:
        raise ValueError('previous target ray is behind the current camera')
    cx, cy = PRINCIPAL[0]+FOCAL_PX*rx/rz, PRINCIPAL[1]+FOCAL_PX*ry/rz
    if not math.isfinite(cx) or not math.isfinite(cy):
        raise ValueError('nonfinite camera projection')
    width, height = x2-x1, y2-y1
    result = [cx-width/2, cy-height/2, cx+width/2, cy+height/2]
    if not all(math.isfinite(value) for value in result):
        raise ValueError('nonfinite camera projection box')
    return result
