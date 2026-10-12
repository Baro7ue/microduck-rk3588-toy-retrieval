"""Bounded visual ball following, independent of transport and robot control."""
import math
from statistics import median

from ball_follow_camera import body_target_error, normalize_quat, predict_box

DEFAULT_STOP_AREA = .012  # Fraction of the full, rotated camera image.


def _finite(value):
    try:
        return not isinstance(value, bool) and math.isfinite(value)
    except (TypeError, ValueError):
        return False


def _box_area(box):
    return (box[2]-box[0])*(box[3]-box[1])


def _box_iou(first, second):
    intersection = (max(0, min(first[2], second[2])-max(first[0], second[0])) *
                    max(0, min(first[3], second[3])-max(first[1], second[1])))
    return intersection / (_box_area(first)+_box_area(second)-intersection)


def _box_near(first, second):
    if _box_iou(first, second) >= .1:
        return True
    width = max(first[2]-first[0], second[2]-second[0])
    height = max(first[3]-first[1], second[3]-second[1])
    dx = abs((first[0]+first[2])-(second[0]+second[2]))/2
    dy = abs((first[1]+first[3])-(second[1]+second[3]))/2
    # Gait head motion can move a far ball beyond box overlap in either image axis.
    return dx <= 1.5*width and dy <= 1.5*height


class FollowControl:
    def __init__(self, started_at, duration=30.0, max_age=.3, speed=.1,
                 stop_area=DEFAULT_STOP_AREA, orange_continuation=False,
                 camera_motion_compensation=False, body_steering=False):
        if not isinstance(orange_continuation, bool):
            raise ValueError('orange_continuation must be True or False')
        if (not isinstance(camera_motion_compensation, bool) or
                camera_motion_compensation and not orange_continuation):
            raise ValueError('camera_motion_compensation must be boolean and requires orange_continuation')
        if (not isinstance(body_steering, bool) or
                body_steering and not camera_motion_compensation):
            raise ValueError('body_steering must be boolean and requires camera_motion_compensation')
        if (not all(_finite(x) for x in (started_at, duration, max_age, speed, stop_area)) or
                not 0 < duration <= 30 or not 0 < max_age <= .3 or
                speed not in (.1, .3) or not 0 < stop_area <= 1):
            raise ValueError('finite started_at, 0<duration<=30, 0<max_age<=0.3, speed=0.1|0.3, 0<stop_area<=1 required')
        self.started_at = float(started_at)
        self.duration = float(duration)
        self.max_age = float(max_age)
        self.speed = float(speed)
        self.stop_area = float(stop_area)
        self.orange_continuation = orange_continuation
        self.camera_motion_compensation = camera_motion_compensation
        self.body_steering = body_steering
        self.reason = None
        self.reached = False
        self.moving_started_at = None
        self.last_frame_at = None
        self.last_frame_index = None
        self.baseline_area = None
        self._startup_areas = []
        self._error = None
        self._area = None
        self._last_box = None
        self._missing_since = None
        self._absence_reason = None
        self._source_class_id = None
        self._association = None
        self._last_camera_quat = None
        self._predicted_center = None
        self._image_error = None
        self._body_bearing = None

    @property
    def target_area(self):
        return self.stop_area

    def stop(self, reason):
        if self.reason is None:
            self.reason = str(reason) if reason else 'stopped'
            self.reached = self.reason == 'target_area_reached'

    def command(self, now):
        if self.reason is not None:
            return (0.0, 0.0, 0.0)
        if not _finite(now) or now < self.started_at:
            self.stop('invalid_time')
        elif now >= self.started_at + self.duration:
            self.stop('duration_limit')
        elif self._missing_since is not None and now >= self._missing_since + self.max_age:
            self.stop(self._absence_reason)
        elif self.last_frame_at is None:
            if now >= self.started_at + 5:
                self.stop('startup_timeout')
        elif now > self.last_frame_at + self.max_age:
            self.stop('frame_timeout')
        if (self.reason is not None or self.baseline_area is None or
                self._missing_since is not None):
            return (0.0, 0.0, 0.0)
        # Include the mathematical boundary despite binary floating-point rounding.
        yaw = (0.0 if abs(self._error) <= .03 or
               math.isclose(abs(self._error), .03, rel_tol=0, abs_tol=1e-12)
               else max(-.2, min(.2, -.6 * self._error)))
        return (self.speed, 0.0, yaw)

    def _record(self, now):
        vx, vy, vyaw = self.command(now)
        record = {'vx': vx, 'vy': vy, 'vyaw': vyaw,
                'mode': ('stopped' if self.reason is not None else
                         'holding' if self._missing_since is not None else
                         'acquiring' if self.baseline_area is None else 'following'),
                'e': self._error, 'area': self._area, 'start_area': self.baseline_area,
                'stop_area': self.target_area,
                'reason': self.reason}
        if self.orange_continuation:
            record.update(source_class_id=self._source_class_id, association=self._association)
        if self.camera_motion_compensation:
            record.update(camera_motion_compensation=True, predicted_center=self._predicted_center)
        if self.body_steering:
            record.update(body_steering=True, image_error=self._image_error,
                          body_bearing_rad=self._body_bearing)
        return record

    def _hold_absence(self, received_at, now, receiver_index, target=None, reason=None):
        self.last_frame_at, self.last_frame_index = float(received_at), receiver_index
        if self._missing_since is None:
            self._missing_since = float(now)
            self._absence_reason = reason or ('clipped_target' if target is not None else 'missing_ball')
        self._error = self._area = None
        self._image_error = self._body_bearing = None
        self._source_class_id = target['class_id'] if target is not None else None
        self._association = ('camera_pose_hold' if reason == 'camera_pose_unavailable' else
                             'clipped_hold' if target is not None else 'missing_hold')
        return self._record(now)

    def observe(self, detections, received_at, now, receiver_index, camera_quat=None,
                trunk_quat=None):
        # Check the previous frame before accepting a new one; a long gap cannot be concealed.
        self.command(now)
        if self.reason is not None:
            return self._record(now)
        if not _finite(received_at):
            self.stop('invalid_time')
        elif received_at > now:
            self.stop('future_frame')
        elif now > received_at + self.max_age:
            self.stop('stale_detection')
        elif isinstance(receiver_index, bool) or not isinstance(receiver_index, int) or receiver_index < 1:
            self.stop('invalid_frame_index')
        elif not isinstance(detections, (list, tuple)):
            self.stop('invalid_detection')
        if self.reason is not None:
            return self._record(now)

        balls, candidates = [], []
        for target in detections:
            if not isinstance(target, dict):
                self.stop('invalid_detection')
                break
            class_id, confidence, box = (target.get('class_id'), target.get('confidence'),
                                         target.get('xyxy'))
            if (isinstance(class_id, bool) or not isinstance(class_id, int) or not 0 <= class_id < 80 or
                    not _finite(confidence) or not 0 <= confidence <= 1 or
                    not isinstance(box, (list, tuple)) or len(box) != 4 or
                    not all(_finite(x) for x in box)):
                self.stop('invalid_detection')
                break
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= 360 and 0 <= y1 < y2 <= 640):
                self.stop('invalid_detection')
                break
            if class_id == 32:
                balls.append(target)
            if class_id in (32, 49):
                candidates.append(target)
        if self.reason is not None:
            return self._record(now)
        if (self.orange_continuation and self.last_frame_index is not None and
                receiver_index <= self.last_frame_index):
            return self._record(now)

        established = self.orange_continuation and self.baseline_area is not None
        predicted_box, current_quat, compensated_only = None, None, False
        current_trunk_quat = None
        if self.camera_motion_compensation:
            self._predicted_center = None
            if len(balls) > 1:
                self.stop('multiple_balls')
                return self._record(now)
            if camera_quat is None:
                return self._hold_absence(received_at, now, receiver_index,
                    balls[0] if balls else None, reason='camera_pose_unavailable')
            try:
                current_quat = normalize_quat(camera_quat)
            except ValueError:
                self.stop('invalid_camera_pose')
                return self._record(now)
            if self.body_steering:
                if trunk_quat is None:
                    return self._hold_absence(received_at, now, receiver_index,
                        balls[0] if balls else None, reason='camera_pose_unavailable')
                try:
                    current_trunk_quat = normalize_quat(trunk_quat)
                except ValueError:
                    self.stop('invalid_camera_pose')
                    return self._record(now)
            if established:
                try:
                    predicted_box = predict_box(self._last_box, self._last_camera_quat, current_quat)
                except ValueError:
                    self.stop('invalid_camera_projection')
                    return self._record(now)
                self._predicted_center = [(predicted_box[0]+predicted_box[2])/2,
                                          (predicted_box[1]+predicted_box[3])/2]

        def near_reference(box):
            return (_box_near(box, self._last_box) or
                    predicted_box is not None and _box_near(box, predicted_box))

        if established:
            if len(balls) > 1:
                self.stop('multiple_balls')
                return self._record(now)
            if balls:
                # Preserve the original detector's unique sports-ball semantics. Camera
                # motion need not scale with a far-away ball's small image footprint.
                target = balls[0]
                separate_local_oranges = [candidate for candidate in candidates
                                          if candidate['class_id'] == 49 and
                                          near_reference(candidate['xyxy']) and
                                          _box_iou(candidate['xyxy'], target['xyxy']) < .7]
                if separate_local_oranges:
                    self.stop('ambiguous_target')
                    return self._record(now)
            elif not candidates:
                return self._hold_absence(received_at, now, receiver_index)
            else:
                matches = [candidate for candidate in candidates
                           if near_reference(candidate['xyxy'])]
                if len(matches) != 1:
                    self.stop('target_jump' if not matches else 'ambiguous_target')
                    return self._record(now)
                target = matches[0]
                compensated_only = not _box_near(target['xyxy'], self._last_box)
        else:
            if len(balls) != 1:
                self.stop('missing_ball' if not balls else 'multiple_balls')
                return self._record(now)
            target = balls[0]

        if target['confidence'] < .25:
            self.stop('low_confidence')
        if self.orange_continuation and self.reason is None:
            x1, y1, x2, y2 = target['xyxy']
            if not (2 <= x1 < x2 <= 358 and 2 <= y1 < y2 <= 638):
                if established:
                    # A partial frame gives neither a reliable size nor a full target box.
                    return self._hold_absence(received_at, now, receiver_index, target)
                self.stop('clipped_target')
        if established and self.reason is None:
            ratio = _box_area(target['xyxy']) / _box_area(self._last_box)
            if not .5 <= ratio <= 2:
                self.stop('target_scale_jump')
        if self.reason is not None:
            return self._record(now)
        if self.last_frame_index is not None and receiver_index <= self.last_frame_index:
            return self._record(now)

        x1, y1, x2, y2 = target['xyxy']
        image_error = ((x1+x2)/2-180)/180
        if self.body_steering:
            try:
                error, bearing = body_target_error(target['xyxy'], current_quat,
                                                   current_trunk_quat)
            except ValueError:
                self.stop('invalid_target_direction')
                return self._record(now)
            self._error, self._body_bearing = error, bearing
            self._image_error = image_error
        else:
            self._error = image_error
        self._area = (x2-x1)*(y2-y1)/(360*640)
        self.last_frame_at, self.last_frame_index = float(received_at), receiver_index
        if self.orange_continuation:
            self._last_box = list(target['xyxy'])
            self._missing_since = None
            self._absence_reason = None
            self._source_class_id = target['class_id']
            self._association = ('sports_ball_acquisition' if not established else
                                 'sports_ball_detection' if target['class_id'] == 32 else
                                 'camera_motion_continuation' if compensated_only else
                                 'spatial_continuation')
            if self.camera_motion_compensation:
                self._last_camera_quat = current_quat
        # Stop at the same visual size from every starting point, including the first frame.
        if self._area >= self.target_area:
            self.stop('target_area_reached')
            return self._record(now)
        if self.baseline_area is None:
            self._startup_areas.append(self._area)
            if len(self._startup_areas) == 5:
                self.baseline_area = median(self._startup_areas)
        if self.baseline_area is not None and self.moving_started_at is None:
            self.moving_started_at = float(now)
        return self._record(now)
