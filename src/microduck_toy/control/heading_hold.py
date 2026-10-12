# Moved 2026-10-11 from work/heading_hold.py (original kept unchanged in archive/2026-10_ball_follow/work/); only paths and imports changed.
"""Relative-heading feedback from gyro + gravity; no simulator pose input.

The wrapper keeps the user's requested velocity separate from the corrected
13D command supplied to the unchanged ONNX policy.
"""
from dataclasses import dataclass
import math


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_rate_from_imu(gyro, gravity):
    """ZYX Euler yaw rate from body gyro and unit downward gravity vector."""
    if len(gyro) != 3 or len(gravity) != 3:
        raise ValueError('Expected three gyro and three gravity components')
    values = [float(v) for v in (*gyro, *gravity)]
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Non-finite IMU observation')
    norm = math.sqrt(sum(v*v for v in gravity))
    if norm < 0.1:
        raise ValueError('Invalid gravity direction')
    gx, gy, gz = (float(v)/norm for v in gravity)
    denominator = gy*gy + gz*gz
    if denominator < 0.25 or -gz < 0.5:
        raise ValueError('Heading estimate unavailable at large tilt')
    return -(gy*float(gyro[1]) + gz*float(gyro[2]))/denominator


@dataclass(frozen=True)
class HeadingConfig:
    kp: float = 1.0
    ki: float = 0.0
    max_correction: float = 0.3
    integral_limit: float = 0.8
    min_forward_speed: float = 0.05
    period_s: float = 0.02

    def __post_init__(self):
        values = (self.kp, self.ki, self.max_correction, self.integral_limit,
                  self.min_forward_speed, self.period_s)
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError('Heading configuration must contain finite numbers')
        if self.kp < 0 or self.ki < 0 or self.min_forward_speed < 0:
            raise ValueError('Heading gains and minimum speed must be non-negative')
        if self.max_correction <= 0 or self.integral_limit <= 0 or self.period_s <= 0:
            raise ValueError('Heading limits and time period must be positive')


class HeadingHold:
    def __init__(self, config=None):
        self.config = config or HeadingConfig()
        self.heading = 0.0
        self.reference = None
        self.integral = 0.0
        self.active = False
        self.direction = 0
        self.error = 0.0
        self.effective_yaw = 0.0
        self.rate = 0.0
        self.status = 'idle'

    def release(self, status='idle'):
        self.reference = None
        self.integral = 0.0
        self.active = False
        self.error = 0.0
        self.direction = 0
        self.status = status

    def update(self, gyro, gravity, dt, requested, permitted=True):
        vx, vy, yaw = (float(v) for v in requested)
        if not all(math.isfinite(v) for v in (vx, vy, yaw)):
            raise ValueError('Non-finite requested velocity')
        self.effective_yaw = yaw
        cfg = self.config
        if not math.isfinite(dt) or dt <= 0.0:
            self.release('invalid_timestamp')
            return yaw
        if dt > 1.5*cfg.period_s:
            # Inference may have been paused while physics kept running. The
            # lost gyro history cannot recover the old heading: start afresh.
            self.heading = 0.0
            self.release('resume')
            dt = cfg.period_s
        try:
            self.rate = yaw_rate_from_imu(gyro, gravity)
        except ValueError:
            self.release('invalid_imu')
            return yaw
        self.heading += self.rate*dt
        straight = (permitted and abs(vx) >= cfg.min_forward_speed
                    and abs(vy) < 1e-6 and abs(yaw) < 1e-6)
        if not straight:
            self.release('manual_or_idle')
            return yaw
        direction = 1 if vx > 0.0 else -1
        if not self.active or self.direction != direction:
            self.reference = self.heading
            self.integral = 0.0
            self.active = True
            self.direction = direction
        self.error = wrap_angle(self.reference - self.heading)
        candidate = max(-cfg.integral_limit,
                        min(cfg.integral_limit, self.integral + self.error*dt))
        raw = cfg.kp*self.error + cfg.ki*candidate
        if abs(raw) <= cfg.max_correction or raw*self.error <= 0.0:
            self.integral = candidate
        raw = cfg.kp*self.error + cfg.ki*self.integral
        self.effective_yaw = max(-cfg.max_correction, min(cfg.max_correction, raw))
        self.status = 'holding'
        return self.effective_yaw


def make_heading_policy(base_class, config=None, log_status=True):
    """Build a subclass around the pinned replay implementation at runtime."""
    class HeadingPolicy(base_class):
        def __init__(self, *args, **kwargs):
            self.heading_hold = HeadingHold(config)
            self._heading_last_time = None
            self._heading_steps = 0
            super().__init__(*args, **kwargs)

        def infer(self):
            now = float(self.data.time)
            period = self.heading_hold.config.period_s
            dt = period if self._heading_last_time is None else now-self._heading_last_time
            self._heading_last_time = now
            # Only the IMU/gravity slots of the policy's existing observation
            # are consumed. No world position or world yaw is read here.
            obs = self.get_observations()
            permitted = (self.new_cmd_obs and self.use_projected_gravity
                         and self.current_policy == 'walking'
                         and not self.ground_pick_mode and not self.sit_mode
                         and not self.slope_mode and self.behavior_mode is None)
            corrected_yaw = self.heading_hold.update(
                obs[:3], obs[3:6], dt, self.vel_cmd, permitted)
            self._update_command()
            if self.heading_hold.active:
                self.command[2] = corrected_yaw
            self._heading_steps += 1
            if log_status and self._heading_steps % 50 == 0:
                state = self.heading_hold
                print(f'[航向保持] {state.status} '
                      f'偏离={math.degrees(-state.error):+.2f}° '
                      f'用户转向={float(self.vel_cmd[2]):+.3f} '
                      f'实际发送={float(self.command[2]):+.3f} rad/s', flush=True)
            return super().infer()

    return HeadingPolicy
