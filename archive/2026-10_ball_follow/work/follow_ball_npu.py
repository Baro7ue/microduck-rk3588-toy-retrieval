"""RKNN ball detection with the existing bounded visual-follow controller."""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from ball_follow_control import DEFAULT_STOP_AREA
from follow_ball_cpu import run_trial as run_follow
from probe_board_yolo11_npu_camera import NPUCameraDetector


def run_trial(model_path, camera_address, robot_socket, directory, duration=30.0,
              detector=None, speed=.1, stop_area=DEFAULT_STOP_AREA, orange_continuation=False,
              camera_motion_compensation=False, body_steering=False):
    return run_follow(model_path, camera_address, robot_socket, directory,
                      duration=duration, detector=detector, speed=speed, stop_area=stop_area,
                      detector_factory=NPUCameraDetector, backend='npu',
                      orange_continuation=orange_continuation,
                      camera_motion_compensation=camera_motion_compensation,
                      body_steering=body_steering)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path,
        default=Path.home()/'microduck-board/yolo11n-follow-npu/models/yolo11n-rk3588-fp.rknn')
    parser.add_argument('--socket', default=str(Path.home()/'microduck-board/run/robotd.sock'))
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--speed', type=float, choices=(.1, .3), default=.1)
    parser.add_argument('--stop-area', type=float, default=DEFAULT_STOP_AREA,
        help='fixed ball box area / full image area (default: 0.012 = 1.2%%)')
    parser.add_argument('--orange-continuation', action='store_true',
        help='continue an established ball track across spatially matched orange labels')
    parser.add_argument('--camera-motion-compensation', action='store_true',
        help='use paired camera orientation for association in the fixed 640x360 simulation camera')
    parser.add_argument('--body-steering', action='store_true',
        help='steer from horizontal target bearing relative to the paired trunk heading')
    args = parser.parse_args()
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S_%fZ')
    output = args.output_dir or Path.home()/'microduck-board/run/G4'/('ball_follow_npu_'+stamp)
    raise SystemExit(run_trial(args.model, ('192.168.50.1', 7901), args.socket, output,
                               speed=args.speed, stop_area=args.stop_area,
                               orange_continuation=args.orange_continuation,
                               camera_motion_compensation=args.camera_motion_compensation,
                               body_steering=args.body_steering))
