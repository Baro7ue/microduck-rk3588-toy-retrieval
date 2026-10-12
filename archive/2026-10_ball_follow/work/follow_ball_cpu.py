"""One bounded visual approach to the original orange ball; defaults to board CPU."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
import signal
import sys
import threading
import time
import traceback

from ball_follow_control import DEFAULT_STOP_AREA, FollowControl
from ball_follow_rpc import RobotLink

MODEL_SHA = '121662c804b9aa906563235ed2909815f2c23bc6e4052e651e01468817551826'
NPU_MODEL_SHA = '72b202f6b6cebb12f60bd1ae2dceffa706ba19cc66c93955e483645ef8c26a02'


class GuardedDrive:
    """Only this thread writes movement; inference and disk I/O never hold its lock."""
    def __init__(self, robot, control, camera):
        self.robot, self.control, self.camera = robot, control, camera
        self.lock = threading.Lock()
        self.cancel = threading.Event()
        self.done = threading.Event()
        self.error = None
        self.records = []
        self.walk_seen = False
        self.thread = threading.Thread(target=self.run, name='ball-drive', daemon=True)

    def start(self):
        self.thread.start()

    def observe(self, detections, received_at, now, receiver_index, camera_quat=None,
                trunk_quat=None):
        with self.lock:
            return self.control.observe(detections, received_at, now, receiver_index,
                                        camera_quat=camera_quat, trunk_quat=trunk_quat)

    def _send(self, command):
        self.robot.move(*command)
        self.records.append({'sent_monotonic_s': time.monotonic(),
            'vx': command[0], 'vy': command[1], 'vyaw': command[2],
            'receiver_index': self.control.last_frame_index,
            'frame_received_at': self.control.last_frame_at,
            'reason': self.control.reason})

    def run(self):
        try:
            next_send = time.monotonic()
            while not self.cancel.is_set():
                if self.cancel.wait(max(0, next_send-time.monotonic())):
                    break
                self.camera.raise_if_failed()
                if self.robot.error:
                    raise RuntimeError(self.robot.error)
                state, received = self.robot.snapshot()
                now = time.monotonic()
                if state is None or received is None or not 0 <= now-received <= .3:
                    raise RuntimeError('超过 0.3 秒没有推进的身体状态')
                if (state['safety']['fallen'] or state['safety'].get('limp', False)
                        or state['policy'] not in {'stand', 'walk'}):
                    raise RuntimeError('身体摔倒、失去控制或出现非预期策略状态')
                with self.lock:
                    now = time.monotonic()
                    if self.cancel.is_set():
                        self.control.stop('cancelled')
                    if not 0 <= now-received <= .3:
                        raise RuntimeError('发送前身体状态已过期')
                    command = self.control.command(now)
                    if state['policy'] == 'walk':
                        self.walk_seen = True
                    began = self.control.moving_started_at
                    if (began is not None and not self.walk_seen and
                            time.monotonic()-began > 2 and self.control.reason is None):
                        raise RuntimeError('前进后 2 秒内未切换到行走策略')
                    self._send(command)
                    if self.control.reason is not None:
                        break
                next_send += .05
                delay = time.monotonic()-next_send
                if delay > 0:
                    next_send += math.ceil(delay/.05)*.05  # Skip missed slots; never catch up.
        except Exception:
            self.error = traceback.format_exc()
            with self.lock:
                self.control.stop('drive_error')
        finally:
            with self.lock:
                self.control.stop('drive_stopped')
                try:
                    self._send((0.0, 0.0, 0.0))
                except Exception:
                    self.error = (self.error or '') + traceback.format_exc()
            self.done.set()

    def stop(self, reason='stopped'):
        self.cancel.set()
        with self.lock:
            self.control.stop(reason)
        if self.thread.ident is not None:
            self.thread.join(.6)
        if self.thread.is_alive():
            self.robot.close_moves()
            raise RuntimeError('速度发送线程尚未退出；不能确认停止顺序')


def confirm_stand(robot, stopped_at, previous_tick):
    deadline = time.monotonic()+3
    stable = None
    while time.monotonic() < deadline:
        if robot.error:
            raise RuntimeError(robot.error)
        state, received = robot.snapshot()
        now = time.monotonic()
        if received is None or not 0 <= now-received <= .3:
            raise RuntimeError('停止后没有新的身体状态')
        move = state['move']
        settled = (received > stopped_at and state['t_ns'] > previous_tick
            and state['policy'] == 'stand' and not state['safety']['fallen']
            and state['safety']['gravity'][2] <= -.9
            and len(move['requested']) == len(move['applied']) == 3
            and all(math.isfinite(v) and abs(v) <= 1e-8 for v in move['requested'])
            and all(math.isfinite(v) and abs(v) < .01 for v in move['applied']))
        stable = (now if stable is None else stable) if settled else None
        if stable is not None and now-stable >= 1:
            return state
        time.sleep(.02)
    raise TimeoutError('停止后 3 秒内未确认连续 1 秒零速度站立')


def interrupt(_signum, _frame):
    raise KeyboardInterrupt


def run_trial(model_path, camera_address, robot_socket, directory, duration=30.0, detector=None,
              speed=.1, stop_area=DEFAULT_STOP_AREA, detector_factory=None, backend='cpu',
              orange_continuation=False, camera_motion_compensation=False, body_steering=False):
    from probe_board_yolo11_camera import CameraDetector, LatestFrameSlot, save_image
    from probe_runtime_visual_load import FrameWorker

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    print(f'证据目录：{directory}', flush=True)
    report = {'scope': f'board {backend.upper()} vision -> board robotd -> Mac simulated body',
        'outcome': 'not_started', 'duration_limit_s': duration, 'speed_command_m_s': speed,
        'stop_rule': 'fixed_image_area_fraction', 'stop_area_fraction': stop_area,
        'orange_continuation': orange_continuation,
        'camera_motion_compensation': camera_motion_compensation,
        'body_steering': body_steering,
        'freshness_limit_s': .3, 'stop_confirmed': False,
        'left_enabled_standing': False,
        'definitions': {'commands': 'Successful socket send, not proof of applied velocity; '
                        'applied velocity and posture are in states.jsonl.',
                        'freshness': 'Local complete-payload receipt and advancing state receipt; '
                        'camera protocol has no source time or source frame ID.',
                        'duration': 'Fixed observation/movement budget after model and robot readiness; '
                        'loading and stop confirmation are outside this budget.',
                        'stop_area': 'Fixed ball box area / full rotated image area, independent of starting size; not a metric distance.',
                        'baseline_area': 'Median of initial five accepted boxes for diagnostics only; never changes the stop threshold.',
                        'left_enabled_standing': 'Confirmed zero-speed balance retained after a completed or controlled-incomplete trial; independent of task completion.'}}
    robot = camera = drive = logic = None
    last_image = None
    detections = []
    success, status = False, 1
    keep_standing = False
    moves_closed = stop_accepted = False
    previous_signal = signal.signal(signal.SIGTERM, interrupt)

    def cleanup_control():
        nonlocal success, status, keep_standing
        keep_standing = False
        if robot is None or drive is None:
            return
        # request opens an independent bounded connection, also after close().
        cleanup = [('robot.enable', {'on': False})]
        if not stop_accepted:
            cleanup.insert(0, ('robot.stop', {}))
        for method, params in cleanup:
            try:
                reply = robot.request(method, params)
                if not reply.get('accepted'):
                    raise RuntimeError(f'{method}: {reply}')
                report[method+'_cleanup'] = reply
            except Exception:
                report[method+'_cleanup_error'] = traceback.format_exc()
                traceback.print_exc()
                print('停止请求未确认；请在终端 B 按 Ctrl+C，保留完整输出。', flush=True)
                success, status = False, 1

    try:
        # Validate the budget before connecting to any robot endpoint.
        if backend not in ('cpu', 'npu'):
            raise ValueError('vision backend must be cpu or npu')
        FollowControl(time.monotonic(), duration=duration, speed=speed, stop_area=stop_area,
                      orange_continuation=orange_continuation,
                      camera_motion_compensation=camera_motion_compensation,
                      body_steering=body_steering)
        factory = detector_factory or CameraDetector
        if detector is None:
            detector = (factory(Path(model_path), include_orange=True) if orange_continuation
                        else factory(Path(model_path)))
        if orange_continuation:
            report['definitions']['orange_continuation'] = (
                'Acquire five sports-ball frames; preserve unique sports-ball semantic detection, '
                'then spatially associate raw orange class49 as continuation; '
                'no acquisition from orange alone; missing detections request zero for at most '
                '0.3s before a latched stop; clipped targets also request zero and cannot '
                'contribute area; ambiguous/jumping targets stop immediately.')
        if camera_motion_compensation:
            report['definitions']['camera_motion_compensation'] = (
                'Association only: raw position OR rotation-predicted position, with the same '
                'spatial gate and ambiguity/scale/confidence checks; raw stop area unchanged. '
                'Pose uses advancing sensor and local receipt times at/before complete-frame arrival, '
                'each within 0.1s. Arrival is an approximate capture-time proxy; no exposure timestamp. '
                'Missing pose joins the shared zero-hold deadline; malformed pose stops without disabling confirmed balance.')
        if body_steering:
            report['definitions']['body_steering'] = (
                'Current box-center camera ray rotated to world, projected onto the horizontal plane, '
                'and compared with trunk-forward horizontal heading from the same paired state. '
                'Image-equivalent error=-focal/180*tan(body bearing); existing -0.6 gain, '
                '0.03 deadband and +/-0.2rad/s limit retained. Raw image_error and body_bearing_rad '
                'are logged separately. No heading PI. Pairing approximates capture time; '
                'current simulation intrinsics must be recalibrated for a physical camera.')
        expected_sha = MODEL_SHA if backend == 'cpu' else NPU_MODEL_SHA
        if detector.model['sha256'] != expected_sha:
            raise ValueError('YOLO 模型哈希不匹配；未发送速度指令')
        report['model'], report['camera_address'] = detector.model, list(camera_address)
        if backend == 'npu':
            report['runtime'] = detector.runtime_info
            report['definitions']['inference_call_ms'] = (
                'Synchronous RKNNLite.inference including runtime/input/output overhead; '
                'excludes image decoding, preprocessing, postprocessing and annotation.')
        robot = RobotLink(robot_socket, directory)
        robot.start()
        report['initial_state'] = robot.wait_ready()
        report['models'] = robot.models
        report['health_start'] = robot.request('robot.health', {})
        slot = LatestFrameSlot()
        camera = FrameWorker(camera_address, directory, on_frame=slot.publish)
        camera.start()
        logic = FollowControl(time.monotonic(), duration=duration, speed=speed, stop_area=stop_area,
                              orange_continuation=orange_continuation,
                              camera_motion_compensation=camera_motion_compensation,
                              body_steering=body_steering)
        drive = GuardedDrive(robot, logic, camera)
        drive.start()
        print(f'开始视觉接近：前 5 帧确认目标；球框占画面达到固定 {stop_area:.2%} 停车；'
              f'前进指令 {speed:g} m/s；最大 {duration:g} 秒。', flush=True)
        if orange_continuation:
            print('近球跟踪：保留 sports ball/orange 原始标签；漏检或截断先归零，最多等待 0.3 秒。',
                  flush=True)
        if camera_motion_compensation:
            print('位置关联加入相机姿态预测；按帧到达时刻配对，固定面积停车不变。', flush=True)
        if body_steering:
            print('转向按球相对机身的水平角度计算，补偿摆头与机身倾斜。', flush=True)
        last_index, last_print = 0, -1
        with (directory/'detections.jsonl').open('w') as log:
            while not drive.done.is_set():
                latest = slot.take_after(last_index, .05)
                if latest is None:
                    continue
                frame, payload = latest
                camera.raise_if_failed()
                prediction, image = detector.process(payload)
                camera_pose = camera_quat = trunk_quat = None
                if camera_motion_compensation:
                    try:
                        camera_pose = robot.camera_pose_at(frame['arrived_monotonic_s'])
                        camera_quat = camera_pose['quat'] if camera_pose is not None else None
                        trunk_quat = camera_pose.get('trunk_quat') if camera_pose is not None else None
                    except ValueError as error:
                        # Malformed orientation is a controlled stop, not a transport failure.
                        camera_pose, camera_quat, trunk_quat = {'error': str(error)}, [], []
                now = time.monotonic()
                decision = drive.observe(prediction['detections'],
                    frame['arrived_monotonic_s'], now, frame['index'], camera_quat=camera_quat,
                    trunk_quat=trunk_quat)
                record = {'receiver_index': frame['index'],
                    'payload_sha256': frame['payload_sha256'],
                    'received_complete_monotonic_s': frame['arrived_monotonic_s'],
                    'processed_monotonic_s': now, **prediction, 'control': decision}
                if camera_motion_compensation:
                    record['camera_pose'] = camera_pose
                detections.append(record)
                log.write(json.dumps(record, allow_nan=False)+'\n')
                log.flush()
                if last_image is None:
                    save_image(directory/'first_annotated.jpg', image)
                last_image, last_index = image, frame['index']
                elapsed = now-logic.started_at
                if elapsed-last_print >= 1 or decision['reason']:
                    print(f'{elapsed:.1f} 秒 | {decision["mode"]} | '
                          f'面积 {decision["area"]} | 转向 {decision["vyaw"]:+.3f} rad/s'
                          f' | {decision["reason"] or "tracking"}', flush=True)
                    last_print = elapsed
        drive.stop()
        report['stop_reason'] = logic.reason
        if drive.error:
            raise RuntimeError(drive.error)
        state, _ = robot.snapshot()
        previous_tick = state['t_ns']
        stopped_at = time.monotonic()
        try:
            report['stop_reply'] = robot.stop_moves()
        finally:
            moves_closed = True  # stop_moves closes its stream even when the reply fails.
        if not report['stop_reply'].get('accepted'):
            raise RuntimeError('robot.stop 未接受停止请求')
        stop_accepted = True
        report['final_state'] = confirm_stand(robot, stopped_at, previous_tick)
        report['stop_confirmed'] = True
        report['health_end'] = robot.request('robot.health', {})
        already_at_target = logic.moving_started_at is None and not any(
            any(abs(row[key]) > 0 for key in ('vx', 'vy', 'vyaw')) for row in drive.records)
        success = logic.reached and (drive.walk_seen or already_at_target)
        report['outcome'] = (
            ('completed_already_at_target_returned_stand' if already_at_target else
             'completed_target_area_returned_stand') if success else
            'stopped_'+('no_walk_observed' if logic.reached else logic.reason))
        status = 0 if success else 2
    except KeyboardInterrupt:
        report['outcome'], status = 'interrupted', 130
    except Exception:
        report['outcome'], report['error'] = 'failed', traceback.format_exc()
        traceback.print_exc()
    finally:
        # No disk writes before terminating the only sender and requesting zero.
        if drive is not None:
            try:
                drive.stop('interrupted' if status == 130 else 'finished')
            except Exception:
                report['sender_stop_error'] = traceback.format_exc()
                traceback.print_exc()
                success, status = False, 1
            if drive.error:
                report['drive_error'] = drive.error
                success, status = False, 1
        # Error exits use the same zero -> stop ordering as the normal exit.
        if robot is not None and drive is not None and not moves_closed:
            try:
                report['stop_reply'] = robot.stop_moves()
                if not report['stop_reply'].get('accepted'):
                    raise RuntimeError('robot.stop 未接受停止请求')
                stop_accepted = True
            except Exception:
                report['ordered_stop_error'] = traceback.format_exc()
                traceback.print_exc()
                success, status = False, 1
            finally:
                moves_closed = True
        # A receiver may report a final-frame/log write failure only after joining.
        # Inspect that error before deciding whether the trial can remain enabled.
        for label, resource in [('camera', camera), ('robot', robot)]:
            if resource is not None:
                try:
                    resource.stop() if label == 'camera' else resource.close()
                except Exception:
                    report[label+'_close_error'] = traceback.format_exc()
                    traceback.print_exc()
                    success, status = False, 1
                if resource.error:
                    report[label+'_error'] = resource.error
                    success, status = False, 1
        # Terminate movement and transports before releasing the inference runtime.
        if backend == 'npu' and detector is not None:
            try:
                detector.close()
            except Exception:
                report['detector_close_error'] = traceback.format_exc()
                traceback.print_exc()
                success, status = False, 1
        # A controlled stop can be incomplete while the robot is safely balancing.
        # Disabling here would replace that verified stand with a fixed HOME joint target.
        keep_standing = report['stop_confirmed'] and status in (0, 2)
        if not keep_standing:
            cleanup_control()
        if logic is not None:
            report.update(stop_reason=logic.reason, baseline_area=logic.baseline_area,
                target_area=logic.target_area)
        report['processed_frame_count'] = len(detections)
        try:
            if drive is not None:
                report['walk_observed'] = drive.walk_seen
                with (directory/'commands.jsonl').open('w') as log:
                    for row in drive.records:
                        log.write(json.dumps(row, allow_nan=False)+'\n')
            if camera is not None:
                report['transport'] = camera.summary([])
            if last_image is not None:
                save_image(directory/'last_annotated.jpg', last_image)
        except Exception:
            report['evidence_error'] = traceback.format_exc()
            traceback.print_exc()
            success, status = False, 1
            keep_standing = False
            if not report.get('robot.enable_cleanup', {}).get('accepted'):
                cleanup_control()
        try:
            if status == 1:
                report['outcome'] = 'failed'
            report['left_enabled_standing'] = keep_standing
            errors = '\n'.join(str(value) for key, value in report.items()
                               if key == 'error' or key.endswith('_error'))
            if errors:
                (directory/'error.txt').write_text(errors)
            (directory/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
            print(f'结果：{report["outcome"]}；停止确认={report["stop_confirmed"]}；'
                  f'保持平衡={report["left_enabled_standing"]}；'
                  f'完整数据：{directory}', flush=True)
        except Exception:
            report.update(outcome='failed', left_enabled_standing=False,
                          final_output_error=traceback.format_exc())
            traceback.print_exc()  # Disk/output failures may prevent a saved summary.
            success, status = False, 1
            keep_standing = False
            if not report.get('robot.enable_cleanup', {}).get('accepted'):
                cleanup_control()
            try:
                (directory/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
            except Exception:
                traceback.print_exc()
        finally:
            signal.signal(signal.SIGTERM, previous_signal)
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path,
        default=Path.home()/'microduck-board/yolo11n-cpu/models/yolo11n.onnx')
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
    output = args.output_dir or Path.home()/'microduck-board/run/G4'/('ball_follow_'+stamp)
    sys.exit(run_trial(args.model, ('192.168.50.1', 7901), args.socket, output,
                       speed=args.speed, stop_area=args.stop_area,
                       orange_continuation=args.orange_continuation,
                       camera_motion_compensation=args.camera_motion_compensation,
                       body_steering=args.body_steering))
