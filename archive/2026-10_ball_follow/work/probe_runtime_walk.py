"""Finite runtime walking trial from an already enabled, standing simulated duck."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
from pathlib import Path
import signal
import socket
import sys
import threading
import time
import traceback

from probe_runtime_stand import PROJECT, interrupted
from heading_hold import HeadingConfig, HeadingHold


def rpc(path: str, method: str, params: dict) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(2.0)
        client.connect(path)
        client.sendall((json.dumps({'jsonrpc': '2.0', 'id': 1,
                                   'method': method, 'params': params}) + '\n').encode())
        with client.makefile('rb') as reader:
            response = json.loads(reader.readline())
    if 'error' in response:
        raise RuntimeError(response)
    return response['result']


def capture_health(path: str) -> dict:
    """Read loop counters without changing robot control."""
    started = time.monotonic()
    try:
        health = rpc(path, 'robot.health', {})
    except Exception:
        return {'error': traceback.format_exc()}
    finished = time.monotonic()
    return {'sample_monotonic_s': (started + finished) / 2,
            'rpc_duration_s': finished - started, 'health': health}


def summarize_loop(report: dict) -> None:
    start, end = report.get('health_walk_start', {}), report.get('health_walk_end', {})
    if 'health' not in start or 'health' not in end:
        print('控制环起止计数未完整取得；详细错误已保存在日志中。', flush=True)
        return
    before, after = start['health']['control_loop'], end['health']['control_loop']
    elapsed = end['sample_monotonic_s'] - start['sample_monotonic_s']
    ticks, missed = after['ticks'] - before['ticks'], after['missed'] - before['missed']
    if elapsed <= 0 or ticks <= 0 or missed < 0:
        report['loop_window_error'] = 'Invalid timing or counters; daemon may have restarted.'
        return
    report['loop_window'] = {'sample_elapsed_s': elapsed, 'ticks': ticks,
        'missed': missed, 'achieved_hz': ticks / elapsed,
        'work_overrun_percent': 100 * missed / ticks,
        'note': 'missed counts loop work exceeding its period; scheduler skips are not counted.'}
    print(f'行走期间控制环：平均 {ticks / elapsed:.2f} Hz，'
          f'新增执行超时 {missed}/{ticks}（{100 * missed / ticks:.3f}%）。', flush=True)


class Drive:
    def __init__(self, path: str, vx: float, heading_config: HeadingConfig | None = None):
        self.path, self.vx = path, vx
        self.heading = HeadingHold(heading_config) if heading_config else None
        self.lock = threading.Lock()
        self.vyaw = 0.0
        self.feedback_at = None
        self.sensor_ns = None
        self.stop_event = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self.run, daemon=True)

    def update_feedback(self, state: dict) -> None:
        if self.heading is None:
            return
        sensor_ns = state['t_ns']
        if self.sensor_ns is not None and sensor_ns == self.sensor_ns:
            return  # Repeated sensor samples must not refresh the feedback deadline.
        dt = (self.heading.config.period_s if self.sensor_ns is None
              else (sensor_ns - self.sensor_ns) / 1e9)
        if not 0 < dt <= 1.5 * self.heading.config.period_s:
            raise RuntimeError(f'PI 的 IMU 采样间隔异常：{dt:.3f} 秒；停止试验。')
        yaw = self.heading.update(state['imu']['gyro'], state['safety']['gravity'],
                                  dt, (self.vx, 0.0, 0.0))
        if not self.heading.active:
            raise RuntimeError(f'PI 航向反馈无效：{self.heading.status}')
        self.sensor_ns = sensor_ns
        with self.lock:
            self.vyaw = yaw
            self.feedback_at = time.monotonic()

    def run(self) -> None:
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(0.3)
                client.connect(self.path)
                while not self.stop_event.is_set():
                    with self.lock:
                        yaw, feedback_at = self.vyaw, self.feedback_at
                    if self.heading and (feedback_at is None or time.monotonic() - feedback_at > 0.2):
                        client.sendall((json.dumps({'jsonrpc': '2.0', 'method': 'robot.move',
                            'params': {'vx': 0.0, 'vy': 0.0, 'vyaw': 0.0}}) + '\n').encode())
                        raise RuntimeError('超过 0.2 秒没有新的 IMU 反馈，停止前进。')
                    message = (json.dumps({'jsonrpc': '2.0', 'method': 'robot.move',
                        'params': {'vx': self.vx, 'vy': 0.0, 'vyaw': yaw}}) + '\n').encode()
                    client.sendall(message)
                    self.stop_event.wait(0.05)
        except Exception:
            self.error = traceback.format_exc()
            self.stop_event.set()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=1.0)
        if self.thread.is_alive():
            raise RuntimeError('速度发送线程尚未退出')
        if self.heading:
            self.heading.release('stopped')


def main(path: str, duration: float = 10.0, vx: float = 0.3,
         heading_config: HeadingConfig | None = None) -> int:
    directory = PROJECT / 'task/evidence' / os.environ.get('MD_EVIDENCE_GATE', 'G2') / ('runtime_walk_' +
        dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    directory.mkdir(parents=True, exist_ok=False)
    print(f'证据目录：{directory}', flush=True)
    print(f'前进指令 {vx:.2f} m/s；观察自训行走策略 {duration:g} 秒。',
          flush=True)
    print('独立 PI 根据 IMU 修正转向；横移为零。' if heading_config
          else '横移和转向指令均为零。', flush=True)
    print('结束后停止前进并恢复站立；摔倒、报错或中断时请求关闭策略。', flush=True)
    report = {'socket': path, 'vx': vx, 'duration_s': duration,
              'heading_compensation': bool(heading_config), 'outcome': 'not_started'}
    if heading_config:
        report['heading_config'] = vars(heading_config)
        report['heading_feedback'] = 'body gyro + gravity, sensor t_ns; computed on Mac'
        report['state_subscription'] = 'every control tick, without server decimation'
    subscriber = None
    stream = None
    drive = None
    attempted = False
    success = False
    result = 1
    walk_started = None
    stop_started = None
    stand_settled = None
    signal.signal(signal.SIGTERM, interrupted)
    try:
        subscriber = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        subscriber.settimeout(2.0)
        subscriber.connect(path)
        stream = subscriber.makefile('rb')
        # PI needs the sensor history. Dashboard-style decimation can turn one
        # delayed control tick into a large gap between the retained samples.
        subscriber.sendall((json.dumps({'jsonrpc': '2.0', 'id': 1,
            'method': 'robot.subscribe',
            'params': {} if heading_config else {'hz': 20}}) + '\n').encode())
        with (directory / 'states.jsonl').open('w') as raw, \
                (directory / 'states.csv').open('w', newline='') as table:
            writer = csv.writer(table)
            writer.writerow(['elapsed_s', 'stage', 'policy', 'fallen', 'gravity_z',
                             'vx_requested', 'vx_applied', 'vyaw_applied', 'hz',
                             'missed', 'odom_x', 'odom_y', 'odom_yaw_rad'])
            heading_log = []
            began = None
            last_print = -1.0
            previous_label = None
            while True:
                line = stream.readline()
                if not line:
                    raise RuntimeError('robotd 状态流已关闭')
                raw.write(line.decode())
                raw.flush()
                message = json.loads(line)
                if message.get('id') == 1:
                    models = message.get('result', {})
                    if 'error' in message or not models.get('accepted'):
                        raise RuntimeError(message)
                    report['models'] = models
                    if models.get('walk') != 'walk_17500.onnx':
                        raise RuntimeError('当前行走模型不是 walk_17500.onnx，未发送速度指令。')
                    continue
                if message.get('method') != 'robot.state':
                    continue
                state = message['params']
                now = time.monotonic()
                elapsed = 0.0 if began is None else now - began
                label = state['policy']
                safety = state['safety']
                move = state['move']
                loop = state['loop']
                odom = state.get('odom', {'position': [0.0] * 3, 'yaw': 0.0})
                stage = 'stopping' if stop_started is not None else 'walking'
                if drive and drive.heading and stop_started is None and not safety['fallen']:
                    drive.update_feedback(state)
                    heading_log.append({'daemon_t': state['t'], 'sensor_ns': state['t_ns'],
                        'relative_heading_rad': drive.heading.heading,
                        'error_rad': drive.heading.error, 'integral': drive.heading.integral,
                        'vyaw_command': drive.vyaw, 'status': drive.heading.status})
                writer.writerow([elapsed, stage, label, safety['fallen'], safety['gravity'][2],
                    move['requested'][0], move['applied'][0], move['applied'][2], loop['hz'],
                    loop['missed'], *odom['position'][:2], odom['yaw']])
                table.flush()
                report['last_state'] = state
                if not attempted:
                    if (label != 'stand' or safety['fallen'] or safety['gravity'][2] > -0.9
                            or any(abs(v) > 1e-8 for field in ('requested', 'applied')
                                   for v in move[field])):
                        raise RuntimeError('初始状态并非零速度稳定站立，未发送前进指令。')
                    if not report.get('models'):
                        raise RuntimeError('尚未收到模型信息，未发送前进指令。')
                    report['initial_state'] = state
                    drive = Drive(path, vx, heading_config)
                    drive.update_feedback(state)
                    attempted = True
                    began = time.monotonic()
                    drive.thread.start()
                    continue
                if drive.error:
                    raise RuntimeError(drive.error)
                if label != previous_label or elapsed - last_print >= 1.0:
                    print(f'{elapsed:5.2f} 秒 | {stage} | {label}'
                          f' | 前进指令 {move["applied"][0]:.2f}'
                          f' | 摔倒={safety["fallen"]} | {loop["hz"]:.1f} Hz', flush=True)
                    if drive.heading and stop_started is None:
                        print(f'航向误差 {math.degrees(drive.heading.error):+.2f}°'
                              f' | PI 转向 {drive.vyaw:+.3f} rad/s', flush=True)
                    previous_label, last_print = label, elapsed
                if safety['fallen']:
                    report['outcome'] = 'fallen_stopping' if stop_started else 'fallen_walking'
                    print('检测到摔倒，结束试验并保留日志。', flush=True)
                    result = 2
                    break
                if stop_started is None:
                    if label == 'walk' and walk_started is None:
                        walk_started = now
                        report['walk_start_s'] = elapsed
                        report['health_walk_start'] = capture_health(path)
                    if label not in {'stand', 'walk'} or (
                            walk_started is not None and label != 'walk'):
                        raise RuntimeError(f'行走阶段出现非预期策略状态：{label}')
                    if walk_started is None and elapsed >= 2.0:
                        raise RuntimeError('2 秒内未切换到自训行走策略')
                    if walk_started is not None and now - walk_started >= duration:
                        report['walk_elapsed_s'] = now - walk_started
                        report['health_walk_end'] = capture_health(path)
                        drive.stop()  # Stop refreshes first, so nothing overwrites robot.stop.
                        report['stop_reply'] = rpc(path, 'robot.stop', {})
                        if not report['stop_reply'].get('accepted'):
                            raise RuntimeError('停止前进请求未被接受')
                        stop_started = time.monotonic()
                else:
                    settled = (label == 'stand' and all(abs(v) < 0.01 for v in move['applied'])
                               and all(abs(v) < 1e-8 for v in move['requested']))
                    stand_settled = (now if stand_settled is None else stand_settled) if settled else None
                    if stand_settled is not None and now - stand_settled >= 1.0:
                        report['outcome'] = 'completed_no_fall_returned_stand'
                        success = True
                        result = 0
                        print('行走观测结束，已停止前进并恢复站立。', flush=True)
                        summarize_loop(report)
                        break
                    if now - stop_started >= 3.0:
                        raise RuntimeError('停止前进后 3 秒内未稳定恢复站立')
    except KeyboardInterrupt:
        report['outcome'] = 'interrupted'
        print('试验已中断。', flush=True)
        result = 130
    except Exception:
        report['outcome'] = 'error'
        report['error'] = traceback.format_exc()
        traceback.print_exc()
    finally:
        if drive is not None:
            try:
                drive.stop()
            except Exception:
                report['sender_stop_error'] = traceback.format_exc()
                traceback.print_exc()
                success = False
                result = 1
        if attempted and not success:
            for method, params in [('robot.stop', {}), ('robot.enable', {'on': False})]:
                try:
                    reply = rpc(path, method, params)
                    report[method + '_cleanup'] = reply
                    if not reply.get('accepted'):
                        raise RuntimeError(reply)
                except Exception:
                    report[method + '_error'] = traceback.format_exc()
                    traceback.print_exc()
                    print('停止请求未获确认；请在 robotd 终端按 Ctrl+C 停止控制程序。', flush=True)
                    result = 1
        if stream is not None:
            stream.close()
        if subscriber is not None:
            subscriber.close()
        report['left_enabled_standing'] = success
        if drive and drive.heading:
            report['heading_final'] = vars(drive.heading).copy()
            report['heading_final'].pop('config')
            (directory / 'heading.json').write_text(json.dumps(heading_log, indent=2) + '\n')
        (directory / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print(f'结果：{report["outcome"]}；完整数据已保存。', flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--socket', default='/private/tmp/md-runtime-ethan/robotd.sock')
    parser.add_argument('--duration', type=float, default=10.0,
                        help='seconds of continuous walk to observe (1 to 600, default 10)')
    parser.add_argument('--vx', type=float, default=0.3)
    parser.add_argument('--heading-hold', action='store_true')
    args = parser.parse_args()
    if not 1.0 <= args.duration <= 600.0:
        parser.error('--duration must be between 1 and 600 seconds')
    if not math.isfinite(args.vx) or not 0.05 < args.vx <= 0.3:
        parser.error('--vx must be greater than 0.05 and no greater than 0.3 m/s')
    config = HeadingConfig(kp=1.0, ki=0.05, period_s=0.05) if args.heading_hold else None
    sys.exit(main(args.socket, duration=args.duration, vx=args.vx, heading_config=config))
