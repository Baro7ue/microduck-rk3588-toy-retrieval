# Moved 2026-10-11 from work/probe_runtime_stand.py (original kept unchanged in archive/2026-10_ball_follow/work/); only paths and imports changed.
"""Bounded runtime stand probe; stop on faults, optionally keep a successful stand."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time
import traceback

from microduck_toy.paths import runs_dir

SOCKET = os.environ.get('MD_ROBOT_SOCKET', '/private/tmp/md-runtime-ethan/robotd.sock')
DURATION = 10.0
STARTUP_LIMIT = 5.0


def rpc(method: str, params: dict) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(3.0)
        client.connect(SOCKET)
        packet = {'jsonrpc': '2.0', 'id': 1, 'method': method, 'params': params}
        client.sendall((json.dumps(packet) + '\n').encode())
        with client.makefile('rb') as stream:
            while line := stream.readline():
                reply = json.loads(line)
                if reply.get('id') == 1:
                    if 'error' in reply:
                        raise RuntimeError(json.dumps(reply, ensure_ascii=False))
                    return reply['result']
    raise RuntimeError(f'{method}: connection closed before its response')


def interrupted(_signum: int, _frame: object) -> None:
    raise KeyboardInterrupt


def main(keep_enabled: bool = False, support_log: Path | None = None) -> int:
    stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    directory = runs_dir() / f'runtime_stand_{stamp}'
    directory.mkdir(parents=True, exist_ok=False)
    print(f'证据目录：{directory}', flush=True)
    print('本次仅检查零速度站立；不发送前进或转向指令。', flush=True)
    print('当前零速度采用 velstand.onnx；此试验尚不验证自训行走模型。', flush=True)
    print('起步最多 5 秒；策略接管并解除起步支撑后再观察 10 秒。'
          if support_log else '起步最多 5 秒；策略接管后再观察 10 秒，起步不计入站立时间。', flush=True)
    print('发现摔倒、报错或中断时会请求关闭策略控制。', flush=True)
    if keep_enabled:
        print('通过后继续保持站立，便于下一步前进试验。', flush=True)
    report = {'socket': SOCKET, 'duration_limit_s': DURATION,
              'outcome': 'not_started', 'disable_confirmed': False,
              'keep_enabled_requested': keep_enabled, 'left_enabled': False,
              'support_log': str(support_log) if support_log else None}
    attempted = False
    states = 0
    active_seen = False
    start = None
    policy_start = None
    result = 1
    subscriber = None
    stream = None
    support_stream = None
    support_released = support_log is None
    signal.signal(signal.SIGTERM, interrupted)
    try:
        if support_log:
            support_stream = support_log.open()
            support_stream.seek(0, 2)  # Only accept a release from this enable attempt.
        subscriber = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        subscriber.settimeout(2.0)
        subscriber.connect(SOCKET)
        stream = subscriber.makefile('rb')
        request = {'jsonrpc': '2.0', 'id': 1, 'method': 'robot.subscribe',
                   'params': {'hz': 20}}
        subscriber.sendall((json.dumps(request) + '\n').encode())
        with (directory / 'states.jsonl').open('w') as raw, \
                (directory / 'states.csv').open('w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['elapsed_s', 'daemon_t', 'policy', 'fallen',
                             'gravity_x', 'gravity_y', 'gravity_z', 'hz', 'missed'])
            last_label = None
            last_print = -1.0
            while True:
                line = stream.readline()
                if not line:
                    raise RuntimeError('robotd 状态流已关闭')
                raw.write(line.decode())
                raw.flush()
                message = json.loads(line)
                if message.get('id') == 1:
                    if 'error' in message or not message.get('result', {}).get('accepted'):
                        raise RuntimeError(json.dumps(message, ensure_ascii=False))
                    report['models'] = message['result']
                    continue
                if message.get('method') != 'robot.state':
                    continue
                state = message['params']
                safety = state['safety']
                gravity = safety['gravity']
                label = state['policy']
                loop = state['loop']
                elapsed = 0.0 if start is None else time.monotonic() - start
                writer.writerow([round(elapsed, 6), state['t'], label, safety['fallen'],
                                 *gravity, loop['hz'], loop['missed']])
                csv_file.flush()
                states += 1
                if not attempted:
                    if label != 'held' or safety['fallen'] or gravity[2] > -0.9:
                        raise RuntimeError('初始状态并非未启用且直立；已停止，不发送启用请求。')
                    if any(abs(v) > 1e-8 for v in state['move']['requested']):
                        raise RuntimeError('检测到非零速度请求；已停止，不发送启用请求。')
                    report['initial_state'] = state
                    attempted = True  # Even a lost reply may follow an accepted enable.
                    start = time.monotonic()
                    reply = rpc('robot.enable', {'on': True})
                    report['enable_reply'] = reply
                    print(f'启用请求：{json.dumps(reply, ensure_ascii=False)}', flush=True)
                    if not reply.get('accepted'):
                        raise RuntimeError('启用请求未被接受')
                    continue
                if elapsed >= STARTUP_LIMIT and not (active_seen and support_released):
                    report['outcome'] = 'no_support_release' if active_seen else 'no_policy_takeover'
                    result = 2
                    print('起步超过 5 秒，尚未完成策略接管和支撑解除，结束试验。', flush=True)
                    break
                if label == 'stand':
                    if not active_seen:
                        report['startup_s'] = elapsed
                    active_seen = True
                if support_stream and not support_released:
                    while True:
                        offset = support_stream.tell()
                        support_line = support_stream.readline()
                        if not support_line:
                            break
                        if not support_line.endswith('\n'):
                            support_stream.seek(offset)
                            break
                        prefix = '== startup assistance ended: '
                        if support_line.startswith(prefix):
                            report['support_release'] = json.loads(support_line[len(prefix):])
                            support_released = True
                            report['support_release_observed_s'] = elapsed
                            print('起步支撑已解除；开始计满 10 秒站立。', flush=True)
                            break
                release_t = report.get('support_release', {}).get('daemon_t', 0.0)
                if (label == 'stand' and support_released and policy_start is None
                        and state['t'] >= release_t):
                    policy_start = time.monotonic()
                    report['observation_initial_state'] = state
                active_elapsed = 0.0 if policy_start is None else time.monotonic() - policy_start
                if label != last_label or elapsed - last_print >= 1.0:
                    print(f'{elapsed:5.2f} 秒 | 阶段 {label} | 重力 z={gravity[2]:+.3f}'
                          f' | 摔倒={safety["fallen"]} | {loop["hz"]:.1f} Hz', flush=True)
                    last_label, last_print = label, elapsed
                report['last_state'] = state
                report['elapsed_s'] = elapsed
                report['active_elapsed_s'] = active_elapsed
                if safety['fallen']:
                    report['outcome'] = 'fallen_after_policy' if active_seen else 'fallen_before_policy'
                    print('检测到摔倒，结束试验；保留起步过程供定位。', flush=True)
                    result = 2
                    break
                if any(abs(v) > 1e-8 for field in ('requested', 'applied')
                       for v in state['move'][field]):
                    report['outcome'] = 'unexpected_command'
                    result = 2
                    print('观测到非零速度指令，结束零速度站立试验。', flush=True)
                    break
                if (active_seen and label != 'stand') or (
                        not active_seen and label not in {'held', 'homing'}):
                    report['outcome'] = 'unexpected_policy'
                    result = 2
                    print(f'站立阶段出现非预期状态 {label}，结束试验。', flush=True)
                    break
                if active_elapsed >= DURATION:
                    daemon_elapsed = state['t'] - report['observation_initial_state']['t']
                    report['daemon_observation_elapsed_s'] = daemon_elapsed
                    if daemon_elapsed < DURATION:
                        if active_elapsed < DURATION + 2.0:
                            continue
                        report['outcome'] = 'insufficient_state_coverage'
                        result = 2
                        print('状态流尚未覆盖完整 10 秒，结束试验并保留数据。', flush=True)
                        break
                    report['outcome'] = 'completed_no_fall' if label == 'stand' else 'unexpected_policy'
                    result = 0 if label == 'stand' else 2
                    initial_loop = report['observation_initial_state']['loop']
                    report['observation_missed_delta'] = loop['missed'] - initial_loop['missed']
                    print('10 秒站立观测结束。', flush=True)
                    break
    except KeyboardInterrupt:
        report['outcome'] = 'interrupted'
        print('试验已中断。', flush=True)
        result = 130
    except Exception:
        report['outcome'] = 'error'
        report['error'] = traceback.format_exc()
        traceback.print_exc()
    finally:
        successful_keep = keep_enabled and result == 0 and report['outcome'] == 'completed_no_fall'
        if attempted and successful_keep:
            report['left_enabled'] = True
            print('试验通过；保持零速度站立，控制程序和身体窗口请继续运行。', flush=True)
        elif attempted:
            try:
                reply = rpc('robot.enable', {'on': False})
                report['disable_reply'] = reply
                report['disable_confirmed'] = bool(reply.get('accepted'))
                print(f'关闭请求：{json.dumps(reply, ensure_ascii=False)}', flush=True)
            except Exception:
                report['disable_error'] = traceback.format_exc()
                traceback.print_exc()
                print('关闭请求未获确认；请在 robotd 终端按 Ctrl+C 停止控制程序。', flush=True)
                result = 1
        if stream is not None:
            stream.close()
        if subscriber is not None:
            subscriber.close()
        if support_stream is not None:
            support_stream.close()
        report['states'] = states
        report['policy_takeover_seen'] = active_seen
        (directory / 'summary.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print(f'结果：{report["outcome"]}；完整数据已保存。', flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--keep-enabled', action='store_true',
                        help='only after a successful stand, keep zero-velocity policy active')
    parser.add_argument('--support-log', type=Path,
                        help='count standing only after a new startup-support release in this log')
    args = parser.parse_args()
    sys.exit(main(args.keep_enabled, args.support_log))
