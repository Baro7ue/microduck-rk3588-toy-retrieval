#!/usr/bin/env python3
"""把两个已验证探针并行运行，保存一次行走与相机负载的关联证据。

前提：body/headless 已在 127.0.0.1:7801/7901 运行；robotd 已启用零速站立。
输入：仅 --socket（默认已有 robotd Unix socket）；使用已有 .venv 和两个探针。
输出：G2/walk_camera_<微秒时间戳>/summary.json，以及两个子进程的 stdout/stderr。
进程：先只读观测；5 秒内确认第一条完整帧记录后才启动 60 秒行走。
75 秒 = 5×15 秒观测，含前后站立；不能把全程均值叫作 60 秒行走指标。
父进程不连接设备、不启停服务、不自动 enable、不修改参数或依赖。
中断/失败：先 SIGINT 自己创建的行走进程并等其清理，再停只读观测。
停止未确认时必须到 robotd 终端 Ctrl+C；父进程不会声称机器人已停止。

实际命令摘要（子进程均直接 Popen，不经过 shell）：
  zsh work/probe_runtime_walk_camera.zsh
  .venv/bin/python -u work/probe_runtime_visual_load.py --camera --seconds 15 --samples 5
  .venv/bin/python -u work/probe_runtime_walk.py --duration 60 --socket <已有socket>
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import traceback

PROJECT = Path('/Users/ethan/Documents/RK3588 Microduck')
EVIDENCE_ROOT = PROJECT / 'task/evidence/G2'
PYTHON = PROJECT / 'task/microduck_r1/.venv/bin/python'
READY_TIMEOUT_S = 5.0
WALK_TIMEOUT_S = 72.0
OBSERVER_TIMEOUT_S = 90.0
CLEANUP_TIMEOUT_S = 12.0
EXPECTED_BYTES = 640 * 360 * 2
PRINT_LOCK = threading.Lock()
TERMINAL_ERRORS: list[str] = []


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec='microseconds')


def say(message: str) -> None:
    with PRINT_LOCK:
        try:
            print(message, flush=True)
        except Exception as error:
            description = f'stdout: {type(error).__name__}: {error}'
            if description not in TERMINAL_ERRORS:
                TERMINAL_ERRORS.append(description)
            # A broken output pipe must never prevent a stop signal or evidence saving.
            try:
                print(message, file=sys.stderr, flush=True)
            except Exception as fallback_error:
                description = f'stderr: {type(fallback_error).__name__}: {fallback_error}'
                if description not in TERMINAL_ERRORS:
                    TERMINAL_ERRORS.append(description)


def finite_number(value) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value))


class Child:
    """Own one new process session; pipe threads only log and parse its directory."""

    def __init__(self, label: str, command: list[str], parent: Path, prefix: str):
        self.label, self.command, self.prefix = label, command, prefix
        self.directory: Path | None = None
        self.errors: list[str] = []
        self.shutdown: dict = {}
        self.started_monotonic_s = time.monotonic()
        self.process = subprocess.Popen(command, cwd=PROJECT, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8',
            errors='replace', bufsize=1, start_new_session=True)
        self.threads = []
        self.logs = {}
        self.logs = {name: str(parent / f'{label}.{name}.log') for name in ('stdout', 'stderr')}

    def start_logs(self) -> None:
        # Call only after main owns this Child, so thread-start failure cannot orphan it.
        for name, stream in (('stdout', self.process.stdout), ('stderr', self.process.stderr)):
            log = Path(self.logs[name])
            thread = threading.Thread(target=self._copy, args=(name, stream, log), daemon=True)
            thread.start()
            self.threads.append(thread)

    def _copy(self, name: str, stream, log: Path) -> None:
        try:
            with log.open('w', encoding='utf-8') as output:
                for line in stream:
                    output.write(line)
                    output.flush()
                    if name == 'stdout' and line.startswith('证据目录：'):
                        path = Path(line.split('：', 1)[1].strip()).resolve()
                        if path.parent != EVIDENCE_ROOT.resolve() or not path.name.startswith(self.prefix):
                            raise RuntimeError(f'{self.label} 输出了非预期证据路径：{path}')
                        if self.directory is not None and self.directory != path:
                            raise RuntimeError(f'{self.label} 输出了多个证据路径')
                        self.directory = path
                    say(f'[{self.label}/{name}] {line.rstrip()}')
        except Exception:
            self.errors.append(traceback.format_exc())
        finally:
            stream.close()

    def join_logs(self) -> None:
        for thread in self.threads:
            thread.join(timeout=1.0)
        if any(thread.is_alive() for thread in self.threads):
            self.errors.append('日志线程未结束；日志可能不完整')

    def read_summary(self) -> dict | None:
        if self.directory is None:
            return None
        path = self.directory / 'summary.json'
        try:
            value = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(value, dict):
                raise ValueError('summary 不是 JSON object')
            return value
        except Exception as error:
            self.errors.append(f'无法读取 {path}：{error}')
            return None

    def record(self, summary: dict | None) -> dict:
        return {'command': self.command, 'pid': self.process.pid,
                'start_new_session': True, 'started_monotonic_s': self.started_monotonic_s,
                'returncode': self.process.poll(), 'evidence_directory': str(self.directory)
                if self.directory else None, 'summary_file': str(self.directory / 'summary.json')
                if self.directory else None, 'outcome': summary.get('outcome') if summary else None,
                'logs': self.logs, 'errors': self.errors, 'shutdown': self.shutdown}


def first_complete_frame(child: Child) -> dict | None:
    if child.directory is None:
        return None
    path = child.directory / 'frames.jsonl'
    try:
        with path.open(encoding='utf-8') as stream:
            line = stream.readline()
    except FileNotFoundError:
        return None
    if not line.endswith('\n'):
        return None
    frame = json.loads(line)
    if (not isinstance(frame, dict) or type(frame.get('index')) is not int or frame.get('index') != 1
            or frame.get('payload_bytes') != EXPECTED_BYTES
            or not finite_number(frame.get('arrived_monotonic_s'))
            or not finite_number(frame.get('receive_duration_s'))
            or frame['receive_duration_s'] < 0
            or not isinstance(frame.get('arrived_wall_utc'), str)
            or not re.fullmatch(r'[0-9a-f]{64}', str(frame.get('payload_sha256', '')))):
        raise RuntimeError('第一条帧记录不符合完整 640×360 UYVY 帧协议；不会启动行走')
    return frame


def check_cancel(cancelled: threading.Event, child: Child | None = None) -> None:
    if cancelled.is_set():
        raise KeyboardInterrupt
    if TERMINAL_ERRORS:
        raise RuntimeError(f'终端输出失败；中断试验并保留证据：{TERMINAL_ERRORS[0]}')
    if child is not None and child.errors:
        raise RuntimeError(f'{child.label} 日志/证据读取失败：{child.errors[-1]}')


def wait_ready(observer: Child, cancelled: threading.Event) -> dict:
    deadline = observer.started_monotonic_s + READY_TIMEOUT_S
    while time.monotonic() < deadline:
        check_cancel(cancelled, observer)
        if observer.process.poll() is not None:
            raise RuntimeError('只读观测在取得完整帧前退出；未启动行走')
        frame = first_complete_frame(observer)
        if frame is not None:
            return frame
        time.sleep(0.05)
    raise RuntimeError('5 秒内未取得完整帧记录；未启动行走')


def interrupt_child(child: Child | None) -> None:
    if child is None or child.process.poll() is not None:
        return
    child.shutdown = {'requested_signal': 'SIGINT', 'requested_wall_utc': utc_now()}
    say(f'先请求 {child.label} 中断并等待其原有清理（最多 {CLEANUP_TIMEOUT_S:g} 秒）。')
    try:
        os.killpg(child.process.pid, signal.SIGINT)
    except ProcessLookupError:
        pass
    except OSError as error:
        child.shutdown['signal_error'] = str(error)
        child.errors.append(f'无法向自己的 {child.label} 子进程发 SIGINT：{error}')
        if child.label == 'walk':
            say('停止/清理未获确认；请立即在 robotd 终端按 Ctrl+C。')
        return
    try:
        child.process.wait(timeout=CLEANUP_TIMEOUT_S)
        child.shutdown['exited_after_sigint'] = True
    except subprocess.TimeoutExpired:
        child.shutdown['exited_after_sigint'] = False
        child.errors.append('SIGINT 后清理超时；未强杀，子进程可能仍运行')
        if child.label == 'walk':
            say('停止/清理未获确认；请立即在 robotd 终端按 Ctrl+C。')


def walk_succeeded(summary: dict | None, child: Child) -> bool:
    return bool(summary and child.process.poll() == 0
                and summary.get('outcome') == 'completed_no_fall_returned_stand'
                and summary.get('left_enabled_standing') is True
                and not summary.get('sender_stop_error'))


def cleanup_status(walk: Child | None, summary: dict | None) -> dict:
    if walk is None:
        return {'status': 'walk_not_started', 'manual_robotd_ctrl_c_required': False}
    if walk_succeeded(summary, walk):
        return {'status': 'child_confirmed_returned_stand', 'manual_robotd_ctrl_c_required': False}
    if summary and walk.process.poll() is not None:
        if 'initial_state' not in summary:
            return {'status': 'child_reported_no_move_attempt', 'manual_robotd_ctrl_c_required': False}
        stop = summary.get('robot.stop_cleanup')
        disable = summary.get('robot.enable_cleanup')
        if (isinstance(stop, dict) and stop.get('accepted') is True
                and isinstance(disable, dict) and disable.get('accepted') is True
                and not summary.get('sender_stop_error')
                and not any(key in summary for key in ('robot.stop_error', 'robot.enable_error'))):
            return {'status': 'child_cleanup_requests_accepted',
                    'note': '只确认 stop/disable 请求被接受；未独立验证机器人实际状态。',
                    'manual_robotd_ctrl_c_required': False}
    return {'status': 'unconfirmed', 'manual_robotd_ctrl_c_required': True,
            'instruction': '停止/清理未获确认；请立即在 robotd 终端按 Ctrl+C。'}


def analyze_windows(observer: dict | None, walk: dict | None, directory: Path | None) -> dict:
    """Use only recorded monotonic times; no extra device requests or Gate decision."""
    result = {'verified': False, 'limitations': [
        '75 秒原始观测含前后站立，其均值不代表完整 60 秒行走实时因子。',
        '行走起止采用 health RPC midpoint；与实际策略切换存在采样边界差异。',
        '帧数为完整负载的传输证据；协议没有源时间戳/序号，不能测真实帧龄或唯一新帧率。'],
        'gate_decision': 'not_assessed'}
    if not observer or not walk or directory is None:
        result['reason'] = '子汇总缺失，窗口关系未验证'
        return result
    start = walk.get('health_walk_start', {}).get('sample_monotonic_s')
    end = walk.get('health_walk_end', {}).get('sample_monotonic_s')
    if not finite_number(start) or not finite_number(end) or end <= start:
        result['reason'] = '行走 health 起止缺失/无效，窗口关系未验证'
        return result
    windows = observer.get('windows', [])
    valid = [w for w in windows if w.get('outcome') == 'completed'
             and finite_number(w.get('start_sample', {}).get('midpoint_monotonic_s'))
             and finite_number(w.get('end_sample', {}).get('midpoint_monotonic_s'))]
    if not valid:
        result['reason'] = '没有完整观测窗口，窗口关系未验证'
        return result
    observation_start = valid[0]['start_sample']['midpoint_monotonic_s']
    observation_end = valid[-1]['end_sample']['midpoint_monotonic_s']
    result.update(walk_health_start_monotonic_s=start, walk_health_end_monotonic_s=end,
        walk_health_elapsed_s=end - start, reported_walk_elapsed_s=walk.get('walk_elapsed_s'),
        observation_start_monotonic_s=observation_start, observation_end_monotonic_s=observation_end,
        body_window_covers_walk=observation_start <= start and end <= observation_end,
        before_walk_observed_s=max(0.0, min(start, observation_end) - observation_start),
        after_walk_observed_s=max(0.0, observation_end - max(end, observation_start)))
    annotated = []
    selected = []
    for window in windows:
        copy = dict(window)
        if window not in valid:
            copy['walk_relation'] = 'incomplete'
        else:
            left = window['start_sample']['midpoint_monotonic_s']
            right = window['end_sample']['midpoint_monotonic_s']
            relation = ('before_walk' if right <= start else 'after_walk' if left >= end
                        else 'fully_inside_walk' if start <= left and right <= end
                        else 'partly_overlaps_walk')
            copy['walk_relation'] = relation
            copy['walk_overlap_s'] = max(0.0, min(right, end) - max(left, start))
            if relation == 'fully_inside_walk':
                selected.append(window)
        annotated.append(copy)
    result['original_observer_windows_with_relations'] = annotated
    wall = sum(w['wall_delta_s'] for w in selected)
    sim = sum(w['sim_delta_s'] for w in selected)
    count = sum(w.get('camera_frames_received', 0) for w in selected)
    result['fully_inside_walk_windows'] = {'verified': bool(selected),
        'window_indices': [w['index'] for w in selected], 'wall_elapsed_s': wall,
        'sim_elapsed_s': sim, 'realtime_factor': sim / wall if wall > 0 else None,
        'complete_transport_frames': count, 'transport_fps': count / wall if wall > 0 else None,
        'note': '仅汇总完全落在行走 health 起止内的窗口；并非完整 60 秒行走 RTF。'
        if selected else '没有完全落在行走区间的窗口，行走 RTF 未验证。'}
    arrivals = []
    try:
        with (directory / 'frames.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                if not line.endswith('\n'):
                    raise ValueError('frames.jsonl 尾部存在不完整行')
                frame = json.loads(line)
                arrived = frame.get('arrived_monotonic_s')
                if frame.get('payload_bytes') != EXPECTED_BYTES or not finite_number(arrived):
                    raise ValueError('frames.jsonl 存在无效完整帧记录')
                arrivals.append(arrived)
        walking = sum(start <= arrived < end for arrived in arrivals)
        result['walk_camera_frames'] = {'verified': bool(arrivals), 'complete_transport_frames': walking,
            'health_window_elapsed_s': end - start, 'transport_fps': walking / (end - start),
            'camera_arrival_span_covers_walk': bool(arrivals and arrivals[0] <= start and end <= arrivals[-1]),
            'frames_file': str(directory / 'frames.jsonl')}
    except Exception as error:
        result['walk_camera_frames'] = {'verified': False, 'error': str(error)}
    result['verified'] = bool(result['body_window_covers_walk'] and selected
                              and result['walk_camera_frames'].get('verified')
                              and result['walk_camera_frames'].get('camera_arrival_span_covers_walk'))
    if not result['verified']:
        result['reason'] = '完整行走覆盖或分段证据有遗漏；查看各 verified/coverage 字段'
    return result


def main(argv: list[str] | None = None) -> int:
    with PRINT_LOCK:
        TERMINAL_ERRORS.clear()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--socket', default='/private/tmp/md-runtime-ethan/robotd.sock',
                        help='传给已有行走探针的 robotd Unix socket')
    args = parser.parse_args(argv)
    directory = EVIDENCE_ROOT / ('walk_camera_' + dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    directory.mkdir(parents=True, exist_ok=False)
    observer = walk = None
    observer_summary = walk_summary = None
    cancelled = threading.Event()
    previous_signals = {sig: signal.signal(sig, lambda _sig, _frame: cancelled.set())
                        for sig in (signal.SIGINT, signal.SIGTERM)}
    report = {'schema_version': 1, 'outcome': 'not_started', 'started_wall_utc': utc_now(),
        'evidence_directory': str(directory), 'requested_observation_s': 75,
        'requested_walk_s': 60, 'ready_timeout_s': READY_TIMEOUT_S,
        'services_started_or_stopped_by_parent': False, 'gate_decision': 'not_assessed'}
    result = 1
    say(f'联合证据目录：{directory}')
    say('先确认完整相机帧，再行走 60 秒；75 秒观测含前后站立。')
    try:
        check_cancel(cancelled)
        observer = Child('observe', [str(PYTHON), '-u', str(PROJECT / 'work/probe_runtime_visual_load.py'),
            '--camera', '--seconds', '15', '--samples', '5', '--label', directory.name],
            directory, 'visual_load_')
        observer.start_logs()
        report['ready_frame'] = wait_ready(observer, cancelled)
        report['ready_confirmed_monotonic_s'] = time.monotonic()
        check_cancel(cancelled, observer)
        if observer.process.poll() is not None:
            raise RuntimeError('完整帧确认后观测已退出；未启动行走')
        say('第一条完整帧协议证据已取得，现在启动 60 秒行走探针。')
        walk = Child('walk', [str(PYTHON), '-u', str(PROJECT / 'work/probe_runtime_walk.py'),
            '--duration', '60', '--socket', args.socket], directory, 'runtime_walk_')
        walk.start_logs()
        while walk.process.poll() is None:
            check_cancel(cancelled, observer)
            check_cancel(cancelled, walk)
            if observer.process.poll() is not None:
                raise RuntimeError('行走尚未完成时只读观测已退出；中断行走以执行原有清理')
            if time.monotonic() - walk.started_monotonic_s > WALK_TIMEOUT_S:
                raise RuntimeError('行走子进程超过预期完成时间；请求中断并清理')
            time.sleep(0.05)
        walk.join_logs()
        walk_summary = walk.read_summary()
        if not walk_succeeded(walk_summary, walk):
            raise RuntimeError('行走未获得成功返回站立的子汇总证据')
        say('行走子汇总已确认返回站立；继续只读观察直到 5 个窗口完成。')
        while observer.process.poll() is None:
            check_cancel(cancelled, observer)
            if time.monotonic() - observer.started_monotonic_s > OBSERVER_TIMEOUT_S:
                raise RuntimeError('只读观测超过预期完成时间')
            time.sleep(0.05)
        observer.join_logs()
        observer_summary = observer.read_summary()
        if (observer.process.returncode != 0 or not observer_summary
                or observer_summary.get('outcome') != 'completed'
                or observer_summary.get('completed_windows') != 5
                or observer_summary.get('camera', {}).get('error')
                or observer_summary.get('camera', {}).get('frame_count', 0) < 1):
            raise RuntimeError('只读观测没有完整完成 5 个含相机的窗口')
        check_cancel(cancelled)
        report['outcome'] = 'children_completed'
        result = 0
    except KeyboardInterrupt:
        report['outcome'] = 'interrupted'
        result = 130
        say('联合试验已收到中断，先等待行走探针的原有清理。')
    except Exception:
        report['outcome'] = 'error'
        report['error'] = traceback.format_exc()
        say(report['error'])
    finally:
        # Always preserve this order. Never send signals to body, robotd, or unrelated PIDs.
        for child in (walk, observer):
            try:
                interrupt_child(child)
            except Exception:
                if child is not None:
                    child.errors.append('清理异常：' + traceback.format_exc())
                    child.shutdown['cleanup_exception'] = child.errors[-1]
                if child is walk and walk is not None:
                    say('停止/清理未获确认；请立即在 robotd 终端按 Ctrl+C。')
        for child in (walk, observer):
            if child is not None:
                try:
                    child.join_logs()
                except Exception:
                    child.errors.append('日志收尾异常：' + traceback.format_exc())
        if walk is not None and walk_summary is None:
            walk_summary = walk.read_summary()
        if observer is not None and observer_summary is None:
            observer_summary = observer.read_summary()
        report['children'] = {'observe': observer.record(observer_summary) if observer else None,
                              'walk': walk.record(walk_summary) if walk else None}
        report['walk_cleanup'] = cleanup_status(walk, walk_summary)
        try:
            report['window_analysis'] = analyze_windows(observer_summary, walk_summary,
                                                       observer.directory if observer else None)
        except Exception:
            report['window_analysis'] = {'verified': False, 'error': traceback.format_exc(),
                                         'gate_decision': 'not_assessed'}
        incomplete_logs = any(child.errors for child in (walk, observer) if child is not None)
        if result == 0 and (not report['window_analysis']['verified'] or incomplete_logs):
            report['outcome'] = 'children_completed_evidence_incomplete'
            result = 1
            say('两个子探针完成，但时间覆盖/分段证据存在遗漏；不把行走指标标为已验证。')
        if report['walk_cleanup']['manual_robotd_ctrl_c_required']:
            result = 1
            say(report['walk_cleanup']['instruction'])
        report['terminal_output_errors'] = list(TERMINAL_ERRORS)
        report['finished_wall_utc'] = utc_now()
        (directory / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n',
                                               encoding='utf-8')
        for sig, handler in previous_signals.items():
            signal.signal(sig, handler)
        say(f'父汇总：{directory / "summary.json"}；结果：{report["outcome"]}')
    return result


if __name__ == '__main__':
    sys.exit(main())
