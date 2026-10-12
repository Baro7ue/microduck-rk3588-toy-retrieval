#!/usr/bin/env python3
"""Read-only body realtime windows, optionally with raw head-camera transport evidence."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import select
import signal
import socket
import struct
import sys
import threading
import time
import traceback

PROJECT = Path('/Users/ethan/Documents/RK3588 Microduck')
EVIDENCE_ROOT = PROJECT / 'task/evidence/G2'
BODY_ADDRESS = ('127.0.0.1', 7801)
CAMERA_ADDRESS = ('127.0.0.1', 7901)
SOCKET_TIMEOUT_S = 2.0
WIDTH, HEIGHT = 640, 360
FRAME_BYTES = WIDTH * HEIGHT * 2
MAX_FRAME_BYTES = 1024 * 1024
MAX_JSON_BYTES = 1024 * 1024
CAMERA_LIMITATIONS = (
    '此协议没有源时间戳或帧编号；这里只测量完整负载到达的传输帧率，'
    '不能测量真实帧龄或唯一新渲染帧率。静态画面重复 SHA-256 不表示跳过拍摄。'
)


class FrameProtocolError(RuntimeError):
    pass


class StopRequested(Exception):
    pass


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec='microseconds')


def bounded_seconds(value: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or not 1 <= seconds <= 60:
        raise argparse.ArgumentTypeError('--seconds 必须为 1 到 60 之间的有限秒数')
    return seconds


def bounded_samples(value: str) -> int:
    count = int(value)
    if not 1 <= count <= 10:
        raise argparse.ArgumentTypeError('--samples 必须为 1 到 10 之间的整数')
    return count


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='只读比较仿真时间与本机单调时钟；可同时保存头部相机传输证据。')
    parser.add_argument('--seconds', type=bounded_seconds, default=10.0,
                        help='每个观察窗口的秒数，1–60，默认 10')
    parser.add_argument('--samples', type=bounded_samples, default=3,
                        help='观察窗口数量，1–10，默认 3')
    parser.add_argument('--label', default=None, help='证据标签，例如 baseline_no_camera')
    parser.add_argument('--camera', action='store_true',
                        help='另接 127.0.0.1:7901，保存收帧记录和首末帧 PNG')
    return parser.parse_args(argv)


class BodyClient:
    """The public request method only accepts hello and read; no control operations exist."""

    def __init__(self, address: tuple[str, int], rpc_times: list[dict]):
        self.address = address
        self.rpc_times = rpc_times
        self.socket: socket.socket | None = None
        self.buffer = bytearray()

    def connect(self) -> None:
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(SOCKET_TIMEOUT_S)
        self.socket.connect(self.address)

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close()
            self.socket = None

    def request(self, operation: str) -> dict:
        if operation not in {'hello', 'read'}:
            raise ValueError('只读客户端仅允许 hello/read')
        request = {'op': operation}
        if operation == 'hello':
            request['protocol'] = 1
        record = {'operation': operation, 'sent_wall_utc': utc_now(),
                  'sent_monotonic_s': time.monotonic()}
        try:
            assert self.socket is not None
            self.socket.settimeout(SOCKET_TIMEOUT_S)
            deadline = record['sent_monotonic_s'] + SOCKET_TIMEOUT_S
            self.socket.sendall((json.dumps(request) + '\n').encode())
            while b'\n' not in self.buffer:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('身体状态回复在 2 秒内未完成')
                self.socket.settimeout(remaining)
                chunk = self.socket.recv(4096)
                if not chunk:
                    raise ConnectionError('身体状态连接在回复完整数据前关闭')
                self.buffer.extend(chunk)
                if len(self.buffer) > MAX_JSON_BYTES:
                    raise ValueError('身体 JSON 回复超过 1 MiB 上限')
            raw, _, remaining = self.buffer.partition(b'\n')
            self.buffer = bytearray(remaining)
            record['received_monotonic_s'] = time.monotonic()
            record['received_wall_utc'] = utc_now()
            record['midpoint_monotonic_s'] = (
                record['sent_monotonic_s'] + record['received_monotonic_s']) / 2
            record['rtt_s'] = record['received_monotonic_s'] - record['sent_monotonic_s']
            record['response'] = json.loads(raw)
            response = record['response']
            if not isinstance(response, dict) or 'error' in response:
                raise RuntimeError(f'身体状态回复异常：{response!r}')
            if operation == 'hello' and response.get('protocol') != 1:
                raise RuntimeError(f'身体协议版本不匹配：{response!r}')
            return record
        except Exception:
            record['error'] = traceback.format_exc()
            raise
        finally:
            self.rpc_times.append(record)


def receive_exact(client: socket.socket, size: int, stopped: threading.Event) -> bytes:
    """Bound each packet part and check cancellation without a blocking two-second wait."""
    received = bytearray()
    deadline = time.monotonic() + SOCKET_TIMEOUT_S
    while len(received) < size:
        if stopped.is_set():
            raise StopRequested
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f'相机数据在 2 秒内未收齐：{len(received)}/{size} 字节')
        ready, _, _ = select.select([client], [], [], min(0.1, remaining))
        if not ready:
            continue
        chunk = client.recv(min(65536, size - len(received)))
        if not chunk:
            raise ConnectionError(f'相机连接提前关闭：{len(received)}/{size} 字节')
        received.extend(chunk)
    return bytes(received)


class FrameWorker(threading.Thread):
    """The worker exclusively owns and closes its camera socket."""

    def __init__(self, address: tuple[str, int], directory: Path, on_frame=None):
        super().__init__(name='visual-load-frames', daemon=True)
        self.address = address
        self.directory = directory
        self.on_frame = on_frame
        self.stopped = threading.Event()
        self.records: list[dict] = []
        self.error: str | None = None
        self.started_monotonic_s: float | None = None
        self.started_process_time_s: float | None = None
        self.ended_monotonic_s: float | None = None
        self.first_payload: bytes | None = None
        self.last_payload: bytes | None = None

    def run(self) -> None:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
                client.settimeout(SOCKET_TIMEOUT_S)
                client.connect(self.address)
                self.started_monotonic_s = time.monotonic()
                self.started_process_time_s = time.process_time()
                print('相机连接成功；正在记录完整帧到达时间和负载哈希。', flush=True)
                with (self.directory / 'frames.jsonl').open('w') as stream:
                    while not self.stopped.is_set():
                        receiving = time.monotonic()
                        header = receive_exact(client, 4, self.stopped)
                        size, = struct.unpack('<I', header)
                        if size > MAX_FRAME_BYTES:
                            raise FrameProtocolError(f'相机帧长度 {size} 超过 1 MiB 上限')
                        if size != FRAME_BYTES:
                            raise FrameProtocolError(
                                f'相机帧长度为 {size}，预期 {FRAME_BYTES} (640×360 UYVY)')
                        payload = receive_exact(client, size, self.stopped)
                        arrived = time.monotonic()
                        record = {'index': len(self.records) + 1,
                                  'arrived_monotonic_s': arrived,
                                  'arrived_wall_utc': utc_now(),
                                  'receive_duration_s': arrived - receiving,
                                  'payload_bytes': size,
                                  'payload_sha256': hashlib.sha256(payload).hexdigest()}
                        if self.first_payload is None:
                            self.first_payload = payload
                            (self.directory / 'camera_first.uyvy').write_bytes(payload)
                        self.last_payload = payload
                        self.records.append(record)
                        stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                        stream.flush()
                        if self.on_frame is not None:
                            self.on_frame(record, payload)
        except StopRequested:
            pass
        except Exception:
            self.error = traceback.format_exc()
        finally:
            self.ended_monotonic_s = time.monotonic()
            try:
                if self.last_payload is not None:
                    (self.directory / 'camera_last.uyvy').write_bytes(self.last_payload)
            except Exception:
                self.error = (self.error or '') + traceback.format_exc()

    def raise_if_failed(self) -> None:
        if self.error is not None:
            raise RuntimeError('相机读取线程失败：\n' + self.error)

    def stop(self) -> None:
        self.stopped.set()
        self.join(SOCKET_TIMEOUT_S + 0.5)
        if self.is_alive():
            raise RuntimeError('相机读取线程未在 2.5 秒内结束')

    def summary(self, windows: list[dict]) -> dict:
        count = len(self.records)
        elapsed = (self.ended_monotonic_s - self.started_monotonic_s
                   if self.ended_monotonic_s is not None and self.started_monotonic_s is not None
                   else None)
        first = self.records[0] if count else None
        last = self.records[-1] if count else None
        arrival_span = (last['arrived_monotonic_s'] - first['arrived_monotonic_s']
                        if count > 1 else None)
        repeated = sum(a['payload_sha256'] == b['payload_sha256']
                       for a, b in zip(self.records, self.records[1:]))
        report = {'requested': True, 'address': list(self.address),
                  'format': 'UYVY', 'raw_width': WIDTH, 'raw_height': HEIGHT,
                  'payload_bytes': FRAME_BYTES, 'frame_count': count,
                  'receiving_elapsed_s': elapsed,
                  'transport_fps': count / elapsed if elapsed is not None and elapsed > 0 else None,
                  'interarrival_fps': (count - 1) / arrival_span if arrival_span else None,
                  'first_frame': first, 'last_frame': last,
                  'repeated_consecutive_hash_count': repeated,
                  'limitations': CAMERA_LIMITATIONS, 'error': self.error,
                  'frames_file': 'frames.jsonl', 'png_rotation_clockwise_degrees': 90}
        for window in windows:
            if 'end_sample' not in window:
                continue
            start = window['start_sample']['midpoint_monotonic_s']
            end = window['end_sample']['midpoint_monotonic_s']
            arrivals = sum(start <= r['arrived_monotonic_s'] < end for r in self.records)
            window['camera_frames_received'] = arrivals
            window['camera_transport_fps'] = arrivals / (end - start)
        return report


def save_rotated_png(payload: bytes, destination: Path) -> None:
    """Decode simulator full-range BT.601 UYVY, then rotate 90 degrees clockwise."""
    import numpy as np
    from PIL import Image

    if len(payload) != FRAME_BYTES:
        raise FrameProtocolError('PNG 转换负载长度不匹配')
    pairs = np.frombuffer(payload, dtype=np.uint8).reshape(HEIGHT, WIDTH // 2, 4)
    y = np.empty((HEIGHT, WIDTH), dtype=np.float32)
    y[:, 0::2] = pairs[:, :, 1]
    y[:, 1::2] = pairs[:, :, 3]
    u = np.repeat(pairs[:, :, 0].astype(np.float32) - 128, 2, axis=1)
    v = np.repeat(pairs[:, :, 2].astype(np.float32) - 128, 2, axis=1)
    rgb = np.stack((y + 1.402 * v, y - 0.344136 * u - 0.714136 * v,
                    y + 1.772 * u), axis=2)
    pixels = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    Image.fromarray(pixels).transpose(Image.Transpose.ROTATE_270).save(destination)


def interrupt(_signum: int, _frame: object) -> None:
    raise KeyboardInterrupt


def wait_until(deadline: float, worker: FrameWorker | None) -> None:
    while True:
        if worker is not None:
            worker.raise_if_failed()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(0.1, remaining))


def sample_body(client: BodyClient, states_file) -> dict:
    record = client.request('read')
    states_file.write(json.dumps(record, ensure_ascii=False) + '\n')
    states_file.flush()
    sim_time = record['response'].get('sim_time')
    if isinstance(sim_time, bool) or not isinstance(sim_time, (int, float)) \
            or not math.isfinite(sim_time) or sim_time < 0:
        raise ValueError(f'身体回复中的 sim_time 无效：{sim_time!r}')
    record['sim_time_s'] = float(sim_time)
    return record


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    directory = EVIDENCE_ROOT / f'visual_load_{stamp}'
    directory.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': 1, 'label': args.label, 'outcome': 'not_started',
              'started_wall_utc': utc_now(), 'evidence_directory': str(directory),
              'body_address': list(BODY_ADDRESS), 'read_only_operations': ['hello', 'read'],
              'seconds_per_window': args.seconds, 'requested_windows': args.samples,
              'timing_method': 'delta sim_time / delta monotonic RPC midpoint',
              'rpc_times': [], 'windows': [], 'states_file': 'states.jsonl',
              'camera': {'requested': args.camera, 'limitations': CAMERA_LIMITATIONS}}
    print(f'证据目录：{directory}', flush=True)
    print(f'只读观测：{args.samples} 个窗口，每个 {args.seconds:g} 秒；标签：{args.label or "未指定"}。',
          flush=True)
    print('身体连接只发送 hello/read；不会启用、关闭或改变行走指令。', flush=True)
    print('实时因子仅保存原始数值，不在脚本中判定通过。', flush=True)
    if args.camera:
        print(CAMERA_LIMITATIONS, flush=True)
    client = BodyClient(BODY_ADDRESS, report['rpc_times'])
    worker = None
    result = 1
    previous_sigterm = signal.signal(signal.SIGTERM, interrupt)
    try:
        with (directory / 'states.jsonl').open('w') as states_file:
            client.connect()
            report['hello'] = client.request('hello')['response']
            print('身体协议 1 已确认。', flush=True)
            if args.camera:
                worker = FrameWorker(CAMERA_ADDRESS, directory)
                worker.start()
            for index in range(1, args.samples + 1):
                if worker is not None:
                    worker.raise_if_failed()
                window = {'index': index, 'outcome': 'incomplete'}
                report['windows'].append(window)
                start = sample_body(client, states_file)
                window['start_sample'] = start
                print(f'窗口 {index}/{args.samples}：开始观察。', flush=True)
                wait_until(start['midpoint_monotonic_s'] + args.seconds, worker)
                end = sample_body(client, states_file)
                window['end_sample'] = end
                wall_delta = end['midpoint_monotonic_s'] - start['midpoint_monotonic_s']
                sim_delta = end['sim_time_s'] - start['sim_time_s']
                window.update(wall_delta_s=wall_delta, sim_delta_s=sim_delta)
                if sim_delta < 0:
                    raise RuntimeError('窗口内 sim_time 倒退；仿真可能已重启')
                if wall_delta <= 0:
                    raise RuntimeError('本机单调时钟间隔无效')
                window.update(realtime_factor=sim_delta / wall_delta, outcome='completed')
                print(f'窗口 {index}：仿真 {sim_delta:.6f} 秒 / 本机 {wall_delta:.6f} 秒'
                      f' = {window["realtime_factor"]:.6f}x；'
                      f'读状态往返 {start["rtt_s"] * 1000:.3f}/{end["rtt_s"] * 1000:.3f} ms。',
                      flush=True)
            if worker is not None:
                worker.raise_if_failed()
            report['outcome'] = 'completed'
            result = 0
    except KeyboardInterrupt:
        report['outcome'] = 'interrupted'
        result = 130
        print('观测已中断，正在保存已有证据。', flush=True)
    except Exception:
        report['outcome'] = 'error'
        report['error'] = traceback.format_exc()
        traceback.print_exc()
    finally:
        client.close()
        if worker is not None:
            try:
                worker.stop()
                worker.raise_if_failed()
                if not worker.records and report['outcome'] == 'completed':
                    raise RuntimeError('请求了相机观测，但未收到任何完整相机帧')
            except Exception:
                error = traceback.format_exc()
                report['error'] = (report.get('error', '') + error)
                traceback.print_exc()
                if report['outcome'] != 'interrupted':
                    report['outcome'] = 'error'
                    result = 1
            report['camera'] = worker.summary(report['windows'])
            try:
                for suffix, payload in (('first', worker.first_payload), ('last', worker.last_payload)):
                    if payload is not None:
                        name = f'camera_{suffix}_rotated.png'
                        save_rotated_png(payload, directory / name)
                        report['camera'][f'{suffix}_png'] = name
            except Exception:
                report['png_error'] = traceback.format_exc()
                traceback.print_exc()
                if report['outcome'] != 'interrupted':
                    report['outcome'] = 'error'
                    result = 1
            fps = report['camera']['transport_fps']
            print(f'相机：收到 {report["camera"]["frame_count"]} 个完整负载；'
                  f'传输帧率 {fps:.3f} fps。' if fps is not None else '相机：未收到完整负载。', flush=True)
        report['completed_windows'] = sum(w['outcome'] == 'completed' for w in report['windows'])
        report['finished_wall_utc'] = utc_now()
        (directory / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        signal.signal(signal.SIGTERM, previous_sigterm)
        print(f'结果：{report["outcome"]}；汇总：{directory / "summary.json"}', flush=True)
    return result


if __name__ == '__main__':
    sys.exit(main())
