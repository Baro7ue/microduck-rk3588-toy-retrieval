"""Plan 2F: the board's live detector (Stage 3 uses it as is) and the deployment measurement.

On the board, with robotd s2 running there and the Mac sim streaming its head camera:

    python -m microduck_toy.stage2.deploy --backend cpu|fp16|int8 --model models/<file> \
        --camera 192.168.50.1:7901 --socket <board>/run/robotd.sock --out <dir>

On the Mac, before (toys around the robot, through the arena body's test port) and after (the table):

    python -m microduck_toy.stage2.deploy --place
    python -m microduck_toy.stage2.deploy --table runs/<UTC>_s2f_board_deploy

The live path, one camera frame:
    receive     the 640x360 UYVY payload arrives on TCP (4-byte little-endian length first);
                FrameReceiver keeps only the newest frame
    preprocess  camera pose from the newest robot.state -> horizon row -> window (search) or
                letterbox (approach); UYVY decode (only the columns the window needs), upright,
                crop and scale to 320, then the backend's input (ONNX: float NCHW, NPU: uint8 NHWC)
    inference   ONNX Runtime on the CPU (split FP32 ONNX), or RKNNLite on the NPU (FP16 / INT8)
    postprocess decode_split (DFL, class scores, NMS), boxes back to frame pixels
The four times, the wait between "frame complete" and "processing starts", and the end-to-end
time from the frame's first byte to boxes out are kept per frame (frames.jsonl). Every second the
sampler adds system CPU, this process's CPU, robotd's CPU, the NPU load (debugfs, through
`sudo -n`; left out when sudo would ask) and the SoC temperature (resources.csv). robotd's own
numbers come from its state stream: every tick's time, missed count and rate.

Meanwhile the robot stands and turns in place (config deploy.motion), at the search head.
summary.json: percentiles, rates, CPU and NPU means, robotd ticks/missed/period, versions, pass,
and the cores the process was allowed (the launcher pins it with taskset, env.sh MD_BOARD_CPUS).
"""
from __future__ import annotations

import argparse
import datetime as dt
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import platform
import re
import select
import socket
import struct
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

from microduck_toy.control import ranging
from microduck_toy.control.robot import Events, Robot, UPRIGHT
from microduck_toy.paths import repo_root
from microduck_toy.vision import uyvy, views
from microduck_toy.vision.detect import OnnxYoloSplit, RknnYolo, decode_split

BACKENDS = ('cpu', 'fp16', 'int8')
RAW_W, RAW_H = 640, 360
FRAME_BYTES = RAW_W * RAW_H * 2
SEGMENTS = ('receive', 'wait', 'preprocess', 'inference', 'postprocess', 'e2e')
NPU_LOAD = '/sys/kernel/debug/rknpu/load'
TICK_S = 0.02


def percentiles(values) -> dict | None:
    if not len(values):
        return None
    a = np.asarray(values, float)
    return {'n': int(a.size), 'p50': round(float(np.percentile(a, 50)), 3),
            'p90': round(float(np.percentile(a, 90)), 3), 'p99': round(float(np.percentile(a, 99)), 3),
            'max': round(float(a.max()), 3), 'mean': round(float(a.mean()), 3)}


# ── camera ──────────────────────────────────────────────────────────────────

class Frame:
    __slots__ = ('seq', 'payload', 'first', 'done')

    def __init__(self, seq: int, payload: bytes, first: float, done: float):
        self.seq, self.payload, self.first, self.done = seq, payload, first, done


class FrameReceiver(threading.Thread):
    """The sim head camera over TCP. Only the newest complete frame is kept; `first` is when its
    length header arrived and `done` when the last payload byte did (time.perf_counter)."""

    def __init__(self, address: tuple[str, int], timeout: float = 2.0):
        super().__init__(name='camera', daemon=True)
        self.address, self.timeout = address, timeout
        self.cond = threading.Condition()
        self.latest: Frame | None = None
        self.count = 0
        self.error: str | None = None
        self.stopped = threading.Event()
        self.connected: float | None = None

    def _exact(self, sock: socket.socket, size: int) -> tuple[bytes, float]:
        buf = bytearray()
        first = None
        deadline = time.monotonic() + self.timeout
        while len(buf) < size:
            if self.stopped.is_set():
                raise EOFError('stopped')
            if time.monotonic() > deadline:
                raise TimeoutError(f'camera: {len(buf)}/{size} bytes in {self.timeout} s')
            ready, _, _ = select.select([sock], [], [], 0.1)
            if not ready:
                continue
            chunk = sock.recv(min(1 << 16, size - len(buf)))
            if not chunk:
                raise ConnectionError(f'camera closed after {len(buf)}/{size} bytes')
            if first is None:
                first = time.perf_counter()
            buf.extend(chunk)
        return bytes(buf), first

    def run(self) -> None:
        try:
            with socket.create_connection(self.address, timeout=self.timeout) as sock:
                self.connected = time.perf_counter()
                while not self.stopped.is_set():
                    header, first = self._exact(sock, 4)
                    size, = struct.unpack('<I', header)
                    if size != FRAME_BYTES:
                        raise ValueError(f'camera frame of {size} bytes, expected {FRAME_BYTES} '
                                         f'({RAW_W}x{RAW_H} UYVY)')
                    payload, _ = self._exact(sock, size)
                    done = time.perf_counter()
                    with self.cond:
                        self.count += 1
                        self.latest = Frame(self.count, payload, first, done)
                        self.cond.notify_all()
        except EOFError:
            pass
        except Exception as error:      # reported by take_after
            with self.cond:
                self.error = f'{type(error).__name__}: {error}'
                self.cond.notify_all()

    def take_after(self, seq: int, timeout: float) -> Frame | None:
        """The newest frame after `seq`, waiting up to `timeout` s for one."""
        with self.cond:
            self.cond.wait_for(lambda: self.error or (self.latest and self.latest.seq > seq),
                               timeout)
            if self.error:
                raise RuntimeError(self.error)
            return self.latest if self.latest and self.latest.seq > seq else None

    def stop(self) -> None:
        self.stopped.set()
        self.join(self.timeout + 0.5)


# ── detector ────────────────────────────────────────────────────────────────

def mapping_for(state: dict | None, view: str, margin_rows: int) -> views.Mapping:
    """Search: the window under the horizon of the robot's own camera pose; approach, or no pose
    (or a camera looking straight down): the letterboxed whole frame."""
    if view == 'search' and state is not None:
        horizon = views.horizon_row(ranging.camera_in_level(state)[1])
        if horizon is not None:
            return views.window_mapping(horizon, margin_rows)
    return views.letterbox_mapping()


def frame_input(payload: bytes, m: views.Mapping) -> np.ndarray:
    """The 320x320 RGB input from a raw UYVY payload. Upright row r is raw column r, so a window
    only needs its 360 raw columns decoded; the result equals views.to_input on the whole frame."""
    pairs = np.frombuffer(payload, np.uint8).reshape(RAW_H, RAW_W // 2, 4)
    if m.kind != 'window':
        return views.to_input(np.ascontiguousarray(np.rot90(uyvy.decode(pairs), 3)), m)
    y0 = int(m.y0)
    a, b = y0 // 2, (y0 + views.WINDOW + 1) // 2
    band = np.rot90(uyvy.decode(pairs[:, a:b]), 3)[y0 - 2 * a:y0 - 2 * a + views.WINDOW]
    return cv2.resize(np.ascontiguousarray(band), (views.INPUT, views.INPUT),
                      interpolation=cv2.INTER_LINEAR)


class LiveDetector:
    """One backend, the three timed steps. Boxes come back in frame pixels (`xyxy`) and input
    pixels (`xyxy_input`), with the mapping used."""

    def __init__(self, backend: str, model: Path, names: list[str], conf: float, iou: float,
                 threads: int = 2, net=None):
        if backend not in BACKENDS:
            raise ValueError(f'backend {backend!r} not in {BACKENDS}')
        self.backend, self.conf, self.iou = backend, conf, iou
        if net is not None:
            self.net = net
        elif backend == 'cpu':
            self.net = OnnxYoloSplit(model, threads=threads, iou=iou)
        else:
            self.net = RknnYolo.board(model, names, iou=iou)
        self.names = list(self.net.names)
        if self.names != list(names):
            raise ValueError(f'model classes {self.names} differ from the config {names}')

    def warm(self, n: int = 5) -> None:
        zero = np.zeros((views.INPUT, views.INPUT, 3), np.uint8)
        for _ in range(n):
            self.net.run(self.net.prepare(zero, bgr=False))

    def preprocess(self, payload: bytes, m: views.Mapping) -> np.ndarray:
        return self.net.prepare(frame_input(payload, m), bgr=False)

    def infer(self, x: np.ndarray):
        return self.net.run(x)

    def postprocess(self, outs, m: views.Mapping) -> list[dict]:
        boxes, scores, classes = decode_split(outs, self.conf, self.iou)
        return [{'cls': self.names[int(c)], 'conf': float(s),
                 'xyxy': [round(v, 2) for v in views.to_frame_box(b, m)],
                 'xyxy_input': [round(float(v), 2) for v in b]}
                for b, s, c in zip(boxes, scores, classes)]

    def detect(self, frame: Frame, state: dict | None, view: str, margin_rows: int):
        """(boxes, mapping, times in ms) for one frame."""
        start = time.perf_counter()
        m = mapping_for(state, view, margin_rows)
        x = self.preprocess(frame.payload, m)
        t1 = time.perf_counter()
        outs = self.infer(x)
        t2 = time.perf_counter()
        dets = self.postprocess(outs, m)
        t3 = time.perf_counter()
        ms = {'receive': frame.done - frame.first, 'wait': start - frame.done,
              'preprocess': t1 - start, 'inference': t2 - t1, 'postprocess': t3 - t2,
              'e2e': t3 - frame.first}
        return dets, m, {k: round(v * 1000.0, 3) for k, v in ms.items()}

    def close(self) -> None:
        close = getattr(self.net, 'close', None)
        if close:
            close()


# ── board resources ─────────────────────────────────────────────────────────

def cpu_jiffies() -> tuple[int, int, int] | None:
    """(busy, total) jiffies over all cores from /proc/stat, and the core count."""
    try:
        lines = Path('/proc/stat').read_text().splitlines()
    except OSError:
        return None
    fields = [int(v) for v in lines[0].split()[1:]]
    idle = fields[3] + (fields[4] if len(fields) > 4 else 0)
    cores = sum(1 for line in lines if re.match(r'cpu\d', line))
    return sum(fields) - idle, sum(fields), cores


def proc_ticks(pid: int) -> int | None:
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    except OSError:
        return None
    return int(fields[11]) + int(fields[12])


def peer_pid(path: str) -> int | None:
    """robotd's PID from its control socket (Linux SO_PEERCRED)."""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(2.0)
            client.connect(path)
            cred = client.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize('3i'))
        return struct.unpack('3i', cred)[0]
    except (OSError, AttributeError):
        return None


def parse_npu_load(text: str) -> list[int]:
    """'NPU load:  Core0: 12%, Core1:  0%, Core2:  0%,' -> [12, 0, 0]."""
    return [int(v) for _, v in sorted((int(c), v) for c, v in re.findall(r'Core(\d+):\s*(\d+)%', text))]


def npu_load() -> tuple[list[int] | None, str | None]:
    try:
        r = subprocess.run(['sudo', '-n', 'cat', NPU_LOAD], capture_output=True, text=True, timeout=2)
    except (OSError, subprocess.TimeoutExpired) as error:
        return None, str(error)
    if r.returncode != 0:
        return None, (r.stderr or r.stdout).strip()[:200] or f'exit {r.returncode}'
    loads = parse_npu_load(r.stdout)
    return (loads, None) if loads else (None, f'unreadable: {r.stdout.strip()[:120]}')


def soc_temp_c() -> float | None:
    try:
        return int(Path('/sys/class/thermal/thermal_zone0/temp').read_text()) / 1000.0
    except (OSError, ValueError):
        return None


class Sampler(threading.Thread):
    """Once a second: system CPU (% of all cores), this process and robotd (% of one core),
    NPU load per core, SoC temperature. Rows go to resources.csv."""

    COLUMNS = ('t_s', 'system_cpu_pct', 'self_cpu_pct_core', 'robotd_cpu_pct_core', 'npu_load',
               'temp_c')

    def __init__(self, out: Path, robotd_pid: int | None, npu: bool, period: float = 1.0):
        super().__init__(name='sampler', daemon=True)
        self.out, self.pid, self.npu, self.period = out, robotd_pid, npu, period
        self.rows: list[dict] = []
        self.npu_error: str | None = None
        self.stopped = threading.Event()
        self.hz = os.sysconf('SC_CLK_TCK') if hasattr(os, 'sysconf') else 100

    def run(self) -> None:
        with self.out.open('w') as f:
            f.write(','.join(self.COLUMNS) + '\n')
            t0 = time.perf_counter()
            prev = (cpu_jiffies(), time.process_time(), proc_ticks(self.pid) if self.pid else None,
                    t0)
            while not self.stopped.wait(self.period):
                now = time.perf_counter()
                cur = (cpu_jiffies(), time.process_time(),
                       proc_ticks(self.pid) if self.pid else None, now)
                wall = now - prev[3]
                row = {'t_s': round(now - t0, 2)}
                if cur[0] and prev[0] and cur[0][1] > prev[0][1]:
                    row['system_cpu_pct'] = round(100.0 * (cur[0][0] - prev[0][0])
                                                  / (cur[0][1] - prev[0][1]), 2)
                row['self_cpu_pct_core'] = round(100.0 * (cur[1] - prev[1]) / wall, 2)
                if cur[2] is not None and prev[2] is not None:
                    row['robotd_cpu_pct_core'] = round(100.0 * (cur[2] - prev[2]) / self.hz / wall, 2)
                if self.npu and self.npu_error is None:
                    loads, self.npu_error = npu_load()
                    if loads is not None:
                        row['npu_load'] = loads
                row['temp_c'] = soc_temp_c()
                self.rows.append(row)
                f.write(','.join('' if row.get(c) is None else
                                 ('/'.join(map(str, row[c])) if isinstance(row[c], list) else str(row[c]))
                                 for c in self.COLUMNS) + '\n')
                f.flush()
                prev = cur

    def stop(self) -> None:
        self.stopped.set()
        self.join(self.period + 1.0)

    def summary(self) -> dict:
        def mean(key):
            v = [r[key] for r in self.rows if r.get(key) is not None]
            return round(float(np.mean(v)), 2) if v else None
        out = {'samples': len(self.rows), 'system_cpu_pct_mean': mean('system_cpu_pct'),
               'self_cpu_pct_core_mean': mean('self_cpu_pct_core'),
               'robotd_cpu_pct_core_mean': mean('robotd_cpu_pct_core'),
               'temp_c_max': max((r['temp_c'] for r in self.rows if r.get('temp_c') is not None),
                                 default=None),
               'cores': (cpu_jiffies() or (0, 0, None))[2]}
        loads = [r['npu_load'] for r in self.rows if r.get('npu_load')]
        if loads:
            a = np.asarray(loads, float)
            out['npu_load_pct_mean'] = [round(float(v), 1) for v in a.mean(axis=0)]
            out['npu_load_pct_max'] = [int(v) for v in a.max(axis=0)]
        elif self.npu:
            out['npu_load_error'] = self.npu_error or 'no samples'
        return out


class TickTap:
    """robotd's state stream, one tuple per tick: (received, t, missed, hz, policy). The state
    carries the loop's numbers as `loop: {hz, missed}` (robotd proto LoopState). Only ticks
    received inside the measurement window (start() .. stop(), time.monotonic) are summarised."""

    def __init__(self):
        self.rows: list[tuple] = []
        self.lock = threading.Lock()
        self.window = (None, None)

    def __call__(self, state: dict, received: float) -> None:
        loop = state.get('loop') or {}
        with self.lock:
            self.rows.append((received, float(state.get('t', 0.0)), loop.get('missed'),
                              loop.get('hz'), state.get('policy')))

    def start(self) -> None:
        self.window = (time.monotonic(), None)

    def stop(self) -> None:
        self.window = (self.window[0], time.monotonic())

    def summary(self, stage0: dict) -> dict:
        lo, hi = self.window
        with self.lock:
            rows = [r for r in self.rows if (lo is None or r[0] >= lo) and (hi is None or r[0] <= hi)]
        if len(rows) < 2:
            return {'ticks_seen': len(rows), 'pass': False, 'rule_text': 'no robot.state stream'}
        periods = np.diff([r[1] for r in rows]) * 1000.0
        missed = [r[2] for r in rows if r[2] is not None]
        hz = [r[3] for r in rows if r[3]]
        span = rows[-1][1] - rows[0][1]
        ticks = int(round(span / TICK_S)) + 1
        missed_delta = (missed[-1] - missed[0]) if missed else None
        rate0 = stage0['missed'] / stage0['ticks']
        allowed = max(1, math.ceil(rate0 * ticks))
        return {'ticks_seen': len(rows), 'ticks_expected': ticks, 'span_s': round(span, 2),
                'stream_gaps': int(np.sum(periods > 1.5 * TICK_S * 1000.0)),
                'period_ms': percentiles(periods), 'missed': missed_delta,
                'missed_per_10k': None if missed_delta is None else round(1e4 * missed_delta / ticks, 2),
                'stage0_missed_per_10k': round(1e4 * rate0, 2),
                'hz_min': round(min(hz), 2) if hz else None,
                'hz_mean': round(float(np.mean(hz)), 2) if hz else None,
                'policies': sorted({r[4] for r in rows if r[4]}),
                'pass': missed_delta is not None and missed_delta <= allowed,
                'rule_text': f'missed <= {allowed} (Stage 0 rate {stage0["missed"]}/{stage0["ticks"]} '
                             f'over {ticks} ticks, at least 1)'}


# ── versions ────────────────────────────────────────────────────────────────

def librknnrt_version(paths=('/usr/lib/librknnrt.so', '/usr/lib64/librknnrt.so',
                             '/usr/local/lib/librknnrt.so')) -> str | None:
    for p in paths:
        try:
            data = Path(p).read_bytes()
        except OSError:
            continue
        m = re.search(rb'librknnrt version: ([0-9][0-9A-Za-z.\-_ ()]*)', data)
        return m.group(1).decode(errors='replace').strip() if m else f'{p} (no version string)'
    return None


def versions(detector: LiveDetector | None) -> dict:
    out = {'python': platform.python_version(), 'numpy': np.__version__, 'opencv': cv2.__version__,
           'kernel': platform.release(), 'machine': platform.machine()}
    for pkg in ('onnxruntime', 'rknn-toolkit-lite2'):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    out['librknnrt'] = librknnrt_version()
    runtime = getattr(getattr(detector, 'net', None), 'runtime', None)
    if runtime is not None and hasattr(runtime, 'get_sdk_version'):
        try:
            out['rknn_sdk'] = str(runtime.get_sdk_version())
        except Exception as error:      # informational only
            out['rknn_sdk'] = f'error: {error}'
    try:
        r = subprocess.run(['sudo', '-n', 'cat', '/sys/kernel/debug/rknpu/version'],
                           capture_output=True, text=True, timeout=2)
        out['rknpu_driver'] = r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        out['rknpu_driver'] = None
    return out


# ── the measurement ─────────────────────────────────────────────────────────

def motion_phase(motion: list, elapsed: float) -> str:
    cycle = sum(float(s) for _, s in motion)
    t = elapsed % cycle
    for name, seconds in motion:
        if t < float(seconds):
            return name
        t -= float(seconds)
    return motion[-1][0]


def measure(args, cfg: dict, stage1: dict) -> int:
    d = cfg['deploy']
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    print(f'运行目录：{out}', flush=True)
    frames_wanted = int(args.frames or d['frames'])
    warmup = int(d['warmup_frames'])
    margin = int(cfg['input']['window_margin_rows'])
    wz = float(stage1['approach']['turn_wz_radps'])
    speeds = {'stand': 0.0, 'left': wz, 'right': -wz}
    host, port = args.camera.rsplit(':', 1)
    summary = {'backend': args.backend, 'model': str(args.model), 'view': d['view'],
               'head': d['head'], 'motion': d['motion'], 'turn_wz_radps': wz,
               'frames_wanted': frames_wanted, 'warmup_frames': warmup, 'cpus': allowed_cpus(),
               'started_utc': dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
               'outcome': 'failed'}
    events = Events(out / 'events.jsonl')
    tap = TickTap()
    robot = detector = rx = sampler = None
    records: list[dict] = []
    status = 1
    try:
        robot = Robot(args.socket, None, events, on_state=tap)
        state = robot.wait(lambda s: True, 5.0, 'first robot.state')
        if state['policy'] == 'held':
            robot.request('robot.enable', {'on': True})
        robot.wait(lambda s: s['policy'] in UPRIGHT, 15.0, 'policy takeover')
        robot.move(0.0, 0.0, 0.0)
        robot.head(tuple(d['head']))
        detector = LiveDetector(args.backend, Path(args.model), cfg['classes'],
                                float(cfg['detect']['conf']), float(cfg['detect']['iou']),
                                threads=int(d['cpu_threads']))
        detector.warm()
        summary['versions'] = versions(detector)
        sampler = Sampler(out / 'resources.csv', peer_pid(args.socket), args.backend != 'cpu')
        rx = FrameReceiver((host, int(port)))
        rx.start()
        time.sleep(2.0)                                 # head settles; the first frames arrive
        sampler.start()
        tap.start()
        t0 = time.perf_counter()
        deadline = t0 + (warmup + frames_wanted) / 15.0 * 2.0 + 30.0
        phase, seq, falls0 = None, 0, len(robot.falls)
        with (out / 'frames.jsonl').open('w') as f:
            while len(records) < warmup + frames_wanted:
                now = time.perf_counter()
                if now > deadline:
                    raise TimeoutError(f'only {len(records)} frames in {now - t0:.0f} s')
                p = motion_phase(d['motion'], now - t0)
                if p != phase:
                    robot.move(0.0, 0.0, speeds[p])
                    phase = p
                frame = rx.take_after(seq, 1.0)
                if frame is None:
                    continue
                try:
                    state = robot.snapshot()
                except TimeoutError:
                    state = None
                dets, m, ms = detector.detect(frame, state, d['view'], margin)
                rec = {'i': len(records), 'seq': frame.seq, 'skipped': frame.seq - seq - 1,
                       'phase': phase, 'policy': state['policy'] if state else None,
                       'map': [m.kind, m.y0], 'ms': ms, 'n': len(dets),
                       'dets': [[x['cls'], round(x['conf'], 3)] for x in dets[:6]]}
                seq = frame.seq
                records.append(rec)
                f.write(json.dumps(rec) + '\n')
                if len(records) % 150 == 0:
                    print(f'{len(records)} 帧 | 端到端 {ms["e2e"]:.1f} ms | 推理 {ms["inference"]:.1f} ms'
                          f' | {phase}', flush=True)
        tap.stop()
        summary['elapsed_s'] = round(time.perf_counter() - t0, 2)
        summary['falls'] = len(robot.falls) - falls0
        summary['outcome'] = 'completed'
        status = 0
    except KeyboardInterrupt:
        summary['outcome'] = 'interrupted'
        status = 130
    except Exception as error:
        summary['error'] = f'{type(error).__name__}: {error}'
        print(summary['error'], file=sys.stderr, flush=True)
    finally:
        if tap.window[0] is not None and tap.window[1] is None:
            tap.stop()
        if sampler is not None and sampler.is_alive():
            sampler.stop()
        if rx is not None:
            rx.stop()
            summary['received_frames'] = rx.count
        if robot is not None:
            try:
                robot.move(0.0, 0.0, 0.0)
                robot.head('H0')
                time.sleep(0.3)
            finally:
                robot.close()
        if detector is not None:
            detector.close()
        events.close()
        kept = records[warmup:]
        summary['frames'] = len(kept)
        summary['ms'] = {s: percentiles([r['ms'][s] for r in kept]) for s in SEGMENTS}
        summary['skipped_frames'] = sum(r['skipped'] for r in kept)
        if kept and summary.get('elapsed_s'):
            summary['processed_fps'] = round(len(records) / summary['elapsed_s'], 2)
        summary['boxes_per_frame'] = round(float(np.mean([r['n'] for r in kept])), 3) if kept else None
        summary['resources'] = sampler.summary() if sampler is not None else None
        summary['robotd'] = tap.summary(d['stage0_robotd'])
        e2e = (summary['ms'].get('e2e') or {}).get('p99')
        summary['e2e_pass'] = e2e is not None and e2e <= float(d['e2e_p99_ms_max'])
        summary['pass'] = bool(summary['outcome'] == 'completed' and summary['e2e_pass']
                               and summary['robotd']['pass'] and len(kept) >= frames_wanted)
        summary['rule_text'] = (f'end-to-end P99 <= {d["e2e_p99_ms_max"]} ms and '
                                f'{summary["robotd"]["rule_text"]}')
        (out / 'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False) + '\n')
        print(json.dumps({'backend': args.backend, 'frames': len(kept), 'e2e': summary['ms']['e2e'],
                          'robotd_missed': summary['robotd'].get('missed'), 'pass': summary['pass']},
                         ensure_ascii=False), flush=True)
    return status


# ── on the Mac ──────────────────────────────────────────────────────────────

def place(cfg: dict, stage1: dict, probe_addr: str) -> dict:
    """Put the robot and the toys of the config's layout seed on the field (arena body test port)."""
    from microduck_toy.stage0.link import Probe
    from microduck_toy.stage2 import datagen
    scene = datagen.sample(int(cfg['deploy']['layout_seed']), 'test_walk', cfg, stage1)
    host, port = probe_addr.rsplit(':', 1)
    probe = Probe(host, int(port))
    try:
        floor = probe.call('info').get('floor')
        probe.call('layout', robot=scene['robot'], toys=scene['toys'], hold_s=1.0)
    finally:
        probe.close()
    return {'seed': scene['seed'], 'floor': floor, 'robot': scene['robot'],
            'toys': sorted(scene['toys'])}


def allowed_cpus() -> list[int] | None:
    """The cores this process may run on (taskset); None where the OS does not say (macOS)."""
    return sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None


def cpus_text(cpus: list[int] | None) -> str:
    """[4, 5, 6, 7] -> '4-7', [0, 2, 3] -> '0,2-3'."""
    if not cpus:
        return '–'
    parts, start = [], cpus[0]
    for a, b in zip(cpus, cpus[1:] + [None]):
        if b != a + 1:
            parts.append(str(start) if start == a else f'{start}-{a}')
            start = b
    return ','.join(parts)


def table(run: Path, pack: Path | None = None) -> dict:
    """runs/<..>_s2f_board_deploy/board/<backend>/summary.json -> deploy_table.json + .md."""
    rows = {}
    for b in BACKENDS:
        p = run / 'board' / b / 'summary.json'
        if p.exists():
            rows[b] = json.loads(p.read_text())
    if not rows:
        raise SystemExit(f'no board/<backend>/summary.json under {run}')
    mac = run / 'mac_versions.json'
    extra = {'mac': json.loads(mac.read_text()) if mac.exists() else None}
    if pack is not None:
        for name in ('export.json', 'conversion_fp16.json', 'conversion_int8.json'):
            p = pack / 'models' / name
            if p.exists():
                c = json.loads(p.read_text())
                extra[name] = {k: c[k] for k in ('ultralytics', 'torch', 'onnx', 'rknn_toolkit2',
                                                 'toolkit', 'algo', 'weights_sha256')
                               if k in c}
    out = {'backends': {}, 'versions': {b: r.get('versions') for b, r in rows.items()}, **extra}
    lines = ['| 档 | 核 | 帧数 | 收帧 P50/P99 | 预处理 P50/P99 | 推理 P50/P99 | 后处理 P50/P99 | '
             '端到端 P50/P99 | 处理帧率 | 本进程 CPU | 系统 CPU | NPU 负载 | robotd 漏拍 | 通过 |',
             '|' + '---|' * 14]
    for b, r in rows.items():
        ms = r.get('ms') or {}
        res = r.get('resources') or {}
        rb = r.get('robotd') or {}
        out['backends'][b] = {
            'cpus': r.get('cpus'), 'frames': r.get('frames'), 'ms': ms,
            'processed_fps': r.get('processed_fps'),
            'skipped_frames': r.get('skipped_frames'), 'resources': res, 'robotd': rb,
            'falls': r.get('falls'), 'e2e_pass': r.get('e2e_pass'), 'pass': r.get('pass'),
            'outcome': r.get('outcome')}

        def pp(seg):
            s = ms.get(seg)
            return f'{s["p50"]:.1f} / {s["p99"]:.1f}' if s else '–'
        npu = res.get('npu_load_pct_mean')
        lines.append(f'| {b} | {cpus_text(r.get("cpus"))} | {r.get("frames")} | {pp("receive")} | '
                     f'{pp("preprocess")} | {pp("inference")} | {pp("postprocess")} | {pp("e2e")} | '
                     f'{r.get("processed_fps")} | {res.get("self_cpu_pct_core_mean")} % | '
                     f'{res.get("system_cpu_pct_mean")} % | '
                     f'{"/".join(str(v) for v in npu) + " %" if npu else "–"} | '
                     f'{rb.get("missed")} / {rb.get("ticks_expected")} | '
                     f'{"是" if r.get("pass") else "否"} |')
    (run / 'deploy_table.json').write_text(json.dumps(out, indent=1, ensure_ascii=False) + '\n')
    (run / 'deploy_table.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--backend', choices=BACKENDS)
    ap.add_argument('--model', type=Path)
    ap.add_argument('--camera', default='192.168.50.1:7901', help='sim head camera HOST:PORT')
    ap.add_argument('--socket', help='robotd control socket on this machine')
    ap.add_argument('--frames', type=int, help='frames to keep (default: config deploy.frames)')
    ap.add_argument('--out', type=Path)
    ap.add_argument('--config', type=Path, help='stage2.json (default: the repo one)')
    ap.add_argument('--place', action='store_true', help='Mac: lay out the field, then exit')
    ap.add_argument('--probe', default='127.0.0.1:7811')
    ap.add_argument('--table', type=Path, help='Mac: summarise a finished run folder')
    ap.add_argument('--pack', type=Path, help='with --table: the evaluation pack (model versions)')
    args = ap.parse_args(argv)
    cfg = json.loads((args.config or repo_root() / 'configs/vision/stage2.json').read_text())
    stage1 = json.loads((repo_root() / 'configs/control/stage1.json').read_text())
    if args.table:
        table(args.table, args.pack)
        return 0
    if args.place:
        print(json.dumps(place(cfg, stage1, args.probe), ensure_ascii=False))
        return 0
    missing = [n for n in ('backend', 'model', 'socket', 'out') if getattr(args, n) is None]
    if missing:
        ap.error('measuring needs --' + ', --'.join(missing))
    return measure(args, cfg, stage1)


if __name__ == '__main__':
    raise SystemExit(main())
