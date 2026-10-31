"""The brain's eyes: a camera frame, the robot.state that goes with it, and the YOLO boxes.

    history = StateHistory()                       # pass as Robot(on_state=history)
    eyes = Eyes(source, detector, history, cam, stage3)
    sight = eyes.frame('search')                   # the next new frame, window view; boxes in it
    xy = eyes.ground(sight, box, plane_z)          # (forward, left) on the floor, level trunk frame

Sources (where frames come from):
    StreamSource  the sim head camera on TCP (Stage 2's FrameReceiver): the Mac with the arena body
                  at 15 fps, and the board. The frame's moment is its first byte, less stream_lag_s.
    ProbeSource   the arena body's "view" op (cloud twin, frames rendered on request): the RGB comes
                  back with the moment the scene was copied, which is used as the frame's moment.
Detectors:
    Yolo          Stage 2's LiveDetector (cpu = split FP32 ONNX, fp16 / int8 = RKNN on the NPU),
                  window view while searching, letterboxed whole frame when approaching
    Oracle        the arena body's segmentation boxes (debugging only, never for a result)
    Colour        Stage 4E's baseline: the toys from Stage 2's colour threshold, the tray from Yolo

The robot.state paired with a frame is the one received closest to the frame's moment (a turning
robot's camera moves about 1 deg every 20 ms at 0.9 rad/s). Everything is in the upright 360x640
image, as in Stages 1 and 2.
"""
from __future__ import annotations

import base64
from collections import deque
import math
import threading
import time

import numpy as np

from microduck_toy.control import ranging
from microduck_toy.control.approach import Correction
from microduck_toy.stage2 import deploy
from microduck_toy.vision import uyvy, views


class StateHistory:
    """The last few seconds of robot.state, by receive time (time.monotonic)."""

    def __init__(self, seconds: float = 3.0):
        self.rows: deque = deque(maxlen=int(seconds * 50) + 10)
        self.lock = threading.Lock()

    def __call__(self, state: dict, received: float) -> None:
        with self.lock:
            self.rows.append((received, state))

    def at(self, mono: float) -> dict | None:
        with self.lock:
            if not self.rows:
                return None
            best = min(self.rows, key=lambda r: abs(r[0] - mono))
        return best[1]

    def latest(self) -> dict | None:
        with self.lock:
            return self.rows[-1][1] if self.rows else None


class Frame(deploy.Frame):
    """deploy.Frame (seq, UYVY payload, first, done on perf_counter) + the frame's moment on
    time.monotonic, and the oracle boxes when the source has them."""
    __slots__ = ('mono', 'oracle')

    def __init__(self, seq, payload, first, done, mono, oracle=None):
        super().__init__(seq, payload, first, done)
        self.mono, self.oracle = mono, oracle


class StreamSource:
    def __init__(self, address: tuple[str, int], lag_s: float = 0.0):
        self.rx = deploy.FrameReceiver(address)
        self.rx.start()
        self.lag_s = lag_s
        self.offset = time.monotonic() - time.perf_counter()

    def next(self, after: int, timeout: float = 2.0) -> Frame | None:
        f = self.rx.take_after(after, timeout)
        if f is None:
            return None
        return Frame(f.seq, f.payload, f.first, f.done, f.first + self.offset - self.lag_s)

    def close(self) -> None:
        self.rx.stop()


def upright_to_uyvy(upright: np.ndarray) -> bytes:
    """The upright 360x640 RGB back to the camera's raw 640x360 UYVY, as the stream would send it."""
    return uyvy.encode(np.ascontiguousarray(np.rot90(upright, 1))).tobytes()


class ProbeSource:
    def __init__(self, probe, rgb: bool = True):
        self.probe, self.rgb = probe, rgb
        self.lock = threading.Lock()

    def next(self, after: int, timeout: float = 5.0) -> Frame | None:
        with self.lock:
            view = self.probe.call('view', after=after, rgb=self.rgb, timeout=timeout)
        payload = b''
        if self.rgb:
            shape = view['rgb_shape']
            img = np.frombuffer(base64.b64decode(view['rgb_b64']), np.uint8).reshape(shape)
            payload = upright_to_uyvy(img)
        now = time.perf_counter()
        mono = view['truth'].get('mono', time.monotonic())
        return Frame(view['seq'], payload, now, now, mono, oracle=view['boxes'])

    def close(self) -> None:
        pass


class Yolo:
    def __init__(self, live: deploy.LiveDetector, margin_rows: int):
        self.live, self.margin = live, margin_rows

    def boxes(self, frame: Frame, state: dict | None, view: str) -> tuple[list[dict], str, dict]:
        dets, m, ms = self.live.detect(frame, state, view, self.margin)
        return ([{'cls': d['cls'], 'conf': round(d['conf'], 4), 'xyxy': d['xyxy']} for d in dets],
                m.kind, ms)

    def close(self) -> None:
        self.live.close()


class Oracle:
    """Segmentation boxes from the arena body (ProbeSource only); debugging, never for results."""

    def boxes(self, frame: Frame, state: dict | None, view: str) -> tuple[list[dict], str, dict]:
        out = [{'cls': b['name'], 'conf': 1.0, 'xyxy': b['xyxy']} for b in (frame.oracle or [])
               if b.get('name')]
        return out, 'oracle', {}

    def close(self) -> None:
        pass


class Colour:
    """Stage 4E (plan claude-docs/stage4_plan_20261028/, 4E): only the toy detector changes. The toys
    come from Stage 2's colour threshold (configs/vision/stage2.json baselines.colour, vision/detect.py
    ColourBoxes) on the same 320x320 input YOLO gets (window when searching, letterbox when
    approaching), keeping blobs that score above YOLO's conf (0.25: about 7 px or more); the tray box
    still comes from YOLO. The baseline has no croc class: croc's green falls inside henry's range."""

    def __init__(self, yolo: Yolo, cfg: dict, conf: float):
        from microduck_toy.vision.detect import ColourBoxes
        self.yolo, self.colour, self.conf = yolo, ColourBoxes(cfg), conf
        self.margin = yolo.margin

    def boxes(self, frame: Frame, state: dict | None, view: str) -> tuple[list[dict], str, dict]:
        found, kind, ms = self.yolo.boxes(frame, state, view)
        start = time.perf_counter()
        m = deploy.mapping_for(state, view, self.margin)
        bgr = np.ascontiguousarray(deploy.frame_input(frame.payload, m)[..., ::-1])   # ColourBoxes takes BGR
        out = [{'cls': d['cls'], 'conf': round(d['conf'], 4),
                'xyxy': [round(v, 2) for v in views.to_frame_box(d['xyxy'], m)]}
               for d in self.colour(bgr, conf=self.conf)]
        out += [b for b in found if b['cls'] == 'tray']
        return out, kind, {**ms, 'colour': round((time.perf_counter() - start) * 1000.0, 3)}

    def close(self) -> None:
        self.yolo.close()


class Sight:
    __slots__ = ('seq', 'state', 'boxes', 'view', 'ms', 'mono', 'received')

    def __init__(self, seq, state, boxes, view, ms, mono, received):
        self.seq, self.state, self.boxes, self.view = seq, state, boxes, view
        self.ms, self.mono, self.received = ms, mono, received

    def of(self, cls: str) -> list[dict]:
        return sorted((b for b in self.boxes if b['cls'] == cls), key=lambda b: -b['conf'])


class Eyes:
    def __init__(self, source, detector, history: StateHistory, cam: ranging.CameraModel,
                 ranging_facts: dict, log=None):
        self.source, self.detector, self.history, self.cam = source, detector, history, cam
        self.corrections = {head: Correction(ranging_facts[head]['correction_table_cm'])
                            for head in ('H1', 'H2')}
        self.seq = -1
        self.log = log
        self.timings: list[dict] = []   # per frame: the detector's ms (receive .. e2e), view
        self.prefer = None          # state -> (forward, left) where the B segment expects the target
        self.last: Frame | None = None  # the last frame taken, for picture()
        self.shoot = None           # (sight, tag) -> None: the brain's picture saver (Brain.shot)

    def frame(self, view: str, fresh: bool = False, timeout: float = 5.0) -> Sight:
        """The next frame after the last one taken (fresh: after this call), with its boxes."""
        if fresh:
            first = self.source.next(-1, timeout)
            self.seq = first.seq if first is not None else self.seq
        frame = None
        while frame is None:
            frame = self.source.next(self.seq, timeout)
            if frame is None:
                raise TimeoutError(f'no camera frame after {self.seq} within {timeout} s')
        self.seq = frame.seq
        self.last = frame
        state = self.history.at(frame.mono) or self.history.latest()
        boxes, kind, ms = self.detector.boxes(frame, state, view)
        if ms:
            self.timings.append({**ms, 'view': kind})
        return Sight(frame.seq, state, boxes, kind, ms, frame.mono, time.monotonic())

    def picture(self, sight: Sight) -> tuple[np.ndarray, int | None] | None:
        """The upright RGB of the frame `sight` came from (it must be the last one taken) and, for a
        window frame, the window's top row; None without the pixels (a probe source run with rgb off)."""
        f = self.last
        if f is None or f.seq != sight.seq or not f.payload:
            return None
        pairs = np.frombuffer(f.payload, np.uint8).reshape(deploy.RAW_H, deploy.RAW_W // 2, 4)
        rgb = np.ascontiguousarray(np.rot90(uyvy.decode(pairs), 3))
        y0 = None
        if sight.view == 'window':
            m = deploy.mapping_for(sight.state, 'search', getattr(self.detector, 'margin', views.MARGIN_ROWS))
            y0 = int(m.y0) if m.kind == 'window' else None
        return rgb, y0

    def latency(self, since: int = 0) -> dict:
        """Percentiles of the detector's times over the frames taken since index `since`."""
        rows = self.timings[since:]
        keys = sorted({k for r in rows for k in r if k != 'view'})
        return {'frames': len(rows), **{k: deploy.percentiles([r[k] for r in rows if k in r])
                                        for k in keys}}

    def ground(self, sight: Sight, box: dict, plane_z: float, point: str = 'centre'):
        if sight.state is None:
            return None
        return ranging.ground_point(self.cam, sight.state, box['xyxy'], plane_z, point)

    def bearing(self, sight: Sight, box: dict) -> float | None:
        """Bearing (rad, left > 0) of the box centre's ray in the level trunk frame."""
        if sight.state is None:
            return None
        x0, y0, x1, y1 = box['xyxy']
        _, rot = ranging.camera_in_level(sight.state)
        ray = rot @ self.cam.pixel_ray((x0 + x1) / 2, (y0 + y1) / 2)
        return math.atan2(ray[1], ray[0])

    def look(self, robot, head: str, toy: str, plane_z: float, frames: int) -> dict:
        """approach.Eyes.look for the B segment (control/approach.py): `frames` new frames at
        `head` (H1 or H2), letterboxed; the toy's corrected (forward, left) averaged over the frames
        that saw it. With several boxes of the class, the one nearest to where `prefer(state)` expects the
        toy is taken."""
        points, raw = [], []
        self.frame('approach', fresh=True)
        for _ in range(frames):
            sight = self.frame('approach')
            best, best_d = None, math.inf
            expect = self.prefer(sight.state) if self.prefer and sight.state else None
            for box in sight.of(toy):
                hit = self.ground(sight, box, plane_z)
                if hit is None:
                    continue
                d = 0.0 if expect is None else math.dist(hit, expect)
                if d < best_d:
                    best, best_d = hit, d
            if best is None:
                continue
            raw.append(best)
            points.append(self.corrections[head].apply(best) if head in self.corrections else best)
        if self.shoot is not None:
            self.shoot(sight, f'b_look_{head}')
        toy_at = None
        if points:
            toy_at = (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points))
        if self.log is not None:
            self.log('b_look', head=head, toy=toy_at, seen=len(points), frames=frames)
        return {'toy': toy_at, 'seen': len(points), 'frames': frames, 'raw': raw}

    def close(self) -> None:
        self.source.close()
        self.detector.close()
