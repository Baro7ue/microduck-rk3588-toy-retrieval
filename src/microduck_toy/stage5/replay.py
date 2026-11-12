"""Stage 5 room demo video, drawn after the run from the logged poses (no viewer during the run: a
viewer, or any other load on the Mac, stalls the physics and robotd then misses ticks, Stage 4).

    python -m microduck_toy.stage5.replay runs/<UTC>_s5_room_demo[_board] [--fps 25] [--speed 1]
        [--from S] [--to S] [--size 1280x720] [--out FILE.mp4]

Reads the run's truth.qpos.f64 (every truth row's full qpos, written by the arena body in room
mode), truth.meta.json, trials.jsonl and each episodes/<n>_<target>/brain.jsonl. Draws, per video
frame: a camera following the robot (main picture), the robot's own head camera (inset, the
picture the detector saw, without boxes), a top-down map (room, furniture, tray, toys, robot) and
a caption (target, brain state, time, toys delivered so far). Time is simulation time, so a stall
of the Mac's physics does not show as a freeze. Output: <run>/demo_video.mp4 (H.264 when OpenCV
can write it, else MPEG-4).
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
from pathlib import Path
import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from microduck_toy.paths import repo_root
from microduck_toy.scene import arena, render as scene_render, toys as toy_lib
from microduck_toy.stage5 import room

KIND_RGB = {'bus': (240, 200, 30), 'henry': (40, 160, 60), 'ketchup': (210, 40, 40),
            'croc': (20, 110, 40), 'thomas': (40, 90, 200)}
STATE_EN = {'SPIN': 'searching (spin)', 'LOOK': 'looking', 'RELOCATE': 'moving to search again',
            'GOTO': 'walking to target', 'APPROACH': 'approaching', 'LAST': 'last few cm',
            'PICK': 'picking up', 'CARRY': 'carrying to tray', 'TRAY_LOOK': 'finding the tray',
            'DROP': 'dropping into tray', 'FALLEN': 'fallen, getting up', 'RECOVER': 'looking for dropped toy'}


def load_qpos(run: Path) -> tuple[np.ndarray, dict]:
    meta = json.loads((run / 'truth.meta.json').read_text())
    log = meta.get('qpos_log')
    if not log:
        raise SystemExit(f'{run}: no qpos log (start the arena body with --room)')
    raw = np.fromfile(run / 'truth.qpos.f64', dtype='<f8')
    width = 2 + int(log['nq'])
    rows = raw[: len(raw) // width * width].reshape(-1, width)
    return rows, meta


def brain_states(run: Path) -> list[tuple[float, str, str]]:
    """(Mac monotonic time, target label, state) for every state change of every brain run."""
    out = []
    for line in (run / 'trials.jsonl').read_text().splitlines():
        rec = json.loads(line)
        offset = float(rec.get('offset') or 0.0)        # board clock -> Mac clock (stage3 remote)
        log = run / 'episodes' / rec['label'] / 'brain.jsonl'
        if not log.exists():
            continue
        for row in map(json.loads, log.read_text().splitlines()):
            if row.get('event') == 'state':
                out.append((float(row['mono']) + offset, rec['label'], row['state']))
            elif row.get('event') == 'end':
                out.append((float(row['mono']) + offset, rec['label'], 'END:' + str(row.get('end'))))
    return sorted(out)


def font(size: int):
    for name in ('/System/Library/Fonts/Helvetica.ttc', '/System/Library/Fonts/SFNS.ttf',
                 '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


class Replay:
    def __init__(self, run: Path, size=(1280, 720)):
        self.run = run
        self.rows, self.meta = load_qpos(run)
        doc = room.load()
        self.doc = doc
        cfg = room.merged(doc, 'stage1')['arena']
        # The tray the run's body built (fix2 runs log it; a run from before fix2 had the Stage 1
        # square tray), so an old run is not drawn with today's round tray.
        cfg['tray'] = {**room.merged({}, 'stage1')['arena']['tray'], 'centre_m': cfg['tray']['centre_m'],
                       **self.meta.get('room_tray', {})}
        rl_root = repo_root() / 'third_party' / 'microduck_rl'
        spec = arena.build_spec(cfg, rl_root, self.meta.get('floor', 'wood'))
        spec.visual.global_.offwidth = max(size[0], 640)
        spec.visual.global_.offheight = max(size[1], 640)
        self.model = spec.compile()
        if self.model.nq != int(self.meta['qpos_log']['nq']):
            raise SystemExit(f'model nq {self.model.nq} != logged {self.meta["qpos_log"]["nq"]}: '
                             'replay with the same room.json as the run')
        self.data = mujoco.MjData(self.model)
        self.ids = arena.ArenaIds(self.model)
        self.head = scene_render.turned_camera(self.model)
        self.size = size
        self.main = mujoco.Renderer(self.model, height=size[1], width=size[0])
        self.eye = mujoco.Renderer(self.model, height=scene_render.RAW_H, width=scene_render.RAW_W)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.distance, self.cam.elevation = 1.5, -32.0
        self.az = None
        self.cfg = cfg
        self.s3 = room.merged(doc, 'stage3')
        self.states = brain_states(run)
        self.trials = [json.loads(l) for l in (run / 'trials.jsonl').read_text().splitlines()]
        demo = run / 'demo.json'
        done = json.loads(demo.read_text()) if demo.exists() else {}
        self.results = done.get('results', [])
        self.total = len(done.get('targets', [])) or len(self.trials)
        self.font, self.small = font(round(26 * size[1] / 720)), font(round(18 * size[1] / 720))
        self.names = list(self.ids.toys)

    def frame(self, i: int, dt: float) -> Image.Image:
        t, mono = self.rows[i, 0], self.rows[i, 1]
        self.data.qpos[:] = self.rows[i, 2:]
        mujoco.mj_forward(self.model, self.data)
        x, y = float(self.data.qpos[0]), float(self.data.qpos[1])
        w, qx, qy, qz = self.data.qpos[3:7]
        yaw = math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        # The follow camera trails behind the robot, its heading smoothed (time constant 3 s) so a
        # spin turns the picture slowly instead of whirling it.
        want = math.degrees(yaw) + 180.0 + 35.0
        if self.az is None:
            self.az = want
        else:
            diff = (want - self.az + 180.0) % 360.0 - 180.0
            self.az += diff * min(1.0, dt / 3.0)
        self.cam.azimuth = self.az + 180.0
        self.cam.lookat[:] = [x, y, 0.12]
        self.main.update_scene(self.data, camera=self.cam)
        img = Image.fromarray(self.main.render())
        self.eye.update_scene(self.data, camera=self.head)
        eye = Image.fromarray(scene_render.upright(self.eye.render()))
        W, H = self.size
        ew = int(H * 0.42 * 360 / 640)
        eh = int(H * 0.42)
        eye = eye.resize((ew, eh))
        draw = ImageDraw.Draw(img)
        draw.rectangle([W - ew - 18, 14, W - 14, 18 + eh + 26], fill=(20, 20, 20))
        img.paste(eye, (W - ew - 16, 16))
        draw.text((W - ew - 14, 18 + eh + 2), "robot's camera", font=self.small, fill=(230, 230, 230))
        self.minimap(img, draw, x, y, yaw)
        self.caption(draw, t, mono)
        return img

    def minimap(self, img, draw, rx, ry, yaw):
        H = self.size[1]
        side = int(H * 0.36)
        x0, y0 = 16, H - side - 16
        half = float(self.cfg['room_half_m'])
        s = side / (2 * half)

        def px(x, y):
            return x0 + (x + half) * s, y0 + (half - y) * s

        draw.rectangle([x0 - 4, y0 - 4, x0 + side + 4, y0 + side + 4], fill=(245, 240, 230), outline=(60, 60, 60))
        f = float(self.cfg['half_size_m'])
        draw.rectangle([*px(-f, f), *px(f, -f)], outline=(170, 150, 120))
        for piece in self.cfg.get('furniture', []):
            for box in piece['boxes']:
                (bx, by, _), (hx, hy, _) = box['pos'], box['size']
                c = tuple(int(255 * v) for v in box.get('rgba', piece['rgba'])[:3])
                draw.rectangle([*px(bx - hx, by + hy), *px(bx + hx, by - hy)], fill=c)
        tx, ty = (float(v) for v in self.cfg['tray']['centre_m'])
        th = float(self.cfg['tray']['size_m']) / 2
        if self.cfg['tray'].get('shape') == 'round':
            draw.ellipse([*px(tx - th, ty + th), *px(tx + th, ty - th)], fill=(77, 77, 87))
        else:
            draw.rectangle([*px(tx - th, ty + th), *px(tx + th, ty - th)], fill=(77, 77, 87))
        for name in self.names:
            q = self.ids.toys[name]['qadr']
            x, y = float(self.data.qpos[q]), float(self.data.qpos[q + 1])
            if abs(x) > half or abs(y) > half:
                continue                    # parked outside the room
            cx, cy = px(x, y)
            draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=KIND_RGB[toy_lib.kind_of(name)],
                         outline=(0, 0, 0))
        cx, cy = px(rx, ry)
        r = 9
        pts = [(cx + r * math.cos(yaw), cy - r * math.sin(yaw)),
               (cx + r * math.cos(yaw + 2.5), cy - r * math.sin(yaw + 2.5)),
               (cx + r * math.cos(yaw - 2.5), cy - r * math.sin(yaw - 2.5))]
        draw.polygon(pts, fill=(255, 140, 0), outline=(0, 0, 0))

    def caption(self, draw, t, mono):
        k = bisect.bisect_right([s[0] for s in self.states], mono) - 1
        label, state = (self.states[k][1], self.states[k][2]) if k >= 0 else (None, None)
        lines = []
        if label:
            n, target = label.split('_', 1)
            total = self.total or '?'
            if state.startswith('END:'):
                # The brain only knows it opened the beak; whether the toy stayed in the tray is the
                # judge's (demo.json), so the caption says released, not delivered (v2).
                what = 'released' if state == 'END:dropped' else state[4:]
            else:
                what = STATE_EN.get(state, state.lower())
            lines.append(f'Target {n}/{total}: {target}   -   {what}')
        else:
            lines.append('Starting up')
        # What is in the tray in this frame (truth, the demo's own rule), so a toy knocked out
        # leaves the list when it leaves the tray.
        row = {}
        for name in self.names:
            q = self.ids.toys[name]['qadr']
            short = toy_lib.SHORT[name]
            row.update({f'{short}_x': float(self.data.qpos[q]), f'{short}_y': float(self.data.qpos[q + 1]),
                        f'{short}_z': float(self.data.qpos[q + 2])})
        centre = [float(v) for v in self.cfg['tray']['centre_m']]
        done = [toy_lib.SHORT[n] for n in self.names
                if room.in_tray(row, toy_lib.SHORT[n], centre, self.cfg['tray'], self.s3)]
        lines.append(f'sim time {t:6.1f} s    in the tray: {", ".join(done) if done else "-"}')
        y = 14
        for i, line in enumerate(lines):
            f = self.font if i == 0 else self.small
            box = draw.textbbox((16, y), line, font=f)
            draw.rectangle([box[0] - 6, box[1] - 4, box[2] + 6, box[3] + 4], fill=(0, 0, 0))
            draw.text((16, y), line, font=f, fill=(255, 255, 255))
            y = box[3] + 10


def writer(path: Path, fps: float, size):
    import cv2
    for codec in ('avc1', 'mp4v'):
        out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if out.isOpened():
            return out, codec
    raise SystemExit('OpenCV could not open a video writer (avc1 or mp4v)')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='stage5.replay', description=__doc__.split('\n')[0])
    ap.add_argument('run')
    ap.add_argument('--fps', type=float, default=25.0)
    ap.add_argument('--speed', type=float, default=1.0, help='2 = twice as fast as the simulation')
    ap.add_argument('--from', dest='t_from', type=float, default=None, help='sim time to start at, s')
    ap.add_argument('--to', dest='t_to', type=float, default=None, help='sim time to stop at, s')
    ap.add_argument('--size', default='1280x720')
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    import cv2
    run = Path(a.run)
    size = tuple(int(v) for v in a.size.lower().split('x'))
    rp = Replay(run, size)
    t = rp.rows[:, 0]
    t0 = a.t_from if a.t_from is not None else float(t[0])
    t1 = a.t_to if a.t_to is not None else float(t[-1])
    step = a.speed / a.fps
    out_path = Path(a.out) if a.out else run / 'demo_video.mp4'
    out, codec = writer(out_path, a.fps, size)
    times = np.arange(t0, t1, step)
    print(f'{len(times)} frames, {codec}, -> {out_path}', flush=True)
    for n, when in enumerate(times):
        i = min(int(np.searchsorted(t, when)), len(t) - 1)
        img = rp.frame(i, step)
        out.write(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))
        if n % 250 == 0:
            print(f'  {n}/{len(times)}', flush=True)
    out.release()
    print(f'done: {out_path}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
