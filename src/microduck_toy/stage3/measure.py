"""Stage 3P: four things measured in the sim before the brain's numbers are trusted (plan section 4).

    python -m microduck_toy.stage3.episode 3p [--parts m_view,m_tof,m_reach,m_spin]

m_view   the held toy in the H1 picture at the staging point: which rows it covers, whether it
         overlaps the tray box, and whether YOLO still finds the tray (3 targets)
m_tof    what the 8x8 ToF sees: per zone floor / tray rim / wall / near (< self_blind_m: the beak
         or the held toy) / other, and which zones the stop rule would fire on, at the staging
         point (H0, H1, with and without a toy), facing a wall at 0.2 / 0.3 / 0.5 m and in the open
m_reach  how far the feet reach ahead of the trunk (collision geoms; plan reference 3.4 cm) and
         where the held toy sits against the beak tip, standing at H0 / H1 / H2
m_spin   the bearing error of spin frames: five toys around the robot, a full turn at the search
         head; truth bearing at the frame's moment vs at its receipt (the lag that would explain it)
Truth is used here only to measure; the brain never sees it.
"""
from __future__ import annotations

import base64
import json
import math
import socket
import time

import numpy as np

from microduck_toy.stage1 import metrics
from microduck_toy.stage3 import tof as tof_lib
from microduck_toy.stage3.world import odom_of, wrap

STAGING = (0.2, 0.0, 0.0)          # the x- staging point facing the tray (stage3.json carry)
SPIN_RING = {'bus': (1.0, 20.0), 'henry': (0.8, 100.0), 'ketchup': (1.3, 200.0),
             'croc': (0.9, 260.0), 'thomas': (1.1, 320.0)}     # distance m, bearing deg from +x


def view(r, rgb: bool = False) -> dict:
    return r.probe.call('view', after=-1, rgb=rgb, timeout=5.0)


def save_png(path, answer: dict, boxes) -> None:
    import cv2
    img = np.frombuffer(base64.b64decode(answer['rgb_b64']), np.uint8).reshape(answer['rgb_shape'])
    img = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2BGR)
    for b in boxes:
        x0, y0, x1, y1 = (int(round(v)) for v in b['xyxy'])
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 255) if b.get('oracle') else (255, 0, 255), 1)
        cv2.putText(img, b['name'], (x0, max(10, y0 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.imwrite(str(path), img)


def overlap(a, b) -> float:
    """Intersection area over a's area."""
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0) / max(1e-9, (a[2] - a[0]) * (a[3] - a[1]))


def m_view(r) -> None:
    targets = ['bus'] if r.args.quick else ['bus', 'henry', 'ketchup']
    for k, toy in enumerate(targets):
        label = f'm_view_{toy}'
        r.reset(head='H2', robot=STAGING)
        r.wait_settled(3.0)
        load = r.load_toy(toy, k, label)
        r.robot.head('H1')
        time.sleep(1.5)
        ans = view(r, rgb=True)
        boxes = {b['name']: b['xyxy'] for b in ans['boxes']}
        yolo = []
        r.eyes.frame('approach', fresh=True)
        for _ in range(3):
            sight = r.eyes.frame('approach')
            yolo.append(sight.boxes)
        rec = {'part': 'm_view', 'label': label, 'toy': toy, 'loaded_by': load['by'],
               'reach': r.probe.call('reach'), 'oracle_toy': boxes.get(toy),
               'oracle_tray': boxes.get('tray'), 'yolo': yolo}
        held = (rec['reach'] or {}).get('held') == toy
        rec['held'] = held
        # a held toy with no segmentation box is outside the H1 frame (below its lower edge)
        rec['toy_in_view'] = bool(rec['oracle_toy'])
        if held and not rec['oracle_toy']:
            rec['tray_covered_by_toy'] = 0.0
        if rec['oracle_toy']:
            t = rec['oracle_toy']
            rec['toy_rows'] = [round(t[1], 1), round(t[3], 1)]
            rec['toy_rows_frac'] = [round(t[1] / 640, 3), round(t[3] / 640, 3)]
        if rec['oracle_toy'] and rec['oracle_tray']:
            rec['tray_covered_by_toy'] = round(overlap(rec['oracle_tray'], rec['oracle_toy']), 4)
        rec['yolo_tray_frames'] = sum(any(b['cls'] == 'tray' for b in f) for f in yolo)
        rec['yolo_toy_frames'] = sum(any(b['cls'] == toy for b in f) for f in yolo)
        save_png(r.dir / f'{label}.png', ans,
                 [{'name': n, 'xyxy': v, 'oracle': True} for n, v in boxes.items()] +
                 [{'name': 'Y:' + b['cls'], 'xyxy': b['xyxy']} for b in yolo[-1]])
        r.robot.mouth(1.0)
        time.sleep(0.8)
        r.robot.mouth(0.0)
        r.trial(rec)
        print(f'm_view {toy}: held {held}, in view {rec["toy_in_view"]}, rows {rec.get("toy_rows")}, '
              f'tray covered {rec.get("tray_covered_by_toy")}, '
              f'YOLO tray {rec["yolo_tray_frames"]}/3', flush=True)


def tof_frame() -> dict:
    with socket.create_connection(('127.0.0.1', 7801), timeout=2.0) as sock:
        stream = sock.makefile('rwb')
        stream.write(b'{"op": "tof"}\n')
        stream.flush()
        return json.loads(stream.readline())


def classify(p: dict, row: dict, cfg: dict, s1: dict) -> str:
    half = float(s1['arena']['half_size_m'])
    tray = s1['arena']['tray']
    th = float(tray['size_m']) / 2
    c, s = math.cos(row['yaw']), math.sin(row['yaw'])
    x = row['x'] + c * p['x'] - s * p['y']
    y = row['y'] + s * p['x'] + c * p['y']
    if p['slant_m'] < float(cfg['self_blind_m']):
        return 'near'
    if p['z'] < 0.006:
        return 'floor'
    cx, cy = tray['centre_m']
    if abs(x - cx) <= th + 0.005 and abs(y - cy) <= th + 0.005 and p['z'] <= 0.015:
        return 'tray'
    if abs(x) >= half - 0.01 or abs(y) >= half - 0.01:
        return 'wall'
    return 'other'


def m_tof(r) -> None:
    half = float(r.cfg['arena']['half_size_m'])
    cfg = r.s3['tof']
    scenes = [('staging_H1_toy', STAGING, 'H1', True), ('staging_H0_toy', STAGING, 'H0', True),
              ('staging_H1_empty', STAGING, 'H1', False), ('staging_H0_empty', STAGING, 'H0', False)]
    for d in (0.20, 0.30, 0.50):
        for head in ('H0', 'H1', (0.0, 0.15)):
            name = head if isinstance(head, str) else 'search'
            scenes.append((f'wall_{d:.2f}_{name}', (-half + d, 0.0, math.pi), head, False))
    for head in ('H0', 'H1', (0.0, 0.15)):
        name = head if isinstance(head, str) else 'search'
        scenes.append((f'open_{name}', (-0.2, -0.3, math.pi / 2), head, False))
    if r.args.quick:
        scenes = scenes[:2] + scenes[4:5]
    for label, pose, head, toy in scenes:
        r.reset(head='H2' if toy else 'H0', robot=pose)
        r.wait_settled(3.0)
        if toy:
            r.load_toy('bus', 0, f'm_tof_{label}')
        r.robot.head(head)
        time.sleep(1.5)
        zones: dict[str, int] = {}
        stops, nearest = 0, None
        frames = []
        for _ in range(3):
            frame = tof_frame()
            state = r.robot.snapshot()
            row = r.probe.truth()
            pts = tof_lib.points(frame, state)
            kinds = [classify(p, row, cfg, r.cfg) for p in pts]
            for k in kinds:
                zones[k] = zones.get(k, 0) + 1
            hits = tof_lib.obstacles(frame, state, cfg)
            stops += bool(len(hits))
            if hits:
                h = min(hits, key=lambda p: p['range_m'])
                nearest = h['range_m'] if nearest is None else min(nearest, h['range_m'])
            frames.append({'valid': len(pts), 'kinds': kinds, 'zones': [p['zone'] for p in pts],
                           'hits': [{'zone': p['zone'], 'range_m': round(p['range_m'], 3), 'z': round(p['z'], 3),
                                     'kind': classify(p, row, cfg, r.cfg)} for p in hits]})
        rec = {'part': 'm_tof', 'label': label, 'head': head, 'toy': toy,
               'zone_kinds_3_frames': zones, 'frames_that_stop': stops,
               'nearest_hit_m': None if nearest is None else round(nearest, 3), 'frames': frames,
               'trunk_to_wall_m': round(half + pose[0], 3) if label.startswith('wall') else None}
        r.trial(rec)
        if toy:
            r.robot.mouth(1.0)
            time.sleep(0.8)
            r.robot.mouth(0.0)
        print(f'm_tof {label}: {zones}, stop frames {stops}/3, nearest {rec["nearest_hit_m"]}', flush=True)


def m_reach(r) -> None:
    for head in ('H0', 'H1', 'H2'):
        r.reset(head=head, robot=(-0.3, 0.0, 0.0))
        r.wait_settled(3.0)
        time.sleep(1.0)
        samples = [r.probe.call('reach') for _ in range(3)]
        rec = {'part': 'm_reach', 'label': f'm_reach_{head}', 'head': head, 'held': None,
               'samples': samples,
               'toe_forward_m': round(max(max(s['left_toe'][0], s['right_toe'][0]) for s in samples), 4),
               'tip_forward_m': round(sum(s['tip'][0] for s in samples) / 3, 4),
               'tip_z_m': round(sum(s['tip'][2] for s in samples) / 3, 4)}
        r.trial(rec)
        print(f'm_reach {head}: toe {rec["toe_forward_m"]}, tip {rec["tip_forward_m"]}', flush=True)
    targets = ['bus'] if r.args.quick else ['bus', 'henry', 'ketchup']
    for k, toy in enumerate(targets):
        r.reset(head='H2', robot=(-0.3, 0.0, 0.0))
        r.wait_settled(3.0)
        load = r.load_toy(toy, k, f'm_reach_{toy}')
        for head in ('H1', 'H0'):
            r.robot.head(head)
            time.sleep(1.5)
            samples = [r.probe.call('reach') for _ in range(3)]
            s = samples[-1]
            rec = {'part': 'm_reach', 'label': f'm_reach_{toy}_{head}', 'head': head, 'held': toy,
                   'loaded_by': load['by'], 'samples': samples}
            if s.get('toy'):
                rec['toy_minus_tip_m'] = [round(a - b, 4) for a, b in zip(s['toy'], s['tip'])]
                rec['toy_forward_m'] = s['toy'][0]
            r.trial(rec)
            print(f'm_reach {toy} {head}: toy - tip {rec.get("toy_minus_tip_m")}', flush=True)
        r.robot.mouth(1.0)
        time.sleep(0.8)
        r.robot.mouth(0.0)


def m_spin(r) -> None:
    cfg = r.s3['search']
    toys = {}
    for name, (d, deg) in SPIN_RING.items():
        a = math.radians(deg)
        toys[name] = [d * math.cos(a), d * math.sin(a), a]
    r.reset(head='H0', robot=(0.0, 0.0, 0.0), toys=toys)
    r.wait_settled(3.0)
    r.robot.head(tuple(cfg['spin_head']))
    time.sleep(1.0)
    label = 'm_spin'
    start = r.mark(label)
    records = []
    _, _, yaw_prev = odom_of(r.robot.snapshot())
    turned = 0.0
    r.eyes.frame('search', fresh=True)
    r.robot.move(0.0, 0.0, float(cfg['spin_wz_radps']))
    try:
        while turned < math.radians(360.0 if not r.args.quick else 120.0):
            sight = r.eyes.frame('search')
            _, _, yaw = odom_of(r.robot.snapshot())
            turned += abs(wrap(yaw - yaw_prev))
            yaw_prev = yaw
            for box in sight.boxes:
                if box['cls'] == 'tray':
                    continue
                b = r.eyes.bearing(sight, box)
                if b is not None:
                    records.append({'cls': box['cls'], 'bearing': b, 'mono_frame': sight.mono,
                                    'mono_receipt': sight.received, 'conf': box['conf']})
    finally:
        r.robot.move(0.0, 0.0, 0.0)
    end = r.mark(label + '_end')
    rec = {'part': 'm_spin', 'label': label, 'start': start, 'end_mark': end, 'toys': toys,
           'records': records}
    r.trial(rec)
    print(f'm_spin: {len(records)} toy boxes over {math.degrees(turned):.0f} deg', flush=True)


PARTS = {'m_view': m_view, 'm_tof': m_tof, 'm_reach': m_reach, 'm_spin': m_spin}


def score_spin(rec: dict, rows: list[dict]) -> dict:
    errs_f, errs_r, rates = [], [], []
    for x in rec['records']:
        toy = x['cls']
        if f'{toy}_x' not in rows[0]:
            continue
        for key, out in (('mono_frame', errs_f), ('mono_receipt', errs_r)):
            row = metrics.at_mark(rows, x[key])
            truth = math.atan2(row[f'{toy}_y'] - row['y'], row[f'{toy}_x'] - row['x']) - row['yaw']
            out.append(math.degrees(wrap(x['bearing'] - truth)))
        row = metrics.at_mark(rows, x['mono_frame'])
        rates.append(row['wz'])

    def stats(v):
        if not v:
            return None
        a = np.abs(np.asarray(v))
        return {'n': len(v), 'mean_deg': round(float(np.mean(v)), 2), 'abs_p50': round(float(np.percentile(a, 50)), 2),
                'abs_p90': round(float(np.percentile(a, 90)), 2)}
    out = {'frame': stats(errs_f), 'receipt': stats(errs_r)}
    if errs_f and rates and abs(np.mean(rates)) > 0.1:
        # A camera that trails the state by `lag` puts the toy ahead of the turn: err ~ wz * lag.
        out['lag_s_from_frame_err'] = round(float(np.mean(errs_f)) / math.degrees(float(np.mean(rates))), 4)
        out['wz_mean_radps'] = round(float(np.mean(rates)), 3)
    return out


def summarise(trials: list[dict], rows: list[dict]) -> dict:
    out: dict = {}
    views = [t for t in trials if t['part'] == 'm_view']
    if views:
        out['m_view'] = [{k: t.get(k) for k in ('toy', 'loaded_by', 'held', 'toy_in_view', 'toy_rows',
                                                 'toy_rows_frac', 'tray_covered_by_toy', 'yolo_tray_frames',
                                                 'yolo_toy_frames')}
                         for t in views]
    tofs = [t for t in trials if t['part'] == 'm_tof']
    if tofs:
        out['m_tof'] = [{k: t.get(k) for k in ('label', 'zone_kinds_3_frames', 'frames_that_stop',
                                                'nearest_hit_m', 'trunk_to_wall_m')} for t in tofs]
    reach = [t for t in trials if t['part'] == 'm_reach']
    if reach:
        out['m_reach'] = [{k: t.get(k) for k in ('label', 'toe_forward_m', 'tip_forward_m', 'tip_z_m',
                                                  'toy_minus_tip_m', 'toy_forward_m')} for t in reach]
    spins = [t for t in trials if t['part'] == 'm_spin']
    if spins:
        out['m_spin'] = score_spin(spins[-1], rows)
    return out
