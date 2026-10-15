"""Stage 1D: how well a box in the image turns into "the toy is X cm ahead, Y cm left".

    python -m microduck_toy.stage1.s1d_range [--parts d_H2,d_H1] [--quick]
    (needs the arena body with its camera: run_arena_body.zsh --camera)

Static test (plan section 6): the robot stands at H1 or H2; one target toy is put at a known spot
in its heading frame; five consecutive camera frames are taken. Per frame:
  oracle  the segmentation box (the Stage 1 "eyes", decision 1A), ranged with control/ranging.py
          from robot.state only (camera pose from robotd's head FK, trunk tilt from the IMU, trunk
          height from odometry) onto the plane at half the toy's height
  colour  the HSV-threshold box on the same frame (control/colour.py), compared with the oracle box
The error is ranged minus true toy centre (truth used only to score).
  d_H2   distances 0.08-0.30 m (the C segment is 0.08-0.25 m)
  d_H1   distances 0.20-0.60 m (the B segment)
3 targets x 8 headings (lateral +1.5 cm on even headings, -3 cm on odd) per distance. The robot
stands at x = -0.5 m facing +x, so a toy 0.6 m ahead is still 0.4 m short of the tray (the first
Mac run, 2026-10-15, stood at the origin and the 0.5/0.6 m toys sat on the tray rim and base).
The colour box is ranged the same way, so its error is also reported in cm per toy.
Pass (plan): C segment |mean| <= 1 cm and sd <= 0.7 cm forward and left (after a correction if the
error is steady); B segment within 3 cm; colour box centre within 5 px of the oracle's for H2 at
<= 25 cm. Both corrections go in the summary: a straight line (error = a + b x distance) and a
per-distance table (the toy runs off the bottom of the image near 8 cm, which bends the error).
"""
from __future__ import annotations

import base64
import math
import sys
import time

import numpy as np

from microduck_toy.control import ranging
from microduck_toy.control.colour import ColourDetector
from microduck_toy.scene import toys as toy_lib
from microduck_toy.stage1 import common, metrics

DISTANCES = {'H2': (0.08, 0.10, 0.12, 0.15, 0.18, 0.21, 0.25, 0.30),
             'H1': (0.20, 0.25, 0.30, 0.40, 0.50, 0.60)}
HEADINGS_DEG = (0, 45, 90, 135, 180, 225, 270, 315)
LATERALS = (0.015, -0.03)
FRAMES = 5
C_RANGE = (0.08, 0.25)
B_RANGE = (0.25, 0.60)
C_MEAN_M, C_SD_M, B_ERR_M, COLOUR_PX = 0.01, 0.007, 0.03, 5.0
ROBOT_AT = (-0.5, 0.0, 0.0)
TARGETS = [toy_lib.SHORT[n] for n in toy_lib.targets()]


def truth_in_trunk(truth: dict, toy: str) -> tuple[float, float, float]:
    x0, y0, _ = truth['trunk']
    yaw = truth['yaw']
    tx, ty, tz, _ = truth[toy]
    dx, dy = tx - x0, ty - y0
    c, s = math.cos(yaw), math.sin(yaw)
    return c * dx + s * dy, -s * dx + c * dy, tz


def frames_for(s, head: str, toy: str, dist: float, heading_i: int, cam, colour) -> dict:
    lateral = LATERALS[heading_i % 2]
    s.reset(head=head, robot=ROBOT_AT)
    s.place_in_trunk(toy, dist, lateral, math.radians(HEADINGS_DEG[heading_i]))
    time.sleep(0.5)
    s.wait_settled(2.0)
    start = s.mark(f'd_{head}_{toy}_{dist:.2f}_{HEADINGS_DEG[heading_i]}')
    plane_z = float(s.meta['toys'][toy]['rest_z'])
    # The latest view may predate the placement (with --oracle-fps 0 it is the last pose's
    # frame); take its number and use only frames rendered after it.
    seq = s.probe.call('view', after=-1, rgb=False, timeout=5.0)['seq']
    frames = []
    for _ in range(FRAMES):
        view = s.probe.call('view', after=seq, rgb=True, timeout=5.0)
        state = s.robot.snapshot()
        seq = view['seq']
        frame = {'seq': seq, 'truth_trunk': truth_in_trunk(view['truth'], toy),
                 'gravity': state['safety']['gravity'], 'odom_z': state['odom']['position'][2],
                 'camera': state['frames']['camera']}
        box = next((b for b in view['boxes'] if b['name'] == toy), None)
        frame['oracle'] = box
        if box is not None:
            hit = ranging.ground_point(cam, state, box['xyxy'], plane_z)
            frame['ranged'] = hit
        rgb = np.frombuffer(base64.b64decode(view['rgb_b64']), np.uint8).reshape(view['rgb_shape'])
        found = colour.detect(rgb, [toy])
        frame['colour'] = found[0] if found else None
        frames.append(frame)
    return {'part': f'd_{head}', 'head': head, 'toy': toy, 'distance': dist,
            'heading_deg': HEADINGS_DEG[heading_i], 'lateral': lateral, 'start': start,
            'plane_z': plane_z, 'frames': frames}


def part_for(head: str):
    def part(s) -> None:
        cam = ranging.CameraModel.from_config(s.cfg['camera'])
        colour = ColourDetector(s.cfg['colour'])
        if s.args.quick:
            plan = [(t, d, h) for t in TARGETS[:1] for d in DISTANCES[head][::3] for h in (0, 1)]
        else:
            plan = [(t, d, h) for t in TARGETS for d in DISTANCES[head]
                    for h in range(len(HEADINGS_DEG))]
        for i, (toy, dist, heading_i) in enumerate(plan):
            s.trial(frames_for(s, head, toy, dist, heading_i, cam, colour))
            if (i + 1) % 8 == 0 or i + 1 == len(plan):
                print(f'd_{head}: {i + 1}/{len(plan)}', flush=True)
    return part


PARTS = {'d_H2': part_for('H2'), 'd_H1': part_for('H1')}


def centre(box) -> tuple[float, float]:
    x0, y0, x1, y1 = box
    return (x0 + x1) / 2, (y0 + y1) / 2


def frame_state(frame: dict) -> dict:
    """The robot.state fields ranging reads, as saved with each frame."""
    return {'safety': {'gravity': frame['gravity']}, 'odom': {'position': [0.0, 0.0, frame['odom_z']]},
            'frames': {'camera': frame['camera']}}


def score_pose(record: dict, cam=None) -> dict:
    errs, colour_errs, colour_px, cut = [], [], [], 0
    for f in record['frames']:
        tf, tl, tz = f['truth_trunk']
        if f.get('ranged'):
            errs.append((f['ranged'][0] - tf, f['ranged'][1] - tl))
        if f.get('oracle') and f['oracle']['cut']:
            cut += 1
        if f.get('oracle') and f.get('colour'):
            (ax, ay), (bx, by) = centre(f['oracle']['xyxy']), centre(f['colour']['xyxy'])
            colour_px.append(math.hypot(ax - bx, ay - by))
        if cam is not None and f.get('colour') and f.get('camera'):
            hit = ranging.ground_point(cam, frame_state(f), f['colour']['xyxy'], record['plane_z'])
            if hit:
                colour_errs.append((hit[0] - tf, hit[1] - tl))
    out = {k: record[k] for k in ('head', 'toy', 'distance', 'heading_deg', 'lateral')}
    out['seen'] = len(errs)
    out['cut_frames'] = cut
    # toy centre height in the frames vs the plane ranging assumed (on the floor: ~0)
    out['toy_raised_m'] = float(np.mean([f['truth_trunk'][2] for f in record['frames']])
                                - record['plane_z'])
    if errs:
        e = np.array(errs)
        out['err_forward_m'] = float(e[:, 0].mean())
        out['err_left_m'] = float(e[:, 1].mean())
        out['frame_sd_forward_m'] = float(e[:, 0].std())
        out['frame_sd_left_m'] = float(e[:, 1].std())
    out['colour_found'] = sum(1 for f in record['frames'] if f.get('colour'))
    out['colour_centre_px'] = float(np.mean(colour_px)) if colour_px else None
    if colour_errs:
        e = np.array(colour_errs)
        out['colour_err_forward_m'] = float(e[:, 0].mean())
        out['colour_err_left_m'] = float(e[:, 1].mean())
    return out


def colour_by_toy(results: list[dict]) -> dict:
    """The HSV box per toy (H2, C segment): centre offset from the oracle box in px, and its
    own ranging error in cm (Stage 2's colour baseline is judged on the second)."""
    out = {}
    for toy in dict.fromkeys(r['toy'] for r in results):
        rs = [r for r in results if r['toy'] == toy]
        out[toy] = {'centre_px': metrics.summary([r['colour_centre_px'] for r in rs
                                                  if r['colour_centre_px'] is not None]),
                    'err_forward_m': metrics.summary([r['colour_err_forward_m'] for r in rs
                                                      if 'colour_err_forward_m' in r]),
                    'err_left_m': metrics.summary([r['colour_err_left_m'] for r in rs
                                                   if 'colour_err_left_m' in r])}
    return out


def fit_line(xs, ys) -> dict | None:
    if len(xs) < 3:
        return None
    b, a = np.polyfit(np.asarray(xs), np.asarray(ys), 1)
    resid = np.asarray(ys) - (a + b * np.asarray(xs))
    return {'a_m': float(a), 'b_per_m': float(b), 'resid_sd_m': float(resid.std())}


def correction_table(results: list[dict]) -> dict:
    """Mean error per distance and axis over every toy and heading: the lookup a controller
    would subtract (interpolating between distances). Unlike the straight line it also covers
    the near end, where the toy runs off the bottom of the image and the box centre moves."""
    table = {}
    for axis in ('forward', 'left'):
        rows = {}
        for r in results:
            if f'err_{axis}_m' in r:
                rows.setdefault(r['distance'], []).append(r[f'err_{axis}_m'])
        table[axis] = {f'{d:.2f}': float(np.mean(v)) for d, v in sorted(rows.items())}
    return table


def lookup(table: dict, axis: str, distance: float) -> float:
    xs = [float(d) for d in table[axis]]
    return float(np.interp(distance, xs, list(table[axis].values())))


def segment(results: list[dict], lo: float, hi: float, fits: dict | None,
            table: dict | None = None) -> dict:
    rs = [r for r in results if lo - 1e-9 <= r['distance'] <= hi + 1e-9 and 'err_forward_m' in r]
    out = {'poses': len(rs), 'seen': sum(1 for r in rs if r['seen']),
           'cut_poses': sum(1 for r in rs if r['cut_frames'])}
    for axis in ('forward', 'left'):
        raw = [r[f'err_{axis}_m'] for r in rs]
        out[f'err_{axis}_m'] = metrics.summary(raw)
        fit = (fits or {}).get(axis)
        if fit:
            corrected = [r[f'err_{axis}_m'] - (fit['a_m'] + fit['b_per_m'] * r['distance'])
                         for r in rs]
            out[f'err_{axis}_linear_m'] = metrics.summary(corrected)
        if table:
            corrected = [r[f'err_{axis}_m'] - lookup(table, axis, r['distance']) for r in rs]
            out[f'err_{axis}_table_m'] = metrics.summary(corrected)
    return out


def score(s, rows) -> dict:
    cam = ranging.CameraModel.from_config(s.cfg['camera'])
    parts: dict[str, list] = {}
    for record in s.trials:
        parts.setdefault(record['part'], []).append(score_pose(record, cam))
    verdict = {}
    quick = s.args.quick
    for name, results in parts.items():
        head = name.split('_')[1]
        seen = [r for r in results if 'err_forward_m' in r]
        fits = {axis: fit_line([r['distance'] for r in seen], [r[f'err_{axis}_m'] for r in seen])
                for axis in ('forward', 'left')}
        table = correction_table(seen)
        by_distance = {f'{d:.2f}': segment(results, d, d, fits)
                       for d in sorted({r['distance'] for r in results})}
        entry = {'fit_error_vs_distance': fits, 'correction_table_m': table,
                 'by_distance': by_distance,
                 'unseen_poses': sum(1 for r in results if not r['seen']),
                 'raised_poses': sum(1 for r in results if r['toy_raised_m'] > 0.002)}
        if head == 'H2':
            c = segment(results, *C_RANGE, fits, table)
            near = [r['colour_centre_px'] for r in results
                    if r['distance'] <= C_RANGE[1] + 1e-9 and r['colour_centre_px'] is not None]
            entry['C_segment'] = c
            entry['colour_centre_px'] = metrics.summary(near)
            entry['colour_by_toy'] = colour_by_toy([r for r in results
                                                    if r['distance'] <= C_RANGE[1] + 1e-9])
            entry['colour_missed_frames'] = sum(FRAMES - r['colour_found'] for r in results
                                                if r['distance'] <= C_RANGE[1] + 1e-9)

            def ok(suffix):
                names = (f'err_forward{suffix}_m', f'err_left{suffix}_m')
                return all(name in c and abs(c[name].get('mean', 9)) <= C_MEAN_M
                           and c[name].get('sd', 9) <= C_SD_M for name in names)
            entry['pass_raw'] = None if quick else ok('')
            entry['pass_linear'] = None if quick else ok('_linear')
            entry['pass_table'] = None if quick else ok('_table')
            entry['colour_pass'] = None if quick else bool(near and max(near) <= COLOUR_PX)
        else:
            b = segment(results, *B_RANGE, fits, table)
            entry['B_segment'] = b
            entry['pass'] = None if quick else all(
                max(abs(b[f'err_{axis}_m'].get('min', 9)), abs(b[f'err_{axis}_m'].get('max', 9)))
                <= B_ERR_M for axis in ('forward', 'left'))
        verdict[name] = entry
    verdict['note'] = ('plan 1D: C segment (H2, 8-25 cm) |mean| <= 1 cm and sd <= 0.7 cm per axis, '
                       'raw, after the fitted line a + b x distance (linear), or after subtracting '
                       'the per-distance mean (table; needed where the toy is cut by the image '
                       'bottom, about 10 cm and nearer at H2); B segment (H1, 0.25-0.6 m) every '
                       'pose within 3 cm; colour box centre within 5 px of the oracle at H2 <= 25 '
                       'cm (colour_by_toy also ranges the colour box). raised_poses counts toys not '
                       'on the floor (should be 0). Errors = ranged - truth; pass is null for '
                       '--quick')
    return {'verdict': verdict, 'parts': parts}


def check_camera(s) -> None:
    if not s.meta.get('camera'):
        raise RuntimeError('the arena body has no camera; start run_arena_body.zsh with --camera')


def main(argv=None) -> int:
    return common.main('d', __doc__, PARTS, score, argv, check=check_camera)


if __name__ == '__main__':
    sys.exit(main())
