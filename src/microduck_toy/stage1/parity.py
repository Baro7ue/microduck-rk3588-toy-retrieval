"""Stage 1A render parity: do the Mac and the cloud draw the toys the same? (Stage 0 left it open.)

    python -m microduck_toy.stage1.parity --floor grid --out runs/<dir>      # render 10 views
    python -m microduck_toy.stage1.parity --compare A/views_grid.json B/views_grid.json

Kinematic, no robotd: the robot posed at the STAND keyframe with a head offset (scene/render.py),
the five toys at fixed spots, the head camera rendered as the detector sees it. Per view it keeps
every segmentation box and, per toy, the mean colour inside its mask. Pass (plan 1A): every box
within 2 px. Colour differences are reported, not judged (they decide where Stage 2 data is made).
One floor per process: a second software-GL context in one process breaks segmentation (OSMesa).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np

from microduck_toy.control.colour import rgb_to_hsv
from microduck_toy.paths import repo_root, runs_dir
from microduck_toy.scene import toys as toy_lib
from microduck_toy.stage1 import common

FAR = {'bus': (0.25, 0.05, 0.3), 'henry': (0.35, -0.08, 1.2), 'ketchup': (0.45, 0.12, 2.0),
       'croc': (0.60, -0.20, 0.7), 'thomas': (0.75, 0.20, -0.5)}
NEAR = {'bus': (0.14, 0.02, 0.8), 'henry': (0.20, -0.05, 0.0), 'ketchup': (0.27, 0.07, 1.5),
        'croc': (0.33, -0.10, 2.5), 'thomas': (0.40, 0.10, 0.4)}
POSES = [('H0', 0.0), ('H1', 0.0), ('H2', 0.0), ('H3', 0.0), ('H1', 0.3)]
BOX_PX = 2


def views() -> list[dict]:
    out = []
    for name, spots in (('far', FAR), ('near', NEAR)):
        for head, yaw in POSES:
            out.append({'id': f'{name}_{head}_yaw{yaw:+.1f}', 'head': head, 'yaw': yaw,
                        'toys': spots})
    return out


def render(floor: str, out_dir: Path) -> Path:
    from PIL import Image
    from microduck_toy.scene.render import ArenaRender
    cfg = common.stage1_config()
    rl_root = repo_root() / 'third_party' / 'microduck_rl'
    scene = ArenaRender(rl_root, cfg['arena'], floor)
    records = []
    for view in views():
        scene.pose(view['head'], yaw=view['yaw'])
        for toy, (fwd, left, yaw) in view['toys'].items():
            x, y = scene.trunk_frame_point(fwd, left)
            scene.place(toy, x, y, view['yaw'] + yaw)
        rgb, boxes = scene.view()
        seg = scene.seg_upright()
        hsv = rgb_to_hsv(rgb)
        colours = {}
        for name, toy in scene.ids.toys.items():
            mask = seg == toy['visual']
            if mask.any():
                h = np.radians(hsv[..., 0][mask] * 2.0)           # hue as an angle, 0-360 deg
                colours[toy_lib.SHORT[name]] = {
                    'rgb': [round(float(v), 2) for v in rgb[mask].mean(0)],
                    'h_deg': round(math.degrees(math.atan2(np.sin(h).mean(), np.cos(h).mean()))
                                   % 360, 2),
                    's': round(float(hsv[..., 1][mask].mean()), 2),
                    'v': round(float(hsv[..., 2][mask].mean()), 2), 'px': int(mask.sum())}
        Image.fromarray(rgb).save(out_dir / f'{floor}_{view["id"]}.png')
        records.append({'id': view['id'], 'boxes': boxes, 'colour': colours})
    import mujoco
    doc = {'floor': floor, 'host': platform.node(), 'platform': platform.platform(),
           'mujoco': mujoco.__version__, 'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
           'views': records}
    path = out_dir / f'views_{floor}.json'
    path.write_text(json.dumps(doc, indent=1))
    return path


def compare(a_path: str, b_path: str) -> dict:
    a, b = (json.loads(Path(p).read_text()) for p in (a_path, b_path))
    rows, worst = [], 0
    b_views = {v['id']: v for v in b['views']}
    for va in a['views']:
        vb = b_views.get(va['id'])
        if vb is None:
            rows.append({'id': va['id'], 'missing_in_b': True})
            worst = math.inf
            continue
        ba = {x['name']: x['xyxy'] for x in va['boxes']}
        bb = {x['name']: x['xyxy'] for x in vb['boxes']}
        for name in sorted(set(ba) | set(bb)):
            if name not in ba or name not in bb:
                rows.append({'id': va['id'], 'name': name, 'only_in': 'a' if name in ba else 'b'})
                worst = math.inf
                continue
            diff = max(abs(p - q) for p, q in zip(ba[name], bb[name]))
            worst = max(worst, diff)
            row = {'id': va['id'], 'name': name, 'box_px': diff}
            ca, cb = va['colour'].get(name), vb['colour'].get(name)
            if ca and cb:
                dh = abs((ca['h_deg'] - cb['h_deg'] + 180) % 360 - 180)
                row.update(dh_deg=round(dh, 2), ds=round(ca['s'] - cb['s'], 2),
                           dv=round(ca['v'] - cb['v'], 2))
            rows.append(row)
    colour = [r for r in rows if 'dh_deg' in r]
    return {'a': {k: a[k] for k in ('host', 'platform', 'mujoco', 'floor')},
            'b': {k: b[k] for k in ('host', 'platform', 'mujoco', 'floor')},
            'max_box_px': worst, 'pass': worst <= BOX_PX,
            'max_dh_deg': max((r['dh_deg'] for r in colour), default=None),
            'max_abs_ds': max((abs(r['ds']) for r in colour), default=None),
            'max_abs_dv': max((abs(r['dv']) for r in colour), default=None),
            'rows': rows}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--floor', choices=('grid', 'wood'), default='grid')
    p.add_argument('--out', default=None, help='output folder (default runs/<UTC>_s1a_parity_<host>)')
    p.add_argument('--compare', nargs=2, metavar=('A_JSON', 'B_JSON'))
    args = p.parse_args(argv)
    if args.compare:
        result = compare(*args.compare)
        out = Path(args.compare[0]).with_name(f'parity_vs_{Path(args.compare[1]).parent.name}.json')
        out.write_text(json.dumps(result, indent=1, ensure_ascii=False))
        print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, ensure_ascii=False))
        print(f'结果：{out}')
        return 0 if result['pass'] else 2
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out_dir = Path(args.out) if args.out else runs_dir() / f'{stamp}_s1a_parity_{platform.node()}'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'结果：{render(args.floor, out_dir)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
