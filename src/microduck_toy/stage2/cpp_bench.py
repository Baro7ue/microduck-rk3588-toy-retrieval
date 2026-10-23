"""Plan 2H: the frames for the board C++ benchmark (cpp/board_bench) and its check against Python.

    # cloud (needs MuJoCo): the first N test frames again, as the camera payloads the board receives
    python -m microduck_toy.stage2.cpp_bench make --data runs/<UTC>_s2a_dataset --n 100 --out PACK/cpp
    # board: the Python pipeline on the same payloads (RKNNLite, INT8), inputs of the first frames kept
    python -m microduck_toy.stage2.cpp_bench reference --pack PACK/cpp --model <int8.rknn> \
        --out ref.jsonl --dump-inputs inputs --dump-n 20
    # anywhere: md_bench's JSONL against the reference
    python -m microduck_toy.stage2.cpp_bench compare --cpp cpp.jsonl --ref ref.jsonl \
        [--cpp-inputs DIR --ref-inputs DIR] --out compare.json

PACK/cpp/frames.tsv lists the first n/2 window and n/2 letterbox test frames, id and mapping (kind x0 y0 gain pad_x pad_y, as views.Mapping);
uyvy/<id>.uyvy is the 640x360 UYVY payload (uyvy.encode of the raw render). `make` checks every
frame: the payload through deploy.frame_input gives the dataset's own input image, byte for byte.

Acceptance (plan 2H): every frame has the same number of boxes; each C++ box pairs with a Python
box of its class at IoU >= 0.99 and their scores differ by <= 1e-3. Also reported: how many input
bytes differ (resize), and the P50 of each step, C++ against Python (numpy).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from microduck_toy.stage2.deploy import RAW_H, RAW_W, frame_input
from microduck_toy.vision import uyvy, views
from microduck_toy.vision.detect import decode_split

IOU_MIN = 0.99
CONF_TOL = 1e-3


def mapping_row(fid: str, m: views.Mapping) -> str:
    return '\t'.join([fid, m.kind] + [repr(float(v)) for v in (m.x0, m.y0, m.gain, m.pad_x, m.pad_y)])


def read_frames(pack: Path) -> list[tuple[str, views.Mapping]]:
    rows = (pack / 'frames.tsv').read_text().splitlines()[1:]
    out = []
    for row in rows:
        fid, kind, *vals = row.split('\t')
        out.append((fid, views.Mapping(kind, *(float(v) for v in vals))))
    return out


def make(data: Path, n: int, out: Path) -> dict:
    import cv2
    from microduck_toy.stage2 import datagen
    cfg, stage1 = datagen.load_configs()
    rend = datagen.Renderer(datagen.repo_root() / 'third_party' / 'microduck_rl', cfg, stage1)
    every = [json.loads(x) for x in (data / 'meta' / 'test.jsonl').read_text().splitlines()]
    # Half search (window), half approach (letterbox): the first of each, in seed order.
    metas = ([m for m in every if m['mapping']['kind'] == 'window'][:n - n // 2]
             + [m for m in every if m['mapping']['kind'] == 'letterbox'][:n // 2])
    (out / 'uyvy').mkdir(parents=True)
    lines = ['id\tkind\tx0\ty0\tgain\tpad_x\tpad_y']
    views_count = {'window': 0, 'letterbox': 0}
    for meta in metas:
        scene = datagen.sample(int(meta['seed']), 'test', cfg, stage1)
        r = rend.pose(scene)
        r.rgb.update_scene(r.data, camera=r.cam)
        r.rgb.render()                                   # settle, as Renderer.render does
        rgb, _ = r.view()
        payload = uyvy.encode(np.ascontiguousarray(np.rot90(rgb, 1))).tobytes()
        img, _, again = rend.render(scene)
        m = views.Mapping.from_dict(again['mapping'])
        if again['mapping'] != meta['mapping']:
            raise SystemExit(f'{meta["id"]}: mapping {again["mapping"]} differs from the dataset')
        disk = cv2.imread(str(data / 'images' / 'test' / f'{meta["id"]}.png'))
        mine = frame_input(payload, m)[:, :, ::-1]
        if not (np.array_equal(mine, img) and np.array_equal(mine, disk)):
            raise SystemExit(f'{meta["id"]}: the payload does not give the dataset input image')
        (out / 'uyvy' / f'{meta["id"]}.uyvy').write_bytes(payload)
        lines.append(mapping_row(meta['id'], m))
        views_count[m.kind] += 1
    (out / 'frames.tsv').write_text('\n'.join(lines) + '\n')
    info = {'frames': len(metas), 'source': str(data), 'split': 'test', 'views': views_count,
            'check': 'payload -> deploy.frame_input == dataset PNG, every frame'}
    (out / 'make.json').write_text(json.dumps(info, indent=1) + '\n')
    return info


def reference(pack: Path, model: Path, out: Path, dump: Path | None, dump_n: int, names: list[str],
              conf: float, iou: float) -> dict:
    from microduck_toy.vision.detect import RknnYolo
    net = RknnYolo.board(model, names, iou=iou)
    if dump:
        dump.mkdir(parents=True, exist_ok=True)
    times = {'preprocess': [], 'inference': [], 'postprocess': []}
    try:
        with out.open('w') as f:
            for i, (fid, m) in enumerate(read_frames(pack)):
                payload = (pack / 'uyvy' / f'{fid}.uyvy').read_bytes()
                if len(payload) != RAW_W * RAW_H * 2:
                    raise SystemExit(f'{fid}: payload of {len(payload)} bytes')
                t0 = time.perf_counter()
                img = frame_input(payload, m)
                x = net.prepare(img, bgr=False)
                t1 = time.perf_counter()
                outs = net.run(x)
                t2 = time.perf_counter()
                boxes, scores, classes = decode_split(outs, conf, iou)
                t3 = time.perf_counter()
                if dump and i < dump_n:
                    (dump / f'{fid}.rgb').write_bytes(img.tobytes())
                ms = {'preprocess': (t1 - t0) * 1e3, 'inference': (t2 - t1) * 1e3,
                      'postprocess': (t3 - t2) * 1e3}
                for k, v in ms.items():
                    times[k].append(v)
                f.write(json.dumps({'id': fid, 'boxes': [[*map(float, b), float(s), int(c)] for b, s, c
                                                          in zip(boxes, scores, classes)],
                                    'ms': {k: round(v, 4) for k, v in ms.items()}}) + '\n')
    finally:
        net.close()
    return {k: round(float(np.median(v)), 3) for k, v in times.items() if v}


def _iou(a, b) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def compare_frame(cpp: list, ref: list) -> dict:
    """Pair boxes in order of the reference, each with the unused C++ box of its class of best
    IoU; returns the count difference, the worst pair and anything unpaired."""
    used, worst_iou, worst_conf, unpaired = set(), 1.0, 0.0, []
    for r in ref:
        best, best_iou = None, 0.0
        for j, c in enumerate(cpp):
            if j in used or int(c[5]) != int(r[5]):
                continue
            v = _iou(c[:4], r[:4])
            if v > best_iou:
                best, best_iou = j, v
        if best is None or best_iou < 0.5:
            unpaired.append(r)
            continue
        used.add(best)
        worst_iou = min(worst_iou, best_iou)
        worst_conf = max(worst_conf, abs(cpp[best][4] - r[4]))
    ok = len(cpp) == len(ref) and not unpaired and worst_iou >= IOU_MIN and worst_conf <= CONF_TOL
    return {'cpp': len(cpp), 'ref': len(ref), 'worst_iou': worst_iou, 'worst_conf': worst_conf,
            'unpaired_ref': len(unpaired), 'ok': ok}


def compare(cpp_path: Path, ref_path: Path, cpp_inputs: Path | None, ref_inputs: Path | None) -> dict:
    cpp = {r['id']: r for r in map(json.loads, cpp_path.read_text().splitlines())}
    ref = {r['id']: r for r in map(json.loads, ref_path.read_text().splitlines())}
    missing = sorted(set(ref) ^ set(cpp))
    rows, first_bad = [], None
    for fid in sorted(set(ref) & set(cpp)):
        c = compare_frame(cpp[fid]['boxes'], ref[fid]['boxes'])
        rows.append(c)
        if not c['ok'] and first_bad is None:
            first_bad = {'id': fid, **c, 'cpp_boxes': cpp[fid]['boxes'][:5], 'ref_boxes': ref[fid]['boxes'][:5]}
    out = {'frames': len(rows), 'missing_or_extra_ids': missing[:10],
           'frames_ok': sum(r['ok'] for r in rows),
           'worst_iou': round(min((r['worst_iou'] for r in rows), default=1.0), 5),
           'worst_conf': round(max((r['worst_conf'] for r in rows), default=0.0), 6),
           'first_bad_frame': first_bad,
           'p50_ms': {'cpp': {k: round(float(np.median([cpp[i]['ms'][k] for i in cpp])), 3)
                              for k in next(iter(cpp.values()))['ms']} if cpp else {},
                      'python': {k: round(float(np.median([ref[i]['ms'][k] for i in ref])), 3)
                                 for k in next(iter(ref.values()))['ms']} if ref else {}}}
    if cpp_inputs and ref_inputs:
        diffs = []
        for p in sorted(ref_inputs.glob('*.rgb')):
            q = cpp_inputs / p.name
            if q.exists():
                a = np.frombuffer(p.read_bytes(), np.uint8).astype(np.int16)
                b = np.frombuffer(q.read_bytes(), np.uint8).astype(np.int16)
                d = np.abs(a - b) if a.size == b.size else np.array([255])
                diffs.append({'id': p.stem, 'bytes_differ': int((d > 0).sum()), 'max': int(d.max())})
        out['inputs'] = {'frames': len(diffs), 'identical': sum(d['bytes_differ'] == 0 for d in diffs),
                         'worst': max(diffs, key=lambda d: d['bytes_differ'], default=None)}
    out['pass'] = not missing and bool(rows) and out['frames_ok'] == len(rows)
    out['rule_text'] = (f'same box count in every frame, pairs IoU >= {IOU_MIN}, scores within '
                        f'{CONF_TOL}')
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    m = sub.add_parser('make')
    m.add_argument('--data', type=Path, required=True)
    m.add_argument('--n', type=int, default=100)
    m.add_argument('--out', type=Path, required=True)
    r = sub.add_parser('reference')
    r.add_argument('--pack', type=Path, required=True)
    r.add_argument('--model', type=Path, required=True)
    r.add_argument('--out', type=Path, required=True)
    r.add_argument('--dump-inputs', type=Path)
    r.add_argument('--dump-n', type=int, default=20)
    r.add_argument('--config', type=Path, help='stage2.json (default: the repo one)')
    c = sub.add_parser('compare')
    c.add_argument('--cpp', type=Path, required=True)
    c.add_argument('--ref', type=Path, required=True)
    c.add_argument('--cpp-inputs', type=Path)
    c.add_argument('--ref-inputs', type=Path)
    c.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    if args.cmd == 'make':
        print(json.dumps(make(args.data, args.n, args.out)))
        return 0
    if args.cmd == 'reference':
        from microduck_toy.paths import repo_root
        cfg = json.loads((args.config or repo_root() / 'configs/vision/stage2.json').read_text())
        p50 = reference(args.pack, args.model, args.out, args.dump_inputs, args.dump_n, cfg['classes'],
                        float(cfg['detect']['conf']), float(cfg['detect']['iou']))
        print(json.dumps({'python_p50_ms': p50}))
        return 0
    out = compare(args.cpp, args.ref, args.cpp_inputs, args.ref_inputs)
    args.out.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: out[k] for k in ('frames', 'frames_ok', 'worst_iou', 'worst_conf', 'p50_ms',
                                          'pass')}))
    return 0 if out['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
