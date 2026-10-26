"""Plan 2E metrics for one dataset split and one detector's predictions (one evaluator for all).

    python -m microduck_toy.stage2.evaluate --data runs/<UTC>_s2a_dataset --split test \
        --pred runs/<UTC>_s2b_coco/test.jsonl [--space coco] --out runs/<UTC>_s2b_coco/test_eval.json

Predictions: one JSON line per frame, {"id", "dets": [{"cls", "conf", "xyxy"}], "ms"?}, xyxy in the
320x320 input, kept down to conf 0.001 so that AP sees the whole ranking. Truth comes from the
dataset's meta/<split>.jsonl (seg boxes, distances, camera state).

Class space: 'ours' scores the 6 classes as they are. 'coco' scores the pretrained model in COCO
names (config baselines.coco_map: bus -> bus, henry and thomas -> train, ketchup -> bottle; croc and
tray have no COCO class and are not scored).

What it reports (plan section 8):
- AP50 per class, mAP50 and mAP50-95 over the scored classes (COCO-style greedy matching,
  101-point interpolation; may differ a little from Ultralytics' val numbers, which use multi-label
  NMS), and mAP50 over the 3 targets' classes, the one number all detectors share.
- Recall vs distance from the trunk (as Stage 1 and 3 measure it; the camera sits ~8 cm ahead),
  0.1 m bins, per view and per toy: a box of the right class with IoU >= 0.5 and conf >= 0.25. Toys cut by the frame edge are left out. 'any' is the d4 measure:
  a box of any class on the toy (IoU >= 0.3, or centre near a small box).
- Search radius: per target, the far edge of the last bin in the unbroken run from 0.5 m where
  recall >= 0.8 (bins with fewer than min_n toys are skipped); the radius is the smallest of the 3.
- Confusion matrix (truth rows, detected columns; 'missed' and a 'background' row for boxes on no
  object), for near (<= 0.6 m), mid and far (> 1.2 m) toys.
- Wrong grab: for every frame with toys and every target t, the top-scoring box of class t; wrong
  when it sits on another toy. 'near' counts only frames with a toy within 0.6 m and grabs that
  land on toys within 0.6 m.
- H1 ranging: approach frames, uncut targets 0.18-0.30 m from the trunk; the detected box and the
  seg box (Stage 1's oracle) both go through ranging.ground_point; the extra error is their
  difference. Expected grasp p = 1 - (blind park RMS^2 + extra RMS^2) / 16 (plan section 8).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from microduck_toy.paths import repo_root
from microduck_toy.vision import views

CONF = 0.25
IOU = 0.5
BIN_M = 0.1
NEAR_M, FAR_M = 0.6, 1.2
H1_RANGE_M = (0.18, 0.30)
GRASP_TOL_CM = 4.0


# ----------------------------------------------------------------------------- geometry

def iou(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def on_toy(box, gt) -> bool:
    """Stage 0 d4's rule (tools/detect.py): IoU >= 0.3, or for boxes a few pixels wide where IoU
    is noisy, the centre inside the true box grown by 25% (at least 3 px) and not more than four
    times its area."""
    if iou(box, gt) >= 0.3:
        return True
    gw, gh = gt[2] - gt[0], gt[3] - gt[1]
    mx, my = max(3.0, 0.25 * gw), max(3.0, 0.25 * gh)
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    area = (box[2] - box[0]) * (box[3] - box[1])
    return (gt[0] - mx <= cx <= gt[2] + mx and gt[1] - my <= cy <= gt[3] + my
            and area <= 4 * gw * gh)


def band(d: float) -> str:
    return 'near' if d <= NEAR_M else 'mid' if d <= FAR_M else 'far'


def trunk_dist(o: dict) -> float:
    return math.hypot(o['fwd_m'], o['left_m'])


# ----------------------------------------------------------------------------- class spaces

class Space:
    """Maps our object names to the scored class names (None = not scored)."""

    def __init__(self, kind: str, cfg: dict):
        self.kind = kind
        if kind == 'ours':
            self.map = {n: n for n in cfg['classes']}
        elif kind == 'coco':
            self.map = {n: cfg['baselines']['coco_map'].get(n) for n in cfg['classes']}
        else:
            raise ValueError(kind)
        self.classes = sorted({c for c in self.map.values() if c}, key=list(self.map.values()).index)
        self.targets = list(cfg['targets'])

    def __call__(self, name: str) -> str | None:
        return self.map.get(name)


# ----------------------------------------------------------------------------- AP

def ap_101(tp: np.ndarray, conf: np.ndarray, n_gt: int) -> float:
    if n_gt == 0:
        return float('nan')
    if len(tp) == 0:
        return 0.0
    order = np.argsort(-conf, kind='stable')
    tp = tp[order].astype(float)
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1e-12)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    points = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, points, side='left')
    return float(np.mean([envelope[i] if i < len(envelope) else 0.0 for i in idx]))


def match_class(gts: list, ignores: list, dets: list, thr: float) -> list[tuple[float, int]]:
    """Greedy COCO matching for one class in one frame: [(conf, 1 = TP / 0 = FP)], ignored boxes
    (on a too-small object of this class) left out."""
    used = [False] * len(gts)
    out = []
    for d in sorted(dets, key=lambda d: -d['conf']):
        best, best_iou = -1, thr
        for i, g in enumerate(gts):
            if not used[i]:
                v = iou(d['xyxy'], g)
                if v >= best_iou:
                    best, best_iou = i, v
        if best >= 0:
            used[best] = True
            out.append((d['conf'], 1))
        elif any(iou(d['xyxy'], g) >= thr for g in ignores):
            continue
        else:
            out.append((d['conf'], 0))
    return out


def average_precision(frames: list, space: Space) -> dict:
    thrs = [round(0.5 + 0.05 * i, 2) for i in range(10)]
    per = {}
    for c in space.classes:
        aps = []
        n_gt = sum(1 for m, _ in frames for o in m['objects'] if space(o['name']) == c)
        for t in thrs:
            pairs = []
            for m, dets in frames:
                gts = [o['xyxy_input'] for o in m['objects'] if space(o['name']) == c]
                ign = [o['xyxy_input'] for o in m['dropped']
                       if o.get('xyxy_input') and space(o['name']) == c]
                pairs += match_class(gts, ign, [d for d in dets if d['cls'] == c], t)
            conf = np.array([p[0] for p in pairs])
            tp = np.array([p[1] for p in pairs])
            aps.append(ap_101(tp, conf, n_gt))
        per[c] = {'n_gt': n_gt, 'ap50': round(aps[0], 4), 'ap50_95': round(float(np.mean(aps)), 4)}
    scored = [v for v in per.values() if v['n_gt']]
    tgt = [per[c] for c in dict.fromkeys(space(t) for t in space.targets) if per[c]['n_gt']]
    return {'per_class': per,
            'map50': round(float(np.mean([v['ap50'] for v in scored])), 4) if scored else None,
            'map50_95': round(float(np.mean([v['ap50_95'] for v in scored])), 4) if scored else None,
            'map50_targets': round(float(np.mean([v['ap50'] for v in tgt])), 4) if tgt else None}


# ----------------------------------------------------------------------------- recall, radius

def found(o: dict, dets: list, cls: str | None) -> bool:
    return cls is not None and any(d['cls'] == cls and d['conf'] >= CONF and
                                   iou(d['xyxy'], o['xyxy_input']) >= IOU for d in dets)


def recall_by_distance(frames: list, space: Space) -> dict:
    """{view: {name: {bin_m: [n, right, any]}}} over uncut toys (tray left out)."""
    out: dict = {}
    for m, dets in frames:
        sure = [d for d in dets if d['conf'] >= CONF]
        for o in m['objects']:
            if o['name'] == 'tray' or o['cut_frame']:
                continue
            b = f"{math.floor(trunk_dist(o) / BIN_M + 1e-9) * BIN_M:.1f}"
            cell = out.setdefault(m['view'], {}).setdefault(o['name'], {}).setdefault(b, [0, 0, 0])
            cell[0] += 1
            cell[1] += found(o, sure, space(o['name']))
            cell[2] += any(on_toy(d['xyxy'], o['xyxy_input']) for d in sure)
    for v in out.values():
        for name in v:
            v[name] = dict(sorted(v[name].items(), key=lambda kv: float(kv[0])))
    return out


def radius(curve: dict, start_m: float = 0.5, need: float = 0.8, min_n: int = 5) -> dict:
    """Far edge of the unbroken run of bins from start_m with recall >= need."""
    reach, stop = 0.0, None
    for b, (n, right, _any) in curve.items():
        lo = float(b)
        if lo + BIN_M <= start_m + 1e-9 or n < min_n:
            continue
        if right / n >= need:
            reach = round(lo + BIN_M, 2)
        else:
            stop = {'bin_m': lo, 'n': n, 'recall': round(right / n, 3)}
            break
    return {'radius_m': reach, 'stopped_at': stop}


def search_radius(rec: dict, space: Space) -> dict:
    per = {t: radius(rec.get('search', {}).get(t, {})) for t in space.targets}
    return {'per_target': per, 'radius_m': min(v['radius_m'] for v in per.values())}


def recall_bands(rec: dict) -> dict:
    """Recall over distance ranges the acceptance lines use: search 0.5-1.2 m, approach 0.2-0.6 m."""
    def pool(view, lo, hi):
        out = {}
        for name, curve in rec.get(view, {}).items():
            n = right = 0
            for b, (k, r, _a) in curve.items():
                if lo - 1e-9 <= float(b) < hi - 1e-9:
                    n, right = n + k, right + r
            out[name] = {'n': n, 'recall': round(right / n, 4) if n else None}
        return out
    return {'search_0.5_1.2': pool('search', 0.5, 1.2), 'approach_0.2_0.6': pool('approach', 0.2, 0.6)}


# ----------------------------------------------------------------------------- confusion, wrong grab

def confusion(frames: list, space: Space) -> dict:
    out = {k: {} for k in ('near', 'mid', 'far', 'all')}

    def add(bands, row, col):
        for k in bands:
            out[k].setdefault(row, {}).setdefault(col, 0)
            out[k][row][col] += 1
    for m, dets in frames:
        sure = [d for d in dets if d['conf'] >= CONF]
        for o in m['objects']:
            if o['cut_frame']:
                continue
            hits = [d for d in sure if iou(d['xyxy'], o['xyxy_input']) >= IOU]
            col = max(hits, key=lambda d: d['conf'])['cls'] if hits else 'missed'
            add((band(trunk_dist(o)), 'all'), o['name'], col)
        boxes = [o['xyxy_input'] for o in m['objects']] + \
                [o['xyxy_input'] for o in m['dropped'] if o.get('xyxy_input')]
        for d in sure:
            if all(iou(d['xyxy'], g) < IOU for g in boxes):
                add(('all',), 'background', d['cls'])
    return out


def wrong_grab(frames: list, space: Space) -> dict:
    count = {k: {'pairs': 0, 'right': 0, 'wrong': 0, 'phantom': 0, 'none': 0}
             for k in ('all', 'near')}
    by_target = {t: {'pairs': 0, 'wrong': 0} for t in space.targets}
    for m, dets in frames:
        toys = [o for o in m['objects'] if o['name'] != 'tray']
        if not toys:
            continue
        has_near = any(trunk_dist(o) <= NEAR_M for o in toys)
        for t in space.targets:
            cls = space(t)
            mine = [d for d in dets if d['cls'] == cls and d['conf'] >= CONF]
            keys = ('all', 'near') if has_near else ('all',)
            for k in keys:
                count[k]['pairs'] += 1
            by_target[t]['pairs'] += 1
            if not mine:
                for k in keys:
                    count[k]['none'] += 1
                continue
            top = max(mine, key=lambda d: d['conf'])
            under = [o for o in toys if on_toy(top['xyxy'], o['xyxy_input'])]
            hit = max(under, key=lambda o: iou(top['xyxy'], o['xyxy_input'])) if under else None
            what = 'phantom' if hit is None else 'right' if hit['name'] == t else 'wrong'
            count['all'][what] += 1
            by_target[t]['wrong'] += what == 'wrong'
            if has_near:
                if hit is not None and trunk_dist(hit) <= NEAR_M:
                    count['near'][what] += 1
                else:
                    count['near']['none'] += 1
    for v in count.values():
        v['wrong_rate'] = round(v['wrong'] / v['pairs'], 4) if v['pairs'] else None
    for v in by_target.values():
        v['wrong_rate'] = round(v['wrong'] / v['pairs'], 4) if v['pairs'] else None
    return {**count, 'by_target': by_target}


# ----------------------------------------------------------------------------- H1 ranging

def h1_ranging(frames: list, space: Space, stage1: dict, blind_rms_cm: float) -> dict:
    from microduck_toy.control import ranging
    cam = ranging.CameraModel.from_config(stage1['camera'])
    extra, base, missed = [], [], 0
    for m, dets in frames:
        if m['view'] != 'approach':
            continue
        mapping = views.Mapping.from_dict(m['mapping'])
        for o in m['objects']:
            if o['name'] not in space.targets or o['cut_frame']:
                continue
            if not H1_RANGE_M[0] <= trunk_dist(o) <= H1_RANGE_M[1]:
                continue
            hits = [d for d in dets if d['cls'] == space(o['name']) and d['conf'] >= CONF
                    and iou(d['xyxy'], o['xyxy_input']) >= IOU]
            if not hits:
                missed += 1
                continue
            d = max(hits, key=lambda d: d['conf'])
            oracle = ranging.ground_point(cam, m['state'], o['xyxy_frame'], o['half_z_m'])
            mine = ranging.ground_point(cam, m['state'], views.to_frame_box(d['xyxy'], mapping),
                                        o['half_z_m'])
            if oracle is None or mine is None:
                missed += 1
                continue
            extra.append((100 * (mine[0] - oracle[0]), 100 * (mine[1] - oracle[1])))
            base.append((100 * (oracle[0] - o['fwd_m']), 100 * (oracle[1] - o['left_m'])))
    if not extra:
        return {'n': 0, 'missed': missed}
    e, b = np.array(extra), np.array(base)
    rms = float(np.sqrt((e ** 2).sum(1).mean()))
    return {'n': len(e), 'missed': missed, 'extra_rms_cm': round(rms, 3),
            'extra_mean_cm': [round(float(v), 3) for v in e.mean(0)],
            'extra_sd_cm': [round(float(v), 3) for v in e.std(0)],
            'extra_max_cm': round(float(np.sqrt((e ** 2).sum(1)).max()), 3),
            'oracle_vs_truth_rms_cm': round(float(np.sqrt((b ** 2).sum(1).mean())), 3),
            'blind_park_rms_cm': blind_rms_cm,
            'expected_grasp_p': round(1 - (blind_rms_cm ** 2 + rms ** 2) / GRASP_TOL_CM ** 2, 3),
            'note': '(forward, left) cm; extra = YOLO box ranged minus seg box ranged'}


# ----------------------------------------------------------------------------- driver

def load(data: Path, split: str, pred: Path) -> list:
    """split may name several ('test,test_far'); pred then names one file per split, same order."""
    metas, preds = [], {}
    pred_files = str(pred).split(',')
    for i, s in enumerate(split.split(',')):
        metas += [json.loads(x) for x in (data / 'meta' / f'{s}.jsonl').read_text().splitlines()]
        for line in Path(pred_files[i] if len(pred_files) > 1 else pred_files[0]).read_text().splitlines():
            p = json.loads(line)
            preds[p['id']] = p
    missing = [m['id'] for m in metas if m['id'] not in preds]
    if missing:
        raise ValueError(f'{len(missing)} frames have no prediction, first {missing[0]}')
    return [(m, preds[m['id']]['dets']) for m in metas], [preds[m['id']].get('ms') for m in metas]


def evaluate(data: Path, split: str, pred: Path, space_kind: str = 'ours') -> dict:
    root = repo_root()
    cfg = json.loads((root / 'configs/vision/stage2.json').read_text())
    stage1 = json.loads((root / 'configs/control/stage1.json').read_text())
    facts1 = json.loads((root / 'configs/control/stage1_facts.json').read_text())
    blind = float(facts1['approach_1f_mac_first']['f_blind30']['park_err_rms_cm'])
    space = Space(space_kind, cfg)
    frames, ms = load(data, split, pred)
    rec = recall_by_distance(frames, space)
    out = {'data': str(data), 'split': split, 'pred': str(pred), 'space': space_kind,
           'frames': len(frames), 'conf': CONF, 'iou': IOU,
           'ap': average_precision(frames, space),
           'search_radius': search_radius(rec, space),
           'recall_bands': recall_bands(rec),
           'wrong_grab': wrong_grab(frames, space),
           'h1_ranging': h1_ranging(frames, space, stage1, blind),
           'confusion': confusion(frames, space),
           'recall_by_distance': rec}
    ms = [v for v in ms if v is not None]
    if ms:
        out['ms_per_frame'] = {'p50': round(float(np.median(ms)), 2),
                               'p99': round(float(np.percentile(ms, 99)), 2),
                               'note': 'detector call on the 320x320 input (pre + infer + post), '
                                       'this machine; not a deployment number'}
    return out


def brief(s: dict) -> str:
    ap = s['ap']
    lines = [f"{s['split']} ({s['frames']} frames, space {s['space']}): mAP50 {ap['map50']}, "
             f"mAP50-95 {ap['map50_95']}, targets mAP50 {ap['map50_targets']}",
             '  AP50 ' + ', '.join(f"{c} {v['ap50']}" for c, v in ap['per_class'].items()),
             f"  search radius {s['search_radius']['radius_m']} m " +
             str({t: v['radius_m'] for t, v in s['search_radius']['per_target'].items()}),
             '  recall search 0.5-1.2 m ' + str({k: v['recall'] for k, v in
                                                 s['recall_bands']['search_0.5_1.2'].items()}),
             '  recall approach 0.2-0.6 m ' + str({k: v['recall'] for k, v in
                                                   s['recall_bands']['approach_0.2_0.6'].items()}),
             f"  wrong grab all {s['wrong_grab']['all']['wrong_rate']}, "
             f"near {s['wrong_grab']['near']['wrong_rate']}"]
    h = s['h1_ranging']
    if h.get('n'):
        lines.append(f"  H1 ranging extra RMS {h['extra_rms_cm']} cm (n {h['n']}, missed "
                     f"{h['missed']}), expected grasp p {h['expected_grasp_p']}")
    else:
        lines.append(f"  H1 ranging: no matched targets (missed {h.get('missed')})")
    return '\n'.join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--pred', required=True, help='one file, or one per split joined by commas')
    ap.add_argument('--space', choices=('ours', 'coco'), default='ours')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    s = evaluate(args.data, args.split, args.pred, args.space)
    args.out.write_text(json.dumps(s, indent=1) + '\n')
    print(brief(s))
    print(args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
