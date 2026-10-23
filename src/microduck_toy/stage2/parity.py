"""Plan 2E consistency: the same frames (or the same views) through two detector runs, compared.

    python -m microduck_toy.stage2.parity --data PACK/data --split test --ids PACK/parity_ids.txt \
        --ref PACK/cloud/fp32_test.jsonl --new runs/<UTC>_s2e_mac/mac_test.jsonl \
        --rule mac --out compare.json [--ref-images DIR --new-images DIR]

Both prediction files are stage2.predict output (boxes down to conf 0.001). Only boxes at the
detector threshold (conf 0.25) are compared, frame by frame, paired by class and IoU >= 0.5
(check_tools.pair_boxes). Rules (plan 2E):
    mac    the Mac against the cloud, same model: every frame has the same number of boxes and
           every pair is within 0.01 in score
    board  the board's NPU against a cloud run: per labelled target toy, "found" (right class,
           IoU >= 0.5 with the label) agrees in >= 95% of cases (Stage 0 d5's tolerance, 2 in 40)
With --ref-images / --new-images (the same views rendered on two machines), the pixel difference
is reported too (mean absolute, and the share of pixels more than 8 levels apart).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from microduck_toy.stage2.check_tools import pair_boxes
from microduck_toy.stage2.evaluate import CONF, IOU, iou

MAC_CONF_TOL = 0.01
BOARD_AGREE = 0.95


def load_preds(path: Path) -> dict:
    return {p['id']: p['dets'] for p in map(json.loads, path.read_text().splitlines())}


def as_arrays(dets: list[dict], names: list[str]):
    sure = [d for d in dets if d['conf'] >= CONF]
    boxes = np.array([d['xyxy'] for d in sure], float).reshape(-1, 4)
    return boxes, np.array([d['conf'] for d in sure], float), np.array([names.index(d['cls'])
                                                                       for d in sure], int)


def found(dets: list[dict], obj: dict) -> bool:
    return any(d['cls'] == obj['name'] and d['conf'] >= CONF and iou(d['xyxy'], obj['xyxy_input']) >= IOU
               for d in dets)


def pixels(a_path: Path, b_path: Path) -> tuple[float, float]:
    a, b = cv2.imread(str(a_path)).astype(np.int16), cv2.imread(str(b_path)).astype(np.int16)
    diff = np.abs(a - b)
    return float(diff.mean()), float((diff.max(axis=2) > 8).mean())


def compare(metas: list[dict], ref: dict, new: dict, names: list[str], targets: list[str],
            rule: str, images: tuple[Path, Path] | None = None) -> dict:
    count_diff, pairs, worst_conf, worst_box = [], 0, 0.0, 0.0
    agree = total = 0
    disagree = []
    pix = []
    for m in metas:
        a, b = as_arrays(ref[m['id']], names), as_arrays(new[m['id']], names)
        r = pair_boxes(a, b)
        pairs += r['pairs']
        worst_conf, worst_box = max(worst_conf, r['conf']), max(worst_box, r['box_px'])
        if len(a[0]) != len(b[0]):
            count_diff.append({'id': m['id'], 'ref': len(a[0]), 'new': len(b[0])})
        for o in m['objects']:
            if o['name'] not in targets or o['cut_frame']:
                continue
            fa, fb = found(ref[m['id']], o), found(new[m['id']], o)
            total += 1
            agree += fa == fb
            if fa != fb:
                disagree.append({'id': m['id'], 'toy': o['name'], 'ref': fa, 'new': fb})
        if images:
            pix.append(pixels(images[0] / f"{m['id']}.png", images[1] / f"{m['id']}.png"))
    out = {'frames': len(metas), 'rule': rule, 'pairs': pairs,
           'frames_with_other_box_count': len(count_diff), 'count_diffs': count_diff[:20],
           'worst_pair_conf': round(worst_conf, 6), 'worst_pair_box_px': round(worst_box, 3),
           'targets': total, 'found_agree': round(agree / total, 4) if total else None,
           'found_disagree': disagree[:20]}
    if pix:
        out['pixels'] = {'mean_abs': round(float(np.mean([p[0] for p in pix])), 3),
                         'share_over_8': round(float(np.mean([p[1] for p in pix])), 5),
                         'worst_frame_share_over_8': round(float(max(p[1] for p in pix)), 5)}
    if rule == 'mac':
        out['pass'] = not count_diff and worst_conf <= MAC_CONF_TOL
        out['rule_text'] = f'same box count in every frame, scores within {MAC_CONF_TOL}'
    else:
        out['pass'] = total > 0 and agree / total >= BOARD_AGREE
        out['rule_text'] = f'"found" agrees for >= {BOARD_AGREE:.0%} of target toys (d5 tolerance)'
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data', type=Path, required=True, help='folder with meta/<split>.jsonl')
    ap.add_argument('--split', required=True)
    ap.add_argument('--ids', type=Path, help='only these frame ids (one per line)')
    ap.add_argument('--ref', type=Path, required=True)
    ap.add_argument('--new', type=Path, required=True)
    ap.add_argument('--rule', choices=('mac', 'board'), required=True)
    ap.add_argument('--ref-images', type=Path)
    ap.add_argument('--new-images', type=Path)
    ap.add_argument('--config', type=Path, help='stage2.json (default: the repo one)')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    from microduck_toy.paths import repo_root
    cfg = json.loads((args.config or repo_root() / 'configs/vision/stage2.json').read_text())
    metas = [json.loads(x) for x in (args.data / 'meta' / f'{args.split}.jsonl').read_text().splitlines()]
    if args.ids:
        keep = set(args.ids.read_text().split())
        metas = [m for m in metas if m['id'] in keep]
    ref, new = load_preds(args.ref), load_preds(args.new)
    missing = [m['id'] for m in metas if m['id'] not in ref or m['id'] not in new]
    if missing:
        raise SystemExit(f'{len(missing)} frames missing from a prediction file, first {missing[0]}')
    images = (args.ref_images, args.new_images) if args.ref_images and args.new_images else None
    out = compare(metas, ref, new, cfg['classes'], cfg['targets'], args.rule, images)
    out.update({'ref': str(args.ref), 'new': str(args.new), 'split': args.split})
    args.out.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: out[k] for k in ('frames', 'frames_with_other_box_count', 'worst_pair_conf',
                                          'found_agree', 'pass')}), out.get('pixels', ''))
    return 0 if out['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
