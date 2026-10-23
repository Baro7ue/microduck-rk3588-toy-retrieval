"""Pairing the boxes of two detectors on the same frame (plans 2D and 2E consistency checks).

A detector result here is decode_single / decode_split's (boxes [n,4], scores [n], classes [n]),
sorted by score. Boxes pair greedily in score order with the unpaired box of the same class that
overlaps most (IoU >= min_iou); the worst corner and score differences are over the pairs.
"""
from __future__ import annotations

import numpy as np

from microduck_toy.stage2.evaluate import iou


def pair_boxes(a, b, min_iou: float = 0.5) -> dict:
    used, pairs = set(), []
    for i in range(len(a[0])):
        best, best_iou = None, min_iou
        for j in range(len(b[0])):
            if j in used or a[2][i] != b[2][j]:
                continue
            u = iou(a[0][i], b[0][j])
            if u >= best_iou:
                best, best_iou = j, u
        if best is not None:
            used.add(best)
            pairs.append((i, best))
    paired_a = {i for i, _ in pairs}
    return {'pairs': len(pairs),
            'box_px': max((float(np.abs(a[0][i] - b[0][j]).max()) for i, j in pairs), default=0.0),
            'conf': max((float(abs(a[1][i] - b[1][j])) for i, j in pairs), default=0.0),
            'unpaired_a': [i for i in range(len(a[0])) if i not in paired_a],
            'unpaired_b': [j for j in range(len(b[0])) if j not in used]}


def nms_tie(i: int, side, nms_iou: float, tol: float = 1e-4) -> bool:
    """Box i of this result has a higher-scoring box of its class at IoU nms_iou +- tol, so float
    rounding alone decides whether NMS keeps it."""
    boxes, scores, classes = side
    return any(classes[j] == classes[i] and scores[j] >= scores[i]
               and abs(iou(boxes[i], boxes[j]) - nms_iou) <= tol
               for j in range(len(boxes)) if j != i)
