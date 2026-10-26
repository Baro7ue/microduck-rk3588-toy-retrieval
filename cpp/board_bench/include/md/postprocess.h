// Post-processing of the 9 INT8 outputs. ETHAN WRITES src/postprocess.cpp (part of the ~200 lines).
//
// The answer must equal Python's detect.decode_split on the same outputs (src/microduck_toy/vision/
// detect.py): same boxes in the same order. Steps, per stride 8, 16, 32 (outputs 3k, 3k+1, 3k+2):
//   1. sum map [1,1,H,W]: skip the cell when dequant(sum) < conf - kSumMargin. (No class score
//      can pass when their clamped sum doesn't.) Cells in row-major order: y, then x.
//   2. cls map [1,6,H,W]: best class = the first index with the highest score (ties -> lower
//      index, like argmax); keep the cell when its score > conf (strictly greater).
//   3. box map [1,64,H,W]: channel side * 16 + bin, side = left, top, right, bottom. dfl() turns
//      16 logits into a distance in grid cells: softmax, then the expected bin.
//   4. anchor (x + 0.5, y + 0.5); box = (ax - left, ay - top, ax + right, ay + bottom) * stride.
// Then over all strides: if more than kMaxCandidates, keep the best by score (stable: equal scores
// keep stride/cell order); class_aware_nms; keep kMaxDet; clip every coordinate to 0..320.
//
// dequant(q) = (q - zp) * scale, in float. Interview point (rknn_model_zoo process_i8): the
// threshold can be compared in int8 without dequantising every value, q > conf / scale + zp;
// only cells that pass need floats. Either way the result must not change.
#pragma once

#include <vector>

#include "md/types.h"

namespace md {

inline float dequant(int8_t q, int32_t zp, float scale) {
    return (static_cast<float>(q) - static_cast<float>(zp)) * scale;
}

// Distance (grid cells) from 16 DFL logits: p_i = exp(l_i - max) / sum_j exp(l_j - max), d = sum_i i * p_i.
float dfl(const float logits[kRegMax]);

// Intersection over union of two xyxy boxes; 0 when the union is 0.
float iou(const Box& a, const Box& b);

// Indices of the boxes kept, best first. Sort by score, highest first, STABLE (equal scores keep
// their input order: INT8 scores take few distinct values, so ties are common and the order
// decides which box survives). Take the first pending box, keep it, drop every pending box of the
// same class whose IoU with it is > iou_thr; repeat. (yolo_postprocess.class_aware_nms)
std::vector<int> class_aware_nms(const std::vector<Box>& boxes, float iou_thr);

// Steps 1-4 and the NMS above, on the 9 raw outputs.
std::vector<Box> decode_split(const std::vector<RawOutput>& outs, float conf, float iou_thr);

}  // namespace md
