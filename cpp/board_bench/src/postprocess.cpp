// ETHAN: DFL, IoU, class-aware NMS and the split-output decode. The contract is in
// include/md/postprocess.h; tests/unit_tests.cpp checks each function, the last test against
// Python's detect.decode_split on generated outputs.
#include "md/postprocess.h"

#include <stdexcept>

namespace md {

float dfl(const float logits[kRegMax]) {
    (void)logits;
    throw std::logic_error("TODO(Ethan): dfl");
}

float iou(const Box& a, const Box& b) {
    (void)a; (void)b;
    throw std::logic_error("TODO(Ethan): iou");
}

std::vector<int> class_aware_nms(const std::vector<Box>& boxes, float iou_thr) {
    (void)boxes; (void)iou_thr;
    throw std::logic_error("TODO(Ethan): class_aware_nms");
}

std::vector<Box> decode_split(const std::vector<RawOutput>& outs, float conf, float iou_thr) {
    (void)outs; (void)conf; (void)iou_thr;
    throw std::logic_error("TODO(Ethan): decode_split");
}

}  // namespace md
