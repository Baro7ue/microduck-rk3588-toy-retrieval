// Shared types and constants of the board C++ benchmark (Stage 2, plan 2H).
// Every number here mirrors the Python side; the file it comes from is named next to it.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace md {

constexpr int kRawW = 640, kRawH = 360;      // camera frame as sent: 640x360 UYVY (camera_stream.py)
constexpr int kFrameW = 360, kFrameH = 640;  // upright frame = np.rot90(raw, 3) (vision/views.py)
constexpr int kInput = 320;                  // detector input, 320x320 RGB (views.INPUT)
constexpr int kWindow = 360;                 // search window side in frame pixels (views.WINDOW)
constexpr uint8_t kPad = 114;                // letterbox padding grey (views.PAD_VALUE)
constexpr int kNumClasses = 6;               // bus henry ketchup croc thomas tray (stage2.json)
constexpr int kRegMax = 16;                  // DFL bins per box side (detect.REG_MAX)
constexpr int kStrides[3] = {8, 16, 32};     // output order: box, cls, sum per stride (detect.STRIDES)
constexpr float kSumMargin = 0.01f;          // skip a cell when sum < conf - margin (detect.SUM_MARGIN)
constexpr int kMaxCandidates = 1000;         // pre-NMS cap, best scores first (detect.MAX_CANDIDATES)
constexpr int kMaxDet = 100;                 // boxes kept after NMS (detect.MAX_DET)

// Frame -> input: input = (frame - (x0, y0)) * gain + (pad_x, pad_y)   (views.Mapping)
//   window:    x0 = 0, y0 = top row of the 360x360 square, gain = 320/360, no padding
//   letterbox: x0 = y0 = 0, gain = 0.5, pad_x = 70 (left and right), pad_y = 0
struct Mapping {
    bool window = false;
    float x0 = 0, y0 = 0, gain = 1, pad_x = 0, pad_y = 0;
};

// One detection in input pixels (0..320), like detect.decode_split's rows.
struct Box {
    float x0, y0, x1, y1;
    float score;
    int cls;
};

// One raw NPU output as rknn_outputs_get returns it with want_float = 0: int8 values in NCHW
// order (n = 1), plus the zero point and scale that turn them into floats: (q - zp) * scale.
struct RawOutput {
    int c = 0, h = 0, w = 0;
    int32_t zp = 0;
    float scale = 1;
    std::vector<int8_t> q;
    int8_t at(int ch, int y, int x) const { return q[(static_cast<size_t>(ch) * h + y) * w + x]; }
};

}  // namespace md
