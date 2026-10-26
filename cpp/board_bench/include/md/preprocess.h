// Window / letterbox preprocessing. ETHAN WRITES src/preprocess.cpp (part of the ~200 lines).
//
// Goal: the same 320x320 RGB bytes as Python's views.to_input, byte for byte. Python resizes with
// cv2.resize(..., interpolation=cv2.INTER_LINEAR); for uint8 images OpenCV does that in fixed
// point, and the formula below reproduces it exactly (checked against OpenCV 4.11 on 360->320 and
// 640x360->320x180: 0 differing bytes). Each output pixel, each channel:
//
//   1. Coefficients, separately for x (src width sw -> dst width dw) and y (sh -> dh):
//        scale = (float)sw / dw
//        f  = (d + 0.5) * scale - 0.5          // centre of dst pixel d in source coordinates
//        s  = floor(f);  f -= s
//        if (s < 0)       { s = 0;      f = 0; }
//        if (s >= sw - 1) { s = sw - 1; f = 0; }
//        a0 = round((1 - f) * 2048)            // 11-bit fixed point, round to nearest
//        a1 = 2048 - a0                        // so a0 + a1 = 2048 exactly
//      The second source pixel is s + 1, kept inside the image (min(s + 1, sw - 1)).
//   2. Horizontal pass, per source row r (int32):  H(r) = src[r][sx] * ax0 + src[r][sx + 1] * ax1
//   3. Vertical pass, rows sy and sy + 1 (int32, the shifts are arithmetic):
//        out = (((ay0 * (H(sy) >> 4)) >> 16) + ((ay1 * (H(sy + 1) >> 4)) >> 16) + 2) >> 2
//      then clamp to 0..255.
//
// Interview point: why not plain float bilinear? A one-level difference in one pixel can move an
// INT8 score by one quantisation step (about 0.004), more than the 1e-3 the C++/Python check allows.
#pragma once

#include <cstdint>
#include <vector>

#include "md/types.h"

namespace md {

// cv2.resize(src, (dw, dh), interpolation=INTER_LINEAR) for a packed 3-channel uint8 image.
// src: sh rows x sw columns x 3; dst: dh x dw x 3 (the caller allocates dw * dh * 3 bytes).
void resize_linear(const uint8_t* src, int sw, int sh, uint8_t* dst, int dw, int dh);

// views.to_input for one upright frame, given as rows [row0, row0 + rows) of the 640-row frame
// (360 wide, RGB; decode_upright_rows makes them). out: 320 x 320 x 3 RGB.
//   window (m.window):   rows y0 .. y0 + 359 of the frame -> resize 360x360 to 320x320.
//                        The caller passes exactly those rows (row0 = y0, rows = 360).
//   letterbox:           the whole frame (row0 = 0, rows = 640) -> resize to 180 wide x 320 high,
//                        placed at columns 70..249; columns 0..69 and 250..319 are grey 114.
void make_input(const std::vector<uint8_t>& rows_rgb, int row0, int rows, const Mapping& m,
                uint8_t* out);

}  // namespace md
