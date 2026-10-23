#include "md/uyvy.h"

#include <cmath>
#include <stdexcept>

namespace md {

static inline uint8_t to_byte(float v) {
    v = std::nearbyint(v);                   // round half to even, as np.rint
    return static_cast<uint8_t>(v < 0.f ? 0.f : (v > 255.f ? 255.f : v));
}

void decode_upright_rows(const uint8_t* payload, int row0, int rows, std::vector<uint8_t>& out) {
    if (row0 < 0 || rows < 0 || row0 + rows > kFrameH)
        throw std::out_of_range("decode_upright_rows: rows outside the 640-row frame");
    out.resize(static_cast<size_t>(rows) * kFrameW * 3);
    for (int i = 0; i < rows; ++i) {
        const int col = row0 + i;                 // raw column = upright row
        const int pair = col / 2, which = col % 2;
        uint8_t* dst = &out[static_cast<size_t>(i) * kFrameW * 3];
        for (int j = 0; j < kFrameW; ++j) {
            const int raw_row = kRawH - 1 - j;    // upright column j = raw row 359 - j
            const uint8_t* p = payload + (static_cast<size_t>(raw_row) * (kRawW / 2) + pair) * 4;
            const float y = p[which ? 3 : 1];
            const float u = static_cast<float>(p[0]) - 128.f;
            const float v = static_cast<float>(p[2]) - 128.f;
            dst[3 * j + 0] = to_byte(y + 1.402f * v);
            dst[3 * j + 1] = to_byte(y - 0.344136f * u - 0.714136f * v);
            dst[3 * j + 2] = to_byte(y + 1.772f * u);
        }
    }
}

}  // namespace md
