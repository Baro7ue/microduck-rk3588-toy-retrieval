// ETHAN: window / letterbox preprocessing. The contract and the exact resize formula are in
// include/md/preprocess.h; tests/unit_tests.cpp checks both functions byte for byte.
#include "md/preprocess.h"

#include <stdexcept>

namespace md {

void resize_linear(const uint8_t* src, int sw, int sh, uint8_t* dst, int dw, int dh) {
    (void)src; (void)sw; (void)sh; (void)dst; (void)dw; (void)dh;
    throw std::logic_error("TODO(Ethan): resize_linear");
}

void make_input(const std::vector<uint8_t>& rows_rgb, int row0, int rows, const Mapping& m,
                uint8_t* out) {
    (void)rows_rgb; (void)row0; (void)rows; (void)m; (void)out;
    throw std::logic_error("TODO(Ethan): make_input");
}

}  // namespace md
