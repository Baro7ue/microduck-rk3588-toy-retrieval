// Unit tests for the board benchmark: no NPU needed. Run ./build/unit_tests; exit code 0 = all pass.
// Expected values come from fixtures.h (made by make_fixtures.py with the Python code each C++
// function must match) or are worked out by hand in the test. The inputs are regenerated here with
// the same pattern() and LCG as make_fixtures.py.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "fixtures.h"
#include "md/postprocess.h"
#include "md/preprocess.h"
#include "md/uyvy.h"

namespace {

int failures = 0;

#define CHECK(cond)                                                                   \
    do {                                                                              \
        if (!(cond)) {                                                                \
            std::printf("    FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);           \
            ++failures;                                                               \
        }                                                                             \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                         \
    do {                                                                              \
        const double _a = (a), _b = (b);                                              \
        if (!(std::fabs(_a - _b) <= (tol))) {                                         \
            std::printf("    FAIL %s:%d  %s = %.6f, expected %.6f\n", __FILE__, __LINE__, \
                        #a, _a, _b);                                                  \
            ++failures;                                                               \
        }                                                                             \
    } while (0)

uint64_t fnv1a(const uint8_t* p, size_t n) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 0x100000001b3ULL;
    return h;
}

std::vector<uint8_t> pattern(int h, int w, int salt) {
    std::vector<uint8_t> img(static_cast<size_t>(h) * w * 3);
    for (int r = 0; r < h; ++r)
        for (int c = 0; c < w; ++c)
            for (int ch = 0; ch < 3; ++ch)
                img[(static_cast<size_t>(r) * w + c) * 3 + ch] =
                    static_cast<uint8_t>((r * 7 + c * 13 + ch * 29 + (r * c) % 11 + salt) % 256);
    return img;
}

std::vector<uint8_t> payload_pattern() {
    std::vector<uint8_t> p(md::kRawW * md::kRawH * 2);
    for (size_t i = 0; i < p.size(); ++i) p[i] = static_cast<uint8_t>((i * 31 + (i >> 9) * 7 + 3) % 256);
    return p;
}

struct Lcg {
    uint32_t x;
    uint32_t next() { x = x * 1664525u + 1013904223u; return x >> 8; }
};

// make_fixtures.raw_outputs: the 9 raw INT8 outputs, generated in the same order.
std::vector<md::RawOutput> raw_outputs(uint32_t seed) {
    Lcg rnd{seed};
    std::vector<md::RawOutput> outs;
    for (int stride : md::kStrides) {
        const int g = md::kInput / stride;
        md::RawOutput box{64, g, g, 0, 0.25f, {}}, cls{6, g, g, -128, 1.0f / 255.0f, {}},
            sum{1, g, g, -128, 1.0f / 255.0f, {}};
        box.q.resize(static_cast<size_t>(64) * g * g);
        for (auto& v : box.q) v = static_cast<int8_t>(static_cast<int>(rnd.next() % 256) - 128);
        cls.q.resize(static_cast<size_t>(6) * g * g);
        for (int y = 0; y < g; ++y)
            for (int x = 0; x < g; ++x) {
                const bool hot = rnd.next() % kHotEvery == 0;
                const int c_hot = (y / 3 + x / 3) % 6;
                for (int ch = 0; ch < 6; ++ch) {
                    const uint32_t r = rnd.next();
                    const int q = (hot && ch == c_hot) ? -128 + 70 + static_cast<int>((r >> 4) % 186)
                                                       : -128 + static_cast<int>((r >> 4) % 20);
                    cls.q[(static_cast<size_t>(ch) * g + y) * g + x] = static_cast<int8_t>(q);
                }
            }
        sum.q.resize(static_cast<size_t>(g) * g);
        for (int y = 0; y < g; ++y)
            for (int x = 0; x < g; ++x) {
                bool hot = false;
                for (int ch = 0; ch < 6; ++ch) hot = hot || cls.at(ch, y, x) > -128 + 40;
                const uint32_t r = rnd.next();
                sum.q[static_cast<size_t>(y) * g + x] =
                    static_cast<int8_t>(hot ? 127 : -128 + static_cast<int>(r % 30));
            }
        outs.push_back(box);
        outs.push_back(cls);
        outs.push_back(sum);
    }
    return outs;
}

// ── tests ──────────────────────────────────────────────────────────────────

void test_decode_rows() {   // Claude's uyvy.cpp
    const auto pay = payload_pattern();
    std::vector<uint8_t> rows;
    md::decode_upright_rows(pay.data(), 137, 360, rows);
    CHECK(fnv1a(rows.data(), rows.size()) == kDecodeRows137);
    md::decode_upright_rows(pay.data(), 0, 640, rows);
    CHECK(fnv1a(rows.data(), rows.size()) == kDecodeAll);
}

void test_resize_small() {
    const auto src = pattern(7, 9, 3);
    std::vector<uint8_t> dst(5 * 4 * 3);
    md::resize_linear(src.data(), 9, 7, dst.data(), 5, 4);
    int wrong = 0;
    for (size_t i = 0; i < dst.size(); ++i) wrong += dst[i] != kResizeSmall[i];
    if (wrong) std::printf("    %d of %zu bytes differ from cv2.resize\n", wrong, dst.size());
    CHECK(wrong == 0);
}

void test_resize_real_sizes() {
    auto src = pattern(360, 360, 5);
    std::vector<uint8_t> dst(320 * 320 * 3);
    md::resize_linear(src.data(), 360, 360, dst.data(), 320, 320);
    CHECK(fnv1a(dst.data(), dst.size()) == kResize360To320);
    src = pattern(640, 360, 7);
    dst.assign(180 * 320 * 3, 0);
    md::resize_linear(src.data(), 360, 640, dst.data(), 180, 320);
    CHECK(fnv1a(dst.data(), dst.size()) == kResize640x360To320x180);
}

void test_make_input() {
    const auto pay = payload_pattern();
    std::vector<uint8_t> rows, out(320 * 320 * 3);
    md::Mapping win;
    win.window = true;
    win.y0 = 137;
    win.gain = 320.0f / 360.0f;
    md::decode_upright_rows(pay.data(), 137, 360, rows);
    md::make_input(rows, 137, 360, win, out.data());
    CHECK(fnv1a(out.data(), out.size()) == kInputWindow137);
    md::Mapping box;
    box.gain = 0.5f;
    box.pad_x = 70;
    md::decode_upright_rows(pay.data(), 0, 640, rows);
    md::make_input(rows, 0, 640, box, out.data());
    CHECK(fnv1a(out.data(), out.size()) == kInputLetterbox);
    CHECK(out[0] == md::kPad && out[(320 * 320 - 1) * 3] == md::kPad);   // corners are padding
}

void test_dfl() {
    float sharp[md::kRegMax];
    for (int i = 0; i < md::kRegMax; ++i) sharp[i] = i == 5 ? 30.f : -30.f;
    CHECK_NEAR(md::dfl(sharp), 5.0, 1e-4);
    float flat[md::kRegMax] = {};
    CHECK_NEAR(md::dfl(flat), 7.5, 1e-5);          // uniform: the mean of 0..15
    float two[md::kRegMax];
    for (int i = 0; i < md::kRegMax; ++i) two[i] = (i == 2 || i == 3) ? 1000.f : 0.f;  // huge: needs the max shift
    CHECK_NEAR(md::dfl(two), 2.5, 1e-4);
}

void test_iou() {
    const md::Box a{0, 0, 10, 10, 1, 0}, b{5, 0, 15, 10, 1, 0}, c{20, 20, 30, 30, 1, 0},
        z{3, 3, 3, 3, 1, 0};
    CHECK_NEAR(md::iou(a, a), 1.0, 1e-6);
    CHECK_NEAR(md::iou(a, b), 50.0 / 150.0, 1e-6);  // half overlap: 50 / (100 + 100 - 50)
    CHECK_NEAR(md::iou(a, c), 0.0, 1e-9);
    CHECK_NEAR(md::iou(z, z), 0.0, 1e-9);           // zero area: union 0 -> 0, not NaN
}

void test_nms() {
    // 0 and 1 tie at 0.8 and overlap (IoU 0.9): the first in input order wins.
    // 2 overlaps 0 but is another class: kept. 3 overlaps 1 only a little (IoU < 0.45): kept.
    const std::vector<md::Box> boxes{{0, 0, 10, 10, 0.8f, 1},  {0, 0, 10, 9, 0.8f, 1},
                                     {0, 0, 10, 10, 0.9f, 2},  {7, 0, 17, 10, 0.5f, 1},
                                     {0, 0, 10, 10, 0.3f, 1}};
    const auto keep = md::class_aware_nms(boxes, 0.45f);
    CHECK((keep == std::vector<int>{2, 0, 3}));
    const std::vector<md::Box> swapped{boxes[1], boxes[0]};
    CHECK((md::class_aware_nms(swapped, 0.45f) == std::vector<int>{0}));
    CHECK(md::class_aware_nms({}, 0.45f).empty());
}

void test_decode_one_cell() {
    // One hot cell at stride 16, (y 4, x 7), class 2, sharp bins left 2 top 1 right 3 bottom 5:
    // box = ((7.5 - 2) * 16, (4.5 - 1) * 16, (7.5 + 3) * 16, (4.5 + 5) * 16) = (88, 56, 168, 152).
    std::vector<md::RawOutput> outs;
    for (int stride : md::kStrides) {
        const int g = md::kInput / stride;
        md::RawOutput box{64, g, g, 0, 0.25f, std::vector<int8_t>(64 * g * g, -120)};
        md::RawOutput cls{6, g, g, -128, 1.0f / 255.0f, std::vector<int8_t>(6 * g * g, -128)};
        md::RawOutput sum{1, g, g, -128, 1.0f / 255.0f, std::vector<int8_t>(g * g, -128)};
        if (stride == 16) {
            const int bins[4] = {2, 1, 3, 5};
            for (int side = 0; side < 4; ++side)
                box.q[((side * 16 + bins[side]) * g + 4) * g + 7] = 120;
            cls.q[(2 * g + 4) * g + 7] = 76;     // (76 + 128) / 255 = 0.8
            sum.q[4 * g + 7] = 76;
        }
        outs.push_back(box);
        outs.push_back(cls);
        outs.push_back(sum);
    }
    const auto got = md::decode_split(outs, 0.25f, 0.45f);
    CHECK(got.size() == 1);
    if (got.size() == 1) {
        CHECK(got[0].cls == 2);
        CHECK_NEAR(got[0].score, 0.8, 1e-6);
        CHECK_NEAR(got[0].x0, 88, 1e-3);
        CHECK_NEAR(got[0].y0, 56, 1e-3);
        CHECK_NEAR(got[0].x1, 168, 1e-3);
        CHECK_NEAR(got[0].y1, 152, 1e-3);
    }
}

void test_decode_like_python() {
    const auto got = md::decode_split(raw_outputs(kDecodeSeed), 0.25f, 0.45f);
    const size_t n = sizeof kDecodeExpected / sizeof kDecodeExpected[0];
    if (got.size() != n) std::printf("    %zu boxes, Python kept %zu\n", got.size(), n);
    CHECK(got.size() == n);
    int first_bad = -1;
    for (size_t i = 0; i < std::min(got.size(), n) && first_bad < 0; ++i) {
        const auto& e = kDecodeExpected[i];
        const auto& g = got[i];
        const bool ok = g.cls == e.cls && std::fabs(g.score - e.score) <= 1e-6f &&
                        std::fabs(g.x0 - e.x0) <= 1e-3f && std::fabs(g.y0 - e.y0) <= 1e-3f &&
                        std::fabs(g.x1 - e.x1) <= 1e-3f && std::fabs(g.y1 - e.y1) <= 1e-3f;
        if (!ok) {
            first_bad = static_cast<int>(i);
            std::printf("    box %d: got (%.3f %.3f %.3f %.3f) %.6f cls %d, Python (%.3f %.3f %.3f %.3f) "
                        "%.6f cls %d\n", first_bad, g.x0, g.y0, g.x1, g.y1, g.score, g.cls,
                        e.x0, e.y0, e.x1, e.y1, e.score, e.cls);
        }
    }
    CHECK(first_bad < 0);
}

}  // namespace

int main() {
    const std::vector<std::pair<const char*, std::function<void()>>> tests{
        {"decode_upright_rows (given)", test_decode_rows},
        {"resize_linear: 9x7 -> 5x4", test_resize_small},
        {"resize_linear: 360->320 and 360x640->180x320", test_resize_real_sizes},
        {"make_input: window and letterbox", test_make_input},
        {"dfl", test_dfl},
        {"iou", test_iou},
        {"class_aware_nms: ties and classes", test_nms},
        {"decode_split: one cell", test_decode_one_cell},
        {"decode_split: same boxes as Python", test_decode_like_python},
    };
    int failed_tests = 0;
    for (const auto& [name, fn] : tests) {
        const int before = failures;
        std::printf("%s\n", name);
        try {
            fn();
        } catch (const std::exception& e) {
            std::printf("    EXCEPTION %s\n", e.what());
            ++failures;
        }
        if (failures != before) ++failed_tests;
        std::printf("  %s\n", failures == before ? "ok" : "FAILED");
    }
    std::printf("%zu tests, %d failed\n", tests.size(), failed_tests);
    return failed_tests ? 1 : 0;
}
