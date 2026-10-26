// The NPU model behind RAII. ETHAN WRITES src/rknn_model.cpp (part of the ~200 lines).
//
// RAII ("resource acquisition is initialisation"): the constructor acquires the NPU context
// (rknn_init) and the destructor always releases it (rknn_destroy), even when an exception leaves
// the scope early. Like a C struct whose open() is the constructor and close() the destructor, but
// the compiler calls close() for you. Copying is forbidden (two owners would destroy twice);
// moving hands the context over and leaves the source empty (ctx_ = 0, its destructor does nothing).
//
// rknn_api calls to use (rknn_api.h; examples: rknn_model_zoo examples/yolo11/cpp/rknpu2/yolo11.cc):
//   rknn_init(&ctx, model_bytes, size, 0, nullptr)
//   rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM / RKNN_QUERY_INPUT_ATTR / RKNN_QUERY_OUTPUT_ATTR /
//              RKNN_QUERY_SDK_VERSION, &value, sizeof value)
//   rknn_inputs_set(ctx, 1, &input)    input: index 0, type RKNN_TENSOR_UINT8, fmt RKNN_TENSOR_NHWC,
//                                      size 320 * 320 * 3, pass_through 0 (the runtime applies the
//                                      mean 0 / std 255 compiled into the model)
//   rknn_run(ctx, nullptr)
//   rknn_outputs_get(ctx, n, outs, nullptr)   want_float = 0: int8 data, NCHW, as the attrs say
//   rknn_outputs_release(ctx, n, outs)        every get needs a release, also on error paths
//   rknn_destroy(ctx)
// Every call returns an int; anything but RKNN_SUCC (0) -> throw std::runtime_error with the call
// name and the code. The model must be the INT8 one: every output attr has type RKNN_TENSOR_INT8,
// 9 outputs in the order box8 cls8 sum8 box16 cls16 sum16 box32 cls32 sum32, shapes
// [1,64,40,40] [1,6,40,40] [1,1,40,40] ... [1,1,10,10]; refuse anything else.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "rknn_api.h"
#include "md/types.h"

namespace md {

class RknnModel {
public:
    explicit RknnModel(const std::string& path);   // read the file, rknn_init, query and check attrs
    ~RknnModel();                                    // rknn_destroy when ctx_ != 0
    RknnModel(const RknnModel&) = delete;
    RknnModel& operator=(const RknnModel&) = delete;
    RknnModel(RknnModel&& other) noexcept;
    RknnModel& operator=(RknnModel&& other) noexcept;

    // One inference on a 320x320x3 RGB uint8 image. Fills `outs` (9 entries, resized as needed)
    // with the int8 data, c/h/w from the output attrs (dims are NCHW), zp and scale from the attrs.
    void run(const uint8_t* rgb, std::vector<RawOutput>& outs);

    std::string sdk_version() const;                // "API: x.y.z / driver: a.b.c"

private:
    rknn_context ctx_ = 0;
    std::vector<rknn_tensor_attr> out_attrs_;
};

}  // namespace md
