// ETHAN: the RAII wrapper around rknn_context. The contract (calls, checks, ownership) is in
// include/md/rknn_model.h; md_bench uses it on the board.
#include "md/rknn_model.h"

#include <stdexcept>

namespace md {

RknnModel::RknnModel(const std::string& path) {
    (void)path;
    throw std::logic_error("TODO(Ethan): RknnModel(path)");
}

RknnModel::~RknnModel() {}

RknnModel::RknnModel(RknnModel&& other) noexcept { (void)other; }

RknnModel& RknnModel::operator=(RknnModel&& other) noexcept {
    (void)other;
    return *this;
}

void RknnModel::run(const uint8_t* rgb, std::vector<RawOutput>& outs) {
    (void)rgb; (void)outs;
    throw std::logic_error("TODO(Ethan): RknnModel::run");
}

std::string RknnModel::sdk_version() const {
    throw std::logic_error("TODO(Ethan): RknnModel::sdk_version");
}

}  // namespace md
