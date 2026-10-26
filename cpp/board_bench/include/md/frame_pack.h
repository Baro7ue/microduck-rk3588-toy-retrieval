// The C++ frame pack and the JSONL output. Given (Claude), not part of the 200 lines.
//
// PACK/cpp/ (made in the cloud by `python -m microduck_toy.stage2.cpp_bench make`):
//   frames.tsv        id, kind (window | letterbox), x0, y0, gain, pad_x, pad_y; one header line
//   uyvy/<id>.uyvy    the camera payload, 640x360 UYVY, 460,800 bytes, as the sim sends it
// On the board the Python reference adds inputs/<id>.rgb (its 320x320x3 input) for the first frames.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "md/types.h"

namespace md {

struct PackFrame {
    std::string id;
    Mapping map;
};

std::vector<PackFrame> read_frames_tsv(const std::string& path);

// Whole file as bytes; throws std::runtime_error when it can't be read or (expect > 0) has
// another size.
std::vector<uint8_t> read_file(const std::string& path, size_t expect = 0);
void write_file(const std::string& path, const uint8_t* data, size_t size);

// One output line: {"id": "...", "boxes": [[x0, y0, x1, y1, score, cls], ...], "ms": {...}}
std::string jsonl_line(const std::string& id, const std::vector<Box>& boxes,
                       const std::map<std::string, double>& ms);

}  // namespace md
