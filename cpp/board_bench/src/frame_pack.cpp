#include "md/frame_pack.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace md {

std::vector<PackFrame> read_frames_tsv(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::string line;
    std::getline(in, line);  // header
    std::vector<PackFrame> frames;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::istringstream row(line);
        PackFrame f;
        std::string kind;
        if (!std::getline(row, f.id, '\t') || !std::getline(row, kind, '\t') ||
            !(row >> f.map.x0 >> f.map.y0 >> f.map.gain >> f.map.pad_x >> f.map.pad_y))
            throw std::runtime_error("bad line in " + path + ": " + line);
        if (kind != "window" && kind != "letterbox")
            throw std::runtime_error("unknown mapping kind " + kind + " in " + path);
        f.map.window = kind == "window";
        frames.push_back(f);
    }
    return frames;
}

std::vector<uint8_t> read_file(const std::string& path, size_t expect) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (expect && data.size() != expect)
        throw std::runtime_error(path + ": " + std::to_string(data.size()) + " bytes, expected " +
                                 std::to_string(expect));
    return data;
}

void write_file(const std::string& path, const uint8_t* data, size_t size) {
    std::ofstream out(path, std::ios::binary);
    if (!out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size)))
        throw std::runtime_error("cannot write " + path);
}

std::string jsonl_line(const std::string& id, const std::vector<Box>& boxes,
                       const std::map<std::string, double>& ms) {
    std::string s = "{\"id\": \"" + id + "\", \"boxes\": [";
    char buf[160];
    for (size_t i = 0; i < boxes.size(); ++i) {
        const Box& b = boxes[i];
        std::snprintf(buf, sizeof buf, "%s[%.4f, %.4f, %.4f, %.4f, %.7f, %d]", i ? ", " : "",
                      b.x0, b.y0, b.x1, b.y1, b.score, b.cls);
        s += buf;
    }
    s += "], \"ms\": {";
    bool first = true;
    for (const auto& [key, value] : ms) {
        std::snprintf(buf, sizeof buf, "%s\"%s\": %.4f", first ? "" : ", ", key.c_str(), value);
        s += buf;
        first = false;
    }
    return s + "}}";
}

}  // namespace md
