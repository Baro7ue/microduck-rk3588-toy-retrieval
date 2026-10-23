// md_bench: the offline board benchmark of plan 2H. Given (Claude), not part of the 200 lines.
//
//   md_bench --pack PACK/cpp --model yolo11n_toys_int8.rknn --out cpp.jsonl
//            [--conf 0.25] [--iou 0.45] [--python-inputs DIR] [--dump-inputs DIR] [--dump-n 20]
//
// Per frame: decode (UYVY -> upright RGB rows), resize (make_input), inference (RknnModel::run),
// postprocess (decode_split); each timed with steady_clock, written with the boxes (input pixels)
// to one JSONL line. --python-inputs feeds the Python reference's own 320x320 inputs instead of
// ours (inputs/<id>.rgb): then any box difference is in inference or post-processing, not in the
// resize. --dump-inputs writes our inputs of the first --dump-n frames for a byte comparison.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "md/frame_pack.h"
#include "md/postprocess.h"
#include "md/preprocess.h"
#include "md/rknn_model.h"
#include "md/uyvy.h"

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

double p50(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

struct Args {
    std::string pack, model, out, python_inputs, dump_inputs;
    float conf = 0.25f, iou = 0.45f;
    int dump_n = 20;
};

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        if (i + 1 >= argc) throw std::invalid_argument("missing value after " + k);
        const std::string v = argv[++i];
        if (k == "--pack") a.pack = v;
        else if (k == "--model") a.model = v;
        else if (k == "--out") a.out = v;
        else if (k == "--conf") a.conf = std::stof(v);
        else if (k == "--iou") a.iou = std::stof(v);
        else if (k == "--python-inputs") a.python_inputs = v;
        else if (k == "--dump-inputs") a.dump_inputs = v;
        else if (k == "--dump-n") a.dump_n = std::stoi(v);
        else throw std::invalid_argument("unknown option " + k);
    }
    if (a.pack.empty() || a.model.empty() || a.out.empty())
        throw std::invalid_argument("usage: md_bench --pack DIR --model FILE.rknn --out FILE.jsonl "
                                    "[--conf 0.25] [--iou 0.45] [--python-inputs DIR] "
                                    "[--dump-inputs DIR] [--dump-n 20]");
    return a;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse(argc, argv);
        const auto frames = md::read_frames_tsv(args.pack + "/frames.tsv");
        md::RknnModel model(args.model);
        std::fprintf(stderr, "md_bench: %zu frames, %s\n", frames.size(), model.sdk_version().c_str());

        std::ofstream out(args.out);
        if (!out) throw std::runtime_error("cannot write " + args.out);
        std::vector<uint8_t> rows, input(md::kInput * md::kInput * 3);
        std::vector<md::RawOutput> outs;
        std::map<std::string, std::vector<double>> times;
        int index = 0;
        for (const auto& f : frames) {
            const auto payload = md::read_file(args.pack + "/uyvy/" + f.id + ".uyvy",
                                               md::kRawW * md::kRawH * 2);
            std::map<std::string, double> ms;
            const int row0 = f.map.window ? static_cast<int>(f.map.y0) : 0;
            const int nrows = f.map.window ? md::kWindow : md::kFrameH;

            auto t = Clock::now();
            md::decode_upright_rows(payload.data(), row0, nrows, rows);
            ms["decode"] = ms_since(t);

            t = Clock::now();
            md::make_input(rows, row0, nrows, f.map, input.data());
            ms["resize"] = ms_since(t);

            if (!args.dump_inputs.empty() && index < args.dump_n)
                md::write_file(args.dump_inputs + "/" + f.id + ".rgb", input.data(), input.size());
            if (!args.python_inputs.empty())
                input = md::read_file(args.python_inputs + "/" + f.id + ".rgb", input.size());

            t = Clock::now();
            model.run(input.data(), outs);
            ms["inference"] = ms_since(t);

            t = Clock::now();
            const auto boxes = md::decode_split(outs, args.conf, args.iou);
            ms["postprocess"] = ms_since(t);

            out << md::jsonl_line(f.id, boxes, ms) << '\n';
            for (const auto& [k, v] : ms) times[k].push_back(v);
            ++index;
        }
        std::fprintf(stderr, "md_bench: P50 ms");
        for (const auto& [k, v] : times) std::fprintf(stderr, "  %s %.3f", k.c_str(), p50(v));
        std::fprintf(stderr, "\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "md_bench: %s\n", e.what());
        return 1;
    }
}
