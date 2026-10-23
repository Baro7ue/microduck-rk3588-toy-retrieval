# 板端 C++ 推理基准（阶段 2，计划 2H）

用 `rknn_api`（C 接口）在 RK3588 的 NPU 上跑 INT8 模型，前后处理用 C++17 自己写。这是离线基准程序，不进闭环（闭环用 Python 版 `stage2/deploy.py`，2F 已经量过）。

## 谁写什么

| 文件 | 谁写 | 内容 |
|---|---|---|
| `include/md/*.h` | Claude | 每个函数的接口和要求（注释里写清楚要和 Python 哪一行一致） |
| `src/uyvy.cpp`、`src/frame_pack.cpp`、`src/main.cpp` | Claude | 相机原始画面解码、读帧包、写 JSONL、计时 |
| `tests/unit_tests.cpp`、`tests/fixtures.h` | Claude | 9 个单元测试；答案由 `tests/make_fixtures.py` 用 Python 算出 |
| **`src/preprocess.cpp`** | **Ethan** | `resize_linear`（和 OpenCV 逐字节一致的双线性缩放）、`make_input`（窗口 / letterbox） |
| **`src/postprocess.cpp`** | **Ethan** | `dfl`、`iou`、`class_aware_nms`、`decode_split`（INT8 反量化在里面） |
| **`src/rknn_model.cpp`** | **Ethan** | `RknnModel`：构造时 `rknn_init`，析构时 `rknn_destroy`（RAII），`run` 一次推理 |

三个文件现在是空壳，每个函数只抛 `TODO(Ethan)`，所以能编译、测试会失败。合起来约 200 行。

参考（读懂思路后自己写，不拷代码）：airockchip/rknn_model_zoo 的 `examples/yolo11/cpp/postprocess.cc`，`nms`（第 109 行）、`compute_dfl`（208）、`process_i8`（300）；`examples/yolo11/cpp/rknpu2/yolo11.cc` 里 `rknn_init` / `rknn_query` / `rknn_outputs_get` 的用法。

## 在 Mac 上先过单元测试（不需要板子）

```
cmake -S cpp/board_bench -B cpp/board_bench/build -DMD_WITH_RKNN=OFF
cmake --build cpp/board_bench/build -j4
./cpp/board_bench/build/unit_tests
```

没有 cmake 的话，一条命令也行（macOS 自带 clang）：

```
c++ -std=c++17 -O2 -ffp-contract=off -Icpp/board_bench/include cpp/board_bench/tests/unit_tests.cpp \
  cpp/board_bench/src/{uyvy,frame_pack,preprocess,postprocess}.cpp -o /tmp/md_unit_tests && /tmp/md_unit_tests
```

现在会显示 `9 tests, 8 failed`（只有我写的 `decode_upright_rows` 那一项过）。你每写完一个函数，对应那一项变成 `ok`。全部 `ok` 之前不用上板。

建议顺序：`iou` → `dfl` → `class_aware_nms` → `decode_split` → `resize_linear` → `make_input` → `RknnModel`（最后这个只能在板上验证）。

## 上板对拍（一条命令）

等评估包做好（里面有 `cpp/` 帧包：100 帧相机原始画面，一半搜索窗口、一半 letterbox），在 Mac 仓库根目录：

```
scripts/board/stage2/run_cpp_bench.zsh runs/<UTC>_s2e_pack
```

它会：在板上用 g++ 原生编译 → 跑单元测试 → C++ 跑 100 帧 → Python（同一个 INT8 模型、同样的帧）跑一遍 → 回到 Mac 对比。第一次会自动下载 Rockchip 的 `rknn_api.h` 到 `offline/rknn-2.3.2/`（不进 git，因为它不是开源许可）。

## 验收（计划 2H）

- 每帧框数相同；每个框和 Python 同类的框 IoU ≥ 0.99；置信度差 ≤ 1e-3。
- 给出 C++ 和 Python 后处理各自的 P50 耗时（`compare.json` 的 `p50_ms`）。
- 对不上：看 `compare.json` 的 `first_bad_frame`，再看 `inputs`（前 20 帧输入逐字节比）。输入不同 → 查 `resize_linear` / `make_input`；输入相同框不同 → 查 `decode_split`。

## 面试时能讲的点

1. RAII：为什么 `RknnModel` 禁止拷贝、允许移动；构造到一半抛异常时谁来 `rknn_destroy`。
2. INT8：`(q - zp) * scale`；为什么阈值可以直接在 int8 上比（`q > conf / scale + zp`），省掉大部分反量化。
3. DFL：16 个格子的 softmax 期望；为什么先减最大值（不减，`exp(1000)` 溢出）。
4. NMS：为什么要稳定排序（INT8 分数离散，常有并列，顺序决定留哪个）。
5. 双线性缩放的定点算法：为什么要和 OpenCV 逐字节一致（一个像素差 1，INT8 分数可能跳一个量化台阶，约 0.004）。
6. 浮点：为什么编译要加 `-ffp-contract=off`（FMA 少一次舍入，结果和 numpy 差 1）。
