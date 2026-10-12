# G3：保存资源证据，测量 CPU 策略推理耗时

## 已确认的结果

2026-10-07 的最新试验完整行走 600.017 秒，没有检测到摔倒，随后恢复零速度站立。控制环起止测量窗口为 600.020 秒，共 30,001 个 tick，平均 50.000 Hz；新增 4 次执行超时，占 0.0133%。这里的执行超时指单轮工作超过 20 ms，并非所有调度延迟的计数。

逐拍状态共 30,067 帧，IMU 新样本间隔最大 29.28 ms，PI 反馈检查没有再次触发。PI 仍由 Mac 计算；策略推理由板端 CPU 执行。

持续行走记录：`task/evidence/G3/runtime_walk_20261007_213452_004199/`。这一轮的资源文件尚未回传完成，CPU 占用和内存结果仍待核对。

## 1. Mac 终端 D：完成资源文件回传

回到资源采样终端。如果出现 `linaro@192.168.50.10's password:`，输入板子的 SSH 密码。这是复制文件时的再次认证；输入密码时屏幕不显示字符。

等待复制完成、返回 Mac 命令提示符。资源记录在板端已经保存，复制完成后会出现在：

`/Users/ethan/Documents/RK3588 Microduck/task/evidence/G3/resources_时间戳/`

如果复制报错，保留完整输出，先处理该错误。

## 2. 结束控制实验，避免影响基准测量

先在终端 B（板端 robotd）按 Ctrl+C，再在终端 A（Mac 仿真身体）按 Ctrl+C。它们都返回命令提示符后继续。

用途：停止在线控制和仿真，避免另一个推理进程与基准程序争用 CPU。已有试验日志保留。

## 3. Mac 新终端：登录板子

```bash
ssh RK3588
```

后面的测量命令全部在 `linaro@ATK-DLRK3588:~$` 这个板端终端执行。

## 4. 板端 SSH 终端：编译、测量、显示结果

整段一起复制执行：

```bash
set -o pipefail
export PATH="$HOME/rust-1.89.0/bin:$PATH"
export ORT_DYLIB_PATH="$HOME/onnxruntime-linux-aarch64-1.28.0/lib/libonnxruntime.so.1.28.0"
RK_POLICY_BENCH_DIR="$HOME/microduck-board/run/policy_cpu_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RK_POLICY_BENCH_DIR" &&
cd "$HOME/microduck-g3-offline/microduck" &&
cargo build --release --locked --offline -j 2 \
  -p duck-control --example policy-rehearsal \
  2>&1 | tee "$RK_POLICY_BENCH_DIR/build.log" &&
target/release/examples/policy-rehearsal \
  "$HOME/microduck-board/policies/walk_17500.onnx" \
  > "$RK_POLICY_BENCH_DIR/result.json" \
  2> >(tee "$RK_POLICY_BENCH_DIR/runtime_stderr.log" >&2) &&
python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps({k:v for k,v in d.items() if k!="actions"},indent=2))' \
  "$RK_POLICY_BENCH_DIR/result.json"
```

这段命令做四件事：

1. 使用已安装的 Rust 1.89.0 与 ONNX Runtime 1.28.0。
2. 编译固定仓库里已有的 `policy-rehearsal` 例程。`--locked --offline` 固定依赖并禁止联网获取依赖，`-j 2` 限制同时编译的任务数。
3. 加载同一个 `walk_17500.onnx`，先预热 50 次，再测量 1,000 次策略推理；输入是例程构造的 61 维观测，输出是 14 维动作。
4. 保存全部动作和延迟统计，屏幕只显示统计摘要。

例程仅执行离线推理，不发送运动指令。它测量 `policy.infer` 耗时，不包括身体通信、状态 JSON 传输或电机读写。

成功时会看到：`steps: 1000`、`latency_ms`、`over_20_ms` 和 `scope`。

- p50：一半推理不超过该时间。
- p95 / p99：95% / 99% 推理不超过该时间。
- max：这 1,000 次中最慢的一次。
- over_20_ms：超过 50 Hz 单周期 20 ms 预算的次数。

任何一步报错，后面的测量不会继续；把完整输出发回。不要升级依赖或更换模型。

## 5. Mac 本地终端：取回推理证据

另开 Mac 本地终端执行：

```bash
scp -r 'RK3588:/home/linaro/microduck-board/run/policy_cpu_*' \
  "/Users/ethan/Documents/RK3588 Microduck/task/evidence/G3/"
```

用途：把编译日志、运行错误日志和完整推理结果保存到项目证据目录。发回第 4 步屏幕上的统计结果；资源和推理文件复制完成后，我们一起核对 G3 测量，再推进下一阶段。

准备状态：命令已按固定仓库例程与离线工具链说明核对，尚未在板端执行本次延迟基准。
