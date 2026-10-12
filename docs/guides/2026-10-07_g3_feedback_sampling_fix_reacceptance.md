# G3：修正反馈采样，重新完成持续行走

## 这次发生了什么

上一轮 600 秒试验在约 305.6 秒停止：`PI 的 IMU 采样间隔异常：0.100 秒；停止试验。` 完整原始记录中尚未检测到摔倒，停止与关闭请求都获确认。这轮没有完成 10 分钟验收。

原脚本请求 20 Hz 的状态，50 Hz 控制环经过服务端抽样后，实际约每 60 ms 才保留一帧。末尾一次读取到发布之间的额外延迟约 35 ms，叠加抽样后，相邻保留帧的 IMU 时间戳跨度达到 100 ms。现有记录不能再区分延迟来自身体读取、推理还是调度。

本次只修改工作目录中的外部试验脚本：PI 开启时接收每个控制周期的状态，按实际 IMU 时间戳积分。PI 增益、75 ms 采样间隔限制、0.2 秒反馈保护、模型、仓库提交和依赖版本均保持原值。PI 仍在 Mac 上，ONNX 推理仍在板子的 CPU 上。

修改前的脚本保存在 `work/probe_runtime_walk_before_unthrottled_20261007.py`。失败完整输出保存在 `task/evidence/G3/board_walk_console_20261007_210126.log`，原始状态与摘要保存在 `task/evidence/G3/runtime_walk_20261007_210126_665631/`。

准备验证：只读接收 8 秒，收到 401 个状态，新的传感器间隔中位数 19.85 ms、最大 20.96 ms。这仅验证反馈通道；修改后的行走仍需下面的实际试验。

## 1. Mac：取回上一轮资源证据

```bash
scp -r 'RK3588:/home/linaro/microduck-board/run/resources_*' \
  "/Users/ethan/Documents/RK3588 Microduck/task/evidence/G3/"
```

用途：保存板端已经记录的 CPU、内存、温度及控制环计数。即便资源观察正常完成，也包含约 5 分钟行走与关闭后的状态，不能算 10 分钟行走成功。

## 2. 结束旧实验，从新 HOME 开始

先在旧 robotd 终端按 Ctrl+C，再在旧身体终端按 Ctrl+C。旧资源采样如果仍在运行，也按 Ctrl+C，然后再执行一次上面的复制命令取回最终记录。已回到命令提示符的窗口直接跳过。

用途：上一轮错误已关闭控制，身体可能随后倒下；必须新建身体和起步支撑，避免直接启用倒下的旧身体。

## 3. Mac 终端 A：新建仿真身体

```bash
zsh "/Users/ethan/Documents/RK3588 Microduck/work/run_board_runtime_body.zsh"
```

用途：创建 HOME 姿态的鸭子，等待板端连接。看到 `starting at HOME` 后继续；窗口保持运行。

## 4. Mac 终端 B：启动板端控制及 SSH 转发

```bash
zsh "/Users/ethan/Documents/RK3588 Microduck/work/run_board_runtime_robotd.zsh"
```

用途：核对已有模型和配置、启动板端 CPU 推理，并把状态查询与速度指令接口转发到 Mac。输入 SSH 密码，看到 `control loop running` 后继续；窗口保持运行。

## 5. Mac 终端 C：连续完成 10 秒站立和 60 秒行走

```bash
zsh "/Users/ethan/Documents/RK3588 Microduck/work/probe_board_runtime_session.zsh"
```

用途：先确认起步支撑解除后能站立，再用修改后的反馈通道验证 60 秒行走。60 秒通过会显示 `completed_no_fall_returned_stand`，并保持零速度站立。

这段正常结束即可继续第 6、7 步，不需要每个小步骤回聊天确认。任何一步报错，停在该处，保留完整输出；不要重复启用旧身体。

## 6. Mac 终端 D：先启动资源采样

```bash
zsh "/Users/ethan/Documents/RK3588 Microduck/work/measure_board_runtime_resources.zsh"
```

用途：每秒记录板端 robotd 的 CPU 占用、内存、实际 CPU 温度和控制环计数。看到首行 `0 秒` 后立即执行下一步；屏幕每分钟显示一行，文件每秒记录一行。首次 CPU 显示 `-` 是因为还没有前后两个样本计算差值。

## 7. 回到 Mac 终端 C：运行 10 分钟

```bash
zsh "/Users/ethan/Documents/RK3588 Microduck/work/probe_board_runtime_walk.zsh" 600
```

用途：持续发送 0.1 m/s 前进指令，PI 用每个控制周期的 IMU 反馈修正航向。计满 600 秒后自动停止前进、确认恢复站立，并保存起止控制环计数。

成功标志：`completed_no_fall_returned_stand`。观察鸭子是否不倒、是否有明显偏转。资源采样持续 660 秒，结束后自动把证据复制回 Mac，可能再次要求 SSH 密码。

将终端 C 的最后完整输出、资源采样最后输出，以及是否明显偏转发回。收到完整证据后，再分析 10 分钟控制环和资源结果；通过后进入下一阶段。
