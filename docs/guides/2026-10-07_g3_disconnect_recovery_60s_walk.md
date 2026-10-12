# G3：断联恢复与 60 秒行走

这一轮连续完成：清理旧实验 → 新 HOME 身体 → 板端控制与转发 → 10 秒站立 → 60 秒前进 → 恢复站立。正常时按顺序直接往下执行，不需要每个小步骤都回聊天确认；出现报错时停在该处并保留完整输出。

板子的 CPU 运行 ONNX 行走策略，Mac 运行 MuJoCo 身体。独立 PI 在 Mac 上读取板端返回的 IMU 信息，修正发送给板子的转向指令。通信中断后，身体失去持续的控制更新，因此本轮从新的 HOME 状态重新开始。

## 1. 清理旧实验

先在旧板端 robotd 终端按 Ctrl+C，再在旧 Mac 身体终端按 Ctrl+C，最后停止旧的独立 SSH 转发终端。某个终端已回到命令提示符时，该项直接跳过。普通 SSH 登录窗口可以保留。

用途：释放旧控制程序、仿真身体与转发端点，避免两个控制程序同时连接，也避免沿用已经解除起步支撑的身体。原有日志保留。

## 2. Mac 终端 A：启动新的仿真身体

```bash
zsh "/Users/ethan/Documents/RK3588 Microduck/work/run_board_runtime_body.zsh"
```

它创建一只位于 HOME 姿态的鸭子，监听 192.168.50.1:7801，等待板端连接；同时记录本轮身体日志，供站立试验确认起步支撑已经解除。

看到 `starting at HOME` 和 `robotd --sim 192.168.50.1:7801` 后继续下一步。这个终端和 MuJoCo 窗口保持运行。

## 3. Mac 终端 B：同时启动板端控制程序和 SSH 转发

```bash
zsh "/Users/ethan/Documents/RK3588 Microduck/work/run_board_runtime_robotd.zsh"
```

按提示输入板子 SSH 密码。此脚本在同一个 SSH 会话里建立控制接口转发，并在板端核对模型与配置校验值、加载已有 ONNX Runtime 1.28.0、启动 robotd。无需再单独运行以前的 `ssh -N -L ...` 命令。

真正的策略推理仍在板子的 CPU 上执行。身体数据经过网线上的 7801 通道；SSH 转发用于状态查询与速度指令。

等待日志出现 `simulated body`、`protocol=1`、`policy loaded`、`control loop running` 后继续。此时模型已加载，运动由下一步启用。这个终端保持运行。

## 4. Mac 终端 C：一次完成站立与行走

```bash
zsh "/Users/ethan/Documents/RK3588 Microduck/work/probe_board_runtime_session.zsh"
```

程序先检查身体处于直立、零速度、未启用状态，然后启用 velstand.onnx。等策略接管且起步支撑解除后，记录完整 10 秒站立。只有这一步成功，才开始下一段行走。

随后运行 walk_17500.onnx，以 0.1 m/s 前进 60 秒，启用独立 PI 航向补偿。试验结束时自动停止前进，再确认恢复零速度站立。这个连接过程没有改变 PPO 网络本身。

观察三件事：鸭子是否不倒，长时间行走是否明显偏转，结束后是否恢复站立。

站立成功输出：`completed_no_fall`。

行走成功输出：`completed_no_fall_returned_stand`。

遇到摔倒、无效反馈或程序错误时，测试会尝试停止前进并关闭策略，保存失败记录；如果停止请求未获确认，按终端提示在终端 B 按 Ctrl+C。不要直接重复行走，重新从本教程第 1 步建立新实验。

## 5. 保存与回传

Mac 日志和原始状态保存在：

`/Users/ethan/Documents/RK3588 Microduck/task/evidence/G3/`

板端 robotd 完整日志保存在：

`/home/linaro/microduck-board/run/robotd-时间戳.log`

把终端 C 最后的完整输出发回，并说明是否仍明显偏转。A、B 继续保持运行；核对这一轮结果后，下一段是按计划进行 10 分钟持续运行测量。

准备状态：新增两个启动/试验组合脚本已经通过 Mac Zsh 与远端 Bash 的语法检查；本次完整恢复与 60 秒实验由你按上述流程执行，尚未验收。此前成功记录仅覆盖 10 秒行走，60 秒断联失败记录仍保留。
