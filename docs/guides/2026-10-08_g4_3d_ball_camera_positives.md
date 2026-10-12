# G4：三维球相机正样本

本段完成 **三维球 → Mac 头部相机 → 网线 → RK3588 → YOLO11n CPU 检出**。运行 30 秒；先验证检测，再接入视觉跟随。当前不发送启用、前进或转向命令，HOME 起步辅助保持姿态不算策略站立成绩。

上一轮板端空场景已核实：450 个负载全部处理，处理吞吐 14.992 fps；纯推理 P50/P95 为 44.494/44.621 ms，完整处理为 53.527/53.747 ms。CPU 186.026% 表示约 1.86 个核心的 CPU 时间。处理完成后曾发生证据回传断联，现已补齐并核验 7 个结果文件。

## 1. 停止旧相机，启动球场景

上一轮检测结束后，在 **Mac 相机终端 A** 按一次 `Ctrl+C`，等 `%` 提示符返回。随后运行：

```zsh
zsh "/Users/ethan/Documents/RK3588 Microduck/work/run_board_runtime_body_camera_ball.zsh"
```

作用：使用外部球场景启动一只鸭子的身体与头部相机，身体端口 7801、相机端口 7901，发送设置 15 fps。仍使用 headless，因此没有 MuJoCo 观察窗口；画面从板端保存的图片检查。相机进程持续运行，请保留此终端。

为什么换场景：原球在前方 0.3 米、半径 0.035 米，相机在 HOME 时约 0.253 米高并朝前。原球太近，落在视野外。这里把同一个球放到前方 **1 米**。不修改鸭子的模型、球的半径/质量/惯量/摩擦/颜色，也不修改检测模型或阈值。

离线验证：Mac 使用实际相机渲染和相同 UYVY 解码，在仿真 0、1、10、30 秒取图，均检出 `sports ball`，置信度约 0.46。此结果仅证明准备的画面可作为正样本，板端结果仍需下面实测。

启动日志先记录场景文件哈希和模型哈希，然后应出现 `scene_ball_1m.xml`、身体地址 `192.168.50.1:7801`、相机地址 `192.168.50.1:7901`。日志保存在 `task/evidence/G4/mac_body_camera_ball_时间戳.log`。场景来源记录位于 `work/g4-scenes/ball_1m/manifest.json`，源提交仍是固定的 `cb70b79`。

## 2. 在 Mac 新终端 C 运行板端检测

在 **Mac 提示符** 下执行，无需先 SSH：

```zsh
zsh "/Users/ethan/Documents/RK3588 Microduck/work/probe_board_yolo11_camera.zsh"
```

作用：把已验证的检测脚本复制到板子、校验哈希，使用现有 CPU 依赖和 YOLO11n 320 模型，接收并检测 30 秒，随后回传结果。输入 SSH 密码时不会显示字符；复制、执行和回传可能分别询问。

保持终端 A 运行。运行中每约 5 秒打印计数，结束应显示 `结果：completed`。本段不会向身体端口或 robotd 发送运动指令。

## 3. 看入镜与检出结果

回传后的目录是：

`/Users/ethan/Documents/RK3588 Microduck/task/evidence/G4/camera_cpu_UTC时间戳/`

查看 `first_annotated.jpg` 和 `last_annotated.jpg`：球应位于画面中下方；`sports ball` 框应覆盖橙色球。`detections.jsonl` 记录每次处理的框、类别和置信度，`summary.json` 记录处理速度和耗时。把终端最后的完整输出发回，我会核对实际检出帧数与总处理帧数，并关联这次场景日志。

先分清两个问题：球有没有出现在图片里；出现后模型有没有检出。可见但零检出就如实记录，不降低阈值来制造通过。若任何命令报错，在该处停止并带回完整输出；模型检测失败的证据仍保留在板端本次目录。若已经显示 completed，但 scp 回传失败，不必重新跑实验，保留输出中的精确目录后补传。

## 4. 本段结束后

这次成功只能证明相机正样本在板端被检出。原计划 4a 的最终验收仍是鸭子根据检测结果走向目标并停止；完成正样本验证后才接视觉控制。

独立结果：`task/evidence/G4/camera-cpu-board-verification-20261008.md`。
离线准备证据：`task/evidence/G4/ball_target_preparation_20261008/verification.json`。

## 启动时出现 Address already in use

这表示旧仿真仍占用身体端口 7801 或相机端口 7901。先在旧相机终端按 Ctrl+C，随后在 Mac 检查：

```zsh
lsof -nP -iTCP:7801 -iTCP:7901 -sTCP:LISTEN
```

无输出后再启动球场景。若有输出，先核对 PID 与完整进程命令，停止确认的旧仿真进程；不要随意终止其他进程，也不要更换端口来跳过冲突。此次失败的完整输出已保存在对应 `mac_body_camera_ball_时间戳.log`，停止进程不会删除日志。
