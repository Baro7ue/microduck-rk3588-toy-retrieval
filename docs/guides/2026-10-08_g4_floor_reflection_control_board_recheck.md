# G4：保留橙色球，复核地面反射影响

本段目标：使用单独的无反射场景，重做一次站立与板端相机检测。先确认策略接管后的真实画面，再接视觉行走控制。

更新：上一轮 `camera_cpu_20261008_073949` 已完整补传，450 帧均检出恰好一个球，首末图片框覆盖真球。第 1 段保留为回传故障记录；现在直接从第 2 段开始。当前身体进程仍使用原有反射场景。

## 为什么做这个对照

Mac 离线实验保持橙球、相机、HOME 姿态、模型、320 输入和 0.25 阈值相同，只改变地板的镜面反射。

| 球的世界 x 坐标（m） | 原地面反射：检出球数 | 关闭反射：检出球数 | 关闭反射：置信度 |
|---:|---:|---:|---:|
| 0.70 | 0 | 1 | 0.348 |
| 0.75 | 0 | 1 | 0.371 |
| 0.80 | 0 | 1 | 0.629 |
| 0.85 | 0 | 1 | 0.541 |
| 0.90 | 0 | 1 | 0.581 |
| 0.95 | 0 | 1 | 0.411 |
| 1.00 | 1 | 1 | 0.527 |
| 1.05 | 2 | 1 | 0.417 |
| 1.10 | 0 | 1 | 0.370 |

1.05 m 原画面中的第二个框落在地面倒影上；关闭反射后只框住真球。缩放与已安装的官方 Ultralytics LetterBox 在两张核对图上逐像素、逐张量一致；换五种缩放插值也没有使原 0.95 m 图超过阈值。

这说明地面反射是本组静态画面中可复现的干扰。9 个静态位置通过不等于行走全过程可靠，也不代表真实相机检测准确率。YOLO 判断依赖球及其周围图像，倒影会改变这些图像特征；本次没有分析网络内部哪层导致分数变化。

## 1. 上一轮结果已保存（本段已完成）

上一轮板端检测 `camera_cpu_20261008_073949` 已显示 completed，处理 450 帧。最后核对时回传进程已退出，Mac 中仍没有该目录，完整结束错误为：

```text
Connection closed by 192.168.50.10 port 22
scp: Connection closed
```

[完整控制台日志](</Users/ethan/Documents/RK3588 Microduck/task/evidence/G4/camera_cpu_console_20261008_073949.log>) 已保留。在 Mac 补传这个精确目录：

```zsh
scp -r RK3588:/home/linaro/microduck-board/run/G4/camera_cpu_20261008_073949 \
  "/Users/ethan/Documents/RK3588 Microduck/task/evidence/G4/"
```

作用：保留已完成实验的图片和检测数据，不重跑上一轮检测。若询问密码，直接在该终端输入 SSH 密码。等复制完成、回到 Mac 的 `%` 提示符，再进入下一段；若再次报错，保留完整输出并在该处停止。

## 2. 结束旧会话，启动新的仿真身体

旧身体日志还记录过 `ConnectionResetError: [Errno 54] Connection reset by peer`，随后出现新的 daemon 连接。完整 traceback 已保存在[旧身体日志](</Users/ethan/Documents/RK3588 Microduck/task/evidence/G4/mac_body_camera_ball_20261008_144259.log>)。重连不能证明当前鸭子姿态正常，本段从新会话开始验证。

先在终端 B 按 `Ctrl+C`，等待板端控制程序退出；再在终端 A 按 `Ctrl+C`，等待旧身体进程退出。保留已有日志。

在 Mac 终端 A 运行：

```zsh
zsh "/Users/ethan/Documents/RK3588 Microduck/work/run_board_runtime_body_camera_ball_no_reflection.zsh"
```

作用：启动新的身体 7801 和相机 7901，球仍在 x=1 m。使用独立场景 `ball_1m_no_reflection`，地板 `reflectance` 从 0.2 改为 0；球仍是原来的橙色 70 mm、15 g 球。新启动日志开头会打印该变更和文件哈希。

已离线验证：两场景编译后的 469 个数组中只有 `mat_reflectance` 不同；球文件字节一致，相机与物理参数一致，新场景的 HOME 原始帧与无反射对照实验哈希相同。原场景和两个固定提交仓库均保留。现在实际运行的旧会话不会被文件创建自动改变。

若出现 Address already in use 或其他报错，在该处停止，把完整输出带回来。不要再开另一个身体进程。

## 3. 启动板端控制，并一次完成站立与检测

在 Mac 终端 B 运行：

```zsh
zsh "/Users/ethan/Documents/RK3588 Microduck/work/run_board_runtime_robotd.zsh"
```

作用：板端加载现有策略，连接新身体，建立控制接口转发。等待 `policy loaded`、`control loop running`，保持 B 运行。

在 Mac 终端 C 运行：

```zsh
zsh "/Users/ethan/Documents/RK3588 Microduck/work/probe_board_stand_with_camera.zsh"
```

作用：头部命令增量归零；启用站立策略，确认新会话的起步支撑解除后观测 10 秒；成功后保持站立，执行 30 秒板端 CPU 相机检测并回传数据。整个流程不发送前进或转向命令。

必须在刚重启的身体与控制程序上运行这一入口；旧身体的起步支撑只解除一次，不能直接重用旧会话作为新的站立实验。零头部命令表示相对名义 HOME 的零增量，实际姿态以本次观测为准。

## 4. 审核结果，再进入视觉行走

站立应出现 `completed_no_fall`，检测应出现 `completed`。等待结果回传后，带回终端 C 的完整输出，以及本次 `first_annotated.jpg`、`last_annotated.jpg`。

我们将核对：球是否可见、框是否覆盖球、漏检或双球框是否仍存在，以及板端处理耗时。检测 completed 只说明程序完成，不能代替这些检查；10 秒站立证据也不能覆盖后续 30 秒全部姿态。

这轮通过后，再按 Claude 计划把水平框误差接到偏航、框面积接到前进，并在球仍可见时停止。视觉接近与停止演示仍未完成。

## 本次证据

- [反射对照原始数据](</Users/ethan/Documents/RK3588 Microduck/task/evidence/G4/orange_ball_reflection_diagnosis_20261008/summary.json>)
- [原场景：真球与倒影两个框](</Users/ethan/Documents/RK3588 Microduck/task/evidence/G4/orange_ball_reflection_diagnosis_20261008/x_1p05_reflect_0p2_annotated.jpg>)
- [无反射：只框住真球](</Users/ethan/Documents/RK3588 Microduck/task/evidence/G4/orange_ball_reflection_diagnosis_20261008/x_1p05_reflect_0p0_annotated.jpg>)
- [预处理一致性核对](</Users/ethan/Documents/RK3588 Microduck/task/evidence/G4/orange_ball_preprocess_diagnosis_20261008/preprocess_comparison.json>)
- [新场景参数及帧哈希核对](</Users/ethan/Documents/RK3588 Microduck/task/evidence/G4/orange_ball_reflection_diagnosis_20261008/scene_variant_verification.json>)

诊断过程中没有安装或升级依赖。导入已隔离的官方 Ultralytics 库时，库自动创建了 `/Users/ethan/Library/Application Support/Ultralytics/settings.json`；这项副作用已记录。
