# G4：YOLO11n 板端 CPU 单图检测

这一段验证：**RK3588 能否用固定的 YOLO11n ONNX 模型，在照片中识别人，并输出正确的框。** 接下来才接实时相机与跟随控制。

## 已准备与已验证

- 相机接收已通过：板子 30 秒收到 450 个完整负载，约 15 fps；回传画面已转正。当前画面只有地板。
- 已从官方 v8.3.0 资产发布下载 `yolo11n.pt`，导出静态 FP32 ONNX，batch=1，输入 `[1,3,320,320]`，输出 `[1,84,2100]`，opset=17，无内嵌 NMS。选择的是原计划的 YOLO11n。
- Mac CPU 测试照片检出 **3 个 person**；框坐标与官方 PyTorch 参考的最大差异约 0.00025 像素。空地板图片检出 0 个目标。**这些是 Mac 验证，板端尚未执行。**
- 检测器筛选 COCO 的 person（0）和 sports ball（32）；本次照片只验证人的正样本，球的正样本稍后另测。

## 1. 在 Mac 新终端运行完整流程

终端提示符应为 `ethan@… %`，无需先进入 SSH。原来的相机窗口可以保留；本实验只读图片，不依赖相机或 robotd。

```zsh
zsh "/Users/ethan/Documents/RK3588 Microduck/work/run_board_yolo11_image.zsh"
```

输入板子的 SSH 密码，密码不会显示；复制、远程运行和回传可能分别询问一次。

脚本按以下顺序执行，前一步失败便停止后续步骤：

| 步骤 | 在哪里 | 做什么与原因 |
|---|---|---|
| 复制离线包 | Mac → 板子 | 发送模型、脚本、两张图片及固定版本 wheel；板子不用联网下载。 |
| 校验 SHA-256 | 板子 | 检查所有文件的内容，防止传输损坏或拿错模型。 |
| 安装 Python CPU 绑定 | 板子 | 仅写入 `~/microduck-board/vision-cpu-python`；运行时通过 `PYTHONPATH` 读取这个隔离目录。ORT 使用与已有原生库相同的 1.28.0；NumPy 1.26.4、OpenCV 4.11.0.86 与已有 NPU 环境版本一致。 |
| 检测测试照片 | 板子 CPU | 图片缩放并补灰边到 320×320，BGR 转 RGB，再转 NCHW float32 `/255`；模型输出框和类别分数，筛选人/球，用 NMS 去除同类重复框，然后映射回原图。 |
| 检测地板图片 | 板子 CPU | 用实际相机的空画面验证无目标情形；没有框是合理结果。 |
| 回传结果 | 板子 → Mac | 保存两张标框图、检测 JSON 和简短推理计时，供核对。 |

这里的 `--no-index` 禁止在线下载，`--no-deps` 禁止自动补装依赖，`--require-hashes` 要求安装包与锁定哈希一致，`--target` 指定独立目录。不会修改现有 venv 的包、系统软件、仓库提交、模型或配置。

检测使用 CPUExecutionProvider、2 个推理线程；置信度阈值 0.25、NMS IoU 阈值 0.45。网络原始输出中的 80 项是类别分数，没有额外 objectness；不能套 YOLOv5 的输出解码。10 次计时仅涵盖 `session.run`，不包含预处理、后处理或相机传输，不能当成实时帧率。

## 2. 看结果并带回证据

预期测试照片提示“检测到 3 个 person/sports ball 目标”，实际类别应均为 person；地板图片提示 0 个。数目以板端实际输出为准，我会结合框的位置和 JSON 检查。

Mac 结果目录：

`/Users/ethan/Documents/RK3588 Microduck/task/evidence/G4/yolo11_cpu_UTC时间戳/`

其中 `bus/annotated.jpg` 应在照片中的三个人周围画框；`bus/detection.json` 记录类别、置信度、原图坐标、版本、模型哈希与推理耗时。`camera_floor/` 保存空画面的同类结果。

把终端完整输出发回即可；成功后文件已自动复制到本项目，我会读取并核对。任何报错都保留完整输出，在该处停止，不自行更换模型、升级依赖或跳过校验。完整控制台记录保存在 Mac `task/evidence/G4/yolo11_cpu_console_UTC时间戳.log`；推理异常另保存在板端相应输出目录的 `error.txt`。

通过后进入**相机连续检测 → 加入可见目标 → 视觉跟随**。本段不发送任何运动指令，不使用 NPU；NPU 对照属于 G4 后续阶段。

官方参考：[YOLO11](https://docs.ultralytics.com/models/yolo11/)、[ONNX Runtime 1.28.0 ARM64 安装包](https://pypi.org/project/onnxruntime/1.28.0/)。
