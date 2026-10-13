# 阶段 0 测量：运行步骤（2026-10-13）

依据：`claude-docs/stage0_plan_20261012/MicroDuck阶段0详细计划_20261012.pdf`（第 2–5 节）。
这一步只新增文件，上游两个仓库和已有脚本一行没改。

## 新增了什么

| 文件 | 作用 |
|---|---|
| `src/microduck_toy/sim/probe_body.py` | 仿真身体 = `bootstrap_body.py` + 真值记录 + 测试端口 127.0.0.1:7811（推一下、摆躺姿、复位、放物体） |
| `src/microduck_toy/stage0/link.py` | 连接 robotd（每个控制周期收一次状态，20 Hz 重发速度和头部指令）和测试端口 |
| `src/microduck_toy/stage0/common.py` | 共用部分：运行目录、启用、复位、等待停稳、判断摔倒、`--rescore` |
| `src/microduck_toy/stage0/metrics.py` | 打分（纯函数）。站立判据照搬 lab 的 bench_getup：双脚着地、别处不碰地、重力 z < −0.85、躯干高 ≥ 站姿 − 4.5 cm；“完整站立”另要求嘴高 ≥ 站姿的 80% |
| `src/microduck_toy/stage0/s0c_getup.py` | 0c 起立：c1 四个头姿各站 10 s；c2 行走中推倒（先扫推力，再每方向 3 次）；c3 仰、俯、侧躺；每次站起后保持 5 s 再走 3 s |
| `src/microduck_toy/stage0/s0a_walk.py` | 0a 低头行走：a5 嘴尖/相机位置，a1v 前进死区，a1w 转向死区（冷/热），a2 直走 1 m（航向 PI），a3f/a3t 点动，a4 停车 |
| `src/microduck_toy/stage0/s0b_pick.py` | 0b GroundPick：头 H0、H2 各 10 次，叼完站 3 s 再走 0.5 m |
| `configs/robotd/stage0/{s1,s1u,s2,s2u}.toml.in` | s1 = alpha_stand + limp_fall；s2 = velstand；带 u 的是官方调参（0.9、低通 0.5/0.7、站立增益比 0.8），只在不通过时用来查原因 |
| `scripts/sim/stage0/run_probe_body.zsh` | 启动上面的仿真身体（默认给 Mac 上的 robotd；`--board` 给板子） |
| `scripts/sim/stage0/run_robotd.zsh` | Mac 上按所选配置启动 robotd |
| `scripts/sim/stage0/run_test.zsh` | 跑 0a / 0b / 0c |
| `scripts/board/stage0/run_board_robotd.zsh` | 板子上按所选配置启动 robotd（流程同 `scripts/board/run_board_robotd.zsh`） |
| `tests/test_stage0_*.py` | 22 个单元测试（打分、仿真身体、配置模板、启动脚本） |

我在云端用同一版本（microduck 43d0967、microduck_rl cb70b79）编译了 robotd，用 s2（walk_17500 + velstand）跑通了 0c、0a 的快速版；0b 用 velstand 临时顶替叼取网络只验证了流程。s1 和真正的 0b 要等两个官方模型。

## 第 1 步：两个官方模型（P1）

下载命令、记录 SHA-256 的命令在聊天回复里（从 PDF 或 Markdown 复制命令容易带错字符）。
下载完把 `alpha_ground_pick.onnx`、`alpha_stand.onnx` 两个文件直接拖进项目聊天，我先在云端预跑 s1 和 0b。

## 第 2 步：单元测试

```
scripts/tools/run_tests.zsh
```

应看到 `OK`。`test_stage0_probe` 需要 microduck_rl 的 Python 环境（就是 `$MD_PY`），没有会自动跳过。

## 第 3 步：每次测量开三个终端

都先 `cd` 到仓库根目录。

| 终端 | 命令 | 说明 |
|---|---|---|
| A | `scripts/sim/stage0/run_probe_body.zsh` | 仿真身体。加 `--viewer` 可以看画面 |
| B | `scripts/sim/stage0/run_robotd.zsh s2` | robotd，`s2` 换成要测的配置 |
| C | `scripts/sim/stage0/run_test.zsh 0c --config s2` | 测试脚本，自己发 `robot.enable`，跑完自动停 |

换配置时 A、B 都按 Ctrl+C 退出再重开（robotd 换网络后，仿真身体的起步支撑只在第一次启用时生效）。

先做冒烟（P4）：`run_test.zsh 0c --config s2 --parts c1 --quick`，四个头姿各站 3 s，不倒、`truth.csv` 每 20 ms 一行即通过。

## 第 4 步：按顺序测

| 顺序 | 终端 C 的命令 | 大约用时 |
|---|---|---|
| 0c，S2 | `run_test.zsh 0c --config s2` | 10 分钟 |
| 0c，S1 | `run_test.zsh 0c --config s1`（B 用 `run_robotd.zsh s1`） | 10 分钟 |
| 0a | `run_test.zsh 0a --config <0c 选中的>` | 20 分钟 |
| 0b | `run_test.zsh 0b --config <0c 选中的>` | 6 分钟 |

可以只跑一部分：`--parts c1,c3`、`--parts a3f,a3t`、`--parts b_H2`。`--quick` 每部分只做 1–2 次，用来先确认脚本能跑。
某项不通过时，用同名带 u 的配置（如 `s1u`）再跑同一项，看是不是调参引起的。

## 第 5 步：板子复跑（选定的配置）

| 终端 | 命令 |
|---|---|
| A | `scripts/sim/stage0/run_probe_body.zsh --board` |
| B | `scripts/board/stage0/run_board_robotd.zsh <配置>` |
| C | 见下 |

- 0c：`run_test.zsh 0c --config <配置> --target board --parts c2 --reps 1 --push <Mac 上扫出的推力>`（4 个方向各推 1 次；推力看 Mac 那次 `summary.json` 里 `c2_sweep` 第一个 `fell: true`）
- 0a：`run_test.zsh 0a --config <配置> --target board --parts a2,a3f,a3t`
- 0b：`run_test.zsh 0b --config <配置> --target board --parts b_H2`

## 结果在哪里

每次运行一个目录 `runs/<UTC>_s0<a|b|c>_<配置>/`：

| 文件 | 内容 |
|---|---|
| `summary.json` | `verdict`（对照计划验收值的结论，`pass` 为 true/false；快速版或只跑一部分时为 null）和 `parts`（每次试验的数字） |
| `truth.csv` | 仿真真值，每 20 ms 一行：躯干位置/朝向/速度、嘴尖 `tip_*`、相机 `cam_*`（`cam_pitch` = 俯角，正值朝下）、脚和其他部位是否碰地、`mark` = 脚本打的标记 |
| `state.jsonl` | robotd 每个周期的状态（去掉了 skeleton 和 frames） |
| `events.jsonl` | 脚本发出的每条指令和测试端口操作 |
| `trials.jsonl` | 每次试验的时间点 |

打分规则改了以后不用重跑：`run_test.zsh 0c --rescore runs/<目录>` 会按新规则重写 `summary.json`。

跑完告诉我目录名，我直接读 `summary.json` 并抽查 `truth.csv`，逐条对照验收值。

## 两个要知道的地方

- 真值只用来打分，不进控制器：robotd 看到的仍然只有上游协议里的传感器数据。
- 摆躺姿和复位是“瞬移”：复位时把机器人放回站姿并扶 1 s（和起步支撑一样），躺姿不扶、离地 5 mm 放下。所以 c3 测的是站立网络从静止躺姿起来的能力，和 lab 的 bench_getup 同一种起法。
