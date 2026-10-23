# 阶段 1 · 1E 末段点动：接口、仿真和你要写的部分（2026-10-15）

依据：阶段 1 计划第 7 节（1E），以及你 Mac 上 1A–1D 和三次重跑的结果。
分工（10-15 你选的“三块都写”）：我写接口、点动仿真和单元测试，`decide()` 由你写，写完我来审。

## 1. 重跑结果（代码版本 30e6d6d）

| 测试 | 结果 | 判定 |
|---|---|---|
| 1B b_fall `runs/20261015T094124Z_s1b_s2` | 3 个玩具都在摔倒时松嘴（校车推左 1.05、Henry 推右 0.6、番茄酱推左 1.05 m/s）；另外 4 次推了没倒，玩具都还叼着 | 通过 |
| 摔倒处理（同一次） | 只用 `safety.fallen` 清零：从真值摔倒到 velstand 接手 0.34、0.35、0.34 s（上限 0.35）；1a、1c、c_trim 正常动作中误判 0 次 | 通过（计划要 4 次，实际 3 次） |
| 1C c_trim `runs/20261015T094236Z_s1c_s2` | 加右转后平均偏航变小，但离散翻倍（标准差 4.2–4.5°，原来 2.0–2.7°），没有一个前进或后退脉冲能走直 | 结论写进出口表 |
| 1D d_H1 `runs/20261015T095725Z_s1d_s2` | 0.20–0.60 m 误差 −0.77 到 −1.17 cm，随距离平滑变化；玩具不再压在托盘上 | 通过 |

所有数字已写进 `configs/control/stage1_facts.json`（阶段 1 出口表，1E 以后的代码只从这里和 `stage0_facts.json` 读数）。

## 2. 两个新发现，改了 1E 的设计

1. **每个前进点动都往左偏。** F2（vx 0.25 × 0.3 s）走 2.1 cm、左转 4.9 ± 2.7°；F5（× 0.5 s）走 5.3 cm、左转 4.2 ± 2.1°。机器人左转，玩具在它眼里就往右移：在 15 cm 处每走一步约右移 1 cm。计划里的“偏右瞄准 + 只左转”只能把玩具往右修，和这个漂移同方向，修不回来。
2. **H2 的画面左边很窄。** 用 1D 的相机位姿算：躯干前 8 cm 处，左边只看得到 4.2 cm，右边能看到 7.4 cm；20 cm 处左 6.6、右 10.5 cm。玩具太靠左会先出画面。

所以我把 R17（原地右转 wz −1.0 × 0.3 s，−17.2 ± 0.9°，很稳）加进了 1E 的动作表，并在线程里问你要不要保留。点动仿真的结果（只看 C 段，每组 1000 次）：

| 动作表 | 停进窗口并叼取的比例 | 每次闭嘴的期望成功率 |
|---|---|---|
| F2、F5、L2（只左转） | 约 25–30%；B 段横向误差压到 ±2 cm 也只有约 53% | 约 0.90 |
| F2、F5、L2、R17 | 约 90% | 约 0.91 |

B 段瞄准偏移 `aim_right_of_toy_m` 从计划的 2 cm 改成 1 cm（加 R17 后 0–2 cm 都差不多，1 cm 在两种 B 段精度下都稳；1F 再调）。

## 3. 这次的文件

| 文件 | 内容 |
|---|---|
| `src/microduck_toy/control/inching.py` | 接口：`Pulse`、`load_pulses`、`move_toy`、`Target`、`Look`、`InchingController`；**`reset()` 和 `decide()` 是你写的** |
| `src/microduck_toy/control/pulse_sim.py` | 点动仿真（不开 MuJoCo）：脉冲按实测均值和标准差抽样，测距按 1D 的误差加噪声，玩具出了 H2 画面就看不见 |
| `tests/test_stage1_inching.py` | 26 个测试：14 个测我给的部分（现在就通过），12 个测你的 `decide()`（现在失败，写好后全部通过） |
| `configs/control/stage1.json` | 新增 `inching` 段：目标点、窗口、预算、可用脉冲、B 段瞄准偏移，以及仿真用的参数 `inching.sim` |
| `configs/control/stage1_facts.json` | 阶段 1 出口表（新文件）：重跑结果、脉冲表 `pulses_mac`、测距校正表、H2 相机位姿 |

## 4. 你要写的：`InchingController.reset()` 和 `decide(look)`

C 段每一轮：站稳 1.0 s → 取 5 帧测距（运行程序已经做了校正和平均）→ 调一次 `decide(look)` → 按返回值发一个脉冲或结束。

**输入 `Look`**（坐标都是躯干水平坐标系，米；x 向前，y 向左）：

| 字段 | 意思 |
|---|---|
| `toy` | 玩具位置 `(前, 左)`；5 帧都没看到时是 `None` |
| `seen`、`frames` | 看到玩具的帧数 / 总帧数 |
| `elapsed_s` | 这次 C 段开始到现在的秒数 |
| `fallen` | 上次看之后 `safety.fallen` 有没有报过 |

**输出**：一个字符串，只能是下面之一：
`self.pulses` 里的名字（`'F2'`、`'F5'`、`'L2'`、`'R17'`），或 `PICK`（叼），或 `RETREAT`（放弃这次，后退约 10 cm 回 B 段重新瞄准），或 `ABORT`（摔倒了）。

**必须遵守的规则**（`ControllerRuleTests`，每条一个测试）：

1. 摔倒了，不管看到什么，返回 `ABORT`。
2. 看不到玩具，返回 `RETREAT`。
3. 玩具离目标点前后、左右都在 ±1.0 cm 以内，必须 `PICK`，预算用完也要叼；任一方向偏出 2.5 cm 以上，绝不 `PICK`。中间这一段由你定（配置里的窗口是 ±1.5 cm）。
4. 已经发了 12 个脉冲，或 `elapsed_s` ≥ 30 s，返回 `RETREAT`；`reset()` 之后重新计数。
5. 只能用 `self.pulses` 里的脉冲（动作表里没有 R17 时就不能发 R17）。
6. 前进脉冲必须满足计划里的规则：**均值 + 2 倍标准差 ≤ 剩余距离 + 窗口半宽**，也就是 `forward_m[0] + 2 * forward_m[1] <= (x − 0.08) + 0.015`。
7. 同样的输入序列，输出一样（不要用随机数）。

**要达到的水平**（`ClosedLoopTests`，用仿真跑 500 次 C 段，种子固定）：

| 指标 | 门槛 |
|---|---|
| 以叼取结束（不是后退）的比例 `picked` | ≥ 0.80 |
| 每次闭嘴的期望成功率 `grasp_p_per_close` | ≥ 0.85 |
| 平均脉冲数 `mean_pulses` | ≤ 7 |

## 5. 能直接用的工具

- `self.pulses['F2']` 是一个 `Pulse`（相当于 C 里的只读结构体）：`forward_m`、`left_m`、`yaw_rad` 都是 `(均值, 标准差)`，`role` 是 `'forward'` 或 `'turn'`。
- `pulse.predict(toy)`：按均值算这个脉冲之后玩具会出现在哪。背后是 `move_toy`，玩具不动、机器人动，所以是“先减平移、再反转偏航”：

```c
/* 机器人走了 (f, l)、左转了 yaw（弧度）；玩具在新的机器人坐标系里的位置 */
double x = toy_x - f, y = toy_y - l;
double new_x =  cos(yaw) * x + sin(yaw) * y;
double new_y = -sin(yaw) * x + cos(yaw) * y;
```

- `self.target.inside(toy)`、`self.target.error(toy)`（返回 `(前后误差, 左右误差)`）。
- `self.max_pulses`、`self.max_s`。

**一个要处理的空档**：F2 的“均值 + 2 倍标准差”是 3.5 cm，比窗口宽度 3 cm 还大，所以玩具在前 9.5–10.0 cm 时（刚过窗口远边），没有一个前进脉冲满足第 6 条。云端预跑时参考写法在这里只会一直发 L2，直到超时后退。你可以在这一段直接叼（前后差 1.5–2 cm，焊接模型的成功率约 0.75–0.86），也可以想别的办法。

**提示（不是答案）**：计划里写的“先修横向、再修前后”，我在仿真里按字面写了一版，`picked` 约 0.69，过不了 0.80，因为它没算前进时的左偏。对每个能发的脉冲先用 `predict` 算一下落点，再挑落点最好的那个，就能过门槛。“最好”怎么定义由你决定。

## 6. 命令

文件已经直接放进你 Mac 的仓库（10-15 10:40Z 左右，md5 核对过）。先看一眼：

```
cd "/Users/ethan/Documents/RK3588 Microduck"
git status --short
git diff --stat
```

应看到 ` M configs/control/stage1.json`（`git diff --stat` 显示只多了 24 行）和 5 个 `??` 新文件。如果没有，就用附件 `stage1_1e_20261015.tar.gz`：下载到“下载”文件夹后 `tar -xzf ~/Downloads/stage1_1e_20261015.tar.gz`（这次要覆盖 `stage1.json`，所以不加 `-k`）。

然后跑测试：

```
scripts/tools/run_tests.zsh -p test_stage1_inching.py
```

现在应看到 `FAILED (failures=12)`，失败的都是 `ControllerRuleTests` 和 `ClosedLoopTests`，提示 `decide is not written yet`。写好后应看到 `OK`，并打印一行 `1E closed loop: {...}`。

单独跑仿真、看每一步：

```
source env.sh
"$MD_PY" -m microduck_toy.control.pulse_sim --episodes 500
"$MD_PY" -m microduck_toy.control.pulse_sim --episodes 3 --show 3
"$MD_PY" -m microduck_toy.control.pulse_sim --episodes 500 --pulses F2,F5,L2
```

`--show 3` 会把前 3 次的每一步打出来（玩具真值、有没有看到、你的决定）。`--pulses` 换动作表，`--aim` 换 B 段瞄准偏移。

## 7. 写完之后

把代码放在 `inching.py` 里（只改 `reset` 和 `decide`，需要的话可以加自己的辅助函数），测试 `OK` 后告诉我，我来审。同时我写 1E/1F 的 MuJoCo 运行程序：B 段用航向 PI 走到约 0.25 m，C 段每一轮调用你的 `decide()`，然后叼取、打分。1F 就用它跑。

## 8. 讲解题（写完后用自己的话回答）

1. F2 让机器人左转 4.9°，为什么玩具在画面里往右移？在 15 cm 处大约移多少？
2. 第 6 条规则为什么用“均值 + 2 倍标准差”，而不是只用均值？
3. 仿真里的测距误差分成“每次 C 段一个偏差”和“每次看的噪声”两部分。为什么多取几帧求平均消不掉前一部分？

## 9. 1F：接近 + 叼取（等你的 `decide()` 测试 `OK` 之后再跑）

新增的文件：

| 文件 | 内容 |
|---|---|
| `src/microduck_toy/control/approach.py` | 接近程序（只用 `robot.state` 和画面里的框，以后板子上原样能用）：B 段在 H1 看一次、原地转向瞄准点或带航向 PI 往前走，再看，直到玩具进入 0.24 m；C 段在 H2 每轮站稳 1.0 s、取 5 帧测距、调用你的 `decide()`、发它选的点动 |
| `src/microduck_toy/stage1/s1f_approach.py` | 1F 测试：起点按种子定好（机器人朝向偏 ±10°，玩具在嘴尖前 0.25 ± 0.03 m、横向 ±5 cm、朝向 0–180°）；每个起点最多闭嘴 3 次，没叼住就后退（B10）重来；按真值打分 |
| `configs/control/stage1.json` | 新增 `approach` 段：B 段的转向、直走、交给 C 段的距离 |
| `tests/test_stage1_approach.py` | 8 个测试：校正表、把框换成位置、起点、判定 |

三个终端（和第 3 步一样，A 要加 `--camera`）：

| 终端 | 命令 |
|---|---|
| A | `scripts/sim/stage1/run_arena_body.zsh --camera` |
| B | `scripts/sim/stage0/run_robotd.zsh s2` |
| C | 先冒烟 `scripts/sim/stage1/run_test.zsh 1f --config s2 --quick`（3 个起点），再正式 `scripts/sim/stage1/run_test.zsh 1f --config s2 --parts f_main`（30 个起点），最后盲走对照 `--parts f_blind`（10 个起点） |

判定（计划第 8 节）：叼住次数 ÷ 闭嘴次数 ≥ 50% 就继续走叼取主线。另外看停位误差均方根 ≤ 2 cm、3 次以内叼住的起点 ≥ 80%、每个起点平均 ≤ 40 s、0 次摔倒。结果在 `summary.json` 的 `verdict.f_main`。
