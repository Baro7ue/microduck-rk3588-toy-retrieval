# 1E：分四步写 `decide()`

要改的文件只有 `src/microduck_toy/control/inching.py`，从 `# ---- Ethan: from here` 那行往下写。每一步写几行，再跑这一步对应的几个测试；全部变 `OK` 再进下一步。代码请自己敲，不要复制，`____` 是你要填的空。

规则 1–7 见 `docs/guides/2026-10-15_stage1_1E_inching.md` 第 4 节。

跑单个测试用 `-k`，后面跟测试名里的一段字（可以写好几个 `-k`）。命令都在仓库根目录跑：

```
cd "/Users/ethan/Documents/RK3588 Microduck"
scripts/tools/run_tests.zsh -p test_stage1_inching.py -k fallen -k lost
```

几个 Python 写法，对照 C 来看：

| Python | C 里相当于 |
|---|---|
| `look.toy is None` | `look->toy == NULL` |
| `look.toy[0]`、`look.toy[1]` | 前后、左右两个分量，像 `toy.x`、`toy.y` |
| `pulse.forward_m[0]`、`pulse.forward_m[1]` | 这个脉冲前进量的均值、标准差（米） |
| `for name, pulse in self.pulses.items():` | `for (i = 0; i < n; i++) { name = tbl[i].name; pulse = &tbl[i]; ... }` |
| `continue`、`return` | 和 C 一样 |

## 第 1 步：先处理意外（规则 1、2）

```python
    def decide(self, look: Look) -> str:
        if ____:                 # 规则 1：摔倒了
            return ABORT
        if ____:                 # 规则 2：这一轮没看到玩具
            return RETREAT
        return RETREAT           # 占位，第 2 步接着往下写
```

提示：`Look` 的字段在同一个文件里往上翻就能看到（`toy`、`seen`、`frames`、`elapsed_s`、`fallen`）。

先预测再回车：下面这条会显示跑了几个测试，结果是什么？

```
scripts/tools/run_tests.zsh -p test_stage1_inching.py -k fallen -k lost
```

## 第 2 步：够近就叼，预算用完就退（规则 3、4）

把第 1 步最后那行占位的 `return RETREAT` 换成：

```python
        if ____:                 # 规则 3：玩具在窗口里
            return PICK
        if ____ or ____:         # 规则 4：脉冲发够 12 个，或者时间到 30 s
            return RETREAT
        return RETREAT           # 占位，第 3 步换掉
```

提示：

- 判断“在窗口里”不用自己算，`self.target.inside(look.toy)` 已经写好了。它用的是配置里 ±1.5 cm 的窗口，正好符合规则 3（±1.0 cm 内必须叼，偏出 2.5 cm 以上绝不叼）。
- 预算的两个上限是 `self.max_pulses` 和 `self.max_s`。已经发了几个脉冲记在 `self.pulses_sent`（`reset()` 里清零）。从这一轮开始过了多久是 `look.elapsed_s`。

想一想：为什么“叼”要写在“预算用完”的**前面**？（有一个测试专门查这个。）

```
scripts/tools/run_tests.zsh -p test_stage1_inching.py -k picks_inside -k far_off -k pick_wins -k time_budget
```

## 第 3 步：挑一个脉冲（规则 5、6）

思路：把能发的脉冲挨个“假装发一下”，用 `pulse.predict(玩具位置)` 算出发完后玩具会出现在哪，挑落点离目标点最近的那个。前进脉冲还要先过规则 6，过不了就跳过。

先在 `decide` 下面加两个小函数（缩进和 `decide` 对齐）：

```python
    def _fits(self, pulse: Pulse, toy) -> bool:
        """规则 6：前进脉冲的 均值 + 2 倍标准差 <= 剩余距离 + 窗口半宽。"""
        room = ____              # 剩余距离 + 窗口半宽（米）
        return ____ <= room

    def _cost(self, point) -> float:
        """落点离目标点有多远，越小越好。"""
        ex, ey = self.target.error(point)      # 前后误差、左右误差（米）
        return ____
```

再把第 2 步最后那行占位的 `return RETREAT` 换成：

```python
        best_name, best_cost = None, None
        for name, pulse in self.pulses.items():
            if pulse.role == 'forward' and ____:   # 过不了规则 6 的前进脉冲：跳过
                continue
            c = self._cost(____)                   # 发完这个脉冲，玩具会在哪
            if best_cost is None or c < best_cost:
                best_name, best_cost = name, c
        if best_name is None:
            return RETREAT
        ____                                       # 记账：又发了一个脉冲
        return best_name
```

提示：

- 剩余距离 = 玩具的前后距离 − 目标前后距离。目标前后距离是 `self.target.forward_m`（0.08 m），窗口半宽是 `self.target.window_m`（0.015 m）。
- 距离用勾股定理，但可以不开根号：只是比大小的话，平方和就够了。想一想为什么。
- 规则 5（只能用 `self.pulses` 里的脉冲）你不用额外写，循环本来就只看 `self.pulses`。
- 为什么记账写在最后，不写在 `decide` 开头？因为 PICK、RETREAT、ABORT 都不是脉冲，不该算进 12 个里。

先预测：这 5 个测试里，哪个最可能因为忘了记账而失败？

```
scripts/tools/run_tests.zsh -p test_stage1_inching.py -k answers_are -k only_the_allowed -k overshoot -k pulse_budget -k same_looks
```

## 第 4 步：闭环仿真（及格线）

```
scripts/tools/run_tests.zsh -p test_stage1_inching.py
```

应看到 `Ran 26 tests` 和 `OK`，中间打印一行 `1E closed loop: {...}`。及格线是 picked ≥ 0.80、grasp_p_per_close ≥ 0.85、mean_pulses ≤ 7。

我用这套骨架、把空按最直接的写法填满，在云端跑过：picked 约 0.86，grasp_p_per_close 约 0.90，平均 6.0 个脉冲。能过线，但离 0.80 和 7 都不远。

看它每一步怎么走：

```
source env.sh
"$MD_PY" -m microduck_toy.control.pulse_sim --episodes 3 --show 3
```

## 第 5 步（选做，建议做）：把余量做大

先跑一次 `"$MD_PY" -m microduck_toy.control.pulse_sim --episodes 500` 记下结果，然后每次只改一处，再跑一次，填进这张表：

| 改了什么 | picked | parked | grasp_p_per_close | mean_pulses |
|---|---|---|---|---|
| 第 4 步原样 | | | | |
| `_cost` 里左右误差乘 2 | | | | |
| `_cost` 里左右误差乘 3 | | | | |
| 你自己的想法 | | | | |

可以从这三个方向想：

1. **左右为什么更要紧？** 每个前进脉冲都让机器人左偏约 5°，玩具在画面里往右跑；H2 的画面又是左边窄、右边宽。
2. **空档。** 玩具在前 9.5–10.0 cm 时，没有前进脉冲能过规则 6。用 `--show` 找一找这种情况，看它是不是在原地一直转。
3. **最大的损失是 `retreat_unseen`**，也就是玩具跑出了画面。看 `ends` 里它占多少，想想是哪种脉冲把玩具送出去的。

## 写完之后

测试 `OK` 后在这里说一声，我直接从你 Mac 上读 `inching.py` 来审，不用发文件。然后用自己的话回答 1E 说明第 8 节的 3 道题，再加一道：

4. 第 3 步里如果把“记账”写在 `decide` 的第一行，哪个测试会失败？为什么？
