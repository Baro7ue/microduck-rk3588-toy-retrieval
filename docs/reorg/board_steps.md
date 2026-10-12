# 板端整理运行步骤（B0 到 B7）

方案：`claude-docs/board_cleanup_20261012/板端整理方案_20261012.md`（4 个问题都选 A）。
所有命令都在 Mac 的仓库根目录运行：

```zsh
cd "/Users/ethan/Documents/RK3588 Microduck"
```

`docs/reorg/board_reorg.zsh` 通过 SSH 操作板子，每次运行都在 `runs/<UTC>_board_reorg_<命令>/` 留下记录（`console.log`）。
同一时间板上只能跑一个步骤。在 `apply`/`undo` 中途按 Ctrl+C 不一定能马上停下板上那一步；先跑 `status` 看清楚，再决定 `undo` 还是重来。
板上的每次移动和复制都记进 `~/microduck/move_log.tsv`，`undo` 按记录倒着撤回。任何一步报错就停下，把输出发给我。

## 先提交整理工具

```zsh
zsh scripts/tools/run_tests.zsh
git add docs/reorg/board_reorg.zsh docs/reorg/board_reorg_remote.sh docs/reorg/board_steps.md \
  docs/reorg/board_switch.patch configs/board tests/test_board_reorg.py
git commit -m "Add board home reorganisation tool"
```

预期：测试全部 OK。`test_board_reorg` 需要 bash 4 和 GNU 工具，Mac 自带的 bash 3.2 会让它显示为 skipped，这是正常的（我在 Linux 上跑过，13 个全过）。

## B0 只读预检

```zsh
zsh docs/reorg/board_reorg.zsh precheck
```

结果存到 `runs/<UTC>_board_reorg_precheck/precheck.txt`。跑完告诉我，我来看：有没有进程在跑、`~/.profile` 里有没有写死的 Rust 路径、vendor 路径是不是相对路径。

## B1 把板上证据拷回 Mac

```zsh
zsh docs/reorg/board_reorg.zsh backup
python3 docs/reorg/reorg.py manifest
git add evidence/MANIFEST.tsv && git commit -m "Back up board evidence before reorganising"
```

预期：打印 `核对通过：N 个文件，两边 SHA-256 一致`，备份在 `evidence/board_home_<日期>/`。没有这一行，B2 会拒绝执行。

## B2 建骨架，移走安装包

```zsh
zsh docs/reorg/board_reorg.zsh plan B2      # 只列出要做什么，每行末尾应是 [ok] 或 [已存在，跳过]
zsh docs/reorg/board_reorg.zsh apply B2
```

验收：启动脚本用到的东西一个都没动，所以按现在的路径跑一次 NPU 检测，结果应当照常（先在另一个终端用 `zsh scripts/sim/run_sim_body.zsh` 开仿真身体）：

```zsh
zsh scripts/board/probe_board_yolo_camera.zsh --npu --duration 10
```

预期最后一行 `实验退出码：0；证据回传退出码：0`。撤回：`zsh docs/reorg/board_reorg.zsh undo B2`。

## B3 Mac 端改动（我已写好，先看不用跑）

改动都在 `docs/reorg/board_switch.patch` 里，B5 才应用。可以先看改了哪些文件：

```zsh
git apply --stat docs/reorg/board_switch.patch
```

## B4 并排装新运行环境

```zsh
zsh docs/reorg/board_reorg.zsh plan B4
zsh docs/reorg/board_reorg.zsh apply B4
```

预期：pip 装 9 个包，最后一行是
`rknn-toolkit-lite2 2.3.2 onnxruntime 1.28.0 numpy 1.26.4 opencv 4.11.0`，然后是 `B4 完成`。
旧环境完全没动。撤回：`zsh docs/reorg/board_reorg.zsh undo B4`（只删 `~/microduck/runtime/` 里的副本）。

## B5 切换并冒烟测试

```zsh
git apply docs/reorg/board_switch.patch
zsh scripts/tools/run_tests.zsh
```

然后按今天的顺序冒烟，三个终端：

```zsh
zsh scripts/sim/run_sim_body.zsh                        # 终端 1：仿真身体（带相机）
zsh scripts/board/run_board_robotd.zsh                  # 终端 2：板上 robotd
zsh scripts/board/probe_board_stand_camera.zsh          # 终端 3：站立 10 秒 + CPU 检测
zsh scripts/board/probe_board_yolo_camera.zsh --npu     # 终端 3：NPU 检测
```

验收（和 10-12 的数字比）：站立 10 秒不倒、50 Hz；CPU 和 NPU 各 450/450 帧检出；推理中位数在 CPU 44.5 ms、NPU 15.4 ms 的 ±10% 以内；
在终端 2 按 Ctrl+C 停掉 robotd 后，`runs/<UTC>_board_robotd/board/robotd.log` 应当拷回来了。全部通过后再提交：

```zsh
git add -A && git commit -m "Switch board paths to ~/microduck"
```

撤回：还没提交时用 `git apply -R docs/reorg/board_switch.patch`；已经提交用 `git revert HEAD`。旧路径在 B6 之前一直完好。

## B6 移走旧目录

```zsh
zsh docs/reorg/board_reorg.zsh plan B6
zsh docs/reorg/board_reorg.zsh apply B6
```

`apply B6` 会先检查 `env.sh` 已经切到新路径，没切就拒绝。完成后再跑一次站立和 NPU 检测（同 B5），并确认 Rust 还能用：

```zsh
ssh RK3588 '~/microduck/toolchains/rust-1.89.0/bin/cargo -V'
```

如果 B0 显示 `~/.profile` 里写了 `rust-1.89.0` 的路径，我会给出要改的那一行。撤回：`zsh docs/reorg/board_reorg.zsh undo B6`。

## B7 收尾

```zsh
zsh docs/reorg/board_reorg.zsh apply B7         # 装 ~/microduck/README.md
zsh docs/reorg/board_reorg.zsh inventory        # 整理后的清点
zsh docs/reorg/board_reorg.zsh status
```

跑完告诉我，我核对清点结果并更新项目记录。

## 撤销的顺序

只能倒着撤：`undo B7` → `undo B6` → `undo B4` → `undo B2`。脚本会拒绝跳步。
撤销只会删除脚本自己复制出来的东西，原来的文件全部移回原位；B6 删掉的 `__pycache__` 是缓存，不恢复。
