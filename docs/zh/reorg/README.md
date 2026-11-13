# 项目文件夹整理与建 git：运行步骤（2026-10-11）

按 P0 到 P6 顺序做。每一步都有"预期结果"；结果不一致就停下，把终端输出发给 Claude，不要继续下一步。
所有命令都在 Mac 终端里、项目根目录下运行：

```zsh
cd "/Users/ethan/Documents/RK3588 Microduck"
```

`reorg.py` 只移动、只新增，不覆盖任何已有文件；每个动作写进 `docs/reorg/move_log.tsv`，可以按阶段撤销。
唯一不能撤销的是 P1 删除的缓存（`.DS_Store`、`__pycache__`），它们会自动重新生成。

## P0 备份并停掉运行中的程序

1. 关掉仿真身体、robotd、板端 SSH 会话（终端 A/B/C 全部 Ctrl+C）。脚本发现 `mjpython`、`robotd`、`body_server`、`tofd` 进程时会拒绝执行 P2、P3、P4（包括撤销）。
2. 做一个 APFS 克隆备份（几秒完成，几乎不占空间，写时复制）。备份放在主目录而不是"文稿"里，避免 iCloud 同步时上传几 GB：

   ```zsh
   cp -cR "/Users/ethan/Documents/RK3588 Microduck" "$HOME/RK3588 Microduck_backup_20261011"
   ```

   预期：命令没有输出；`ls ~` 能看到备份文件夹。整理验收完成后再自己删除这个备份。

## P1 检查计划，安装新文件

```zsh
python3 docs/reorg/reorg.py plan
```

预期：四行 `P1: ok …`、`P2: ok …`、`P3: ok …`、`P4: ok …`，没有以 `ERROR` 开头的行。
完整计划写在 `docs/reorg/move_plan.tsv`，可以用表格软件打开看每个文件去哪里。

```zsh
python3 docs/reorg/reorg.py apply P1
```

预期：删除约 50 个缓存，生成 `docs/reorg/snapshot_before.tsv`（整理前每个文件的 SHA-256），
把 `docs/reorg/staged/` 里的新文件装到正式位置（`src/`、`scripts/`、`configs/`、`tests/`、`env.sh`、README 等），最后 `staged/` 消失。

## P2 文档、证据、参考资料

```zsh
python3 docs/reorg/reorg.py apply P2
```

预期：`evidence/`、`docs/guides/`、`docs/analyses/`、`docs/plans/`、`references/`、`scripts/learning/` 出现；
`output/`、`outputs/`、`reports/`、`research_notes/`、`learning/`、`task1/`、`tmp/`、`参考资料/` 消失。

## P3 上游仓库、策略模型，重建环境

```zsh
python3 docs/reorg/reorg.py apply P3
```

预期：`task/` 消失；出现 `third_party/microduck`、`third_party/microduck_rl`、`models/policies`、`models/checkpoints/walk_qktrrkg5`；
旧的 Python 环境被挪到 `third_party/microduck_rl/.venv.pre_reorg`（里面写死了旧路径，只作回退用）。

确认两个上游仓库原样搬过来了：

```zsh
git -C third_party/microduck rev-parse --short HEAD
git -C third_party/microduck_rl rev-parse --short HEAD
git -C third_party/microduck status --short
git -C third_party/microduck_rl status --short
```

预期：前两行是 `43d0967` 和 `cb70b79`；第三条没有输出；第四条没有输出，或只有一行 `?? .venv.pre_reorg/`。

重建两样东西（路径变了，旧的编译结果和环境不能直接用）：

```zsh
(cd third_party/microduck_rl && uv sync --locked)
third_party/microduck_rl/.venv/bin/python -c "import mujoco, onnxruntime, numpy, mjlab_microduck; print('env ok', onnxruntime.__version__, numpy.__version__)"
zsh scripts/sim/build_mac_runtime.zsh
```

预期：第二行打印 `env ok 1.24.4 2.4.1`；第三行最后打印 `编译成功`，完整日志在 `runs/<时间>_build_mac_runtime/build.log`。

## P4 活跃代码归档，冒烟测试

```zsh
python3 docs/reorg/reorg.py apply P4
```

预期：`work/` 消失；出现 `models/yolo/yolo11n_coco/`（5 个文件）、`offline/`、`archive/2026-10_ball_follow/work/`。

1. 单元测试：

   ```zsh
   zsh scripts/tools/run_tests.zsh
   ```

   预期：`Ran 97 tests`，最后一行 `OK`（如果 Mac 上没有 `sha256sum` 命令，会显示 `OK (skipped=4)`，也算通过）。
   `test_camera_stream` 和 `test_yolo_camera_probe` 是 GPT 原来的计时测试，电脑很忙时偶尔失败一次；只有这两个失败时再跑一遍。

2. 板端冒烟测试（和 10-08 的站立加相机检测相同）。开三个终端，都先 `cd` 到项目根目录：

   ```zsh
   # 终端 A：仿真身体 + 头部相机 + 黄色球场景
   zsh scripts/sim/run_sim_body.zsh --scene sim/scenes/legacy_ball/ball_1m_yellow/scene_ball_1m.xml
   # 终端 B：等 A 出现监听信息后，启动板端 robotd 并转发控制接口
   zsh scripts/board/run_board_robotd.zsh
   # 终端 C：等 B 出现 robotd 运行日志后，站立 10 秒，再跑 30 秒板端 CPU 检测
   zsh scripts/board/probe_board_stand_camera.zsh
   # 终端 C：再跑一次 NPU 检测（站立保持，不需要重启 A、B）
   zsh scripts/board/probe_board_yolo_camera.zsh --npu
   ```

   预期：站立检查通过（和 10-08 一样，策略接管后站立 10 秒）；CPU 和 NPU 两次检测都在大多数帧里检出 `sports ball`
   （10-08 晚上的基线：450 帧全部检出，置信度约 0.83 到 0.89）。每次运行的证据在 `runs/<时间>_board_stand/`、
   `runs/<时间>_board_yolo_cpu/`、`runs/<时间>_board_yolo_npu/`，其中 `board/` 是从板子复制回来的结果。

3. 新代码里不再有写死的旧路径：

   ```zsh
   grep -rIn --exclude-dir=__pycache__ "/Users/ethan" src scripts/sim scripts/board scripts/vision scripts/tools scripts/lib.zsh configs sim tests env.sh
   ```

   预期：没有输出。

## P5 核对与清单

```zsh
python3 docs/reorg/reorg.py check
python3 docs/reorg/reorg.py manifest
```

预期：`check` 打印 `check passed`（整理前的每个文件都在新位置、内容不变）。后面可能跟一段 `note: … symlinks … point nowhere now`，
列出整理前能用、现在指向旧位置的链接；它们应该都在 `archive/` 里（旧场景指向 `task/` 的 `assets` 链接），属于正常。列出的链接如果不在 `archive/` 里，停下发给我。
`manifest` 在 `evidence/`、`offline/`、`references/`、`models/`、`data/`、`runs/`
和归档的两个大文件夹里各写一个 `MANIFEST.tsv`（路径、大小、SHA-256）。

## P6 建 git（提交由你自己做）

```zsh
git init -b main
git -C third_party/microduck_rl remote set-url origin https://github.com/pollen-robotics/microduck_rl.git
git submodule add https://github.com/pollen-robotics/microduck.git third_party/microduck
git submodule add https://github.com/pollen-robotics/microduck_rl.git third_party/microduck_rl
git config -f .gitmodules submodule.third_party/microduck.ignore untracked
git config -f .gitmodules submodule.third_party/microduck_rl.ignore untracked
chmod +x scripts/*/*.zsh docs/reorg/reorg.py
git add -A
git -c core.quotePath=false diff --cached --name-only | grep -E "^(evidence|runs|offline|references|data)/" | grep -v -E "/(README.md|MANIFEST.tsv)$"
git diff --cached --name-only -z | xargs -0 stat -f '%z %N' | awk '$1 > 1000000'
git diff --cached --name-only | wc -l
```

预期：两条 `submodule add` 显示 `Adding existing repo at 'third_party/…' to the index`（用已有的仓库，不重新下载）；
第一条检查没有输出（大文件夹只提交 README 和 MANIFEST）；第二条只列出 `models/yolo/yolo11n_coco/` 里的 `.onnx` 和 `.rknn` 两个文件（约 10 MB 和 7 MB），
列出别的大于 1 MB 的文件就先停下发给我；最后一条大约几百行。确认后：

```zsh
git commit -m "Reorganise project into repository layout"
du -sh .git
```

预期：`.git` 小于 50 MB。

然后在 GitHub 网页上新建私有仓库 `microduck-rk3588-toy-retrieval`（不要勾选 README、.gitignore、License），再：

```zsh
git remote add origin https://github.com/<你的用户名>/microduck-rk3588-toy-retrieval.git
git push -u origin main
```

## 出问题时

- `python3 docs/reorg/reorg.py status` 显示每个阶段做到哪里。
- 某一阶段中途停下（例如目标已存在）：按提示处理后重新 `apply` 同一阶段，已完成的动作不会重复。
- 撤销：`python3 docs/reorg/reorg.py undo P4`，再 `undo P3`……必须从最后一个阶段倒着撤。撤销前脚本先检查一遍，
  有任何障碍就列出来并且什么都不改。常见的两种：
  - 已经跑过 `uv sync`，撤销 P3 会说新的 `.venv` 挡路：先 `mv third_party/microduck_rl/.venv third_party/microduck_rl/.venv.rebuilt`，再撤销；之后旧的 `.venv` 回到原处。
  - 冒烟测试在 `runs/` 里留下了运行记录，撤销 P1 会要求先把它们挪出 `runs/`。
- `third_party/microduck_rl/.venv.pre_reorg` 占 1.3 GB：P6 推送成功、冒烟测试通过以后再删（`rm -rf third_party/microduck_rl/.venv.pre_reorg`）。删掉以后就不能再撤销 P3。
- 板端冒烟测试失败而整理本身没问题时：旧启动器都还在 `archive/2026-10_ball_follow/work/`，但它们指向旧路径，不能直接运行；把输出发给 Claude。
- 板子上的 `~/microduck-board` 这次不动；冒烟测试通过后再按计划整理板端（Q9）。
