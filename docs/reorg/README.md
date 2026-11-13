# Reorganising the project folder and setting up git: steps to run (2026-10-11)

Do P0 to P6 in order. Every step has an "expected result"; if the result differs, stop and send the terminal output to Claude instead of going on to the next step.
Every command runs in a Mac terminal from the project root:

```zsh
cd "/Users/ethan/Documents/RK3588 Microduck"
```

`reorg.py` only moves and adds; it overwrites no existing file. Every action is written to `docs/reorg/move_log.tsv` and can be undone phase by phase.
The only thing that cannot be undone is the caches P1 deletes (`.DS_Store`, `__pycache__`), which are regenerated automatically.

## P0 back up and stop the running programs

1. Shut down the sim body, robotd and the SSH sessions to the board (Ctrl+C in terminals A/B/C). The script refuses to run P2, P3 and P4 (undo included) when it finds `mjpython`, `robotd`, `body_server` or `tofd` processes.
2. Make an APFS clone backup (done in seconds, takes almost no space, copy-on-write). The backup goes in the home folder rather than in Documents, so that iCloud sync does not upload several GB:

   ```zsh
   cp -cR "/Users/ethan/Documents/RK3588 Microduck" "$HOME/RK3588 Microduck_backup_20261011"
   ```

   Expected: the command prints nothing; `ls ~` shows the backup folder. Delete this backup yourself after the reorganisation has been accepted.

## P1 check the plan, install the new files

```zsh
python3 docs/reorg/reorg.py plan
```

Expected: four lines `P1: ok …`, `P2: ok …`, `P3: ok …`, `P4: ok …`, and no line starting with `ERROR`.
The full plan is written to `docs/reorg/move_plan.tsv`; open it in a spreadsheet to see where each file goes.

```zsh
python3 docs/reorg/reorg.py apply P1
```

Expected: about 50 caches deleted, `docs/reorg/snapshot_before.tsv` generated (the SHA-256 of every file before the reorganisation),
the new files in `docs/reorg/staged/` installed in their proper places (`src/`, `scripts/`, `configs/`, `tests/`, `env.sh`, README and so on), and finally `staged/` gone.

## P2 documents, evidence, reference material

```zsh
python3 docs/reorg/reorg.py apply P2
```

Expected: `evidence/`, `docs/guides/`, `docs/analyses/`, `docs/plans/`, `references/` and `scripts/learning/` appear;
`output/`, `outputs/`, `reports/`, `research_notes/`, `learning/`, `task1/`, `tmp/` and `参考资料/` are gone.

## P3 upstream repositories, policy models, rebuilding the environment

```zsh
python3 docs/reorg/reorg.py apply P3
```

Expected: `task/` is gone; `third_party/microduck`, `third_party/microduck_rl`, `models/policies` and `models/checkpoints/walk_qktrrkg5` appear;
the old Python environment has been moved to `third_party/microduck_rl/.venv.pre_reorg` (it has the old paths hard-coded and is only for falling back).

Confirm that the two upstream repositories came across unchanged:

```zsh
git -C third_party/microduck rev-parse --short HEAD
git -C third_party/microduck_rl rev-parse --short HEAD
git -C third_party/microduck status --short
git -C third_party/microduck_rl status --short
```

Expected: the first two print `43d0967` and `cb70b79`; the third prints nothing; the fourth prints nothing, or just one line `?? .venv.pre_reorg/`.

Rebuild two things (the paths changed, so the old build and environment cannot be used as they are):

```zsh
(cd third_party/microduck_rl && uv sync --locked)
third_party/microduck_rl/.venv/bin/python -c "import mujoco, onnxruntime, numpy, mjlab_microduck; print('env ok', onnxruntime.__version__, numpy.__version__)"
zsh scripts/sim/build_mac_runtime.zsh
```

Expected: the second line prints `env ok 1.24.4 2.4.1`; the third ends by printing `编译成功` (build succeeded), with the full log in `runs/<time>_build_mac_runtime/build.log`.

## P4 archive the active code, smoke test

```zsh
python3 docs/reorg/reorg.py apply P4
```

Expected: `work/` is gone; `models/yolo/yolo11n_coco/` (5 files), `offline/` and `archive/2026-10_ball_follow/work/` appear.

1. Unit tests:

   ```zsh
   zsh scripts/tools/run_tests.zsh
   ```

   Expected: `Ran 97 tests` and a last line `OK` (if the Mac has no `sha256sum` command it shows `OK (skipped=4)`, which also passes).
   `test_camera_stream` and `test_yolo_camera_probe` are GPT's original timing tests and fail once in a while when the computer is busy; run again when only these two fail.

2. Board smoke test (the same standing plus camera detection as on 10-08). Open three terminals and `cd` to the project root in each:

   ```zsh
   # terminal A: sim body + head camera + the yellow ball scene
   zsh scripts/sim/run_sim_body.zsh --scene sim/scenes/legacy_ball/ball_1m_yellow/scene_ball_1m.xml
   # terminal B: once A shows that it is listening, start robotd on the board and forward the control interface
   zsh scripts/board/run_board_robotd.zsh
   # terminal C: once B shows robotd's run log, stand for 10 seconds, then run 30 seconds of CPU detection on the board
   zsh scripts/board/probe_board_stand_camera.zsh
   # terminal C: one more run with NPU detection (still standing; no need to restart A and B)
   zsh scripts/board/probe_board_yolo_camera.zsh --npu
   ```

   Expected: the standing check passes (as on 10-08: 10 seconds standing after the policy takes over); both the CPU and the NPU run detect `sports ball` in most frames
   (the baseline of the evening of 10-08: detected in all 450 frames, confidence about 0.83 to 0.89). The evidence of each run is in `runs/<time>_board_stand/`,
   `runs/<time>_board_yolo_cpu/` and `runs/<time>_board_yolo_npu/`, where `board/` holds the results copied back from the board.

3. The new code no longer has hard-coded old paths:

   ```zsh
   grep -rIn --exclude-dir=__pycache__ "/Users/ethan" src scripts/sim scripts/board scripts/vision scripts/tools scripts/lib.zsh configs sim tests env.sh
   ```

   Expected: no output.

## P5 check and manifests

```zsh
python3 docs/reorg/reorg.py check
python3 docs/reorg/reorg.py manifest
```

Expected: `check` prints `check passed` (every file from before the reorganisation is in its new place with unchanged content). It may be followed by a paragraph `note: … symlinks … point nowhere now`,
listing links that worked before and now point at the old location; they should all be in `archive/` (the `assets` links of old scenes that point into `task/`), which is normal. If a listed link is not in `archive/`, stop and send it to me.
`manifest` writes one `MANIFEST.tsv` (path, size, SHA-256) each in `evidence/`, `offline/`, `references/`, `models/`, `data/`, `runs/`
and the two big archived folders.

## P6 set up git (the commit is yours to make)

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

Expected: both `submodule add` lines show `Adding existing repo at 'third_party/…' to the index` (the existing repository is used, nothing is downloaded again);
the first check prints nothing (the big folders commit only their README and MANIFEST); the second lists only the `.onnx` and `.rknn` files in `models/yolo/yolo11n_coco/` (about 10 MB and 7 MB);
if it lists any other file over 1 MB, stop and send it to me; the last prints a few hundred. After confirming:

```zsh
git commit -m "Reorganise project into repository layout"
du -sh .git
```

Expected: `.git` is under 50 MB.

Then create a new private repository `microduck-rk3588-toy-retrieval` on the GitHub website (do not tick README, .gitignore or License), and:

```zsh
git remote add origin https://github.com/<your user name>/microduck-rk3588-toy-retrieval.git
git push -u origin main
```

## When something goes wrong

- `python3 docs/reorg/reorg.py status` shows how far each phase has got.
- A phase stopped part-way (for example, a destination already exists): deal with it as the message says and `apply` the same phase again; actions already done are not repeated.
- Undo: `python3 docs/reorg/reorg.py undo P4`, then `undo P3`, and so on; it has to go backwards from the last phase. Before undoing, the script checks everything first,
  lists any obstacle and changes nothing if there is one. The two common ones:
  - After `uv sync` has run, undoing P3 says the new `.venv` is in the way: first `mv third_party/microduck_rl/.venv third_party/microduck_rl/.venv.rebuilt`, then undo; the old `.venv` then returns to its place.
  - The smoke test left run records in `runs/`, and undoing P1 asks for them to be moved out of `runs/` first.
- `third_party/microduck_rl/.venv.pre_reorg` takes 1.3 GB: delete it (`rm -rf third_party/microduck_rl/.venv.pre_reorg`) after the P6 push has succeeded and the smoke test has passed. Once it is deleted, P3 can no longer be undone.
- When the board smoke test fails while the reorganisation itself is fine: the old launchers are all still in `archive/2026-10_ball_follow/work/`, but they point at the old paths and cannot be run directly; send the output to Claude.
- `~/microduck-board` on the board is not touched this time; the board is tidied according to plan (Q9) after the smoke test has passed.
