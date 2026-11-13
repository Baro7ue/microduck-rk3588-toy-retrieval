# Tidying the board: steps to run (B0 to B7)

Plan: `claude-docs/board_cleanup_20261012/板端整理方案_20261012.md` (option A chosen for all 4 questions).
Every command runs from the repository root on the Mac:

```zsh
cd "/Users/ethan/Documents/RK3588 Microduck"
```

`docs/reorg/board_reorg.zsh` works on the board over SSH, and every run leaves a record (`console.log`) in `runs/<UTC>_board_reorg_<command>/`.
Only one step may run on the board at a time. Pressing Ctrl+C in the middle of `apply`/`undo` does not necessarily stop that step on the board at once; run `status` first to see where things stand, then decide whether to `undo` or start over.
Every move and copy on the board is recorded in `~/microduck/move_log.tsv`, and `undo` takes them back in reverse order. If any step reports an error, stop and send me the output.

The tool prints its messages in Chinese; the lines quoted below are what it prints, with the meaning in brackets.

## Commit the tidying tool first

```zsh
zsh scripts/tools/run_tests.zsh
git add docs/reorg/board_reorg.zsh docs/reorg/board_reorg_remote.sh docs/reorg/board_steps.md \
  docs/reorg/board_switch.patch configs/board tests/test_board_reorg.py
git commit -m "Add board home reorganisation tool"
```

Expected: every test OK. `test_board_reorg` needs bash 4 and GNU tools; with the Mac's own bash 3.2 it shows as skipped, which is normal (I ran it on Linux: all 13 pass).

## B0 read-only precheck

```zsh
zsh docs/reorg/board_reorg.zsh precheck
```

The result is saved in `runs/<UTC>_board_reorg_precheck/precheck.txt`. Tell me when it has run and I will look at: whether any process is running, whether `~/.profile` has a hard-coded Rust path, and whether the vendor paths are relative.

## B1 copy the board's evidence back to the Mac

```zsh
zsh docs/reorg/board_reorg.zsh backup
python3 docs/reorg/reorg.py manifest
git add evidence/MANIFEST.tsv && git commit -m "Back up board evidence before reorganising"
```

Expected: it prints `核对通过：N 个文件，两边 SHA-256 一致` (check passed: N files, SHA-256 equal on both sides), and the backup is in `evidence/board_home_<date>/`. Without that line, B2 refuses to run.

## B2 build the skeleton, move the install packages away

```zsh
zsh docs/reorg/board_reorg.zsh plan B2      # only lists what it would do; each line should end in [ok] or [已存在，跳过] (already there, skipped)
zsh docs/reorg/board_reorg.zsh apply B2
```

Acceptance: nothing the launchers use has been touched, so one NPU detection run on the present paths should give the usual result (first start the sim body with the yellow ball in another terminal: `zsh scripts/sim/run_sim_body.zsh --scene sim/scenes/legacy_ball/ball_1m_yellow/scene_ball_1m.xml`):

```zsh
zsh scripts/board/probe_board_yolo_camera.zsh --npu --duration 10
```

Expected last line: `实验退出码：0；证据回传退出码：0` (experiment exit code 0; evidence copy-back exit code 0). To take it back: `zsh docs/reorg/board_reorg.zsh undo B2`.

## B3 changes on the Mac side (already written; look, no need to run)

The changes are all in `docs/reorg/board_switch.patch` and are applied only in B5. To see which files change:

```zsh
git apply --stat docs/reorg/board_switch.patch
```

## B4 install the new runtime side by side

```zsh
zsh docs/reorg/board_reorg.zsh plan B4
zsh docs/reorg/board_reorg.zsh apply B4
```

Expected: pip installs 9 packages, and the last lines are
`rknn-toolkit-lite2 2.3.2 onnxruntime 1.28.0 numpy 1.26.4 opencv 4.11.0`, then `B4 完成` (B4 done).
The old environment is untouched. To take it back: `zsh docs/reorg/board_reorg.zsh undo B4` (deletes only the copies in `~/microduck/runtime/`).

## B5 switch over and smoke test

```zsh
git apply docs/reorg/board_switch.patch
zsh scripts/tools/run_tests.zsh
```

Then the smoke test in today's order, in three terminals:

```zsh
zsh scripts/sim/run_sim_body.zsh --scene sim/scenes/legacy_ball/ball_1m_yellow/scene_ball_1m.xml   # terminal 1: sim body with the yellow ball (the same scene as on the morning of 10-12)
zsh scripts/board/run_board_robotd.zsh                  # terminal 2: robotd on the board
zsh scripts/board/probe_board_stand_camera.zsh          # terminal 3: stand 10 s + CPU detection
zsh scripts/board/probe_board_yolo_camera.zsh --npu     # terminal 3: NPU detection
```

Acceptance (against the numbers of 10-12): stands 10 seconds without falling, at 50 Hz; CPU and NPU each detect in 450/450 frames; median inference within ±10% of 44.5 ms on the CPU and 15.4 ms on the NPU;
after stopping robotd with Ctrl+C in terminal 2, `runs/<UTC>_board_robotd/board/robotd.log` should have been copied back. Commit only after all of it passes:

```zsh
git add -A && git commit -m "Switch board paths to ~/microduck"
```

To take it back: before committing, `git apply -R docs/reorg/board_switch.patch`; after committing, `git revert HEAD`. The old paths stay intact until B6.

## B6 move the old folders away

```zsh
zsh docs/reorg/board_reorg.zsh plan B6
zsh docs/reorg/board_reorg.zsh apply B6
```

`apply B6` first checks that `env.sh` has switched to the new paths and refuses if it has not. Afterwards run the stand and the NPU detection once more (as in B5), and confirm that Rust still works:

```zsh
ssh RK3588 '~/microduck/toolchains/rust-1.89.0/bin/cargo -V'
```

If B0 showed a `rust-1.89.0` path written in `~/.profile`, I will give the one line to change. To take it back: `zsh docs/reorg/board_reorg.zsh undo B6`.

## B7 finishing

```zsh
zsh docs/reorg/board_reorg.zsh apply B7         # installs ~/microduck/README.md
zsh docs/reorg/board_reorg.zsh inventory        # stock-take after tidying
zsh docs/reorg/board_reorg.zsh status
```

Tell me when it has run; I will check the inventory and update the project record.

## Order of undoing

Undo only goes backwards: `undo B7` -> `undo B6` -> `undo B4` -> `undo B2`. The script refuses to skip a step.
An undo deletes only what the script itself copied, and moves every original file back where it was; the `__pycache__` that B6 deleted is a cache and is not restored.
