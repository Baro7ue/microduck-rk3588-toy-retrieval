# ~/microduck on the RK3588 (not a git checkout)

Board side of the Mac repository `microduck-rk3588-toy-retrieval`. Installed from
`configs/board/README_board.md` by `docs/reorg/board_reorg.zsh` (step B7); the Mac launchers in
`scripts/board/` drive everything here over SSH, using the paths in the Mac `env.sh`.

| Path | What it is | Changes when |
|---|---|---|
| `runtime/bin/robotd` (+ `robotd.sha256`) | robotd built on this board from microduck@43d0967, copied out of `build/` | robotd is rebuilt and copied here on purpose |
| `runtime/onnxruntime-linux-aarch64-1.28.0/` | ONNX Runtime that robotd dlopens (`ORT_DYLIB_PATH`) | ORT upgrade |
| `runtime/venv/` | the only board Python (3.11): RKNN Toolkit Lite2, ONNX Runtime, OpenCV, NumPy; pinned by `runtime/venv_requirements_lock.txt` | lock file changes; rebuild, do not edit in place |
| `bundles/<run>/` | code, models, robotd config and policies sent by one Mac launcher run, checked with `SHA256SUMS` | every run (new folder) |
| `run/<run>/` | that run's output (robotd log, detections, summaries); copied back to the Mac `runs/<run>/board/` | every run (new folder) |
| `run/robotd.sock` | robotd control socket, forwarded to the Mac by `run_board_robotd.zsh` | while robotd runs |
| `build/microduck-g3-offline/` | robotd source, vendored crates and build output (offline) | only when rebuilding robotd |
| `toolchains/rust-1.89.0/` | Rust toolchain used for that build | never (offline install) |
| `downloads/` | installers and their checksums, kept only for reinstalling | never |
| `archive/2026-10_ball_follow/` | everything from before 2026-10-12, as it was (old `microduck-board/`, old venv, NPU smoke test, baseline) | never |
| `move_log.tsv` | every move and copy of the reorganisation (step, operation, from, to, UTC) | only by `board_reorg.zsh` |

Rules: nothing here is edited by hand. Experiment code, configs and policies come from the Mac repo in
`bundles/`; results go to `run/` and are copied back. Paths written inside `archive/` are the old ones
(`/home/linaro/microduck-board/...`); `move_log.tsv` maps them to where they are now.
