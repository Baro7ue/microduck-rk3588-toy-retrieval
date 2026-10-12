# MicroDuck on RK3588: toy retrieval

MicroDuck is Pollen Robotics' small bipedal robot. This project runs its control stack on an
RK3588 board (正点原子 ATK-DLRK3588) against a MuJoCo "sim twin" of the robot running on a Mac:
the board runs `robotd` (50 Hz ONNX locomotion policy) and YOLO detection on the robot's head
camera stream; the Mac simulates the body, camera and ToF sensor over TCP.

- **Goal (due 2026-10-31):** find a named toy among distractors, walk to it, pick it up and drop it
  in a tray, with detection fine-tuned on the toys and deployed on the RK3588 NPU.
- **Done so far (2026-10-03 to 10-10):** own walking policy trained, robotd running on the board
  against the sim body, YOLO11n running on board CPU and NPU on the sim camera, and a ball-follow
  prototype. That prototype is archived in `archive/2026-10_ball_follow/`.

## Layout

| Path | What | In git |
|---|---|---|
| `src/microduck_toy/` | Python package: `vision/` (camera stream, YOLO, RKNN), `rpc/` (robotd client), `control/`, `sim/` | yes |
| `scripts/` | zsh launchers (`sim/`, `board/`, `tools/`) and one-off Python scripts | yes |
| `configs/` | robotd and controller parameters | yes |
| `sim/scenes/` | MuJoCo scenes passed to the sim body with `--scene` | yes |
| `tests/` | unit tests (`zsh scripts/tools/run_tests.zsh`) | yes |
| `docs/` | `plans/`, `guides/`, `analyses/`, `research/`, `reorg/` (folder reorganisation tool and log) | yes |
| `results/`, `report/` | final result tables and the course report | yes |
| `models/` | `policies/` (robotd ONNX), `yolo/<name>/` (ONNX, RKNN + metadata), `checkpoints/` | small final models only |
| `third_party/` | upstream `microduck` (Rust runtime) and `microduck_rl` (MuJoCo + training), git submodules | submodule |
| `evidence/` | logs and measurements from gates G1 to G4 and training | MANIFEST.tsv only |
| `runs/` | every launcher writes `runs/<UTC YYYYMMDDTHHMMSSZ>_<name>/` | README only |
| `offline/` | offline install bundles for the board (G3 runtime, NPU, vision export, flashing) | MANIFEST.tsv only |
| `references/` | vendor documentation and AI-generated reference notes (original file names) | MANIFEST.tsv only |
| `data/` | datasets (toy images, labels) | MANIFEST.tsv only |
| `archive/` | frozen earlier work, kept unchanged with its original paths | code yes, bundles no |

Local-only folders are listed file by file, with SHA-256, in their `MANIFEST.tsv`
(`python3 docs/reorg/reorg.py manifest`).

## Setup on the Mac

```zsh
git clone --recurse-submodules <this repo> "RK3588 Microduck"
cd "RK3588 Microduck"
(cd third_party/microduck_rl && uv sync --locked)   # Python 3.12 env: mujoco, onnxruntime, numpy
zsh scripts/sim/build_mac_runtime.zsh              # robotd, robotctl, tofd (needs Rust 1.89)
source env.sh                                       # optional in a shell; launchers source it themselves
```

`env.sh` derives every path from its own location, so the folder can be moved or renamed.
Board addresses and board-side paths are `MD_*` variables in `env.sh`; set them before running to override.

## Smoke test (sim body on the Mac, robotd and YOLO on the board)

```zsh
# terminal A: sim body + head camera, reachable from the board
zsh scripts/sim/run_sim_body.zsh --scene sim/scenes/legacy_ball/ball_1m_yellow/scene_ball_1m.xml
# terminal B: robotd on the board, control socket forwarded to the Mac
zsh scripts/board/run_board_robotd.zsh
# terminal C: 10 s stand check, then 30 s of board YOLO on the camera (add --npu for the NPU)
zsh scripts/board/probe_board_stand_camera.zsh
```

Mac only (no board): `run_sim_body.zsh --local`, then `zsh scripts/sim/run_mac_robotd.zsh`.

## Conventions

- File and folder names: ASCII, lowercase, `snake_case`. Dated documents: `YYYY-MM-DD_topic.md`.
  Run folders: `YYYYMMDDTHHMMSSZ_name` (UTC). `references/` keeps original file names.
- Upstream code is never edited in place; changes to upstream behaviour live in `src/` and are
  documented there.
- Evidence and run folders are append-only: old paths inside old logs are historical and stay as they are.
- Who wrote what: see [PROVENANCE.md](PROVENANCE.md).
