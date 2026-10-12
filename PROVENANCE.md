# Provenance

Who wrote what in this repository, as recorded on 2026-10-11 when the folder was reorganised.
"GPT" means OpenAI Codex/ChatGPT sessions that Ethan directed; "Claude" means Anthropic Claude
sessions in the project planning space. Ethan made the decisions, ran every hardware and sim
experiment, and reviewed the results.

## Upstream (unchanged, Apache-2.0)

| Path | Source |
|---|---|
| `third_party/microduck` | [pollen-robotics/microduck](https://github.com/pollen-robotics/microduck) @ `43d0967`: Rust runtime (`robotd`, `robotctl`, `tofd`) |
| `third_party/microduck_rl` | [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl) @ `cb70b79`: MuJoCo sim body and RL training |
| `models/policies/velstand.onnx`, `models/policies/manifest.json` | official MicroDuck policy set (see manifest) |
| `models/yolo/yolo11n_coco/yolo11n_coco.pt` | Ultralytics YOLO11n COCO weights v8.3.0 (AGPL-3.0) |

## Trained or exported in this project

| Path | How |
|---|---|
| `models/checkpoints/walk_qktrrkg5/`, `models/policies/walk_*.onnx` | walking policy trained by Ethan on Hugging Face cloud, 2026-10-03 (run `qktrrkg5`) |
| `models/yolo/yolo11n_coco/*.onnx`, `*.rknn`, `*.json` | exported (ONNX FP32, 320) and converted (RKNN FP16, rknn-toolkit2 2.3.2) by GPT scripts, 2026-10-08/09 |

## Project code

| Path | Author |
|---|---|
| `src/microduck_toy/` (except `paths.py`), `scripts/vision/`, `scripts/board/measure_board_resources.py`, `configs/control/` and their tests | GPT, 2026-10-04 to 10-10, in `work/`; moved on 2026-10-11 with only paths and imports changed (each file's first line names its original) |
| `src/microduck_toy/paths.py`, `scripts/**/*.zsh`, `scripts/lib.zsh`, `env.sh`, `configs/robotd/`, `tests/test_paths.py`, `tests/test_launchers.py`, `docs/reorg/` | Claude, 2026-10-11 (launchers rewritten from GPT's `work/*.zsh`, same commands and checks) |
| `scripts/learning/` | Claude, 2026-10-04 (learning exercises) |
| `sim/scenes/legacy_ball/` | GPT scene (2026-10-08), include path made relative by Claude on 2026-10-11 |
| `archive/2026-10_ball_follow/work/` | GPT, 2026-10-04 to 10-10, frozen as it was |

## Documents

| Path | Author |
|---|---|
| `docs/guides/`, `docs/analyses/` | GPT step guides and analyses, 2026-10-03 to 10-08 |
| `docs/plans/`, `docs/research/` | not recorded in the files; Ethan to confirm |
| `references/vendor_alientek/` | 正点原子 (ALIENTEK) board documentation pack |
| `references/ai_notes/` | Claude and GPT PDFs and notes (original names kept) |

Ethan's own code: none yet as of 2026-10-11; new code from Ethan goes in `src/` with his commits.
