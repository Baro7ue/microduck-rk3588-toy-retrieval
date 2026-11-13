# Provenance

Where the models, data and documents in this repository came from. First recorded on 2026-10-11,
when the folder was reorganised; brought up to date on 2026-11-01. "GPT" means OpenAI Codex/ChatGPT
sessions that Ethan directed; "Claude" means Anthropic Claude sessions in the project's planning space.
Ethan chose the direction and the option at every stage gate, ran every experiment on his Mac and the
RK3588 board, and trained the walking policy and the toy detector.

## Upstream (unchanged)

| Path | Source |
|---|---|
| `third_party/microduck` | [pollen-robotics/microduck](https://github.com/pollen-robotics/microduck) @ `43d0967`: Rust runtime (`robotd`, `robotctl`, `tofd`), Apache-2.0 |
| `third_party/microduck_rl` | [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl) @ `cb70b79`: MuJoCo sim body and RL training, Apache-2.0 |
| `models/policies/velstand.onnx`, `alpha_stand.onnx`, `alpha_ground_pick.onnx`, `manifest.json` | official MicroDuck policy set, Hugging Face `pollen-robotics/microduck-policies` rev `d5a8b55` |
| `models/yolo/yolo11n_coco/yolo11n_coco.pt` (local only) | Ultralytics YOLO11n COCO weights v8.3.0 (AGPL-3.0) |
| `sim/assets/toys/` | Google Scanned Objects via kevinzakka/mujoco_scanned_objects (CC BY 4.0), rescaled |

Licences of all third-party files: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Trained, exported or measured in this project

| Path | How |
|---|---|
| `models/checkpoints/walk_qktrrkg5/` (local only), `models/policies/walk_*.onnx` | walking policy trained by Ethan with microduck_rl on Hugging Face cloud, 2026-10-03 (run `qktrrkg5`) |
| `models/yolo/yolo11n_coco/*.onnx`, `*.rknn`, `*.json` | exported (ONNX FP32, 320) and converted (RKNN FP16, rknn-toolkit2 2.3.2) by GPT scripts, 2026-10-08/09 |
| `models/yolo/yolo11n_toys/` | fine-tuned by Ethan on Google Colab (60 epochs, 2026-10-25) with `scripts/vision/stage2/train_colab.ipynb` on the Stage 2 dataset; exported to ONNX and converted to RKNN FP16 and INT8 with the Stage 2 scripts |
| numbers in `configs/*/stage*_facts.json` | measured in Ethan's runs on the Mac and the board; some were first tried in a cloud copy of the setup, and only the Mac and board runs count |
| `results/stage4/` | computed by `scripts/stage4/tables.py` from Ethan's Stage 4 runs on the Mac and the board (2026-10-28 to 10-31) |
| `docs/media/stage5_room_demo_rk3588_int8.mp4`, `.gif` | Stage 5 room demo, one board run (YOLO INT8 on the NPU, layout seed 5000) of 2026-11-12 on branch `microduck-stage5-v4` (`799c2aa`). GPT rendered the video afterwards from that run's logs (an offline replay at real speed, not a screen recording); Claude re-encoded it with ffmpeg (H.264, CRF 27) and made the GIF preview from it (8x speed, 640 px) |

## Documents

| Path | Author |
|---|---|
| `docs/guides/` up to 2026-10-10 (removed from `main` on 2026-11-01; at tag `stage4-results`), `docs/analyses/` | GPT step guides and analyses, 2026-10-03 to 10-10 |
| `docs/guides/` from 2026-10-13 | Claude |
| `docs/plans/2026-10-11_toy_retrieval_plan_for_review.md` | Claude (review version of the toy-retrieval plan) |
| `docs/plans/2026-10-10_ball_search_navigation_plan.md`, `docs/research/` | not recorded in the files |
| `README.md`, `THIRD_PARTY_NOTICES.md`, this file | Claude, 2026-11-01; README demo section and `README.zh-CN.md` (the Chinese version), Claude, 2026-11-12 |
| English text of `docs/guides/`, `docs/plans/`, `docs/analyses/*.md`, `docs/research/`, `docs/reorg/README.md`, `docs/reorg/board_steps.md` | translated from the Chinese originals by Claude, 2026-11-12; the originals are kept unchanged in `docs/zh/` |
| `references/vendor_alientek/` (local only) | 正点原子 (ALIENTEK) board documentation pack |
| `references/ai_notes/` (local only) | Claude and GPT PDFs and notes (original names kept) |
