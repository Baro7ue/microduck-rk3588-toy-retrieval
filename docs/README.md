# docs/

- `plans/`: project plans and plans sent for review
- `guides/`: step-by-step operating guides for Stages 0 and 1. The earlier guides for
  board gates G3 and G4 (2026-10-03 to 10-10) were removed from `main` on 2026-11-01; they are at tag `stage4-results`.
- `analyses/`: training and drift analyses, notebooks and figures
- `research/`: background research notes (sources in `references/research_ball_navigation/`)
- `reorg/`: the 2026-10-11 reorganisation tool, its README and the move log
- `media/`: the Stage 5 room demo video and its preview GIF, shown in the top-level README
- `zh/`: the Chinese originals of the documents above (中文原文), under the same paths and file names

## Language

The documents here were written in Chinese and translated to English on 2026-11-12; `zh/` keeps every
original unchanged. Still in Chinese, as generated or as programs print them:

- the labels inside the figures in `analyses/figures/` and the notebook code that draws them;
- the messages the tools in `reorg/` print (`reorg.py`, `board_reorg.zsh`, `board_reorg_remote.sh`), the
  notes in `reorg/move_log.tsv` and the text inside `reorg/board_switch.patch`, which has to match those files;
- old file and folder names, quoted as they were (also the `local` paths in `research/ball_navigation/robocup_sources.json`).

## Renamed on 2026-10-11

Documents were renamed to ASCII `YYYY-MM-DD_topic` names. Links and paths written inside them still use
the old names; this table (and `reorg/move_log.tsv`) maps them.

| New path | Old path |
|---|---|
| `analyses/2026-10-04_heading_hold_notes.md` | `outputs/MicroDuck航向保持说明_20261004.md` |
| `analyses/2026-10-04_left_drift_circling_diagnosis.md` | `outputs/MicroDuck左偏绕圈诊断_20261004.md` |
| `analyses/2026-10-04_next_stage_action_guide.md` | `output/pdf/MicroDuck_下一阶段行动指南_20261004.md` |
| `analyses/2026-10-04_training_plateau_analysis.ipynb` | `outputs/MicroDuck训练平台期分析_20261004.ipynb` |
| `analyses/2026-10-04_training_plateau_analysis.json` | `outputs/MicroDuck训练平台期分析_20261004.json` |
| `analyses/2026-10-04_training_plateau_analysis.md` | `outputs/MicroDuck训练平台期分析_20261004.md` |
| `analyses/2026-10-04_wandb_training_duration_check.ipynb` | `outputs/WandB训练时长核对_20261004.ipynb` |
| `analyses/2026-10-04_wandb_training_duration_check.json` | `outputs/WandB训练时长核对_20261004.json` |
| `analyses/figures/2026-10-04_heading_hold_5min_comparison.png` | `outputs/MicroDuck航向保持_5分钟对照_20261004.png` |
| `analyses/figures/2026-10-04_heading_hold_comparison.png` | `outputs/MicroDuck航向保持对照_20261004.png` |
| `analyses/figures/2026-10-04_heading_hold_comparison.svg` | `outputs/MicroDuck航向保持对照_20261004.svg` |
| `analyses/figures/2026-10-04_training_curves.png` | `outputs/MicroDuck训练曲线分析_20261004.png` |
| `analyses/figures/2026-10-04_training_curves.svg` | `outputs/MicroDuck训练曲线分析_20261004.svg` |
| `plans/2026-10-10_ball_search_navigation_plan.md` | `reports/MicroDuck 寻球 导航方案.md` |
| `plans/2026-10-11_toy_retrieval_plan_for_review.md` | `reports/MicroDuck_玩具回收总体方案_Claude评估版_2026-10-11.md` |
| `research/ball_navigation` | `research_notes/MicroDuck 寻球 导航方案` |
