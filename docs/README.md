# docs/

- `plans/`: project plans and plans sent for review
- `guides/`: step-by-step operating guides for each gate (G3, G4), written in Chinese
- `analyses/`: training and drift analyses, notebooks and figures
- `research/`: background research notes (sources in `references/research_ball_navigation/`)
- `reorg/`: the 2026-10-11 reorganisation tool, its README and the move log

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
| `guides/2026-10-03_task_card1_mac_serial_boot_log.md` | `outputs/任务卡1_Mac串口启动日志_逐步教程.md` |
| `guides/2026-10-07_g3_10min_run_resource_sampling.md` | `output/guides/G3_10分钟持续运行与资源采样_20261007.md` |
| `guides/2026-10-07_g3_cpu_policy_inference_timing.md` | `output/guides/G3_CPU策略推理耗时测量_20261007.md` |
| `guides/2026-10-07_g3_disconnect_recovery_60s_walk.md` | `output/guides/G3_断联恢复与60秒行走_20261007.md` |
| `guides/2026-10-07_g3_feedback_sampling_fix_reacceptance.md` | `output/guides/G3_反馈采样修复与重新验收_20261007.md` |
| `guides/2026-10-07_g4_board_receives_sim_camera.md` | `output/guides/G4_板端接收仿真相机画面_20261007.md` |
| `guides/2026-10-08_g4_3d_ball_camera_positives.md` | `output/guides/G4_三维球相机正样本_20261008.md` |
| `guides/2026-10-08_g4_camera_continuous_cpu_detection.md` | `output/guides/G4_相机连续CPU检测_20261008.md` |
| `guides/2026-10-08_g4_floor_reflection_control_board_recheck.md` | `output/guides/G4_地面反射对照与板端复核_20261008.md` |
| `guides/2026-10-08_g4_speed_comparison.md` | `output/guides/G4_速度对照_20261008.md` |
| `guides/2026-10-08_g4_stand_policy_with_camera_detection.md` | `output/guides/G4_站立策略与相机检测联合检查_20261008.md` |
| `guides/2026-10-08_g4_unified_launch_and_recording.md` | `output/guides/G4_统一启动与录屏_20261008.md` |
| `guides/2026-10-08_g4_visual_approach_and_stop.md` | `output/guides/G4_视觉接近与停止_20261008.md` |
| `guides/2026-10-08_g4_yolo11n_board_cpu_single_image.md` | `output/guides/G4_YOLO11n板端CPU单图检测_20261008.md` |
| `guides/2026-10-09_g4_fixed_area_stop.md` | `output/guides/G4_固定面积停车_20261009.md` |
| `guides/2026-10-09_g4_one_cell_distance_stop.md` | `output/guides/G4_一格距离停车_20261009.md` |
| `guides/2026-10-10_g4_steering_reference_fix.md` | `output/guides/G4_转向参考修正与验收_20261010.md` |
| `plans/2026-10-10_ball_search_navigation_plan.md` | `reports/MicroDuck 寻球 导航方案.md` |
| `plans/2026-10-11_toy_retrieval_plan_for_review.md` | `reports/MicroDuck_玩具回收总体方案_Claude评估版_2026-10-11.md` |
| `research/ball_navigation` | `research_notes/MicroDuck 寻球 导航方案` |
