# Third-party material

The project's own code and documents are under the Apache License 2.0 ([LICENSE](LICENSE)).
The files below come from other people and keep their own terms.

| Path | What | Source | Licence |
|---|---|---|---|
| `third_party/microduck` (git submodule, not copied) | MicroDuck runtime: `robotd`, `robotctl`, `tofd` (Rust) | [pollen-robotics/microduck](https://github.com/pollen-robotics/microduck) @ `43d0967` | Apache-2.0 |
| `third_party/microduck_rl` (git submodule, not copied) | MuJoCo model, sim body and RL training code | [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl) @ `cb70b79` | Apache-2.0; its 3D model files are CC BY-NC-SA (see its README) |
| `models/policies/velstand.onnx`, `alpha_stand.onnx`, `alpha_ground_pick.onnx`, `manifest.json` | official MicroDuck policies (stand and get-up, legacy stand, ground pick) | Pollen Robotics, Hugging Face `pollen-robotics/microduck-policies` rev `d5a8b55`; trained with microduck_rl | as published by Pollen Robotics |
| `models/policies/walk_5000.onnx`, `walk_17500.onnx` | walking policy trained in this project with microduck_rl | this project | Apache-2.0 |
| `models/yolo/yolo11n_coco/*` | YOLO11n COCO weights exported to ONNX and RKNN | [Ultralytics](https://github.com/ultralytics/ultralytics) YOLO11n, v8.3.0 | AGPL-3.0 |
| `models/yolo/yolo11n_toys/*` | YOLO11n fine-tuned on this project's toy dataset, exported to ONNX and RKNN | derived from Ultralytics YOLO11n | AGPL-3.0 |
| `sim/assets/toys/*` | five scanned toys (mesh and texture), rescaled; `FOOD_BEVERAGE_SET__2` is one piece split out of `FOOD_BEVERAGE_SET` | Google Scanned Objects (Downs et al., ICRA 2022), MuJoCo form from [kevinzakka/mujoco_scanned_objects](https://github.com/kevinzakka/mujoco_scanned_objects) @ `6ff8d27` | CC BY 4.0 |

Read but not copied:

- `docs/research/ball_navigation/` summarises code from [B-Human](https://github.com/bhuman/BHumanCodeRelease)
  and [Bit-Bots](https://github.com/bit-bots/bitbots_main); `robocup_sources.json` lists every file with its
  commit and link. The source files themselves are not in this repository.
- The grasp score in `configs/control/stage1.json` follows the idea in
  [jonathanhawkins/microduck-lab](https://github.com/jonathanhawkins/microduck-lab) (Apache-2.0,
  `world/arena.py`); no code was copied.
- Board vendor documents (ALIENTEK ATK-DLRK3588) are used locally and not included; `references/MANIFEST.tsv`
  only lists their names.
