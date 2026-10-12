# ball_1m_yellow (legacy G4 scene)

Yellow 70 mm / 15 g ball 1 m in front of the duck, used for the G4 ball-follow trials (2026-10-08 to 10-10).
Kept as the smoke-test scene after the 2026-10-11 reorganisation; the toy-retrieval scenes will live next to it.

- `scene_ball_1m.xml`: same as `archive/2026-10_ball_follow/work/g4-scenes/ball_1m_yellow/scene_ball_1m.xml`
  except the robot include, which is now relative to this folder
  (`../../../../third_party/microduck_rl/src/mjlab_microduck/robot/microduck/robot_groundcontact.xml`).
- `ball.xml`: unchanged copy (sha256 `42fc2052893212d33841599bdcf3ec7f24ef0671f9f376f8555c342d6dd55eec`).
- `assets`: relative symlink to `third_party/microduck_rl/src/mjlab_microduck/robot/microduck/assets`
  (MuJoCo looks for meshes next to the main scene file). Created by `docs/reorg/reorg.py apply P3`.
- The original `manifest.json` (colour choice evidence, source hashes) stays in the archive copy.

Run: `zsh scripts/sim/run_sim_body.zsh --scene sim/scenes/legacy_ball/ball_1m_yellow/scene_ball_1m.xml`
