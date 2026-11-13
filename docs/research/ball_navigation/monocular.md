# MicroDuck: monocular ball localisation, visual servoing and navigating after a fix

Research date: 2026-10-10. Only papers, official documentation and project code were searched; no simulation or board experiment was run and no product was changed. The old `gait_response` diagnostic script has not been started.

## 1. With a known 70 mm ball, can "stop by area" be upgraded to a metric position?

### Takeaway

A metric ball observation with explicit assumptions can be built, but the area of a YOLO box cannot simply be called a distance. The minimum route is to calibrate the camera, recover the ball's direction, then estimate the position from the known diameter and the constraint that the ball is on the ground, and output validity and uncertainty. [OpenCV's official camera model](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html); [B-Human's plane projection implementation](https://raw.githubusercontent.com/bhuman/BHumanCodeRelease/master/Src/Tools/Math/Transformation.cpp)

### Cited Findings

- OpenCV's pinhole model describes the mapping from a 3-D point to a pixel with intrinsics, extrinsics and distortion; `undistortPoints` recovers undistorted normalised pixel coordinates. When the image is scaled the intrinsics must be scaled with it. [OpenCV official documentation](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html)
- B-Human's `imageToRobotHorizontalPlane` rotates the camera ray into the robot's frame and, with the camera translation, intersects it with a plane at a given height; the implementation checks the ray's vertical component. It does not judge distance from image area alone. [B-Human source, lines 61-84](https://raw.githubusercontent.com/bhuman/BHumanCodeRelease/master/Src/Tools/Math/Transformation.cpp)
- B-Human builds the camera transform from the torso transform, the robot's dimensions and the current joint angles; the official documentation warns that a small error in camera orientation strongly affects the ground projection of distant objects. [B-Human official perception documentation](https://docs.b-human.de/master/perception/perception-infrastructure/)

### Inferences

- **The known-diameter route**: from the pinhole model, far away, near the optical axis and with a whole outline, `Z≈f·D/d`, with `D=0.070 m` in this project. This approximates depth along the optical axis; it is not the slant range from camera to ball, nor the distance from the foot to the ball's surface. Near the image edge or at close range a sphere outline model should be used. A YOLO box can only serve as a proxy for the diameter with error; it cannot be assumed to fit the ball exactly. [Basis of the derivation: the OpenCV model](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html)
- **The on-the-ground route**: with the ground at `z=0`, the ball centre's height `R=0.035 m`, the camera centre `c` and the estimated direction to the ball centre `r`, then `p=c+λr`, `λ=(R-c_z)/r_z`. This needs the camera position and its relation to the ground; IMU attitude alone does not give the camera's height above the ground. When the ray is almost parallel to the ground, the intersection is behind the camera or the ball is in the air, the observation should be rejected. [Basis of the derivation: B-Human's plane intersection code](https://raw.githubusercontent.com/bhuman/BHumanCodeRelease/master/Src/Tools/Math/Transformation.cpp)
- **Cross-check the two routes first**: when the diameter depth and the ground depth clearly disagree, mark the observation invalid or down-weight it; do not average them unconditionally. A fixed area threshold can stand for a rough scale under ideal fixed imaging, but head motion, cut-off and box error break the assumption "same area = same distance in front of the feet". [Basis: OpenCV projection and B-Human camera geometry](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html); [camera geometry](https://docs.b-human.de/master/perception/perception-infrastructure/)

### Gaps

- The deviation of the existing boxes from the true ball outline, the intrinsics and extrinsics of a physical camera and the error in camera height while moving are not yet quantified, so centimetre accuracy cannot be promised. Geometry that can compute a position in theory does not mean this project has verified metric ranging. [Basis of the calibration need](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html)

## 2. Continuous visual servoing, or navigate by position after seeing the ball once?

### Takeaway

The suggestion is to keep continuous visual correction and add a short-term memory of the ball's state; freezing the target for good after one sighting is not recommended. Visual observations mainly update the ball's state; a ball that may move cannot be used as a fixed landmark to correct the robot's odom. [The original visual servoing tutorial](https://web.mit.edu/amcp/OldFiles/drg/Chaumette_Part_I.pdf)

### Cited Findings

- The tutorial by Chaumette and Hutchinson distinguishes IBVS, which forms its error from image features, from PBVS, which forms it from 3-D parameters estimated by vision. Its basic derivation is mainly for camera velocity control and cannot serve directly as a stability proof for a biped gait. [Original paper, Part I](https://web.mit.edu/amcp/OldFiles/drg/Chaumette_Part_I.pdf)
- The authors' introduction to hybrid visual servoing points out that image-based and pose-based control each have constraint problems in the other space, and that features leaving the field of view can cause failure; a hybrid scheme needs explicit switching conditions. [Hutchinson's research page](https://faculty.cc.gatech.edu/~seth/res.php?u=vs)
- B-Human's projection infrastructure matches joint angles to the moment the image was taken, not just to the latest joint angles when the image arrives. [B-Human official documentation, BodyContour](https://docs.b-human.de/master/perception/perception-infrastructure/)

### Inferences

| Scheme | When it applies | Limits in this project |
|---|---|---|
| Image centre / area feedback only | The ball stays visible and the task only asks to approach to some visual scale | The feedback depends on a detection in every frame; with the ball lost there is no memory of position, and area itself does not express the gap in front of the feet |
| Navigate after one fix | The target is static, the robot's displacement estimate is reliable, and the path and the end point can be verified | Ball movement, foot slip or yaw drift make the old target stale; one observation cannot remove later error |
| Metric observation + short-term prediction + re-observation | Brief misses are allowed and the prediction's uncertainty can be monitored | The better next step; when the ball stays out of sight for long, stop and re-acquire it; a prediction must not be treated as a new detection |

The table is an architectural judgement for this project, based on the classification of visual servoing and the field-of-view limit above; it is not an experimental conclusion about a finished controller. [Paper](https://web.mit.edu/amcp/OldFiles/drg/Chaumette_Part_I.pdf); [authors' notes](https://faculty.cc.gatech.edu/~seth/res.php?u=vs)

- Use odom to transform past ball observations into the current body frame, and **correct the ball estimate in the odom frame**; do not claim the reverse, that the ball observation corrected the robot's odom. Unless the ball is known to be static and a joint localisation model is built, the two must not be mixed. [Basis of the coordinate transform](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html)
- This project's TCP stream has no capture timestamp from the source, so the existing "pair the pose with the arrival time of the whole frame" only approximates the capture moment; the head swings, and the time offset also becomes an error in direction and distance. Adding a frame number / capture time and a clock mapping across machines should come first; network arrival time cannot be renamed capture time. [The project's source protocol and its limits](</Users/ethan/Documents/RK3588 Microduck/work/probe_runtime_visual_load.py>); [project reference for image / pose matching](https://docs.b-human.de/master/perception/perception-infrastructure/)

### Gaps

- There are no distributions yet of ball position error, odom drift and time offset under motion, turning and occlusion, so "how many seconds it may walk blind" or "it will stop within some centimetres" cannot be laid down. The bound on short-term prediction should come from data. [Basis of the error sources](https://docs.b-human.de/master/perception/perception-infrastructure/)

## 3. What is the least this project must add, and how to verify it layer by layer?

### Takeaway

First add a metric ball observation and state estimate, then decide on the approach control; the existing chain of RK3588 NPU detection, CPU gait and the MuJoCo body on the Mac can stay. The latest record is a controlled early stop after losing the ball; it must not be retold as a fall, nor may memory of position cover up the gait's turning response, which is not yet verified. [This project's latest record](</Users/ethan/Documents/RK3588 Microduck/task/evidence/G4/ball_follow_npu_20261010_035720/summary.json>)

### Cited Findings

- The current record is `stopped_missing_ball`, `stop_confirmed=true`, `left_enabled_standing=true`; the entry notes also state that there is no heading PI. The record proves a controlled stop in this run, not successful navigation. [Project record](</Users/ethan/Documents/RK3588 Microduck/task/evidence/G4/ball_follow_npu_20261010_035720/summary.json>)
- The existing ball centre / radius observation, the coordinate transform and the closed loop on visual error are layers that can be separated; B-Human describes ball detection and camera geometry separately, and the visual servoing tutorial describes visual features and control error separately. [B-Human ball detection](https://docs.b-human.de/master/perception/ball-and-penalty-mark-detection/); [camera geometry](https://docs.b-human.de/master/perception/perception-infrastructure/); [visual servoing paper](https://web.mit.edu/amcp/OldFiles/drg/Chaumette_Part_I.pdf)

### Inferences

The following are suggested new modules, not yet implemented or verified, based on the model and control layering above. [Geometric basis](https://raw.githubusercontent.com/bhuman/BHumanCodeRelease/master/Src/Tools/Math/Transformation.cpp); [control basis](https://web.mit.edu/amcp/OldFiles/drg/Chaumette_Part_I.pdf)

1. **BallMetricObservation**: takes the ball outline / box in the original image, the camera calibration, the paired pose and the relation to the ground; outputs the ball centre in the body or odom frame, distance / bearing, time, quality flags and uncertainty. Keeps the original detection label and the state "whole / cut off / occluded".
2. **BallState**: keeps the latest valid ball state; propagates it with odom's relative motion and updates it with new ball observations; distinguishes observed / predicted / lost explicitly. Ball motion and the robot's own motion are modelled separately; when the prediction's uncertainty grows, stop advancing and re-acquire the target.
3. **ApproachSupervisor**: manages searching, aligning, approaching, brief occlusion, re-acquisition and completion; the end point becomes an explicit geometric gap to the body / in front of the feet, keeping freshness, a controlled stop and posture safety conditions. An estimate of the target position does not guarantee that the gait achieves the turning needed, so the actual response should be checked separately.

Suggested order of verification, in each case collecting evidence first and choosing thresholds afterwards, with no preset seconds or accuracy guarantees: [calibration and projection basis](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html); [basis of the field-of-view limit](https://faculty.cc.gatech.edu/~seth/res.php?u=vs)

- **Static position offline**: cover different known ball positions, head yaw / pitch, and whole / edge / cut-off detections; measure the diameter method, the ground method and their agreement separately. MuJoCo truth is used only for scoring and must not enter the deployed algorithm.
- **Static occlusion**: with the robot still, hide the ball and show it again, or move it; verify that a prediction is not logged as a new observation, that a wrong association can be rejected, and that a stale state leads to a stop.
- **Motion and turning**: verify the actual body displacement / heading response and the estimation error for walking straight, turning left and right, and switching between stop and go, each separately; then connect the approach supervisor. Showing only that "a positive yaw was sent" does not show that "the body turned left".
- **Occlusion and re-acquisition while moving**: collect prediction error, uncertainty, re-acquisition association, the final gap in front of the feet and safe stops; until these results exist, walking on indefinitely on an old target is not approved.

### Gaps

- Calibration of the monocular metric observation, the state estimate implementation, verification of the motion response and occlusion navigation experiments are all not done. This note gives the minimum direction that the research supports and is no claim of success in hardware or in a closed loop. [Record of the project's current limits](</Users/ethan/Documents/RK3588 Microduck/task/evidence/G4/ball_follow_npu_20261010_035720/summary.json>)
