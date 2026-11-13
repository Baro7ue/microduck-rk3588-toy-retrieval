# How RoboCup humanoid robots search for the ball, remember it and approach it

Research date: 2026-10-10. Only official material and source code were read; in this round the MicroDuck product was not changed, and no board, closed-loop simulation or tuning was run.

## Versions and limits of the evidence

- B-Human: the 2025 annual release of the official BHumanCodeRelease, pinned commit [`d89d603`](https://github.com/bhuman/BHumanCodeRelease/commit/d89d603f0388208cecd1c3144b6e8eb92aef4a93), commit date 2025-11-06. An annual release cannot be equated with the team's internal competition version at this moment. [Official repository notes](https://github.com/bhuman/BHumanCodeRelease)
- Bit-Bots: a snapshot of main of the official bitbots_main, pinned commit [`a9b6276`](https://github.com/bit-bots/bitbots_main/commit/a9b62760085e77f36e541ea00eacab71d5a4775a), commit date 2026-10-08. This is a verifiable public development version; whether all its branches were used in some match was not checked. [Official repository](https://github.com/bit-bots/bitbots_main)
- All the source below was fetched on 2026-10-10. The full list of sources (repository, pinned commit, file path, link to the original) is in `robocup_sources.json` in this folder; the original copies are not in the public repository (they belong to B-Human and Bit-Bots, each under its own licence), so read the originals through the `url` field of the list. Parameters are the defaults of the code read or the values of that config, not targets that can be copied to MicroDuck.

## Key question 1: after seeing the ball once, what does a mature robot remember?

### Takeaway

They keep a ball state with a frame and a confidence, which later vision keeps correcting. On a missed detection the estimate can stay, but that does not mean walking on along the old command without limit. B-Human models the ball's motion explicitly; this Bit-Bots filter mainly keeps a static position and grows its uncertainty, and the two cannot be lumped together as the same kind of Kalman ball tracker.

### Cited Findings

- **B-Human's representation**: `BallModel` holds the position relative to the robot (mm), the velocity (mm/s), the position covariance, the last observation, and `timeWhenLastSeen` and `timeWhenDisappeared` (it should have been seen but was not). The estimated state and the latest visual measurement are stored separately. [BallModel.h](https://github.com/bhuman/BHumanCodeRelease/blob/d89d603f0388208cecd1c3144b6e8eb92aef4a93/Src/Representations/Modeling/BallModel.h)
- **B-Human's prediction**: every frame it applies the inverse odometry transform to bring the resting-ball and rolling-ball hypotheses into the current robot frame; the rolling hypothesis also advances the position and handles deceleration. When vision sees the ball again it updates the hypotheses and selects a result; the covariance also propagates with motion and noise. After 4 seconds a new observation resets the old filter, which cannot be read as permission to walk blind for 4 seconds. [BallStateEstimator.cpp](https://github.com/bhuman/BHumanCodeRelease/blob/d89d603f0388208cecd1c3144b6e8eb92aef4a93/Src/Modules/Modeling/BallStateEstimator/BallStateEstimator.cpp)
- **Bit-Bots' representation**: the input is already a 3-D `BallArray`, transformed by timestamped TF into the filter frame such as `odom`, and position and covariance are kept. The current implementation has no ball-velocity state; without a new measurement it keeps the position in that frame and raises the uncertainty on a timer. After matching an observation it updates the position and the measurement covariance directly, so the module name alone does not justify saying it uses a full motion Kalman model. [ball_filter.py](https://github.com/bit-bots/bitbots_main/blob/a9b62760085e77f36e541ea00eacab71d5a4775a/src/bitbots_world_model/bitbots_ball_filter/bitbots_ball_filter/ball_filter.py)
- **How Bit-Bots consumes it**: the blackboard keeps the ball's field position, obtains the forward / left coordinates in `base_footprint` through the current body transform, and provides a metric distance and an `atan2` bearing. Getting a position back does not guarantee it is still credible; the behaviour layer has to check validity as well. [world_model_capsule.py](https://github.com/bit-bots/bitbots_main/blob/a9b62760085e77f36e541ea00eacab71d5a4775a/src/bitbots_behavior/bitbots_blackboard/bitbots_blackboard/capsules/world_model_capsule.py)

### Inferences

For MicroDuck the minimum form worth borrowing is `position / bearing + frame + observation time + uncertainty / validity`. If only a resting ball is considered for now, keeping one short-term static ball position hypothesis already has a clearer meaning than "keep walking along the last pixel error"; ball velocity, multi-robot communication or whole-field localisation need not be copied yet.

### Gaps

This round did not audit the whole of Bit-Bots' upstream implementation that produces 3-D ball measurements from images, nor verify the error, units and time alignment of MicroDuck's current odometry. Without those checks, reliable metric navigation cannot be claimed.

## Key question 2: how long can prediction go on after the ball is lost, and when to slow, stop or search again?

### Takeaway

There is no constant common to both teams of the form "keep walking for N seconds after losing the ball". Three things have to be looked at separately: whether the estimator keeps the state, whether the behaviour layer accepts it, and whether the search motion allows the body to move. Mature implementations also turn, back off or go to a search position when the ball is not seen, but these are conditional recovery actions, not a repeat of the last approach command.

### Cited Findings

| Implementation | Trigger conditions verified | Actions and limits verified |
|---|---|---|
| B-Human near-ball recovery | Handling the ball, ball distance <700 mm; should-see-but-not-seen >500 ms; occlusion by obstacles excluded | Sweeps the head and backs off a little; backs off for at most 1 s, or for at least 0.5 s with more than 100 mm of backward odometry, then turns towards the last ball direction. Leaves as soon as the ball is seen again. Applies to this near-ball branch only. |
| Bit-Bots ball confidence | Every diagonal term of the own ball covariance < the configured 2.0, or a valid team-mate ball | `BallSeen` is re-evaluated every time; without a credible ball it enters a role-dependent search flow. This decision uses no fixed number of seconds since the last observation. |

The two rows come from [HandleStrikerLostBall.cpp](https://github.com/bhuman/BHumanCodeRelease/blob/d89d603f0388208cecd1c3144b6e8eb92aef4a93/Src/Modules/BehaviorControl/SkillBehaviorControl/Options/HandleStrikerLostBall.cpp); and from [ball_seen.py](https://github.com/bit-bots/bitbots_main/blob/a9b62760085e77f36e541ea00eacab71d5a4775a/src/bitbots_behavior/bitbots_body_behavior/bitbots_body_behavior/behavior_dsd/decisions/ball_seen.py), [blackboard](https://github.com/bit-bots/bitbots_main/blob/a9b62760085e77f36e541ea00eacab71d5a4775a/src/bitbots_behavior/bitbots_blackboard/bitbots_blackboard/capsules/world_model_capsule.py) and [body_behavior.yaml](https://github.com/bit-bots/bitbots_main/blob/a9b62760085e77f36e541ea00eacab71d5a4775a/src/bitbots_behavior/bitbots_body_behavior/config/body_behavior.yaml).

- B-Human first checks the projection of the physical ball and occlusion by its own body before deciding that the ball should be visible in the current view. The default checkable distance is 1 m, and a disappearance is recorded after 7 should-see-but-not-seen counts; `700 ms` is how long an old observation may be used for computing velocity, not a general permission to move. [BallLocatorTools.h](https://github.com/bhuman/BHumanCodeRelease/blob/d89d603f0388208cecd1c3144b6e8eb92aef4a93/Src/Tools/Modeling/BallLocatorTools.h), [estimator parameters](https://github.com/bhuman/BHumanCodeRelease/blob/d89d603f0388208cecd1c3144b6e8eb92aef4a93/Src/Modules/Modeling/BallStateEstimator/BallStateEstimator.h)
- Bit-Bots' default filter rate is 62 Hz, and each timer step adds 0.002 to the planar covariance; if the predicted ball is in view but no detection matches, the uncertainty is also multiplied by 1.5. Since the initial confidence, the rate of negative observations and team-mate balls differ, no fixed safe blind-walking time follows from this. [Filter config](https://github.com/bit-bots/bitbots_main/blob/a9b62760085e77f36e541ea00eacab71d5a4775a/src/bitbots_world_model/bitbots_ball_filter/config/ball_filter_parameters.yaml), [filter_step and negative observation handling](https://github.com/bit-bots/bitbots_main/blob/a9b62760085e77f36e541ea00eacab71d5a4775a/src/bitbots_world_model/bitbots_ball_filter/bitbots_ball_filter/ball_filter.py)
- Bit-Bots' `SearchBall` branch first walks on the spot for 1 second, then turns towards the side of the last ball for 6 seconds, then goes to a search point on the field, turns for 15 seconds, changes position, and finally stands; other roles may take other branches. The 6 / 15 seconds are durations of search motions, not a tolerance for walking blind towards the ball. The turning action itself sends only a rotation speed. [main.dsd](https://github.com/bit-bots/bitbots_main/blob/a9b62760085e77f36e541ea00eacab71d5a4775a/src/bitbots_behavior/bitbots_body_behavior/bitbots_body_behavior/behavior_dsd/main.dsd), [turn.py](https://github.com/bit-bots/bitbots_main/blob/a9b62760085e77f36e541ea00eacab71d5a4775a/src/bitbots_behavior/bitbots_body_behavior/bitbots_body_behavior/behavior_dsd/actions/turn.py)
- B-Human's wider search uses field particles / a grid and chooses the next area to observe from the views already covered and the occlusions. A search request points at a field position, which reflects actively changing the viewing direction and position. [BallSearchParticlesProvider.cpp](https://github.com/bhuman/BHumanCodeRelease/blob/d89d603f0388208cecd1c3144b6e8eb92aef4a93/Src/Modules/BehaviorControl/BallSearchParticlesProvider/BallSearchParticlesProvider.cpp), [BallSearch.h](https://github.com/bhuman/BHumanCodeRelease/blob/d89d603f0388208cecd1c3144b6e8eb92aef4a93/Src/Modules/BehaviorControl/StrategyBehaviorControl/BallSearch.h)
- Bit-Bots' navigation controller sets the translation speed to the remaining path distance times a gain, with a speed limit and smoothing; this is slowing with distance when nearing the goal, not a verified "slowing with time since the ball was lost". Cancelling the plan does not stop the walk automatically either; the interface has a separate `stop_walk`. [controller.py](https://github.com/bit-bots/bitbots_main/blob/a9b62760085e77f36e541ea00eacab71d5a4775a/src/bitbots_navigation/bitbots_path_planning/bitbots_path_planning/controller.py), [pathfinding_capsule.py](https://github.com/bit-bots/bitbots_main/blob/a9b62760085e77f36e541ea00eacab71d5a4775a/src/bitbots_behavior/bitbots_blackboard/bitbots_blackboard/capsules/pathfinding_capsule.py)

### Inferences

MicroDuck is advised to keep three things apart: update the estimate during a short miss; leave the approach when the uncertainty goes out of bounds; turn the head or the body actively when searching again. While evidence of the odometry's reliability is missing, keeping the existing zero-velocity protection on a lost ball and adding a controlled re-search is easier to verify than simply lengthening the tolerance for missed detections. This is a suggestion for the project, not yet implemented or tested.

### Gaps

This round did not trace a single threshold at which every B-Human role enters whole-field search, nor did it find a "covariance -> walking speed" formula shared by the two teams. The 500 ms above cannot be generalised to all B-Human behaviours; a comment in Bit-Bots' config calls the covariance a standard deviation, and the quantity the code actually compares should be taken as authoritative.

## Key question 3: how does a 2-D tracker differ from a metric ball state, and how do approach and head work together?

### Takeaway

A 2-D tracker answers "does this frame's box continue the previous frame's target"; a metric state answers "where is the ball relative to the robot, and where should it be after the robot moves". The former can improve association and short-term continuity of detection; with only pixel boxes, areas and frame numbers, a stopping condition of 20 cm from the front of the foot to the ball cannot be defined.

### Cited Findings

- B-Human's approach skill builds a `kickPose` from the ball's predicted end point and a kick offset, uses path planning far away and switches to local walking round the ball close by; head control looks at the ball when about 250 mm from the target and goes back to other observation beyond 310 mm, with hysteresis. These distances belong to the skill's target and do not equal MicroDuck's clearance in front of the foot. [GoToBallAndKick.cpp](https://github.com/bhuman/BHumanCodeRelease/blob/d89d603f0388208cecd1c3144b6e8eb92aef4a93/Src/Modules/BehaviorControl/SkillBehaviorControl/Skills/Ball/GoToBallAndKick.cpp), [GoToBallHeadControl.cpp](https://github.com/bhuman/BHumanCodeRelease/blob/d89d603f0388208cecd1c3144b6e8eb92aef4a93/Src/Modules/BehaviorControl/SkillBehaviorControl/Skills/Head/GoToBallHeadControl.cpp)
- Bit-Bots' `GoToBall` keeps computing and publishing a goal pose; the near-ball goal is the ball's position relative to the current body minus a given approach distance and side offset. The 0.2 m in the config is its navigation / kick reference distance and cannot be equated with our 20 cm from toe to ball surface. [go_to_ball.py](https://github.com/bit-bots/bitbots_main/blob/a9b62760085e77f36e541ea00eacab71d5a4775a/src/bitbots_behavior/bitbots_body_behavior/bitbots_body_behavior/behavior_dsd/actions/go_to_ball.py), [goal geometry](https://github.com/bit-bots/bitbots_main/blob/a9b62760085e77f36e541ea00eacab71d5a4775a/src/bitbots_behavior/bitbots_blackboard/bitbots_blackboard/capsules/pathfinding_capsule.py), [approach distance config](https://github.com/bit-bots/bitbots_main/blob/a9b62760085e77f36e541ea00eacab71d5a4775a/src/bitbots_behavior/bitbots_body_behavior/config/body_behavior.yaml)
- Bit-Bots' head mode `TRACK_BALL` transforms the metric ball point by TF and computes joint targets; the search modes generate smooth pan / tilt scan trajectories. The body behaviour selects the head mode explicitly, dividing the work between head scanning and body navigation. [move_head.cpp](https://github.com/bit-bots/bitbots_main/blob/a9b62760085e77f36e541ea00eacab71d5a4775a/src/bitbots_motion/bitbots_head_mover/src/move_head.cpp), [head_modes.py](https://github.com/bit-bots/bitbots_main/blob/a9b62760085e77f36e541ea00eacab71d5a4775a/src/bitbots_behavior/bitbots_body_behavior/bitbots_body_behavior/behavior_dsd/actions/head_modes.py)

### Inferences

| Layer | The question it should answer for MicroDuck | Can a 2-D box alone do it? |
|---|---|---|
| Detection / association | Is this the ball, and is it still the same target? | It can carry most of this, though false and missed detections remain |
| Geometry and state estimation | How many metres ahead and to the left of the body is the ball? How credible is the estimate? | Needs camera calibration, pose / extrinsics, a depth constraint and timing as well |
| Motion prediction | After I walk or turn, how should a resting ball change relative to me? | Extrapolating pixel boxes alone cannot reliably tell ball motion from camera motion; own-motion information is needed |
| Behaviour / control | Keep approaching, slow down, stand still, or search again actively? | Should use state validity and a metric target; the tracker cannot decide this alone |

The suggested research route is to define the reference frame of the ball position and the geometry of the "clearance in front of the foot" first, then verify static distance and own-motion compensation, and only then discuss the estimator and the search state machine. What is worth borrowing from RoboCup is the duties of these layers; the specific thresholds, the NAO / Bit-Bots ball sizes, field coordinates and gait speeds cannot be carried over.

### Gaps

These public implementations show design mechanisms; they do not prove that they would work directly with MicroDuck's camera, gait and hardware. This research ran no new closed-loop verification of MicroDuck's current failed trials, and did not evaluate how YOLO trackers, SLAM or navigation packages actually perform.
