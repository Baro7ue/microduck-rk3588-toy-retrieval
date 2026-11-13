# MicroDuck: action guide for the next stage

From sustained walking on the Mac to camera acceptance and preparing the RK3588 deployment

4 October 2026 · follows the stage order of Claude's original plan · each step explains its purpose, the operation and the evidence

**What to do first now:** complete Stage 2's check that "the simulation stays real-time with the camera on". Then check the board preparation of Stages 0/1, and go on to Stage 3, CPU policy inference on the board. The heading PI, as you decided, is left to be connected to the current runtime chain before going on the board.

## 1. Where things stand

| Item | Verified state |
| --- | --- |
| Own trained policy | The local walk_17500.onnx is loaded and run by the native robotd; standing at zero speed uses velstand.onnx. |
| Sustained walking | The latest actual walk lasted 60.018 seconds without a fall and returned to standing at the end. Forward command 0.3 m/s. |
| Control loop | In the same test window: 50.002 Hz on average, 3001 loops, 1 loop whose work overran the period, 0.0333%. This was not measured on the RK3588. |
| Heading error | The user still sees circling; heading_compensation=false in this run. The outer PI of the earlier replay program is not connected to robotd's command chain. |
| Camera and real time | The pinned commit already has camera code, but starting the camera, grabbing frames and the real-time factor have not actually been accepted yet. |
| Board preparation | No acceptable G1 evidence so far; the existence and integrity of the system image and the flashing tool, and the state of the system on the board, each need checking. |

**Limits of the evidence:** the existing start-up assistance in the simulation permanently releases its hold after the policy first takes over. The assisted period does not count towards standing / walking results; this implementation note is kept. The 7.4 V and 32°C in the logs are simulation constants, not measurements of the board's supply or temperature.

## 2. The order from here

1. Measure the physics real-time factor without a camera, as the baseline of the current environment.
2. Restart the body with one head camera on, reconnect robotd and verify standing.
3. Check the actual image, frame format and arrival rate, then walk for 60 seconds with the camera on.
4. Put the G2 evidence in order; connect the PI before going on the board, keeping the results before and after compensation separately.
5. Check G0/G1: serial log, image integrity, system, SSH, NPU smoke test.
6. Enter G3: the CPU policy runs on the RK3588 and drives the simulated body on the Mac over the network cable.

No more PPO training is needed for now. YOLO will later use the paid Colab resources you already have; fine-tuning starts once the dataset is ready.

<!-- page -->

## 3. Step 1: the real-time factor without a camera

**What this step does:** reads the sim_time the body returns and compares how far the physics world's time advanced with the real time that passed on the Mac. robotd showing 50 Hz does not prove by itself that the physics world also runs in real time.

Keep the current body window and the robotd terminal running, with the duck standing at zero velocity. In a free terminal run:

```bash
zsh \
  "/Users/ethan/Documents/RK3588 Microduck/work/probe_runtime_visual_load.zsh" \
  --label baseline_no_camera
```

**What will happen:** the script sends only hello/read queries, no velocity, enable, disable or joint control commands. By default it samples 3 windows of about 10 seconds each, about 30 seconds in all; the result is saved automatically in a new folder under task/evidence/G2/.

**How to read the output:** real-time factor RTF = simulated time elapsed / real time elapsed. 1.0 means 10 simulated seconds take about 10 real seconds; 0.8 means the simulation advances only about 8 seconds in 10 real seconds. Keep the raw result of every window.

**Evidence to bring back:** the full terminal output, the evidence folder, summary.json. Send this step's result before doing the next step.

## 4. Step 2: turn on the head camera

**What this step does:** ends the old local experiment and starts one duck's body with one head camera. Camera and body use different TCP channels, so later vision will not push pictures into the control protocol.

After confirming that step 1's result is saved, run in a free terminal:

```bash
zsh \
  "/Users/ethan/Documents/RK3588 Microduck/work/restart_runtime_body_camera.zsh"
```

**What will happen:** the script first checks process identity and stops only this project's old body on 7801 and its robotd; existing logs are kept. It then opens a new MuJoCo window: the body listens on 127.0.0.1:7801, the camera on 127.0.0.1:7901.

**Sign of success:** the terminal shows camera on 127.0.0.1:7901 and the body stays at HOME waiting for the control program. At this point only the body and the camera are ready; standing under the policy is not yet verified. Keep that window and terminal running.

**Evidence to bring back:** a screenshot of the new window, the full terminal output, the path of the log stamped runtime_body_camera_. If the port is taken or the process identity does not match, keep the error and do not start again.

<!-- page -->

## 5. Step 3: connect robotd to the new body

**What this step does:** loads the same own trained ONNX and the stand model, and connects the native control program to the body again. It continues the CPU inference configuration that already passed.

Run in another free terminal, keeping the camera body window running:

```bash
zsh \
  "/Users/ethan/Documents/RK3588 Microduck/work/run_runtime_robotd.zsh"
```

**Sign of success:** the log shows simulated body, policy loaded, control loop running. Motion control is not yet enabled at this step. Keep that terminal running.

## 6. Step 4: accept standing at zero velocity again

**What this step does:** enables the policy, waits for the start-up assistance to be released, then watches 10 seconds of standing that the policy really maintains. Verifying standing first avoids mixing camera load with walking problems.

Run in a third free terminal:

```bash
zsh \
  "/Users/ethan/Documents/RK3588 Microduck/work/probe_runtime_stand.zsh" \
  --keep-enabled
```

**Sign of success:** the result is completed_no_fall, stating that the trial passed and the robot is kept standing. The body log should show startup assistance ended. After success the robot stays standing at zero velocity, ready for grabbing frames in the next step; a failure / error / interruption requests the policy off and saves the evidence.

## 7. Step 5: grab camera images and measure the load while standing

**What this step does:** reads raw frames from the camera's TCP channel, saves the first and last frames turned upright, and measures the physics real-time factor again.

After standing has passed, run in a free terminal:

```bash
zsh \
  "/Users/ethan/Documents/RK3588 Microduck/work/probe_runtime_visual_load.zsh" \
  --camera --label camera_stand
```

**Check the result:** open camera_first_rotated.png and camera_last_rotated.png in the evidence folder. The default scene is mostly floor; this round accepts only the camera bridge and does not count "seeing the floor" as a PERSON/BALL detection result. Confirm that the images decode and are the right way up, and that the terminal reports the frame count, the arrival rate and 3 RTF windows.

The raw frame is 640×360 UYVY; turned 90° clockwise, the saved image is 360×640. There is no YOLO inference now, and no video stream to a browser.

<!-- page -->

## 8. Step 6: a 60-second walk with the camera on

**What this step does:** checks whether camera rendering, image transfer and native walking control can run steadily together. The walking command is the same as before, and the result without PI is still kept.

First start the walking test in terminal C:

```bash
zsh \
  "/Users/ethan/Documents/RK3588 Microduck/work/probe_runtime_walk.zsh" \
  --duration 60
```

When the output enters the walk phase, start the read-only camera / real-time sampling in another terminal D. It takes about 30 seconds and should finish before this 60-second walk ends:

```bash
zsh \
  "/Users/ethan/Documents/RK3588 Microduck/work/probe_runtime_visual_load.zsh" \
  --camera --label camera_walk
```

**Do not put the two commands one after the other in one terminal:** the second test would then only start after the walk has ended. Terminal C handles the velocity command and the walking record; terminal D only reads sensor and time data.

**Evidence to bring back:** the full output of both sides, both summary.json files, the camera PNGs, the body log and a screen recording of the 60-second walk. Confirm that there was no fall and that it returned to stand at the end; record the actual RTF, the control loop's mean frequency, and the increase in ticks and missed.

## 9. How to judge whether this round is done

| Check | Verdict and limits |
| --- | --- |
| Physics real time | As in the original plan, at least real-time, ideally an RTF of about 1.0. The body advances in 20 ms batches, so limited sampling may fluctuate around 1.0; keep the raw increments and do not quietly change the standard to 0.95. If it stays below 1.0, troubleshoot first. |
| Walking stability | 60 seconds of continuous walking without a fall after the policy takes over, returning to standing after the stop. The current circling stays on record; holding a straight line does not count as done. |
| Control loop | Report the mean Hz and new missed / new ticks. missed counts only loops whose work overran the period, not every skipped schedule; it cannot stand alone for all missed deadlines. |
| Camera | The images decode and turn upright properly; report the arrival frame rate. The protocol has no source timestamp or frame number, so this round cannot claim to have measured end-to-end frame age or the unique rendered frame rate. |

If real time falls short, keep the failure record first, then retry headless with the same procedure: add --headless to the camera restart command of step 2. That closes the third-person window and keeps the head camera; then connect, stand and run the load tests again.

<!-- page -->

## 10. After the camera check: preparing to go on the board

Stages 1 and 2 of Claude's plan were meant to run in parallel. The Mac branch being done does not mean the board branch has passed. Check the existing evidence first; tasks that already passed are not redone wholesale.

### A. Complete the board evidence of Stages 0/1

**Purpose:** to confirm that later compiling, network communication and NPU deployment stand on a reliable system.

1. Find the serial boot log of task card #1 and the output of uname -a and cat /etc/os-release. Without this evidence, first complete the read-only serial record following the existing step guide for task card 1 (Mac serial boot log).
2. Take stock of the system image and the flashing tool, checking the real file paths, sizes, checksums and the vendor's notes; a file existing does not mean it downloaded completely. A .downloading fragment cannot serve as a flashable image. This guide has not confirmed that the current Drive / local image is complete.
3. After the serial record and the manual have been checked, handle the Linux 6.1 Debian 12 installation, turning off automatic sleep, SSH and the direct network as in the original plan. The exact flashing steps follow the vendor's manual for the board model and come as a separate task card.
4. Save the kernel / system version, CPU / memory / storage, network latency, board temperature, and the versions of the NPU driver and librknnrt.
5. Complete the NPU smoke test with a ready-made RKNN model. It belongs to G1; the own trained PPO walking policy is still deployed first to CPU ONNX Runtime.

**Evidence that completes G1:** Linux 6.1, SSH over the direct network, the detection result and timing of a ready-made NPU model, and the driver / runtime library versions. Adjustments to the image, flashing and software versions are checked separately and not carried out automatically in this guide.

### B. Connect the heading PI before going on the board

**Purpose:** to give the accumulated heading error a feedback correction while keeping the version of the PPO network itself.

The original PI sits in the Mac replay wrapper and changes the turn command before inference. The same feedback logic later has to be connected to the velocity command chain actually sent to robotd and verified with its sensor feedback; a TCP connection will not carry this logic over by itself.

As you decided, this is handled before going on the board. First keep the current baseline without compensation, then compare heading, stability and control loop timing before and after compensation at the same forward speed. Manual turning and stopping must release the heading lock and clear the integral; the turning of future visual control and the heading hold must be given clearly separate control authority.

**The earlier PI parameters are not taken directly as the final parameters on the board.** First complete the camera and real-time evidence, then start this task card.

<!-- page -->

## 11. The next stage: G3, the brain goes on the board

**Goal:** the RK3588's CPU runs robotd and the own trained ONNX and drives the MuJoCo body on the Mac over the direct network. The control loop stays on the board; the physics world and image rendering stay on the Mac.

| Order | Purpose and acceptance |
| --- | --- |
| 1. Align the files | Pin the upstream commits, transfer walk_17500.onnx, velstand.onnx and the config; compare SHA-256. The model takes 61 inputs and gives 14 outputs; use the exported file with normalisation baked in. |
| 2. Build on the board | Install the build tools the board's system needs and build robotd, robotctl and tofd with the lock file. Keep the full output; do not change dependencies or work round errors yet. |
| 3. Set up the CPU inference library | Use the Linux aarch64 ONNX Runtime shared library and check the Rust ort version and that it actually loads; the Mac's dylib cannot be copied to the board. |
| 4. Bring up the cable | Use a dedicated direct network and have the Mac body listen on an address the board can reach; check connectivity first. The current 127.0.0.1 is reachable only on the Mac itself and must be reconfigured for the board. |
| 5. Start on the board | Run robotd --sim and tofd --sim, each pointing at the Mac; verify standing at zero velocity first, then walk for 60 seconds. The exact commands are generated once the IP, system and library paths are confirmed. |
| 6. Sustained timing | Run for 10 minutes and record the control loop, latency, CPU / memory and the real board temperature; check the below-1% target with the proposal's definition of a missed tick. Keep records of failures and interruptions. |

**G3 passes when:** inference on the board's CPU drives the duck to walk for 60 seconds without falling, with control timing meeting the original plan's targets. This conclusion must come from records taken on the board.

## 12. When YOLO starts

Once G3 works, Stage 4 first does the narrowest visual closed loop with a pretrained model, then NPU deployment, the dataset and fine-tuning. You already have paid Colab, which later training will use; no more HF PPO jobs need submitting now.

Before training, prepare the PERSON/BALL assets and labels and two datasets, with fixed and with random appearance, and split training / validation / test by episode, seed, texture and asset. As in the original plan, complete the experiment of 2 appearances × 3 seeds, 320×320, at most 60 epochs; choose the checkpoint and the thresholds only on the validation set. A separate Colab operation card is generated once the data is ready.

## 13. Versions, scripts and index of material

Training repository: microduck_r1, commit cb70b792312d559a4da09064d92009079671815f; this is the folder name of the local microduck_rl. Runtime repository: microduck, commit 43d09674221da03e1a8f8f6f5cf66c21a9978b2c. Neither had source changes at this check.

The new camera start-up and read-only measurement scripts are all in the project's work/; neither repository above was modified and no new cloud job was started. So far only the source check and the syntax / probe checks are done; the real acceptance of the camera on this Mac is left for you to carry out step by step.

Main sources: Stages 0-4 of Claude's step-by-step implementation plan "RK3588 × MicroDuck"; body_server.py, camera.py, simulation.md and the duck-sim realtime implementation at the locally pinned commits; the latest G2 walking summary.json. The collaboration protocol in Claude's document is content of that material; the order of operations and the pause points of this guide follow what you asked for directly: "one step at a time, explain the purpose, tune the PI just before going on the board".

Current evidence folder: task/evidence/G2/runtime_walk_20261004_201611_099005/. Every later script prints its own evidence path; send me the full output and the path, and I will check this step first and then give the next one.
