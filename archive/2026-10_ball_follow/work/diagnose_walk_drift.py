"""Measure fixed-command drift using the pinned upstream CPU replay code.

This diagnostic imports the upstream PolicyInference and BAM controller without
changing their code. It runs the same 50 Hz / four-substep loop without a viewer.
"""
import argparse
import contextlib
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import time

import mujoco
import numpy as np


ROOT = Path('/Users/ethan/Documents/RK3588 Microduck')
REPO = ROOT / 'task/microduck_r1'
parser = argparse.ArgumentParser()
parser.add_argument('--scene', default='src/mjlab_microduck/robot/microduck/scene.xml')
parser.add_argument('--label', required=True)
parser.add_argument('--vx', type=float, default=0.1)
parser.add_argument('--yaw', type=float, default=0.0)
parser.add_argument('--policy', default='walk_17500.onnx')
parser.add_argument('--prelude-seconds', type=float, default=0.0)
parser.add_argument('--seconds', type=float, default=30.0)
parser.add_argument('--delay', type=int, default=0)
parser.add_argument('--heading-hold', action='store_true')
parser.add_argument('--heading-kp', type=float, default=1.0)
parser.add_argument('--heading-ki', type=float, default=0.0)
parser.add_argument('--schedule', type=Path, default=None)
args = parser.parse_args()
schedule = [] if args.schedule is None else json.loads(args.schedule.read_text())
os.chdir(REPO)
spec = importlib.util.spec_from_file_location('upstream_infer', REPO / 'scripts/infer_policy.py')
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)
if args.heading_hold:
    from heading_hold import HeadingConfig, make_heading_policy
    replay.PolicyInference = make_heading_policy(
        replay.PolicyInference,
        HeadingConfig(kp=args.heading_kp, ki=args.heading_ki), log_status=False)
out = ROOT / 'task/evidence/walk_drift_20261004'
out.mkdir(parents=True, exist_ok=True)
rows = []
started = time.monotonic()
with (out / f'{args.label}.full.log').open('w') as log, contextlib.redirect_stdout(log):
    bam_model = replay.load_bam_model(replay.BAM_KP_FW, 7.4, 0.0)
    model, data, bam_ctrl, _ = replay.load_mujoco_with_bam(
        args.scene, bam_model, 0.005, 0.1, replay.BAM_VIN_MIN)
    policy = replay.PolicyInference(
        model, data, walking_onnx_path=str(ROOT / 'task/policies' / args.policy),
        action_scale=1.0, bam_ctrl=bam_ctrl,
        delay_min_lag=args.delay, delay_max_lag=args.delay,
        use_projected_gravity=True, new_cmd_obs=True)
    policy.set_vel_cmd(0.1 if args.prelude_seconds else args.vx, 0.0,
                       0.0 if args.prelude_seconds else args.yaw)
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'trunk_base_freejoint')
    qpos_adr = model.jnt_qposadr[joint_id]
    qvel_adr = model.jnt_dofadr[joint_id]
    data.qpos[qpos_adr:qpos_adr+3] = [0.0, 0.0, 0.125]
    data.qpos[qpos_adr+3:qpos_adr+7] = [1.0, 0.0, 0.0, 0.0]
    for i, idx in enumerate(policy.joint_qpos_indices):
        data.qpos[idx] = policy.default_pose[i]
    bam_ctrl.reset(data.qpos)
    policy.set_position_targets(policy.default_pose)
    mujoco.mj_forward(model, data)
    assert policy.get_observations().shape == (61,)
    expected_initial = [0.1, 0, 0] if args.prelude_seconds else [args.vx, 0, args.yaw]
    assert np.array_equal(policy.command[:3], np.array(expected_initial, dtype=np.float32))
    prelude_step = int(round(args.prelude_seconds / 0.02))
    schedule_index = 0
    for step in range(int(round(args.seconds / 0.02))):
        if args.prelude_seconds and step == prelude_step:
            policy.set_vel_cmd(args.vx, 0.0, args.yaw)
        while schedule_index < len(schedule) and step*0.02 >= schedule[schedule_index]['time_s']:
            policy.set_vel_cmd(*schedule[schedule_index]['command'])
            schedule_index += 1
        action = policy.infer()
        policy.apply_action(action)
        for _ in range(4):
            bam_ctrl.update()
            mujoco.mj_step(model, data)
        quat = data.qpos[qpos_adr+3:qpos_adr+7].astype(np.float32)
        v_world = data.qvel[qvel_adr:qvel_adr+3].astype(np.float32)
        v_body = policy.quat_rotate_inverse(quat, v_world)
        rot = data.xmat[policy.trunk_base_id].reshape(3, 3)
        heading = math.atan2(rot[1, 0], rot[0, 0])
        tilt = math.degrees(math.acos(float(np.clip(rot[2, 2], -1.0, 1.0))))
        rows.append(dict(time_s=(step+1)*0.02,
            x_m=float(data.qpos[qpos_adr]), y_m=float(data.qpos[qpos_adr+1]),
            z_m=float(data.qpos[qpos_adr+2]), heading_rad=heading,
            tilt_deg=tilt, vx_body_m_s=float(v_body[0]), vy_body_m_s=float(v_body[1]),
            wz_body_rad_s=float(data.qvel[qvel_adr+5]),
            cmd_vx_m_s=float(policy.vel_cmd[0]), cmd_vy_m_s=0.0,
            cmd_wz_rad_s=float(policy.vel_cmd[2])))
        rows[-1]['effective_cmd_wz_rad_s'] = float(policy.command[2])
        if args.heading_hold:
            rows[-1]['estimated_relative_heading_rad'] = policy.heading_hold.heading
            rows[-1]['estimated_heading_error_rad'] = policy.heading_hold.error
            rows[-1]['heading_hold_active'] = policy.heading_hold.active
        if not np.isfinite(data.qpos).all() or tilt >= 70.0:
            break
with (out / f'{args.label}.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
tail = [r for r in rows if r['time_s'] > args.prelude_seconds + 2.0]
angles = np.unwrap([r['heading_rad'] for r in rows])
summary = dict(label=args.label, scene=args.scene, command=[args.vx, 0.0, args.yaw],
    policy=args.policy, upstream_commit='cb70b792312d559a4da09064d92009079671815f',
    prelude_seconds_at_vx01=args.prelude_seconds,
    averaging_start_s=args.prelude_seconds + 2.0,
    initial_state='upstream default, no random perturbation',
    duration_s=rows[-1]['time_s'],
    mean_vx_body_after_2s_m_s=float(np.mean([r['vx_body_m_s'] for r in tail])),
    mean_vy_body_after_2s_m_s=float(np.mean([r['vy_body_m_s'] for r in tail])),
    mean_wz_body_after_2s_rad_s=float(np.mean([r['wz_body_rad_s'] for r in tail])),
    heading_change_deg=float(np.degrees(angles[-1] - angles[0])),
    max_tilt_deg=max(r['tilt_deg'] for r in rows),
    final_xy_m=[rows[-1]['x_m'], rows[-1]['y_m']],
    tilt_70deg_reached=any(r['tilt_deg'] >= 70.0 for r in rows),
    actuator='upstream BAM M6, vin=7.4, sag=0.1, kp=200, no current limit',
    action_delay_control_steps=args.delay,
    heading_hold=args.heading_hold,
    heading_hold_kp=args.heading_kp if args.heading_hold else None,
    heading_hold_ki=args.heading_ki if args.heading_hold else None,
    command_schedule=schedule,
    wall_seconds=time.monotonic()-started,
    scope='single deterministic CPU replay; no training environment or board validation')
(out / f'{args.label}.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n')
print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
