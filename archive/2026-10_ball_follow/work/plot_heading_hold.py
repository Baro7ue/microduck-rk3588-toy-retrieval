"""Render measured CPU-replay comparison; truth is evaluation-only."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

root = Path('/Users/ethan/Documents/RK3588 Microduck')
source = root / 'task/evidence/walk_drift_20261004'
font = FontProperties(fname='/System/Library/Fonts/STHeiti Medium.ttc')
plt.rcParams['axes.unicode_minus'] = False
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
records = {}
for label, name, color in [('heading_off_60s', '仅对齐执行器延迟', '#c87928'),
                           ('heading_pi_60s', '再加航向保持', '#177d86')]:
    rows = list(csv.DictReader((source / f'{label}.csv').open()))
    t = np.array([float(r['time_s']) for r in rows])
    x = np.array([float(r['x_m']) for r in rows])
    y = np.array([float(r['y_m']) for r in rows])
    heading = np.unwrap([float(r['heading_rad']) for r in rows])
    axes[0].plot(x, y, color=color, lw=2, label=name)
    axes[1].plot(t, np.degrees(heading-heading[0]), color=color, lw=1.8, label=name)
    records[label] = json.loads((source / f'{label}.json').read_text())
axes[0].axhline(0, color='#89939c', ls='--', lw=1)
axes[0].set_xlabel('前进方向 X（米，评估真值）', fontproperties=font)
axes[0].set_ylabel('横向 Y（米，评估真值）', fontproperties=font)
axes[0].set_title('实际行走轨迹', fontproperties=font, fontsize=13)
axes[1].axhline(0, color='#89939c', ls='--', lw=1)
axes[1].axvspan(0, 5, color='#e9eef1', alpha=.7)
axes[1].set_xlabel('仿真时间（秒）', fontproperties=font)
axes[1].set_ylabel('相对初始航向（度，评估真值）', fontproperties=font)
axes[1].set_title('朝向漂移', fontproperties=font, fontsize=13)
for ax in axes:
    ax.grid(alpha=.18)
    ax.spines[['top','right']].set_visible(False)
    ax.legend(prop=font, frameon=False)
fig.suptitle('MicroDuck：增加航向保持后的直行对照', fontproperties=font, fontsize=17)
fig.text(.07, .04, '同一初始状态：5秒启动 + 60秒前进；同一策略、20毫秒动作延迟。理想模拟IMU，单次CPU仿真。',
         fontproperties=font, fontsize=10, color='#536675')
fig.subplots_adjust(top=.80, bottom=.20, left=.07, right=.97, wspace=.25)
for extension in ('png','svg'):
    fig.savefig(root / f'outputs/MicroDuck航向保持对照_20261004.{extension}', dpi=170)

baseline, hold = records['heading_off_60s'], records['heading_pi_60s']
report = f'''# MicroDuck 直行航向保持（2026-10-04）

## 改动

独立包装原来的CPU回放，不修改microduck_rl仓库、依赖、ONNX或训练。保留20毫秒动作延迟。

外层控制器在直行时锁定当前相对航向；通过模拟陀螺仪和原策略已有的重力方向估计朝向，计算小幅转向修正，写入61维观测中的转向命令槽。用户命令与自动修正分开保存。控制器不读取世界位置或世界航向；这些真值只在诊断程序中离线评价。

已先验证P反馈，再加入小积分项以减小恒定偏差。当前参数：Kp=1.0、Ki=0.05、修正限幅±0.3 rad/s、积分限幅±0.8 rad·s。默认库配置仍为Ki=0，启动脚本显式选择本次验证过的Ki=0.05。不对14维策略动作做滤波或修改。

## 同条件对照

同一初始姿态，先给0.1 m/s指令5秒，再给0.3 m/s指令前进60秒，转向用户指令始终为零。策略：17,500轮；BAM M6；kp=200；vin=7.4 V；sag=0.1；20 ms控制步、5 ms物理步。每条件一次确定性CPU仿真；不是板上测试。

| 指标 | 仅20 ms延迟 | 再加航向保持 |
|---|---:|---:|
| 结束航向偏差绝对值 | {abs(baseline['heading_change_deg']):.2f}° | {abs(hold['heading_change_deg']):.2f}° |
| 结束横向偏移绝对值 | {abs(baseline['final_xy_m'][1]):.3f} m | {abs(hold['final_xy_m'][1]):.3f} m |
| 结束前进位置 | {baseline['final_xy_m'][0]:.2f} m | {hold['final_xy_m'][0]:.2f} m |
| 最大身体倾斜角 | {baseline['max_tilt_deg']:.2f}° | {hold['max_tilt_deg']:.2f}° |
| 是否达70°倾倒阈值 | 否 | 否 |

这些是结束值，不是全程最大误差；步态周期内仍有摆动。航向保持明显改善本工况，但不保证零位置偏差。后续必须按相同配置单独验证随机化场景与板上表现。

## 控制状态验收

已运行60秒的“直行→左转→停→直行→右转→停→直行”序列：手动转向与停止时纠偏退出，重启直行时重新锁定方向；两次转向实际方向正确，停段平均前进速度低于0.002 m/s，未摔倒。6项单元检查验证了反馈符号、倾斜坐标变换、手动优先、暂停和反向重锁、限幅与配置/传感异常。

若按T暂停，恢复时会重锁当前方向；不会尝试追回暂停前未观测到的转动。A/E是持续命令，恢复直行请先空格停止，再按↑。

## 复测

在终端运行：

```bash
zsh "/Users/ethan/Documents/RK3588 Microduck/work/run_straight_hold.zsh"
```

先等5秒，再在终端按↑，观察60秒；空格停止，Q退出。日志与CSV以时间戳保存到task/evidence/G2/。查看终端“[航向保持]”字段：用户转向为零时，实际发送的转向修正会随误差变化。

## 边界

当前使用理想的模拟陀螺仪，长时间真实陀螺仪零偏仍会导致相对航向漂移。航向控制也无法单独消除横向滑移；位置严格保持需要可靠的里程计或外部观测。该包装用于本地验证，不应未经测试直接接入视觉跟随：外层视觉转向与航向保持必须明确分配控制权。

原策略的正式基线应保留关闭航向保持的记录；新增控制器应在模型与实验清单中单独注明。
'''
(root / 'outputs/MicroDuck航向保持说明_20261004.md').write_text(report)
print('Saved comparison PNG/SVG and report.')
