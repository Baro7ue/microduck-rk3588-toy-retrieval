from pathlib import Path
import json
import hashlib
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

BASE = Path('/Users/ethan/Documents/RK3588 Microduck')
SOURCE = BASE / 'task/evidence/wandb-history-qktrrkg5-20261004-clean.csv'
OUT = BASE / 'outputs'
OUT.mkdir(exist_ok=True)
with SOURCE.open() as f:
    reader = csv.DictReader(f)
    columns = reader.fieldnames
    records = list(reader)
d = {key: np.array([float(row[key]) for row in records]) for key in columns}
assert len(records) == 17680 and len(np.unique(d['_step'])) == 17680
assert set(d['_step']) == set(range(17680))
assert all(np.isfinite(values).all() for values in d.values())

windows = [('2000–2499', 2000, 2500), ('17000–17499', 17000, 17500)]
metrics = [
    ('Train/mean_reward', '平均回报', 'higher'),
    ('Train/mean_episode_length', '平均 episode 长度（控制步）', 'higher'),
    ('Metrics/twist/error_vel_xy', '平面速度误差日志值', 'lower'),
    ('Metrics/twist/error_vel_yaw', '偏航速度误差日志值', 'lower'),
    ('Episode_Metrics/mean_action_acc', '动作二阶差分绝对值', 'lower'),
    ('Episode_Termination/fell_over', '摔倒终止计数均值（非摔倒率）', 'lower'),
]
segments = {
    name: {key: values[(d['_step'] >= lo) & (d['_step'] < hi)] for key, values in d.items()}
    for name, lo, hi in windows
}
comparison = []
for key, label, direction in metrics:
    early, late = [float(segments[name][key].mean()) for name, *_ in windows]
    comparison.append({
        'metric': key, 'label': label, 'early_window_mean': early,
        'late_window_mean': late, 'percent_change': (late / early - 1) * 100,
        'preferred_direction': direction,
    })
result = {
    'run_id': 'qktrrkg5', 'observed_steps': len(records),
    'last_step': int(d['_step'].max()),
    'comparison_windows': [name for name, *_ in windows],
    'rows_per_window': [len(segments[name]['_step']) for name, *_ in windows],
    'comparison': comparison,
    'runtime_at_iteration_2000_seconds': float(d['_runtime'][d['_step'] == 2000][0]),
    'runtime_at_iteration_5000_seconds': float(d['_runtime'][d['_step'] == 5000][0]),
    'runtime_at_final_iteration_seconds': float(d['_runtime'][-1]),
    'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
}
(OUT / 'MicroDuck训练平台期分析_20261004.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))

font_path = Path('/System/Library/Fonts/STHeiti Medium.ttc')
if font_path.exists():
    font = FontProperties(fname=str(font_path))
    plt.rcParams['font.family'] = font.get_name()
plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=False)
colors = ['#137F8B', '#137F8B', '#B26918', '#B26918', '#6A59A1', '#A34B4B']
for ax, (key, label, direction), color in zip(axes.flat, metrics, colors):
    ax.plot(d['_step'][::10], d[key][::10], color=color, alpha=0.14, linewidth=0.7)
    smooth = np.convolve(d[key], np.ones(200), mode='same') / np.convolve(np.ones(len(records)), np.ones(200), mode='same')
    ax.plot(d['_step'], smooth, color=color, linewidth=1.8)
    ax.axvspan(0, 2000, color='#72818A', alpha=0.08)
    ax.axvline(2000, color='#667681', linestyle='--', linewidth=1)
    ax.set_title(label + (' ↑' if direction == 'higher' else ' ↓'), loc='left', fontsize=11, pad=10)
    ax.set_xlabel('PPO 更新轮数（W&B _step）')
    ax.set_xlim(0, 17679)
    ax.grid(axis='y', alpha=0.16)
    ax.tick_params(labelsize=9)
    if key == 'Train/mean_episode_length':
        ax.axhline(1000, color='#89969E', linewidth=0.7, linestyle=':')
        ax.set_ylim(0, 1040)
        ax.text(17100, 1007, '20 秒上限', fontsize=9, ha='right', color='#667681')
    if key == 'Episode_Termination/fell_over':
        inset = ax.inset_axes([0.35, 0.32, 0.61, 0.56])
        mask = d['_step'] >= 2000
        inset.plot(d['_step'][mask], smooth[mask], color=color, linewidth=1.4)
        inset.set_title('2000 轮以后放大', fontsize=9, loc='left')
        inset.set_xlim(2000, 17679)
        inset.set_ylim(0.18, 0.46)
        inset.set_xticks([2000, 10000, 17000])
        inset.set_yticks([0.2, 0.3, 0.4])
        inset.tick_params(labelsize=8)
        inset.grid(axis='y', alpha=0.15)
fig.suptitle('MicroDuck：2,000 轮后进入慢速改善阶段', fontsize=19, x=0.06, ha='left', y=0.973)
fig.text(0.06, 0.936, '运行 qktrrkg5 · 4096 个并行环境 · seed 42 · 实测记录 0–17679 轮', fontsize=11, color='#556570')
fig.text(0.06, 0.035,
         '实线：200 轮中心移动平均；淡线：每 10 轮采样原值。虚线：课程最后阶段约在第 2000 轮。\n'
         '速度误差为上游累计归一化日志值；动作 acc 指 acceleration；终止计数不能直接当作摔倒率。',
         fontsize=10, color='#556570', linespacing=1.6)
fig.subplots_adjust(left=0.07, right=0.97, top=0.89, bottom=0.105, hspace=0.44, wspace=0.23)
fig.savefig(OUT / 'MicroDuck训练曲线分析_20261004.png', dpi=170)
fig.savefig(OUT / 'MicroDuck训练曲线分析_20261004.svg')
plt.close(fig)

lines = ['# MicroDuck 训练平台期核对', '',
         '分析范围：W&B run qktrrkg5 的 0–17679 轮。读取公开 SDK 下的已授权账号数据；未启动训练。', '',
         '两组均为 500 轮平均：2000–2499 对比 17000–17499，降低单点噪声。', '',
         '| 指标 | 2000–2499 | 17000–17499 | 变化 |', '|---|---:|---:|---:|']
for row in comparison:
    lines.append(f"| {row['label']} | {row['early_window_mean']:.4f} | {row['late_window_mean']:.4f} | {row['percent_change']:+.2f}% |")
lines += ['', '主要观察：平面速度误差接近平台；偏航跟踪、平滑度、回报和终止计数仍缓慢改善。'
          '课程参数到约 2000 轮已到最终阶段，因此此前和此后的总回报不是完全相同难度下的直接比较。', '',
          '指标边界：mean_action_acc 是动作二阶差分，不是准确率；1000 控制步等于 20 秒 episode 上限。'
          '速度误差是上游按最大指令时间归一化的累计值，不直接等于逐步平均物理误差。'
          'fell_over 是日志平均计数，不是独立评测摔倒率。', '',
          '数据检查：第一次分页导出出现 17 个完全相同的重复记录与 17 个缺少的轮号；'
          '保留原文件，针对缺少轮号重读后另存清理版，得到 17680 个唯一连续轮号，32 列无缺失。', '',
          '建议：先对比 model_2000.pt、model_5000.pt、model_17500.pt 的确定性固定指令回放，'
          '使用相同种子、相同初始状态和相同指令序列，统计直行/转向跟踪、停止及 60 秒不倒。'
          '用验证场景选择 checkpoint，正式测试场景留到配置冻结后。当前没有曲线依据要求继续跑到 50000 轮。', '',
          '此次曲线分析不能代替 MuJoCo 回放或 RK3588 板上闭环验收。']
(OUT / 'MicroDuck训练平台期分析_20261004.md').write_text('\n'.join(lines))

# Preserve the exact calculations and generated preview in an editable notebook.
code = Path(__file__).read_text().split('# Preserve the exact calculations')[0]
nb = {'nbformat': 4, 'nbformat_minor': 5,
      'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}},
      'cells': [
          {'cell_type': 'markdown', 'id': 'context', 'metadata': {},
           'source': ['# MicroDuck 训练平台期核对\n', '本 notebook 读取已保存的 W&B 历史数据，不连接云端或发起训练。\n',
                      '比较 2000–2499 与 17000–17499 两个各 500 轮的窗口；训练指标不是独立验证成绩。']},
          {'cell_type': 'code', 'id': 'analysis', 'metadata': {}, 'execution_count': 1,
           'source': code.replace("code = Path(__file__).read_text()", "code = ''").splitlines(keepends=True),
           'outputs': [{'output_type': 'stream', 'name': 'stdout',
                        'text': [json.dumps(result, ensure_ascii=False, indent=2) + '\n']}]}]}
(OUT / 'MicroDuck训练平台期分析_20261004.ipynb').write_text(json.dumps(nb, ensure_ascii=False, indent=2))
print(json.dumps(result, ensure_ascii=False, indent=2))
