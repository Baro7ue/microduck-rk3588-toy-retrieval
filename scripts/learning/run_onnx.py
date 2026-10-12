# run_onnx.py —— 不经过 robotd，自己加载行走策略 ONNX、推理一次、计时。
# 用法（Mac，在训练仓库目录里用它的虚拟环境运行）：
#   uv run --frozen python run_onnx.py <策略.onnx>
# 板子上：pip 装好 onnxruntime 和 numpy 后，python3 run_onnx.py <策略.onnx>
import sys
import time

import numpy as np
import onnxruntime as ort

path = sys.argv[1]

# 1. 建立推理会话：ONNX Runtime 读入文件里的"网络结构 + 权重"，
#    相当于 C 程序 fopen 一个数据文件，再按里面的配方调用自己的矩阵乘函数。
sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
print("输入:", [(i.name, i.shape) for i in sess.get_inputs()])
print("输出:", [(o.name, o.shape) for o in sess.get_outputs()])

# 2. 准备一个 61 维观测。全 0 表示"关节都在默认姿态、静止、没有指令"；
#    索引 3..6 是重力方向，身体竖直时约为 (0, 0, -1)。布局见 duck-control/src/obs.rs。
obs = np.zeros((1, 61), dtype=np.float32)
obs[0, 3:6] = [0.0, 0.0, -1.0]

# 3. 推理一次：61 个数进去，14 个数出来（14 个关节的目标偏移，单位 rad）。
act = sess.run(None, {"obs": obs})[0]
print("14 个动作:", np.round(act, 3))

# 4. 计时：连续推理 2000 次取平均，和 50 Hz 控制环的 20 ms 周期比。
N = 2000
t0 = time.perf_counter()
for _ in range(N):
    sess.run(None, {"obs": obs})
dt = (time.perf_counter() - t0) / N
print(f"平均每次推理 {dt * 1e6:.1f} 微秒，占 20 ms 控制周期的 {dt / 0.020 * 100:.2f}%")
