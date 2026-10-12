# peek_pt.py —— 打开训练存档 .pt，看里面到底装了什么，以及为什么它比 ONNX 大 6 倍。
# 用法（Mac，在训练仓库目录里用它的虚拟环境运行，需要 torch）：
#   uv run --frozen python peek_pt.py <model_17500.pt>
import sys

import torch

ck = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
print("顶层的键:", list(ck.keys()))
print("训练到第几轮 iter =", ck["iter"])
print("infos:", ck["infos"])  # common_step_counter = 每只鸭子走过的控制步数


def numbers(d):
    """数一个字典里所有张量一共有多少个数。"""
    return sum(v.numel() for v in d.values() if torch.is_tensor(v))


total = 0
for key in ["actor_state_dict", "critic_state_dict"]:
    sd = ck[key]
    n = numbers(sd)
    total += n
    print(f"\n{key}: {len(sd)} 个张量，共 {n} 个数")
    for name, t in sd.items():
        print(f"    {name:28s} {tuple(t.shape)}")

opt = ck["optimizer_state_dict"]
n_opt = sum(numbers(s) for s in opt["state"].values())
total += n_opt
print(f"\noptimizer_state_dict（Adam 的两份动量）: 共 {n_opt} 个数")
print("    当前学习率:", opt["param_groups"][0]["lr"])

print(f"\n合计 {total} 个数 × 4 字节 ≈ {total * 4 / 1e6:.2f} MB（对照 .pt 文件大小）")
