# peek_onnx.py —— 在终端里看 ONNX 的结构：输入输出、每一层的权重形状、算子顺序、元数据。
# 用法（Mac，在训练仓库目录里运行）：
#   uv run --frozen python peek_onnx.py <策略.onnx>
import sys

import onnx
from onnx import numpy_helper

m = onnx.load(sys.argv[1])
g = m.graph

for i in g.input:
    print("输入", i.name, [d.dim_value for d in i.type.tensor_type.shape.dim])
for o in g.output:
    print("输出", o.name, [d.dim_value for d in o.type.tensor_type.shape.dim])

total = 0
print("\n权重（initializer）:")
for t in g.initializer:
    a = numpy_helper.to_array(t)
    total += a.size
    print(f"    {t.name:24s} {a.shape}")
print(f"共 {total} 个数 × 4 字节 = {total * 4} 字节（对照 .onnx 文件大小）")

print("\n计算顺序（node）:")
for n in g.node:
    print(f"    {n.op_type:5s} {list(n.input)} -> {list(n.output)}")

print("\n元数据（导出时写进去的说明）:")
for p in m.metadata_props:
    print(f"    {p.key} = {p.value[:90]}")
