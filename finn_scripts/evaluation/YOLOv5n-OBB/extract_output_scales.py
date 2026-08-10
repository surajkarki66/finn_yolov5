import numpy as np

from qonnx.core.modelwrapper import ModelWrapper

model = ModelWrapper("./dataflow_parent.onnx")

mul_nodes = [n for n in model.graph.node if n.op_type == "Mul"]

for i, node in enumerate(mul_nodes):
    print(f"Mul node {i}: {node.name}")
    print("Inputs:", node.input)

# Extract both constants
mul0 = model.get_initializer("Mul_0_param0")
mul1 = model.get_initializer("Mul_1_param0")
mul2 = model.get_initializer("Mul_2_param0")

# Save as .npy
np.save("mul_0.npy", mul0)
np.save("mul_1.npy", mul1)
np.save("mul_2.npy", mul1)

print("Mul_0 shape:", mul0.shape)
print("Mul_1 shape:", mul1.shape)
print("Mul_2 shape:", mul2.shape)