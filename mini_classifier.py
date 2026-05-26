import mindspore as ms
from mindspore import Tensor
import mindspore.ops as ops
import numpy as np

input_data = Tensor(
    np.array([2.0,1.0,0.5]),
    ms.float
)

#第一层权重

w1 = Tensor(
    np.array([
    [2.0,0.8],
    [0.5,0.1],
    [0.3,0.4]
    ]),
    ms.float32
)
# 输入了三维的数据，输出2维
#乘上了一个3x2的矩阵
hidden = ops.matmul(input_data,w1)

#推理的核心部分
#输入*权重

print("Hidden layer output")
print(hidden)

activated = ops.relu(hidden)

print("After ReLU:")
print(activated)

#第二层
w2 = Tensor(
    np.array([
    [0.7,0.2],
    [0.1,0.9]
]),
    ms.float32
)
#矩阵的形状是2x2的一个矩阵

output = ops.matmul(activated,w2)

print("Raw output scores:")
print(output)


probabilities = ops.softmax(output)

print("Classification probabilities:")
print(probabilities)

predicted = ops.argmax(probabilities)
print("Predicted class:")
print(predicted)

# 这个程序里的数字都是瞎写的
# 但是在实际的机器学习中的数字不是
# 而是通过训练得来的 w1 和 w2
# 训练是让模型找到更好的权重
# Training:输入数据->前向计算->算损失->反向传播->更新权重
# 这个程序不是在训练而是在inference（推理）
# 输入数据->用训练好的权重->前向计算->输出预测
# 不学习只计算
# 神经网络 本质就是Tensor在operator pipeline里流动

