import mindspore as ms
from mindspore import Tensor
import mindspore.ops as ops
import numpy as np

x = Tensor(np.array([[1,2],[3,4]]),ms.float32)
y = Tensor(np.array([[10,20],[30,40]]),ms.float32)

print("Add operator:")
print(ops.add(x,y))

print("Multiply operator:")
print(ops.mul(x,y))

print("Matrix Multiplication operator:")
print(ops.matmul(x,y))

z = Tensor(np.array([[-1,2],[3,-4]]),ms.float32)


print("ReLU operator")
print(ops.relu(z))


scores = Tensor(np.array([2.0,1.0,0.1]),ms.float32)

print("Softmax operator:")
print(ops.softmax(scores))


