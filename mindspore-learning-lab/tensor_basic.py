import mindspore as ms
from mindspore import Tensor
import numpy as np

# create a tensor
x = Tensor(np.array([[1,2],[3,4]]))

# print tensor
print("Tensor x:")
print(x)

#print shape
print("Shape of x:")
print(x.shape)
