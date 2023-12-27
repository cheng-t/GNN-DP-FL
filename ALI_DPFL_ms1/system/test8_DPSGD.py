import numpy as np
import mindspore
from mindspore import Tensor, ops, nn
from mindspore import value_and_grad

# Cell object to be differentiated
class Net(nn.Cell):
    def construct(self, x, y, z):
        return x * y * z
x = Tensor([1, 2], mindspore.float32)
y = Tensor([-2, 3], mindspore.float32)
z = Tensor([0, 3], mindspore.float32)
net = Net()
grad_fn = value_and_grad(net, grad_position=(0,1,2))
output, inputs_gradient = grad_fn(x, y, z)
print(output)

print(inputs_gradient)