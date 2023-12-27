# import torch
# PyTorch
import torch
from torch import tensor
import numpy as np

x_ = np.ones((64, 1, 28, 28))
x = tensor(x_, dtype=torch.float32)
net_conv = torch.nn.Conv2d(1, 32, kernel_size=5)
net_pool = torch.nn.MaxPool2d(kernel_size=(2,2),stride=2)
x = net_conv(x)
output = net_pool(x).detach().numpy().shape
print(output)
# (1, 240, 1021, 637)

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import numpy as np

x_ = np.ones((64, 1, 28, 28))
x = Tensor(x_, mindspore.float32)
net_conv = nn.Conv2d(1, 32, kernel_size=5, pad_mode='valid')
net_pool = nn.MaxPool2d(kernel_size=(2,2),stride=2)
x = net_conv(x)
output = net_pool(x).shape
print(output)
# (1, 240, 1021, 637)
