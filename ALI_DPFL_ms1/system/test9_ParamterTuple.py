import mindspore.nn as nn  
from mindspore import Tensor  
from mindspore import ParameterTuple  
from mindspore import context  
from mindspore import Parameter  
import mindspore
  
# 假设有一个模型 model  
class MyModel(nn.Cell):  
    def __init__(self):  
        super(MyModel, self).__init__()  
        self.conv1 = nn.Conv2d(3, 64, 3, has_bias=False)  
        self.conv2 = nn.Conv2d(64, 128, 3, has_bias=False)  
        self.fc1 = nn.Dense(128 * 10 * 10, 256)  
        self.fc2 = nn.Dense(256, 10)  
  
        self.params = ParameterTuple(self.trainable_params())  
  
    def construct(self, x):  
        x = self.conv1(x)  
        x = self.conv2(x)  
        x = x.view(-1, 128 * 10 * 10)  
        x = self.fc1(x)  
        x = self.fc2(x)  
        return x  
  
# 创建模型  
model = MyModel()  
  
# 设置不同参数组的学习率  
lr = 0.01  
lr_fc = 0.001  
for param in model.params:  
    if isinstance(param, Parameter):  
        if 'fc' in param.name:  
            param.set_parameter_data(Tensor(param.data, mindspore.float32))  
            param.set_parameter_data(Tensor(lr_fc, mindspore.float32))  
        else:  
            param.set_parameter_data(Tensor(param.data, mindspore.float32))  
            param.set_parameter_data(Tensor(lr, mindspore.float32))  