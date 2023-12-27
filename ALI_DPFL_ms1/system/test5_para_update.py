# OK

import mindspore as ms

from mindspore import nn,ops

class FedAvgCNN(nn.Cell):
    def __init__(self,in_features=1, num_classes=10, dim=1024):
        super().__init__()

        self.conv1 = nn.SequentialCell(
            nn.Conv2d(in_features,
                      32,
                      kernel_size=5,
                      padding=0,
                      stride=1,
                      has_bias=True,
                      pad_mode='valid'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2),stride=2)
        )
        self.conv2 = nn.SequentialCell(
            nn.Conv2d(32,
                      64,
                      kernel_size=5,
                      padding=0,
                      stride=1,
                      has_bias=True,
                      pad_mode='valid'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.fcs = nn.SequentialCell(
            nn.Dense(dim,512),
            nn.ReLU(),
            nn.Dense(512,num_classes)
        )
    
    def construct(self,x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = ops.flatten(x,start_dim=1)

        x = self.fcs(x)

        return x
net1 = FedAvgCNN()
net = FedAvgCNN()
para_dict = net.parameters_dict()
netlist = [net1,net]

for param_tensor in net.parameters_dict():
    print(param_tensor)
    # print(net.parameters_dict()[param_tensor].value())
    # print(net1.parameters_dict()[param_tensor].value())
    
    para_sum = sum(c.parameters_dict()[param_tensor] for c in netlist)
    para_sum_copy = para_sum.copy()

    # print(para_sum_copy.value())

    net.parameters_dict()[param_tensor].set_data(para_sum_copy)
    # print(net.parameters_dict()[param_tensor].value())


# import mindspore.nn as nn
# from mindspore import Tensor
# import mindspore as ms
# # 定义模型
# class MyModel(nn.Cell):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         self.fc = nn.Dense(10, 5)

#     def construct(self, x):
#         x = self.fc(x)
#         return x

# # 创建模型实例
# model = MyModel()

# # 获取模型的参数字典
# params = model.parameters_dict()

# # 修改参数值
# new_value = Tensor([1.0], dtype=ms.float32)
# params['fc.weight'] = new_value

# # 打印修改后的参数值
# print(model.parameters_dict()['fc.weight'])
