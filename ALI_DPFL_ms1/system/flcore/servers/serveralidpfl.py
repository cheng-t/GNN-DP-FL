import copy
import math
import os

import ujson
import h5py

from flcore.clients.clientalidpfl import clientALIDPFL
from flcore.optimizer.utils.RDP.get_max_steps import get_max_steps
from flcore.servers.serverbase import Server

from utils.data_utils import read_server_testset
from flcore.optimizer.utils.RDP.compute_dp_sgd import apply_dp_sgd_analysis


import mindspore
from mindspore import nn,ops

def compute_new_tau(mu, C, Gamma, sigma, d, hat_B, Rs, Rc, tau_star):
    '''
    mu: 强凸系数
    C: Clipping norm
    Gamma: 体现异质性的系数
    sigma: 噪声乘子
    hat_B: 跟batchsize有关的一个值
    T = min{Rs·tau_star,Rc}
    '''
    T = min(Rs * tau_star, Rc)
    print(f"mu={mu}, C={C}, Gamma={Gamma}, sigma={sigma}, d={d}, "
          f"hat_B={hat_B}, Rs={Rs}, Rc={Rc}, T={T}, tau_star={tau_star}")
    dp_noise_bound = (sigma ** 2 * C ** 2 * d) / (hat_B ** 2)
    # 分子
    molecule = (4 / (mu ** 2)) + 3 * (C ** 2) + 2 * Gamma * T * mu + dp_noise_bound
    # 分母
    denominator = (2 + 2 / T) * (C ** 2 + dp_noise_bound)
    ret = math.sqrt(1 + molecule / (denominator + 1e-6))
    print(f"分子 = {molecule}, 分母 = {denominator}, 原始tau = {ret}")
    ret = int(ret + 0.5)  # 十分位四舍五入
    ret = max(1, ret)
    ret = min(ret, 100)
    return ret




class ALIDPFL(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_clients(clientALIDPFL)
        self.rs_server_acc = []  # 中心方测出来的准确率
        self.rs_server_loss = []  # 中心方测出来的loss，不是各client的loss的加权
        self.loss = nn.CrossEntropyLoss()  # 交叉熵损失函数,用来测server_loss的
        self.batch_sample_ratio = args.batch_sample_ratio
        self.dp_sigma = args.dp_sigma  # 算epsilon的时候要用
        self.dp_norm = args.dp_norm  # 裁剪范数C
        self.need_adaptive_tau = args.need_adaptive_tau
        self.tau_star = args.local_iterations  # optimal tau
        self.rs_tau_list = [self.tau_star]  # adap local iteration list
        self.dp_epsilon = args.dp_epsilon

        self.global_rounds = args.global_rounds
        self.local_iterations = args.local_iterations

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

        self.hat_B = int(self.batch_sample_ratio * min([client.train_samples for client in self.clients]))
        # 所有参数的数量
        self.dimension_of_model = sum(ops.numel(param) for name,param in self.global_model.parameters_dict().items())

        delta = 10 ** (-5)
        orders = (list(range(2, 64)) + [128, 256, 512])  # 默认的lamda

        self.Rc = get_max_steps(self.dp_epsilon, delta, self.batch_sample_ratio, self.dp_sigma, orders)

        print(f"===================== Rc={self.Rc} =====================")

        if self.need_adaptive_tau and self.global_rounds >= self.Rc:  # Rs>=Rc 取1最好
            self.need_adaptive_tau = False
            self.local_iterations = 1
            self.tau_star = 1
            for client in self.clients:
                client.need_adaptive_tau = False
                self.local_iterations = 1
                self.tau_star = 1

    def send_models(self):  # sever->client
        assert (len(self.clients) > 0)

        for client in self.clients:
             client.set_tau(self.tau_star)  # 发模型的时候，把tau_star发下去

             client.set_parameters(self.global_model)

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        if self.need_adaptive_tau:
            self.global_model.set_train(False)
            for client in self.clients:
                client.model.set_train(False)
        
        self.global_model = copy.deepcopy(self.uploaded_models[0])
       
