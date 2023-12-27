import numpy as np
import time
import copy

from flcore.clients.clientbase import Client
from flcore.optimizer.dp_optimizer import DPAdam, DPSGD
from mindspore import ops


class clientALIDPFL(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.dp_norm = args.dp_norm
        self.batch_sample_ratio = args.batch_sample_ratio
        self.auto_s = args.auto_s
        self.local_iterations = args.local_iterations
        self.need_adaptive_tau = args.need_adaptive_tau
        self.tau_star = args.local_iterations

        self.loss_global_model = 1.0
        self.grad_global_model = 1.0
        self.loss_client_model = 1.0
        self.grad_client_model = 1.0

        if self.privacy:
            self.optimizer = DPSGD(
                l2_norm_clip=self.dp_norm,  # 裁剪范数
                noise_multiplier=self.dp_sigma,
                minibatch_size=self.batch_size,  # batch_size
                microbatch_size=1,  # 几个样本梯度进行一次裁剪
                # 后面这些参数是继承父类的（SGD优化器的一些参数）
                params=self.model.trainable_params(),
                learning_rate=self.learning_rate,
            )

    def set_tau(self, tau_star):
        self.tau_star = tau_star

    def train(self):
        print("---------------------------------------")
        print(f"Client {self.id} is training, privacy={self.privacy}, AUTO-S={self.auto_s}")
        minibatch_size = int(self.train_samples * self.batch_sample_ratio)
        trainloader = self.load_train_data_minibatch(minibatch_size=minibatch_size,
                                                     iterations=self.tau_star)   

        self.model.set_train()
        
        max_local_epochs = self.local_epochs  # 在DP里，一般epoch = 1，甚至都不会跑完全部的数据，只会来几个iterations
        for step in range(max_local_epochs):  # FedPRF中限定epochs=1
            
            for x,y in trainloader:

                if self.need_adaptive_tau:
                    global_model = copy.deepcopy(self.model)
                    output_of_global_model = global_model(x)
                    loss_of_global_model = self.loss_fn(output_of_global_model, y)
                    self.loss_global_model = loss_of_global_model.item()
                    # 这一句是不是可以不加
                    # loss_of_global_model.backward()   #反向传播获得梯度
                    self.grad_global_model = copy.deepcopy(global_model)

                gradients_list = []
                losses_list = []
                for x_single,y_single in zip(x,y):
                    x_single = ops.expand_dims(x_single,0)
                    y_single = ops.expand_dims(y_single,0)
                    # print(x_single)
                    # print(y_single)
                    loss,grad = self.train_step_dpsgd(x_single,y_single)
                    gradients_list.append(grad)
                    losses_list.append(loss)
                self.optimizer(gradients_list)
                self.loss_batch_avg = sum(losses_list) / len(losses_list)  # 对逐样本的loss做平均，回传

                if self.need_adaptive_tau:
                    # 拷贝是为了拿梯度，但是不能影响正常的梯度下降    
                    client_model = copy.deepcopy(self.model)
                    output_of_client_model_batch = client_model(x)
                    loss_of_client_model_batch = self.loss_fn(output_of_client_model_batch, y)    
                    self.loss_client_model = loss_of_client_model_batch.item()  
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             