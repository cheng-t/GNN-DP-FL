from mindspore import nn,ops

import mindspore as ms
import copy
class DPSGD(nn.Optimizer):
    def __init__(self, params, learning_rate,l2_norm_clip, noise_multiplier, minibatch_size,microbatch_size):
        super(DPSGD,self).__init__(learning_rate,params)

        self.l2_norm_clip = l2_norm_clip
        # self.clip_bound_C = clip_bound_C
        self.noise_multiplier = noise_multiplier
        self.minibatch_size = minibatch_size
        self.microbatch_size = microbatch_size
        # self.epsilon = epsilon
        self.assign = ops.Assign()

    # gradients_list是一次迭代样本的梯度列表
    def construct(self,gradients_list):

        # l2_norm_list用于存储一次迭代所有样本的clip后的值
        l2_norm_list = self.l2_norm_calculate(gradients_list)
        # 对梯度进行裁剪
        gradients_bar = self.clip(gradients_list,l2_norm_list)
        # 加噪
        gradient_final = self.add_noise(gradients_bar)
        # 参数更新
        return self.update_parameters(gradient_final)



    # gradients_list是一次迭代样本的梯度列表
    def l2_norm_calculate(self,gradients_list):

        # l2_norm_list用于存储一次迭代所有样本的clip后的值
        l2_norm_list = []
        i=0
        # gradients是一个样本的梯度
        for gradients in gradients_list:
            total_norm = 0
            # l2_norm_list.append([])
            
            for grad in gradients:
                grad_l2_norm = ops.norm(grad)
                total_norm+=grad_l2_norm**2

            # 至此，完成了单样本的二范数的计算
            total_norm =total_norm ** .5
            l2_norm_list.append(total_norm)
            i+=1
        
        return l2_norm_list
    
    # gradients_list是一次迭代样本的梯度列表
    # l2_norm是一次迭代的所有样本中，每个样本的l2范数
    def clip(self,gradients_list,l2_norm_list):

        # clip_coef = 1.0
        # for total_norm in l2_norm_list:
        #     clip_coef = min(self.l2_norm_clip / (total_norm + 1e-6), 1.)  # 范数比较，得到等下要裁剪的部分

        # 每个样本
        for i in range(len(gradients_list)):
            clip_coef = min(self.l2_norm_clip / (l2_norm_list[i] + 1e-6), 1.)  # 范数比较，得到等下要裁剪的部分
            # 一个样本的每一层参数
            for j in range(len(gradients_list[i])):
                if(type(clip_coef) == float):
                    self.assign(gradients_list[i][j],gradients_list[i][j]*clip_coef)
                else:
                    self.assign(gradients_list[i][j],gradients_list[i][j]*clip_coef.value())
                # gradients_list[i][j] = gradients_list[i][j]*clip_coef.value()

        return gradients_list



    def add_noise(self,gradients_bar):
        
        gradients_bar = [list(gradients_bar[i]) for i in range(len(gradients_bar))]

        gradients_sum = copy.deepcopy(gradients_bar[0])
        for i in range(len(gradients_bar)-1):
            # gradients_sum += gradients_bar[i+1]
            for j in range(len(list(gradients_bar[i+1]))):
                gradients_sum[j]+=gradients_bar[i+1][j]
            # gradients_sum = ops.add(gradients_sum,gradients_bar[i+1])

        for i in range(len(gradients_sum)):
            gradients_sum[i] += self.l2_norm_clip * self.noise_multiplier * ops.randn_like(gradients_sum[i])
            gradients_sum[i] *= (self.microbatch_size / self.minibatch_size)
        
        return gradients_sum

    # 更新参数
    def update_parameters(self,gradient):
        
        lr = self.get_lr()
        for i in range(len(self.parameters)):
            update = self.parameters[i] - lr*gradient[i]
            self.assign(self.parameters[i],update)
        return self.parameters



    


