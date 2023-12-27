import mindspore as ms
import copy
import sys

import numpy as np
from utils.data_utils import read_client_data
from mindspore import nn,ops
from mindspore.dataset import GeneratorDataset
import mindspore.dataset as ds


# 实现泊松采样
class IIDBatchSampler:
    def __init__(self, dataset, minibatch_size, iterations):
        self.dataset = dataset
        self.data_size = len(dataset)
        self.minibatch_size = minibatch_size    # batch大小
        self.iterations = iterations            # batch个数

    def generator(self):

        for i in range(self.iterations):                                # 64/6272
            indices = np.where(np.random.rand(len(self.dataset)) < (self.minibatch_size / len(self.dataset)))[0]
            # print("indices size:",indices.size)
            # print('indices:',indices)
            if indices.size > 0:
                yield indices

    def get_dataset(self):
        indices = self.generator()
        indices = next(indices)
        pseudo_batch_size = indices.size
        # print(type(indices))
        # indices = np.array([next(indices) for i in range(indices_size)])
        dataloader = GeneratorDataset(self.dataset, sampler=indices , column_names=['x','y'])
        dataloader = dataloader.batch(pseudo_batch_size)
        # dataloader = dataloader.batch(1)

        # print(self.dataset[91][1])
        # print(self.dataset[516][1])
        # print(self.dataset[532][1])
        # for x,y in dataloader:
        #     # print(x)
        #     print(y)
        return dataloader


class Client():
    """
    Base class for clients in federated learning.
    """
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset  # str
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs       

        self.mini_batch_size = 0 

        # check BatchNorm
        self.has_BatchNorm = False  # 这段代码用于检查神经网络模型中是否包含 批归一化（Batch Normalization）层
        for cell_name, cell in self.model.cells_and_names():  # 迭代神经网络模型的所有子层

            if isinstance(cell, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break
        
        self.privacy = args.privacy
        self.dp_sigma = args.dp_sigma

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = nn.SGD(self.model.trainable_params(), learning_rate=self.learning_rate)

    def __lt__(self,other):
        if self.id<other.id:
            return True
        else:
            return False

    def load_train_data(self,batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        dataloader = GeneratorDataset(train_data,shuffle=True,column_names=["x",'y'])
        dataloader = dataloader.batch(batch_size,drop_remainder=True)
        return dataloader

    def load_train_data_minibatch(self, minibatch_size=None, iterations=None):
        if minibatch_size is None:
            minibatch_size = self.batch_size
        if iterations is None:
            iterations = 1
        train_data = read_client_data(self.dataset, self.id, is_train=True)   
        sampler = IIDBatchSampler(train_data,minibatch_size,iterations)
        # sampler.generator()
        return sampler.get_dataset() 



    
    def load_test_data(self,batch_size=None):
        if batch_size==None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset,self.id,is_train=False)
        dataloader = GeneratorDataset(test_data,shuffle=True,column_names=['x','y'])
        dataloader = dataloader.batch(batch_size,drop_remainder=False)
        return dataloader
    
    def forward_fn(self,data,label):
        logits = self.model(data)
        label = label.astype(ms.int32)
        loss = self.loss_fn(logits,label)
        return loss, logits
    
    def train_step(self,data,label):
        grad_fn = ms.value_and_grad(self.forward_fn,None,self.optimizer.parameters,has_aux=True)
        (loss,_),grads = grad_fn(data,label)
        # for grad in grads:
        #     grad_l2_norm = ops.norm(grad)
            # print("grad:",grad)
            # print('l2_norm:',grad_l2_norm)
        self.optimizer(grads)
        return loss
    
    def train_step_dpsgd(self,data,label):
        grad_fn = ms.value_and_grad(self.forward_fn,None,self.optimizer.parameters,has_aux=True)
        (loss,_),grads = grad_fn(data,label)
        return loss,grads

    def train_metrics(self):
        trainloader = self.load_train_data()
        # trainloader = self.load_train_data_minibatch()

        train_num = 0
        losses = 0
        self.model.set_train()
        # 此处一次迭代就是一个epoch
        i=0
        for x,y in trainloader:
            loss = self.train_step(x,y)
            losses += loss.asnumpy() *y.shape[0]
            train_num += y.shape[0]
            # i+=1
            # test_acc,test_num,auc = self.test_metric()
            # print('test_acc:',test_acc)
        # print(i)
        return losses,train_num
    
    def test_metric(self):
        testloader = self.load_test_data()
        self.model.set_train(False)

        test_acc = 0
        test_num = 0
        test_loss = 0
        for x,y in testloader:
            pred = self.model(x)
            label = y.astype(ms.int32)
            loss = self.loss_fn(pred,label)
            test_loss += loss.asnumpy()

            test_num += y.shape[0]
            test_acc += (pred.argmax(1) == y).asnumpy().sum()
        
        test_acc /= test_num
        auc = 0

        return test_acc,test_num,auc,test_loss
    
    # 将服务器下发的模型复制给自己
    def set_parameters(self,model):

        for param_tensor in model.parameters_dict():

            debug = model.parameters_dict()[param_tensor]

            param_tensor_copy = model.parameters_dict()[param_tensor].copy()

            self.model.parameters_dict()[param_tensor].set_data(param_tensor_copy)

    
    # 将自己的模型参数上传给服务器
    def update_parameters(self, model, new_params):
        
        # TODO
        pass





    
    
    

if __name__ == "__main__":
    """
    OK
    """
    # train_data = read_client_data('mnist', 0, is_train=True)
    # dataloader = GeneratorDataset(train_data,shuffle=True,column_names=["x",'y'])
    # dataloader = dataloader.batch(64,drop_remainder=True)


    pass