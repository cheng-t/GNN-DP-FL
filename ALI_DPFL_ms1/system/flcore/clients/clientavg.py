from flcore.clients.clientbase import Client

import numpy as np
import mindspore as ms
from mindspore import ops,nn
import sys

class clientAVG(Client):
    def __init__(self,args,id,train_sample,test_sample,**kwargs):
        super().__init__(args,id,train_sample,test_sample,**kwargs)
    
    def train(self):
        # print(f"Clinet {self.id} is training……")
        print(self.id,end=' ')
        if self.id==9:
            print('')
        sys.stdout.flush()
        
        trainloader = self.load_train_data_minibatch(iterations=1)
        # test
        # trainloader = self.load_train_data_minibatch(minibatch_size=313,iterations=5)
        
        # for x,y in trainloader:
        #     for x_single,y_single in zip(x,y):
        #         x_single = ops.expand_dims(x_single,0)
        #         y_single = ops.expand_dims(y_single,0)
        #         # print(x_single)
        #         print(y_single)
        #         loss = self.train_step(x_single,y_single)
        #     print('')

        self.model.set_train()

        max_local_epochs = self.local_epochs

        train_losses = 0

        for step in range(max_local_epochs):
            for x,y in trainloader:
                loss = self.train_step(x,y)
                train_losses += loss.asnumpy() * y.shape[0]
        # return train_losses
        # print(f"Client {self.id} lossed: {train_losses}",end=' ')
        # print(f"Clinet {self.id} had trained.")

if __name__ == '__main__':
    pass

