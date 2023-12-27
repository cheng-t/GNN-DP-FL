import mindspore as ms
import copy
import sys
  
from utils.data_utils import read_client_data
from mindspore import nn
from mindspore.dataset import GeneratorDataset

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

        # check BatchNorm
        self.has_BatchNorm = False  # 这段代码用于检查神经网络模型中是否包含 批归一化（Batch Normalization）层
        for layer in self.model.cells():  # 迭代神经网络模型的所有子层
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break
        
        self.privacy = args.privacy
        self.dp_sigma = args.dp_sigma

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = nn.SGD(self.model.trainable_params(), learning_rate=self.learning_rate)

    def load_train_data(self,batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        dataloader = GeneratorDataset(train_data,shuffle=True)
        dataloader = dataloader.batch(batch_size,drop_remainder=True)
        return dataloader
    
# OK
if __name__ == "__main__":
    train_data = read_client_data('mnist', 0, is_train=True)
    dataloader = GeneratorDataset(train_data,shuffle=True,column_names=["x",'y'])
    dataloader1 = dataloader.batch(64,drop_remainder=True)
    print(dataloader1)
    for x,y in dataloader1:
        print(x,y)
    pass