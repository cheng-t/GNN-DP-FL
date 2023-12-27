import numpy as np
import mindspore as ms
import numpy as np
from utils.data_utils import read_client_data
import copy
from random import sample

class Server():
    def __init__(self,args):
        self.args = args
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)  # 深拷贝
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio  # Ratio of clients per round,args.join_ratio = 1.0 default
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.algorithm = args.algorithm  # AVG/Prox……
        self.save_folder_name = args.save_folder_name
        
        self.eval_gap = args.eval_gap
        self.clients = []
        self.selected_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []
        

    def set_clients(self,clientObj):
        for i in range(self.num_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args,i,len(train_data),len(test_data))
            self.clients.append(client)
        # self.clients = [clientObj(self.args, i,None,None) for i in range(self.num_clients)]
    
    # 按比例选择客户端，并按照id从小到大排序
    def select_clients(self):

        selected_clients = sample(self.clients,self.num_join_clients)

        self.current_num_join_clients = self.num_join_clients

        return sorted(selected_clients)

        # if self.random_join_ratio:  # 这组判断确定current_num_join_clients（数量）
        #     self.current_num_join_clients = \
        #         np.random.choice(range(self.num_join_clients, self.num_clients + 1), 1, replace=False)[0]
        # else:
        #     self.current_num_join_clients = self.num_join_clients
        # selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))

        # selected_clients.sort(key=lambda x: x.id)

        # return selected_clients
    
    # sever->client
    def send_models(self):
        
        assert (len(self.clients) > 0)

        for client in self.clients:
            client.set_parameters(self.global_model)

    
    # client->server
    # def receive_model(self):
    #     # TODO
    #     pass

    def aggregate_parameters(self):

        assert(len(self.selected_clients)>0)

        for param_tensor in self.global_model.parameters_dict():
            # 此处为平均聚合
            param_tensor_avg = sum(c.model.parameters_dict()[param_tensor] for c in self.selected_clients)/len(self.selected_clients)

            param_tensor_avg_copy = param_tensor_avg.copy()

            self.global_model.parameters_dict()[param_tensor].set_data(param_tensor_avg_copy)

    def evaluate(self):

        acc_sum = 0
        test_loss = 0
        for client in self.clients:
            test_acc,test_num,auc,loss = client.test_metric()
            acc_sum += test_acc
            test_loss += loss
        
        return acc_sum/self.num_clients,test_loss

    # def receive_model(self):
        
