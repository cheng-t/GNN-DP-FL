from flcore.clients.clientbase import Client
from flcore.trainmodels.models import FedAvgCNN
from flcore.servers.serverdpfl import DPFL
import argparse
import mindspore as ms

ms.set_seed(0)



# def test_client(args):

#     if "mnist" in args.dataset:
#         args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024)

#     test_client = Client(args,id=0,train_samples=None,test_samples=None)

#     train_loss ,train_num = test_client.train_metrics()

#     test_acc,test_num,auc = test_client.test_metric()
#     print('test_acc:',test_acc)
def run_DPFL(args):
    if 1:
        if "mnist" in args.dataset:
            args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024)
        elif "Cifar10" in args.dataset:
            args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
        elif "omniglot" in args.dataset:
            args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=33856).to(args.device)
            # args.model = CifarNet(num_classes=args.num_classes).to(args.device)
        # elif "Digit5" in args.dataset:
        #     args.model = Digit5CNN().to(args.device)
        else:
            args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816).to(args.device)

        # print(args.model)    
    fedavg_model = DPFL(args)
    fedavg_model.train()
    fedavg_model.test()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    # goal
    parser.add_argument('-go', "--goal", type=str, default="test",
                        help="The goal for this experiment")
    # need_daptive_tau
    parser.add_argument('-nat', "--need_adaptive_tau", type=bool, default=True,
                        help="use the adaptive tau(True) or fixed tau(False)")
    # global_rounds
    parser.add_argument('-gr', "--global_rounds", type=int, default=500,
                        help="Rs in the ALIDPFL")
    # local_iterations
    parser.add_argument('-li', "--local_iterations", type=int, default=1,
                        help="DP-FedSGD need li")
    # batch sample ratio (Poisson sampling)
    parser.add_argument('-bsr', "--batch_sample_ratio", type=float, default=0.05,
                        help="The ratio of Poisson sampling")
    # sigma
    parser.add_argument('-dps', "--dp_sigma", type=float, default=2.0)
    # epsilon
    parser.add_argument('-dpe', "--dp_epsilon", type=float, default=2.0)

    # AUTO-S
    parser.add_argument('-as', "--auto_s", type=bool, default=False,
                        help="Clipping method: AUTO-S(True) or Abadi(False)")
    # norm
    parser.add_argument('-dpn', "--dp_norm", type=float, default=0.1)
    # 数据集
    parser.add_argument('-data', "--dataset", type=str, default="mnist")  # mnsit, Cifar10, fmnist
    # algorithm
    parser.add_argument('-algo', "--algorithm", type=str, default="ALIDPFL")
    # local_learning_rate
    # DPSGD 学习率在0.1这个级别，再低一个数量级acc升不了，再高一个数量级loss指数爆炸
    # 传统SGD 学习率取0.01/0.001这个级别，太高了不利于收敛
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.5,
                        help="Local learning rate")
    # batch_size (DP相关的使用泊松采样，这个batch size用不上了)
    parser.add_argument('-lbs', "--batch_size", type=int, default=64)
    # num_clients
    parser.add_argument('-nc', "--num_clients", type=int, default=10,
                        help="Total number of clients")
    # local_epochs
    parser.add_argument('-ls', "--local_epochs", type=int, default=1,
                        help="Multiple update steps in one local epoch.")

    # privacy:
    dp_help = r"""
        when you use FedPRF, please only {privacy = True};
        If you do not need DP (For algorithm comparison), please set:
        1. auto_s = False , ie. use Abadi Clipping
        2. dp_norm = 1e10 , ie. do not clipping (FedPRF dp_norm=1)
        3. dp_sigma = 0, ie. do not add noise
        4. IMPORTANT: lr = 0.1 for FL+DP; lr = 0.01 for FL without DP
    """
    parser.add_argument('-dp', "--privacy", type=bool, default=True,
                        help="differential privacy" + dp_help)

    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-nb', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="cnn")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)

    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")  # 客户端参加的比例(client drift程度)
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")

    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")  # 这里如果设成3，就会自动跑三次 wow!
    parser.add_argument('-eg', "--eval_gap", type=int, default=10,
                        help="Rounds gap for evaluation")  # 几轮一次test的意思

    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)  # ？
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)  # ？ ↓下面应该都是跟具体算法有关了
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    parser.add_argument('-fte', "--fine_tuning_epoch", type=int, default=0)
    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")

    args = parser.parse_args()  

    if args.device == "cuda" and ms.context.get_context("device_target") == "GPU":
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    run_DPFL(args)
