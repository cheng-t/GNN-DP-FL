# OK
import mindspore as ms
from mindspore import dataset as ds
import mindspore.dataset.transforms as transforms
from mindspore.dataset import GeneratorDataset
from download import download
import mindspore.dataset.vision as vision
import numpy as np

from dataset.utils.dataset_utils import check, separate_data, split_data, save_file
from generate_server_testset import generate_server_testset

dir_path = "mnist_ms/"

train_path = dir_path + "train/"
test_path = dir_path + "test/"


def download_minst(path):
    mnist_url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/MNIST_Data.zip"
    download(mnist_url,path,kind= 'zip',replace = False)

download_minst(dir_path+'rawdata')

def generate_mnist(dir_path, num_clients, num_classes, niid, balance, partition, need_server_testset=False):

    transform = transforms.Compose([vision.ToTensor(), vision.Normalize([0.5], [0.5],is_hwc=False)])

    trainset = ds.MnistDataset(dir_path+'rawdata/MNIST_Data/train',usage = 'train');
    trainset.map(operations=transform)
    testset = ds.MnistDataset(dir_path+'rawdata/MNIST_Data/test',usage = 'test');
    testset.map(operations=transform)

    trainloader = GeneratorDataset(trainset,column_names=['x','y'],shuffle=False);
    trainloader.batch(60000)
    testloader = GeneratorDataset(testset,column_names=['x','y'],shuffle=False);
    testloader.batch(len(testset))

    i = 0
    x_all = []
    y_all = []
    for x,y in trainloader:

        x_all.append(np.transpose(x,(2,0,1)).asnumpy())
        y_all.append(y.asnumpy())

    for x,y in testloader:

        x_all.append(np.transpose(x,(2,0,1)).asnumpy())
        y_all.append(y.asnumpy())

    dataset_image = np.array(x_all)
    dataset_label = np.array(y_all)



    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes,
                                        niid, balance, partition)
    train_data, test_data = split_data(X, y)

    if need_server_testset:
        generate_server_testset(test_data, test_path)

    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes,
              statistic, niid, balance, partition)





