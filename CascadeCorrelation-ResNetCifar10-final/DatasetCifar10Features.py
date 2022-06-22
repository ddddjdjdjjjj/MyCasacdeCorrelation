
import torch
import numpy as np
import torchvision
import torch.utils.data.dataset
from torch.utils.data import Dataset

class Cifar10FeaturesTrainData(): #60000
    def __init__(self):


        train_features = np.loadtxt("train_features.txt", dtype=np.float32)
        train_labels=np.loadtxt("train_labels.txt", dtype=np.int32)
        print(train_features.shape)
        print(train_labels.shape)




        self.train_data=train_features
        self.train_label=train_labels



    def getData(self):
        return self.train_data

    def getLabels(self):
        return self.train_label

class Cifar10FeaturesTesData(): #10000  写Test被pycharm检测成测试模式
    def __init__(self):

        test_features = np.loadtxt("test_features.txt", dtype=np.float32)
        test_labels = np.loadtxt("test_labels.txt", dtype=np.int32)
        print(test_features.shape)
        print(test_labels.shape)

        self.test_data = test_features
        self.test_label = test_labels

        self.test_data = np.array(self.test_data)
        self.test_label = np.array(self.test_label)

        print(self.test_data.shape)


    def getData(self):
        return self.test_data

    def getLabels(self):
        return self.test_label


class Cifar10FeaturesDataset(Dataset):
    # dirname 为训练/测试数据地址，使得训练/测试分开
    def __init__(self, data, labels):  # 0 train 1 val 2 test
        super(Cifar10FeaturesDataset, self).__init__()

        self.data=data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        return self.data[index], self.labels[index]



#d=Cifar10FeaturesTrainData()
'''
from torch.utils.data import DataLoader

train_datas = MnistFeaturesTrainData()
train_data = train_datas.getData()
train_label = train_datas.getLabels()
train_data = torch.Tensor(train_data)              #numpy->tensor
train_label = torch.Tensor(train_label).long()
train_dataset = MnistFeaturesDataset(train_data, train_label)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)

with torch.no_grad():

    features=[]
    labels=[]

    for train_data in train_loader:
        train_datas, train_labels = train_data
        print(train_datas.shape)
        #print(len(model.hiddenunits))
        #print(model.hiddenunits[0].shape)
        break
'''

