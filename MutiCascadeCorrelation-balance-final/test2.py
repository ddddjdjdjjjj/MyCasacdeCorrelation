import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from DatasetMnist import *
import numpy as np

save_path = './net/cascadenet-0.pth'
model=torch.load(save_path)
print(len(model.hiddenunits))


device=torch.device('cpu')
print(device)


model = model.to(device)


train_datas = MnistTrainData()
train_data = train_datas.getData()
train_label = train_datas.getLabels()
train_data = torch.Tensor(train_data)              #numpy->tensor
train_label = torch.Tensor(train_label).long()
train_dataset = MnistDataset(train_data, train_label)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)

with torch.no_grad():


    acc = 0
    for train_data in train_loader:
        train_datas, train_labels = train_data
        #print(train_datas.shape)
        #print(len(model.hiddenunits))
        #print(model.hiddenunits[0].shape)
        outputs = model.test1(train_datas.to(device))

        predict_y = torch.max(outputs, dim=1)[1]
        acc += (predict_y == train_labels.to(device)).sum().item()


    print("trainacc:",acc/len(train_dataset))




    #features=np.loadtxt("train_featrures.txt",dtype=np.float32)

test_datas = MnistTestData()
test_data = test_datas.getData()
test_label = test_datas.getLabels()
test_data = torch.Tensor(test_data)              #numpy->tensor
test_label = torch.Tensor(test_label).long()
test_dataset = MnistDataset(test_data, test_label)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)

with torch.no_grad():

    acc = 0
    for test_data in test_loader:
        test_datas, test_labels = test_data

        outputs = model.test1(test_datas.to(device))

        predict_y = torch.max(outputs, dim=1)[1]
        acc += (predict_y == test_labels.to(device)).sum().item()

    print("testacc:", acc / len(test_dataset))


    #features=np.loadtxt("train_featrures.txt",dtype=np.float32)


