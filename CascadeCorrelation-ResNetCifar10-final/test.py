from cascadenet import CascadeNet

#from DatasetFourSpirals import FourSpiralsDataset,FourSpiralsData,drawClass
from DatasetMnistFeatures import MnistFeaturesTrainData,MnistFeaturesDataset
from torch.utils.data import DataLoader
import torch
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')#使用非交互式后端，避免创建窗口

n_output=10

datas = MnistFeaturesTrainData()

train_data = datas.getData()
train_label = datas.getLabels()

datalen = train_data.shape[0]
ts = np.zeros((datalen, n_output), dtype=np.float32)  # 60000*10
ts[np.arange(datalen), train_label] = 1  # 75*3
estrainlabel = ts

print(type(train_data))
train_data = torch.Tensor(train_data)  # numpy->tensor
print(type(train_label))
train_label = torch.Tensor(train_label).long()
