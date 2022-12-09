from torch.utils.data import Dataset
import numpy as np
import torch
from sklearn import datasets
from PIL import Image
from torch.utils.data import DataLoader
class IrirsData():

    def __init__(self):

        # 加载数据,修改数据类型
        dataset = datasets.load_iris()
        self.data = dataset['data'].astype(np.float32)  # numpy 150*4
        self.labels = dataset['target'].astype(np.int64)  # numpy 150
        # 打乱顺序 dataset会在打乱一次
        indexs = np.arange(len(self.labels))
        np.random.shuffle(indexs)
        self.data = self.data[indexs]
        self.labels = self.labels[indexs]

    def getData(self):
        return self.data

    def getLabels(self):
        return self.labels

class IrirsDataset(Dataset):
    # dirname 为训练/测试数据地址，使得训练/测试分开
    def __init__(self,data,labels)  :  # 0 train 1 val 2 test
        super( IrirsDataset, self).__init__()


        self.data=data
        self.labels=labels




    def __len__(self):
        return len(self.labels)


    def __getitem__(self, index):
        return self.data[index], self.labels[index]


'''
datalen=50
batch_size=1
twospiralsdata=IrirsData()
data=twospiralsdata.getData()
label=twospiralsdata.getLabels()


train_data=data[0:datalen]
train_label=label[0:datalen]

train_dataset=IrirsDataset(train_data,train_label)

train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

for step, data in enumerate(train_loader, start=0):
    datas, labels = data
    print(datas,labels)
'''