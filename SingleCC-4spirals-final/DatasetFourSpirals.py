import math
import matplotlib
#matplotlib.use('Agg')#使用非交互式后端，避免创建窗口

import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import numpy as np
import torch
from sklearn import datasets
from PIL import Image
import torchvision.transforms as transforms

#x [-10,10]   x<0 0 x>0 1
class FourSpiralsData():
    def __init__(self, datalen):

        self.data,self.labels=self.generate_four_spirals5(datalen)

        # 打乱顺序
        indexs = np.arange(len(self.labels))
        np.random.shuffle(indexs)
        self.data = self.data[indexs]
        self.labels = self.labels[indexs]

    def getData(self):
        return self.data

    def getLabels(self):
        return self.labels



    def generate_two_spirals2(self,n_points, noise=.5):
        n = np.sqrt(np.random.rand(n_points, 1)) * 780 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(n_points, 1) * noise
        d1y = np.sin(n) * n + np.random.rand(n_points, 1) * noise
        d1x=d1x.astype(np.float32)
        d1y = d1y.astype(np.float32)
        return (np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))),
                np.hstack((np.zeros(n_points,dtype=np.int64), np.ones(n_points,dtype=np.int64))))

    def generate_two_spirals3(self, datalen):
        datalen1 = datalen // 2
        datalen2 = datalen - datalen // 2

        train_i1 = np.arange(0,  datalen1)
        train_i2 = np.arange(0,  datalen2)
        # 1 - 104.584

        alpha=9
        beta=360/alpha
        alpha1 =  0.5*math.pi-math.pi * (train_i1 - 1) / alpha  # (train_i-1)/15  105
        alpha2 =  -0.5*math.pi-math.pi * (train_i2 - 1) / alpha  # (train_i-1)/15  105

        beta1 = 0.4+2.4 * train_i1 / beta  # r
        beta2 = 0.4+2.4 * train_i2 / beta  # r

        x0 = beta1 * np.cos(alpha1)
        y0 = beta1 * np.sin(alpha1)

        x1 = beta2 * np.cos(alpha2)
        y1 = beta2 * np.sin(alpha2)

        x = np.append(x0, x1).reshape(datalen, 1)
        y = np.append(y0, y1).reshape(datalen, 1)
        data=np.append(x, y, axis=1).astype(np.float32)
        label=np.append(np.zeros((datalen1, 1)).astype(np.int32), np.ones((datalen2, 1)).astype(np.int64))
        return data,label

    def generate_four_spirals4(self, datalen):#归一化至[-1,1]
        datalen1 = datalen // 4
        datalen2 = datalen // 4
        datalen3 = datalen // 4
        datalen4 = datalen-datalen//4*3



        train_i1 = np.arange(0,  datalen1)
        train_i2 = np.arange(0,  datalen2)
        train_i3 = np.arange(0, datalen3)
        train_i4 = np.arange(0, datalen4)
        # 1 - 104.584

        alpha=16
        beta=104
        alpha1 =  math.pi * (train_i1 ) / alpha  # (train_i-1)/15  105
        alpha2 =  math.pi * (train_i2 ) / alpha  # (train_i-1)/15  105
        alpha3 = math.pi * (train_i3) / alpha  # (train_i-1)/15  105
        alpha4 = math.pi * (train_i4) / alpha  # (train_i-1)/15  105

        beta1 = 6.5 * (beta-train_i1) / beta  # r   6.5*(104-96)/104=0.5
        beta2 = 6.5  * (beta-train_i2) / beta  # r
        beta3 = 0.5+6.5 * (beta - train_i3) / beta  # r
        beta4 = 0.5+6.5 * (beta - train_i4) / beta  #

        x0 = beta1 * np.sin(alpha1)
        y0 = beta1 * np.cos(alpha1)

        x1 = -x0
        y1 = -y0

        x2 = beta3 * np.sin(alpha3)
        y2 = beta3 * np.cos(alpha3)

        x3 = -x2
        y3 = -y2

        low=-1
        up=1
        x0=low+(x0-(-7))/(7-(-7))*(up-low)
        x1 =low + (x1 - (-7)) / (7 - (-7)) * (up-low)
        x2 =low + (x2 - (-7)) / (7 - (-7)) * (up-low)
        x3 = low + (x3 - (-7)) / (7 - (-7)) * (up-low)

        y0 = low + (y0 - (-7)) / (7 - (-7)) * (up-low)
        y1 = low + (y1 - (-7)) / (7 - (-7)) * (up-low)
        y2 = low + (y2 - (-7)) / (7 - (-7)) * (up-low)
        y3 = low + (y3 - (-7)) / (7 - (-7)) * (up-low)



        x = np.append(x0, x1)
        x = np.append(x, x2)
        x = np.append(x, x3).reshape(datalen, 1)
        y = np.append(y0, y1)
        y = np.append(y, y2)
        y = np.append(y, y3).reshape(datalen, 1)
        data=np.append(x, y, axis=1).astype(np.float32)
        label=np.append(np.zeros((datalen1, 1)).astype(np.int64), np.ones((datalen2, 1)).astype(np.int64))
        label=np.append(label, np.full((datalen3, 1),2).astype(np.int64))
        label = np.append(label, np.full((datalen4, 1), 3).astype(np.int64))

        return data,label

    def generate_four_spirals5(self, datalen):#
        datalen1 = datalen // 4
        datalen2 = datalen // 4
        datalen3 = datalen // 4
        datalen4 = datalen-datalen//4*3



        train_i1 = np.arange(0,  datalen1)
        train_i2 = np.arange(0,  datalen2)
        train_i3 = np.arange(0, datalen3)
        train_i4 = np.arange(0, datalen4)
        # 1 - 104.584

        alpha=16
        beta=104
        alpha1 =  math.pi * (train_i1 ) / alpha  # (train_i-1)/15  105
        alpha2 =  math.pi * (train_i2 ) / alpha  # (train_i-1)/15  105
        alpha3 = math.pi * (train_i3) / alpha  # (train_i-1)/15  105
        alpha4 = math.pi * (train_i4) / alpha  # (train_i-1)/15  105

        beta1 = 6.5 * (beta-train_i1) / beta  # r   6.5*(104-96)/104=0.5
        beta2 = 6.5  * (beta-train_i2) / beta  # r
        beta3 = 0.5+6.5 * (beta - train_i3) / beta  # r
        beta4 = 0.5+6.5 * (beta - train_i4) / beta  #

        x0 = beta1 * np.sin(alpha1)
        y0 = beta1 * np.cos(alpha1)

        x1 = -x0
        y1 = -y0

        x2 = beta3 * np.sin(alpha3)
        y2 = beta3 * np.cos(alpha3)

        x3 = -x2
        y3 = -y2




        x = np.append(x0, x1)
        x = np.append(x, x2)
        x = np.append(x, x3).reshape(datalen, 1)
        y = np.append(y0, y1)
        y = np.append(y, y2)
        y = np.append(y, y3).reshape(datalen, 1)
        data=np.append(x, y, axis=1).astype(np.float32)
        label=np.append(np.zeros((datalen1, 1)).astype(np.int64), np.ones((datalen2, 1)).astype(np.int64))
        label=np.append(label, np.full((datalen3, 1),2).astype(np.int64))
        label = np.append(label, np.full((datalen4, 1), 3).astype(np.int64))

        return data,label


class FourSpiralsDataset(Dataset):
    # dirname 为训练/测试数据地址，使得训练/测试分开
    def __init__(self, data, labels):  # 0 train 1 val 2 test
        super(FourSpiralsDataset, self).__init__()
        self.transform = [transforms.ToTensor()]
        self.data=data

       # if not isinstance(data,torch.Tensor):
        #  for t in self.transform:
        #     self.data = t(data)[0]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        return self.data[index], self.labels[index]





def drawClass(net,train_data,train_label,epoch,tip,file_path):
    x=[]
    y=[]


    for i in np.arange(-7,7,0.1):
        for j in np.arange(-7,7,0.1):
            y.append(i)
            x.append(j)

    class1x=[]
    class1y = []
    class2x = []
    class2y = []
    class3x = []
    class3y = []
    class4x = []
    class4y = []






    #low = -1
    #up = 1
    for index in range(len(x)):
            #label=net.test1(torch.Tensor([np.append(low+(x[index]-(-7))/(7-(-7))*(up-low),
            #                                        low + (y[index] - (-7)) / (7 - (-7)) * (up - low))]))
            label = net.test1(torch.Tensor([np.append(x[index], y[index])]))

            _,maxi=torch.max(label,dim=1)

            if(maxi==0):
                class1x.append(x[index])
                class1y.append(y[index])
            elif(maxi==1):
                class2x.append(x[index])
                class2y.append(y[index])
            elif (maxi == 2):
                class3x.append(x[index])
                class3y.append(y[index])
            elif (maxi == 3):
                class4x.append(x[index])
                class4y.append(y[index])

    #dataxy=(train_data.numpy()-low)/(up-low)*(7-(-7))+(-7)
    #labels=train_label.numpy()
    dataxy=train_data.numpy()
    labels=train_label.numpy()

    mask=labels==0
    x0=dataxy[mask,0]
    y0=dataxy[mask,1]
    mask = labels== 1
    x1 = dataxy[mask, 0]
    y1 = dataxy[mask, 1]
    mask = labels == 2
    x2 = dataxy[mask, 0]
    y2 = dataxy[mask, 1]
    mask = labels == 3
    x3 = dataxy[mask, 0]
    y3 = dataxy[mask, 1]

    ax = plt.gca()
    ax.set_aspect(1)

    #plt.figure(figsize=(5,5),dpi=100)
    plt.xlim([-7, 7])
    plt.ylim([-7, 7])

  #  plt.axis('equal')
    #black green gray blue white gold red yellow olive pink purple

    plt.scatter(class1x,class1y,c='black',s=1) #黑白有叠加
    plt.scatter(class3x, class3y, c='gold', s=1)  # 黑白有叠加
    plt.scatter(class4x, class4y, c='green', s=1)  # 黑白有叠加
   # axes.scatter(class2x, class2y,c='w',s=15)
    plt.scatter(x0, y0, c='red', s=10)
    plt.scatter(x1, y1, c='blue', s=10)
    plt.scatter(x2, y2, c='pink', s=10)
    plt.scatter(x3, y3, c='purple', s=10)


    plt.savefig( file_path+'drawclass-'+str(epoch)+'-'+str(tip)+'.png')

   # plt.show()
    plt.clf()


#c= b c g k m r w y

from torch.utils.data import DataLoader

datalen=96*4
batch_size=1
fourspiralsdata=FourSpiralsData(datalen)
data=fourspiralsdata.getData()
label=fourspiralsdata.getLabels()


train_data=data[0:datalen,:]
train_label=label[0:datalen]

#val_data=data[datalen//3:datalen//3*2,:]
#val_label=label[datalen//3:datalen//3*2]

#test_data=data[datalen//3*2:datalen,:]
#test_label=label[datalen//3*2:datalen]

train_dataset=FourSpiralsDataset(train_data,train_label)
#val_dataset=TwoSpiralsDataset(val_data,val_label)
#test_dataset=TwoSpiralsDataset(test_data,test_label)

train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
#val_loader=DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
#test_loader=DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

ax = plt.gca()
ax.set_aspect(1)
low=-1
up=1
for step, data in enumerate(train_loader, start=0):
    datas, labels = data
    data=datas.numpy()
    label=labels.numpy()
    x=data[0][0]
    y=data[0][1]
    if labels[0]==0:
        plt.scatter(x, y, c="r", s=10, marker='o')
    elif labels[0]==1:
        plt.scatter(x, y, c="b", s=10, marker='x')
    elif labels[0]==2:
        plt.scatter(x, y, c="m", s=10, marker='v')
    elif labels[0]==3:
        plt.scatter(x, y, c="g", s=10, marker='D')
plt.savefig('./4-spirals.jpg')
plt.show()
'''
for step, data in enumerate(val_loader, start=0):
    datas, labels = data
    data=datas.numpy()
    label=labels.numpy()
    x=data[0][0]
    y=data[0][1]
    if labels[0]==0:
        plt.scatter(x,y,c="g")
    else:
        plt.scatter(x, y, c="y")
plt.show()
for step, data in enumerate(test_loader, start=0):
    datas, labels = data
    data=datas.numpy()
    label=labels.numpy()
    x=data[0][0]
    y=data[0][1]
    if labels[0]==0:
        plt.scatter(x,y,c="c")
    else:
        plt.scatter(x, y, c="k")
plt.show()
'''





