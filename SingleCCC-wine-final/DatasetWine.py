from torch.utils.data import Dataset
import numpy as np
import torch

class WineData():

    def __init__(self):
        self.data = []
        self.labels = []

        # 加载数据,修改数据类型
        file=open("./data/wine.data")
        for line in file:
            line=line[:-1]
            lines=line.split(",")
            classtype=int(lines[0])-1
            lines=lines[1:]
            features=[]
            for fe in lines:
                features.append(float(fe))

            self.data.append(np.array(features))
            self.labels.append(classtype)

        self.data=np.array(self.data,dtype=np.float32) #178*13
        self.labels = np.array(self.labels, dtype=np.int64)

        #打乱
        indexs=np.arange(self.labels.shape[0])
        np.random.shuffle(indexs)
        self.data=self.data[indexs,:]
        self.labels=self.labels[indexs]



        #特征归一化到-1，1
        maxlist=[]
        minlist=[]
        for i in range(13):
            maxv=self.data[0][i]
            minv=self.data[0][i]
            for g in range(self.data.shape[0]):

                maxv=max(maxv,self.data[g][i])
                minv = min(minv, self.data[g][i])

            maxlist.append(maxv)
            minlist.append(minv)

        for i in range(13):
            for g in range(self.data.shape[0]):
                self.data[g][i]=-1+2*(self.data[g][i]-minlist[i])/(maxlist[i]-minlist[i])

        #print(self.data)



        print(self.data.shape)
        print(self.labels.shape)

    def fiveSlpit(self):
        rate=[0.0,0.2,0.4,0.6,0.8,1.0]
        splitdatas=[]
        splitlabels = []
        len=self.data.shape[0]
        for i in range(5):
            #print(str(int(len*rate[i]))+" "+str(int(len*rate[i+1])))
            splitdatas.append(self.data[int(len*rate[i]):int(len*rate[i+1])][:])
            splitlabels.append(self.labels[int(len*rate[i]):int(len*rate[i+1])])

        self.traindata=[]
        self.trainlabel=[]
        self.testdata=[]
        self.testlabel=[]



        for i in range(5):
            self.testdata.append(splitdatas[i])
            self.testlabel.append(splitlabels[i])
            tempdata=None
            templabel=None
            for j in range(5):
                if(i==j):
                    continue
                if(tempdata is None):
                    tempdata=splitdatas[j].copy()
                else:
                    tempdata=np.concatenate((tempdata,splitdatas[j]),axis=0)
                if(templabel is None):
                    templabel=splitlabels[j].copy()
                else:
                    templabel=np.concatenate((templabel,splitlabels[j]),axis=0)


            tempdata=np.array(tempdata)
            templabel=np.array(templabel)

            self.traindata.append(tempdata)
            self.trainlabel.append(templabel)

       # print(datas)


    def getData(self):
        return self.data

    def getLabels(self):
        return self.labels

    def getTrainData(self,id):
        return self.traindata[id]
    def getTrainLabel(self,id):
        return self.trainlabel[id]
    def getTestData(self,id):
        return self.testdata[id]
    def getTestLabel(self,id):
        return self.testlabel[id]


class WineDataset(Dataset):
    # dirname 为训练/测试数据地址，使得训练/测试分开
    def __init__(self,data,labels)  :  # 0 train 1 val 2 test
        super( WineDataset, self).__init__()

        self.data=data
        self.labels=labels



    def __len__(self):
        return len(self.labels)


    def __getitem__(self, index):
        return self.data[index], self.labels[index]


#d1=WineData()
#d1.fiveSlpit()
