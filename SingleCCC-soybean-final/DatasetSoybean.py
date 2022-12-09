from torch.utils.data import Dataset
import numpy as np
import torch



class SoybeanData():

    def __init__(self):
        self.data = []            #classes*len*featurenums
        self.labels = []          #classes*len

        # 加载数据,修改数据类型
        file=open("./data/soybean-small.data") #(47,35) 4

        self.classes=4
        self.featurenums=35

        for i in range(self.classes):
            self.data.append([])
            self.labels.append([])

        for line in file:
            #print(line)

            lines=line.split(",")
            classtype = None
            if (lines[-1] == "D1\n"):
               classtype = 0
            elif (lines[-1] == "D2\n"):
               classtype = 1
            elif (lines[-1] == "D3\n"):
               classtype = 2
            else:
               classtype = 3

            lines=lines[:-1]
            features=[]
            for fe in lines:
                if(fe == ""):
                    continue
               # print(fe)
                features.append(float(fe))

            self.data[classtype].append(np.array(features))
            self.labels[classtype].append(classtype)

        for i in range(self.classes):
            self.data[i] = np.array(self.data[i], dtype=np.float32)  # 178*13
            self.labels[i] = np.array(self.labels[i], dtype=np.int64)
            # 打乱
            indexs = np.arange(self.labels[i].shape[0])
            np.random.shuffle(indexs)
            self.data[i] = self.data[i][indexs, :]
            self.labels[i] = self.labels[i][indexs]



        #特征归一化到-1，1
        maxlist=[]
        minlist=[]
        for i in range(self.featurenums):
            maxv=self.data[0][0][i]
            minv=self.data[0][0][i]
            for j in range(self.classes):
               for g in range(self.data[j].shape[0]):

                  maxv=max(maxv,self.data[j][g][i])
                  minv = min(minv, self.data[j][g][i])

            maxlist.append(maxv)
            minlist.append(minv)

        for i in range(self.featurenums):
            for j in range(self.classes):
              for g in range(self.data[j].shape[0]):
                if(maxlist[i]-minlist[i]==0):
                    self.data[j][g][i]=0
                else:
                    self.data[j][g][i]=-1+2*(self.data[j][g][i]-minlist[i])/(maxlist[i]-minlist[i])

        #print(self.data)

        len=0
        for i in range(self.classes):
           #print(self.data[i].shape)
           #print(self.labels[i].shape)
           len=len+self.labels[i].shape[0]
        print(str(len)+"*"+str(self.featurenums)+"-"+str(classtype))

    def fiveSlpit(self):
        rate=[0.0,0.2,0.4,0.6,0.8,1.0]
        splitdatas=[]
        splitlabels = []

        for i in range(5):
            #print(str(int(len*rate[i]))+" "+str(int(len*rate[i+1])))
            splitdata=None
            splitlabel=None
            for j in range(self.classes):
                len=self.data[j].shape[0]
                tempdata=self.data[j][int(len*rate[i]):int(len*rate[i+1])][:]
                templabel=self.labels[j][int(len*rate[i]):int(len*rate[i+1])]
                if (splitdata is None):
                    splitdata = tempdata.copy()
                else:
                    splitdata=np.concatenate((splitdata,tempdata),axis=0)
                if (splitlabel is None):
                    splitlabel = templabel.copy()
                else:
                    splitlabel = np.concatenate((splitlabel, templabel), axis=0)

            splitdatas.append(splitdata)
            splitlabels.append(splitlabel)


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


class SoybeanDataset(Dataset):
    # dirname 为训练/测试数据地址，使得训练/测试分开
    def __init__(self,data,labels)  :  # 0 train 1 val 2 test
        super( SoybeanDataset, self).__init__()

        self.data=data
        self.labels=labels



    def __len__(self):
        return len(self.labels)


    def __getitem__(self, index):
        return self.data[index], self.labels[index]


#d1=SoybeanData()
#d1.fiveSlpit()
