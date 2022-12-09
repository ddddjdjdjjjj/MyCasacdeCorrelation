import torch
import torch.nn.functional as Fun
import time
from maxoptim import maxoptimsgd
import numpy as np
import matplotlib.pyplot as mp
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

from JDE import JDE,JDE1,JDE2,JDE3,JDE4
from MOEAD2 import MOEAD2


class hiddenUnits2(torch.nn.Module):
    def __init__(self, input_size, output_size,device):#实际上是self.I*self.O  但训练过程中是self.I*1
        super(hiddenUnits2, self).__init__()
        self.I = input_size
        self.O = output_size
        self.device=device
        self.maxLoss=0

    def forward(self, x):
        y = torch.tanh(self.out(x))

        return y

    def starttrain(self,maxepoches,xs,es):  #es datalen*outputsize




        self.optimizer=JDE(maxepoches, xs, es, self.I)
        self.optimizer.startTrain()
        bestx,self.maxLoss=self.optimizer.getBest()
        iter=self.optimizer.getFinalGen()



        print("hidden unit maxloss:", self.maxLoss, "iter:", iter)
        '''
        with open("./data.txt", "w") as f:
            data1 = xs.tolist()
            data2 = es.tolist()
            data3 = bestx.tolist()


            f.write(str(data1) + "\n")
            f.write(str(data2) + "\n")
            f.write(str(data3))
        '''
        #tweights = bestx
        #bias=1

        tweights=bestx[:-1]
        bias=bestx[-1]




        self.out = torch.nn.Linear(self.I, self.O)  # 输出层
        self.out.state_dict()["weight"].copy_(torch.Tensor(tweights))

        bias=torch.tensor(bias)
        self.out.state_dict()["bias"].copy_(bias)

       # print(bestx)

        #self.out.state_dict()["bias"].copy_(torch.Tensor([tbias]))

        # 释放，减少模型大小
        self.optimizer = None


        return iter

class hiddenUnitsPool():
    def __init__(self, input_size, output_size, num_candidates,device, file_path):
        self.I = input_size
        self.O = output_size
        self.K = num_candidates


        self.device=device
        self.file_path= file_path

        self.hiddenunits=[]
        for i in range(num_candidates):
            h=hiddenUnits2( input_size, output_size,device).to(device)
          #  h = hiddenUnit1(input_size, output_size, device).to(device)
            self.hiddenunits.append(h)

    def starttrain(self,hiddenindex,maxepoches,xs,es):

        self.es=es
        self.xs=xs

        trainiter=0
        iter=0

        for i in range(self.K):
            trainiter=self.hiddenunits[i].starttrain(maxepoches, xs.data.numpy(), es.data.numpy())
            #trainiter = self.hiddenunits[i].starttrain(maxepoches, xs, es)
            iter=iter+trainiter


        #plt.xlabel('Iteration')
       # plt.ylabel('Loss')
        #for i in range(self.K):
        #    plt.plot(self.hiddenunits[i].train_loss)
        #plt.savefig('./record/hidden_loss'+str(hiddenindex)+'.png')
        #plt.clf()


        #with open(self.file_path+"hidden_weight-"+str(hiddenindex)+".txt", "w") as f:
        #    maxdata=-float('inf')
        #    mindata=float('inf')
        #    for i in range(self.K):
        #        data1=self.hiddenunits[i].out.state_dict()["weight"].numpy()
        #        data2=self.hiddenunits[i].out.state_dict()["bias"].numpy()
        #        f.write(str(data1)+"\n")
        #        f.write(str(data2)+"\n")
        #        maxdata=max(maxdata,data1.max())
        #        mindata = min(mindata, data1.min())
        #    f.write("max:"+str(maxdata)+",min:"+str(mindata))

        return iter

    def getBestUnit(self,hiddenindex):
        bestindex=0
        for i in range(self.K):
            if(self.hiddenunits[i].maxLoss<self.hiddenunits[bestindex].maxLoss):
                bestindex=i
            #if (self.hiddenunits[i].maxLoss > self.hiddenunits[bestindex].maxLoss):
            #    bestindex = i

        with open(self.file_path+"hidden_best-"+str(hiddenindex)+".txt", "w") as f:

            data1=self.es.numpy().tolist()
            with torch.no_grad():
              outputs = self.hiddenunits[bestindex](self.xs.to(self.device))
            data2=outputs.data.tolist()


            f.write(str(data1)+"\n")
            f.write(str(data2) + "\n")
            f.write(str(self.hiddenunits[bestindex].maxLoss))

        return self.hiddenunits[bestindex]


