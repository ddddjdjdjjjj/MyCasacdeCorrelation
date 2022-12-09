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
     #   y = torch.tanh(self.out(x))
      #  x=torch.relu(self.out(x))

     #   x0=self.out(x)
     #   y1=torch.exp(self.a*x0)
     #   y2=torch.exp(-self.a*x0)
     #   x1=(y1-y2)/(y1+y2)

     #   mask=x1.isnan()
     #   mask1=y1>0
     #   mask1=torch.logical_and(mask,mask1)
      #  mask2=torch.logical_and(mask,torch.logical_not(mask1))
     #   x1[mask1]=0.999
     #   x1[mask2]=-0.999
     #   return x1
        y=self.out(x)
        #y=self.slope2*torch.minimum(torch.Tensor([0]),y)+self.slope1*torch.maximum(torch.Tensor([0]),y)
        mask=y>0
        y[:,:]=self.low
        y[mask]=self.up
        return y

    def starttrain(self,maxepoches,xs,es):  #es datalen*outputsize


        #xs 192*self.I
        #es 192*outputnum

        #maxepoches=1000
        #self.optimizer=MOEAD2(maxepoches,self.I,es.shape[1],25,xs,es,self.O)
        #self.train_loss=[]
        #self.optimizer.startTrain()
        #iter=self.optimizer.getFinalGen()
        #bestx,self.maxLoss=self.optimizer.getBest()

        #self.slope1=np.max(es)/(self.I*80)
        #self.slope2=-np.min(es)/(self.I*80)




        self.up = torch.Tensor([1])
        self.low = torch.Tensor([-1])
        iter=0

        self.optimizer=JDE1(maxepoches, xs, es, self.I)
        self.optimizer.startTrain()
        bestx1,loss1=self.optimizer.getBest()
        iter=iter+self.optimizer.getFinalGen()

        self.optimizer = JDE2(maxepoches, xs, es, self.I)
        self.optimizer.startTrain()
        bestx2, loss2 = self.optimizer.getBest()
        iter = iter + self.optimizer.getFinalGen()

        self.maxLoss=[loss1,loss2]
        bestx=np.array([bestx1,bestx2])



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
        #bias=np.ones(self.O)
        tweights=bestx[:,:-1]
        bias=bestx[:,-1].reshape(-1)




        #tweights=bestx.reshape(-1,bestx.shape[0])
        #tbias=bestx[-1]
        #tweights1=bestx1.reshape(-1,bestx1.shape[0])
        #tweights2 = bestx2.reshape(-1,bestx2.shape[0])
        #tweights=np.append(tweights1,tweights2,axis=0)
        #tweights=bestx[0].reshape(-1,bestx[0].shape[0])
        #for i in range(1,len(bestx)):
        #    bestx[i]=bestx[i].reshape(-1,bestx[i].shape[0])
        #    tweights=np.append(tweights,bestx[i],axis=0)

        #tweights=bestx.reshape(-1,bestx.shape[0])

        self.out = torch.nn.Linear(self.I, self.O)  # 输出层
        self.out.state_dict()["weight"].copy_(torch.Tensor(tweights))

        bias=torch.tensor(bias)
        self.out.state_dict()["bias"].copy_(bias)

        print(bestx)

        #self.out.state_dict()["bias"].copy_(torch.Tensor([tbias]))


        return iter




    def drawloss(self,xs,es):
        low = -4*np.sqrt(2)*10
        up = 4*np.sqrt(2)*10
        space = 1
        num = int((up - low) / space + 1)
        x = np.zeros(num * num)
        y = np.zeros(num * num)
        l = np.zeros(num * num)
        with torch.no_grad():
            for i in range(num):
                print(i)
                for g in range(num):
                    x[i * num + g] = low + space * i
                    y[i * num + g] = low + space * g
                    self.state_dict()['out.weight'].copy_(torch.Tensor([[x[i * num + g],y[i * num + g]]]))
                    acc1 = 0

                    outputs = self(xs.to(self.device))

                    loss = self.get_correlationLoss(outputs, es)
                    l[i * num + g] = loss

        print(x, y, l)
        mp.figure("3D Scatter", facecolor="lightgray")
        ax3d = mp.gca(projection="3d")  # 创建三维坐标
        mp.title('3D Scatter', fontsize=20)
        ax3d.set_xlabel('x', fontsize=14)
        ax3d.set_ylabel('y', fontsize=14)
        ax3d.set_zlabel('z', fontsize=14)
        mp.tick_params(labelsize=10)
        ax3d.view_init(elev=20, azim=-75)  # 仰角  方位角
        ax3d.scatter(x, y, l, cmap="jet", marker="o")
        mp.show()



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
           # h = hiddenUnit1(input_size, output_size, device).to(device)
            self.hiddenunits.append(h)

    def starttrain(self,hiddenindex,maxepoches,xs,es):

        self.es=es
        self.xs=xs

        trainiter=0
        iter=0

        for i in range(self.K):
            trainiter=self.hiddenunits[i].starttrain(maxepoches, xs.data.numpy(), es.data.numpy())
          #  trainiter = self.hiddenunits[i].starttrain(maxepoches, xs, es)
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


