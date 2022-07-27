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



class hiddenUnit1(torch.nn.Module):
    def __init__(self, input_size, output_size,device):
        super(hiddenUnit1, self).__init__()
        self.I = input_size
        self.O = output_size
        self.device=device
        #self.convergedt = 10
        self.out = torch.nn.Linear(input_size, output_size)  # 输出层


       # val =  4*np.sqrt(6 / (self.O + self.I))  # tanh   self.O+self.I过大,导致随机值为0，影响
        val=10  #val过小容易导致相关性为0

        self.out.state_dict()["weight"].copy_(torch.Tensor(np.random.uniform(-val, val, (1,self.I))))
       # self.out.state_dict()["bias"].copy_(torch.zeros(output_size))
        #for name, param in self.named_parameters():
       #     if "out.bias" in name:
        #        param.requires_grad = False
        self.out.state_dict()["bias"].copy_(torch.ones(output_size))

        self.lr=0.01
        self.alpha=0.998

        #self.optimizer = maxoptimsgd(self.parameters(), lr=self.lr)  # 优化方法SGD
        #self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)# 优化方法Adam
        self.optimizer = torch.optim.SGD(self.parameters(), momentum=0.9, lr=self.lr)  # 优化方法SGD
        self.maxLoss=0


    def forward(self, x):

        #x = torch.sigmoid(self.out(x))
        x = torch.tanh(self.out(x))
        return x

    def get_correlationLoss(self, vs, es):

        v_term = vs - torch.mean(vs, axis=0)  # 199*1 vs-vm
        e_term = es - torch.mean(es, axis=0)  # 199*2  es-em

        t=v_term.permute(1,0)
        corr = t.mm(e_term)
       # print(v_term,e_term)

       # return -1.0*torch.abs(corr[0][0])
        #print(-1.0*torch.sum(torch.abs(corr)).data.numpy())



        return -1.0*torch.sum(torch.abs(corr))






    def starttrain(self,maxepoches,xs,es):  #es datalen*outputsize

       # self.drawloss(xs, es)
        iter=0
        converged=False
        self.convergednum = 0


        self.train_loss=[]
        for epoch in range(1, maxepoches + 1):
            # train

            if (self.lr > 0.001):
               self.lr = self.lr * self.alpha
               self.optimizer = torch.optim.SGD(self.parameters(), momentum=0.9,lr=self.lr)  # 优化方法SGD

            self.train()  # 开启训练模式
            running_loss = 0.0  # 每一轮损失值
            acc1 = 0
            #t1 = time.perf_counter()  # 开始时间

            self.optimizer.zero_grad()
            outputs = self(xs.to(self.device))

            loss = self.get_correlationLoss(outputs, es)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            #print(outputs,running_loss)


            #print("train loss: %f " % loss)
            #print('[epoch %d] train time:%f' % (epoch,time.perf_counter() - t1))

            self.train_loss.append(running_loss)


            converged = self.check_io_convergence(self.train_loss,epoch)

            if (converged):
                break

        outputs = self(xs.to(self.device))
        self.maxLoss = self.get_correlationLoss(outputs, es)
        print("hidden unit maxloss:",self.maxLoss,"iter:",epoch)

        iter=epoch
        return iter

    def check_io_convergence(self,train_loss,epoch):
        if len(train_loss) >= 50 and abs(train_loss[-1] - train_loss[-40] )< 0.001:  # 训练次数>=2且损失值下降少于eps abs表波动
            # print(train_loss[-1],train_loss[-2])


            return True
        else:
            return False


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
            #h=hiddenUnits2( input_size, output_size,device).to(device)
            h = hiddenUnit1(input_size, output_size, device).to(device)
            self.hiddenunits.append(h)

    def starttrain(self,hiddenindex,maxepoches,xs,es):

        self.es=es
        self.xs=xs

        trainiter=0
        iter=0

        for i in range(self.K):
           # trainiter=self.hiddenunits[i].starttrain(maxepoches, xs.data.numpy(), es.data.numpy())
            trainiter = self.hiddenunits[i].starttrain(maxepoches, xs, es)
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


