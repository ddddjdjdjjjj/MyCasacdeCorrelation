import torch
import torch.nn.functional as Fun
import time
from hiddenunit import hiddenUnitsPool
import matplotlib.pyplot as plt
import numpy as np


# 定义BP神经网络
class CascadeNet(torch.nn.Module):
    def __init__(self, n_feature, n_output,lr,device, file_path):
        super(CascadeNet, self).__init__()
        self.I = n_feature
        self.O = n_output
        self.hiddenus = 1  # 每层隐藏单元数

        self.lr=lr
        self.alpha=0.99

        self.hiddenunits=[]
        self.netLoss=float('inf')


        self.out = torch.nn.Linear(n_feature, n_output)    #输出层
        val =  4*np.sqrt(6 / (self.O + self.I))  # 0.34
        tweights= np.random.uniform(-val, val, (self.O, self.I))  #


        self.out.state_dict()["weight"].copy_(torch.tensor(tweights))



       # self.out.state_dict()["bias"].copy_(torch.zeros(self.O))
        self.out.state_dict()["bias"].copy_(torch.ones(n_output))
        for name, param in self.named_parameters():
            if "out.bias" in name:
                param.requires_grad = False


        self.convergedt=2

        self.leastepoch=2


       # self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)  # 优化方法SGD
        self.optimizer = torch.optim.SGD(self.parameters(), momentum=0.5, lr=self.lr)  # 优化方法SGD

        self.loss_func = torch.nn.CrossEntropyLoss()
        self.loss_func2 = torch.nn.MSELoss(reduction='none')
        self.device=device
        self. file_path= file_path

    def setlr(self,lr):
        self.lr=lr
        #self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)  # 优化方法SGD
        self.optimizer = torch.optim.SGD(self.parameters(), momentum=0.5, lr=self.lr)  # 优化方法SGD


    def forward(self, x):


        x=self.out(x)
        return x

    def get_loss(self, x, labels):
        outputs=self.forward(x)
        return 0.5*(self.loss_func2(outputs, labels.to(self.device)).data)/len(labels)

    def get_loss2(self, x, labels):
        outputs=Fun.softmax(self.forward(x),dim=1)
        return (labels-outputs).data



    def train_io(self,maxepoches,train_dataset,train_loader):


        iter=0
        converged=False
        self.convergednum=0
        best_acc = 0
        iters=[]
        self.train_loss = []
        for epoch in range(1, maxepoches + 1):
            # train
            if (self.lr > 0.001):
                self.lr = self.lr * self.alpha
            self.optimizer = torch.optim.SGD(self.parameters(), momentum=0.5,lr=self.lr)  # 优化方法SGD



            self.train()  # 开启训练模式
            running_loss = 0.0  # 每一轮损失值
            train_accurate=0
            test_accurate=0
            t1 = time.perf_counter()  # 开始时间
            for step, data in enumerate(train_loader, start=0):
                datas, labels = data

                self.optimizer.zero_grad()
                outputs = self(datas.to(self.device))
                loss = self.loss_func(outputs, labels.to(self.device))
                loss.backward()
                self.optimizer.step()

            #    rate = (step + 1) / len(train_loader)
            #    a = "*" * int(rate * 50)
            #    b = "." * int((1 - rate) * 50)
            #    print("\rtrain loss: {:^3.0f}%[{}->{}]".format(int(rate * 100), a, b), end="")

            acc1 = 0
            with torch.no_grad():
                for step, data in enumerate(train_loader, start=0):
                    datas, labels = data

                    outputs = self(datas.to(self.device))
                    loss = self.loss_func(outputs, labels.to(self.device))
                    running_loss += loss.item()
                    predict_y = torch.max(outputs, dim=1)[1]
                    acc1 += (predict_y == labels.to(self.device)).sum().item()
                    #      acc1 += (predict_y == torch.max(test_labels.to(self.device), dim=1)[1]).sum().item()

                train_accurate = acc1 / len(train_dataset)

                    # print train process


           # print('train time:', time.perf_counter() - t1)





            self.train_accurate = train_accurate

            self.train_loss.append(running_loss / step)
            iters.append(epoch)

            converged = self.check_io_convergence(self.train_loss,epoch,train_accurate)
            if (converged):
                break;
            if((epoch-1)%100==0):
                print(self.train_loss[-1],train_accurate,test_accurate)

              #  print(self.out.state_dict()["bias"])

        self.netLoss = self.train_loss[-1]
        self.train_acc=train_accurate

        print('out of iter:', self.train_loss[-1], self.train_acc)



        iter=epoch
        return iter

    def check_io_convergence(self,train_loss,epoch,acc1):
        if epoch >= self.leastepoch and abs(train_loss[-1] - train_loss[-2] )< 0.0001:  # 训练次数>=2且损失值下降少于eps abs表波动
            print('loss convergence:',train_loss[-1],train_loss[-2],acc1)

            return True
        elif self.train_accurate==1.0:
            print('train_data ok:', train_loss[-1], acc1)
            return True
        #    self.lr=self.lr*0.1
       #     print(self.lr,epoch,train_loss[-1],acc1,acc2)

          #  self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)  # 优化方法SGD
        else:
            return False


    def  add_hidden_unit(self,hiddenindex,xs,es):

        hiddenunitspool=hiddenUnitsPool(self.I,self.hiddenus,1,self.device, self.file_path)

        trainiter=hiddenunitspool.starttrain(hiddenindex,300,xs,es)

        h=hiddenunitspool.getBestUnit(hiddenindex)
        y=h(xs)


        #print(h.maxLoss)

        xs=torch.cat([xs,y],dim=1)



        self.hiddenunits.append(h)
      #  print(xs.shape,ts.shape)



        self.I=self.I+self.hiddenus

      #  print(self.out.state_dict()["weight"],self.out.state_dict()["bias"])
        temp=self.out
        self.out = torch.nn.Linear(self.I, self.O)
        #self.out.state_dict()["weight"].copy_(torch.cat((temp.state_dict()["weight"],torch.zeros(self.O).reshape(self.O,1)), 1))
        #self.out.state_dict()["weight"].copy_(torch.cat((temp.state_dict()["weight"], self.out.state_dict()["weight"][:,-1].reshape(self.O,1)),1))

        val = 4*np.sqrt(6 / (self.O + self.I))  # 0.34
        tweights = np.random.uniform(-val, val, (self.O,self.hiddenus))
        #tweights[1][0]=0
        self.out.state_dict()["weight"].copy_(
        torch.cat((temp.state_dict()["weight"],torch.Tensor(tweights)),1))

        self.out.state_dict()["bias"].copy_(temp.state_dict()["bias"])
        for name, param in self.named_parameters():
            if "out.bias" in name:
                param.requires_grad = False
        #print(self.out.state_dict()["weight"], self.out.state_dict()["bias"])

        #将新参数置入优化方法中
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)  # 优化方法SGD



       # return xs.data,ts.data  #新的训练数据，将冻结节点化作训练数据新特征
        return trainiter,xs.data # 新的训练数据，将冻结节点化作训练数据新特征


    def test3(self, xs):

        print('total hiddenunits:',len(self.hiddenunits))
        lunit=0
        munit=0
        runit=0
        for hiddenunit in self.hiddenunits:

            ys=hiddenunit(xs)
            if(ys>0.95):
                runit=runit+1
            elif(ys<-0.95):
                lunit=lunit+1
            else:
                munit=munit+1
            xs = torch.cat([xs, ys], dim=1)

        print('lunit:',lunit)
        print('munit:', munit)
        print('runit:', runit)


        ys = self(xs)

        return ys

    def test1(self, xs):

        for hiddenunit in self.hiddenunits:

            ys=hiddenunit(xs)
            xs = torch.cat([xs, ys], dim=1)


        ys = self(xs)

        return ys

    def test2(self, xs, yt):

        for hiddenunit in self.hiddenunits:

            ys=hiddenunit(xs)
            xs = torch.cat([xs, ys], dim=1)


        ys = self(xs)
        print(ys.shape)

        predict_y = torch.max(ys, dim=1)[1]

        acc2 = (predict_y == torch.max(yt.to(self.device), dim=1)[1]).sum().item()

        test_accurate = acc2 / len(yt)


        print(test_accurate)









