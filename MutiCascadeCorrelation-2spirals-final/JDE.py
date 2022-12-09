# coding:UTF-8
'''
Created on 2015年5月12日
@author: zhaozhiyong
'''

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


class JDE():
    def __init__(self,maxgen=100,xs=[],es=[],dim=1):
        self.NP = 100
        self.dim = dim+1
        #self.dim = dim
        self.LBOUND = np.full([1, self.dim], -80)
        self.UBOUND = np.full([1, self.dim], 80)

        #self.LBOUND[0,-1]=-100
        #self.UBOUND[0,-1] = 100


        self.tau1=0.1
        self.tau2=0.1

        self.F = np.full([self.NP, 1], 0.5)
        self.CR = np.full([self.NP, 1], 0.9)

        self.oldF=self.F
        self.oldCR=self.CR

        self.MAXGEN = maxgen
        self.es=es
        self.xs=xs.T     #DIM*train_len

        #self.slope1 = np.max(es) / (self.dim * 80)
       # self.slope2 = -np.min(es) / (self.dim * 80)
       # self.up = np.max(es)  # avg up
       # self.low = np.min(es)  # avg low
        self.up=1
        self.low = -1

       # print(self.slope1,self.slope2)

        #地形图
        '''
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x0=[]
        y0=[]
        z0=[]
        for y in np.arange(-10,10,0.5):
            for x in np.arange(-10,10,0.5):
                X=np.array([y,x])
                X=X.reshape(-1,X.shape[0])
                f=self.calFitness( X)
                x0.append(x)
                y0.append(y)
                z0.append(f)

        ax.scatter(x0, y0, z0, c='r', marker='.',alpha=0.5)
        plt.show()
        '''



        self.bestvalue=float('inf')
        self.bestx=[]

        # 初始化
        self.gen = 0

        self.pop_x = self.LBOUND + np.random.rand(self.NP, self.dim) * (self.UBOUND - self.LBOUND)  # NP*DIM
        self.pop_value = self.calFitness(self.pop_x)  # NP*1

    def getFinalGen(self):
        return self.gen


    def tanh(self,os):
        return np.tanh(os)

    def sin(self,os):
        return np.sin(os)

    def sigmoid(self,os):
        return 1/(1+np.exp(-os))

    def relu(self,os):
        return (abs(os) + os) / 2

    def fun1(self,os):

       # y = self.slope2 * np.minimum(os,0) + self.slope1 * np.maximum(os,0)
        mask = os > 0
        os[:, :] = self.low
        os[mask] = self.up
        return os

    def calFitness(self, X):  #NP*DIM

       # bias=X[:,-1].reshape(X.shape[0],1)
        #X=X[:,:-1]
        bias=1

        #vs=self.sigmoid(X.dot(self.xs)+bias)
        #vs=self.tanh(X.dot(self.xs)+bias)
        #vs = self.relu(X.dot(self.xs) + bias)
        vs= self.fun1(X.dot(self.xs)+bias)

        v_term = vs - np.mean(vs, axis=1).reshape(vs.shape[0],1)  # 199*1 vs-vm
        e_term = self.es - np.mean(self.es, axis=0)  # 199*2  es-em
        t = v_term
        corr = t.dot(e_term)

        fitness=-1.0 * np.sum(np.abs(corr),axis=1)
        #fitness=-1.0*corr[:,0]

        return fitness

    def mutation(self):
        m, n = self.pop_x.shape

        r0 = np.arange(m)  # 0,1,....,popsize-1
        r1 = np.around(np.random.rand(m) * (m - 1)).astype(np.int32)

        while True:
            pos = r0 == r1
            repeats = np.sum(pos)
            if repeats == 0:
                break
            else:
                r1[pos] = np.around(np.random.rand(repeats) * (m - 1)).astype(np.int32)

        r2 = np.around(np.random.rand(m) * (m - 1)).astype(np.int32)

        while True:
            pos = np.logical_or((r2 == r0), (r2 == r1))
            repeats = np.sum(pos)
            if repeats == 0:
                break
            else:
                r2[pos] = np.around(np.random.rand(repeats) * (m - 1)).astype(np.int32)

        r3 = np.around(np.random.rand(m) * (m - 1)).astype(np.int32)

        while True:
            pos = np.logical_or((r3 == r0), (r3 == r1), (r3 == r2))
            repeats = np.sum(pos)
            if repeats == 0:
                break
            else:
                r3[pos] = np.around(np.random.rand(repeats) * (m - 1)).astype(np.int32)

        XMutationTmp = self.pop_x[r1, :] + self.F * (self.pop_x[r2, :] - self.pop_x[r3, :])

        # 越界反弹处理
        xl = np.repeat(self.LBOUND, m, axis=0)
        xu = np.repeat(self.UBOUND, m, axis=0)

        # LBOUND
        pos1 = XMutationTmp < xl
        XMutationTmp[pos1] = 2 * xl[pos1] - XMutationTmp[pos1]

        pos2 = XMutationTmp > xu
        pos_ = np.logical_and(pos1, pos2)
        XMutationTmp[pos_] = xu[pos_]  # 反弹两次的话碰壁

        # UBOUND
        pos1 = XMutationTmp > xu
        XMutationTmp[pos1] = 2 * xu[pos1] - XMutationTmp[pos1]

        pos2 = XMutationTmp < xl
        pos_ = np.logical_and(pos1, pos2)
        XMutationTmp[pos_] = xl[pos_]  # 反弹两次的话碰壁

        return XMutationTmp

    def crossover(self, XMutationTmp):
        m, n = self.pop_x.shape

        mask = np.random.rand(m, n) > self.CR

        jrand = np.zeros([m, n]).astype(np.bool)
        for index in range(m):
            choosen = np.around(np.random.rand() * (n - 1)).astype(np.int32)
            jrand[index,choosen] = True

        mask[jrand] = False

        XCorssOverTmp = XMutationTmp
        XCorssOverTmp[mask] = self.pop_x[mask]

        return XCorssOverTmp

    def selection(self, XCorssOverTmp):
        m, n = self.pop_x.shape



        fitnessCrossOverVal = self.calFitness(XCorssOverTmp)
        fitnessVal=self.calFitness(self.pop_x)

        mask1=fitnessCrossOverVal<fitnessVal  #新生儿优秀
        mask2=fitnessCrossOverVal>=fitnessVal #父代优秀


        #子代优秀，替代父代
        self.pop_x[mask1,:]=XCorssOverTmp[mask1,:]
        self.pop_value[mask1]=fitnessCrossOverVal[mask1]

        #父代优秀，回退F,CR
        self.F[mask2]=self.oldF[mask2]
        self.CR[mask2]=self.oldCR[mask2]



    def saveBest(self, fitnessVal):
        m = fitnessVal.shape[0]
        tmp = 0
        for i in range(1, m):
            if (fitnessVal[tmp] > fitnessVal[i]):
                tmp = i

        self.bestvalue=fitnessVal[tmp]
        self.bestx=self.pop_x[tmp,:]
       # print(fitnessVal[tmp])


    def checkConvergence(self):#检测种群是否收敛到一定程度，爆炸扩散   收敛0.5


        temp=np.abs(self.pop_value-self.bestvalue)
        mask=temp<=1e-16
        ifconver=sum(mask)>=self.NP*0.5

        if(ifconver):
            print("conver:",self.bestvalue,"iter:",self.gen)
            self.pop_x = self.LBOUND + np.random.rand(self.NP, self.dim) * (self.UBOUND - self.LBOUND)  # NP*DIM
            self.pop_value = self.calFitness(self.pop_x)  # NP
            self.F = np.full([self.NP, 1], 0.5)
            self.CR = np.full([self.NP, 1], 0.9)
            self.oldF = self.F
            self.oldCR = self.CR

            self.pop_x[0,:]=self.bestx
            self.pop_value[0]=self.bestvalue



    def startTrain(self):
        self.gen = 0

        while self.gen <= self.MAXGEN:


            self.Fold=self.F
            self.CRold=self.CR

            #print(self.F)

            IF=np.random.rand(self.NP,1)<self.tau1
            ICR = np.random.rand(self.NP,1) < self.tau2



            self.F[IF]= 0.0 + 1.5 * np.random.rand(np.sum(IF))
            self.CR[ICR] = 0.0 + 1.0 * np.random.rand(np.sum(ICR))

            XMutationTmp = self.mutation()
            XCorssOverTmp = self.crossover(XMutationTmp)

            self.selection(XCorssOverTmp)

            self.saveBest(self.pop_value)
            self.gen += 1
            if self.gen%100==0:
                self.checkConvergence()



    def getBest(self):
        return self.bestx,self.bestvalue


class JDE1(JDE):#tanh
    def __init__(self,maxgen=100,xs=[],es=[],dim=1):
        super(JDE1,self).__init__(maxgen,xs,es,dim)



    def calFitness(self, X):  #NP*DIM

        bias = X[:, -1].reshape(X.shape[0], 1)
        X = X[:, :-1]
        #bias=1

        # vs=self.sigmoid(X.dot(self.xs)+bias)
        # vs=self.tanh(X.dot(self.xs)+bias)
        # vs = self.relu(X.dot(self.xs) + bias)
        vs = self.fun1(X.dot(self.xs) + bias)

        v_term = vs - np.mean(vs, axis=1).reshape(vs.shape[0],1)  # 199*1 vs-vm
        e_term = self.es - np.mean(self.es, axis=0)  # 199*2  es-em
        t = v_term
        corr = t.dot(e_term)

        fitness=-1.0 * np.abs(corr[:,0])
        return fitness


class JDE2(JDE):
    def __init__(self, maxgen=100, xs=[], es=[], dim=1):
        super(JDE2, self).__init__(maxgen, xs, es, dim)

    def calFitness(self, X):  #NP*DIM

        bias = X[:, -1].reshape(X.shape[0], 1)
        X = X[:, :-1]
        #bias=1

        # vs=self.sigmoid(X.dot(self.xs)+bias)
        # vs=self.tanh(X.dot(self.xs)+bias)
        # vs = self.relu(X.dot(self.xs) + bias)
        vs = self.fun1(X.dot(self.xs) + bias)

        v_term = vs - np.mean(vs, axis=1).reshape(vs.shape[0],1)  # 199*1 vs-vm
        e_term = self.es - np.mean(self.es, axis=0)  # 199*2  es-em
        t = v_term
        corr = t.dot(e_term)

        fitness=-1.0 *np.abs(corr[:,1])
        return fitness

class JDE3(JDE):
    def __init__(self, maxgen=100, xs=[], es=[], dim=1):
        super(JDE3, self).__init__(maxgen, xs, es, dim)

    def calFitness(self, X):  #NP*DIM

        #bias = X[:, -1].reshape(X.shape[0], 1)
        #X = X[:, :-1]
        bias=1

        # vs=self.sigmoid(X.dot(self.xs)+bias)
        # vs=self.tanh(X.dot(self.xs)+bias)
        # vs = self.relu(X.dot(self.xs) + bias)
        vs = self.fun1(X.dot(self.xs) + bias)

        v_term = vs - np.mean(vs, axis=1).reshape(vs.shape[0],1)  # 199*1 vs-vm
        e_term = self.es - np.mean(self.es, axis=0)  # 199*2  es-em
        t = v_term
        corr = t.dot(e_term)

        fitness=-1.0 *np.abs(corr[:,2])
        return fitness

class JDE4(JDE):
    def __init__(self, maxgen=100, xs=[], es=[], dim=1):
        super(JDE4, self).__init__(maxgen, xs, es, dim)

    def calFitness(self, X):  #NP*DIM

        #bias = X[:, -1].reshape(X.shape[0], 1)
        #X = X[:, :-1]
        bias=1

        # vs=self.sigmoid(X.dot(self.xs)+bias)
        # vs=self.tanh(X.dot(self.xs)+bias)
        # vs = self.relu(X.dot(self.xs) + bias)
        vs = self.fun1(X.dot(self.xs) + bias)

        v_term = vs - np.mean(vs, axis=1).reshape(vs.shape[0],1)  # 199*1 vs-vm
        e_term = self.es - np.mean(self.es, axis=0)  # 199*2  es-em
        t = v_term
        corr = t.dot(e_term)

        fitness=-1.0 *np.abs(corr[:,3])
        return fitness