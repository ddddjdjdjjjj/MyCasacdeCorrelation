import time
from JDE import JDE,JDE1,JDE2,JDE3,JDE4,JDE5,JDE6,JDE7,JDE8,JDE9,JDE10

import numpy as np
#from utils import Utils


class MOEAD2():

    optim_fun=0
    name = 'problem1'

    Pop_size = -1
    max_gen =50
    T_size = 3

    MU=0.2
    CR=0.8

    EP_X_ID = []# 支配前沿ID
    EP_X_FV = [] # 支配前沿 的 函数值

    sindex=[] #边界索引
    Pop = [] # 种群
    Pop_FV = []    # 种群计算出的函数值

    W = [] # 权重
    W_Bi_T = [] # 权重的T个邻居 index

    Z = [] # 理想点

    gen = 0# 当前迭代代数

    need_dynamic = False# 是否动态展示
    draw_w = True # 是否画出权重图
    now_y = [] # 用于绘图：当前进化种群中，哪个，被，正在 进化。draw_w=true的时候可见

    # draw_w=True

    def __init__(self,maxepoches,dim,m,H,xs,es,hiddenus):
        self.max_gen=maxepoches
        self.dim = dim
        #self.dim = dim+1
        self.m=m
        self.H = H

        self.low=-10  #要手动调节JDE的
        self.up=10

        self.xs0=xs
        #self.optim_fun=optim_fun
        self.xs=xs.T   #192*2
        self.es=es   #192*2

        self.hiddenus=hiddenus


        self.LBOUND = np.full(( self.dim), self.low)
        self.UBOUND = np.full(( self.dim), self.up)
        self.Init_data()

    def Init_data(self):
        # 加载权重

        self.Gen_W()
        print("gen_wb")
        # 计算每个权重Wi的T个邻居
        self.cpt_W_Bi_T()
        # 创建种群
        #self.GA_DE_Utils.Creat_Pop(self)
        self.Creat_Pop()
        # 初始化Z集，最小问题0,0
        self.cpt_Z()

        #self.show()

    def Gen_W(self):        #生成平均权重
        mv = self.Mean_vector(self.H, self.m)
        self.W ,self.sindex=mv.generate()
      #  print('created namda')
     #   print(self.W)
        self.Pop_size = self.W.shape[0]
       # print("pop_size:",self.Pop_size)

    # 计算T个邻居
    def cpt_W_Bi_T(self):
        for bi in range(self.W.shape[0]):
            Bi = self.W[bi]
            DIS = np.sum((self.W - Bi) ** 2, axis=1)
            B_T = np.argsort(DIS)
            # 第0个是自己（距离永远最小）
            B_T = B_T[1:self.T_size + 1]
            self.W_Bi_T.append(B_T)
        #print(self.W_Bi_T)

    def Creat_Pop(self):
        # 创建moead.Pop_size个种群
        Pop = []
        Pop_FV = []

        while len(Pop) != self.Pop_size:
            X =self.LBOUND + (self.UBOUND - self.LBOUND) * np.random.rand(self.dim)
            Pop.append(X)
            Pop_FV.append(self.calFitness(X))
        self.Pop, self.Pop_FV = Pop, Pop_FV
        #return Pop, Pop_FV

    def cpt_Z(self):
        Z = self.Pop_FV[0];
        for fv in self.Pop_FV:
            for index in range(self.m):

                if(Z[index]>fv[index]):
                    Z[index]=fv[index]
        self.Z = Z

        #self.Z=[-200,-200,-200,-200,-200,-200,-200,-200,-200,-200]
     #   print("Z:",self.Z)
        return Z


    #def calFitness(self, x):
    #    return Func(x)

    def fun1(self,os):

       # y = self.slope2 * np.minimum(os,0) + self.slope1 * np.maximum(os,0)
        mask = os > 0
        os[:, :] = self.low
        os[mask] = self.up
        return os

    def calFitness(self, x):  #DIM

       # bias=x[-1]  #非固定bias
       # x=x[:-1]
        bias=1
        x=x.reshape(-1,x.shape[0])
        vs=self.fun1(x.dot(self.xs)+bias)

      #  vs = np.tanh(x.dot(self.xs) + bias)
       # vs=X.dot(self.xs)+bias
      #  vs=(abs(vs) + vs) / 2

        y=[]

        v_term = vs - np.mean(vs, axis=1).reshape(vs.shape[0], 1)  # 1*192 vs-vm
        e_term = self.es - np.mean(self.es, axis=0)  # 192*2  es-em
        for index in range(self.m):


          t = v_term
          corr = t.dot(e_term[:,index])
          fitness=-1.0* np.abs(corr[0])#np.abs(corr)
          y.append(fitness)

        return y

   # def show(self):
        #print("show")
    #    if self.draw_w:
     #       Utils.draw_W(self)
     #   Utils.draw_MOEAD_Pareto(self, self.name + "num:" + str(self.max_gen) + "")
     #   Utils.show()

    def startTrain(self):


        iter = 0

        maxepoches=self.max_gen

        print("jde epoche:",self.max_gen)

        self.optimizer = JDE1(maxepoches, self.xs0, self.es, self.dim)
        self.optimizer.startTrain()
        bestx1, loss1 = self.optimizer.getBest()
        iter = iter + self.optimizer.getFinalGen()

        self.optimizer = JDE2(maxepoches, self.xs0, self.es, self.dim)
        self.optimizer.startTrain()
        bestx2, loss2 = self.optimizer.getBest()
        iter = iter + self.optimizer.getFinalGen()

        self.optimizer = JDE3(maxepoches, self.xs0, self.es, self.dim)
        self.optimizer.startTrain()
        bestx3, loss3 = self.optimizer.getBest()
        iter = iter + self.optimizer.getFinalGen()

        self.optimizer = JDE4(maxepoches, self.xs0, self.es, self.dim)
        self.optimizer.startTrain()
        bestx4, loss4 = self.optimizer.getBest()
        iter = iter + self.optimizer.getFinalGen()

        bestx = [bestx4,bestx3,bestx2,bestx1]
        loss=[loss4,loss3,loss2,loss1]

        self.max_gen=1
        #print("jde loss",loss)
        for index in range(len(self.sindex)):
            self.Pop[self.sindex[index]] =bestx[index]
            self.Pop_FV[self.sindex[index]]=self.calFitness(bestx[index])

            print(self.Pop_FV[self.sindex[index]])

        # EP_X_ID：支配前沿个体解，的ID。在上面数组：Pop，中的序号
        # envolution开始进化
        EP_X_ID = self.envolution()
        #EP_X_ID=self.GA_DE_Utils.envolution(self)
        #print('你拿以下序号到上面数组：Pop中找到对应个体，就是多目标优化的函数的解集啦!')
        #print("支配前沿个体解，的ID（在上面数组：Pop，中的序号）：", EP_X_ID)
        #for id in EP_X_ID:
          #  print(self.Pop_FV[id])

       # self.show()

    def Tchebycheff_dist(self,w, f, z):
        # 计算切比雪夫距离
        return w * abs(f - z)

    def cpt_tchbycheff(self, idx, F_X):
        # idx：X在种群中的位置
        # 计算X的切比雪夫距离（与理想点Z的）
        max = self.Z[0]
        ri = self.W[idx]
        for i in range(self.m):
            fi = self.Tchebycheff_dist(ri[i], F_X[i], self.Z[i])
            if fi > max:
                max = fi
        return max


    def mutate2(self, y1):
        mask = np.random.rand(self.dim) < self.MU
        temp = self.LBOUND + np.random.rand(self.dim) * (self.UBOUND - self.LBOUND)
        y1[mask] = temp[mask]
        return y1

    def crossover2(self, y1, y2):
        mask1=np.random.rand(self.dim)<self.CR
        mask2=~mask1
        y1[mask1]=y2[mask1]
        y2[mask2]=y1[mask2]
        return y1,y2


    def mutate(self, y1):#该策略只适用于ZDT
        # 突变个体的策略2
        dj = 0
        uj = np.random.rand()
        if uj < self.MU:
            dj = (2 * uj) ** (1 / 6) - 1
        else:
            dj = 1 - 2 * (1 - uj) ** (1 / 6)
        y1 = y1 + dj
        mask=y1>self.UBOUND
        y1[mask] = self.UBOUND[mask]
        mask = y1 < self.LBOUND
        y1[mask] = self.LBOUND[mask]
        return y1

    def crossover(self, y1, y2):#该策略只适用于ZDT
        # 交叉个体的策略2
        var_num = self.dim
        yj = 0
        uj = np.random.rand()
        if uj < self.CR:
            yj = (2 * uj) ** (1 / 3)
        else:
            yj = (1 / (2 * (1 - uj))) ** (1 / 3)
        y1 = 0.5 * (1 + yj) * y1 + (1 - yj) * y2
        y2 = 0.5 * (1 - yj) * y1 + (1 + yj) * y2
        mask = y1 > self.UBOUND
        y1[mask] = self.UBOUND[mask]
        mask = y1 < self.LBOUND
        y1[mask] = self.LBOUND[mask]
        mask = y2 > self.UBOUND
        y2[mask] = self.UBOUND[mask]
        mask = y2 < self.LBOUND
        y2[mask] = self.LBOUND[mask]
        return y1, y2

    def cross_mutation(self, p1, p2):
        y1 = np.copy(p1)
        y2 = np.copy(p2)
        c_rate = 1
        m_rate = 0.5
        if np.random.rand() < c_rate:
            y1, y2 = self.crossover2( y1, y2)
        if np.random.rand() < m_rate:
            y1 = self.mutate2( y1)
            y2 = self.mutate2( y2)
        return y1, y2

    def EO(self, wi, p1): #在每一个维度上尝试一个微小突变，若有好的改变则返回
        m = p1.shape[0]
        tp_best = np.copy(p1)   #复制
        F_tp_best=self.calFitness(tp_best)
        qbxf_tp = self.cpt_tchbycheff(wi, F_tp_best)  #计算聚合值
        Up = np.sqrt(self.UBOUND - self.LBOUND) / 2 #0.5
        h = 0
        for i in range(m):
            if h == 1:
                return tp_best
            temp_best = np.copy(p1)      #复制
            rd = np.random.normal(0, Up[i], 1)  #从正太高斯分布取样1个
            temp_best[i] = temp_best[i] + rd    #随机+rd
            temp_best[temp_best > self.UBOUND[i]] = self.UBOUND[i]
            temp_best[temp_best < self.LBOUND[i]] = self.LBOUND[i]

            F_tempbest=self.calFitness(temp_best)
            qbxf_te = self.cpt_tchbycheff( wi, F_tempbest)
            if qbxf_te < qbxf_tp:
                h = 1
                qbxf_tp = qbxf_te
                tp_best[:] = temp_best[:]
        return tp_best

    def generate_next(self, gen, wi, p0, p1, p2,F_p0,F_p1,F_p2):
        # 进化下一代个体。基于自身Xi+邻居中随机选择的2个Xk，Xl 还考虑gen 去进化下一代
        #F_p0=self.calFitness(p0)
       # F_p1 = self.calFitness(p1)
       # F_p2 = self.calFitness(p2)
        qbxf_p0 = self.cpt_tchbycheff( wi, F_p0)
        qbxf_p1 = self.cpt_tchbycheff( wi, F_p1)
        qbxf_p2 = self.cpt_tchbycheff( wi, F_p2)

        qbxf = np.array([qbxf_p0, qbxf_p1, qbxf_p2])
        best = np.argmin(qbxf)
        # 选中切比雪夫距离最小（最好的）个体
        Y1 = [p0, p1, p2][best]
        # 需要深拷贝成独立的一份
        n_p0, n_p1, n_p2 = np.copy(p0), np.copy(p1), np.copy(p2)

        #if gen % 10 == 0:
            # 每格10代，有小概率进行EO优化（效果好，但是复杂度高）
         #   if np.random.rand() < 0.1:
         #       n_p0 = self.EO( wi, n_p0)
        # 交叉
        n_p0, n_p1 = self.cross_mutation( n_p0, n_p1)
        # n_p0, n_p1 = moead.cross_mutation(n_p0, n_p1)
        n_p1, n_p2 = self.cross_mutation( n_p1, n_p2)
        # n_p1, n_p2 = moead.cross_mutation( n_p1, n_p2)
        # 交叉后的切比雪夫距离
        F_np0=self.calFitness(n_p0)
        F_np1=self.calFitness(n_p1)
        F_np2=self.calFitness(n_p2)
        qbxf_np0 = self.cpt_tchbycheff( wi, F_np0)
        qbxf_np1 = self.cpt_tchbycheff( wi, F_np1)
        qbxf_np2 = self.cpt_tchbycheff( wi, F_np2)

        qbxf = np.array([qbxf_p0, qbxf_p1, qbxf_p2, qbxf_np0, qbxf_np1, qbxf_np2])
        best = np.argmin(qbxf)
        # 选中切比雪夫距离最小（最好的）个体
        Y2 = [p0, p1, p2, n_p0, n_p1, n_p2][best]

        # 随机选中目标中的某一个目标进行判断，目标太多，不要贪心，随机选一个目标就好
        fm = np.random.randint(0, self.m)
        # 如果是极小化目标求解，以0。5的概率进行更详细的判断。（返回最优解策略不能太死板，否则容易陷入局部最优）
        if  np.random.rand() < 0.5:
            FY1 = self.calFitness(Y1)
            FY2 = self.calFitness(Y2)
            # 如果随机选的这个目标Y2更好，就返回Y2的
            if FY2[fm] < FY1[fm]:
                return Y2
            else:
                return Y1
        return Y2

    def update_BTX(self, P_B, Y,F_Y):
        # 根据Y更新P_B集内邻居
        for j in P_B:
            d_x = self.cpt_tchbycheff( j, self.Pop_FV[j])
            d_y = self.cpt_tchbycheff( j, F_Y)
            if d_y <= d_x:
                # d_y 的切比雪夫距离更小
                self.Pop[j] = Y[:]
                F_Y = self.calFitness(Y)
                self.Pop_FV[j] = F_Y
                self.update_EP_By_ID(j, F_Y)

    def update_EP_By_ID(self, id, F_Y):
        # 如果id存在，则更新其对应函数集合的值
        if id in self.EP_X_ID:
            # 拿到所在位置
            position_pi = self.EP_X_ID.index(id)
            # 更新函数值
            self.EP_X_FV[position_pi][:] = F_Y[:]

    def update_Z(self, Y):
        # 根据Y更新Z坐标。。ps:实验结论：如果你知道你的目标，比如是极小化了，且理想极小值(假设2目标)是[0,0]，
        # 那你就一开始的时候就写死moead.Z=[0,0]把
      #  dz = np.random.rand()
        F_y = self.calFitness(Y)
        for j in range(self.m):
            if self.Z[j] > F_y[j]:
                self.Z[j] = F_y[j] #- dz
       # print("Z:",self.Z)

    def is_dominate(self, F_X, F_Y):
        # 判断F_X是否支配F_Y
        if type(F_Y) != list:
            F_X = self.calFitness(F_X)
            F_Y = self.calFitness(F_Y)
        i = 0
        for xv, yv in zip(F_X, F_Y):
            if xv < yv:
                 i = i + 1
            if xv > yv:
                return False
        if i != 0:
            return True
        return False


    def update_EP_By_Y(self, id_Y):
        # 根据Y更新前沿
        # 根据Y更新EP
        i = 0
        # 拿到id_Y的函数值
        F_Y = self.Pop_FV[id_Y]
        # 需要被删除的集合
        Delet_set = []
        # 支配前沿集合，的数量
        Len = len(self.EP_X_FV)
        for pi in range(Len):
            # F_Y是否支配pi号个体，支配？哪pi就完了，被剔除。。
            if self.is_dominate( F_Y, self.EP_X_FV[pi]):
                # 列入被删除的集合
                Delet_set.append(pi)
                break
            if i != 0:
                break
            if self.is_dominate(self.EP_X_FV[pi], F_Y):
                # 它有被别人支配！！记下来能支配它的个数
                i += 1
        # 新的支配前沿的ID集合，种群个体ID，
        new_EP_X_ID = []
        # 新的支配前沿集合的函数值
        new_EP_X_FV = []
        for save_id in range(Len):
            if save_id not in Delet_set:
                # 不需要被删除，那就保存
                new_EP_X_ID.append(self.EP_X_ID[save_id])
                new_EP_X_FV.append(self.EP_X_FV[save_id])
        # 更新上面计算好的新的支配前沿
        self.EP_X_ID = new_EP_X_ID
        self.EP_X_FV = new_EP_X_FV
        # 如果i==0，意味着没人支配id_Y
        # 没人支配id_Y？太好了，加进支配前沿呗
        if i == 0:
            # 不在里面直接加新成员
            if id_Y not in self.EP_X_ID:
                self.EP_X_ID.append(id_Y)
                self.EP_X_FV.append(F_Y)
            else:
                # 本来就在里面的，更新它
                idy = self.EP_X_ID.index(id_Y)
                self.EP_X_FV[idy] = F_Y[:]
        # over
        return self.EP_X_ID, self.EP_X_FV

    def getFinalGen(self):
        return self.gen

    def getBest(self):
        best=[]
        loss=[]

        for index in self.sindex:
            best.append(self.Pop[index])
            loss.append(self.Pop_FV[index])

        for i in range(self.hiddenus-len(self.sindex)):
            index=np.around(np.random.rand()*(self.Pop_size-1)).astype(np.int32)
            best.append(self.Pop[index])
            loss.append(self.Pop_FV[index])
        return best, loss

        #best1=self.Pop[0]
        #best2=self.Pop[-1]
        #loss1=self.Pop_FV[0]
        #loss2=self.Pop_FV[-1]
        #return [best1,best2],[loss1,loss2]



    def envolution(self):
        for gen in range(self.max_gen):
            self.gen = gen
            for pi, p in enumerate(self.Pop):
                # 第pi号个体的邻居集
                Bi = self.W_Bi_T[pi]
                # 随机选一个T内的数，作为pi的邻居。
                # （邻居你可以想象成：物种，你总不能人狗杂交吧？所以个体pi只能与他的T个前后的邻居权重，管的个体杂交进化）
                # 比如：T=2，权重(0.1,0.9)约束的个体的邻居是：权重(0,1)、(0.2,0.8)约束的个体。永远固定不变
                k = np.random.randint(self.T_size)
                l = np.random.randint(self.T_size)
                # 随机从邻居内选2个个体，产生新解
                ik = Bi[k]
                il = Bi[l]
                Xi = self.Pop[pi]
                Xk = self.Pop[ik]
                Xl = self.Pop[il]
                # 进化下一代个体。基于自身Xi+邻居中随机选择的2个Xk，Xl 还考虑gen 去进化下一代
                Y = self.generate_next(gen, pi, Xi, Xk, Xl,self.Pop_FV[pi],self.Pop_FV[ik],self.Pop_FV[il])
                F_Y = self.calFitness(Y)
                #Y = generate_next(self,gen, pi, Xi, Xk, Xl)


                cbxf_i = self.cpt_tchbycheff(pi, self.Pop_FV[pi])

                cbxf_y = self.cpt_tchbycheff( pi, F_Y)
                # 不能随随便便一点点好就要了（自己的策略设计）。超过d才更新
                d = 0.001
                # 开始比较是否进化出了更好的下一代，这样才保留
                if cbxf_y < cbxf_i:
                    # 用于绘图：当前进化种群中，哪个，被，正在 进化。draw_w=true的时候可见
                    self.now_y = pi

                    #self.update_EP_By_ID( pi, F_Y)

                    # 都进化出更好切比雪夫下一代了，有可能有更好的理想点，尝试更新理想点
                    self.update_Z(Y)
                    if abs(cbxf_y - cbxf_i) > d:
                        # 超过d才更新。更新支配前沿。红色点那些
                        self.update_EP_By_Y(pi)
                self.update_BTX( Bi, Y,F_Y)
            # 是否需要动态展示
            #print(self.Pop_FV)
           # print('迭代 %s,支配前沿个体数量len(moead.EP_X_ID) :%s,moead.Z:%s' % (gen, len(moead.EP_X_ID), moead.Z))
            #if gen%10==0:
             #   print("iter:",gen);
        return self.EP_X_ID

    class Mean_vector():
        # 对m维空间，目标方向个数H
        def __init__(self, H=5, m=3):
            self.H = H
            self.m = m
            self.stepsize = 1 / H

        def perm(self, sequence):
            # ！！！ 序列全排列，且无重复
            l = sequence
            if (len(l) <= 1):
                return [l]
            r = []
            for i in range(len(l)):
                if i != 0 and sequence[i - 1] == sequence[i]:
                    continue
                else:
                    s = l[:i] + l[i + 1:]
                    p = self.perm(s)
                    for x in p:
                        r.append(l[i:i + 1] + x)
            return r

        def get_mean_vectors2(self):  # m2 H4
            H = self.H
            m = self.m
            sequence = []
            for ii in range(H):
                sequence.append(0)
            for jj in range(m - 1):
                sequence.append(1)  # 0 0 0 0 1 1
            ws = []

            pe_seq = self.perm(sequence)
            for sq in pe_seq:
                s = -1
                weight = []
                for i in range(len(sq)):
                    if sq[i] == 1:
                        w = i - s
                        w = (w - 1) / H
                        s = i
                        weight.append(w)
                nw = H + m - 1 - s
                nw = (nw - 1) / H
                weight.append(nw)
             #   print(weight)
                if weight not in ws:
                    ws.append(weight)
            return ws

        def get_mean_vectors(self):#H是分割  m是维度
            all=pow(self.H+1,self.m-1)
            ws=[]
            for i in range(all):
                weight = np.zeros(self.m)
                total = 0
                for g in range(self.m):
                    weight[g] = int(i/pow(self.H+1,self.m-1-g-1))%(self.H + 1)
                    total = total + weight[g]

                if total <= self.H:
                    weight[-1] = self.H - total
                    for g in range(self.m):
                        weight[g] = weight[g] / self.H
                    ws.append(weight)
            #print(ws)

            sindex=[]
            sweight=[]
            for i in range(self.m):
                weight=np.zeros(self.m)
                weight[self.m-1-i]=1.0
                sweight.append(weight)

            index2=0
           # print(ws[0],sweight[0])
            for index in range(len(ws)):
                if sum(ws[index]==sweight[index2])==self.m:
                    sindex.append(index)
                    index2=index2+1
                    if(index2>=len(sweight)):
                        break
            return ws,sindex









        def generate(self):
            ws,sindex = self.get_mean_vectors()  #[0,0,0,1]  [0,0,1,0] ...

            return np.array(ws),sindex


def Func(X):
    f1 = F1(X)
    gx = g(X)
    f2 = F2(gx, X)
    return [f1, f2]

def F1(X):
    return X[0]
def F2(gx, X):
    x = X[0]
    f2 = gx * (1 - np.sqrt(x / gx))
    return f2
def g(X):
    g = 1 + 9 * (np.sum(X[1:], axis=0) / (X.shape[0] - 1))
    return g

if __name__ == '__main__':
    # np.random.seed(1)
    moead = MOEAD2(50,30,2,100,None,None)
    moead.startTrain()
