import matplotlib.pyplot as plt
import numpy as np

lines=open("./data.txt", "r")
lists=[]
for line in lines:
  lists.append(eval(line))


print(lists[0])#192*2
print(len(lists))

data1=lists[0]
xs=np.array(data1)
xs=xs.T

data2=lists[1]
es=np.array(data2)


data3 = lists[2]
x=np.array(data3)
x=x.reshape(-1,x.shape[0])
print(xs.shape)
print(es.shape)
print(x.shape)


def calFitness1(X):  # NP*DIM

    bias = 1
    vs = np.tanh(X.dot(xs) + bias)
    v_term = vs - np.mean(vs, axis=1).reshape(vs.shape[0], 1)  # 1*199 vs-vm
    e_term = es - np.mean(es, axis=0)  # 199*2  es-em
    t = v_term
    corr = t.dot(e_term)
    fitness = -1.0 * np.sum(np.abs(corr), axis=1)
    return fitness

f1=calFitness1(x)
print(f1)


def calFitness( x):  # DIM


    bias = 1
   # x = x.reshape(-1, x.shape[0])
    vs = np.tanh(x.dot(xs) + bias)
    y = []
    v_term = vs - np.mean(vs, axis=1).reshape(vs.shape[0], 1)  # 1*192 vs-vm
    e_term = es - np.mean(es, axis=0)  # 192*2  es-em
    for index in range(2):
        t = v_term
        corr = t.dot(e_term[:, index])
        fitness = -1.0 * corr[0]  # np.abs(corr)
        y.append(fitness)

    return y

f2=calFitness(x)
print(f2)
print(abs(f2[0])+abs(f2[1]))

'''
#plt.hist(x = data)
plt.figure(dpi=100,figsize=(8,8))
plt.subplot(311)
plt.plot(data1)
plt.plot([0,len(data1)],[avg1,avg1],linewidth=6,color='r')



plt.subplot(312)
plt.plot(data2)
plt.plot([0,len(data2)],[avg2,avg2],linewidth=6,color='r')

plt.subplot(313)
plt.plot(data3)
plt.plot([0,len(data3)],[avg3,avg3],linewidth=6,color='r')

plt.show()
'''