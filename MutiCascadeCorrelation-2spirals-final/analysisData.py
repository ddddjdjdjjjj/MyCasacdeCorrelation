import matplotlib.pyplot as plt
import numpy as np

lines=open("./record/0/hidden_best-" + str(1) + ".txt", "r")

lists=[]
for line in lines:
  lists.append(eval(line))


print(lists[0])#192*2
print(len(lists[0]))

data1=[]  #os1
avg1=0
for d in range(len(lists[0])):
   data1.append(lists[0][d][0])
   avg1=avg1+lists[0][d][0]
avg1=avg1/len(data1)

data2=[]   #os2
avg2=0
for d in range(len(lists[0])):
   data2.append(lists[0][d][1])
   avg2 = avg2 + lists[0][d][1]
avg2 = avg2 / len(data2)



data5 = []
avg5=0
for d in range(len(lists[1])):
   data5.append(lists[1][d][0])
   avg5 = avg5 + lists[1][d][0]
avg5 = avg5 / len(data5)



print(data1)
print(data2)

print(data5)

print(avg1,avg2,avg5)

print("loss:",lists[2])

#plt.hist(x = data)
plt.figure(dpi=100,figsize=(16,8))
plt.subplot(421) #将图分割未几行几列，当前为第几个，从左到右从上到下
plt.plot(data1)
plt.plot([0,len(data1)],[avg1,avg1],linewidth=6,color='r')

plt.subplot(422)
plt.plot(data2)
plt.plot([0,len(data2)],[avg2,avg2],linewidth=6,color='r')


plt.subplot(425)
plt.plot(data5)
plt.plot([0,len(data5)],[avg5,avg5],linewidth=6,color='r')

plt.show()


