import matplotlib.pyplot as plt
import numpy as np

lines=open("./record/0/hidden_best-" + str(1) + ".txt", "r")

lists=[]
for line in lines:
  print(line)
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

data3=[]   #os3
avg3=0
for d in range(len(lists[0])):
   data3.append(lists[0][d][2])
   avg3 = avg3 + lists[0][d][2]
avg3 = avg3 / len(data3)

data4=[]   #os4
avg4=0
for d in range(len(lists[0])):
   data4.append(lists[0][d][3])
   avg4 = avg4 + lists[0][d][3]
avg4 = avg4 / len(data4)

data5 = []
avg5=0
for d in range(len(lists[1])):
   data5.append(lists[1][d][0])
   avg5 = avg5 + lists[1][d][0]
avg5 = avg5 / len(data5)

data6 = []
avg6=0
for d in range(len(lists[1])):
   data6.append(lists[1][d][1])
   avg6 = avg6 + lists[1][d][1]
avg6 = avg6 / len(data6)


data7 = []
avg7=0
for d in range(len(lists[1])):
   data7.append(lists[1][d][2])
   avg7 = avg6 + lists[1][d][2]
avg7 = avg7 / len(data7)


data8 = []
avg8=0
for d in range(len(lists[1])):
   data8.append(lists[1][d][3])
   avg8 = avg8 + lists[1][d][3]
avg8 = avg8 / len(data8)


loss1=[]
for d in range(len(lists[2])):
   loss1.append(lists[2][d])


data9 = []
avg9=0
for d in range(len(lists[3])):
   data9.append(lists[3][d][0])
   avg9 = avg9 + lists[3][d][0]
avg9 = avg9 / len(data9)

loss2=lists[4]
#for d in range(len(lists[4])):
#   loss1.append(lists[4][d])

#print(data1)
#print(data2)
#print(data3)
#print(data4)
#print(data5)
#print(data6)
#print(avg1,avg2,avg3,avg4,avg5)

#print("loss:",lists[2])
print('loss1:',loss1)
for loss in loss1:
   sum=0
   for l in loss:
     sum=sum+l
   print(sum)
print('loss2',loss2)
sum=0
for loss in loss2[0]:
   sum=sum+abs(loss)
print('total loss:',sum)

#plt.hist(x = data)
plt.figure(dpi=100,figsize=(16,8))
plt.subplot(422) #将图分割未几行几列，当前为第几个，从左到右从上到下
plt.plot(data1)
#plt.plot([0,len(data1)],[avg1,avg1],linewidth=6,color='r')

plt.subplot(424)
plt.plot(data2)
#plt.plot([0,len(data2)],[avg2,avg2],linewidth=6,color='r')

plt.subplot(426)
plt.plot(data3)
#plt.plot([0,len(data3)],[avg3,avg3],linewidth=6,color='r')

plt.subplot(428)
plt.plot(data4)
#plt.plot([0,len(data4)],[avg4,avg4],linewidth=6,color='r')

plt.show()


plt.figure(dpi=100,figsize=(16,8))
plt.subplot(421)
plt.plot(data9)
#plt.plot([0,len(data9)],[avg9,avg9],linewidth=6,color='r')
plt.show()

plt.figure(dpi=100,figsize=(16,8))
plt.subplot(421)
plt.plot(data5)
#plt.plot([0,len(data5)],[avg5,avg5],linewidth=6,color='r')

plt.subplot(423)
plt.plot(data6)
#plt.plot([0,len(data6)],[avg6,avg6],linewidth=6,color='r')


plt.subplot(425)
plt.plot(data7)
#plt.plot([0,len(data7)],[avg7,avg7],linewidth=6,color='r')

plt.subplot(427)
plt.plot(data8)
#plt.plot([0,len(data8)],[avg8,avg8],linewidth=6,color='r')

#plt.savefig('./4-4 error.png')
plt.show()
