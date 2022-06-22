from main import main



iters=[]
hiddennums=[]
for i in range(1):

    iter,hiddennum=main(i)

    iters.append(iter)
    hiddennums.append(hiddennum)


sum=0
for i in range(len(iters)):
    sum+=iters[i]

avgiter=sum/(len(iters))
print("avgiter:",avgiter)

sum = 0
for i in range(len(hiddennums)):
    sum += hiddennums[i]

avghiddenum=sum / (len(hiddennums))
print("avghiddenums:", avghiddenum)

with open("./record/total.txt", "w") as f:
    data1 = iters
    data2= hiddennums
    f.write(str(data1) + "\n")
    f.write(str(data2) + "\n")
    f.write(str(avgiter) + "\n")
    f.write(str(avghiddenum) + "\n")