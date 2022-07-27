from main import main



acc1list=[]
acc2list=[]
acc3list=[]
hiddenlist=[]

times=10
for i in range(times):
    acc1list0,acc2list0,acc3list0,hiddenlist0=main(i)

    acc1list.append(acc1list0)
    acc2list.append(acc2list0)
    acc3list.append(acc3list0)
    hiddenlist.append(hiddenlist0)





sum = 0
num=0
for i in range(times):
    for hiddennums in hiddenlist[i]:
        sum = sum+hiddennums
        num=num+1

avghiddenum=sum /num
print("avghiddenums:", avghiddenum)

with open("./record/train_acc.txt", "w") as f:
    f.write(str(acc1list) + "\n")

with open("./record/test_acc.txt", "w") as f:
    f.write(str(acc2list) + "\n")

with open("./record/all_acc.txt", "w") as f:
    f.write(str(acc3list) + "\n")

with open("./record/hiddennums.txt", "w") as f:
    f.write(str(hiddenlist) + "\n")
