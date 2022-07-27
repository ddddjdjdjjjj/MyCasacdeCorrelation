from cascadenet import CascadeNet

from DatasetIris import *
from torch.utils.data import DataLoader
import torch
import time
import numpy as np
import matplotlib


import os






def main(testindex):
  # n_feature=4
  # n_output=3

  acc1list=[]
  acc2list=[]
  acc3list = []
  hiddenlist=[]

  n_feature = 4
  n_output = 3

  maxioepoches = 500
  batch_size = 15
  lr = 0.1
  alpha = 0.9998

  save_path = './net/cascadenet-final-' + str(testindex) + '.pth'
  file_path="./record/"+str(testindex)+"/"

  if not os.path.exists(file_path):
      os.mkdir(file_path)

  save_path = './net/cascadenet-'+str(testindex)+'.pth'

  iters = []
  losss = []



  datas = IrisData()
  datas.fiveSlpit()


  for id in range(5):


    train_data = datas.getTrainData(id)
    train_label = datas.getTrainLabel(id)

    test_data = datas.getTestData(id)
    test_label = datas.getTestLabel(id)

    datalen = train_data.shape[0]
    ts = np.zeros((datalen, n_output), dtype=np.float32)  # 60000*10
    ts[np.arange(datalen), train_label] = 1  # 75*3
    estrainlabel = ts

    train_data = torch.Tensor(train_data)  # numpy->tensor
    train_label = torch.Tensor(train_label).long()
    estrainlabel = torch.Tensor(estrainlabel)

    test_data = torch.Tensor(test_data)  # numpy->tensor
    test_label = torch.Tensor(test_label).long()



    train_dataset = IrisDataset(train_data, train_label)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataset = IrisDataset(test_data, test_label)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device=torch.device("cuda")
    device = torch.device("cpu")
    print(device)



    train_loss = []
    layer_loss = []

    iter = 0
    net = CascadeNet(n_feature, n_output, lr, device, file_path)
    net = net.to(device)

    # drawClass(net, train_data, train_label, 0, 0, file_path)
    trainiter = net.train_io(maxioepoches, train_dataset, train_loader)

    # tweights=np.array([[-1.3844,  0.6109],[-1.4218,  0.5941]])
    # net.out.state_dict()["weight"].copy_(torch.tensor(tweights)) #loss 0.7870291216032845
    # print(net.out.state_dict()["weight"])

    layer_loss.append(net.train_loss[-1])
    train_loss.extend(net.train_loss)
    iter = iter + trainiter
    print("iter:", iter)




    for epoch in range(1, 300):

      es = net.get_loss2(train_data.to(device), estrainlabel.to(device))
      trainiter, train_data = net.add_hidden_unit(epoch, train_data.to(device), es.to(device))
      #trainiter= net.add_hidden_unit(epoch, train_data1.to(device), es.to(device))

      waitloss = [train_loss[-1] for _ in range(trainiter)]
      train_loss.extend(waitloss)
      iter = iter + trainiter
      print("iter:", iter)

      #drawClass(net, train_data, train_label, epoch, 0, file_path)




      #train_data1=net.get_features(train_data0)#clone.detach过

      #print(train_data1.shape)


      train_dataset = IrisDataset(train_data, train_label)
      train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

      print("hidden unitnum:", epoch)

      if (lr > 0.001):
        lr = lr * alpha

      print("lr:", lr)
      net.setlr(lr)
      #net.leastepoch=net.leastepoch+10
      trainiter = net.train_io(maxioepoches, train_dataset, train_loader)
      train_loss.extend(net.train_loss)
      iter = iter + trainiter
      print("iter:", iter)


      if (net.train_accurate == 1.0):
        break
      if (net.netLoss < 0.01):
        break

      if (epoch%10==0):
        torch.save(net, './net/cascadenet-'+str(epoch)+'.pth')

      #if (epoch%5==0):
       # train_data0, train_label, estrainlabel = changeTrainData()
       # train_data1 = net.get_features(train_data0)  # clone.detach过
       # print("chagne:"+str(train_data1.shape))

   #   drawClass(net, train_data,train_label,epoch,1, file_path)

    print("iter:", iter)

    hiddenlist.append(len(net.hiddenunits))

    torch.save(net, save_path)

    with torch.no_grad():

      train_data = datas.getTrainData(id)
      train_label = datas.getTrainLabel(id)
      train_data = torch.Tensor(train_data)  # numpy->tensor
      train_label = torch.Tensor(train_label).long()
      train_dataset = IrisDataset(train_data, train_label)
      train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

      acc1 = 0
      for train_data in train_loader:
        train_datas, train_labels = train_data

        outputs = net.test1(train_datas.to(device))

        predict_y = torch.max(outputs, dim=1)[1]
        acc1 += (predict_y == train_labels.to(device)).sum().item()

      acc1list.append(acc1 / len(train_dataset))
      print("trainacc:", acc1list[-1])

      acc2 = 0
      for test_data in test_loader:
        test_datas, test_labels = test_data

        outputs = net.test1(test_datas.to(device))

        predict_y = torch.max(outputs, dim=1)[1]
        acc2 += (predict_y == test_labels.to(device)).sum().item()

      acc2list.append(acc2 / len(test_dataset))
      print("testacc:", acc2list[-1])

      acc3= (acc1+acc2)/(len(train_dataset)+len(test_dataset))
      acc3list.append(acc3)
      print("allacc:",acc3);


    #with open('./record/' + "acc"+str(testindex) + ".txt", "w") as f:
    #  data1 = layer_loss
    #  f.write(str(data1))

  return acc1list,acc2list,acc3list,hiddenlist











