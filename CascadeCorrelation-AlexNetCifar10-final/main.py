from cascadenet import CascadeNet

#from DatasetFourSpirals import FourSpiralsDataset,FourSpiralsData,drawClass
from DatasetCifar10Features import *
from torch.utils.data import DataLoader
import torch
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')#使用非交互式后端，避免创建窗口

import os

def main(testindex):
  # n_feature=4
  # n_output=3

  n_feature = 6400#4096
  n_output = 10

  maxioepoches = 50
  batch_size = 250
  lr = 0.1
  alpha = 0.9

  save_path = './net/cascadenet' + str(testindex) + '.pth'
  file_path="./record/"+str(testindex)+"/"

  if not os.path.exists(file_path):
      os.mkdir(file_path)

  save_path = './net/cascadenet-'+str(testindex)+'.pth'

  iters = []
  losss = []

  datas = Cifar10FeaturesTrainData()

  train_data = datas.getData()
  train_label = datas.getLabels()

  datalen = train_data.shape[0]
  ts = np.zeros((datalen, n_output), dtype=np.float32)  # 60000*10
  ts[np.arange(datalen), train_label] = 1  # 75*3
  estrainlabel = ts


  train_data = torch.Tensor(train_data)  # numpy->tensor
  train_label = torch.Tensor(train_label).long()

  train_dataset = Cifar10FeaturesDataset(train_data, train_label)
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)


  #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  #device=torch.device("cuda")
  device = torch.device("cpu")
  print(device)




  estrainlabel = torch.Tensor(estrainlabel)

  train_loss = []
  layer_loss = []

  iter = 0
  net = CascadeNet(n_feature, n_output, lr, device, file_path)
  net=net.to(device)


  #drawClass(net, train_data, train_label, 0, 0, file_path)
  trainiter = net.train_io(maxioepoches, train_dataset,  train_loader)


  #tweights=np.array([[-1.3844,  0.6109],[-1.4218,  0.5941]])
  #net.out.state_dict()["weight"].copy_(torch.tensor(tweights)) #loss 0.7870291216032845
  #print(net.out.state_dict()["weight"])

  layer_loss.append(net.train_loss[-1])
  train_loss.extend(net.train_loss)
  iter = iter + trainiter
  print("iter:", iter)
  #drawClass(net, train_data, train_label, 0, 1, file_path)



  for epoch in range(1, 6):

    es = net.get_loss2(train_data.to(device), estrainlabel.to(device))
    trainiter, train_data = net.add_hidden_unit(epoch, train_data.to(device), es.to(device))

    waitloss = [train_loss[-1] for _ in range(trainiter)]
    train_loss.extend(waitloss)
    iter = iter + trainiter
    print("iter:", iter)

    #drawClass(net, train_data, train_label, epoch, 0, file_path)



    train_dataset = Cifar10FeaturesDataset(train_data, train_label)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    print("hidden unitnum:", epoch)

    if (lr > 0.001):
      lr = lr * alpha
    print("lr:", lr)
    net.setlr(lr)
    net.leastepoch=net.leastepoch+5
    trainiter = net.train_io(maxioepoches, train_dataset, train_loader)

    layer_loss.append(net.train_loss[-1])
    train_loss.extend(net.train_loss)
    iter = iter + trainiter
    print("iter:", iter)


    if (net.train_accurate == 2.0):
      break
    if (net.netLoss < -1):
      break

    if(epoch%10==0):
      torch.save(net, save_path)


 #   drawClass(net, train_data,train_label,epoch,1, file_path)

  print("iter:", iter)
  #drawClass(net, train_data, train_label, 100000,1, file_path)

  device=torch.device("cpu")
  net=net.to(device)
  torch.save(net, save_path)


  #plt.xlabel('Iteration')
  #plt.ylabel('Loss')
  #plt.plot(train_loss, color='red', label='Training loss')
  #plt.savefig('./record/'+str(testindex)+'/train_loss.png')
  #plt.clf()
  with open('./record/' + str(testindex) + "/loss.txt", "w") as f:
    data1 = layer_loss

    f.write(str(data1))

  return iter,epoch











