from cascadenet import CascadeNet

from DatasetTwoSpirals import TwoSpiralsDataset,TwoSpiralsData,drawClass
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

  n_feature = 2
  n_output = 2

  maxioepoches = 1000
  batch_size = 96
  lr = 0.1
  alpha = 0.8

  save_path = './net/cascadenet' + str(testindex) + '.pth'
  file_path="./record/"+str(testindex)+"/"

  if not os.path.exists(file_path):
      os.mkdir(file_path)

  save_path = './net/cascadenet-'+str(testindex)+'.pth'

  iters = []
  losss = []


  datas = TwoSpiralsData(96*2)
  data = datas.getData()
  label = datas.getLabels()

  datalen = len(label)
  print("datalen:", datalen)

  ts = np.zeros((datalen, n_output), dtype=np.float32)  # 60000*10
  ts[np.arange(datalen), label] = 1  # 75*3
  estrainlabel = ts

  train_data = data[0:datalen, :]
  train_label = label[0:datalen]
  estrainlabel = estrainlabel[0:datalen]




  train_dataset = TwoSpiralsDataset(train_data, train_label)
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)


  # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  device = torch.device("cpu")
  print(device)

  train_data = torch.Tensor(train_data)
  train_label = torch.Tensor(train_label).long()


  estrainlabel = torch.Tensor(estrainlabel)

  train_loss = []
  iter = 0
  net = CascadeNet(n_feature, n_output, lr, device, file_path).to(device)

  #drawClass(net, train_data, train_label, 0, 0, file_path)
  trainiter = net.train_io(maxioepoches, train_dataset,  train_loader)


  #tweights=np.array([[-1.3844,  0.6109],[-1.4218,  0.5941]])
  #net.out.state_dict()["weight"].copy_(torch.tensor(tweights)) #loss 0.7870291216032845
  #print(net.out.state_dict()["weight"])

  train_loss.extend(net.train_loss)
  iter = iter + trainiter
  print("iter:", iter)
  #drawClass(net, train_data, train_label, 0, 1, file_path)



  for epoch in range(1, 50):

    es = net.get_loss2(train_data.to(device), estrainlabel.to(device))
    trainiter, train_data = net.add_hidden_unit(epoch, train_data.to(device), es.to(device))

    waitloss = [train_loss[-1] for _ in range(trainiter)]
    train_loss.extend(waitloss)
    iter = iter + trainiter
    print("iter:", iter)

    #drawClass(net, train_data, train_label, epoch, 0, file_path)



    train_dataset = TwoSpiralsDataset(train_data, train_label)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    print("hidden unitnum:", epoch)

    if (lr > 0.001):
      lr = lr * alpha
    print("lr:", lr)
    net.setlr(lr)
    net.leastepoch=net.leastepoch+100
    trainiter = net.train_io(maxioepoches, train_dataset, train_loader)
    train_loss.extend(net.train_loss)
    iter = iter + trainiter
    print("iter:", iter)


    if (net.train_accurate == 1.0):
      break
    if (net.netLoss < 0.01):
      break

 #   drawClass(net, train_data,train_label,epoch,1, file_path)

  print("iter:", iter)
  drawClass(net, train_data, train_label, 100000,1, file_path)
#  torch.save(net, save_path)


  plt.xlabel('Iteration')
  plt.ylabel('Loss')
  plt.plot(train_loss, color='red', label='Training loss')
  plt.savefig('./record/'+str(testindex)+'/train_loss.png')
  plt.clf()

  return iter,epoch











