import matplotlib.pyplot as plt
import numpy as np
import torch

a=torch.nn.Linear(4,2)


a.state_dict()["weight"].copy_(torch.Tensor([[1,1,1,1],[2,2,2,2]]))

for name in a.state_dict():
    print(name)
    print(a.state_dict()[name])




