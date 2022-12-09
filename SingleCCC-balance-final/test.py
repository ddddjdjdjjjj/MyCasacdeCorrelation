import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from DatasetMnist import *
import numpy as np

import numpy as np

p=0.5
r=np.random.rand(5)<p

print(r)