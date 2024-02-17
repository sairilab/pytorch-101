# %%

100
# %%

import torch
# %%

torch.cuda.is_available()
# %%

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
# %%

import numpy as np

# %% 

def setup_all_seed(seed=0):
    # numpyに関係する乱数シードの設定
    np.random.seed(seed)
    
    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# %%

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=False,
                                           transform=transforms.ToTensor(),
                                           download = True)
# %%

len(train_dataset), len(test_dataset)
# %%

train_dataset[0][0].shape
# %%

fig, label = train_dataset[0]
# %%

fig, label
# %%

fig.size()

# import matplotlib.pyplot as plt
# %%
